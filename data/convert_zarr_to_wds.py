"""
Convert Zarr datasets to WebDataset shard format.

Each sample stores per-frame raw data identical to zarr structure.
All frames of an episode are written contiguously to the same shard.

`zarr_list` format:
    zarr_path [human|real_world] [wds_dataset]

If `wds_dataset` is omitted, it defaults to the zarr directory stem.
Datasets with the same `wds_dataset` are written into the same output folder.

Usage:
    python data/convert_zarr_to_wds.py \
        --zarr_list /path/to/zarr_paths.txt \
        --output_dir /cfs/data/wds/ \
        --num_workers 120
"""

import io
import sys
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from collections import Counter

import numpy as np
import zarr
import webdataset as wds
from PIL import Image

from zarr_list_utils import HUMAN_KEY_MAPPING, REAL_WORLD_KEY_MAPPING, parse_zarr_list


def process_episodes(zarr_path, episode_batch, output_pattern, dataset_name,
                     key_mapping, worker_id=0):
    """Process a batch of episodes from a single zarr dataset.

    Each sample stores per-frame raw data identical to zarr structure.
    All frames of an episode are written contiguously to the same shard.
    """
    try:
        src = zarr.open_consolidated(zarr_path, mode='r')
    except Exception:
        src = zarr.open(zarr_path, mode='r')

    data_root = src['data'] if 'data' in src else src

    def get_array(key_path):
        node = data_root
        for part in key_path.split('/'):
            node = node[part]
        return node

    # presence is per-timestep in data group, shape (total_frames,) int8
    # Only human datasets have presence; real_world datasets do not.
    has_presence = 'presence' in key_mapping
    presence_array = get_array(key_mapping['presence']) if has_presence else None

    # Get array references once (no data loaded yet)
    image_array = get_array(key_mapping['image'])
    depth_array = get_array(key_mapping['depth'])

    total_eps = len(episode_batch)
    # Report ~10 times per worker, at least every episode
    report_interval = max(1, total_eps // 10)
    frames_done = 0
    t0 = time.time()

    maxcount = 20000
    maxsize = int(1e9)
    shard_idx = 0
    shard_count = 0
    shard_size = 0
    writer = wds.TarWriter(output_pattern % shard_idx)

    for ep_i, (ep_start, ep_end, ep_idx) in enumerate(episode_batch):
        T = ep_end - ep_start

        # Start a new shard BEFORE this episode if current shard is near limits.
        # This guarantees no episode is split across shards.
        # When shard_count == 0 the condition short-circuits, so an episode
        # that alone exceeds maxcount/maxsize still gets written to one shard.
        if shard_count > 0 and (shard_count + T > maxcount or shard_size > maxsize):
            writer.close()
            shard_idx += 1
            writer = wds.TarWriter(output_pattern % shard_idx)
            shard_count = 0
            shard_size = 0

        if T > maxcount:
            print(f"  [w{worker_id:03d}] WARNING: episode {ep_idx} has {T} frames "
                  f"(> maxcount={maxcount}), shard will exceed limit",
                  flush=True)

        # Batch-load all episode data in single contiguous reads.
        # This turns N per-frame random reads into one sequential read
        # per array, which is critical for CFS performance.
        wrist_s = get_array(key_mapping['wrist_state'])[ep_start:ep_end]
        hand_s = get_array(key_mapping['hand_state'])[ep_start:ep_end]
        wrist_a = get_array(key_mapping['wrist_action'])[ep_start:ep_end]
        hand_a = get_array(key_mapping['hand_action'])[ep_start:ep_end]
        ext = get_array(key_mapping['extrinsic'])[ep_start:ep_end]
        intr = get_array(key_mapping['intrinsic'])[ep_start:ep_end]
        instructions = get_array(key_mapping['instruction'])[ep_start:ep_end]
        instruction_nums = get_array(key_mapping['instruction_num'])[ep_start:ep_end]
        presence = presence_array[ep_start:ep_end] if has_presence else None

        # Batch-load images and depth for the entire episode (one sequential read)
        images = image_array[ep_start:ep_end]  # (T, H, W, C) uint8
        depths = depth_array[ep_start:ep_end]  # (T, H, W) uint16

        # Vectorized lowdim: concatenate once for all frames -> (T, 116)
        lowdim_all = np.concatenate([
            wrist_s.reshape(T, -1),
            hand_s.reshape(T, -1),
            wrist_a.reshape(T, -1),
            hand_a.reshape(T, -1),
            ext.reshape(T, -1),
            intr.reshape(T, -1),
        ], axis=1).astype(np.float32)

        for t in range(T):
            # JPEG encode
            buf = io.BytesIO()
            Image.fromarray(images[t]).save(buf, format='JPEG', quality=95)

            # Instruction
            instr = instructions[t]
            if isinstance(instr, np.ndarray):
                instr = instr.tolist()
            if isinstance(instr, bytes):
                instr = instr.decode('utf-8')

            meta_dict = {
                "dataset_name": dataset_name,
                "episode_index": int(ep_idx),
                "instruction": instr,
                "instruction_num": int(instruction_nums[t]),
            }
            if has_presence:
                meta_dict["presence"] = presence[t].tolist()

            sample = {
                "__key__": f"{dataset_name}_ep{ep_idx:06d}_f{t:05d}",
                "image.jpg": buf.getvalue(),
                "depth.npy": depths[t].astype(np.uint16),
                "lowdim.npy": lowdim_all[t],
                "meta.json": meta_dict,
            }
            writer.write(sample)
            shard_count += 1
            shard_size += len(sample["image.jpg"]) + depths[t].nbytes + lowdim_all[t].nbytes + 512 * 4

        frames_done += T
        ep_done = ep_i + 1
        if ep_done % report_interval == 0 or ep_done == total_eps:
            elapsed = time.time() - t0
            fps = frames_done / elapsed if elapsed > 0 else 0
            print(f"  [w{worker_id:03d}] {dataset_name}: "
                  f"{ep_done}/{total_eps} eps, "
                  f"{frames_done} frames, "
                  f"{elapsed:.1f}s ({fps:.0f} frames/s)",
                  flush=True)

    writer.close()


def convert_zarr_dataset(zarr_path, output_dir, dataset_name, key_mapping,
                         num_workers=120, shard_prefix=None):
    """Convert a single zarr dataset using multiple workers."""
    try:
        src = zarr.open_consolidated(zarr_path, mode='r')
    except Exception:
        src = zarr.open(zarr_path, mode='r')

    meta = src['meta'] if 'meta' in src else None
    if meta is None or 'episode_ends' not in meta:
        print(f"Warning: No episode_ends in {zarr_path}, skipping.")
        return

    episode_ends = meta['episode_ends'][:]

    # Build episode list: (start, end, index)
    episodes = []
    for i, end in enumerate(episode_ends):
        start = 0 if i == 0 else int(episode_ends[i - 1])
        episodes.append((int(start), int(end), i))

    # Split episodes into chunks for workers
    actual_workers = min(num_workers, len(episodes))
    if actual_workers == 0:
        return
    chunk_size = max(1, len(episodes) // actual_workers)
    chunks = [episodes[i:i + chunk_size]
              for i in range(0, len(episodes), chunk_size)]

    ds_output_dir = Path(output_dir)
    ds_output_dir.mkdir(parents=True, exist_ok=True)

    args_list = []
    for worker_id, chunk in enumerate(chunks):
        if shard_prefix:
            pattern = str(ds_output_dir / f"shard-{shard_prefix}-w{worker_id:04d}-%06d.tar")
        else:
            pattern = str(ds_output_dir / f"shard-w{worker_id:04d}-%06d.tar")
        args_list.append((zarr_path, chunk, pattern, dataset_name, key_mapping,
                          worker_id))

    if actual_workers <= 1:
        # Single process for small datasets
        for args in args_list:
            process_episodes(*args)
    else:
        with mp.Pool(actual_workers) as pool:
            pool.starmap(process_episodes, args_list)

    print(f"  Done: {dataset_name} ({len(episodes)} episodes, "
          f"{int(episode_ends[-1])} frames)")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Zarr datasets to WebDataset shards")
    parser.add_argument("--zarr_list", type=str, required=True,
                        help="Text file: each line is 'zarr_path [human|real_world] [wds_dataset]'")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for WebDataset shards")
    parser.add_argument("--num_workers", type=int, default=120,
                        help="Number of parallel workers per dataset")
    args = parser.parse_args()

    try:
        entries = parse_zarr_list(args.zarr_list)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    group_sizes = Counter(entry.wds_dataset for entry in entries)

    for entry in entries:
        key_mapping = (REAL_WORLD_KEY_MAPPING if entry.mapping_type == "real_world"
                       else HUMAN_KEY_MAPPING)
        ds_output_dir = Path(args.output_dir) / entry.wds_dataset
        shard_prefix = entry.dataset_name if group_sizes[entry.wds_dataset] > 1 else None

        print(f"Converting {entry.zarr_path} ({entry.mapping_type}) -> {entry.wds_dataset}")
        convert_zarr_dataset(
            entry.zarr_path, ds_output_dir, entry.dataset_name,
            key_mapping=key_mapping,
            num_workers=args.num_workers,
            shard_prefix=shard_prefix,
        )
