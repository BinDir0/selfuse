"""
Convert Zarr datasets to WebDataset shard format.

Each sample stores per-frame raw data identical to zarr structure.
All frames of an episode are written contiguously to the same shard.

Usage:
    python data/convert_zarr_to_wds.py \
        --zarr_list /path/to/zarr_paths.txt \
        --output_dir /cfs/data/wds/ \
        --num_workers 120
"""

import io
import json
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import zarr
import webdataset as wds
from PIL import Image


def process_episodes(zarr_path, episode_batch, output_pattern, dataset_name,
                     key_mapping):
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

    with wds.ShardWriter(output_pattern, maxcount=20000, maxsize=int(1e9)) as sink:
        for ep_start, ep_end, ep_idx, ep_presence in episode_batch:
            T = ep_end - ep_start

            # Preload episode low-dim data (fits in memory)
            wrist_s = get_array(key_mapping['wrist_state'])[ep_start:ep_end]
            hand_s = get_array(key_mapping['hand_state'])[ep_start:ep_end]
            wrist_a = get_array(key_mapping['wrist_action'])[ep_start:ep_end]
            hand_a = get_array(key_mapping['hand_action'])[ep_start:ep_end]
            ext = get_array(key_mapping['extrinsic'])[ep_start:ep_end]
            intr = get_array(key_mapping['intrinsic'])[ep_start:ep_end]
            instructions = get_array(key_mapping['instruction'])[ep_start:ep_end]
            instruction_nums = get_array(key_mapping['instruction_num'])[ep_start:ep_end]

            # Image array (lazy-loaded per frame to save memory)
            image_array = get_array(key_mapping['image'])

            for t in range(T):
                # Pack all low-dim into a single array (116,) float32
                # Layout: state/wrist[0:18] | state/hand[18:48] | action/wrist[48:66]
                #         | action/hand[66:96] | extrinsic[96:112] | intrinsic[112:116]
                lowdim = np.concatenate([
                    wrist_s[t].flatten().astype(np.float32),
                    hand_s[t].flatten().astype(np.float32),
                    wrist_a[t].flatten().astype(np.float32),
                    hand_a[t].flatten().astype(np.float32),
                    ext[t].flatten().astype(np.float32),
                    intr[t].flatten().astype(np.float32),
                ]).astype(np.float32)

                # Image: JPEG encode
                img = Image.fromarray(image_array[ep_start + t])
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=95)

                # Instruction: handle per-frame string or array
                instr = instructions[t]
                if isinstance(instr, np.ndarray):
                    instr = instr.tolist()
                if isinstance(instr, bytes):
                    instr = instr.decode('utf-8')
                instr_num = int(instruction_nums[t])

                sink.write({
                    "__key__": f"{dataset_name}_ep{ep_idx:06d}_f{t:05d}",
                    "image.jpg": buf.getvalue(),
                    "lowdim.npy": lowdim,
                    "meta.json": {
                        "dataset_name": dataset_name,
                        "episode_index": int(ep_idx),
                        "instruction": instr,
                        "instruction_num": instr_num,
                        "presence": int(ep_presence),
                    },
                })


def convert_zarr_dataset(zarr_path, output_dir, dataset_name, key_mapping,
                         num_workers=120):
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
    presence = meta['presence'][:] if 'presence' in meta else np.ones(len(episode_ends), dtype=np.int8)

    # Build episode list: (start, end, index, presence)
    episodes = []
    for i, end in enumerate(episode_ends):
        start = 0 if i == 0 else int(episode_ends[i - 1])
        episodes.append((int(start), int(end), i, int(presence[i])))

    # Split episodes into chunks for workers
    actual_workers = min(num_workers, len(episodes))
    if actual_workers == 0:
        return
    chunk_size = max(1, len(episodes) // actual_workers)
    chunks = [episodes[i:i + chunk_size]
              for i in range(0, len(episodes), chunk_size)]

    # Output directory
    ds_output_dir = Path(output_dir) / dataset_name
    ds_output_dir.mkdir(parents=True, exist_ok=True)

    # Each worker writes its own shard sequence
    args_list = []
    for worker_id, chunk in enumerate(chunks):
        pattern = str(ds_output_dir / f"shard-w{worker_id:04d}-%06d.tar")
        args_list.append((zarr_path, chunk, pattern, dataset_name, key_mapping))

    if actual_workers <= 1:
        # Single process for small datasets
        for args in args_list:
            process_episodes(*args)
    else:
        with mp.Pool(actual_workers) as pool:
            pool.starmap(process_episodes, args_list)

    print(f"  Done: {dataset_name} ({len(episodes)} episodes, "
          f"{int(episode_ends[-1])} frames)")


# Default key mappings matching vla_dataset_paths.yaml
HUMAN_KEY_MAPPING = {
    'image': 'image',
    'wrist_state': 'state/wrist',
    'hand_state': 'state/fingertips',
    'wrist_action': 'action/wrist',
    'hand_action': 'action/fingertips',
    'extrinsic': 'extrinsic',
    'intrinsic': 'intrinsic',
    'instruction': 'instruction',
    'instruction_num': 'instruction_num',
}

REAL_WORLD_KEY_MAPPING = {
    'image': 'image-head',
    'wrist_state': 'state/wrist-head',
    'hand_state': 'state/fingertips-head',
    'wrist_action': 'action/wrist-head',
    'hand_action': 'action/fingertips-head',
    'extrinsic': 'extrinsic',
    'intrinsic': 'intrinsic/head',
    'instruction': 'instruction',
    'instruction_num': 'instruction_num',
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Zarr datasets to WebDataset shards")
    parser.add_argument("--zarr_list", type=str, required=True,
                        help="Text file: each line is 'zarr_path [human|real_world]'")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for WebDataset shards")
    parser.add_argument("--num_workers", type=int, default=120,
                        help="Number of parallel workers per dataset")
    args = parser.parse_args()

    with open(args.zarr_list) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    for line in lines:
        parts = line.split()
        zarr_path = parts[0]
        mapping_type = parts[1] if len(parts) > 1 else "human"
        key_mapping = (REAL_WORLD_KEY_MAPPING if mapping_type == "real_world"
                       else HUMAN_KEY_MAPPING)
        dataset_name = Path(zarr_path).stem
        print(f"Converting {zarr_path} ({mapping_type}) -> {dataset_name}")
        convert_zarr_dataset(
            zarr_path, args.output_dir, dataset_name,
            key_mapping=key_mapping,
            num_workers=args.num_workers,
        )
