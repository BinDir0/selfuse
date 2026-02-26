"""
Verify numerical consistency between WebDataset shards and original Zarr data.

Randomly samples shards, streams through all samples, and compares each field
against the original Zarr source. All fields should match exactly (PNG and npy
are lossless).

Usage:
    python data/verify_wds_consistency.py \
        --zarr_list /path/to/zarr_paths.txt \
        --wds_dir /cfs/data/wds/ \
        --num_shards 100
"""

import io
import re
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import zarr
import webdataset as wds
from PIL import Image


# Must match convert_zarr_to_wds.py
HUMAN_KEY_MAPPING = {
    'image': 'image',
    'depth': 'depth',
    'wrist_state': 'state/wrist',
    'hand_state': 'state/fingertips',
    'wrist_action': 'action/wrist',
    'hand_action': 'action/fingertips',
    'extrinsic': 'extrinsic',
    'intrinsic': 'intrinsic',
    'instruction': 'instruction',
    'instruction_num': 'instruction_num',
    'presence': 'presence',
}

REAL_WORLD_KEY_MAPPING = {
    'image': 'image-head',
    'depth': 'depth-head',
    'wrist_state': 'state/wrist-head',
    'hand_state': 'state/fingertips-head',
    'wrist_action': 'action/wrist-head',
    'hand_action': 'action/fingertips-head',
    'extrinsic': 'extrinsic',
    'intrinsic': 'intrinsic/head',
    'instruction': 'instruction',
    'instruction_num': 'instruction_num',
}

KEY_PATTERN = re.compile(r"^(.+)_ep(\d+)_f(\d+)$")


def load_zarr_registry(zarr_list):
    """Build mapping: dataset_name -> (zarr_handle, key_mapping, episode_starts)."""
    registry = {}
    with open(zarr_list) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    for line in lines:
        parts = line.split()
        zarr_path = parts[0]
        mapping_type = parts[1] if len(parts) > 1 else "human"
        key_mapping = (REAL_WORLD_KEY_MAPPING if mapping_type == "real_world"
                       else HUMAN_KEY_MAPPING)
        dataset_name = Path(zarr_path).stem

        try:
            src = zarr.open_consolidated(zarr_path, mode='r')
        except Exception:
            src = zarr.open(zarr_path, mode='r')

        data_root = src['data'] if 'data' in src else src
        episode_ends = src['meta/episode_ends'][:]
        episode_starts = np.zeros_like(episode_ends)
        episode_starts[1:] = episode_ends[:-1]

        registry[dataset_name] = {
            "data_root": data_root,
            "key_mapping": key_mapping,
            "episode_starts": episode_starts,
            "episode_ends": episode_ends,
        }
    return registry


def get_zarr_array(data_root, key_path):
    node = data_root
    for part in key_path.split('/'):
        node = node[part]
    return node


def verify_sample(sample, registry):
    """Verify a single WebDataset sample against Zarr source.

    Returns (is_ok, error_message).
    """
    key = sample["__key__"]
    m = KEY_PATTERN.match(key)
    if not m:
        return False, f"Cannot parse key: {key}"

    dataset_name, ep_idx, frame_idx = m.group(1), int(m.group(2)), int(m.group(3))
    if dataset_name not in registry:
        return False, f"Dataset {dataset_name} not in registry"

    reg = registry[dataset_name]
    km = reg["key_mapping"]
    data_root = reg["data_root"]
    abs_idx = int(reg["episode_starts"][ep_idx]) + frame_idx

    # --- Image: allow minor JPEG compression error (quality=95, typically ±2) ---
    wds_img = np.array(Image.open(io.BytesIO(sample["image.jpg"])))
    zarr_img = get_zarr_array(data_root, km['image'])[abs_idx]
    if wds_img.shape != zarr_img.shape:
        return False, f"{key}: image shape mismatch {wds_img.shape} vs {zarr_img.shape}"
    diff = np.abs(wds_img.astype(int) - zarr_img.astype(int))
    if diff.max() > 10:
        return False, f"{key}: image mismatch, max_diff={diff.max()}"

    # --- Depth: exact match (npy is lossless) ---
    wds_depth = sample["depth.npy"]
    if isinstance(wds_depth, bytes):
        wds_depth = np.load(io.BytesIO(wds_depth))
    zarr_depth = get_zarr_array(data_root, km['depth'])[abs_idx]
    if not np.array_equal(wds_depth, zarr_depth):
        return False, f"{key}: depth mismatch"

    # --- Lowdim: exact match ---
    wds_ld = sample["lowdim.npy"]
    if isinstance(wds_ld, bytes):
        wds_ld = np.load(io.BytesIO(wds_ld))

    zarr_fields = [
        get_zarr_array(data_root, km['wrist_state'])[abs_idx].reshape(-1),
        get_zarr_array(data_root, km['hand_state'])[abs_idx].reshape(-1),
        get_zarr_array(data_root, km['wrist_action'])[abs_idx].reshape(-1),
        get_zarr_array(data_root, km['hand_action'])[abs_idx].reshape(-1),
        get_zarr_array(data_root, km['extrinsic'])[abs_idx].reshape(-1),
        get_zarr_array(data_root, km['intrinsic'])[abs_idx].reshape(-1),
    ]
    zarr_ld = np.concatenate(zarr_fields).astype(np.float32)
    if not np.array_equal(wds_ld, zarr_ld):
        diff = np.abs(wds_ld - zarr_ld)
        return False, f"{key}: lowdim mismatch, max_diff={diff.max()}"

    # --- Meta: instruction, instruction_num ---
    meta = sample["meta.json"]
    if isinstance(meta, bytes):
        meta = json.loads(meta.decode("utf-8"))

    zarr_instr = get_zarr_array(data_root, km['instruction'])[abs_idx]
    if isinstance(zarr_instr, bytes):
        zarr_instr = zarr_instr.decode('utf-8')
    if isinstance(zarr_instr, np.ndarray):
        zarr_instr = zarr_instr.tolist()
    if str(meta["instruction"]) != str(zarr_instr):
        return False, f"{key}: instruction mismatch"

    zarr_instr_num = int(get_zarr_array(data_root, km['instruction_num'])[abs_idx])
    if meta["instruction_num"] != zarr_instr_num:
        return False, f"{key}: instruction_num mismatch"

    return True, ""


def main():
    parser = argparse.ArgumentParser(
        description="Verify WebDataset shards against original Zarr data")
    parser.add_argument("--zarr_list", type=str, required=True,
                        help="Text file: each line is 'zarr_path [human|real_world]'")
    parser.add_argument("--wds_dir", type=str, required=True,
                        help="Root directory containing WebDataset shards")
    parser.add_argument("--num_shards", type=int, default=100,
                        help="Number of shards to randomly sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading Zarr registry...")
    registry = load_zarr_registry(args.zarr_list)
    print(f"  Loaded {len(registry)} datasets: {list(registry.keys())}")

    # Find all shards
    all_shards = sorted(Path(args.wds_dir).rglob("shard-*.tar"))
    print(f"  Found {len(all_shards)} total shards")

    sampled = random.sample(all_shards, min(args.num_shards, len(all_shards)))
    print(f"  Sampled {len(sampled)} shards for verification\n")

    total_samples = 0
    total_errors = 0
    error_summary = defaultdict(int)

    for i, shard_path in enumerate(sampled):
        shard_str = str(shard_path)
        shard_samples = 0
        shard_errors = 0

        dataset = wds.WebDataset(shard_str)
        for sample in dataset:
            ok, msg = verify_sample(sample, registry)
            shard_samples += 1
            if not ok:
                shard_errors += 1
                error_type = msg.split(":")[1].strip().split(",")[0] if ":" in msg else msg
                error_summary[error_type] += 1
                if shard_errors <= 3:
                    print(f"  ERROR: {msg}")

        total_samples += shard_samples
        total_errors += shard_errors
        status = "OK" if shard_errors == 0 else f"FAIL ({shard_errors} errors)"
        print(f"[{i+1}/{len(sampled)}] {shard_path.name}: "
              f"{shard_samples} samples - {status}")

    print(f"\n{'='*60}")
    print(f"Total: {total_samples} samples, {total_errors} errors")
    if error_summary:
        print("Error breakdown:")
        for err_type, count in sorted(error_summary.items()):
            print(f"  {err_type}: {count}")
    else:
        print("All samples verified OK.")


if __name__ == "__main__":
    main()
