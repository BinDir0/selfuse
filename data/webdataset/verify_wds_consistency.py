"""
Verify numerical consistency between WebDataset shards and original Zarr data.

Randomly samples shards, streams through all samples, and compares each field
against the original Zarr source. Depth/lowdim/meta should match exactly;
image allows minor JPEG compression error.

Usage:
    python data/webdataset/verify_wds_consistency.py \
        --zarr_list /path/to/zarr_paths.txt \
        --wds_dir /cfs/data/wds/ \
        --num_shards 100 \
        --num_workers 16
"""

import io
import re
import json
import random
import argparse
import multiprocessing as mp
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
IMAGE_MAX_DIFF = 100   # JPEG quality=95 can produce per-pixel diffs up to ~30
IMAGE_AVG_DIFF = 5.0  # JPEG quality=95 average diff is typically < 2


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
            "zarr_path": zarr_path,
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


def verify_sample(sample, data_root, km, abs_idx, key):
    """Verify a single sample. Returns (checks, error, img_max_diff, img_avg_diff)."""
    checks = {
        "image": False, "depth": False, "lowdim": False,
        "instruction": False, "instruction_num": False,
    }

    # --- Image ---
    wds_img = np.array(Image.open(io.BytesIO(sample["image.jpg"])))
    zarr_img = get_zarr_array(data_root, km['image'])[abs_idx]
    if wds_img.shape != zarr_img.shape:
        return checks, f"{key}: image shape mismatch {wds_img.shape} vs {zarr_img.shape}", None, None
    img_abs = np.abs(wds_img.astype(int) - zarr_img.astype(int))
    img_max_diff = int(img_abs.max())
    img_avg_diff = float(img_abs.mean())
    if img_max_diff > IMAGE_MAX_DIFF:
        return checks, f"{key}: image max_diff={img_max_diff} exceeds {IMAGE_MAX_DIFF}", img_max_diff, img_avg_diff
    if img_avg_diff > IMAGE_AVG_DIFF:
        return checks, f"{key}: image avg_diff={img_avg_diff:.2f} exceeds {IMAGE_AVG_DIFF}", img_max_diff, img_avg_diff
    checks["image"] = True

    # --- Depth ---
    wds_depth = sample.get("depth.npy")
    if wds_depth is not None:
        if isinstance(wds_depth, bytes):
            wds_depth = np.load(io.BytesIO(wds_depth))
        zarr_depth = get_zarr_array(data_root, km['depth'])[abs_idx]
        if not np.array_equal(wds_depth, zarr_depth):
            return checks, f"{key}: depth mismatch", img_max_diff, img_avg_diff
    checks["depth"] = True

    # --- Lowdim ---
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
        ld_diff = float(np.abs(wds_ld - zarr_ld).max())
        return checks, f"{key}: lowdim mismatch, max_diff={ld_diff}", img_max_diff, img_avg_diff
    checks["lowdim"] = True

    # --- Meta ---
    meta = sample["meta.json"]
    if isinstance(meta, bytes):
        meta = json.loads(meta.decode("utf-8"))

    zarr_instr = get_zarr_array(data_root, km['instruction'])[abs_idx]
    if isinstance(zarr_instr, bytes):
        zarr_instr = zarr_instr.decode('utf-8')
    if isinstance(zarr_instr, np.ndarray):
        zarr_instr = zarr_instr.tolist()
    if str(meta["instruction"]) != str(zarr_instr):
        return checks, f"{key}: instruction mismatch", img_max_diff, img_avg_diff
    checks["instruction"] = True

    zarr_instr_num = int(get_zarr_array(data_root, km['instruction_num'])[abs_idx])
    if meta["instruction_num"] != zarr_instr_num:
        return checks, f"{key}: instruction_num mismatch", img_max_diff, img_avg_diff
    checks["instruction_num"] = True

    return checks, None, img_max_diff, img_avg_diff


def verify_shard(args):
    """Verify all samples in a single shard. Runs in a worker process."""
    shard_path, zarr_list = args
    registry = load_zarr_registry(zarr_list)

    # Open zarr handles per dataset (process-local)
    zarr_handles = {}
    for ds_name, reg in registry.items():
        try:
            src = zarr.open_consolidated(reg["zarr_path"], mode='r')
        except Exception:
            src = zarr.open(reg["zarr_path"], mode='r')
        zarr_handles[ds_name] = src['data'] if 'data' in src else src

    shard_samples = 0
    shard_errors = 0
    check_totals = defaultdict(int)
    errors = []
    img_max_diffs = []
    img_avg_diffs = []

    dataset = wds.WebDataset(str(shard_path), shardshuffle=False)
    for sample in dataset:
        key = sample["__key__"]
        m = KEY_PATTERN.match(key)
        if not m:
            errors.append(f"Cannot parse key: {key}")
            shard_errors += 1
            shard_samples += 1
            continue

        ds_name = m.group(1)
        ep_idx, frame_idx = int(m.group(2)), int(m.group(3))

        if ds_name not in registry:
            errors.append(f"Dataset {ds_name} not in registry")
            shard_errors += 1
            shard_samples += 1
            continue

        reg = registry[ds_name]
        abs_idx = int(reg["episode_starts"][ep_idx]) + frame_idx
        data_root = zarr_handles[ds_name]

        checks, err, i_max, i_avg = verify_sample(sample, data_root, reg["key_mapping"], abs_idx, key)
        shard_samples += 1
        for field, ok in checks.items():
            if ok:
                check_totals[field] += 1
        if err:
            shard_errors += 1
            errors.append(err)

        # Track image diff stats (reuse values from verify_sample)
        if i_max is not None:
            img_max_diffs.append(i_max)
            img_avg_diffs.append(i_avg)

    img_stats = ""
    if img_max_diffs:
        img_stats = (f"image max_diff: mean={np.mean(img_max_diffs):.1f}, "
                     f"max={np.max(img_max_diffs)}, p99={int(np.percentile(img_max_diffs, 99))}; "
                     f"image avg_diff: mean={np.mean(img_avg_diffs):.2f}, "
                     f"max={np.max(img_avg_diffs):.2f}, p99={np.percentile(img_avg_diffs, 99):.2f}")

    return {
        "shard": shard_path.name,
        "samples": shard_samples,
        "errors": shard_errors,
        "error_msgs": errors[:5],
        "check_totals": dict(check_totals),
        "img_stats": img_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify WebDataset shards against original Zarr data")
    parser.add_argument("--zarr_list", type=str, required=True,
                        help="Text file: each line is 'zarr_path [human|real_world]'")
    parser.add_argument("--wds_dir", type=str, required=True,
                        help="Root directory containing WebDataset shards")
    parser.add_argument("--num_shards", type=int, default=100,
                        help="Number of shards to randomly sample")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel verification workers")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Find and sample shards
    all_shards = sorted(Path(args.wds_dir).rglob("shard-*.tar"))
    print(f"Found {len(all_shards)} total shards")
    sampled = random.sample(all_shards, min(args.num_shards, len(all_shards)))
    print(f"Sampled {len(sampled)} shards, verifying with {args.num_workers} workers\n")

    total_samples = 0
    total_errors = 0
    all_check_totals = defaultdict(int)

    work_args = [(s, args.zarr_list) for s in sampled]

    with mp.Pool(args.num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(verify_shard, work_args)):
            total_samples += result["samples"]
            total_errors += result["errors"]
            for field, cnt in result["check_totals"].items():
                all_check_totals[field] += cnt

            status = "OK" if result["errors"] == 0 else f"FAIL ({result['errors']} errors)"
            print(f"[{i+1}/{len(sampled)}] {result['shard']}: "
                  f"{result['samples']} samples - {status}")
            if result["img_stats"]:
                print(f"  {result['img_stats']}")
            # Print per-field pass counts
            ct = result["check_totals"]
            print(f"  passed: image={ct.get('image',0)} depth={ct.get('depth',0)} "
                  f"lowdim={ct.get('lowdim',0)} instruction={ct.get('instruction',0)}")
            for err in result["error_msgs"]:
                print(f"  ERROR: {err}")

    print(f"\n{'='*60}")
    print(f"Total: {total_samples} samples, {total_errors} errors")
    print(f"Field pass totals:")
    for field in ["image", "depth", "lowdim", "instruction", "instruction_num"]:
        print(f"  {field}: {all_check_totals.get(field, 0)}/{total_samples}")
    if total_errors == 0:
        print("All samples verified OK.")


if __name__ == "__main__":
    main()
