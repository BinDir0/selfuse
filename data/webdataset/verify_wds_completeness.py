"""
Verify data completeness between WebDataset shards and original Zarr datasets.

Multi-process scan all shards to collect every sample key, then compare against
Zarr meta/episode_ends to detect missing or extra episodes/frames.

Usage:
    python data/webdataset/verify_wds_completeness.py \
        --zarr_list /path/to/zarr_paths.txt \
        --wds_dir /cfs/data/wds/ \
        --num_workers 16
"""

import re
import argparse
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

import numpy as np
import zarr
import webdataset as wds


KEY_PATTERN = re.compile(r"^(.+)_ep(\d+)_f(\d+)$")


def scan_shard(shard_path):
    """Extract all (dataset_name, episode_idx, frame_idx) from a shard."""
    keys = []
    bad_keys = []
    dataset = wds.WebDataset(str(shard_path), shardshuffle=False)
    for sample in dataset:
        key = sample["__key__"]
        m = KEY_PATTERN.match(key)
        if not m:
            bad_keys.append(key)
            continue
        keys.append((m.group(1), int(m.group(2)), int(m.group(3))))
    return shard_path.name, keys, bad_keys


def main():
    parser = argparse.ArgumentParser(
        description="Verify WebDataset completeness against Zarr episode_ends")
    parser.add_argument("--zarr_list", type=str, required=True,
                        help="Text file: each line is 'zarr_path [human|real_world]'")
    parser.add_argument("--wds_dir", type=str, required=True,
                        help="Root directory containing WebDataset shards")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel scan workers")
    args = parser.parse_args()

    # --- Build expected frame set from Zarr ---
    print("Loading Zarr episode metadata...")
    # expected[dataset_name] = {ep_idx: num_frames}
    expected = {}
    with open(args.zarr_list) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    for line in lines:
        parts = line.split()
        zarr_path = parts[0]
        dataset_name = Path(zarr_path).stem

        try:
            src = zarr.open_consolidated(zarr_path, mode='r')
        except Exception:
            src = zarr.open(zarr_path, mode='r')

        episode_ends = src['meta/episode_ends'][:]
        ep_frames = {}
        for i, end in enumerate(episode_ends):
            start = 0 if i == 0 else int(episode_ends[i - 1])
            ep_frames[i] = int(end) - start
        expected[dataset_name] = ep_frames
        total_frames = int(episode_ends[-1])
        print(f"  {dataset_name}: {len(ep_frames)} episodes, {total_frames} frames")

    # --- Scan all shards in parallel ---
    all_shards = sorted(Path(args.wds_dir).rglob("shard-*.tar"))
    print(f"\nScanning {len(all_shards)} shards with {args.num_workers} workers...")

    # actual[dataset_name][ep_idx] = set of frame indices
    actual = defaultdict(lambda: defaultdict(set))
    total_samples = 0
    total_bad_keys = 0

    with mp.Pool(args.num_workers) as pool:
        for i, (shard_name, keys, bad_keys) in enumerate(
                pool.imap_unordered(scan_shard, all_shards)):
            total_samples += len(keys)
            total_bad_keys += len(bad_keys)
            for ds_name, ep_idx, frame_idx in keys:
                actual[ds_name][ep_idx].add(frame_idx)
            if (i + 1) % 100 == 0 or (i + 1) == len(all_shards):
                print(f"  [{i+1}/{len(all_shards)}] scanned, "
                      f"{total_samples} samples so far", flush=True)

    # --- Compare ---
    print(f"\n{'='*60}")
    print("Completeness check:\n")

    all_ok = True

    for ds_name, ep_frames in sorted(expected.items()):
        ds_actual = actual.get(ds_name, {})
        missing_eps = []
        incomplete_eps = []
        extra_eps = []
        total_missing_frames = 0
        total_extra_frames = 0

        for ep_idx, expected_len in ep_frames.items():
            expected_set = set(range(expected_len))
            actual_set = ds_actual.get(ep_idx, set())

            if not actual_set:
                missing_eps.append(ep_idx)
                total_missing_frames += expected_len
            else:
                missing = expected_set - actual_set
                extra = actual_set - expected_set
                if missing:
                    incomplete_eps.append((ep_idx, len(missing), expected_len))
                    total_missing_frames += len(missing)
                if extra:
                    total_extra_frames += len(extra)

        # Episodes in WDS but not in Zarr
        for ep_idx in ds_actual:
            if ep_idx not in ep_frames:
                extra_eps.append(ep_idx)
                total_extra_frames += len(ds_actual[ep_idx])

        expected_total = sum(ep_frames.values())
        actual_total = sum(len(fs) for fs in ds_actual.values())
        ok = (not missing_eps and not incomplete_eps and not extra_eps
              and total_missing_frames == 0 and total_extra_frames == 0)

        status = "OK" if ok else "INCOMPLETE"
        if not ok:
            all_ok = False
        print(f"[{status}] {ds_name}: "
              f"{actual_total}/{expected_total} frames, "
              f"{len(ds_actual)}/{len(ep_frames)} episodes")

        if missing_eps:
            shown = missing_eps[:10]
            print(f"  Missing episodes ({len(missing_eps)}): {shown}"
                  f"{'...' if len(missing_eps) > 10 else ''}")
        if incomplete_eps:
            shown = incomplete_eps[:5]
            for ep_idx, n_missing, n_expected in shown:
                print(f"  Episode {ep_idx}: {n_missing}/{n_expected} frames missing")
            if len(incomplete_eps) > 5:
                print(f"  ... and {len(incomplete_eps) - 5} more incomplete episodes")
        if extra_eps:
            print(f"  Extra episodes not in Zarr ({len(extra_eps)}): {extra_eps[:10]}")
        if total_missing_frames:
            print(f"  Total missing frames: {total_missing_frames}")
        if total_extra_frames:
            print(f"  Total extra frames: {total_extra_frames}")

    # Datasets in WDS but not in zarr_list
    unknown_ds = set(actual.keys()) - set(expected.keys())
    if unknown_ds:
        all_ok = False
        print(f"\nUnknown datasets in WDS (not in zarr_list): {unknown_ds}")

    print(f"\n{'='*60}")
    print(f"Total: {total_samples} samples across {len(all_shards)} shards")
    if total_bad_keys:
        print(f"Unparseable keys: {total_bad_keys}")
    if all_ok:
        print("All datasets complete.")
    else:
        print("Some datasets have missing or extra data.")


if __name__ == "__main__":
    main()
