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
import sys
import argparse
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

import zarr
import webdataset as wds


DATA_DIR = Path(__file__).resolve().parents[1]
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from zarr_list_utils import parse_zarr_list


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
                        help="Text file: each line is 'zarr_path [human|real_world] [wds_dataset]'")
    parser.add_argument("--wds_dir", type=str, required=True,
                        help="Root directory containing WebDataset shards")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel scan workers")
    args = parser.parse_args()

    try:
        entries = parse_zarr_list(args.zarr_list)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    all_shards = sorted(Path(args.wds_dir).rglob("shard-*.tar"))
    print(f"Scanning {len(all_shards)} shards with {args.num_workers} workers...")

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

    direct_shards = any(Path(args.wds_dir).glob("shard-*.tar"))
    entries_to_check = entries
    group_entries = [entry for entry in entries if entry.wds_dataset == Path(args.wds_dir).name]
    actual_dataset_names = set(actual.keys())
    if direct_shards and group_entries:
        group_dataset_names = {entry.dataset_name for entry in group_entries}
        if not actual_dataset_names or actual_dataset_names.issubset(group_dataset_names):
            entries_to_check = group_entries
            print(f"\nUsing {len(entries_to_check)} zarr entries for wds_dataset '{Path(args.wds_dir).name}'")

    print("Loading Zarr episode metadata...")
    expected = {}
    for entry in entries_to_check:
        try:
            src = zarr.open_consolidated(entry.zarr_path, mode='r')
        except Exception:
            src = zarr.open(entry.zarr_path, mode='r')

        episode_ends = src['meta/episode_ends'][:]
        ep_frames = {}
        for i, end in enumerate(episode_ends):
            start = 0 if i == 0 else int(episode_ends[i - 1])
            ep_frames[i] = int(end) - start
        expected[entry.dataset_name] = ep_frames
        total_frames = int(episode_ends[-1])
        print(f"  {entry.dataset_name}: {len(ep_frames)} episodes, {total_frames} frames")

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
