"""
Verify data completeness between local LeRobot datasets and original Zarr datasets.

Checks whether every source episode/frame from zarr_list exists in the target LeRobot
root directory or in a single LeRobot dataset directory.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from _common import (
    load_lerobot_dataset,
    open_zarr_root,
    parse_zarr_list,
    resolve_dataset_dirs,
)


def main():
    parser = argparse.ArgumentParser(description="Verify LeRobot completeness against Zarr episode_ends")
    parser.add_argument("--zarr_list", type=str, required=True)
    parser.add_argument("--lerobot_dir", type=str, required=True)
    parser.add_argument("--repo_id_prefix", type=str, default="local")
    args = parser.parse_args()

    try:
        entries = parse_zarr_list(args.zarr_list)
        dataset_dirs = resolve_dataset_dirs(args.lerobot_dir)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    actual = defaultdict(lambda: defaultdict(set))
    total_samples = 0

    print(f"Scanning {len(dataset_dirs)} LeRobot dataset(s)...")
    for dataset_dir in dataset_dirs:
        dataset = load_lerobot_dataset(dataset_dir, args.repo_id_prefix)
        for row in dataset.hf_dataset:
            dataset_name = row["source_dataset"]
            episode_index = int(row["source_episode_index"])
            frame_index = int(row["source_frame_index"])
            actual[dataset_name][episode_index].add(frame_index)
            total_samples += 1

    if len(dataset_dirs) == 1:
        group_name = dataset_dirs[0].name
        group_entries = [entry for entry in entries if entry.target_dataset == group_name]
        entries_to_check = group_entries or [entry for entry in entries if entry.dataset_name in actual]
        if group_entries:
            print(f"Using {len(entries_to_check)} zarr entries for target_dataset '{group_name}'")
    else:
        entries_to_check = entries

    print("Loading Zarr episode metadata...")
    expected = {}
    for entry in entries_to_check:
        src = open_zarr_root(entry.zarr_path)
        episode_ends = src["meta/episode_ends"][:]
        ep_frames = {}
        for ep_idx, ep_end in enumerate(episode_ends):
            ep_start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
            ep_frames[ep_idx] = int(ep_end) - ep_start
        expected[entry.dataset_name] = ep_frames
        print(f"  {entry.dataset_name}: {len(ep_frames)} episodes, {int(episode_ends[-1])} frames")

    print(f"\n{'='*60}")
    print("Completeness check:\n")

    all_ok = True
    for dataset_name, ep_frames in sorted(expected.items()):
        ds_actual = actual.get(dataset_name, {})
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
                continue

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
        actual_total = sum(len(frame_set) for frame_set in ds_actual.values())
        ok = not missing_eps and not incomplete_eps and not extra_eps and total_missing_frames == 0 and total_extra_frames == 0
        all_ok &= ok

        status = "OK" if ok else "INCOMPLETE"
        print(
            f"[{status}] {dataset_name}: {actual_total}/{expected_total} frames, "
            f"{len(ds_actual)}/{len(ep_frames)} episodes"
        )
        if missing_eps:
            shown = missing_eps[:10]
            print(f"  Missing episodes ({len(missing_eps)}): {shown}{'...' if len(missing_eps) > 10 else ''}")
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

    unknown_datasets = set(actual.keys()) - set(expected.keys())
    if unknown_datasets:
        all_ok = False
        print(f"\nUnknown datasets in LeRobot (not in zarr_list): {sorted(unknown_datasets)}")

    print(f"\n{'='*60}")
    print(f"Total: {total_samples} samples across {len(dataset_dirs)} LeRobot dataset(s)")
    if all_ok:
        print("All datasets complete.")
    else:
        print("Some datasets have missing or extra data.")


if __name__ == "__main__":
    main()
