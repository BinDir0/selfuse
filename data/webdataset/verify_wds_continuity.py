"""
Verify episode continuity within WebDataset shards.

Checks:
1. Within each shard, frames of the same episode have consecutive indices (no gaps).
2. No episode is split across multiple shards.

Usage:
    python data/verify_wds_continuity.py --wds_dir /cfs/data/wds/
"""

import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

import webdataset as wds


KEY_PATTERN = re.compile(r"^(.+)_ep(\d+)_f(\d+)$")


def verify_shard_continuity(shard_path):
    """Check frame continuity within a single shard.

    Returns:
        errors: list of error messages
        episodes: dict of (dataset_name, ep_idx) -> (min_frame, max_frame, count)
    """
    errors = []
    # Track per-episode: last seen frame index
    episode_last_frame = {}
    # Track per-episode: frame count and range
    episode_info = {}

    dataset = wds.WebDataset(str(shard_path))
    for sample in dataset:
        key = sample["__key__"]
        m = KEY_PATTERN.match(key)
        if not m:
            errors.append(f"Cannot parse key: {key}")
            continue

        ds_name, ep_idx, frame_idx = m.group(1), int(m.group(2)), int(m.group(3))
        ep_key = (ds_name, ep_idx)

        if ep_key in episode_last_frame:
            expected = episode_last_frame[ep_key] + 1
            if frame_idx != expected:
                errors.append(
                    f"{key}: expected frame {expected}, got {frame_idx} "
                    f"(gap or out-of-order)")
        else:
            # First frame of this episode in this shard
            if frame_idx != 0:
                errors.append(
                    f"{key}: episode starts at frame {frame_idx}, expected 0 "
                    f"(possible cross-shard split)")

        episode_last_frame[ep_key] = frame_idx

        if ep_key not in episode_info:
            episode_info[ep_key] = [frame_idx, frame_idx, 0]
        info = episode_info[ep_key]
        info[0] = min(info[0], frame_idx)
        info[1] = max(info[1], frame_idx)
        info[2] += 1

    # Verify frame count matches range
    for ep_key, (min_f, max_f, count) in episode_info.items():
        expected_count = max_f - min_f + 1
        if count != expected_count:
            errors.append(
                f"{ep_key[0]}_ep{ep_key[1]:06d}: "
                f"frame range [{min_f}, {max_f}] but only {count} frames "
                f"(expected {expected_count})")

    return errors, episode_info


def main():
    parser = argparse.ArgumentParser(
        description="Verify episode continuity in WebDataset shards")
    parser.add_argument("--wds_dir", type=str, required=True,
                        help="Root directory containing WebDataset shards")
    args = parser.parse_args()

    all_shards = sorted(Path(args.wds_dir).rglob("shard-*.tar"))
    print(f"Found {len(all_shards)} shards\n")

    total_errors = 0
    total_episodes = 0
    total_frames = 0

    # Track which shard each episode appears in (for cross-shard split detection)
    episode_to_shards = defaultdict(list)

    for i, shard_path in enumerate(all_shards):
        errors, episode_info = verify_shard_continuity(shard_path)

        shard_frames = sum(info[2] for info in episode_info.values())
        total_frames += shard_frames
        total_episodes += len(episode_info)

        for ep_key in episode_info:
            episode_to_shards[ep_key].append(shard_path.name)

        if errors:
            total_errors += len(errors)
            print(f"[{i+1}/{len(all_shards)}] {shard_path.name}: "
                  f"FAIL ({len(errors)} errors, "
                  f"{len(episode_info)} eps, {shard_frames} frames)")
            for err in errors[:5]:
                print(f"  {err}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more")
        else:
            print(f"[{i+1}/{len(all_shards)}] {shard_path.name}: "
                  f"OK ({len(episode_info)} eps, {shard_frames} frames)")

    # Check cross-shard episode splits
    print(f"\n{'='*60}")
    print("Checking cross-shard episode splits...")
    split_count = 0
    for ep_key, shards in episode_to_shards.items():
        if len(shards) > 1:
            split_count += 1
            if split_count <= 10:
                print(f"  {ep_key[0]}_ep{ep_key[1]:06d} appears in "
                      f"{len(shards)} shards: {shards}")
    if split_count > 10:
        print(f"  ... and {split_count - 10} more")

    print(f"\n{'='*60}")
    print(f"Total: {len(all_shards)} shards, {total_episodes} episodes, "
          f"{total_frames} frames")
    print(f"Continuity errors: {total_errors}")
    print(f"Cross-shard splits: {split_count}")
    if total_errors == 0 and split_count == 0:
        print("All episodes verified OK.")


if __name__ == "__main__":
    main()
