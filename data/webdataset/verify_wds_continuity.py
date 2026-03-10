"""
Verify episode continuity within WebDataset shards.

Checks:
1. Within each shard, frames of the same episode have consecutive indices (no gaps).
2. No episode is split across multiple shards.

Usage:
    python data/webdataset/verify_wds_continuity.py \
        --wds_dir /cfs/data/wds/ \
        --num_workers 16
"""

import re
import argparse
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

import webdataset as wds


KEY_PATTERN = re.compile(r"^(.+)_ep(\d+)_f(\d+)$")


def verify_shard(shard_path):
    """Check frame continuity within a single shard.

    Returns dict with shard name, errors, and episode info.
    """
    errors = []
    episode_last_frame = {}
    episode_info = {}

    dataset = wds.WebDataset(str(shard_path), shardshuffle=False)
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

    shard_frames = sum(info[2] for info in episode_info.values())
    ep_lengths = [info[2] for info in episode_info.values()]

    return {
        "shard": shard_path.name,
        "errors": errors,
        "num_episodes": len(episode_info),
        "num_frames": shard_frames,
        "episode_keys": list(episode_info.keys()),
        "ep_len_min": min(ep_lengths) if ep_lengths else 0,
        "ep_len_max": max(ep_lengths) if ep_lengths else 0,
        "ep_len_mean": sum(ep_lengths) / len(ep_lengths) if ep_lengths else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify episode continuity in WebDataset shards")
    parser.add_argument("--wds_dir", type=str, required=True,
                        help="Root directory containing WebDataset shards")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel verification workers")
    args = parser.parse_args()

    all_shards = sorted(Path(args.wds_dir).rglob("shard-*.tar"))
    print(f"Found {len(all_shards)} shards, verifying with {args.num_workers} workers\n")

    total_errors = 0
    total_episodes = 0
    total_frames = 0
    episode_to_shards = defaultdict(list)

    with mp.Pool(args.num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(verify_shard, all_shards)):
            total_frames += result["num_frames"]
            total_episodes += result["num_episodes"]

            for ep_key in result["episode_keys"]:
                episode_to_shards[ep_key].append(result["shard"])

            n_err = len(result["errors"])
            total_errors += n_err

            status = "OK" if n_err == 0 else f"FAIL ({n_err} errors)"
            print(f"[{i+1}/{len(all_shards)}] {result['shard']}: {status} "
                  f"({result['num_episodes']} eps, {result['num_frames']} frames, "
                  f"ep_len: {result['ep_len_min']}-{result['ep_len_max']}, "
                  f"mean={result['ep_len_mean']:.0f})")
            for err in result["errors"][:3]:
                print(f"  {err}")
            if n_err > 3:
                print(f"  ... and {n_err - 3} more")

    # Cross-shard split detection
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
