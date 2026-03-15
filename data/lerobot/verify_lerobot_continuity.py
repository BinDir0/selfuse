"""
Verify continuity of local LeRobot datasets converted from zarr.

Checks:
1. A LeRobot episode contains frames from exactly one source episode.
2. source_frame_index starts at 0 and is contiguous within each LeRobot episode.
3. A single source episode does not appear in multiple LeRobot episodes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from _common import load_lerobot_dataset, resolve_dataset_dirs


def main():
    parser = argparse.ArgumentParser(description="Verify continuity in local LeRobot datasets")
    parser.add_argument("--lerobot_dir", type=str, required=True)
    parser.add_argument("--repo_id_prefix", type=str, default="local")
    args = parser.parse_args()

    dataset_dirs = resolve_dataset_dirs(args.lerobot_dir)
    total_errors = 0
    total_frames = 0
    total_episodes = 0
    source_to_targets = defaultdict(set)

    print(f"Found {len(dataset_dirs)} LeRobot dataset(s)\n")

    for dataset_dir in dataset_dirs:
        dataset = load_lerobot_dataset(dataset_dir, args.repo_id_prefix)
        episode_state = {}
        errors = []

        for row in dataset.hf_dataset:
            target_ep = int(row["episode_index"])
            source_dataset = row["source_dataset"]
            source_ep = int(row["source_episode_index"])
            source_frame = int(row["source_frame_index"])
            source_key = (source_dataset, source_ep)
            target_key = (dataset_dir.name, target_ep)
            source_to_targets[source_key].add(target_key)

            state = episode_state.get(target_ep)
            if state is None:
                state = {
                    "source_key": source_key,
                    "first_frame": source_frame,
                    "last_frame": source_frame,
                    "count": 1,
                }
                episode_state[target_ep] = state
                if source_frame != 0:
                    errors.append(
                        f"{dataset_dir.name}/episode-{target_ep}: starts at source_frame_index={source_frame}, expected 0"
                    )
            else:
                if state["source_key"] != source_key:
                    errors.append(
                        f"{dataset_dir.name}/episode-{target_ep}: mixes {state['source_key']} and {source_key}"
                    )
                expected = state["last_frame"] + 1
                if source_frame != expected:
                    errors.append(
                        f"{dataset_dir.name}/episode-{target_ep}: expected source_frame_index {expected}, got {source_frame}"
                    )
                state["last_frame"] = source_frame
                state["count"] += 1

        for target_ep, state in episode_state.items():
            expected_count = state["last_frame"] - state["first_frame"] + 1
            if state["count"] != expected_count:
                errors.append(
                    f"{dataset_dir.name}/episode-{target_ep}: frame range [{state['first_frame']}, {state['last_frame']}] "
                    f"but {state['count']} frames stored"
                )

        dataset_frames = sum(state["count"] for state in episode_state.values())
        total_frames += dataset_frames
        total_episodes += len(episode_state)
        total_errors += len(errors)

        status = "OK" if not errors else f"FAIL ({len(errors)} errors)"
        print(f"{dataset_dir.name}: {status} ({len(episode_state)} episodes, {dataset_frames} frames)")
        for err in errors[:5]:
            print(f"  {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    print(f"\n{'='*60}")
    print("Checking whether a source episode was split across multiple LeRobot episodes...")
    split_count = 0
    for source_key, targets in source_to_targets.items():
        if len(targets) > 1:
            split_count += 1
            if split_count <= 10:
                print(f"  {source_key[0]} episode {source_key[1]} appears in {sorted(targets)}")
    if split_count > 10:
        print(f"  ... and {split_count - 10} more")

    print(f"\n{'='*60}")
    print(f"Total: {len(dataset_dirs)} dataset(s), {total_episodes} episodes, {total_frames} frames")
    print(f"Continuity errors: {total_errors}")
    print(f"Source episode splits: {split_count}")
    if total_errors == 0 and split_count == 0:
        print("All episodes verified OK.")


if __name__ == "__main__":
    main()
