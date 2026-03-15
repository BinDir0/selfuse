"""
Verify numerical consistency between local LeRobot datasets and original zarr data.

Randomly samples LeRobot episodes and checks every frame against the original zarr
source arrays using source_dataset/source_episode_index/source_frame_index.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict

import numpy as np

from _common import (
    build_source_registry,
    compare_images,
    get_array,
    get_lerobot_item,
    image_to_hwc_uint8,
    load_lerobot_dataset,
    normalize_instruction,
    numeric_array_to_shape,
    parse_zarr_list,
    resolve_dataset_dirs,
    to_python_int,
    to_python_str,
)


def build_episode_registry(dataset, dataset_dir_name: str):
    episode_to_indices = defaultdict(list)
    episode_to_source = {}

    for frame_idx, row in enumerate(dataset.hf_dataset):
        target_ep = int(row["episode_index"])
        source_key = (row["source_dataset"], int(row["source_episode_index"]))
        episode_to_indices[target_ep].append(frame_idx)
        episode_to_source.setdefault(target_ep, source_key)

    registry = []
    for target_ep, frame_indices in episode_to_indices.items():
        registry.append(
            {
                "dataset_dir_name": dataset_dir_name,
                "target_episode_index": target_ep,
                "source_key": episode_to_source[target_ep],
                "frame_indices": frame_indices,
            }
        )
    return registry


def verify_feature(item, source_value, spec):
    if spec.is_visual:
        ok, max_diff, avg_diff = compare_images(item[spec.feature_name], source_value)
        if not ok:
            return False, f"{spec.feature_name}: image mismatch (max_diff={max_diff}, avg_diff={avg_diff:.2f})"
        return True, None

    if spec.is_string:
        actual = to_python_str(item[spec.feature_name])
        expected = normalize_instruction(source_value)
        if actual != expected:
            return False, f"{spec.feature_name}: string mismatch"
        return True, None

    actual = numeric_array_to_shape(item[spec.feature_name], spec)
    expected = numeric_array_to_shape(source_value, spec)
    if not np.array_equal(actual, expected):
        max_diff = float(np.abs(actual.astype(np.float64) - expected.astype(np.float64)).max())
        return False, f"{spec.feature_name}: numeric mismatch (max_diff={max_diff})"
    return True, None


def main():
    parser = argparse.ArgumentParser(description="Verify LeRobot consistency against zarr")
    parser.add_argument("--zarr_list", type=str, required=True)
    parser.add_argument("--lerobot_dir", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--repo_id_prefix", type=str, default="local")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        entries = parse_zarr_list(args.zarr_list)
        dataset_dirs = resolve_dataset_dirs(args.lerobot_dir)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    random.seed(args.seed)
    registry = build_source_registry(entries)

    datasets = {}
    episode_registry = []
    for dataset_dir in dataset_dirs:
        dataset = load_lerobot_dataset(dataset_dir, args.repo_id_prefix)
        datasets[dataset_dir.name] = dataset
        episode_registry.extend(build_episode_registry(dataset, dataset_dir.name))

    if not episode_registry:
        print("No episodes found.")
        return

    sampled = random.sample(episode_registry, min(args.num_episodes, len(episode_registry)))
    print(f"Found {len(episode_registry)} total episodes")
    print(f"Sampled {len(sampled)} episodes for verification\n")

    total_frames = 0
    total_errors = 0
    field_pass_totals = defaultdict(int)

    for index, episode_info in enumerate(sampled, 1):
        dataset = datasets[episode_info["dataset_dir_name"]]
        source_dataset, source_episode_index = episode_info["source_key"]
        source_registry = registry[source_dataset]
        source_specs = source_registry["feature_specs"]
        errors = []
        frame_count = 0

        for item_index in episode_info["frame_indices"]:
            item = get_lerobot_item(dataset, item_index)
            source_frame_index = to_python_int(item["source_frame_index"])
            abs_index = int(source_registry["episode_starts"][source_episode_index]) + source_frame_index
            data_root = source_registry["data_root"]

            if to_python_str(item["source_dataset"]) != source_dataset:
                errors.append(f"frame {item_index}: source_dataset mismatch")
                continue
            if to_python_int(item["source_episode_index"]) != source_episode_index:
                errors.append(f"frame {item_index}: source_episode_index mismatch")
                continue

            instruction_value = None
            frame_ok = True
            for spec in source_specs.values():
                source_value = get_array(data_root, spec.raw_key)[abs_index]
                ok, message = verify_feature(item, source_value, spec)
                if ok:
                    field_pass_totals[spec.feature_name] += 1
                    if spec.feature_name == "instruction":
                        instruction_value = normalize_instruction(source_value)
                else:
                    frame_ok = False
                    errors.append(f"frame {item_index}: {message}")
                    break

            if not frame_ok:
                frame_count += 1
                continue

            if instruction_value is None:
                instruction_value = normalize_instruction(get_array(data_root, "instruction")[abs_index])
            if to_python_str(item["task"]) != instruction_value:
                errors.append(f"frame {item_index}: task mismatch")
            else:
                field_pass_totals["task"] += 1

            field_pass_totals["source_dataset"] += 1
            field_pass_totals["source_episode_index"] += 1
            field_pass_totals["source_frame_index"] += 1
            frame_count += 1

        total_frames += frame_count
        total_errors += len(errors)
        status = "OK" if not errors else f"FAIL ({len(errors)} errors)"
        print(
            f"[{index}/{len(sampled)}] {episode_info['dataset_dir_name']} episode {episode_info['target_episode_index']} "
            f"<- {source_dataset}/ep{source_episode_index:06d}: {status} ({frame_count} frames)"
        )
        for err in errors[:5]:
            print(f"  ERROR: {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    print(f"\n{'='*60}")
    print(f"Total: {total_frames} frames, {total_errors} errors")
    print("Field pass totals:")
    for feature_name in sorted(field_pass_totals):
        print(f"  {feature_name}: {field_pass_totals[feature_name]}/{total_frames}")
    if total_errors == 0:
        print("All sampled frames verified OK.")


if __name__ == "__main__":
    main()
