"""
Convert Zarr datasets to local LeRobotDataset v3 format.

`zarr_list` format:
    zarr_path [human|real_world] [target_dataset]

If `target_dataset` is omitted, it defaults to the zarr directory stem.
Datasets with the same `target_dataset` are written into the same LeRobot dataset.

This converter preserves original zarr keys as LeRobot feature names, except that
`/` is replaced with `.` because LeRobot feature names cannot contain `/`.

Examples:
    python data/lerobot/convert_zarr_to_lerobot_v30.py \
        --zarr_list data/zarr_list.txt \
        --output_dir /path/to/lerobot \
        --fps 15
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from _common import (
    METADATA_FEATURE_SPECS,
    build_group_feature_specs,
    default_value_for_spec,
    episode_ranges,
    get_array,
    get_data_root,
    normalize_instruction,
    open_zarr_root,
    parse_zarr_list,
    repo_id_for_dataset,
    to_frame_value,
)


def convert_group(
    group_name: str,
    group_entries,
    output_root: Path,
    fps: int,
    repo_id_prefix: str,
    robot_type: str | None,
    use_videos: bool,
    image_writer_processes: int,
    image_writer_threads: int,
    vcodec: str,
):
    output_dir = output_root / group_name
    if output_dir.exists():
        raise ValueError(f"Output dataset already exists: {output_dir}")

    group_specs, entry_specs = build_group_feature_specs(group_entries, use_videos)
    dataset = LeRobotDataset.create(
        repo_id=repo_id_for_dataset(output_dir, repo_id_prefix),
        fps=fps,
        root=output_dir,
        robot_type=robot_type,
        features={name: spec.as_lerobot_feature() for name, spec in group_specs.items()},
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        vcodec=vcodec,
    )

    try:
        for entry in group_entries:
            src = open_zarr_root(entry.zarr_path)
            data_root = get_data_root(src)
            source_specs = entry_specs[entry.dataset_name]

            print(f"Converting {entry.zarr_path} ({entry.mapping_type}) -> {group_name}")
            for ep_start, ep_end, ep_idx in episode_ranges(src):
                source_batches = {
                    feature_name: get_array(data_root, spec.raw_key)[ep_start:ep_end]
                    for feature_name, spec in source_specs.items()
                    if spec.raw_key is not None
                }

                episode_length = ep_end - ep_start
                for frame_idx in range(episode_length):
                    frame = {}
                    for feature_name, spec in group_specs.items():
                        if spec.raw_key is None:
                            continue
                        if feature_name in source_specs:
                            frame[feature_name] = to_frame_value(
                                source_batches[feature_name][frame_idx], source_specs[feature_name]
                            )
                        else:
                            frame[feature_name] = default_value_for_spec(spec)

                    frame["source_dataset"] = entry.dataset_name
                    frame["source_episode_index"] = to_frame_value(ep_idx, METADATA_FEATURE_SPECS["source_episode_index"])
                    frame["source_frame_index"] = to_frame_value(
                        frame_idx, METADATA_FEATURE_SPECS["source_frame_index"]
                    )

                    task = normalize_instruction(frame.get("instruction", entry.dataset_name))
                    frame["task"] = task
                    dataset.add_frame(frame)

                dataset.save_episode(parallel_encoding=False)

        dataset.finalize()
    except Exception:
        try:
            dataset.finalize()
        except Exception:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description="Convert Zarr datasets to local LeRobotDataset v3")
    parser.add_argument("--zarr_list", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--repo_id_prefix", type=str, default="local")
    parser.add_argument("--robot_type", type=str, default=None)
    parser.add_argument("--use_videos", action="store_true")
    parser.add_argument("--image_writer_processes", type=int, default=0)
    parser.add_argument("--image_writer_threads", type=int, default=2)
    parser.add_argument("--vcodec", type=str, default="h264")
    args = parser.parse_args()

    try:
        entries = parse_zarr_list(args.zarr_list)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    groups = defaultdict(list)
    for entry in entries:
        groups[entry.target_dataset].append(entry)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for group_name in groups:
        output_dir = output_root / group_name
        if output_dir.exists():
            print(f"Error: output dataset already exists: {output_dir}")
            sys.exit(1)

    for group_name, group_entries in groups.items():
        try:
            convert_group(
                group_name=group_name,
                group_entries=group_entries,
                output_root=output_root,
                fps=args.fps,
                repo_id_prefix=args.repo_id_prefix,
                robot_type=args.robot_type,
                use_videos=args.use_videos,
                image_writer_processes=args.image_writer_processes,
                image_writer_threads=args.image_writer_threads,
                vcodec=args.vcodec,
            )
        except Exception as exc:
            print(f"Error while converting target_dataset '{group_name}': {exc}")
            sys.exit(1)


if __name__ == "__main__":
    main()
