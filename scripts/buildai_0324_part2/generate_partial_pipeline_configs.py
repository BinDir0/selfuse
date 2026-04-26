#!/usr/bin/env python3
"""Generate per-part BuildAI 0324 part2 pipeline configs and run commands."""

from __future__ import annotations

import argparse
from pathlib import Path


CONFIG_TEMPLATE = """run_tag: {run_tag}

dataset:
  adapter: buildai
  source_id: {source_id}
  split: train
  # When using `--descriptor_manifest`, the pipeline starts from that manifest
  # and these factory bounds are ignored.
  start_factory_id: {start_factory_id}
  end_factory_id: {end_factory_id}

paths:
  shard_root: {shard_root}
  annotation_root: {annotation_root}
  final_dataset_root: {final_dataset_root}
  log_root: {log_root}

runtimes:
  hawor_python: {hawor_python}
  slam_python: {slam_python}

infer:
  common:
    gpus: {gpus}
    resume: true
    checkpoint: {checkpoint}
    infiller_weight: {infiller_weight}
  detect_motion:
    chunk_batch_size: 128
    num_workers: 16
    detect_batch_size: 128
    detect_io_workers: 8
  slam:
    any4d_batch_size: 32
    depth_predict_all_frames: true
    any4d_repo_root: {any4d_repo_root}
    any4d_checkpoint_path: {any4d_checkpoint_path}
  infiller:
    infiller_window_batch_size: 64

filter:
  stages: detect_track,motion,slam,infiller
  workers: 8
  drop_nonfinite_world_res: true
  drop_nonfinite_slam: true
  # max_hand_translation_step: 0.35
  # max_camera_translation_step: 0.5
  # max_camera_rotation_step: 1.0

build:
  require_annotation: {require_annotation}
  preprocess_workers: 8
  writer_workers: 4
  frames_per_shard: 10000
  repeat_episodes: 1
  mano_device: cuda:0
  source_fps: 30.0
  target_fps: 30.0
  interpolate_labels: false
  export_depth: true

validation:
  max_clips: 200
  dataset_sample_checks: 20
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_dir",
        default="configs/generated_buildai_0324_part2",
        help="Directory where per-part config yaml files will be written",
    )
    parser.add_argument(
        "--commands_out",
        default=None,
        help="Optional shell script path for generated run commands; defaults to <config_dir>/run_partial_pipeline.sh",
    )
    parser.add_argument(
        "--manifest_prefix",
        required=True,
        help="Prefix for partition manifests, e.g. /path/buildai_part2 so manifests become <prefix>.part0000.jsonl",
    )
    parser.add_argument(
        "--part_count",
        type=int,
        default=8,
        help="How many part configs to generate",
    )
    parser.add_argument(
        "--config_prefix",
        default="dataset_pipeline_buildai_100k_0324_part2",
        help="Generated config filename prefix",
    )
    parser.add_argument(
        "--run_tag_prefix",
        default="buildai_0324_part2",
        help="Run-tag prefix for generated configs",
    )
    parser.add_argument(
        "--final_dataset_root_base",
        default="/share_data/guantianrui/datasets/Egocentric-100K/BuildAI-100k-part2",
        help="Base output root; each part writes to <base>/partXXXX",
    )
    parser.add_argument(
        "--source_id",
        default="buildai_100k_0324_part2",
        help="dataset.source_id value",
    )
    parser.add_argument(
        "--start_factory_id",
        type=int,
        default=1,
        help="dataset.start_factory_id value",
    )
    parser.add_argument(
        "--end_factory_id",
        type=int,
        default=238,
        help="dataset.end_factory_id value",
    )
    parser.add_argument(
        "--shard_root",
        default="/share_data/guantianrui/datasets/Egocentric-100K/processed_0324_jpg",
        help="paths.shard_root value",
    )
    parser.add_argument(
        "--annotation_root",
        default="/share_data/guantianrui/datasets/Egocentric-100K/annotations_0324",
        help="paths.annotation_root value",
    )
    parser.add_argument(
        "--log_root",
        default="/share_data/guantianrui/dataset_pipeline_logs",
        help="paths.log_root value",
    )
    parser.add_argument(
        "--hawor_python",
        default="/share_data/guantianrui/environment/anaconda3/envs/hawor/bin/python3.10",
        help="runtimes.hawor_python value",
    )
    parser.add_argument(
        "--slam_python",
        default="/share_data/guantianrui/environment/anaconda3/envs/any4d/bin/python3.10",
        help="runtimes.slam_python value",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="infer.common.gpus value")
    parser.add_argument(
        "--checkpoint",
        default="/share_data/guantianrui/webhaworset/facHaWoRy/weights/hawor/checkpoints/hawor.ckpt",
        help="infer.common.checkpoint value",
    )
    parser.add_argument(
        "--infiller_weight",
        default="/share_data/guantianrui/webhaworset/facHaWoRy/weights/hawor/checkpoints/infiller.pt",
        help="infer.common.infiller_weight value",
    )
    parser.add_argument(
        "--any4d_repo_root",
        default="/share_data/guantianrui/RoWaH/thirdparty/Any4D",
        help="infer.slam.any4d_repo_root value",
    )
    parser.add_argument(
        "--any4d_checkpoint_path",
        default="/share_data/guantianrui/Any4D/checkpoints/any4d_4v_combined.pth",
        help="infer.slam.any4d_checkpoint_path value",
    )
    parser.add_argument(
        "--require_annotation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether build.require_annotation should be true",
    )
    parser.add_argument(
        "--stages",
        default="slam,infiller,filter,build,validate",
        help="Stage list written into generated run commands",
    )
    return parser


def _bool_yaml(value: bool) -> str:
    return "true" if value else "false"


def main() -> None:
    args = build_parser().parse_args()
    if args.part_count < 1:
        raise ValueError("--part_count must be >= 1")

    config_dir = Path(args.config_dir).expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    commands_out = Path(args.commands_out).expanduser().resolve() if args.commands_out else config_dir / "run_partial_pipeline.sh"

    commands = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    generated_configs = []

    for part_id in range(args.part_count):
        part_name = f"part{part_id:04d}"
        config_name = f"{args.config_prefix}_{part_name}.yaml"
        config_path = config_dir / config_name
        manifest_path = f"{args.manifest_prefix}.{part_name}.jsonl"
        final_dataset_root = f"{args.final_dataset_root_base}/{part_name}"
        run_tag = f"{args.run_tag_prefix}_{part_name}"

        config_text = CONFIG_TEMPLATE.format(
            run_tag=run_tag,
            source_id=args.source_id,
            start_factory_id=args.start_factory_id,
            end_factory_id=args.end_factory_id,
            shard_root=args.shard_root,
            annotation_root=args.annotation_root,
            final_dataset_root=final_dataset_root,
            log_root=args.log_root,
            hawor_python=args.hawor_python,
            slam_python=args.slam_python,
            gpus=args.gpus,
            checkpoint=args.checkpoint,
            infiller_weight=args.infiller_weight,
            any4d_repo_root=args.any4d_repo_root,
            any4d_checkpoint_path=args.any4d_checkpoint_path,
            require_annotation=_bool_yaml(bool(args.require_annotation)),
        )
        config_path.write_text(config_text, encoding="utf-8")
        generated_configs.append(str(config_path))

        commands.extend(
            [
                f"# {part_name}",
                "sudo nice -n -15 "
                f"{args.hawor_python} "
                "scripts/run_dataset_pipeline.py "
                f"--config {config_path} "
                f"--descriptor_manifest {manifest_path} "
                f"--stages {args.stages}",
                "",
            ]
        )

    commands_out.write_text("\n".join(commands) + "\n", encoding="utf-8")

    print("Generated configs:")
    for path_text in generated_configs:
        print(path_text)
    print()
    print(f"Run commands: {commands_out}")


if __name__ == "__main__":
    main()
