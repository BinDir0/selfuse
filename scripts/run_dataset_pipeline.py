#!/usr/bin/env python3
"""Production orchestrator for raw-source -> clip shards -> HaWoR -> final dataset."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


STAGE_ORDER = [
    "preprocess",
    "manifest",
    "detect_motion",
    "slam",
    "infiller",
    "annotate",
    "build",
    "validate",
]

BATCH_INFER_NEGATIVE_BOOL_FLAGS = {
    "resume",
    "detect_half_precision",
    "depth_predict_all_frames",
    "any4d_use_amp",
}


def get_parser():
    parser = argparse.ArgumentParser(description="Run the whole dataset production pipeline")
    parser.add_argument("--config", type=str, required=True, help="YAML pipeline config")
    parser.add_argument(
        "--stages",
        type=str,
        default=",".join(STAGE_ORDER),
        help="Comma-separated stage list",
    )
    parser.add_argument("--run_tag", type=str, default=None, help="Optional run tag override")
    return parser


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def cli_args_from_mapping(mapping: dict | None, *, negative_bool_flags: set[str] | None = None) -> list[str]:
    args = []
    negative_bool_flags = negative_bool_flags or set()
    for key, value in (mapping or {}).items():
        flag = f"--{key}"
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(flag)
            elif key in negative_bool_flags:
                args.append(f"--no-{key}")
            continue
        if isinstance(value, list):
            args.append(flag)
            args.extend(str(item) for item in value)
            continue
        args.extend([flag, str(value)])
    return args


def stream_command(name: str, cmd: list[str], log_path: Path, *, cwd: str | Path | None = None, env: dict | None = None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + shlex.join(cmd) + "\n\n")
        log_handle.flush()

        process = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"{name} failed with exit code {return_code}")


def selected_stages(raw: str) -> list[str]:
    stages = [stage.strip() for stage in raw.split(",") if stage.strip()]
    invalid = [stage for stage in stages if stage not in STAGE_ORDER]
    if invalid:
        raise ValueError(f"Unknown stages: {invalid}. Valid stages: {STAGE_ORDER}")
    return stages


def buildai_group_names(start_factory_id: int, end_factory_id: int) -> list[str]:
    return [f"factory{factory_id:03d}" for factory_id in range(start_factory_id, end_factory_id + 1)]


def format_annotation_command(template: str, context: dict) -> list[str]:
    command = template.format(**context)
    return ["/bin/bash", "-lc", command]


def main():
    args = get_parser().parse_args()
    config = load_yaml(args.config)
    stages = selected_stages(args.stages)

    dataset_cfg = config.get("dataset", {})
    paths_cfg = config.get("paths", {})
    runtimes_cfg = config.get("runtimes", {})
    batch_cfg = config.get("batch_infer", {})
    build_cfg = config.get("build", {})
    buildai_cfg = config.get("buildai", {})
    annotation_cfg = config.get("annotation", {})
    validation_cfg = config.get("validation", {})

    run_root = Path(paths_cfg.get("log_root", PROJECT_ROOT / "pipeline_runs"))
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "clip_manifest.jsonl"
    shard_dirs_list_path = run_dir / "shard_dirs.txt"
    summary_path = run_dir / "run_summary.json"

    source_type = dataset_cfg.get("source_type", "buildai")
    source_id = dataset_cfg.get("source_id", source_type)
    split = dataset_cfg.get("split", "train")
    shard_root = Path(paths_cfg["shard_root"])
    annotation_root = paths_cfg.get("annotation_root")
    final_dataset_root = Path(paths_cfg["final_dataset_root"])
    hawor_python = runtimes_cfg["hawor_python"]
    slam_python = runtimes_cfg.get("slam_python", hawor_python)
    buildai_repo_root = Path(paths_cfg.get("buildai_repo_root", "/root/buildai_processing"))
    buildai_config = paths_cfg.get("buildai_config")

    run_summary = {
        "config": str(Path(args.config).resolve()),
        "run_dir": str(run_dir.resolve()),
        "source_type": source_type,
        "source_id": source_id,
        "split": split,
        "manifest_path": str(manifest_path.resolve()),
        "annotation_root": annotation_root,
        "final_dataset_root": str(final_dataset_root.resolve()),
        "stages": stages,
    }
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def run_logged(name: str, cmd: list[str], *, cwd: str | Path | None = None):
        print(f"\n[{name}] {shlex.join(cmd)}\n")
        stream_command(name, cmd, run_dir / f"{name}.log", cwd=cwd)

    if "preprocess" in stages:
        if source_type != "buildai":
            raise RuntimeError("preprocess stage currently supports source_type=buildai only")
        start_factory_id = int(dataset_cfg["start_factory_id"])
        end_factory_id = int(dataset_cfg["end_factory_id"])
        preprocess_cmd = [
            runtimes_cfg.get("buildai_shell", "/bin/bash"),
            str(buildai_repo_root / "run_buildai_pipeline.sh"),
            "--config",
            str(buildai_config),
            "--start-factory-id",
            str(start_factory_id),
            "--end-factory-id",
            str(end_factory_id),
            "--stages",
            buildai_cfg.get("stages", "1,2,3"),
        ]
        if buildai_cfg.get("setup_decord"):
            preprocess_cmd.append("--setup-decord")
        if buildai_cfg.get("clean_stage3_output"):
            preprocess_cmd.append("--clean-stage3-output")
        run_logged("preprocess", preprocess_cmd, cwd=buildai_repo_root)

    if "manifest" in stages:
        manifest_cmd = [
            hawor_python,
            str(PROJECT_ROOT / "scripts" / "build_clip_manifest.py"),
            "--shard_root",
            str(shard_root),
            "--source_id",
            source_id,
            "--split",
            split,
            "--manifest_out",
            str(manifest_path),
            "--shard_dirs_out",
            str(shard_dirs_list_path),
        ]
        if source_type == "buildai":
            include_dirs = buildai_group_names(
                int(dataset_cfg["start_factory_id"]),
                int(dataset_cfg["end_factory_id"]),
            )
            manifest_cmd.append("--include_dirs")
            manifest_cmd.extend(include_dirs)
        elif dataset_cfg.get("include_dirs"):
            manifest_cmd.append("--include_dirs")
            manifest_cmd.extend(str(item) for item in dataset_cfg["include_dirs"])
        run_logged("manifest", manifest_cmd)

    common_batch_args = cli_args_from_mapping(
        batch_cfg.get("common"),
        negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
    )
    if "detect_motion" in stages:
        detect_motion_cmd = [
            hawor_python,
            str(PROJECT_ROOT / "scripts" / "batch_infer.py"),
            "--descriptor_manifest",
            str(manifest_path),
            "--stages",
            "detect_track,motion",
            *common_batch_args,
            *cli_args_from_mapping(
                batch_cfg.get("detect_motion"),
                negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
            ),
        ]
        run_logged("detect_motion", detect_motion_cmd)

    if "slam" in stages:
        slam_cmd = [
            slam_python,
            str(PROJECT_ROOT / "scripts" / "batch_infer.py"),
            "--descriptor_manifest",
            str(manifest_path),
            "--stages",
            "slam",
            *common_batch_args,
            *cli_args_from_mapping(
                batch_cfg.get("slam"),
                negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
            ),
        ]
        run_logged("slam", slam_cmd)

    if "infiller" in stages:
        infiller_cmd = [
            hawor_python,
            str(PROJECT_ROOT / "scripts" / "batch_infer.py"),
            "--descriptor_manifest",
            str(manifest_path),
            "--stages",
            "infiller",
            *common_batch_args,
            *cli_args_from_mapping(
                batch_cfg.get("infiller"),
                negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
            ),
        ]
        run_logged("infiller", infiller_cmd)

    if "annotate" in stages:
        annotation_command = annotation_cfg.get("command")
        if not annotation_command:
            raise RuntimeError("annotate stage selected but annotation.command is missing in config")
        context = {
            "manifest": str(manifest_path),
            "annotation_root": str(annotation_root or ""),
            "run_dir": str(run_dir),
            "hawor_python": hawor_python,
            "slam_python": slam_python,
            "project_root": str(PROJECT_ROOT),
        }
        run_logged("annotate", format_annotation_command(annotation_command, context))

    if "build" in stages:
        build_cmd = [
            hawor_python,
            str(PROJECT_ROOT / "scripts" / "build_vla_from_manifest.py"),
            "--descriptor_manifest",
            str(manifest_path),
            "--output_dir",
            str(final_dataset_root),
            *cli_args_from_mapping(build_cfg),
        ]
        if annotation_root:
            build_cmd.extend(["--annotation_root", str(annotation_root)])
        run_logged("build", build_cmd)

    if "validate" in stages:
        validate_cmd = [
            hawor_python,
            str(PROJECT_ROOT / "scripts" / "validate_pipeline_run.py"),
            "--descriptor_manifest",
            str(manifest_path),
            "--dataset_dir",
            str(final_dataset_root),
            *cli_args_from_mapping(validation_cfg),
        ]
        if annotation_root:
            validate_cmd.extend(["--annotation_root", str(annotation_root)])
        run_logged("validate", validate_cmd)

    print(f"\nRun complete: {run_dir}")


if __name__ == "__main__":
    main()
