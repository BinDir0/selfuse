#!/usr/bin/env python3
"""Official orchestrator for dataset preparation, inference, filtering, and build."""

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

from lib.pipeline.clip_manifest import build_manifest_records_from_descriptors, write_clip_manifest, write_shard_dir_list
from lib.pipeline.datasets import DatasetAdapterContext, get_dataset_adapter
from lib.pipeline.pipeline_config import normalize_pipeline_config


OFFICIAL_STAGE_ORDER = [
    "prepare",
    "annotate",
    "infer",
    "filter",
    "build",
    "validate",
]

INTERNAL_STAGE_ORDER = [
    "preprocess",
    "manifest",
    "annotate",
    "detect_motion",
    "slam",
    "infiller",
    "filter",
    "build",
    "validate",
]

STAGE_ALIAS_MAP = {
    "prepare": ["preprocess", "manifest"],
    "infer": ["detect_motion", "slam", "infiller"],
}

LEGACY_STAGE_NAMES = {"preprocess", "manifest", "detect_motion", "slam", "infiller"}

BATCH_INFER_NEGATIVE_BOOL_FLAGS = {
    "resume",
    "detect_half_precision",
    "depth_predict_all_frames",
    "any4d_use_amp",
}


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run the official dataset pipeline. Preferred stages: prepare, annotate, infer, filter, build, validate"
    )
    parser.add_argument("--config", type=str, required=True, help="YAML pipeline config")
    parser.add_argument(
        "--stages",
        type=str,
        default=",".join(OFFICIAL_STAGE_ORDER),
        help="Comma-separated stage list. Preferred: prepare,annotate,infer,filter,build,validate",
    )
    parser.add_argument("--run_tag", type=str, default=None, help="Optional run tag override")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume compatible stages. Currently forwarded to the build stage to skip existing non-empty shards.",
    )
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


def selected_stages(raw: str) -> dict:
    requested = [stage.strip() for stage in raw.split(",") if stage.strip()]
    valid = set(OFFICIAL_STAGE_ORDER) | set(INTERNAL_STAGE_ORDER)
    invalid = [stage for stage in requested if stage not in valid]
    if invalid:
        raise ValueError(
            f"Unknown stages: {invalid}. Valid official stages: {OFFICIAL_STAGE_ORDER}. "
            f"Legacy compatibility stages: {sorted(LEGACY_STAGE_NAMES)}"
        )

    expanded = set()
    requested_public = []
    deprecated = []
    for stage in requested:
        if stage in STAGE_ALIAS_MAP:
            canonical = stage
            internal_names = STAGE_ALIAS_MAP[stage]
        else:
            if stage in LEGACY_STAGE_NAMES:
                deprecated.append(stage)
            if stage in ("preprocess", "manifest"):
                canonical = "prepare"
            elif stage in ("detect_motion", "slam", "infiller"):
                canonical = "infer"
            else:
                canonical = stage
            internal_names = [stage]

        if canonical not in requested_public:
            requested_public.append(canonical)
        expanded.update(internal_names)

    return {
        "requested_tokens": requested,
        "requested_public": requested_public,
        "internal": [stage for stage in INTERNAL_STAGE_ORDER if stage in expanded],
        "deprecated": deprecated,
    }

def format_annotation_command(template: str, context: dict) -> list[str]:
    command = template.format(**context)
    return ["/bin/bash", "-lc", command]


def main():
    args = get_parser().parse_args()
    config = normalize_pipeline_config(load_yaml(args.config))
    stage_selection = selected_stages(args.stages)
    requested_stage_tokens = stage_selection["requested_tokens"]
    stages = stage_selection["internal"]
    public_stages = stage_selection["requested_public"]
    deprecated_stages = stage_selection["deprecated"]

    dataset_cfg = config.get("dataset", {})
    paths_cfg = config.get("paths", {})
    runtimes_cfg = config.get("runtimes", {})
    infer_cfg = config.get("infer", config.get("batch_infer", {}))
    build_cfg = config.get("build", {})
    filter_cfg = config.get("filter", {})
    adapter_cfg = config.get("adapter_config", config.get("buildai", {}))
    annotation_cfg = config.get("annotation", {})
    validation_cfg = config.get("validation", {})

    run_root = Path(paths_cfg.get("log_root", PROJECT_ROOT / "pipeline_runs"))
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "clip_manifest.jsonl"
    filtered_manifest_path = run_dir / "clip_manifest.filtered.jsonl"
    shard_dirs_list_path = run_dir / "shard_dirs.txt"
    summary_path = run_dir / "run_summary.json"
    filter_report_path = run_dir / "filter_report.json"

    adapter_name = dataset_cfg.get("adapter") or dataset_cfg.get("source_type", "buildai")
    source_type = adapter_name
    source_id = dataset_cfg.get("source_id", adapter_name)
    split = dataset_cfg.get("split", "train")
    annotation_root = paths_cfg.get("annotation_root")
    final_dataset_root = Path(paths_cfg["final_dataset_root"])
    hawor_python = runtimes_cfg["hawor_python"]
    slam_python = runtimes_cfg.get("slam_python", hawor_python)
    adapter = get_dataset_adapter(adapter_name)
    adapter_context = DatasetAdapterContext(
        project_root=PROJECT_ROOT,
        run_dir=run_dir,
        manifest_path=manifest_path,
        shard_dirs_list_path=shard_dirs_list_path,
        summary_path=summary_path,
    )
    prepared = None

    run_summary = {
        "config": str(Path(args.config).resolve()),
        "run_dir": str(run_dir.resolve()),
        "resume": bool(args.resume),
        "source_type": source_type,
        "source_id": source_id,
        "split": split,
        "manifest_path": str(manifest_path.resolve()),
        "active_manifest_path": str(manifest_path.resolve()),
        "annotation_root": annotation_root,
        "final_dataset_root": str(final_dataset_root.resolve()),
        "stages": public_stages,
        "requested_stage_tokens": requested_stage_tokens,
        "expanded_internal_stages": stages,
    }
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    active_manifest_path = manifest_path
    annotation_manifest_path = manifest_path

    if deprecated_stages:
        print(
            "Warning: legacy stage names are deprecated. "
            f"Use official stages from {OFFICIAL_STAGE_ORDER}. "
            f"Received legacy names: {sorted(set(deprecated_stages))}"
        )

    def run_logged(name: str, cmd: list[str], *, cwd: str | Path | None = None):
        print(f"\n[{name}] {shlex.join(cmd)}\n")
        stream_command(name, cmd, run_dir / f"{name}.log", cwd=cwd)

    if "preprocess" in stages:
        prepared = adapter.prepare(
            dataset_cfg=dataset_cfg,
            adapter_cfg=adapter_cfg,
            paths_cfg=paths_cfg,
            runtimes_cfg=runtimes_cfg,
            context=adapter_context,
            run_logged=run_logged,
        )

    if "manifest" in stages:
        descriptors = list(
            adapter.build_descriptors(
                dataset_cfg=dataset_cfg,
                adapter_cfg=adapter_cfg,
                paths_cfg=paths_cfg,
                context=adapter_context,
                prepared=prepared,
            )
        )
        records = build_manifest_records_from_descriptors(
            descriptors,
            source_id=source_id,
            split=split,
        )
        if not records:
            raise RuntimeError(f"No clips found while building manifest for adapter={adapter_name}")
        write_clip_manifest(records, manifest_path)
        from lib.pipeline.frame_sources import classify_descriptor_storage

        shard_root = paths_cfg.get("shard_root")
        if shard_root and source_type == "buildai":
            from lib.pipeline.clip_manifest import discover_shard_dirs

            include_dirs = None
            if prepared is not None:
                include_dirs = prepared.payload.get("include_dirs")
            if include_dirs is None:
                include_dirs = adapter_cfg.get("include_dirs") or dataset_cfg.get("include_dirs")
            shard_dirs = discover_shard_dirs(shard_root, include_dirs=include_dirs)
            write_shard_dir_list(shard_dirs, shard_dirs_list_path)
        print(
            json.dumps(
                {
                    "adapter": adapter_name,
                    "source_id": source_id,
                    "split": split,
                    "clip_count": len(records),
                    "descriptor_paths": {
                        kind: sum(1 for record in records if classify_descriptor_storage(record.descriptor) == kind)
                        for kind in sorted({classify_descriptor_storage(record.descriptor) for record in records})
                    },
                    "manifest_out": str(manifest_path.resolve()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if "annotate" in stages:
        annotation_command = annotation_cfg.get("command")
        if not annotation_command:
            raise RuntimeError("annotate stage selected but annotation.command is missing in config")
        annotation_context = adapter.resolve_annotation_context(
            dataset_cfg=dataset_cfg,
            adapter_cfg=adapter_cfg,
            paths_cfg=paths_cfg,
            context=adapter_context,
            prepared=prepared,
        )
        context = {
            "manifest": str(annotation_manifest_path),
            "active_manifest": str(active_manifest_path),
            "annotation_root": str(annotation_root or ""),
            "run_dir": str(run_dir),
            "hawor_python": hawor_python,
            "slam_python": slam_python,
            "project_root": str(PROJECT_ROOT),
        }
        context.update(annotation_context)
        run_logged("annotate", format_annotation_command(annotation_command, context))

    common_batch_args = cli_args_from_mapping(
        infer_cfg.get("common"),
        negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
    )
    if "detect_motion" in stages:
        detect_motion_cmd = [
            hawor_python,
            str(PROJECT_ROOT / "scripts" / "batch_infer.py"),
            "--descriptor_manifest",
            str(active_manifest_path),
            "--stages",
            "detect_track,motion",
            *common_batch_args,
            *cli_args_from_mapping(
                infer_cfg.get("detect_motion"),
                negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
            ),
        ]
        run_logged("detect_motion", detect_motion_cmd)

    if "slam" in stages:
        slam_cmd = [
            slam_python,
            str(PROJECT_ROOT / "scripts" / "batch_infer.py"),
            "--descriptor_manifest",
            str(active_manifest_path),
            "--stages",
            "slam",
            *common_batch_args,
            *cli_args_from_mapping(
                infer_cfg.get("slam"),
                negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
            ),
        ]
        run_logged("slam", slam_cmd)

    if "infiller" in stages:
        infiller_cmd = [
            hawor_python,
            str(PROJECT_ROOT / "scripts" / "batch_infer.py"),
            "--descriptor_manifest",
            str(active_manifest_path),
            "--stages",
            "infiller",
            *common_batch_args,
            *cli_args_from_mapping(
                infer_cfg.get("infiller"),
                negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
            ),
        ]
        run_logged("infiller", infiller_cmd)

    if "filter" in stages:
        filter_cmd = [
            hawor_python,
            str(PROJECT_ROOT / "scripts" / "filter_manifest_by_quality.py"),
            "--input_manifest",
            str(active_manifest_path),
            "--output_manifest",
            str(filtered_manifest_path),
            "--report_out",
            str(filter_report_path),
            *cli_args_from_mapping(filter_cfg),
        ]
        run_logged("filter", filter_cmd)
        active_manifest_path = filtered_manifest_path
        run_summary["active_manifest_path"] = str(active_manifest_path.resolve())
        run_summary["filter_report_path"] = str(filter_report_path.resolve())
        summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if "build" in stages:
        build_cmd = [
            hawor_python,
            str(PROJECT_ROOT / "scripts" / "build_vla_from_manifest.py"),
            "--descriptor_manifest",
            str(active_manifest_path),
            "--output_dir",
            str(final_dataset_root),
            *cli_args_from_mapping(build_cfg),
        ]
        if args.resume:
            build_cmd.append("--resume")
        if annotation_root:
            build_cmd.extend(["--annotation_root", str(annotation_root)])
        run_logged("build", build_cmd)

    if "validate" in stages:
        source_validation = adapter.validate_source(
            dataset_cfg=dataset_cfg,
            adapter_cfg=adapter_cfg,
            paths_cfg=paths_cfg,
            context=adapter_context,
            prepared=prepared,
        )
        if source_validation.summary:
            print(json.dumps({"source_validation": source_validation.summary}, ensure_ascii=False, indent=2))
        if not source_validation.ok:
            raise RuntimeError(f"Source validation failed for adapter={adapter_name}: {source_validation.summary}")
        validate_cmd = [
            hawor_python,
            str(PROJECT_ROOT / "scripts" / "validate_pipeline_run.py"),
            "--descriptor_manifest",
            str(active_manifest_path),
            "--dataset_dir",
            str(final_dataset_root),
            *cli_args_from_mapping(validation_cfg),
        ]
        if annotation_root:
            validate_cmd.extend(["--annotation_root", str(annotation_root)])
            if build_cfg.get("annotation_suffix"):
                validate_cmd.extend(["--annotation_suffix", str(build_cfg["annotation_suffix"])])
        run_logged("validate", validate_cmd)

    print(f"\nRun complete: {run_dir}")


if __name__ == "__main__":
    main()
