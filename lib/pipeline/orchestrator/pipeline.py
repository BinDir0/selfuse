"""Official dataset pipeline orchestration."""

from __future__ import annotations

import json
import shlex
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.clip_manifest import (
    build_manifest_records_from_descriptors,
    load_clip_manifest,
    write_clip_manifest,
    write_shard_dir_list,
)
from lib.pipeline.batch.state import load_status_payload_with_fallback
from lib.pipeline.datasets import DatasetAdapterContext, get_dataset_adapter
from lib.pipeline.frame_sources import classify_descriptor_storage
from lib.pipeline.multihost import (
    MultihostStageQueueRunner,
    MultihostStageSpec,
    parse_multihost_config,
    sanitize_infer_args_for_multihost,
)
from lib.pipeline.pipeline_config import normalize_pipeline_config
from lib.pipeline.stage_api import get_stage_done_marker

from .cli import get_parser
from .constants import BATCH_INFER_NEGATIVE_BOOL_FLAGS, MULTIHOST_DISALLOWED_INFER_KEYS, OFFICIAL_STAGE_ORDER
from .helpers import cli_args_from_mapping, format_annotation_command, load_yaml, stream_command
from .stage_selection import selected_stages
from .validation import (
    infer_stage_worker_count_per_gpu,
    validate_multihost_infer_alignment,
    validate_pipeline_cli_alignment,
)


def _resolve_runtime_python_path(raw_path: str | None, *, runtime_name: str) -> str | None:
    if raw_path is None:
        return None
    path_text = str(raw_path)
    candidate = Path(path_text)
    if candidate.exists():
        return path_text

    # Backward-compatibility shim for the old Any4D env typo: `any4` -> `any4d`.
    fixed_text = path_text.replace("/envs/any4/", "/envs/any4d/")
    if fixed_text != path_text:
        fixed_candidate = Path(fixed_text)
        if fixed_candidate.exists():
            print(
                f"[runtime] {runtime_name} python not found at {path_text}; "
                f"using compatible fallback {fixed_text}",
                flush=True,
            )
            return fixed_text
    return path_text


def run_pipeline(args) -> None:
    config_path = Path(args.config).resolve()
    config = normalize_pipeline_config(load_yaml(config_path))
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

    validate_pipeline_cli_alignment(
        stages=stages,
        infer_cfg=infer_cfg,
        build_cfg=build_cfg,
        filter_cfg=filter_cfg,
        validation_cfg=validation_cfg,
    )

    run_root = Path(paths_cfg.get("log_root", PROJECT_ROOT / "pipeline_runs"))
    run_tag = args.run_tag or config.get("run_tag") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "clip_manifest.jsonl"
    filtered_manifest_path = run_dir / "clip_manifest.filtered.jsonl"
    shard_dirs_list_path = run_dir / "shard_dirs.txt"
    summary_path = run_dir / "run_summary.json"
    filter_report_path = run_dir / "filter_report.json"
    shared_feature_cache_dir = run_dir / "_episode_feature_cache"

    adapter_name = dataset_cfg.get("adapter") or dataset_cfg.get("source_type", "buildai")
    source_type = adapter_name
    source_id = dataset_cfg.get("source_id", adapter_name)
    split = dataset_cfg.get("split", "train")
    annotation_root = paths_cfg.get("annotation_root")
    final_dataset_root = Path(paths_cfg["final_dataset_root"])
    hawor_python = _resolve_runtime_python_path(runtimes_cfg["hawor_python"], runtime_name="hawor")
    slam_python = _resolve_runtime_python_path(runtimes_cfg.get("slam_python", hawor_python), runtime_name="slam")
    infer_multihost_cfg = parse_multihost_config(
        infer_cfg.get("multihost"),
        default_project_root=PROJECT_ROOT,
        default_hawor_python=hawor_python,
        default_slam_python=slam_python,
    )
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
        "config": str(config_path),
        "run_dir": str(run_dir.resolve()),
        "resume": bool(args.resume),
        "source_type": source_type,
        "source_id": source_id,
        "split": split,
        "manifest_path": str(manifest_path.resolve()),
        "active_manifest_path": str(manifest_path.resolve()),
        "annotation_root": annotation_root,
        "final_dataset_root": str(final_dataset_root.resolve()),
        "feature_cache_dir": str(shared_feature_cache_dir.resolve()),
        "stages": public_stages,
        "requested_stage_tokens": requested_stage_tokens,
        "expanded_internal_stages": stages,
    }
    if infer_multihost_cfg.enabled:
        run_summary["infer_multihost"] = infer_multihost_cfg.to_summary()
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    active_manifest_path = manifest_path
    annotation_manifest_path = manifest_path

    if deprecated_stages:
        print(
            "Warning: legacy stage names are deprecated. "
            f"Use official stages from {OFFICIAL_STAGE_ORDER}. "
            f"Received legacy names: {sorted(set(deprecated_stages))}"
        )

    def run_logged(
        name: str,
        cmd: list[str],
        *,
        cwd: str | Path | None = None,
        raise_on_error: bool = True,
    ) -> int:
        print(f"\n[{name}] {shlex.join(cmd)}\n")
        return stream_command(name, cmd, run_dir / f"{name}.log", cwd=cwd, raise_on_error=raise_on_error)

    def ensure_manifest_exists(stage_label: str, manifest_to_check: Path) -> None:
        if manifest_to_check.exists():
            return
        raise RuntimeError(
            f"{stage_label} requires descriptor manifest: {manifest_to_check}\n"
            "Run the manifest stage first, or reuse the previous run directory via --run_tag."
        )

    def manifest_uses_only_native_features(manifest_to_check: Path) -> bool:
        from lib.pipeline.exporters.manifest_vla import descriptor_uses_native_features

        records = load_clip_manifest(manifest_to_check)
        return bool(records) and all(descriptor_uses_native_features(record.descriptor) for record in records)

    def build_completed_stage_manifest(stage_name: str, source_manifest: Path) -> tuple[Path, dict]:
        status_path = run_dir / "status.json"
        events_path = run_dir / "events.jsonl"
        if not status_path.exists() and not events_path.exists():
            raise RuntimeError(f"Missing status.json after {stage_name}: {status_path}")
        source_records = load_clip_manifest(source_manifest)
        status_payload, status_meta = load_status_payload_with_fallback(
            status_path,
            events_path=events_path,
            video_paths=[record.descriptor.video_key for record in source_records],
            stages=[stage_name],
        )
        if status_payload is None:
            raise RuntimeError(f"Missing recoverable batch status after {stage_name}: {status_path}")
        if status_meta.get("source") != "status":
            print(
                f"[{stage_name}] recovered completed-stage manifest state from {status_meta.get('source')}",
                flush=True,
            )
        tasks = status_payload.get("tasks", {})
        completed_records = []
        failed_clip_ids = []
        incomplete_clip_ids = []

        for record in source_records:
            task = tasks.get(record.descriptor.video_key) or tasks.get(record.clip_id) or {}
            stage_status = (task.get("stage_status") or {}).get(stage_name)
            if stage_status != "completed" and get_stage_done_marker(Path(record.descriptor.seq_folder), stage_name).exists():
                stage_status = "completed"

            if stage_status == "completed":
                completed_records.append(record)
            elif stage_status == "failed":
                failed_clip_ids.append(record.clip_id)
            else:
                incomplete_clip_ids.append(record.clip_id)

        subset_path = run_dir / f"{source_manifest.stem}.{stage_name}.completed.jsonl"
        write_clip_manifest(completed_records, subset_path)
        summary = {
            "source_manifest": str(source_manifest.resolve()),
            "completed_manifest": str(subset_path.resolve()),
            "total": len(source_records),
            "completed": len(completed_records),
            "failed": len(failed_clip_ids),
            "incomplete": len(incomplete_clip_ids),
            "failed_clip_ids_preview": failed_clip_ids[:16],
            "incomplete_clip_ids_preview": incomplete_clip_ids[:16],
        }
        return subset_path, summary

    def handle_partial_infer_stage(
        *,
        stage_label: str,
        completed_stage_name: str,
        source_manifest: Path,
        return_code: int,
    ) -> Path:
        status_path = run_dir / "status.json"
        if not status_path.exists():
            raise RuntimeError(
                f"{stage_label} failed with exit code {return_code} before status.json was created. "
                f"Check {run_dir / f'{stage_label}.log'}."
            )
        subset_manifest, subset_summary = build_completed_stage_manifest(completed_stage_name, source_manifest)
        run_summary.setdefault("infer_stage_manifests", {})[stage_label] = subset_summary
        run_summary["active_manifest_path"] = str(subset_manifest.resolve())
        summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        if subset_summary["completed"] <= 0:
            raise RuntimeError(
                f"{stage_label} failed with exit code {return_code} and produced no successful clips. "
                f"Summary: {json.dumps(subset_summary, ensure_ascii=False)}"
            )
        if return_code != 0:
            print(
                f"[{stage_label}] partial failure tolerated: exit_code={return_code}, "
                f"continuing with completed subset {subset_summary['completed']}/{subset_summary['total']}",
                flush=True,
            )
        return subset_manifest

    def handle_partial_external_stage(
        *,
        stage_label: str,
        source_manifest: Path,
        return_code: int,
        output_exists,
    ) -> Path:
        source_records = load_clip_manifest(source_manifest)
        completed_records = []
        incomplete_clip_ids = []

        for record in source_records:
            seq_folder = Path(record.descriptor.seq_folder)
            if output_exists(seq_folder):
                completed_records.append(record)
            else:
                incomplete_clip_ids.append(record.clip_id)

        subset_path = run_dir / f"{source_manifest.stem}.{stage_label}.completed.jsonl"
        write_clip_manifest(completed_records, subset_path)
        summary = {
            "source_manifest": str(source_manifest.resolve()),
            "completed_manifest": str(subset_path.resolve()),
            "total": len(source_records),
            "completed": len(completed_records),
            "incomplete": len(incomplete_clip_ids),
            "incomplete_clip_ids_preview": incomplete_clip_ids[:16],
        }
        run_summary.setdefault("infer_stage_manifests", {})[stage_label] = summary
        run_summary["active_manifest_path"] = str(subset_path.resolve())
        summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        if summary["completed"] <= 0:
            raise RuntimeError(
                f"{stage_label} failed with exit code {return_code} and produced no successful clips. "
                f"Summary: {json.dumps(summary, ensure_ascii=False)}"
            )
        print(
            f"[{stage_label}] partial failure tolerated: exit_code={return_code}, "
            f"continuing with completed subset {summary['completed']}/{summary['total']}",
            flush=True,
        )
        return subset_path

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
        ensure_manifest_exists("annotate", annotation_manifest_path)
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
    native_depth_cfg = infer_cfg.get("native_depth") or {}
    native_infer_stages = [stage for stage in ("detect_motion", "slam", "infiller") if stage in stages]
    if native_infer_stages:
        ensure_manifest_exists("infer", active_manifest_path)
        if manifest_uses_only_native_features(active_manifest_path):
            print(
                "[infer] native-feature manifest detected; skipping ordinary "
                f"infer sub-stages {native_infer_stages}. "
                "HOT3D final build reads lowdim/mano/cameras directly from raw WDS.",
                flush=True,
            )
            stages = [stage for stage in stages if stage not in set(native_infer_stages)]
            run_summary["native_feature_infer_skip"] = {
                "skipped_internal_stages": native_infer_stages,
                "reason": "native lowdim/mano/camera features are provided by source WDS",
            }
            if bool(native_depth_cfg.get("enabled")) and "native_depth" not in stages:
                slam_idx = stages.index("slam") if "slam" in stages else None
                if slam_idx is not None:
                    stages.insert(slam_idx + 1, "native_depth")
                else:
                    stages.append("native_depth")
                run_summary["native_feature_infer_skip"]["appended_internal_stages"] = ["native_depth"]
            run_summary["expanded_internal_stages_after_native_skip"] = stages
            summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    multihost_runner = None
    multihost_common_batch_args = ()
    if infer_multihost_cfg.enabled and any(stage in stages for stage in ("detect_motion", "slam", "infiller")):
        validate_multihost_infer_alignment(infer_cfg)
        multihost_common_batch_args = tuple(
            cli_args_from_mapping(
                sanitize_infer_args_for_multihost(
                    infer_cfg.get("common"),
                    reserved_keys=MULTIHOST_DISALLOWED_INFER_KEYS,
                ),
                negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
            )
        )
        multihost_runner = MultihostStageQueueRunner(
            config=infer_multihost_cfg,
            manifest_path=active_manifest_path,
            run_dir=run_dir,
            infer_resume=bool((infer_cfg.get("common") or {}).get("resume", True)),
        )

    if "detect_motion" in stages:
        ensure_manifest_exists("detect_motion", active_manifest_path)
        detect_motion_args = tuple(
            cli_args_from_mapping(
                infer_cfg.get("detect_motion"),
                negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
            )
        )
        if multihost_runner is not None:
            result = multihost_runner.run_stage(
                MultihostStageSpec(
                    pipeline_stage="detect_motion",
                    batch_stages="detect_track,motion",
                    runtime_key="hawor",
                    extra_args=multihost_common_batch_args
                    + tuple(
                        cli_args_from_mapping(
                            sanitize_infer_args_for_multihost(
                                infer_cfg.get("detect_motion"),
                                reserved_keys=MULTIHOST_DISALLOWED_INFER_KEYS,
                            ),
                            negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
                        )
                    ),
                    worker_count_per_gpu=infer_stage_worker_count_per_gpu(
                        pipeline_stage="detect_motion",
                        infer_cfg=infer_cfg,
                    ),
                )
            )
            run_summary.setdefault("multihost_dispatch", {})["detect_motion"] = result["dispatch_path"]
            summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
            if not result["success"]:
                raise RuntimeError(
                    "detect_motion multihost stage failed: "
                    + json.dumps(result["failed_shards"], ensure_ascii=False)
                )
        else:
            detect_motion_return_code = run_logged(
                "detect_motion",
                [
                    hawor_python,
                    str(PROJECT_ROOT / "scripts" / "batch_infer.py"),
                    "--descriptor_manifest",
                    str(active_manifest_path),
                    "--run_dir",
                    str(run_dir),
                    "--stages",
                    "detect_track,motion",
                    *common_batch_args,
                    *detect_motion_args,
                ],
                raise_on_error=False,
            )
            active_manifest_path = handle_partial_infer_stage(
                stage_label="detect_motion",
                completed_stage_name="motion",
                source_manifest=active_manifest_path,
                return_code=detect_motion_return_code,
            )

    if "slam" in stages:
        ensure_manifest_exists("slam", active_manifest_path)
        slam_args = tuple(
            cli_args_from_mapping(
                infer_cfg.get("slam"),
                negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
            )
        )
        if multihost_runner is not None:
            result = multihost_runner.run_stage(
                MultihostStageSpec(
                    pipeline_stage="slam",
                    batch_stages="slam",
                    runtime_key="slam",
                    extra_args=multihost_common_batch_args
                    + tuple(
                        cli_args_from_mapping(
                            sanitize_infer_args_for_multihost(
                                infer_cfg.get("slam"),
                                reserved_keys=MULTIHOST_DISALLOWED_INFER_KEYS,
                            ),
                            negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
                        )
                    ),
                    worker_count_per_gpu=infer_stage_worker_count_per_gpu(
                        pipeline_stage="slam",
                        infer_cfg=infer_cfg,
                    ),
                )
            )
            run_summary.setdefault("multihost_dispatch", {})["slam"] = result["dispatch_path"]
            summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
            if not result["success"]:
                raise RuntimeError(
                    "slam multihost stage failed: "
                    + json.dumps(result["failed_shards"], ensure_ascii=False)
                )
        else:
            slam_return_code = run_logged(
                "slam",
                [
                    slam_python,
                    str(PROJECT_ROOT / "scripts" / "batch_infer.py"),
                    "--descriptor_manifest",
                    str(active_manifest_path),
                    "--run_dir",
                    str(run_dir),
                    "--stages",
                    "slam",
                    *common_batch_args,
                    *slam_args,
                ],
                raise_on_error=False,
            )
            active_manifest_path = handle_partial_infer_stage(
                stage_label="slam",
                completed_stage_name="slam",
                source_manifest=active_manifest_path,
                return_code=slam_return_code,
            )

    if "native_depth" in stages:
        ensure_manifest_exists("native_depth", active_manifest_path)
        common_infer_cfg = infer_cfg.get("common") or {}
        gpus = native_depth_cfg.get("gpus", common_infer_cfg.get("gpus", "0"))
        if isinstance(gpus, (list, tuple)):
            gpus = ",".join(str(item).strip() for item in gpus if str(item).strip())
        native_depth_args = tuple(
            cli_args_from_mapping(
                {
                    key: value
                    for key, value in native_depth_cfg.items()
                    if key not in {"enabled", "gpus"}
                },
                negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
            )
        )
        native_depth_cmd = [
            slam_python,
            str(PROJECT_ROOT / "scripts" / "run_hot3d_native_depth.py"),
            "--descriptor_manifest",
            str(active_manifest_path),
            "--run_dir",
            str(run_dir),
            "--gpus",
            str(gpus),
            *native_depth_args,
        ]
        if bool(common_infer_cfg.get("resume", False)):
            native_depth_cmd.append("--resume")
        native_depth_return_code = run_logged(
            "native_depth",
            native_depth_cmd,
            raise_on_error=False,
        )
        if native_depth_return_code != 0:
            active_manifest_path = handle_partial_external_stage(
                stage_label="native_depth",
                source_manifest=active_manifest_path,
                return_code=native_depth_return_code,
                output_exists=lambda seq_folder: (
                    get_stage_done_marker(seq_folder, "native_depth").exists()
                    and (seq_folder / "NATIVE_DEPTH" / "any4d_depth.npz").is_file()
                ),
            )

    if "infiller" in stages:
        ensure_manifest_exists("infiller", active_manifest_path)
        fpha_skeleton_cfg = (
            adapter_cfg.get("fpha_skeleton")
            if adapter_name == "fpha_tar"
            else None
        ) or {}
        use_fpha_skeleton_infiller = bool(fpha_skeleton_cfg.get("enabled"))
        if use_fpha_skeleton_infiller:
            if multihost_runner is not None:
                raise ValueError("FPHA skeleton infiller path does not support infer.multihost")
            common_infer_cfg = infer_cfg.get("common") or {}
            device = str(fpha_skeleton_cfg.get("device") or "cuda:0")
            raw_gpus = common_infer_cfg.get("gpus")
            if fpha_skeleton_cfg.get("device") is None and raw_gpus is not None:
                if isinstance(raw_gpus, list):
                    first_gpu = str(raw_gpus[0]).strip() if raw_gpus else ""
                else:
                    first_gpu = str(raw_gpus).split(",")[0].strip()
                if first_gpu:
                    device = first_gpu if first_gpu.startswith("cuda:") else f"cuda:{first_gpu}"
            fpha_cmd = [
                hawor_python,
                str(PROJECT_ROOT / "scripts" / "generate_fpha_world_res.py"),
                "--descriptor_manifest",
                str(active_manifest_path),
                "--device",
                device,
                "--num_iters",
                str(int(fpha_skeleton_cfg.get("num_iters", 180))),
                "--lr",
                str(float(fpha_skeleton_cfg.get("lr", 1e-2))),
                "--pose_reg",
                str(float(fpha_skeleton_cfg.get("pose_reg", 1e-4))),
                "--shape_reg",
                str(float(fpha_skeleton_cfg.get("shape_reg", 1e-3))),
                "--temporal_reg",
                str(float(fpha_skeleton_cfg.get("temporal_reg", 1e-3))),
            ]
            if fpha_skeleton_cfg.get("shape_iters") is not None:
                fpha_cmd.extend(["--shape_iters", str(int(fpha_skeleton_cfg["shape_iters"]))])
            if fpha_skeleton_cfg.get("shape_sample_size") is not None:
                fpha_cmd.extend(["--shape_sample_size", str(int(fpha_skeleton_cfg["shape_sample_size"]))])
            if fpha_skeleton_cfg.get("chunk_size") is not None:
                fpha_cmd.extend(["--chunk_size", str(int(fpha_skeleton_cfg["chunk_size"]))])
            if fpha_skeleton_cfg.get("skeleton_root"):
                fpha_cmd.extend(["--skeleton_root", str(fpha_skeleton_cfg["skeleton_root"])])
            if bool(common_infer_cfg.get("resume", False)):
                fpha_cmd.append("--resume")
            if not bool(fpha_skeleton_cfg.get("preserve_existing_left", True)):
                fpha_cmd.append("--no-preserve_existing_left")

            infiller_return_code = run_logged(
                "infiller",
                fpha_cmd,
                raise_on_error=False,
            )
            if infiller_return_code != 0:
                active_manifest_path = handle_partial_external_stage(
                    stage_label="infiller",
                    source_manifest=active_manifest_path,
                    return_code=infiller_return_code,
                    output_exists=lambda seq_folder: (
                        (seq_folder / "world_space_res.pth").is_file()
                        and get_stage_done_marker(seq_folder, "infiller").exists()
                    ),
                )
        else:
            infiller_args = tuple(
                cli_args_from_mapping(
                    infer_cfg.get("infiller"),
                    negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
                )
            )
            if multihost_runner is not None:
                result = multihost_runner.run_stage(
                    MultihostStageSpec(
                        pipeline_stage="infiller",
                        batch_stages="infiller",
                        runtime_key="hawor",
                        extra_args=multihost_common_batch_args
                        + tuple(
                            cli_args_from_mapping(
                                sanitize_infer_args_for_multihost(
                                    infer_cfg.get("infiller"),
                                    reserved_keys=MULTIHOST_DISALLOWED_INFER_KEYS,
                                ),
                                negative_bool_flags=BATCH_INFER_NEGATIVE_BOOL_FLAGS,
                            )
                        )
                        ,
                        worker_count_per_gpu=infer_stage_worker_count_per_gpu(
                            pipeline_stage="infiller",
                            infer_cfg=infer_cfg,
                        ),
                    )
                )
                run_summary.setdefault("multihost_dispatch", {})["infiller"] = result["dispatch_path"]
                summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
                if not result["success"]:
                    raise RuntimeError(
                        "infiller multihost stage failed: "
                        + json.dumps(result["failed_shards"], ensure_ascii=False)
                    )
            else:
                infiller_return_code = run_logged(
                    "infiller",
                    [
                        hawor_python,
                        str(PROJECT_ROOT / "scripts" / "batch_infer.py"),
                        "--descriptor_manifest",
                        str(active_manifest_path),
                        "--run_dir",
                        str(run_dir),
                        "--stages",
                        "infiller",
                        *common_batch_args,
                        *infiller_args,
                    ],
                    raise_on_error=False,
                )
                active_manifest_path = handle_partial_infer_stage(
                    stage_label="infiller",
                    completed_stage_name="infiller",
                    source_manifest=active_manifest_path,
                    return_code=infiller_return_code,
                )

    if "filter" in stages:
        ensure_manifest_exists("filter", active_manifest_path)
        filter_runtime_cfg = dict(filter_cfg)
        filter_runtime_cfg.setdefault("annotation_root", annotation_root)
        filter_runtime_cfg.setdefault("annotation_suffix", build_cfg.get("annotation_suffix"))
        filter_runtime_cfg.setdefault("require_annotation", build_cfg.get("require_annotation"))
        filter_runtime_cfg.setdefault("source_fps", build_cfg.get("source_fps"))
        filter_runtime_cfg.setdefault("target_fps", build_cfg.get("target_fps"))
        filter_runtime_cfg.setdefault("interpolate_labels", build_cfg.get("interpolate_labels"))
        filter_runtime_cfg.setdefault("mano_device", build_cfg.get("mano_device"))
        if build_cfg.get("mano_gpus") is not None:
            filter_runtime_cfg.setdefault("mano_gpus", build_cfg.get("mano_gpus"))
        if build_cfg.get("mano_dir") is not None:
            filter_runtime_cfg.setdefault("mano_dir", build_cfg.get("mano_dir"))
        filter_runtime_cfg.setdefault("feature_cache_dir", str(shared_feature_cache_dir))
        run_logged(
            "filter",
            [
                hawor_python,
                str(PROJECT_ROOT / "scripts" / "filter_manifest_by_quality.py"),
                "--input_manifest",
                str(active_manifest_path),
                "--output_manifest",
                str(filtered_manifest_path),
                "--report_out",
                str(filter_report_path),
                *cli_args_from_mapping(filter_runtime_cfg),
            ],
        )
        active_manifest_path = filtered_manifest_path
        run_summary["active_manifest_path"] = str(active_manifest_path.resolve())
        run_summary["filter_report_path"] = str(filter_report_path.resolve())
        summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if "build" in stages:
        ensure_manifest_exists("build", active_manifest_path)
        build_runtime_cfg = dict(build_cfg)
        build_runtime_cfg.setdefault("feature_cache_dir", str(shared_feature_cache_dir))
        build_cmd = [
            hawor_python,
            str(PROJECT_ROOT / "scripts" / "build_vla_from_manifest.py"),
            "--descriptor_manifest",
            str(active_manifest_path),
            "--output_dir",
            str(final_dataset_root),
            *cli_args_from_mapping(build_runtime_cfg),
        ]
        if args.resume:
            build_cmd.append("--resume")
        if annotation_root:
            build_cmd.extend(["--annotation_root", str(annotation_root)])
        run_logged("build", build_cmd)

    if "validate" in stages:
        ensure_manifest_exists("validate", active_manifest_path)
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


def main(argv: list[str] | None = None) -> None:
    args = get_parser().parse_args(argv)
    run_pipeline(args)
