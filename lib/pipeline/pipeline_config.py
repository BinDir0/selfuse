"""Helpers for the official nested dataset-pipeline configuration format."""

from __future__ import annotations

from copy import deepcopy


_OFFICIAL_TOP_LEVEL_KEYS = {
    "run_tag",
    "dataset",
    "paths",
    "runtimes",
    "adapter_config",
    "infer",
    "annotation",
    "build",
    "filter",
    "validation",
}

_LEGACY_TOP_LEVEL_KEYS = {
    "adapter",
    "source_id",
    "split",
    "factory_range",
    "start_factory_id",
    "end_factory_id",
    "buildai_repo_root",
    "buildai_config",
    "shard_root",
    "processed_root",
    "seq_folder_root",
    "annotation_root",
    "final_dataset_root",
    "log_root",
    "buildai_shell",
    "hawor_python",
    "slam_python",
    "batch_infer",
    "annotation_command",
    "require_annotation",
    "gpus",
    "workers_per_gpu",
    "resume",
    "checkpoint",
    "infiller_weight",
    "img_focal",
    "detect_motion",
    "slam",
    "native_depth",
    "infiller",
    "buildai",
}


def _as_dict(value):
    return dict(value) if isinstance(value, dict) else {}


def _ensure_mapping(raw: dict, key: str) -> dict:
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Pipeline config `{key}` must be a mapping.")
    return dict(value)


def _reject_legacy_layout(raw: dict) -> None:
    invalid = sorted(key for key in raw if key not in _OFFICIAL_TOP_LEVEL_KEYS)
    if not invalid:
        return
    legacy_hits = [key for key in invalid if key in _LEGACY_TOP_LEVEL_KEYS]
    if legacy_hits:
        raise ValueError(
            "Compact/legacy dataset-pipeline configs are no longer supported. "
            "Move top-level dataset/path/runtime/infer keys into the official nested sections. "
            f"Legacy keys found: {legacy_hits}"
        )
    raise ValueError(f"Unsupported top-level pipeline config keys: {invalid}")


def normalize_pipeline_config(raw_config: dict | None) -> dict:
    raw = deepcopy(raw_config or {})
    if not isinstance(raw, dict):
        raise ValueError("Pipeline config must be a mapping.")

    _reject_legacy_layout(raw)

    dataset_cfg = _ensure_mapping(raw, "dataset")
    paths_cfg = _ensure_mapping(raw, "paths")
    runtimes_cfg = _ensure_mapping(raw, "runtimes")
    build_cfg = _ensure_mapping(raw, "build")
    filter_cfg = _ensure_mapping(raw, "filter")
    validation_cfg = _ensure_mapping(raw, "validation")
    annotation_cfg = _ensure_mapping(raw, "annotation")
    adapter_cfg = _ensure_mapping(raw, "adapter_config")
    infer_cfg = _ensure_mapping(raw, "infer")

    adapter_name = dataset_cfg.get("adapter") or "buildai"
    dataset_cfg.setdefault("adapter", adapter_name)
    dataset_cfg.setdefault("source_id", adapter_name)
    dataset_cfg.setdefault("split", "train")

    if adapter_name == "buildai":
        adapter_cfg.setdefault("stages", "1,2,3")
        adapter_cfg.setdefault("setup_decord", False)
        adapter_cfg.setdefault("clean_stage3_output", False)

    for key, default in (
        ("require_annotation", False),
        ("preprocess_workers", 8),
        ("writer_workers", 4),
        ("frames_per_shard", 10000),
        ("repeat_episodes", 1),
        ("mano_device", "cuda:0"),
        ("annotation_suffix", ".annotation.json"),
        ("source_fps", 5.0),
        ("target_fps", 30.0),
        ("interpolate_labels", True),
    ):
        build_cfg.setdefault(key, default)

    for key, default in (
        ("stages", "detect_track,motion,slam,infiller"),
        ("workers", 8),
        ("chunksize", 16),
        ("outlier_checks", True),
        ("camera_space_auto_method", "iqr_bounds"),
        ("camera_space_iqr_multiplier", 2.5),
        ("camera_space_axis_abs_cap", 1.5),
        ("camera_space_abs_percentile", 99.0),
        ("camera_space_abs_scale", 2.5),
    ):
        filter_cfg.setdefault(key, default)

    for key, default in (("max_clips", 200), ("dataset_sample_checks", 20)):
        validation_cfg.setdefault(key, default)

    normalized_infer_cfg = {
        "common": _as_dict(infer_cfg.get("common")),
        "detect_motion": _as_dict(infer_cfg.get("detect_motion")),
        "slam": _as_dict(infer_cfg.get("slam")),
        "native_depth": _as_dict(infer_cfg.get("native_depth")),
        "infiller": _as_dict(infer_cfg.get("infiller")),
        "multihost": _as_dict(infer_cfg.get("multihost")),
    }

    return {
        "run_tag": raw.get("run_tag"),
        "dataset": dataset_cfg,
        "paths": paths_cfg,
        "runtimes": runtimes_cfg,
        "adapter_config": adapter_cfg,
        "infer": normalized_infer_cfg,
        "annotation": annotation_cfg,
        "build": build_cfg,
        "filter": filter_cfg,
        "validation": validation_cfg,
    }
