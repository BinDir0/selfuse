"""Helpers for compact dataset-pipeline configuration files."""

from __future__ import annotations

from copy import deepcopy


def _as_dict(value):
    return dict(value) if isinstance(value, dict) else {}


def _maybe_set(target: dict, key: str, value):
    if value is not None and key not in target:
        target[key] = value


def _parse_factory_range(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raw = str(value).strip()
    if not raw:
        return None
    if "-" in raw:
        start, end = raw.split("-", 1)
        return int(start), int(end)
    raise ValueError(f"Invalid factory_range: {value!r}. Expected [start, end] or 'start-end'.")


def normalize_pipeline_config(raw_config: dict | None) -> dict:
    """Accept both the original nested config and a compact top-level shorthand."""
    raw = deepcopy(raw_config or {})

    dataset_cfg = _as_dict(raw.get("dataset"))
    if isinstance(raw.get("dataset"), str):
        dataset_cfg["adapter"] = raw["dataset"]
    adapter_name = raw.get("adapter") or dataset_cfg.get("adapter") or dataset_cfg.get("source_type") or "buildai"
    dataset_cfg.setdefault("adapter", adapter_name)
    dataset_cfg.setdefault("source_id", raw.get("source_id") or dataset_cfg.get("source_id") or adapter_name)
    dataset_cfg.setdefault("split", raw.get("split") or dataset_cfg.get("split") or "train")

    parsed_factory_range = _parse_factory_range(raw.get("factory_range"))
    if parsed_factory_range is not None:
        dataset_cfg.setdefault("start_factory_id", parsed_factory_range[0])
        dataset_cfg.setdefault("end_factory_id", parsed_factory_range[1])
    _maybe_set(dataset_cfg, "start_factory_id", raw.get("start_factory_id"))
    _maybe_set(dataset_cfg, "end_factory_id", raw.get("end_factory_id"))

    paths_cfg = _as_dict(raw.get("paths"))
    for key in ("buildai_repo_root", "buildai_config", "shard_root", "annotation_root", "final_dataset_root", "log_root"):
        _maybe_set(paths_cfg, key, raw.get(key))

    runtimes_cfg = _as_dict(raw.get("runtimes"))
    for key in ("buildai_shell", "hawor_python", "slam_python"):
        _maybe_set(runtimes_cfg, key, raw.get(key))

    batch_cfg = _as_dict(raw.get("batch_infer"))
    common_cfg = _as_dict(batch_cfg.get("common"))
    for key in (
        "gpus",
        "workers_per_gpu",
        "resume",
        "checkpoint",
        "infiller_weight",
        "img_focal",
        "detect_half_precision",
        "detect_device",
        "enable_profiler",
    ):
        _maybe_set(common_cfg, key, raw.get(key))

    detect_motion_cfg = _as_dict(batch_cfg.get("detect_motion"))
    detect_motion_cfg.update(_as_dict(raw.get("detect_motion")))

    slam_cfg = _as_dict(batch_cfg.get("slam"))
    slam_cfg.update(_as_dict(raw.get("slam")))

    infiller_cfg = _as_dict(batch_cfg.get("infiller"))
    infiller_cfg.update(_as_dict(raw.get("infiller")))

    build_cfg = _as_dict(raw.get("build"))
    for key, default in (
        ("require_annotation", False),
        ("preprocess_workers", 8),
        ("writer_workers", 4),
        ("frames_per_shard", 10000),
        ("repeat_episodes", 1),
        ("mano_device", "cuda:0"),
    ):
        _maybe_set(build_cfg, key, raw.get(key))
        build_cfg.setdefault(key, default)

    validation_cfg = _as_dict(raw.get("validation"))
    for key, default in (("max_clips", 200), ("dataset_sample_checks", 20)):
        _maybe_set(validation_cfg, key, raw.get(key))
        validation_cfg.setdefault(key, default)

    annotation_cfg = _as_dict(raw.get("annotation"))
    _maybe_set(annotation_cfg, "command", raw.get("annotation_command"))

    adapter_cfg = _as_dict(raw.get("adapter_config"))
    adapter_cfg.update(_as_dict(raw.get(adapter_name)))
    if adapter_name == "buildai":
        adapter_cfg.setdefault("stages", "1,2,3")
        adapter_cfg.setdefault("setup_decord", False)
        adapter_cfg.setdefault("clean_stage3_output", False)

    return {
        "dataset": dataset_cfg,
        "paths": paths_cfg,
        "runtimes": runtimes_cfg,
        "adapter_config": adapter_cfg,
        "batch_infer": {
            "common": common_cfg,
            "detect_motion": detect_motion_cfg,
            "slam": slam_cfg,
            "infiller": infiller_cfg,
        },
        "annotation": annotation_cfg,
        "build": build_cfg,
        "validation": validation_cfg,
    }
