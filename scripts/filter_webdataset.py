#!/usr/bin/env python3
"""Analyze and optionally filter an existing WebDataset shard directory."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.webdataset_rewriter import (  # noqa: E402
    iter_shard_paths,
    iter_shard_samples,
    split_sample_member_name,
    validate_sample_record,
    write_sample_to_tar,
)
from lib.pipeline.quality_metrics import (  # noqa: E402
    decide_clip_quality,
    decode_lowdim,
    finalize_clip_quality_metrics,
    new_clip_quality_stats,
    parse_instruction_metadata,
    parse_frame_index,
    resolve_auto_quality_thresholds,
    update_clip_quality_stats,
)


DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))
_WORKER_ARGS = None
_WORKER_OUTPUT_DIR = None
_WORKER_KEEP_BY_CLIP = None
_SHARD_DATA_EXCEPTIONS = (OSError, tarfile.TarError, ValueError)
TAR_BLOCK_SIZE = 512
REGULAR_TAR_TYPES = {b"", b"0", b"\0", b"7"}


def _auto_chunksize(total_items: int, workers: int) -> int:
    if total_items <= 0:
        return 1
    # Shard analysis/rewrite is heavy enough that large pool chunks make tqdm
    # look stuck for minutes before the first worker returns a batch.
    return max(1, min(4, total_items // max(1, workers * 16) or 1))


def _parse_tar_int(field: bytes) -> int:
    raw = field.rstrip(b"\0 ").strip()
    if not raw:
        return 0
    return int(raw, 8)


def _round_tar_size(size: int) -> int:
    return ((int(size) + TAR_BLOCK_SIZE - 1) // TAR_BLOCK_SIZE) * TAR_BLOCK_SIZE


def build_parser():
    parser = argparse.ArgumentParser(description="Analyze and filter existing WebDataset shards")
    parser.add_argument("--source_shard_dir", required=True, help="Source directory containing shard tar files")
    parser.add_argument("--output_dir", default=None, help="Optional output directory for filtered shards")
    parser.add_argument("--report_out", default=None, help="Optional JSON report path")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel shard workers")
    parser.add_argument(
        "--chunksize",
        type=int,
        default=0,
        help="Pool chunksize; 0 selects an automatic chunksize",
    )
    parser.add_argument(
        "--start_shard",
        type=int,
        default=0,
        help="Start shard index in sorted shard order (inclusive)",
    )
    parser.add_argument(
        "--end_shard",
        type=int,
        default=None,
        help="End shard index in sorted shard order (exclusive)",
    )
    parser.add_argument(
        "--min_instruction_num",
        type=int,
        default=None,
        help="Optional minimum instruction_num required to keep a clip",
    )
    parser.add_argument(
        "--min_presence_ratio",
        type=float,
        default=None,
        help="Optional minimum fraction of frames with presence > 0",
    )
    parser.add_argument(
        "--max_hand_translation_step",
        type=float,
        default=None,
        help="Optional max allowed per-frame wrist translation step in meters",
    )
    parser.add_argument(
        "--max_camera_translation_step",
        type=float,
        default=None,
        help="Optional max allowed per-frame camera translation step in meters",
    )
    parser.add_argument(
        "--max_camera_rotation_step",
        type=float,
        default=None,
        help="Optional max allowed per-frame camera rotation delta (Frobenius norm)",
    )
    parser.add_argument(
        "--max_camera_space_wrist_abs",
        type=float,
        default=None,
        help="Optional max absolute camera-space coordinate allowed for wrist positions in meters",
    )
    parser.add_argument(
        "--max_camera_space_hand_abs",
        type=float,
        default=None,
        help="Optional max absolute camera-space coordinate allowed for stored hand keypoints in meters",
    )
    parser.add_argument(
        "--camera_space_auto_method",
        type=str,
        default="iqr_bounds",
        choices=("iqr_bounds", "percentile_abs"),
        help="Automatic camera-space filter mode when manual abs thresholds are not provided",
    )
    parser.add_argument(
        "--camera_space_iqr_multiplier",
        type=float,
        default=2.5,
        help="IQR multiplier used for automatic camera-space lower/upper bounds",
    )
    parser.add_argument(
        "--camera_space_axis_abs_cap",
        type=float,
        default=1.5,
        help="Hard absolute cap applied to camera-space x/y/z coordinates for wrist and hand points",
    )
    parser.add_argument(
        "--camera_space_abs_percentile",
        type=float,
        default=99.0,
        help="Percentile used for automatic camera-space absolute-value thresholds",
    )
    parser.add_argument(
        "--camera_space_abs_scale",
        type=float,
        default=2.5,
        help="Scale multiplier applied to the chosen percentile for automatic camera-space thresholds",
    )
    parser.add_argument(
        "--outlier_checks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable optional outlier checks; NaN/Inf and missing-language hard filters always stay enabled",
    )
    return parser


def _new_clip_stats(clip_id: str) -> dict:
    return new_clip_quality_stats(clip_id)


def _parse_instruction_flags(meta: dict | None) -> tuple[int, bool, bool, bool]:
    parsed = parse_instruction_metadata(meta)
    return (
        int(parsed["instruction_num"]),
        bool(parsed["missing_instruction"]),
        bool(parsed["empty_instruction"]),
        bool(parsed["instruction_num_mismatch"]),
    )


def _update_clip_stats(
    stats: dict,
    sample_key: str,
    meta: dict,
    lowdim,
    *,
    count_invalid_lowdim: bool = True,
    compute_motion_metrics: bool = True,
    compute_camera_space_metrics: bool = True,
) -> None:
    frame_idx = parse_frame_index(sample_key)
    instruction_num, missing_instruction, empty_instruction, instruction_num_mismatch = _parse_instruction_flags(meta)
    update_clip_quality_stats(
        stats,
        frame_idx,
        instruction_num,
        int(meta.get("presence", 0) or 0),
        lowdim,
        missing_instruction=missing_instruction,
        empty_instruction=empty_instruction,
        instruction_num_mismatch=instruction_num_mismatch,
        count_invalid_lowdim=count_invalid_lowdim,
        compute_motion_metrics=compute_motion_metrics,
        compute_camera_space_metrics=compute_camera_space_metrics,
    )


def _finalize_clip_metrics(stats: dict) -> dict:
    return finalize_clip_quality_metrics(stats)


def _sample_clip_id(sample: dict, meta: dict | None) -> str:
    if meta is not None:
        clip_id = meta.get("clip_id")
        if clip_id:
            return str(clip_id)
    return sample["key"].rsplit("_f", 1)[0]


def _sample_missing_fields(sample: dict) -> list[str]:
    missing = []
    for field_name in ("image_bytes", "lowdim_bytes", "meta_bytes"):
        if sample.get(field_name) is None:
            missing.append(field_name)
    return missing


def _new_analysis_sample(sample_key: str) -> dict:
    return {
        "key": sample_key,
        "image_present": False,
        "lowdim_bytes": None,
        "meta_bytes": None,
    }


def _iter_analysis_samples_fast(shard_path: str):
    current_sample = None

    with open(shard_path, "rb", buffering=1024 * 1024) as handle:
        while True:
            header = handle.read(TAR_BLOCK_SIZE)
            if not header:
                break
            if len(header) != TAR_BLOCK_SIZE:
                raise ValueError(f"Short tar header in {shard_path}")
            if header == b"\0" * TAR_BLOCK_SIZE:
                break

            typeflag = header[156:157]
            try:
                size = _parse_tar_int(header[124:136])
            except Exception as error:
                raise ValueError(f"Invalid tar size field in {shard_path}: {error}") from error

            name = header[0:100].rstrip(b"\0")
            prefix = header[345:500].rstrip(b"\0")
            member_name = (prefix + b"/" + name).decode("utf-8") if prefix else name.decode("utf-8")
            rounded_size = _round_tar_size(size)

            if typeflag not in REGULAR_TAR_TYPES:
                handle.seek(rounded_size, os.SEEK_CUR)
                continue

            sample_key, _, field_name = split_sample_member_name(member_name)
            if sample_key is None:
                raise ValueError(f"Unsupported shard member: {member_name}")

            if current_sample is None:
                current_sample = _new_analysis_sample(sample_key)
            elif current_sample["key"] != sample_key:
                yield current_sample
                current_sample = _new_analysis_sample(sample_key)

            if field_name == "image_bytes":
                current_sample["image_present"] = True
                handle.seek(rounded_size, os.SEEK_CUR)
                continue

            if field_name == "mano_bytes":
                handle.seek(rounded_size, os.SEEK_CUR)
                continue

            payload = handle.read(size)
            if len(payload) != size:
                raise ValueError(f"Short tar payload for {member_name} in {shard_path}")
            padding = rounded_size - size
            if padding > 0:
                handle.seek(padding, os.SEEK_CUR)

            current_sample[field_name] = payload

    if current_sample is not None:
        yield current_sample


def _flush_analyze_clip_block(
    *,
    clip_id: str | None,
    clip_stats: dict | None,
    shard_result: dict,
) -> dict:
    if clip_id is None or clip_stats is None:
        return shard_result

    shard_result["clip_metrics"].append(
        {
            "clip_id": clip_id,
            "metrics": _finalize_clip_metrics(clip_stats),
        }
    )
    shard_result["clips_total"] += 1
    return shard_result


def _serialize_shard_error(exc: Exception) -> dict:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }


def analyze_shard(
    shard_path: str,
    *,
    compute_motion_metrics: bool = True,
    compute_camera_space_metrics: bool = True,
) -> dict:
    if not compute_motion_metrics and not compute_camera_space_metrics:
        return analyze_shard_fast_hard_rules(shard_path)

    shard_name = os.path.basename(shard_path)
    shard_result = {
        "shard_name": shard_name,
        "samples_total": 0,
        "clips_total": 0,
        "incomplete_samples": 0,
        "clip_metrics": [],
        "shard_error": None,
    }

    current_clip_id = None
    current_clip_stats = None

    try:
        for sample in iter_shard_samples(shard_path):
            shard_result["samples_total"] += 1
            missing_fields = _sample_missing_fields(sample)

            meta = None
            if "meta_bytes" not in missing_fields:
                try:
                    meta = json.loads(sample["meta_bytes"].decode("utf-8"))
                except Exception:
                    meta = None
            clip_id = _sample_clip_id(sample, meta)

            if current_clip_id is None:
                current_clip_id = clip_id
                current_clip_stats = _new_clip_stats(clip_id)
            elif clip_id != current_clip_id:
                shard_result = _flush_analyze_clip_block(
                    clip_id=current_clip_id,
                    clip_stats=current_clip_stats,
                    shard_result=shard_result,
                )
                current_clip_id = clip_id
                current_clip_stats = _new_clip_stats(clip_id)

            if missing_fields:
                shard_result["incomplete_samples"] += 1
                current_clip_stats["incomplete_sample_frames"] += 1
                if "meta_bytes" in missing_fields:
                    current_clip_stats["invalid_meta_frames"] += 1
                if "image_bytes" in missing_fields or "lowdim_bytes" in missing_fields:
                    current_clip_stats["invalid_lowdim_frames"] += 1
                _update_clip_stats(
                    current_clip_stats,
                    sample["key"],
                    meta if meta is not None else {},
                    None,
                    count_invalid_lowdim=False,
                    compute_motion_metrics=compute_motion_metrics,
                    compute_camera_space_metrics=compute_camera_space_metrics,
                )
                continue

            validate_sample_record(sample)

            if meta is None:
                current_clip_stats["invalid_meta_frames"] += 1
                _update_clip_stats(
                    current_clip_stats,
                    sample["key"],
                    {},
                    None,
                    count_invalid_lowdim=False,
                    compute_motion_metrics=compute_motion_metrics,
                    compute_camera_space_metrics=compute_camera_space_metrics,
                )
            else:
                try:
                    lowdim = decode_lowdim(sample["lowdim_bytes"])
                except Exception:
                    lowdim = None
                _update_clip_stats(
                    current_clip_stats,
                    sample["key"],
                    meta,
                    lowdim,
                    compute_motion_metrics=compute_motion_metrics,
                    compute_camera_space_metrics=compute_camera_space_metrics,
                )
    except _SHARD_DATA_EXCEPTIONS as exc:
        shard_result["clips_total"] = 0
        shard_result["clip_metrics"] = []
        shard_result["shard_error"] = _serialize_shard_error(exc)
        return shard_result

    shard_result = _flush_analyze_clip_block(
        clip_id=current_clip_id,
        clip_stats=current_clip_stats,
        shard_result=shard_result,
    )

    return shard_result


def analyze_shard_fast_hard_rules(shard_path: str) -> dict:
    shard_name = os.path.basename(shard_path)
    shard_result = {
        "shard_name": shard_name,
        "samples_total": 0,
        "clips_total": 0,
        "incomplete_samples": 0,
        "clip_metrics": [],
        "shard_error": None,
    }

    current_clip_id = None
    current_clip_stats = None

    try:
        for sample in _iter_analysis_samples_fast(shard_path):
            shard_result["samples_total"] += 1
            missing_fields = []
            if not sample["image_present"]:
                missing_fields.append("image_bytes")
            if sample["lowdim_bytes"] is None:
                missing_fields.append("lowdim_bytes")
            if sample["meta_bytes"] is None:
                missing_fields.append("meta_bytes")

            meta = None
            if sample["meta_bytes"] is not None:
                try:
                    meta = json.loads(sample["meta_bytes"].decode("utf-8"))
                except Exception:
                    meta = None
            clip_id = _sample_clip_id(sample, meta)

            if current_clip_id is None:
                current_clip_id = clip_id
                current_clip_stats = _new_clip_stats(clip_id)
            elif clip_id != current_clip_id:
                shard_result = _flush_analyze_clip_block(
                    clip_id=current_clip_id,
                    clip_stats=current_clip_stats,
                    shard_result=shard_result,
                )
                current_clip_id = clip_id
                current_clip_stats = _new_clip_stats(clip_id)

            if missing_fields:
                shard_result["incomplete_samples"] += 1
                current_clip_stats["incomplete_sample_frames"] += 1
                if "meta_bytes" in missing_fields:
                    current_clip_stats["invalid_meta_frames"] += 1
                if "image_bytes" in missing_fields or "lowdim_bytes" in missing_fields:
                    current_clip_stats["invalid_lowdim_frames"] += 1
                _update_clip_stats(
                    current_clip_stats,
                    sample["key"],
                    meta if meta is not None else {},
                    None,
                    count_invalid_lowdim=False,
                    compute_motion_metrics=False,
                    compute_camera_space_metrics=False,
                )
                continue

            if meta is None:
                current_clip_stats["invalid_meta_frames"] += 1
                _update_clip_stats(
                    current_clip_stats,
                    sample["key"],
                    {},
                    None,
                    count_invalid_lowdim=False,
                    compute_motion_metrics=False,
                    compute_camera_space_metrics=False,
                )
                continue

            try:
                lowdim = decode_lowdim(sample["lowdim_bytes"])
            except Exception:
                lowdim = None
            _update_clip_stats(
                current_clip_stats,
                sample["key"],
                meta,
                lowdim,
                compute_motion_metrics=False,
                compute_camera_space_metrics=False,
            )
    except _SHARD_DATA_EXCEPTIONS as exc:
        shard_result["clips_total"] = 0
        shard_result["clip_metrics"] = []
        shard_result["shard_error"] = _serialize_shard_error(exc)
        return shard_result

    shard_result = _flush_analyze_clip_block(
        clip_id=current_clip_id,
        clip_stats=current_clip_stats,
        shard_result=shard_result,
    )
    return shard_result


def rewrite_shard(shard_path: str, output_dir: str, keep_by_clip: dict[str, bool]) -> dict:
    shard_name = os.path.basename(shard_path)
    output_path = os.path.join(output_dir, shard_name)
    tmp_path = f"{output_path}.tmp"
    result = {
        "shard_name": shard_name,
        "output_path": output_path,
        "samples_total": 0,
        "incomplete_samples": 0,
        "frames_written": 0,
        "clips_written": 0,
        "frames_dropped": 0,
        "clips_dropped": 0,
        "shard_written": 0,
        "shard_error": None,
    }

    tar_writer = None
    current_clip_id = None
    current_keep = False
    clip_wrote_frames = False
    current_clip_frames = 0
    try:
        for sample in iter_shard_samples(shard_path):
            result["samples_total"] += 1
            missing_fields = _sample_missing_fields(sample)
            if missing_fields:
                result["incomplete_samples"] += 1
                continue

            validate_sample_record(sample)

            meta = None
            try:
                meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            except Exception:
                meta = None
            clip_id = _sample_clip_id(sample, meta)

            if clip_id != current_clip_id:
                if current_clip_id is not None and current_keep and clip_wrote_frames:
                    result["clips_written"] += 1
                elif current_clip_id is not None and not current_keep and current_clip_frames > 0:
                    result["clips_dropped"] += 1
                    result["frames_dropped"] += current_clip_frames
                current_clip_id = clip_id
                current_keep = bool(keep_by_clip.get(clip_id, False))
                clip_wrote_frames = False
                current_clip_frames = 0

            current_clip_frames += 1
            if not current_keep:
                continue

            if tar_writer is None:
                os.makedirs(output_dir, exist_ok=True)
                tar_writer = tarfile.open(tmp_path, "w")

            write_sample_to_tar(
                tar_writer,
                sample["key"],
                sample["image_bytes"],
                sample["lowdim_bytes"],
                sample["meta_bytes"],
                mano_bytes=sample.get("mano_bytes"),
                depth_bytes=sample.get("depth_bytes"),
            )
            result["frames_written"] += 1
            clip_wrote_frames = True

        if current_clip_id is not None and current_keep and clip_wrote_frames:
            result["clips_written"] += 1
        elif current_clip_id is not None and not current_keep and current_clip_frames > 0:
            result["clips_dropped"] += 1
            result["frames_dropped"] += current_clip_frames
    except _SHARD_DATA_EXCEPTIONS as exc:
        if tar_writer is not None:
            tar_writer.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        result["frames_written"] = 0
        result["clips_written"] = 0
        result["frames_dropped"] = 0
        result["clips_dropped"] = 0
        result["shard_written"] = 0
        result["shard_error"] = _serialize_shard_error(exc)
        return result

    if tar_writer is not None:
        tar_writer.close()
    if result["frames_written"] == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    else:
        os.replace(tmp_path, output_path)
        result["shard_written"] = 1
    return result


def _worker_init(args_dict: dict, output_dir: str | None, keep_by_clip: dict[str, bool] | None = None):
    global _WORKER_ARGS, _WORKER_OUTPUT_DIR, _WORKER_KEEP_BY_CLIP
    _WORKER_ARGS = dict(args_dict)
    _WORKER_OUTPUT_DIR = output_dir
    _WORKER_ARGS["output_dir"] = output_dir
    _WORKER_KEEP_BY_CLIP = keep_by_clip or {}


def _worker_analyze_shard(shard_path: str) -> dict:
    return analyze_shard(
        shard_path,
        compute_motion_metrics=bool(_WORKER_ARGS.get("compute_motion_metrics", True)),
        compute_camera_space_metrics=bool(_WORKER_ARGS.get("compute_camera_space_metrics", True)),
    )


def _worker_rewrite_shard(shard_path: str) -> dict:
    return rewrite_shard(shard_path, _WORKER_OUTPUT_DIR, _WORKER_KEEP_BY_CLIP)


def build_report(
    source_shard_dir: Path,
    output_dir: Path | None,
    analysis_results: list[dict],
    clip_decisions: list[dict],
    args_dict: dict,
    threshold_info: dict,
    rewrite_results: list[dict] | None = None,
) -> dict:
    reason_counts = Counter()
    kept_clips = 0
    dropped = []
    total_clips = 0
    total_samples = 0
    shards = {}
    rewrite = {
        "shards_written": 0,
        "frames_written": 0,
        "clips_written": 0,
        "frames_dropped": 0,
        "clips_dropped": 0,
        "incomplete_samples": 0,
        "errored_shards": 0,
        "errored_shard_details": [],
    }
    total_incomplete_samples = 0
    analysis_errored_shards = []

    decision_by_clip = {item["clip_id"]: item for item in clip_decisions}

    for result in analysis_results:
        total_samples += result["samples_total"]
        total_clips += result["clips_total"]
        total_incomplete_samples += int(result.get("incomplete_samples", 0))
        shard_kept = 0
        shard_dropped = 0
        for item in result["clip_metrics"]:
            decision = decision_by_clip[item["clip_id"]]
            if decision["keep"]:
                shard_kept += 1
            else:
                shard_dropped += 1
        shard_summary = {
            "samples_total": result["samples_total"],
            "clips_total": result["clips_total"],
            "incomplete_samples": int(result.get("incomplete_samples", 0)),
            "kept_clips": shard_kept,
            "dropped_clips": shard_dropped,
        }
        if result.get("shard_error") is not None:
            shard_summary["analysis_error"] = result["shard_error"]
            analysis_errored_shards.append(
                {
                    "shard_name": result["shard_name"],
                    **result["shard_error"],
                }
            )
        shards[result["shard_name"]] = shard_summary

    for item in clip_decisions:
        if item["keep"]:
            kept_clips += 1
            continue
        reason_counts.update(item["reasons"])
        dropped.append(
            {
                "clip_id": item["clip_id"],
                "shard_name": item["shard_name"],
                "reasons": item["reasons"],
                "metrics": item["metrics"],
            }
        )

    if rewrite_results is not None:
        rewrite_by_name = {item["shard_name"]: item for item in rewrite_results}
        for shard_name, shard_summary in shards.items():
            rewrite_item = rewrite_by_name.get(shard_name)
            if rewrite_item is None:
                shard_summary["frames_written"] = 0
                continue
            shard_summary["frames_written"] = rewrite_item["frames_written"]
            rewrite["shards_written"] += rewrite_item["shard_written"]
            rewrite["frames_written"] += rewrite_item["frames_written"]
            rewrite["clips_written"] += rewrite_item["clips_written"]
            rewrite["frames_dropped"] += int(rewrite_item.get("frames_dropped", 0))
            rewrite["clips_dropped"] += int(rewrite_item.get("clips_dropped", 0))
            rewrite["incomplete_samples"] += int(rewrite_item.get("incomplete_samples", 0))
            if rewrite_item.get("shard_error") is not None:
                shard_summary["rewrite_error"] = rewrite_item["shard_error"]
                rewrite["errored_shards"] += 1
                rewrite["errored_shard_details"].append(
                    {
                        "shard_name": shard_name,
                        **rewrite_item["shard_error"],
                    }
                )

    report = {
        "source_shard_dir": str(source_shard_dir.resolve()),
        "output_dir": str(output_dir.resolve()) if output_dir else None,
        "shard_selection": {
            "start_shard": int(args_dict["start_shard"]),
            "end_shard": int(args_dict["end_shard"]),
            "selected_shards": len(analysis_results),
            "total_shards_available": int(args_dict["total_shards_available"]),
        },
        "mode": (
            "hard_rules_only"
            if not args_dict["outlier_checks"]
            else (
                f"two_pass_auto_{threshold_info['auto_rule']['method']}"
                if args_dict["use_auto_camera_space_thresholds"]
                else "two_pass_manual_threshold"
            )
        ),
        "criteria": {
            "hard_rules": {
                "drop_nonfinite_lowdim": True,
                "require_instruction_every_frame": True,
            },
            "min_instruction_num": args_dict["min_instruction_num"],
            "outlier_checks": bool(args_dict["outlier_checks"]),
            "min_presence_ratio": args_dict["min_presence_ratio"],
            "max_hand_translation_step": args_dict["max_hand_translation_step"],
            "max_camera_translation_step": args_dict["max_camera_translation_step"],
            "max_camera_rotation_step": args_dict["max_camera_rotation_step"],
            "camera_space_auto_method": args_dict["camera_space_auto_method"],
            "camera_space_iqr_multiplier": args_dict["camera_space_iqr_multiplier"],
            "max_camera_space_wrist_abs": threshold_info["resolved"]["max_camera_space_wrist_abs"],
            "max_camera_space_hand_abs": threshold_info["resolved"]["max_camera_space_hand_abs"],
            "camera_space_wrist_bounds": threshold_info["resolved"]["camera_space_wrist_bounds"],
            "camera_space_hand_bounds": threshold_info["resolved"]["camera_space_hand_bounds"],
            "camera_space_axis_abs_cap": args_dict["camera_space_axis_abs_cap"],
            "compute_motion_metrics": bool(args_dict["compute_motion_metrics"]),
            "compute_camera_space_metrics": bool(args_dict["compute_camera_space_metrics"]),
        },
        "auto_thresholds": threshold_info,
        "total_shards": len(analysis_results),
        "total_samples": total_samples,
        "total_incomplete_samples": total_incomplete_samples,
        "total_clips": total_clips,
        "kept_clips": kept_clips,
        "dropped_clips": total_clips - kept_clips,
        "analysis_errored_shard_count": len(analysis_errored_shards),
        "analysis_errored_shards": analysis_errored_shards,
        "reason_counts": dict(sorted(reason_counts.items())),
        "shards": shards,
        "dropped": dropped,
    }
    if output_dir:
        report["rewrite"] = rewrite
    return report


def _new_analysis_progress_state() -> dict:
    return {
        "samples_total": 0,
        "clips_total": 0,
        "incomplete_samples": 0,
        "errored_shards": 0,
        "dropped_clips": 0,
        "dropped_frames": 0,
    }


def _update_analysis_progress_state(state: dict, shard_result: dict, decision_criteria: dict | None = None) -> None:
    state["samples_total"] += int(shard_result.get("samples_total", 0))
    state["clips_total"] += int(shard_result.get("clips_total", 0))
    state["incomplete_samples"] += int(shard_result.get("incomplete_samples", 0))
    if shard_result.get("shard_error") is not None:
        state["errored_shards"] += 1
    if decision_criteria is None:
        return
    for item in shard_result.get("clip_metrics", []):
        keep, _ = decide_clip_quality(item["metrics"], decision_criteria)
        if keep:
            continue
        state["dropped_clips"] += 1
        state["dropped_frames"] += int(item["metrics"].get("frames_total", 0))


def _set_analysis_progress_postfix(progress, state: dict, *, show_drop: bool) -> None:
    postfix = {
        "samples": int(state["samples_total"]),
        "clips": int(state["clips_total"]),
        "incomplete": int(state["incomplete_samples"]),
    }
    if int(state["errored_shards"]) > 0:
        postfix["err"] = int(state["errored_shards"])
    if show_drop:
        postfix["drop_clips"] = int(state["dropped_clips"])
        postfix["drop_frames"] = int(state["dropped_frames"])
    progress.set_postfix(refresh=False, **postfix)


def _new_rewrite_progress_state() -> dict:
    return {
        "frames_written": 0,
        "clips_written": 0,
        "frames_dropped": 0,
        "clips_dropped": 0,
        "incomplete_samples": 0,
        "errored_shards": 0,
    }


def _update_rewrite_progress_state(state: dict, shard_result: dict) -> None:
    state["frames_written"] += int(shard_result.get("frames_written", 0))
    state["clips_written"] += int(shard_result.get("clips_written", 0))
    state["frames_dropped"] += int(shard_result.get("frames_dropped", 0))
    state["clips_dropped"] += int(shard_result.get("clips_dropped", 0))
    state["incomplete_samples"] += int(shard_result.get("incomplete_samples", 0))
    if shard_result.get("shard_error") is not None:
        state["errored_shards"] += 1


def _set_rewrite_progress_postfix(progress, state: dict) -> None:
    postfix = {
        "keep_clips": int(state["clips_written"]),
        "drop_clips": int(state["clips_dropped"]),
        "keep_frames": int(state["frames_written"]),
        "drop_frames": int(state["frames_dropped"]),
        "incomplete": int(state["incomplete_samples"]),
    }
    if int(state["errored_shards"]) > 0:
        postfix["err"] = int(state["errored_shards"])
    progress.set_postfix(refresh=False, **postfix)


def _is_same_or_nested(path_a: Path, path_b: Path) -> bool:
    if path_a == path_b:
        return True
    try:
        path_a.relative_to(path_b)
        return True
    except ValueError:
        return False


def validate_io_dirs(source_dir: Path, output_dir: Path | None):
    if output_dir is None:
        return

    source_resolved = source_dir.resolve()
    output_resolved = output_dir.resolve()

    if source_resolved == output_resolved:
        raise ValueError("--output_dir must be different from --source_shard_dir")
    if _is_same_or_nested(output_resolved, source_resolved):
        raise ValueError("--output_dir must not be inside --source_shard_dir")
    if _is_same_or_nested(source_resolved, output_resolved):
        raise ValueError("--source_shard_dir must not be inside --output_dir")


def main():
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.chunksize < 0:
        raise ValueError("--chunksize must be >= 0")

    source_dir = Path(args.source_shard_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source shard dir not found: {source_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else None
    validate_io_dirs(source_dir, output_dir)

    shard_paths = list(iter_shard_paths(str(source_dir)))
    if not shard_paths:
        raise RuntimeError(f"No shard tar files found in {source_dir}")
    total_shards = len(shard_paths)
    start_shard = int(args.start_shard)
    end_shard = total_shards if args.end_shard is None else int(args.end_shard)
    if start_shard < 0 or start_shard >= total_shards:
        raise ValueError(f"--start_shard {start_shard} is out of range [0, {total_shards})")
    if end_shard < start_shard or end_shard > total_shards:
        raise ValueError(f"--end_shard {end_shard} is out of range [{start_shard}, {total_shards}]")
    shard_paths = shard_paths[start_shard:end_shard]
    if not shard_paths:
        raise RuntimeError(f"No shards selected in range [{start_shard}, {end_shard}) from {source_dir}")

    if not args.outlier_checks:
        args.min_presence_ratio = None
        args.max_hand_translation_step = None
        args.max_camera_translation_step = None
        args.max_camera_rotation_step = None
        args.max_camera_space_wrist_abs = None
        args.max_camera_space_hand_abs = None
        args.camera_space_axis_abs_cap = None

    use_auto_camera_space_thresholds = (
        bool(args.outlier_checks)
        and (args.max_camera_space_wrist_abs is None or args.max_camera_space_hand_abs is None)
    )
    compute_motion_metrics = any(
        value is not None
        for value in (
            args.max_hand_translation_step,
            args.max_camera_translation_step,
            args.max_camera_rotation_step,
        )
    )
    compute_camera_space_metrics = (
        use_auto_camera_space_thresholds
        or args.max_camera_space_wrist_abs is not None
        or args.max_camera_space_hand_abs is not None
        or args.camera_space_axis_abs_cap is not None
    )
    chunksize = int(args.chunksize) if args.chunksize > 0 else _auto_chunksize(len(shard_paths), args.workers)

    args_dict = {
        "min_instruction_num": args.min_instruction_num,
        "outlier_checks": bool(args.outlier_checks),
        "min_presence_ratio": args.min_presence_ratio,
        "max_hand_translation_step": args.max_hand_translation_step,
        "max_camera_translation_step": args.max_camera_translation_step,
        "max_camera_rotation_step": args.max_camera_rotation_step,
        "max_camera_space_wrist_abs": args.max_camera_space_wrist_abs,
        "max_camera_space_hand_abs": args.max_camera_space_hand_abs,
        "camera_space_auto_method": args.camera_space_auto_method,
        "camera_space_iqr_multiplier": args.camera_space_iqr_multiplier,
        "camera_space_axis_abs_cap": args.camera_space_axis_abs_cap,
        "camera_space_abs_percentile": args.camera_space_abs_percentile,
        "camera_space_abs_scale": args.camera_space_abs_scale,
        "use_auto_camera_space_thresholds": use_auto_camera_space_thresholds,
        "compute_motion_metrics": bool(compute_motion_metrics),
        "compute_camera_space_metrics": bool(compute_camera_space_metrics),
        "chunksize": chunksize,
        "start_shard": start_shard,
        "end_shard": end_shard,
        "total_shards_available": total_shards,
        "output_dir": str(output_dir) if output_dir else None,
    }
    progress_decision_criteria = None if args.outlier_checks else dict(args_dict)

    if args.workers <= 1:
        analysis_results = []
        progress_state = _new_analysis_progress_state()
        with tqdm(shard_paths, desc="Analyze shards") as progress:
            for shard_path in progress:
                shard_result = analyze_shard(
                    shard_path,
                    compute_motion_metrics=compute_motion_metrics,
                    compute_camera_space_metrics=compute_camera_space_metrics,
                )
                analysis_results.append(shard_result)
                _update_analysis_progress_state(progress_state, shard_result, progress_decision_criteria)
                _set_analysis_progress_postfix(progress, progress_state, show_drop=progress_decision_criteria is not None)
    else:
        mp_context = get_context()
        with mp_context.Pool(
            args.workers,
            initializer=_worker_init,
            initargs=(args_dict, str(output_dir) if output_dir else None, None),
        ) as pool:
            analysis_results = []
            progress_state = _new_analysis_progress_state()
            with tqdm(total=len(shard_paths), desc="Analyze shards") as progress:
                for shard_result in pool.imap_unordered(_worker_analyze_shard, shard_paths, chunksize=chunksize):
                    analysis_results.append(shard_result)
                    progress.update(1)
                    _update_analysis_progress_state(progress_state, shard_result, progress_decision_criteria)
                    _set_analysis_progress_postfix(progress, progress_state, show_drop=progress_decision_criteria is not None)

    analysis_results.sort(key=lambda item: item["shard_name"])
    clip_metrics = []
    clip_to_shard = {}
    for result in analysis_results:
        for item in result["clip_metrics"]:
            clip_metrics.append(item["metrics"])
            clip_to_shard[item["clip_id"]] = result["shard_name"]

    threshold_info = resolve_auto_quality_thresholds(clip_metrics, args_dict)
    resolved_args = dict(args_dict)
    resolved_args.update(threshold_info["resolved"])

    clip_decisions = []
    keep_by_clip = {}
    for result in analysis_results:
        for item in result["clip_metrics"]:
            keep, reasons = decide_clip_quality(item["metrics"], resolved_args)
            keep_by_clip[item["clip_id"]] = bool(keep)
            clip_decisions.append(
                {
                    "clip_id": item["clip_id"],
                    "shard_name": clip_to_shard[item["clip_id"]],
                    "keep": bool(keep),
                    "reasons": reasons,
                    "metrics": item["metrics"],
                }
            )

    rewrite_results = None
    if output_dir:
        if args.workers <= 1:
            rewrite_results = []
            progress_state = _new_rewrite_progress_state()
            with tqdm(shard_paths, desc="Rewrite shards") as progress:
                for shard_path in progress:
                    shard_result = rewrite_shard(shard_path, str(output_dir), keep_by_clip)
                    rewrite_results.append(shard_result)
                    _update_rewrite_progress_state(progress_state, shard_result)
                    _set_rewrite_progress_postfix(progress, progress_state)
        else:
            mp_context = get_context()
            with mp_context.Pool(
                args.workers,
                initializer=_worker_init,
                initargs=(resolved_args, str(output_dir), keep_by_clip),
            ) as pool:
                rewrite_results = []
                progress_state = _new_rewrite_progress_state()
                with tqdm(total=len(shard_paths), desc="Rewrite shards") as progress:
                    for shard_result in pool.imap_unordered(_worker_rewrite_shard, shard_paths, chunksize=chunksize):
                        rewrite_results.append(shard_result)
                        progress.update(1)
                        _update_rewrite_progress_state(progress_state, shard_result)
                        _set_rewrite_progress_postfix(progress, progress_state)
        rewrite_results.sort(key=lambda item: item["shard_name"])

    report = build_report(source_dir, output_dir, analysis_results, clip_decisions, resolved_args, threshold_info, rewrite_results)

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
