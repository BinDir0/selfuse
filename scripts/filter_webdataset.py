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

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.webdataset_rewriter import (  # noqa: E402
    iter_shard_paths,
    iter_shard_samples,
    validate_sample_record,
    write_sample_to_tar,
)
from lib.pipeline.quality_metrics import (  # noqa: E402
    decide_clip_quality,
    decode_lowdim,
    finalize_clip_quality_metrics,
    new_clip_quality_stats,
    parse_frame_index,
    resolve_auto_quality_thresholds,
    update_clip_quality_stats,
)


DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))
_WORKER_ARGS = None
_WORKER_OUTPUT_DIR = None
_WORKER_KEEP_BY_CLIP = None
_SHARD_DATA_EXCEPTIONS = (OSError, tarfile.TarError, ValueError)


def build_parser():
    parser = argparse.ArgumentParser(description="Analyze and filter existing WebDataset shards")
    parser.add_argument("--source_shard_dir", required=True, help="Source directory containing shard tar files")
    parser.add_argument("--output_dir", default=None, help="Optional output directory for filtered shards")
    parser.add_argument("--report_out", default=None, help="Optional JSON report path")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel shard workers")
    parser.add_argument(
        "--drop_nonfinite_lowdim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop clips containing any NaN/Inf lowdim frame",
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
        "--camera_space_abs_percentile",
        type=float,
        default=99.0,
        help="Percentile used for automatic camera-space absolute-value thresholds",
    )
    parser.add_argument(
        "--camera_space_abs_scale",
        type=float,
        default=3.0,
        help="Scale multiplier applied to the chosen percentile for automatic camera-space thresholds",
    )
    return parser


def _new_clip_stats(clip_id: str) -> dict:
    return new_clip_quality_stats(clip_id)


def _update_clip_stats(stats: dict, sample_key: str, meta: dict, lowdim, *, count_invalid_lowdim: bool = True) -> None:
    frame_idx = parse_frame_index(sample_key)
    update_clip_quality_stats(
        stats,
        frame_idx,
        int(meta.get("instruction_num", 0) or 0),
        int(meta.get("presence", 0) or 0),
        lowdim,
        count_invalid_lowdim=count_invalid_lowdim,
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


def analyze_shard(shard_path: str) -> dict:
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
                _update_clip_stats(current_clip_stats, sample["key"], {}, None, count_invalid_lowdim=False)
                continue

            validate_sample_record(sample)

            if meta is None:
                current_clip_stats["invalid_meta_frames"] += 1
                _update_clip_stats(current_clip_stats, sample["key"], {}, None, count_invalid_lowdim=False)
            else:
                try:
                    lowdim = decode_lowdim(sample["lowdim_bytes"])
                except Exception:
                    lowdim = None
                _update_clip_stats(current_clip_stats, sample["key"], meta, lowdim)
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
        "shard_written": 0,
        "shard_error": None,
    }

    tar_writer = None
    current_clip_id = None
    current_keep = False
    clip_wrote_frames = False
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
                current_clip_id = clip_id
                current_keep = bool(keep_by_clip.get(clip_id, False))
                clip_wrote_frames = False

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
            )
            result["frames_written"] += 1
            clip_wrote_frames = True

        if current_clip_id is not None and current_keep and clip_wrote_frames:
            result["clips_written"] += 1
    except _SHARD_DATA_EXCEPTIONS as exc:
        if tar_writer is not None:
            tar_writer.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        result["frames_written"] = 0
        result["clips_written"] = 0
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
    return analyze_shard(shard_path)


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
        "mode": "two_pass_auto_threshold" if args_dict["use_auto_camera_space_thresholds"] else "two_pass_manual_threshold",
        "criteria": {
            "drop_nonfinite_lowdim": bool(args_dict["drop_nonfinite_lowdim"]),
            "min_instruction_num": args_dict["min_instruction_num"],
            "min_presence_ratio": args_dict["min_presence_ratio"],
            "max_hand_translation_step": args_dict["max_hand_translation_step"],
            "max_camera_translation_step": args_dict["max_camera_translation_step"],
            "max_camera_rotation_step": args_dict["max_camera_rotation_step"],
            "max_camera_space_wrist_abs": threshold_info["resolved"]["max_camera_space_wrist_abs"],
            "max_camera_space_hand_abs": threshold_info["resolved"]["max_camera_space_hand_abs"],
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

    source_dir = Path(args.source_shard_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source shard dir not found: {source_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else None
    validate_io_dirs(source_dir, output_dir)

    shard_paths = list(iter_shard_paths(str(source_dir)))
    if not shard_paths:
        raise RuntimeError(f"No shard tar files found in {source_dir}")

    args_dict = {
        "drop_nonfinite_lowdim": bool(args.drop_nonfinite_lowdim),
        "min_instruction_num": args.min_instruction_num,
        "min_presence_ratio": args.min_presence_ratio,
        "max_hand_translation_step": args.max_hand_translation_step,
        "max_camera_translation_step": args.max_camera_translation_step,
        "max_camera_rotation_step": args.max_camera_rotation_step,
        "max_camera_space_wrist_abs": args.max_camera_space_wrist_abs,
        "max_camera_space_hand_abs": args.max_camera_space_hand_abs,
        "camera_space_abs_percentile": args.camera_space_abs_percentile,
        "camera_space_abs_scale": args.camera_space_abs_scale,
        "use_auto_camera_space_thresholds": (
            args.max_camera_space_wrist_abs is None or args.max_camera_space_hand_abs is None
        ),
        "output_dir": str(output_dir) if output_dir else None,
    }

    if args.workers <= 1:
        analysis_results = [analyze_shard(shard_path) for shard_path in tqdm(shard_paths, desc="Analyze shards")]
    else:
        mp_context = get_context()
        with mp_context.Pool(
            args.workers,
            initializer=_worker_init,
            initargs=(args_dict, str(output_dir) if output_dir else None, None),
        ) as pool:
            analysis_results = list(
                tqdm(
                    pool.imap_unordered(_worker_analyze_shard, shard_paths, chunksize=1),
                    total=len(shard_paths),
                    desc="Analyze shards",
                )
            )

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
            rewrite_results = [
                rewrite_shard(shard_path, str(output_dir), keep_by_clip)
                for shard_path in tqdm(shard_paths, desc="Rewrite shards")
            ]
        else:
            mp_context = get_context()
            with mp_context.Pool(
                args.workers,
                initializer=_worker_init,
                initargs=(resolved_args, str(output_dir), keep_by_clip),
            ) as pool:
                rewrite_results = list(
                    tqdm(
                        pool.imap_unordered(_worker_rewrite_shard, shard_paths, chunksize=1),
                        total=len(shard_paths),
                        desc="Rewrite shards",
                    )
                )
        rewrite_results.sort(key=lambda item: item["shard_name"])

    report = build_report(source_dir, output_dir, analysis_results, clip_decisions, resolved_args, threshold_info, rewrite_results)

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
