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
    decode_lowdim,
    extract_lowdim_components,
    is_finite_array,
    parse_frame_index,
)


DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))
_WORKER_KEEP_CLIPS = None
_WORKER_OUTPUT_DIR = None


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
    return parser


def _new_clip_stats(clip_id: str) -> dict:
    return {
        "clip_id": clip_id,
        "frames_total": 0,
        "frames_kept_candidate": 0,
        "presence_nonzero_frames": 0,
        "nonfinite_lowdim_frames": 0,
        "invalid_meta_frames": 0,
        "invalid_lowdim_frames": 0,
        "instruction_num_max": 0,
        "shards": set(),
        "max_hand_translation_step": 0.0,
        "max_camera_translation_step": 0.0,
        "max_camera_rotation_step": 0.0,
        "_prev_frame_idx": None,
        "_prev_left": None,
        "_prev_right": None,
        "_prev_extrinsic": None,
        "_prev_finite": False,
    }


def _update_clip_stats(stats: dict, sample_key: str, meta: dict, lowdim, *, count_invalid_lowdim: bool = True) -> None:
    stats["frames_total"] += 1
    frame_idx = parse_frame_index(sample_key)
    instruction_num = int(meta.get("instruction_num", 0) or 0)
    stats["instruction_num_max"] = max(stats["instruction_num_max"], instruction_num)
    presence = int(meta.get("presence", 0) or 0)
    if presence > 0:
        stats["presence_nonzero_frames"] += 1

    if lowdim is None:
        if count_invalid_lowdim:
            stats["invalid_lowdim_frames"] += 1
        stats["_prev_frame_idx"] = frame_idx
        stats["_prev_left"] = None
        stats["_prev_right"] = None
        stats["_prev_extrinsic"] = None
        stats["_prev_finite"] = False
        return

    if not is_finite_array(lowdim):
        stats["nonfinite_lowdim_frames"] += 1
        stats["_prev_frame_idx"] = frame_idx
        stats["_prev_left"] = None
        stats["_prev_right"] = None
        stats["_prev_extrinsic"] = None
        stats["_prev_finite"] = False
        return

    stats["frames_kept_candidate"] += 1
    parts = extract_lowdim_components(lowdim)
    current_left = parts["left_translation"]
    current_right = parts["right_translation"]
    current_extrinsic = parts["extrinsic"]

    prev_idx = stats["_prev_frame_idx"]
    if stats["_prev_finite"] and prev_idx is not None:
        frame_gap = max(1, frame_idx - prev_idx)
        left_step = float((((current_left - stats["_prev_left"]) ** 2).sum() ** 0.5) / frame_gap)
        right_step = float((((current_right - stats["_prev_right"]) ** 2).sum() ** 0.5) / frame_gap)
        prev_rot = stats["_prev_extrinsic"][:3, :3]
        prev_trans = stats["_prev_extrinsic"][:3, 3]
        curr_rot = current_extrinsic[:3, :3]
        curr_trans = current_extrinsic[:3, 3]
        camera_translation_step = float((((curr_trans - prev_trans) ** 2).sum() ** 0.5) / frame_gap)
        camera_rotation_step = float((((curr_rot - prev_rot).reshape(-1) ** 2).sum() ** 0.5) / frame_gap)

        stats["max_hand_translation_step"] = max(
            stats["max_hand_translation_step"],
            left_step,
            right_step,
        )
        stats["max_camera_translation_step"] = max(
            stats["max_camera_translation_step"],
            camera_translation_step,
        )
        stats["max_camera_rotation_step"] = max(
            stats["max_camera_rotation_step"],
            camera_rotation_step,
        )

    stats["_prev_frame_idx"] = frame_idx
    stats["_prev_left"] = current_left
    stats["_prev_right"] = current_right
    stats["_prev_extrinsic"] = current_extrinsic
    stats["_prev_finite"] = True


def _finalize_partial_stats(partial: dict) -> dict:
    finalized = {}
    for clip_id, stats in partial.items():
        clip_stats = dict(stats)
        clip_stats["shards"] = sorted(stats["shards"])
        for key in ("_prev_frame_idx", "_prev_left", "_prev_right", "_prev_extrinsic", "_prev_finite"):
            clip_stats.pop(key, None)
        finalized[clip_id] = clip_stats
    return finalized


def analyze_shard(shard_path: str) -> dict:
    shard_name = os.path.basename(shard_path)
    clip_stats = {}
    samples_total = 0
    for sample in iter_shard_samples(shard_path):
        samples_total += 1
        validate_sample_record(sample)
        meta = None
        clip_id = None
        try:
            meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            clip_id = str(meta.get("clip_id") or sample["key"].rsplit("_f", 1)[0])
        except Exception:
            clip_id = sample["key"].rsplit("_f", 1)[0]
        stats = clip_stats.setdefault(clip_id, _new_clip_stats(clip_id))
        stats["shards"].add(shard_name)

        if meta is None:
            stats["invalid_meta_frames"] += 1
            _update_clip_stats(stats, sample["key"], {}, None, count_invalid_lowdim=False)
            continue

        try:
            lowdim = decode_lowdim(sample["lowdim_bytes"])
        except Exception:
            lowdim = None
        _update_clip_stats(stats, sample["key"], meta, lowdim)

    return {
        "shard_name": shard_name,
        "samples_total": samples_total,
        "clips": _finalize_partial_stats(clip_stats),
    }


def merge_clip_stats(partials: list[dict]) -> tuple[dict, dict]:
    merged = {}
    shard_stats = {}
    for partial in partials:
        shard_stats[partial["shard_name"]] = {"samples_total": partial["samples_total"]}
        for clip_id, clip_stats in partial["clips"].items():
            target = merged.setdefault(
                clip_id,
                {
                    "clip_id": clip_id,
                    "frames_total": 0,
                    "frames_kept_candidate": 0,
                    "presence_nonzero_frames": 0,
                    "nonfinite_lowdim_frames": 0,
                    "invalid_meta_frames": 0,
                    "invalid_lowdim_frames": 0,
                    "instruction_num_max": 0,
                    "shards": set(),
                    "max_hand_translation_step": 0.0,
                    "max_camera_translation_step": 0.0,
                    "max_camera_rotation_step": 0.0,
                },
            )
            target["frames_total"] += clip_stats["frames_total"]
            target["frames_kept_candidate"] += clip_stats["frames_kept_candidate"]
            target["presence_nonzero_frames"] += clip_stats["presence_nonzero_frames"]
            target["nonfinite_lowdim_frames"] += clip_stats["nonfinite_lowdim_frames"]
            target["invalid_meta_frames"] += clip_stats["invalid_meta_frames"]
            target["invalid_lowdim_frames"] += clip_stats["invalid_lowdim_frames"]
            target["instruction_num_max"] = max(target["instruction_num_max"], clip_stats["instruction_num_max"])
            target["shards"].update(clip_stats["shards"])
            target["max_hand_translation_step"] = max(
                target["max_hand_translation_step"],
                clip_stats["max_hand_translation_step"],
            )
            target["max_camera_translation_step"] = max(
                target["max_camera_translation_step"],
                clip_stats["max_camera_translation_step"],
            )
            target["max_camera_rotation_step"] = max(
                target["max_camera_rotation_step"],
                clip_stats["max_camera_rotation_step"],
            )

    for clip_stats in merged.values():
        clip_stats["shards"] = sorted(clip_stats["shards"])
    return merged, shard_stats


def decide_clip_keep(clip_stats: dict, args) -> tuple[bool, list[str], dict]:
    reasons = []
    metrics = {
        "frames_total": int(clip_stats["frames_total"]),
        "frames_kept_candidate": int(clip_stats["frames_kept_candidate"]),
        "presence_ratio": (
            float(clip_stats["presence_nonzero_frames"]) / float(clip_stats["frames_total"])
            if clip_stats["frames_total"] > 0
            else 0.0
        ),
        "instruction_num_max": int(clip_stats["instruction_num_max"]),
        "nonfinite_lowdim_frames": int(clip_stats["nonfinite_lowdim_frames"]),
        "invalid_meta_frames": int(clip_stats["invalid_meta_frames"]),
        "invalid_lowdim_frames": int(clip_stats["invalid_lowdim_frames"]),
        "max_hand_translation_step": float(clip_stats["max_hand_translation_step"]),
        "max_camera_translation_step": float(clip_stats["max_camera_translation_step"]),
        "max_camera_rotation_step": float(clip_stats["max_camera_rotation_step"]),
        "shards": list(clip_stats["shards"]),
    }

    if clip_stats["invalid_meta_frames"] > 0:
        reasons.append("invalid_meta")
    if clip_stats["invalid_lowdim_frames"] > 0:
        reasons.append("invalid_lowdim")
    if args.drop_nonfinite_lowdim and clip_stats["nonfinite_lowdim_frames"] > 0:
        reasons.append("nonfinite_lowdim")
    if args.min_instruction_num is not None and clip_stats["instruction_num_max"] < args.min_instruction_num:
        reasons.append("instruction_num_below_min")
    if args.min_presence_ratio is not None and metrics["presence_ratio"] < args.min_presence_ratio:
        reasons.append("presence_ratio_below_min")
    if (
        args.max_hand_translation_step is not None
        and clip_stats["max_hand_translation_step"] > args.max_hand_translation_step
    ):
        reasons.append("hand_translation_step_exceeded")
    if (
        args.max_camera_translation_step is not None
        and clip_stats["max_camera_translation_step"] > args.max_camera_translation_step
    ):
        reasons.append("camera_translation_step_exceeded")
    if (
        args.max_camera_rotation_step is not None
        and clip_stats["max_camera_rotation_step"] > args.max_camera_rotation_step
    ):
        reasons.append("camera_rotation_step_exceeded")
    return not reasons, reasons, metrics


def build_report(source_shard_dir: Path, output_dir: Path | None, clip_stats: dict, shard_stats: dict, args) -> tuple[dict, set[str]]:
    reason_counts = Counter()
    kept = 0
    dropped = []
    keep_clips = set()

    for clip_id in sorted(clip_stats):
        keep, reasons, metrics = decide_clip_keep(clip_stats[clip_id], args)
        if keep:
            kept += 1
            keep_clips.add(clip_id)
            continue
        reason_counts.update(reasons)
        dropped.append(
            {
                "clip_id": clip_id,
                "reasons": reasons,
                "metrics": metrics,
            }
        )

    report = {
        "source_shard_dir": str(source_shard_dir.resolve()),
        "output_dir": str(output_dir.resolve()) if output_dir else None,
        "criteria": {
            "drop_nonfinite_lowdim": bool(args.drop_nonfinite_lowdim),
            "min_instruction_num": args.min_instruction_num,
            "min_presence_ratio": args.min_presence_ratio,
            "max_hand_translation_step": args.max_hand_translation_step,
            "max_camera_translation_step": args.max_camera_translation_step,
            "max_camera_rotation_step": args.max_camera_rotation_step,
        },
        "total_shards": len(shard_stats),
        "total_clips": len(clip_stats),
        "kept_clips": kept,
        "dropped_clips": len(clip_stats) - kept,
        "reason_counts": dict(sorted(reason_counts.items())),
        "shards": shard_stats,
        "dropped": dropped,
    }
    return report, keep_clips


def _rewrite_worker_init(output_dir: str, keep_clips: set[str]):
    global _WORKER_OUTPUT_DIR, _WORKER_KEEP_CLIPS
    _WORKER_OUTPUT_DIR = output_dir
    _WORKER_KEEP_CLIPS = keep_clips


def rewrite_shard(shard_path: str, output_dir: str, keep_clips: set[str]) -> dict:
    output_path = os.path.join(output_dir, os.path.basename(shard_path))
    tmp_path = f"{output_path}.tmp"
    frames_written = 0
    clips_written = set()
    tar_writer = None

    try:
        for sample in iter_shard_samples(shard_path):
            validate_sample_record(sample)
            meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            clip_id = str(meta.get("clip_id") or sample["key"].rsplit("_f", 1)[0])
            if clip_id not in keep_clips:
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
            )
            frames_written += 1
            clips_written.add(clip_id)
    except Exception:
        if tar_writer is not None:
            tar_writer.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if tar_writer is not None:
        tar_writer.close()
    if frames_written == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    else:
        os.replace(tmp_path, output_path)

    return {
        "shard_name": os.path.basename(shard_path),
        "frames_written": frames_written,
        "clips_written": len(clips_written),
        "shard_written": 1 if frames_written > 0 else 0,
    }


def _rewrite_worker(shard_path: str) -> dict:
    return rewrite_shard(shard_path, _WORKER_OUTPUT_DIR, _WORKER_KEEP_CLIPS)


def rewrite_shards(shard_paths: list[str], output_dir: Path, keep_clips: set[str], workers: int) -> dict:
    totals = {
        "shards_written": 0,
        "frames_written": 0,
        "clips_written": 0,
    }
    if workers <= 1:
        iterator = (rewrite_shard(shard_path, str(output_dir), keep_clips) for shard_path in shard_paths)
    else:
        mp_context = get_context()
        with mp_context.Pool(
            workers,
            initializer=_rewrite_worker_init,
            initargs=(str(output_dir), keep_clips),
        ) as pool:
            iterator = pool.imap_unordered(_rewrite_worker, shard_paths, chunksize=1)

    for result in tqdm(iterator, total=len(shard_paths), desc="Rewrite shards"):
        totals["shards_written"] += result["shard_written"]
        totals["frames_written"] += result["frames_written"]
        totals["clips_written"] += result["clips_written"]
    return totals


def main():
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    source_dir = Path(args.source_shard_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source shard dir not found: {source_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None and output_dir.resolve() == source_dir.resolve():
        raise ValueError("--output_dir must be different from --source_shard_dir")

    shard_paths = list(iter_shard_paths(str(source_dir)))
    if not shard_paths:
        raise RuntimeError(f"No shard tar files found in {source_dir}")

    if args.workers <= 1:
        partials = [analyze_shard(shard_path) for shard_path in tqdm(shard_paths, desc="Analyze shards")]
    else:
        mp_context = get_context()
        with mp_context.Pool(args.workers) as pool:
            partials = list(tqdm(pool.imap_unordered(analyze_shard, shard_paths, chunksize=1), total=len(shard_paths), desc="Analyze shards"))

    clip_stats, shard_stats = merge_clip_stats(partials)
    report, keep_clips = build_report(source_dir, output_dir, clip_stats, shard_stats, args)

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if output_dir is not None:
        rewrite_totals = rewrite_shards(shard_paths, output_dir, keep_clips, args.workers)
        report["rewrite"] = rewrite_totals
        if args.report_out:
            Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
