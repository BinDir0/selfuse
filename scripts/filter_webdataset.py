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
_WORKER_ARGS = None
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


def _finalize_clip_metrics(stats: dict) -> dict:
    return {
        "frames_total": int(stats["frames_total"]),
        "frames_kept_candidate": int(stats["frames_kept_candidate"]),
        "presence_ratio": (
            float(stats["presence_nonzero_frames"]) / float(stats["frames_total"])
            if stats["frames_total"] > 0
            else 0.0
        ),
        "instruction_num_max": int(stats["instruction_num_max"]),
        "nonfinite_lowdim_frames": int(stats["nonfinite_lowdim_frames"]),
        "invalid_meta_frames": int(stats["invalid_meta_frames"]),
        "invalid_lowdim_frames": int(stats["invalid_lowdim_frames"]),
        "max_hand_translation_step": float(stats["max_hand_translation_step"]),
        "max_camera_translation_step": float(stats["max_camera_translation_step"]),
        "max_camera_rotation_step": float(stats["max_camera_rotation_step"]),
    }


def decide_clip_keep(metrics: dict, args_dict: dict) -> tuple[bool, list[str]]:
    reasons = []
    if metrics["invalid_meta_frames"] > 0:
        reasons.append("invalid_meta")
    if metrics["invalid_lowdim_frames"] > 0:
        reasons.append("invalid_lowdim")
    if args_dict["drop_nonfinite_lowdim"] and metrics["nonfinite_lowdim_frames"] > 0:
        reasons.append("nonfinite_lowdim")
    if args_dict["min_instruction_num"] is not None and metrics["instruction_num_max"] < args_dict["min_instruction_num"]:
        reasons.append("instruction_num_below_min")
    if args_dict["min_presence_ratio"] is not None and metrics["presence_ratio"] < args_dict["min_presence_ratio"]:
        reasons.append("presence_ratio_below_min")
    if (
        args_dict["max_hand_translation_step"] is not None
        and metrics["max_hand_translation_step"] > args_dict["max_hand_translation_step"]
    ):
        reasons.append("hand_translation_step_exceeded")
    if (
        args_dict["max_camera_translation_step"] is not None
        and metrics["max_camera_translation_step"] > args_dict["max_camera_translation_step"]
    ):
        reasons.append("camera_translation_step_exceeded")
    if (
        args_dict["max_camera_rotation_step"] is not None
        and metrics["max_camera_rotation_step"] > args_dict["max_camera_rotation_step"]
    ):
        reasons.append("camera_rotation_step_exceeded")
    return not reasons, reasons


def _sample_clip_id(sample: dict, meta: dict | None) -> str:
    if meta is not None:
        clip_id = meta.get("clip_id")
        if clip_id:
            return str(clip_id)
    return sample["key"].rsplit("_f", 1)[0]


def _flush_clip_block(
    *,
    clip_id: str | None,
    clip_stats: dict | None,
    clip_samples: list[dict],
    args_dict: dict,
    tar_writer,
    shard_result: dict,
) -> tuple[object, dict]:
    if clip_id is None or clip_stats is None:
        return tar_writer, shard_result

    metrics = _finalize_clip_metrics(clip_stats)
    keep, reasons = decide_clip_keep(metrics, args_dict)
    decision = {
        "clip_id": clip_id,
        "keep": bool(keep),
        "reasons": reasons,
        "metrics": metrics,
    }
    shard_result["clip_decisions"].append(decision)
    shard_result["clips_total"] += 1
    if keep:
        shard_result["kept_clips"] += 1
        if tar_writer is None and args_dict["output_dir"]:
            os.makedirs(args_dict["output_dir"], exist_ok=True)
            tar_writer = tarfile.open(shard_result["tmp_path"], "w")
        if tar_writer is not None:
            frames_written = 0
            for sample in clip_samples:
                write_sample_to_tar(
                    tar_writer,
                    sample["key"],
                    sample["image_bytes"],
                    sample["lowdim_bytes"],
                    sample["meta_bytes"],
                )
                frames_written += 1
            shard_result["frames_written"] += frames_written
            shard_result["clips_written"] += 1
    else:
        shard_result["dropped_clips"] += 1
    return tar_writer, shard_result


def process_shard(shard_path: str, args_dict: dict) -> dict:
    shard_name = os.path.basename(shard_path)
    output_path = None
    tmp_path = None
    if args_dict["output_dir"]:
        output_path = os.path.join(args_dict["output_dir"], shard_name)
        tmp_path = f"{output_path}.tmp"

    shard_result = {
        "shard_name": shard_name,
        "output_path": output_path,
        "tmp_path": tmp_path,
        "samples_total": 0,
        "frames_written": 0,
        "clips_total": 0,
        "kept_clips": 0,
        "dropped_clips": 0,
        "clips_written": 0,
        "clip_decisions": [],
    }

    current_clip_id = None
    current_clip_stats = None
    current_clip_samples = []
    tar_writer = None

    try:
        for sample in iter_shard_samples(shard_path):
            shard_result["samples_total"] += 1
            validate_sample_record(sample)

            meta = None
            try:
                meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            except Exception:
                meta = None
            clip_id = _sample_clip_id(sample, meta)

            if current_clip_id is None:
                current_clip_id = clip_id
                current_clip_stats = _new_clip_stats(clip_id)
            elif clip_id != current_clip_id:
                tar_writer, shard_result = _flush_clip_block(
                    clip_id=current_clip_id,
                    clip_stats=current_clip_stats,
                    clip_samples=current_clip_samples,
                    args_dict=args_dict,
                    tar_writer=tar_writer,
                    shard_result=shard_result,
                )
                current_clip_id = clip_id
                current_clip_stats = _new_clip_stats(clip_id)
                current_clip_samples = []

            if meta is None:
                current_clip_stats["invalid_meta_frames"] += 1
                _update_clip_stats(current_clip_stats, sample["key"], {}, None, count_invalid_lowdim=False)
            else:
                try:
                    lowdim = decode_lowdim(sample["lowdim_bytes"])
                except Exception:
                    lowdim = None
                _update_clip_stats(current_clip_stats, sample["key"], meta, lowdim)

            if args_dict["output_dir"]:
                current_clip_samples.append(sample)

        tar_writer, shard_result = _flush_clip_block(
            clip_id=current_clip_id,
            clip_stats=current_clip_stats,
            clip_samples=current_clip_samples,
            args_dict=args_dict,
            tar_writer=tar_writer,
            shard_result=shard_result,
        )
    except Exception:
        if tar_writer is not None:
            tar_writer.close()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if tar_writer is not None:
        tar_writer.close()
    if output_path:
        if shard_result["frames_written"] == 0:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        else:
            os.replace(tmp_path, output_path)

    shard_result["shard_written"] = 1 if shard_result["frames_written"] > 0 else 0
    shard_result.pop("tmp_path", None)
    return shard_result


def _worker_init(args_dict: dict, output_dir: str | None):
    global _WORKER_ARGS, _WORKER_OUTPUT_DIR
    _WORKER_ARGS = dict(args_dict)
    _WORKER_OUTPUT_DIR = output_dir
    _WORKER_ARGS["output_dir"] = output_dir


def _worker_process_shard(shard_path: str) -> dict:
    return process_shard(shard_path, _WORKER_ARGS)


def build_report(source_shard_dir: Path, output_dir: Path | None, shard_results: list[dict], args_dict: dict) -> dict:
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
    }

    for result in shard_results:
        total_samples += result["samples_total"]
        total_clips += result["clips_total"]
        kept_clips += result["kept_clips"]
        shards[result["shard_name"]] = {
            "samples_total": result["samples_total"],
            "clips_total": result["clips_total"],
            "kept_clips": result["kept_clips"],
            "dropped_clips": result["dropped_clips"],
            "frames_written": result["frames_written"],
        }
        rewrite["shards_written"] += result["shard_written"]
        rewrite["frames_written"] += result["frames_written"]
        rewrite["clips_written"] += result["clips_written"]

        for item in result["clip_decisions"]:
            if item["keep"]:
                continue
            reason_counts.update(item["reasons"])
            dropped.append(
                {
                    "clip_id": item["clip_id"],
                    "shard_name": result["shard_name"],
                    "reasons": item["reasons"],
                    "metrics": item["metrics"],
                }
            )

    report = {
        "source_shard_dir": str(source_shard_dir.resolve()),
        "output_dir": str(output_dir.resolve()) if output_dir else None,
        "mode": "single_pass_streaming",
        "criteria": {
            "drop_nonfinite_lowdim": bool(args_dict["drop_nonfinite_lowdim"]),
            "min_instruction_num": args_dict["min_instruction_num"],
            "min_presence_ratio": args_dict["min_presence_ratio"],
            "max_hand_translation_step": args_dict["max_hand_translation_step"],
            "max_camera_translation_step": args_dict["max_camera_translation_step"],
            "max_camera_rotation_step": args_dict["max_camera_rotation_step"],
        },
        "total_shards": len(shard_results),
        "total_samples": total_samples,
        "total_clips": total_clips,
        "kept_clips": kept_clips,
        "dropped_clips": total_clips - kept_clips,
        "reason_counts": dict(sorted(reason_counts.items())),
        "shards": shards,
        "dropped": dropped,
    }
    if output_dir:
        report["rewrite"] = rewrite
    return report


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

    args_dict = {
        "drop_nonfinite_lowdim": bool(args.drop_nonfinite_lowdim),
        "min_instruction_num": args.min_instruction_num,
        "min_presence_ratio": args.min_presence_ratio,
        "max_hand_translation_step": args.max_hand_translation_step,
        "max_camera_translation_step": args.max_camera_translation_step,
        "max_camera_rotation_step": args.max_camera_rotation_step,
        "output_dir": str(output_dir) if output_dir else None,
    }

    if args.workers <= 1:
        shard_results = [process_shard(shard_path, args_dict) for shard_path in tqdm(shard_paths, desc="Process shards")]
    else:
        mp_context = get_context()
        with mp_context.Pool(
            args.workers,
            initializer=_worker_init,
            initargs=(args_dict, str(output_dir) if output_dir else None),
        ) as pool:
            shard_results = list(
                tqdm(
                    pool.imap_unordered(_worker_process_shard, shard_paths, chunksize=1),
                    total=len(shard_paths),
                    desc="Process shards",
                )
            )

    shard_results.sort(key=lambda item: item["shard_name"])
    report = build_report(source_dir, output_dir, shard_results, args_dict)

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
