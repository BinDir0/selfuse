#!/usr/bin/env python3
"""Filter a clip manifest using completed stage outputs and quality thresholds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.clip_manifest import load_clip_manifest, write_clip_manifest  # noqa: E402
from lib.pipeline.exporters.webdataset_geometry import interpolate_extrinsics, normalize_slam_keyframes  # noqa: E402
from lib.pipeline.quality_metrics import is_finite_array, max_camera_step, max_translation_step  # noqa: E402
from lib.pipeline.stage_api import get_track_range, validate_stage_output  # noqa: E402


DEFAULT_STAGES = "detect_track,motion,slam,infiller"
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))
_WORKER_CONFIG = None


def build_parser():
    parser = argparse.ArgumentParser(description="Filter a manifest using finished stage outputs")
    parser.add_argument("--input_manifest", required=True, help="Input manifest JSONL")
    parser.add_argument("--output_manifest", required=True, help="Output manifest JSONL for kept clips")
    parser.add_argument("--report_out", default=None, help="Optional JSON report path")
    parser.add_argument("--stages", default=DEFAULT_STAGES, help="Comma-separated stage outputs that must validate")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel workers")
    parser.add_argument(
        "--drop_nonfinite_world_res",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop clips when world_space_res contains NaN/Inf",
    )
    parser.add_argument(
        "--drop_nonfinite_slam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop clips when interpolated camera extrinsics contain NaN/Inf",
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
    parser.add_argument("--dry_run", action="store_true", help="Analyze only; do not write output manifest")
    return parser


def parse_stage_list(raw: str) -> list[str]:
    stages = [stage.strip() for stage in str(raw).split(",") if stage.strip()]
    if not stages:
        raise ValueError("Expected at least one stage in --stages")
    return stages


def _load_world_prediction(seq_folder: Path):
    world_file = seq_folder / "world_space_res.pth"
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(world_file)
    return {
        "pred_trans": np.asarray(pred_trans, dtype=np.float32),
        "pred_rot": np.asarray(pred_rot, dtype=np.float32),
        "pred_hand_pose": np.asarray(pred_hand_pose, dtype=np.float32),
        "pred_betas": np.asarray(pred_betas, dtype=np.float32),
        "pred_valid": np.asarray(pred_valid),
    }


def _validate_world_prediction(prediction: dict):
    pred_trans = prediction["pred_trans"]
    pred_rot = prediction["pred_rot"]
    pred_hand_pose = prediction["pred_hand_pose"]
    pred_betas = prediction["pred_betas"]
    pred_valid = prediction["pred_valid"]
    if pred_trans.ndim != 3 or pred_trans.shape[:1] != (2,) or pred_trans.shape[-1] != 3:
        raise ValueError(f"pred_trans shape invalid: {pred_trans.shape}")
    if pred_rot.ndim != 3 or pred_rot.shape[:1] != (2,) or pred_rot.shape[-1] != 3:
        raise ValueError(f"pred_rot shape invalid: {pred_rot.shape}")
    if pred_hand_pose.ndim != 3 or pred_hand_pose.shape[:1] != (2,) or pred_hand_pose.shape[-1] != 45:
        raise ValueError(f"pred_hand_pose shape invalid: {pred_hand_pose.shape}")
    if pred_betas.ndim != 3 or pred_betas.shape[:1] != (2,) or pred_betas.shape[-1] != 10:
        raise ValueError(f"pred_betas shape invalid: {pred_betas.shape}")
    if pred_valid.ndim not in (1, 2):
        raise ValueError(f"pred_valid shape invalid: {pred_valid.shape}")
    if pred_valid.ndim == 2 and pred_valid.shape[0] != 2:
        raise ValueError(f"pred_valid shape invalid: {pred_valid.shape}")


def _normalize_pred_valid(pred_valid: np.ndarray, num_frames: int) -> np.ndarray:
    valid = np.asarray(pred_valid, dtype=np.float32)
    if valid.ndim == 1:
        valid = np.tile(valid.reshape(1, -1), (2, 1))
    if valid.shape[1] != num_frames:
        raise ValueError(f"pred_valid frame count mismatch: {valid.shape[1]} vs {num_frames}")
    return valid > 0.5


def _load_interpolated_camera_extrinsics(seq_folder: Path, start_idx: int, end_idx: int, num_frames: int) -> np.ndarray:
    slam_file = seq_folder / "SLAM" / f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
    slam_data = np.load(str(slam_file), allow_pickle=True)
    if "tstamp" not in slam_data or "traj" not in slam_data or "scale" not in slam_data:
        raise ValueError(f"Invalid SLAM npz keys in {slam_file}")
    tstamps = np.asarray(slam_data["tstamp"], dtype=np.int64)
    traj = np.asarray(slam_data["traj"], dtype=np.float32)
    scale = float(slam_data["scale"])
    tstamps, traj = normalize_slam_keyframes(tstamps, traj)
    if len(tstamps) == 0 or len(traj) == 0:
        raise ValueError(f"No valid SLAM keyframes in {slam_file}")
    return interpolate_extrinsics(tstamps, traj, scale, num_frames)


def evaluate_record(record, stages: list[str], config: dict) -> dict:
    seq_folder = Path(record.descriptor.seq_folder)
    result = {
        "clip_id": record.clip_id,
        "seq_folder": str(seq_folder),
        "keep": False,
        "reasons": [],
        "metrics": {},
    }

    try:
        start_idx, end_idx = get_track_range(seq_folder, fast=False)
    except Exception as error:
        result["reasons"].append("missing_track_range")
        result["metrics"]["track_range_error"] = str(error)
        return result

    result["metrics"]["track_range"] = [int(start_idx), int(end_idx)]
    for stage in stages:
        try:
            validate_stage_output(stage, seq_folder, start_idx, end_idx)
        except Exception as error:
            result["reasons"].append(f"invalid_stage_output:{stage}")
            result["metrics"][f"{stage}_error"] = str(error)
            return result

    try:
        prediction = _load_world_prediction(seq_folder)
        _validate_world_prediction(prediction)
    except Exception as error:
        result["reasons"].append("invalid_world_res")
        result["metrics"]["world_res_error"] = str(error)
        return result

    pred_trans = prediction["pred_trans"]
    pred_rot = prediction["pred_rot"]
    pred_hand_pose = prediction["pred_hand_pose"]
    pred_betas = prediction["pred_betas"]
    pred_valid = _normalize_pred_valid(prediction["pred_valid"], pred_trans.shape[1])

    nonfinite_fields = [
        name
        for name, value in (
            ("pred_trans", pred_trans),
            ("pred_rot", pred_rot),
            ("pred_hand_pose", pred_hand_pose),
            ("pred_betas", pred_betas),
            ("pred_valid", pred_valid),
        )
        if not is_finite_array(value)
    ]
    if nonfinite_fields:
        result["metrics"]["nonfinite_world_res_fields"] = nonfinite_fields
        if config["drop_nonfinite_world_res"]:
            result["reasons"].append("nonfinite_world_res")
            return result

    num_frames = int(pred_trans.shape[1])
    result["metrics"]["num_frames"] = num_frames
    left_hand = max_translation_step(pred_trans[0], valid_mask=pred_valid[0])
    right_hand = max_translation_step(pred_trans[1], valid_mask=pred_valid[1])
    result["metrics"]["left_hand"] = left_hand
    result["metrics"]["right_hand"] = right_hand
    result["metrics"]["max_hand_translation_step"] = max(left_hand["max_step"], right_hand["max_step"])
    if (
        config["max_hand_translation_step"] is not None
        and result["metrics"]["max_hand_translation_step"] > config["max_hand_translation_step"]
    ):
        result["reasons"].append("hand_translation_step_exceeded")

    need_camera_metrics = (
        "slam" in stages
        or config["drop_nonfinite_slam"]
        or config["max_camera_translation_step"] is not None
        or config["max_camera_rotation_step"] is not None
    )
    if need_camera_metrics:
        try:
            extrinsics = _load_interpolated_camera_extrinsics(seq_folder, start_idx, end_idx, num_frames)
        except Exception as error:
            result["reasons"].append("invalid_slam_extrinsics")
            result["metrics"]["slam_error"] = str(error)
            return result

        if not is_finite_array(extrinsics):
            result["metrics"]["nonfinite_slam"] = True
            if config["drop_nonfinite_slam"]:
                result["reasons"].append("nonfinite_slam")
                return result

        camera_metrics = max_camera_step(extrinsics)
        result["metrics"]["camera"] = camera_metrics
        if (
            config["max_camera_translation_step"] is not None
            and camera_metrics["max_translation_step"] > config["max_camera_translation_step"]
        ):
            result["reasons"].append("camera_translation_step_exceeded")
        if (
            config["max_camera_rotation_step"] is not None
            and camera_metrics["max_rotation_step"] > config["max_camera_rotation_step"]
        ):
            result["reasons"].append("camera_rotation_step_exceeded")

    result["keep"] = not result["reasons"]
    return result


def _worker_init(config: dict):
    global _WORKER_CONFIG
    _WORKER_CONFIG = config


def _worker_eval(task):
    index, record = task
    result = evaluate_record(record, _WORKER_CONFIG["stages"], _WORKER_CONFIG)
    result["index"] = index
    return result


def build_report(results: list[dict], input_manifest: Path, output_manifest: Path, config: dict) -> dict:
    reason_counts = Counter()
    kept = 0
    dropped = []
    for item in results:
        if item["keep"]:
            kept += 1
            continue
        dropped.append(
            {
                "clip_id": item["clip_id"],
                "seq_folder": item["seq_folder"],
                "reasons": item["reasons"],
                "metrics": item["metrics"],
            }
        )
        reason_counts.update(item["reasons"])

    return {
        "input_manifest": str(input_manifest.resolve()),
        "output_manifest": str(output_manifest.resolve()),
        "criteria": {
            "stages": list(config["stages"]),
            "drop_nonfinite_world_res": bool(config["drop_nonfinite_world_res"]),
            "drop_nonfinite_slam": bool(config["drop_nonfinite_slam"]),
            "max_hand_translation_step": config["max_hand_translation_step"],
            "max_camera_translation_step": config["max_camera_translation_step"],
            "max_camera_rotation_step": config["max_camera_rotation_step"],
        },
        "total_clips": len(results),
        "kept_clips": kept,
        "dropped_clips": len(results) - kept,
        "reason_counts": dict(sorted(reason_counts.items())),
        "dropped": dropped,
    }


def run_filter(args) -> dict:
    records = load_clip_manifest(args.input_manifest)
    config = {
        "stages": parse_stage_list(args.stages),
        "drop_nonfinite_world_res": bool(args.drop_nonfinite_world_res),
        "drop_nonfinite_slam": bool(args.drop_nonfinite_slam),
        "max_hand_translation_step": args.max_hand_translation_step,
        "max_camera_translation_step": args.max_camera_translation_step,
        "max_camera_rotation_step": args.max_camera_rotation_step,
    }

    tasks = list(enumerate(records))
    if args.workers <= 1:
        results = [dict(evaluate_record(record, config["stages"], config), index=index) for index, record in tqdm(tasks, desc="Filter manifest")]
    else:
        mp_context = get_context()
        with mp_context.Pool(args.workers, initializer=_worker_init, initargs=(config,)) as pool:
            results = list(tqdm(pool.imap(_worker_eval, tasks, chunksize=8), total=len(tasks), desc="Filter manifest"))

    results.sort(key=lambda item: item["index"])
    kept_records = [record for record, result in zip(records, results) if result["keep"]]

    if not args.dry_run:
        write_clip_manifest(kept_records, args.output_manifest)

    report = build_report(results, Path(args.input_manifest), Path(args.output_manifest), config)
    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.dry_run and Path(args.output_manifest).exists():
        print(f"Dry run: not writing {args.output_manifest}")
    report = run_filter(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
