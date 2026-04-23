#!/usr/bin/env python3
"""Audit old BuildAI interpolated WebDataset shards against current export logic."""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, **kwargs):
        return iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.webdataset_rewriter import iter_shard_paths, iter_shard_samples, validate_sample_record
from lib.pipeline.quality_metrics import decode_lowdim, parse_frame_index, validate_lowdim_numeric_sanity
from scripts.rewrite_webdataset_lowdim import (
    DEFAULT_WORKERS,
    _sample_clip_id,
    build_clip_index_from_processed_root,
    build_legacy_episode_index,
    build_legacy_episode_index_from_processed_root,
    source_contains_legacy_buildai_keys,
)


COMPARE_TOL = 1e-4
ROT6_OLD_TO_NEW = np.array([0, 2, 4, 1, 3, 5], dtype=np.int64)
_WORKER_CLIP_INDEX = None
_WORKER_LEGACY_EPISODES = None
_WORKER_FEATURE_CACHE_DIR = None
_WORKER_DEVICE = None
_WORKER_MANO_RIGHT = None
_WORKER_MANO_LEFT = None
_WORKER_MANO_DIR = None
_WORKER_EPISODE_CACHE = {}


def build_parser():
    parser = argparse.ArgumentParser(description="Audit old BuildAI interpolated WDS shards against current export logic")
    parser.add_argument("--source_shard_dir", required=True, help="Source directory containing old WDS shard tar files")
    parser.add_argument("--buildai_processed_root", required=True, help="BuildAI processed root containing stage outputs")
    parser.add_argument("--shard_start", type=int, default=0, help="Inclusive shard index in sorted shard order")
    parser.add_argument("--shard_end", type=int, default=None, help="Exclusive shard index in sorted shard order")
    parser.add_argument("--report_out", default=None, help="Optional JSON summary report path")
    parser.add_argument("--clip_report_out", default=None, help="Optional JSONL per-clip report path")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel shard workers")
    parser.add_argument("--mano_device", type=str, default="cuda:0", help="Device for MANO forward pass")
    parser.add_argument("--mano_gpus", type=str, default=None, help="Optional comma-separated GPU list for MANO workers")
    parser.add_argument("--mano_dir", type=str, default=None, help="Optional MANO model directory")
    parser.add_argument("--feature_cache_dir", type=str, default=None, help="Optional cache dir for current export features")
    parser.add_argument("--source_fps", type=float, default=5.0, help="Source label fps used during build")
    parser.add_argument("--target_fps", type=float, default=30.0, help="Target label fps used during build")
    parser.add_argument("--interpolate_labels", action=argparse.BooleanOptionalAction, default=True, help="Use interpolated label path")
    parser.add_argument("--legacy_buildai_input_dir", default=None, help="Optional old builder input_dir for resolving buildai_epXXXX keys")
    parser.add_argument("--legacy_episode_list", default=None, help="Optional old builder episode_list")
    parser.add_argument("--legacy_factory_range", default=None, help="Optional old builder factory range like 1-50")
    parser.add_argument("--legacy_episode_cache", default=None, help="Optional path to old builder _vla_episodes_cache.json")
    return parser


def _load_cam_space_chunks(seq_folder: Path) -> dict[int, list[dict]]:
    import joblib
    from lib.pipeline.stage_api import get_track_range

    start_idx, end_idx = get_track_range(seq_folder, fast=True)
    tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"
    frame_chunks_all = joblib.load(tracks_dir / "frame_chunks_all.npy")
    results: dict[int, list[dict]] = {0: [], 1: []}
    for hand_idx in (0, 1):
        for frame_chunk in frame_chunks_all.get(hand_idx, []):
            frame_chunk = np.asarray(frame_chunk, dtype=np.int64)
            if frame_chunk.size == 0:
                continue
            results[hand_idx].append({"frame_chunk": frame_chunk})
    return results


def _load_world_prediction(seq_folder: Path) -> dict | None:
    from lib.pipeline.exporters.webdataset_features import _load_world_space_prediction

    world_path = seq_folder / "world_space_res.pth"
    if not world_path.is_file():
        return None
    return _load_world_space_prediction({"episode_id": seq_folder.name}, str(world_path))


def _action_consistency_stats(lowdim_all: np.ndarray) -> dict:
    if lowdim_all.shape[0] <= 1:
        return {"wrist_max_abs_diff": 0.0, "hand_max_abs_diff": 0.0}
    wrist_diff = np.abs(lowdim_all[:-1, 48:66] - lowdim_all[1:, 0:18])
    hand_diff = np.abs(lowdim_all[:-1, 66:96] - lowdim_all[1:, 18:48])
    return {
        "wrist_max_abs_diff": float(wrist_diff.max()),
        "hand_max_abs_diff": float(hand_diff.max()),
    }

def _worker_init(
    device_specs,
    mano_dir,
    clip_index: dict[str, dict],
    legacy_episodes: dict[int, dict] | None,
    feature_cache_dir: str | None,
):
    global _WORKER_CLIP_INDEX, _WORKER_LEGACY_EPISODES, _WORKER_FEATURE_CACHE_DIR, _WORKER_DEVICE
    global _WORKER_MANO_RIGHT, _WORKER_MANO_LEFT, _WORKER_MANO_DIR, _WORKER_EPISODE_CACHE

    from multiprocessing import current_process
    import torch

    identity = current_process()._identity
    worker_idx = identity[0] - 1 if identity else 0
    device_str = device_specs[worker_idx % len(device_specs)]
    _WORKER_DEVICE = torch.device(device_str)
    _WORKER_CLIP_INDEX = clip_index
    _WORKER_LEGACY_EPISODES = legacy_episodes or {}
    _WORKER_FEATURE_CACHE_DIR = feature_cache_dir
    _WORKER_MANO_RIGHT = None
    _WORKER_MANO_LEFT = None
    _WORKER_MANO_DIR = mano_dir
    _WORKER_EPISODE_CACHE = {}


def _ensure_worker_models():
    global _WORKER_MANO_RIGHT, _WORKER_MANO_LEFT
    from lib.pipeline.exporters.webdataset_features import build_mano_models

    if _WORKER_MANO_RIGHT is not None and _WORKER_MANO_LEFT is not None:
        return
    _WORKER_MANO_RIGHT, _WORKER_MANO_LEFT = build_mano_models(_WORKER_DEVICE, mano_dir=_WORKER_MANO_DIR)
    _WORKER_MANO_RIGHT.eval()
    _WORKER_MANO_LEFT.eval()


def _resolve_clip_info(sample_key: str, clip_id: str) -> dict:
    clip_info = _WORKER_CLIP_INDEX.get(clip_id)
    if clip_info is not None:
        return clip_info

    if clip_id.startswith("buildai_ep"):
        from lib.pipeline.exporters.webdataset_rewriter import parse_episode_index

        episode_index = int(parse_episode_index(sample_key))
        clip_info = _WORKER_LEGACY_EPISODES.get(episode_index)
        if clip_info is None:
            raise KeyError(f"Legacy episode index {episode_index} not found for {sample_key}")
        return clip_info

    raise KeyError(f"Clip {clip_id} not found in processed root index")


def _get_episode_data(clip_info: dict, *, source_fps: float, target_fps: float, interpolate_labels: bool) -> dict:
    from lib.pipeline.exporters.manifest_vla import load_descriptor_episode_features

    clip_id = clip_info["clip_id"]
    cache_key = (clip_id, float(source_fps), float(target_fps), bool(interpolate_labels))
    if cache_key in _WORKER_EPISODE_CACHE:
        return _WORKER_EPISODE_CACHE[cache_key]

    _ensure_worker_models()
    episode_data = load_descriptor_episode_features(
        dict(clip_info),
        _WORKER_MANO_RIGHT,
        _WORKER_MANO_LEFT,
        _WORKER_DEVICE,
        _WORKER_FEATURE_CACHE_DIR,
        _WORKER_MANO_DIR,
        source_fps=source_fps,
        target_fps=target_fps,
        interpolate_labels=interpolate_labels,
    )
    if episode_data is None:
        raise RuntimeError(f"Failed to load current export features for clip {clip_id}")
    _WORKER_EPISODE_CACHE[cache_key] = episode_data
    return episode_data


def _load_world_summary(seq_folder: Path) -> dict:
    prediction = _load_world_prediction(seq_folder)
    if prediction is None:
        return {
            "source_frame_count": 0,
            "world_valid_left": 0,
            "world_valid_right": 0,
        }
    pred_valid = np.asarray(prediction["pred_valid"])
    return {
        "source_frame_count": int(prediction["pred_trans"].shape[1]),
        "world_valid_left": int((pred_valid[0] > 0.5).sum()),
        "world_valid_right": int((pred_valid[1] > 0.5).sum()),
    }


def _load_motion_summary(seq_folder: Path) -> dict:
    cam_chunks = _load_cam_space_chunks(seq_folder)
    return {
        "cam_space_left_chunks": int(len(cam_chunks[0])),
        "cam_space_right_chunks": int(len(cam_chunks[1])),
    }


def _load_slam_summary(seq_folder: Path) -> dict:
    slam_dir = seq_folder / "SLAM"
    slam_files = sorted(slam_dir.glob("hawor_slam_w_scale_*.npz")) if slam_dir.is_dir() else []
    if not slam_files:
        return {"slam_traj_count": 0}
    data = np.load(str(slam_files[0]), allow_pickle=True)
    traj = np.asarray(data["traj"])
    return {"slam_traj_count": int(traj.shape[0])}


def _permute_old_rot6_layout(lowdim_all: np.ndarray) -> np.ndarray:
    output = np.asarray(lowdim_all, dtype=np.float32).copy()
    for rot_slice in (slice(6, 12), slice(12, 18), slice(54, 60), slice(60, 66)):
        output[:, rot_slice] = output[:, rot_slice][:, ROT6_OLD_TO_NEW]
    return output


def _lowdim_diff_summary(a: np.ndarray, b: np.ndarray) -> dict:
    compare_count = min(int(a.shape[0]), int(b.shape[0]))
    if compare_count <= 0:
        return {
            "compare_frames": 0,
            "overall_max_abs_diff": 0.0,
            "state_max_abs_diff": 0.0,
            "action_max_abs_diff": 0.0,
            "extrinsic_max_abs_diff": 0.0,
            "intrinsic_max_abs_diff": 0.0,
        }
    diff = np.abs(np.asarray(a[:compare_count], dtype=np.float32) - np.asarray(b[:compare_count], dtype=np.float32))
    return {
        "compare_frames": compare_count,
        "overall_max_abs_diff": float(diff.max()),
        "state_max_abs_diff": float(diff[:, :48].max()),
        "action_max_abs_diff": float(diff[:, 48:96].max()),
        "extrinsic_max_abs_diff": float(diff[:, 96:112].max()),
        "intrinsic_max_abs_diff": float(diff[:, 112:116].max()),
    }


def _current_export_metrics(lowdim_all: np.ndarray) -> dict:
    invalid_rot6d = 0
    invalid_extrinsic = 0
    invalid_intrinsic = 0
    for lowdim in lowdim_all:
        sanity = validate_lowdim_numeric_sanity(lowdim)
        invalid_rot6d += int(bool(sanity["invalid_rot6d"]))
        invalid_extrinsic += int(bool(sanity["invalid_extrinsic"]))
        invalid_intrinsic += int(bool(sanity["invalid_intrinsic"]))
    action_stats = _action_consistency_stats(lowdim_all)
    return {
        "current_export_invalid_rot6d_frames": int(invalid_rot6d),
        "current_export_invalid_extrinsic_frames": int(invalid_extrinsic),
        "current_export_invalid_intrinsic_frames": int(invalid_intrinsic),
        "current_export_wrist_action_next_state_max_abs_diff": float(action_stats["wrist_max_abs_diff"]),
        "current_export_hand_action_next_state_max_abs_diff": float(action_stats["hand_max_abs_diff"]),
    }


def _classify_clip(metrics: dict) -> str:
    if metrics["has_infiller_hallucinated_hand"]:
        return "world_or_infiller_bug"
    if (
        metrics["current_export_invalid_rot6d_frames"] > 0
        or metrics["current_export_invalid_extrinsic_frames"] > 0
        or metrics["current_export_invalid_intrinsic_frames"] > 0
        or metrics["current_export_wrist_action_next_state_max_abs_diff"] > COMPARE_TOL
        or metrics["current_export_hand_action_next_state_max_abs_diff"] > COMPARE_TOL
    ):
        return "current_export_bug"
    raw = metrics.get("old_wds_vs_current_raw", {})
    permuted = metrics.get("old_wds_vs_current_rot6d_permuted", {})
    if not raw or raw.get("compare_frames", 0) <= 0:
        return "no_wds_compare"
    if raw["overall_max_abs_diff"] <= COMPARE_TOL:
        return "matches_current_export"
    if (
        permuted["overall_max_abs_diff"] <= COMPARE_TOL
        and permuted["state_max_abs_diff"] <= COMPARE_TOL
        and permuted["action_max_abs_diff"] <= COMPARE_TOL
        and permuted["extrinsic_max_abs_diff"] <= COMPARE_TOL
        and permuted["intrinsic_max_abs_diff"] <= COMPARE_TOL
    ):
        return "legacy_rot6d_layout_only"
    return "legacy_export_bug_beyond_rot6d"


def _analyze_clip_samples(samples: list[dict], *, source_fps: float, target_fps: float, interpolate_labels: bool) -> dict:
    ordered_samples = sorted(samples, key=lambda sample: parse_frame_index(sample["key"]))
    meta = None
    try:
        meta = json.loads(ordered_samples[0]["meta_bytes"].decode("utf-8"))
    except Exception:
        meta = None
    clip_id = _sample_clip_id(ordered_samples[0], meta)
    clip_info = _resolve_clip_info(ordered_samples[0]["key"], clip_id)
    seq_folder = Path(clip_info["seq_folder"]).resolve()

    current_episode = _get_episode_data(
        clip_info,
        source_fps=source_fps,
        target_fps=target_fps,
        interpolate_labels=interpolate_labels,
    )
    frame_indices = [int(parse_frame_index(sample["key"])) for sample in ordered_samples]
    current_lowdim = np.stack(
        [current_episode["lowdim_all"][frame_idx] for frame_idx in frame_indices],
        axis=0,
    ).astype(np.float32)
    old_lowdim = np.stack(
        [decode_lowdim(sample["lowdim_bytes"]) for sample in ordered_samples],
        axis=0,
    )
    old_lowdim_permuted = _permute_old_rot6_layout(old_lowdim)

    motion_summary = _load_motion_summary(seq_folder)
    world_summary = _load_world_summary(seq_folder)
    slam_summary = _load_slam_summary(seq_folder)
    current_metrics = _current_export_metrics(current_episode["lowdim_all"])

    metrics = {
        "clip_id": clip_info["clip_id"],
        "sample_clip_id": clip_id,
        "seq_folder": str(seq_folder),
        "old_wds_frames_in_shard": int(len(samples)),
        "old_wds_frame_index_min": int(min(frame_indices)),
        "old_wds_frame_index_max": int(max(frame_indices)),
        **motion_summary,
        **world_summary,
        **slam_summary,
        **current_metrics,
    }
    metrics["has_slam_frame_mismatch"] = bool(
        metrics["slam_traj_count"] > 0 and metrics["source_frame_count"] > 0 and metrics["slam_traj_count"] < metrics["source_frame_count"]
    )
    metrics["has_infiller_hallucinated_hand"] = bool(
        (metrics["cam_space_left_chunks"] == 0 and metrics["world_valid_left"] > 0)
        or (metrics["cam_space_right_chunks"] == 0 and metrics["world_valid_right"] > 0)
    )
    metrics["old_wds_vs_current_raw"] = _lowdim_diff_summary(old_lowdim, current_lowdim)
    metrics["old_wds_vs_current_rot6d_permuted"] = _lowdim_diff_summary(old_lowdim_permuted, current_lowdim)
    metrics["category"] = _classify_clip(metrics)
    return metrics


def _merge_clip_metrics(existing: dict | None, incoming: dict) -> dict:
    if existing is None:
        return incoming
    merged = dict(existing)
    merged["old_wds_frames_in_shard"] = int(existing["old_wds_frames_in_shard"] + incoming["old_wds_frames_in_shard"])
    merged["old_wds_frame_index_min"] = int(min(existing["old_wds_frame_index_min"], incoming["old_wds_frame_index_min"]))
    merged["old_wds_frame_index_max"] = int(max(existing["old_wds_frame_index_max"], incoming["old_wds_frame_index_max"]))
    for key in (
        "cam_space_left_chunks",
        "cam_space_right_chunks",
        "world_valid_left",
        "world_valid_right",
        "source_frame_count",
        "slam_traj_count",
        "current_export_invalid_rot6d_frames",
        "current_export_invalid_extrinsic_frames",
        "current_export_invalid_intrinsic_frames",
    ):
        merged[key] = int(max(existing[key], incoming[key]))
    for key in (
        "current_export_wrist_action_next_state_max_abs_diff",
        "current_export_hand_action_next_state_max_abs_diff",
    ):
        merged[key] = float(max(existing[key], incoming[key]))
    for key in ("has_slam_frame_mismatch", "has_infiller_hallucinated_hand"):
        merged[key] = bool(existing[key] or incoming[key])
    for compare_key in ("old_wds_vs_current_raw", "old_wds_vs_current_rot6d_permuted"):
        merged_compare = dict(existing[compare_key])
        incoming_compare = incoming[compare_key]
        merged_compare["compare_frames"] = int(existing[compare_key]["compare_frames"] + incoming_compare["compare_frames"])
        for diff_key in (
            "overall_max_abs_diff",
            "state_max_abs_diff",
            "action_max_abs_diff",
            "extrinsic_max_abs_diff",
            "intrinsic_max_abs_diff",
        ):
            merged_compare[diff_key] = float(max(existing[compare_key][diff_key], incoming_compare[diff_key]))
        merged[compare_key] = merged_compare
    merged["category"] = _classify_clip(merged)
    return merged


def _process_shard(args_tuple) -> dict:
    shard_path, source_fps, target_fps, interpolate_labels = args_tuple
    clip_metrics = {}
    current_clip_id = None
    current_samples = []
    for sample in iter_shard_samples(shard_path):
        validate_sample_record(sample)
        meta = None
        try:
            meta = json.loads(sample["meta_bytes"].decode("utf-8"))
        except Exception:
            meta = None
        clip_id = _sample_clip_id(sample, meta)
        if current_clip_id is None:
            current_clip_id = clip_id
        if clip_id != current_clip_id:
            metrics = _analyze_clip_samples(
                current_samples,
                source_fps=source_fps,
                target_fps=target_fps,
                interpolate_labels=interpolate_labels,
            )
            clip_metrics[metrics["clip_id"]] = _merge_clip_metrics(clip_metrics.get(metrics["clip_id"]), metrics)
            current_samples = []
            current_clip_id = clip_id
        current_samples.append(sample)

    if current_samples:
        metrics = _analyze_clip_samples(
            current_samples,
            source_fps=source_fps,
            target_fps=target_fps,
            interpolate_labels=interpolate_labels,
        )
        clip_metrics[metrics["clip_id"]] = _merge_clip_metrics(clip_metrics.get(metrics["clip_id"]), metrics)

    return {
        "shard_name": os.path.basename(shard_path),
        "clips": list(clip_metrics.values()),
    }


def _build_report(source_dir: Path, buildai_processed_root: Path, clip_reports: list[dict], feature_cache_dir: Path, shard_count: int) -> dict:
    category_counts = {}
    for item in clip_reports:
        category_counts[item["category"]] = int(category_counts.get(item["category"], 0) + 1)
    return {
        "source_shard_dir": str(source_dir.resolve()),
        "buildai_processed_root": str(buildai_processed_root.resolve()),
        "feature_cache_dir": str(feature_cache_dir.resolve()),
        "shards_scanned": int(shard_count),
        "clips_audited": int(len(clip_reports)),
        "category_counts": category_counts,
        "clips_with_infiller_hallucinated_hand": int(sum(1 for item in clip_reports if item["has_infiller_hallucinated_hand"])),
        "clips_with_current_export_bug": int(
            sum(
                1
                for item in clip_reports
                if (
                    item["current_export_invalid_rot6d_frames"] > 0
                    or item["current_export_invalid_extrinsic_frames"] > 0
                    or item["current_export_invalid_intrinsic_frames"] > 0
                    or item["current_export_wrist_action_next_state_max_abs_diff"] > COMPARE_TOL
                    or item["current_export_hand_action_next_state_max_abs_diff"] > COMPARE_TOL
                )
            )
        ),
    }


def main():
    args = build_parser().parse_args()
    import torch
    from lib.pipeline.exporters.webdataset_workers import normalize_mano_devices

    source_dir = Path(args.source_shard_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source shard dir not found: {source_dir}")
    buildai_processed_root = Path(args.buildai_processed_root).resolve()
    if not buildai_processed_root.is_dir():
        raise FileNotFoundError(f"BuildAI processed root not found: {buildai_processed_root}")

    feature_cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else source_dir / "_audit_episode_feature_cache"
    feature_cache_dir.mkdir(parents=True, exist_ok=True)

    clip_index = build_clip_index_from_processed_root(
        str(buildai_processed_root),
        factory_range=args.legacy_factory_range,
    )

    legacy_episode_cache = args.legacy_episode_cache
    if legacy_episode_cache:
        legacy_episode_cache = str(Path(legacy_episode_cache).resolve())
    else:
        default_cache = buildai_processed_root / "_vla_episodes_cache.json"
        if default_cache.is_file():
            legacy_episode_cache = str(default_cache)

    legacy_episodes = {}
    if args.legacy_buildai_input_dir or legacy_episode_cache:
        legacy_input_dir = str(Path(args.legacy_buildai_input_dir or buildai_processed_root).resolve())
        legacy_episodes = build_legacy_episode_index(
            legacy_input_dir,
            episode_list=args.legacy_episode_list,
            factory_range=args.legacy_factory_range,
            cache_file=legacy_episode_cache,
        )
    if not legacy_episodes:
        legacy_episodes = build_legacy_episode_index_from_processed_root(
            str(buildai_processed_root),
            factory_range=args.legacy_factory_range,
        )

    shard_paths = list(iter_shard_paths(str(source_dir)))
    if not shard_paths:
        raise RuntimeError(f"No shard tar files found in {source_dir}")
    if source_contains_legacy_buildai_keys(shard_paths) and not legacy_episodes:
        raise RuntimeError("Detected legacy buildai_ep keys but no legacy episode index could be built")

    total = len(shard_paths)
    start = int(args.shard_start)
    end = total if args.shard_end is None else min(int(args.shard_end), total)
    selected = shard_paths[start:end]

    mano_device_obj = torch.device(args.mano_device if torch.cuda.is_available() else "cpu")
    device_specs = normalize_mano_devices(str(mano_device_obj), args.mano_gpus if mano_device_obj.type == "cuda" else None)
    workers = min(int(args.workers), len(device_specs)) if mano_device_obj.type == "cuda" else int(args.workers)
    tasks = [(path, float(args.source_fps), float(args.target_fps), bool(args.interpolate_labels)) for path in selected]

    if workers <= 1:
        _worker_init(
            device_specs,
            args.mano_dir,
            clip_index,
            legacy_episodes,
            str(feature_cache_dir),
        )
        shard_results = [_process_shard(task) for task in tqdm(tasks, desc="Audit shards")]
    else:
        mp_context = get_context("spawn") if mano_device_obj.type == "cuda" else get_context()
        with mp_context.Pool(
            workers,
            initializer=_worker_init,
            initargs=(
                device_specs,
                args.mano_dir,
                clip_index,
                legacy_episodes,
                str(feature_cache_dir),
            ),
        ) as pool:
            shard_results = list(tqdm(pool.imap_unordered(_process_shard, tasks), total=len(tasks), desc="Audit shards"))

    merged = {}
    for shard_result in shard_results:
        for item in shard_result["clips"]:
            merged[item["clip_id"]] = _merge_clip_metrics(merged.get(item["clip_id"]), item)

    clip_reports = sorted(merged.values(), key=lambda item: item["clip_id"])
    report = _build_report(source_dir, buildai_processed_root, clip_reports, feature_cache_dir, len(selected))

    if args.clip_report_out:
        clip_report_out = Path(args.clip_report_out).expanduser().resolve()
        clip_report_out.parent.mkdir(parents=True, exist_ok=True)
        with clip_report_out.open("w", encoding="utf-8") as handle:
            for item in clip_reports:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    if args.report_out:
        report_out = Path(args.report_out).expanduser().resolve()
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
