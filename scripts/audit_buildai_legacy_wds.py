#!/usr/bin/env python3
"""Audit old BuildAI interpolated WebDataset shards against current export logic."""

from __future__ import annotations

import argparse
import json
import os
import re
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

from lib.pipeline.datasets.descriptors import ClipDescriptor
from lib.pipeline.exporters.webdataset_rewriter import iter_shard_paths, iter_shard_samples, validate_sample_record
from lib.pipeline.quality_metrics import decode_lowdim, parse_frame_index
from scripts.rewrite_webdataset_lowdim import (
    DEFAULT_WORKERS,
    _sample_clip_id,
    build_legacy_episode_index,
    iter_buildai_seq_folders,
)


COMPARE_TOL = 1e-4
ROT6_OLD_TO_NEW = np.array([0, 2, 4, 1, 3, 5], dtype=np.int64)
BUILDAI_CLIP_RE = re.compile(r"^factory(\d{3})_worker(\d{3})_")
BUILDAI_SHORT_CLIP_RE = re.compile(r"^f(\d{3})_w(\d{3})_")
ROT6D_UNIT_NORM_TOL = 0.2
ROT6D_ORTHOGONALITY_TOL = 0.2
ROT6D_MIN_CROSS_NORM = 0.5
EXTRINSIC_BOTTOM_ROW_TOL = 1e-3
EXTRINSIC_ROTATION_ORTHO_FROB_TOL = 0.2
EXTRINSIC_ROTATION_DET_TOL = 0.2
_WORKER_CLIP_INDEX = None
_WORKER_LEGACY_EPISODES = None
_WORKER_FEATURE_CACHE_DIR = None
_WORKER_DEVICE = None
_WORKER_MANO_RIGHT = None
_WORKER_MANO_LEFT = None
_WORKER_MANO_DIR = None
_WORKER_EPISODE_CACHE = {}

try:
    from lib.pipeline.quality_metrics import validate_lowdim_numeric_sanity  # type: ignore
except ImportError:
    def _rot6d_is_sane(rot6d: np.ndarray) -> bool:
        array = np.asarray(rot6d, dtype=np.float32).reshape(-1)
        if array.shape != (6,) or not np.isfinite(array).all():
            return False
        col_a = array[:3]
        col_b = array[3:]
        norm_a = float(np.linalg.norm(col_a))
        norm_b = float(np.linalg.norm(col_b))
        if norm_a <= 1e-8 or norm_b <= 1e-8:
            return False
        if abs(norm_a - 1.0) > ROT6D_UNIT_NORM_TOL or abs(norm_b - 1.0) > ROT6D_UNIT_NORM_TOL:
            return False
        unit_a = col_a / norm_a
        unit_b = col_b / norm_b
        if abs(float(np.dot(unit_a, unit_b))) > ROT6D_ORTHOGONALITY_TOL:
            return False
        if float(np.linalg.norm(np.cross(unit_a, unit_b))) < ROT6D_MIN_CROSS_NORM:
            return False
        return True

    def _extrinsic_is_sane(extrinsic: np.ndarray) -> bool:
        matrix = np.asarray(extrinsic, dtype=np.float32).reshape(4, 4)
        if not np.isfinite(matrix).all():
            return False
        if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=EXTRINSIC_BOTTOM_ROW_TOL):
            return False
        rotation = matrix[:3, :3].astype(np.float64)
        det = float(np.linalg.det(rotation))
        if not np.isfinite(det) or abs(det - 1.0) > EXTRINSIC_ROTATION_DET_TOL:
            return False
        ortho_err = float(np.linalg.norm(rotation.T @ rotation - np.eye(3, dtype=np.float64), ord="fro"))
        if ortho_err > EXTRINSIC_ROTATION_ORTHO_FROB_TOL:
            return False
        return True

    def _intrinsic_is_sane(intrinsic: np.ndarray) -> bool:
        array = np.asarray(intrinsic, dtype=np.float32).reshape(-1)
        if array.shape != (4,) or not np.isfinite(array).all():
            return False
        return float(array[0]) > 0.0 and float(array[1]) > 0.0

    def validate_lowdim_numeric_sanity(lowdim: np.ndarray) -> dict:
        array = np.asarray(lowdim, dtype=np.float32).reshape(-1)
        invalid_rot6d = any(
            not _rot6d_is_sane(array[rot_slice])
            for rot_slice in (
                slice(6, 12),
                slice(12, 18),
                slice(54, 60),
                slice(60, 66),
            )
        )
        invalid_extrinsic = not _extrinsic_is_sane(array[96:112].reshape(4, 4))
        invalid_intrinsic = not _intrinsic_is_sane(array[112:116])
        issues = []
        if invalid_rot6d:
            issues.append("invalid_rot6d")
        if invalid_extrinsic:
            issues.append("invalid_extrinsic")
        if invalid_intrinsic:
            issues.append("invalid_intrinsic")
        return {
            "valid": not issues,
            "invalid_rot6d": bool(invalid_rot6d),
            "invalid_extrinsic": bool(invalid_extrinsic),
            "invalid_intrinsic": bool(invalid_intrinsic),
            "issues": issues,
        }


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


def _build_clip_index_with_progress(processed_root: Path, factory_range) -> dict[str, dict]:
    print(f"[audit] scanning processed root for clip index: {processed_root}", flush=True)
    clip_index = {}
    for idx, seq_folder in enumerate(iter_buildai_seq_folders(str(processed_root), factory_range=factory_range), start=1):
        clip_id = seq_folder.name
        clip_index[clip_id] = {
            "clip_id": clip_id,
            "episode_id": clip_id,
            "seq_folder": str(seq_folder),
            "source_id": "buildai",
            "split": "unknown",
            "descriptor": _make_legacy_buildai_descriptor(seq_folder, clip_id),
        }
        if idx <= 5 or idx % 2000 == 0:
            print(f"[audit] indexed clips={idx}", flush=True)
    if not clip_index:
        raise RuntimeError(f"No BuildAI seq_folder with world_space_res.pth found under {processed_root}")
    print(f"[audit] clip index ready: clips={len(clip_index)}", flush=True)
    return clip_index


def _make_legacy_buildai_descriptor(seq_folder: Path, clip_id: str) -> ClipDescriptor:
    frame_dir = seq_folder / "extracted_images"
    frame_names: list[str] = []
    if frame_dir.is_dir():
        jpgs = sorted(path.name for path in frame_dir.glob("*.jpg"))
        pngs = sorted(path.name for path in frame_dir.glob("*.png"))
        frame_names = jpgs if jpgs else pngs
    root_dir = str(seq_folder.parent.resolve()) if seq_folder.parent.exists() else str(seq_folder.resolve())
    return ClipDescriptor.from_image_sequence(
        clip_id=clip_id,
        clip_name=clip_id,
        root_dir=root_dir,
        seq_folder=str(seq_folder.resolve()),
        frame_dir=str(frame_dir.resolve()) if frame_dir.is_dir() else str(seq_folder.resolve()),
        frame_names=frame_names,
        media_path=None,
        fps=30.0,
        extra={},
    )


def _resolve_direct_buildai_clip_info(processed_root: Path, clip_id: str) -> dict | None:
    match = BUILDAI_CLIP_RE.match(clip_id)
    short_match = BUILDAI_SHORT_CLIP_RE.match(clip_id)
    if match is not None:
        factory_id = int(match.group(1))
        worker_id = int(match.group(2))
    elif short_match is not None:
        factory_id = int(short_match.group(1))
        worker_id = int(short_match.group(2))
    else:
        return None
    candidates = [
        processed_root / f"factory_{factory_id:03d}" / f"worker_{worker_id:03d}" / "processed" / clip_id,
        processed_root / f"factory{factory_id:03d}" / "outputs" / clip_id,
        processed_root / f"factory_{factory_id:03d}" / "outputs" / clip_id,
    ]
    seq_folder = next((path for path in candidates if path.is_dir() and (path / "world_space_res.pth").is_file()), None)
    if seq_folder is None:
        return None
    return {
        "clip_id": clip_id,
        "episode_id": clip_id,
        "seq_folder": str(seq_folder.resolve()),
        "source_id": "buildai",
        "split": "unknown",
        "descriptor": _make_legacy_buildai_descriptor(seq_folder, clip_id),
    }


def _collect_selected_clip_ids(shard_paths: list[str]) -> tuple[set[str], bool]:
    print(f"[audit] scanning selected shards for clip ids: shards={len(shard_paths)}", flush=True)
    clip_ids: set[str] = set()
    has_legacy_keys = False
    sample_count = 0
    for shard_idx, shard_path in enumerate(shard_paths, start=1):
        for sample in iter_shard_samples(shard_path):
            meta = None
            try:
                meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            except Exception:
                meta = None
            clip_id = _sample_clip_id(sample, meta)
            clip_ids.add(clip_id)
            has_legacy_keys = has_legacy_keys or clip_id.startswith("buildai_ep")
            sample_count += 1
        if shard_idx <= 4 or shard_idx % 8 == 0 or shard_idx == len(shard_paths):
            print(
                f"[audit] selected-shard scan {shard_idx}/{len(shard_paths)} clips={len(clip_ids)} samples={sample_count}",
                flush=True,
            )
    print(
        f"[audit] selected clip-id scan ready: unique_clips={len(clip_ids)} legacy_keys={has_legacy_keys}",
        flush=True,
    )
    return clip_ids, has_legacy_keys


def _build_subset_clip_index(processed_root: Path, clip_ids: set[str]) -> tuple[dict[str, dict], list[str]]:
    print(f"[audit] resolving direct clip ids without full processed-root scan: clips={len(clip_ids)}", flush=True)
    clip_index = {}
    unresolved = []
    ordered_clip_ids = sorted(clip_ids)
    for idx, clip_id in enumerate(ordered_clip_ids, start=1):
        clip_info = _resolve_direct_buildai_clip_info(processed_root, clip_id)
        if clip_info is None:
            unresolved.append(clip_id)
        else:
            clip_index[clip_id] = clip_info
        if idx <= 5 or idx % 2000 == 0 or idx == len(ordered_clip_ids):
            print(
                f"[audit] direct resolve progress {idx}/{len(ordered_clip_ids)} resolved={len(clip_index)} unresolved={len(unresolved)}",
                flush=True,
            )
    return clip_index, unresolved


def _build_legacy_episode_index_from_clip_index(clip_index: dict[str, dict]) -> dict[int, dict]:
    legacy_episodes = {}
    for episode_index, clip_info in enumerate(clip_index.values()):
        record = dict(clip_info)
        record["legacy_episode_index"] = int(episode_index)
        legacy_episodes[episode_index] = record
    return legacy_episodes

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


def _get_episode_data(
    clip_info: dict,
    *,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
    requested_frame_count: int | None = None,
) -> dict:
    from lib.pipeline.exporters.manifest_vla import load_descriptor_episode_features

    clip_id = clip_info["clip_id"]
    normalized_requested_frame_count = None if requested_frame_count is None else int(requested_frame_count)
    cache_key = (
        clip_id,
        float(source_fps),
        float(target_fps),
        bool(interpolate_labels),
        normalized_requested_frame_count,
    )
    if cache_key in _WORKER_EPISODE_CACHE:
        return _WORKER_EPISODE_CACHE[cache_key]

    _ensure_worker_models()
    feature_request = dict(clip_info)
    if normalized_requested_frame_count is not None:
        feature_request["num_valid_frames"] = int(normalized_requested_frame_count)
    episode_data = load_descriptor_episode_features(
        feature_request,
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
    frame_indices = [int(parse_frame_index(sample["key"])) for sample in ordered_samples]
    requested_frame_count = int(max(frame_indices) + 1) if frame_indices else None

    current_episode = _get_episode_data(
        clip_info,
        source_fps=source_fps,
        target_fps=target_fps,
        interpolate_labels=interpolate_labels,
        requested_frame_count=requested_frame_count,
    )
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

    legacy_episode_cache = args.legacy_episode_cache
    if legacy_episode_cache:
        legacy_episode_cache = str(Path(legacy_episode_cache).resolve())
    else:
        default_cache = buildai_processed_root / "_vla_episodes_cache.json"
        if default_cache.is_file():
            legacy_episode_cache = str(default_cache)

    shard_paths = list(iter_shard_paths(str(source_dir)))
    if not shard_paths:
        raise RuntimeError(f"No shard tar files found in {source_dir}")

    total = len(shard_paths)
    start = int(args.shard_start)
    end = total if args.shard_end is None else min(int(args.shard_end), total)
    selected = shard_paths[start:end]
    print(
        f"[audit] shard selection: total={total} range=[{start}, {end}) selected={len(selected)}",
        flush=True,
    )
    selected_clip_ids, selected_has_legacy_keys = _collect_selected_clip_ids(selected)

    legacy_episodes = {}
    if not selected_has_legacy_keys:
        clip_index, unresolved_clip_ids = _build_subset_clip_index(buildai_processed_root, selected_clip_ids)
        if unresolved_clip_ids:
            preview = unresolved_clip_ids[:8]
            print(
                f"[audit] direct clip resolution missed {len(unresolved_clip_ids)} clips; fallback to full processed-root scan. preview={preview}",
                flush=True,
            )
            clip_index = _build_clip_index_with_progress(
                buildai_processed_root,
                factory_range=args.legacy_factory_range,
            )
            print("[audit] reusing full clip index order for legacy episode mapping", flush=True)
            legacy_episodes = _build_legacy_episode_index_from_clip_index(clip_index)
        else:
            print(f"[audit] subset clip index ready: clips={len(clip_index)}", flush=True)
    else:
        print("[audit] selected shards contain legacy buildai_ep keys; need processed-root/legacy mapping", flush=True)
        clip_index = _build_clip_index_with_progress(
            buildai_processed_root,
            factory_range=args.legacy_factory_range,
        )
        if args.legacy_buildai_input_dir or legacy_episode_cache:
            print("[audit] building legacy episode mapping from explicit legacy source/cache", flush=True)
            legacy_input_dir = str(Path(args.legacy_buildai_input_dir or buildai_processed_root).resolve())
            legacy_episodes = build_legacy_episode_index(
                legacy_input_dir,
                episode_list=args.legacy_episode_list,
                factory_range=args.legacy_factory_range,
                cache_file=legacy_episode_cache,
            )
        if not legacy_episodes:
            print("[audit] reusing full clip index order for legacy episode mapping", flush=True)
            legacy_episodes = _build_legacy_episode_index_from_clip_index(clip_index)
    if legacy_episodes:
        print(f"[audit] legacy episode mapping ready: episodes={len(legacy_episodes)}", flush=True)
    else:
        print("[audit] no legacy episode mapping needed for selected shards", flush=True)

    if selected_has_legacy_keys and not legacy_episodes:
        raise RuntimeError("Detected legacy buildai_ep keys but no legacy episode index could be built")

    mano_device_obj = torch.device(args.mano_device if torch.cuda.is_available() else "cpu")
    device_specs = normalize_mano_devices(str(mano_device_obj), args.mano_gpus if mano_device_obj.type == "cuda" else None)
    workers = min(int(args.workers), len(device_specs)) if mano_device_obj.type == "cuda" else int(args.workers)
    print(
        f"[audit] runtime: device={mano_device_obj} workers={workers} feature_cache_dir={feature_cache_dir}",
        flush=True,
    )
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
