#!/usr/bin/env python3
"""Diagnose one clip across motion cam-space, world-space, and optional final WDS export."""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
from pathlib import Path

import cv2
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hawor.utils.process import run_mano, run_mano_left
from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
from lib.pipeline.exporters.manifest_build.resample import (
    build_source_target_times,
    resample_episode_features,
)
from lib.pipeline.exporters.mano_codec import rot6d_to_rotmat
from lib.pipeline.exporters.webdataset_features import (
    _build_lowdim_features,
    _compute_joint_states,
    _compute_presence_per_frame,
    _load_episode_camera_features,
    _load_world_space_prediction,
    build_mano_models,
)
from lib.pipeline.quality_metrics import decode_lowdim
from lib.pipeline.stage_api import get_track_range
from lib.pipeline.wds_sanity import LOWDIM_DIMENSION_NAMES


FRAME_INDEX_RE = re.compile(r"_f(\d+)$")
MANO_JOINT_TREE = [
    [(0, 1), (1, 2), (2, 3), (3, 4)],
    [(0, 5), (5, 6), (6, 7), (7, 8)],
    [(0, 9), (9, 10), (10, 11), (11, 12)],
    [(0, 13), (13, 14), (14, 15), (15, 16)],
    [(0, 17), (17, 18), (18, 19), (19, 20)],
]
HAND_COLORS = {
    0: (80, 200, 255),
    1: (120, 255, 120),
}
WRIST_STATE_SLICE = slice(0, 18)
HAND_STATE_SLICE = slice(18, 48)
WRIST_ACTION_SLICE = slice(48, 66)
HAND_ACTION_SLICE = slice(66, 96)
EXTRINSIC_SLICE = slice(96, 112)
INTRINSIC_SLICE = slice(112, 116)
WDS_MEMBER_SUFFIXES = {
    ".image.jpg": "image_bytes",
    ".lowdim.npy": "lowdim_bytes",
    ".meta.json": "meta_bytes",
    ".mano.npy": "mano_bytes",
}
ROT6D_UNIT_NORM_TOL = 0.2
ROT6D_ORTHOGONALITY_TOL = 0.2
ROT6D_MIN_CROSS_NORM = 0.5
EXTRINSIC_BOTTOM_ROW_TOL = 1e-3
EXTRINSIC_ROTATION_ORTHO_FROB_TOL = 0.2
EXTRINSIC_ROTATION_DET_TOL = 0.2

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
        invalid_extrinsic = not _extrinsic_is_sane(array[EXTRINSIC_SLICE].reshape(4, 4))
        invalid_intrinsic = not _intrinsic_is_sane(array[INTRINSIC_SLICE])
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose one clip across motion/world/final-WDS stages")
    parser.add_argument(
        "--seq-folder",
        "--seq_folder",
        dest="seq_folders",
        action="append",
        required=True,
        help="Processed clip output folder, e.g. factory*/outputs/<clip_id>. Repeat for multiple clips.",
    )
    parser.add_argument("--frame-dir", default=None, help="Optional extracted_images directory override")
    parser.add_argument("--mano-dir", default=None, help="Optional MANO model directory override")
    parser.add_argument("--device", default="cpu", help="Torch device for MANO forward, e.g. cpu or cuda:0")
    parser.add_argument("--chunk-limit", type=int, default=8, help="How many motion chunks to print per hand")
    parser.add_argument("--render-dir", default=None, help="Optional directory for diagnostic videos")
    parser.add_argument("--render-max-frames", type=int, default=240, help="Maximum number of frames to render per stage")
    parser.add_argument("--render-fps", type=int, default=15, help="FPS for diagnostic videos")
    parser.add_argument("--source-fps", type=float, default=5.0, help="Source label fps used during build")
    parser.add_argument("--target-fps", type=float, default=30.0, help="Target label fps used during build")
    parser.add_argument("--interpolate-labels", action=argparse.BooleanOptionalAction, default=True, help="Use the current build-time label interpolation path")
    parser.add_argument("--wds-shard", default=None, help="Optional final WDS shard path for comparing exported lowdim/image samples")
    parser.add_argument("--wds-clip-id", default=None, help="Optional clip_id override for WDS lookup; default uses seq-folder name")
    parser.add_argument("--report-out", "--report_out", default=None, help="Optional JSON report output path")
    return parser


def _hand_name(hand_idx: int) -> str:
    return "left" if int(hand_idx) == 0 else "right"


def _format_range(array: np.ndarray) -> str:
    arr = np.asarray(array, dtype=np.float64)
    if arr.size == 0:
        return "empty"
    return f"[{arr.min():.6g}, {arr.max():.6g}]"


def _format_norm_range(array: np.ndarray, axis: int = -1) -> str:
    arr = np.asarray(array, dtype=np.float64)
    if arr.size == 0:
        return "empty"
    norms = np.linalg.norm(arr, axis=axis)
    return _format_range(norms)


def _parse_frame_index_from_key(sample_key: str) -> int:
    match = FRAME_INDEX_RE.search(sample_key)
    if match is None:
        raise ValueError(f"Failed to parse frame index from sample key: {sample_key}")
    return int(match.group(1))


def _resolve_render_dir(seq_folder: Path, render_dir: str | None) -> Path:
    if render_dir:
        return Path(render_dir).expanduser().resolve()
    return (seq_folder / "diagnostics").resolve()


def _resolve_frame_dir(seq_folder: Path, frame_dir: str | None) -> Path | None:
    candidates: list[Path] = []
    if frame_dir:
        candidates.append(Path(frame_dir).expanduser().resolve())

    candidates.append((seq_folder / "extracted_images").resolve())

    if seq_folder.parent.name == "outputs":
        factory_dir = seq_folder.parent.parent
        candidates.extend(
            sorted(factory_dir.glob(f"worker_*/processed/{seq_folder.name}/extracted_images"))
        )

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _resolve_image_paths(seq_folder: Path, frame_dir: str | None) -> list[Path]:
    resolved_frame_dir = _resolve_frame_dir(seq_folder, frame_dir)
    if resolved_frame_dir is None:
        raise FileNotFoundError(f"Could not resolve extracted_images for {seq_folder}")
    image_paths = sorted(resolved_frame_dir.glob("*.jpg"))
    if not image_paths:
        image_paths = sorted(resolved_frame_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg/.png frames found under {resolved_frame_dir}")
    return image_paths


def _resolve_focal(seq_folder: Path) -> float:
    path = seq_folder / "est_focal.txt"
    if path.is_file():
        try:
            return float(path.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    return 600.0


def _load_cam_space_chunks(seq_folder: Path) -> dict[int, list[dict]]:
    start_idx, end_idx = get_track_range(seq_folder, fast=True)
    tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"
    frame_chunks_all = joblib.load(tracks_dir / "frame_chunks_all.npy")
    results: dict[int, list[dict]] = {0: [], 1: []}
    for hand_idx in (0, 1):
        for frame_chunk in frame_chunks_all.get(hand_idx, []):
            frame_chunk = np.asarray(frame_chunk, dtype=np.int64)
            if frame_chunk.size == 0:
                continue
            key = f"{int(frame_chunk[0])}_{int(frame_chunk[-1])}"
            path = seq_folder / "cam_space" / str(hand_idx) / f"{key}.json"
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            results[hand_idx].append(
                {
                    "key": key,
                    "frame_chunk": frame_chunk,
                    "init_trans": np.asarray(payload["init_trans"], dtype=np.float32),
                    "init_root_orient": np.asarray(payload["init_root_orient"], dtype=np.float32),
                    "init_hand_pose": np.asarray(payload["init_hand_pose"], dtype=np.float32),
                    "init_betas": np.asarray(payload["init_betas"], dtype=np.float32),
                }
            )
    return results


def _summarize_cam_space(cam_chunks: dict[int, list[dict]], chunk_limit: int) -> dict:
    print("\n=== Motion cam_space summary ===")
    summary = {}
    for hand_idx in (0, 1):
        hand_name = _hand_name(hand_idx)
        chunks = cam_chunks[hand_idx]
        print(f"[{hand_name}] chunks={len(chunks)}")
        hand_summary = {
            "chunks": int(len(chunks)),
        }
        if not chunks:
            summary[hand_name] = hand_summary
            continue

        all_trans = np.concatenate([chunk["init_trans"].reshape(-1, 3) for chunk in chunks], axis=0)
        all_root = np.concatenate([chunk["init_root_orient"].reshape(-1, 3, 3) for chunk in chunks], axis=0)
        all_hand_pose = np.concatenate([chunk["init_hand_pose"].reshape(-1, 15, 3, 3) for chunk in chunks], axis=0)
        root_dets = np.linalg.det(all_root.astype(np.float64))
        root_ortho_err = np.linalg.norm(
            np.matmul(np.transpose(all_root, (0, 2, 1)), all_root) - np.eye(3, dtype=np.float32),
            axis=(1, 2),
        )
        pose_dets = np.linalg.det(all_hand_pose.astype(np.float64).reshape(-1, 3, 3))
        pose_ortho_err = np.linalg.norm(
            np.matmul(
                np.transpose(all_hand_pose.reshape(-1, 3, 3), (0, 2, 1)),
                all_hand_pose.reshape(-1, 3, 3),
            ) - np.eye(3, dtype=np.float32),
            axis=(1, 2),
        )

        print(f"  init_trans xyz range={_format_range(all_trans)}")
        print(f"  init_trans norm range={_format_norm_range(all_trans)}")
        print(f"  root_rot det range={_format_range(root_dets)}")
        print(f"  root_rot ortho_err range={_format_range(root_ortho_err)}")
        print(f"  hand_pose det range={_format_range(pose_dets)}")
        print(f"  hand_pose ortho_err range={_format_range(pose_ortho_err)}")

        chunk_examples = []
        for example in chunks[: max(0, int(chunk_limit))]:
            example_trans = example["init_trans"].reshape(-1, 3)
            trans_norm_range = _format_norm_range(example_trans)
            print(f"  chunk={example['key']} frames={len(example['frame_chunk'])} trans_norm={trans_norm_range}")
            chunk_examples.append(
                {
                    "key": example["key"],
                    "frames": int(len(example["frame_chunk"])),
                    "trans_norm_range": trans_norm_range,
                }
            )

        hand_summary.update(
            {
                "init_trans_range": _format_range(all_trans),
                "init_trans_norm_range": _format_norm_range(all_trans),
                "root_rot_det_range": _format_range(root_dets),
                "root_rot_ortho_err_range": _format_range(root_ortho_err),
                "hand_pose_det_range": _format_range(pose_dets),
                "hand_pose_ortho_err_range": _format_range(pose_ortho_err),
                "chunk_examples": chunk_examples,
            }
        )
        summary[hand_name] = hand_summary
    return summary


def _project_cam_points(points_cam: np.ndarray, focal: float, center_xy: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_cam, dtype=np.float32)
    z = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (z > 1e-6)
    uv = np.zeros((pts.shape[0], 2), dtype=np.float32)
    uv[valid, 0] = pts[valid, 0] / z[valid] * float(focal) + float(center_xy[0])
    uv[valid, 1] = pts[valid, 1] / z[valid] * float(focal) + float(center_xy[1])
    return uv, valid


def _project_world_points(points_world: np.ndarray, w2c: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float32)
    extrinsic = np.asarray(w2c, dtype=np.float32).reshape(4, 4)
    homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (extrinsic @ homo.T).T[:, :3]
    fx, fy, cx, cy = np.asarray(intrinsic, dtype=np.float32).reshape(4)
    z = cam[:, 2]
    valid = np.isfinite(cam).all(axis=1) & (z > 1e-6)
    uv = np.zeros((points.shape[0], 2), dtype=np.float32)
    uv[valid, 0] = cam[valid, 0] / z[valid] * fx + cx
    uv[valid, 1] = cam[valid, 1] / z[valid] * fy + cy
    return uv, valid


def _clip_uv_mask(uv: np.ndarray, valid: np.ndarray, shape) -> np.ndarray:
    h, w = shape[:2]
    return valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)


def _draw_projected_axes_cam(
    image: np.ndarray,
    origin_cam: np.ndarray,
    rotmat_cam: np.ndarray,
    focal: float,
    center_xy: tuple[float, float],
    axis_length: float = 0.04,
) -> None:
    origin_and_axes = np.stack(
        [
            origin_cam,
            origin_cam + rotmat_cam[:, 0] * axis_length,
            origin_cam + rotmat_cam[:, 1] * axis_length,
            origin_cam + rotmat_cam[:, 2] * axis_length,
        ],
        axis=0,
    ).astype(np.float32)
    uv, valid = _project_cam_points(origin_and_axes, focal, center_xy)
    mask = _clip_uv_mask(uv, valid, image.shape)
    if not mask[0]:
        return
    origin = tuple(uv[0].astype(np.int32))
    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for axis_idx in range(3):
        if mask[axis_idx + 1]:
            cv2.arrowedLine(image, origin, tuple(uv[axis_idx + 1].astype(np.int32)), axis_colors[axis_idx], 2, tipLength=0.2)


def _draw_projected_axes_world(
    image: np.ndarray,
    origin_world: np.ndarray,
    rotmat_world: np.ndarray,
    w2c: np.ndarray,
    intrinsic: np.ndarray,
    axis_length: float = 0.04,
) -> None:
    origin_and_axes = np.stack(
        [
            origin_world,
            origin_world + rotmat_world[:, 0] * axis_length,
            origin_world + rotmat_world[:, 1] * axis_length,
            origin_world + rotmat_world[:, 2] * axis_length,
        ],
        axis=0,
    ).astype(np.float32)
    uv, valid = _project_world_points(origin_and_axes, w2c, intrinsic)
    mask = _clip_uv_mask(uv, valid, image.shape)
    if not mask[0]:
        return
    origin = tuple(uv[0].astype(np.int32))
    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for axis_idx in range(3):
        if mask[axis_idx + 1]:
            cv2.arrowedLine(image, origin, tuple(uv[axis_idx + 1].astype(np.int32)), axis_colors[axis_idx], 2, tipLength=0.2)


def _draw_cam_hand_overlay(
    image: np.ndarray,
    joints_cam: np.ndarray,
    root_rotmat: np.ndarray,
    hand_idx: int,
    focal: float,
    center_xy: tuple[float, float],
) -> int:
    color = HAND_COLORS[int(hand_idx)]
    uv, valid = _project_cam_points(joints_cam, focal, center_xy)
    mask = _clip_uv_mask(uv, valid, image.shape)
    for chain in MANO_JOINT_TREE:
        for j1, j2 in chain:
            if mask[j1] and mask[j2]:
                cv2.line(image, tuple(uv[j1].astype(np.int32)), tuple(uv[j2].astype(np.int32)), color, 2)
    for joint_idx, pt in enumerate(uv.astype(np.int32)):
        if mask[joint_idx]:
            cv2.circle(image, tuple(pt), 5 if joint_idx == 0 else 3, color, -1)
    if mask[0]:
        _draw_projected_axes_cam(image, joints_cam[0], root_rotmat, focal, center_xy)
    return int(mask.sum())


def _draw_world_hand_overlay(
    image: np.ndarray,
    joints_world: np.ndarray,
    root_rotmat: np.ndarray,
    hand_idx: int,
    w2c: np.ndarray,
    intrinsic: np.ndarray,
) -> int:
    color = HAND_COLORS[int(hand_idx)]
    uv, valid = _project_world_points(joints_world, w2c, intrinsic)
    mask = _clip_uv_mask(uv, valid, image.shape)
    for chain in MANO_JOINT_TREE:
        for j1, j2 in chain:
            if mask[j1] and mask[j2]:
                cv2.line(image, tuple(uv[j1].astype(np.int32)), tuple(uv[j2].astype(np.int32)), color, 2)
    for joint_idx, pt in enumerate(uv.astype(np.int32)):
        if mask[joint_idx]:
            cv2.circle(image, tuple(pt), 5 if joint_idx == 0 else 3, color, -1)
    if mask[0]:
        _draw_projected_axes_world(image, joints_world[0], root_rotmat, w2c, intrinsic)
    return int(mask.sum())


def _annotate_frame(image: np.ndarray, lines: list[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28


def _render_motion_video(
    seq_folder: Path,
    cam_chunks: dict[int, list[dict]],
    *,
    image_paths: list[Path],
    mano_dir: str | None,
    device: str,
    render_out: Path,
    render_max_frames: int,
    render_fps: int,
) -> dict:
    import torch

    render_frame_count = min(len(image_paths), max(1, int(render_max_frames)))
    first_frame = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read frame: {image_paths[0]}")
    height, width = first_frame.shape[:2]
    focal = _resolve_focal(seq_folder)
    center_xy = (width / 2.0, height / 2.0)
    torch_device = torch.device(device)
    mano_right, mano_left = build_mano_models(torch_device, mano_dir=mano_dir)

    per_frame: dict[int, dict[int, dict]] = {}
    for hand_idx in (0, 1):
        for chunk in cam_chunks[hand_idx]:
            frame_chunk = chunk["frame_chunk"]
            if frame_chunk.size == 0:
                continue
            with torch.inference_mode():
                init_trans = torch.from_numpy(chunk["init_trans"]).float().to(torch_device)
                init_root_orient = torch.from_numpy(chunk["init_root_orient"]).float().to(torch_device)
                init_hand_pose = torch.from_numpy(chunk["init_hand_pose"]).float().to(torch_device)
                init_betas = torch.from_numpy(chunk["init_betas"]).float().to(torch_device)
                root_aa = rotation_matrix_to_angle_axis(init_root_orient)
                hand_aa = rotation_matrix_to_angle_axis(init_hand_pose)
                if int(hand_idx) == 0:
                    outputs = run_mano_left(init_trans, root_aa, hand_aa, betas=init_betas, mano_model=mano_left)
                else:
                    outputs = run_mano(init_trans, root_aa, hand_aa, betas=init_betas, mano_model=mano_right)
            joints = outputs["joints"][0].detach().cpu().numpy().astype(np.float32)
            root_rotmats = init_root_orient[0].detach().cpu().numpy().astype(np.float32)
            for local_idx, frame_idx in enumerate(frame_chunk.tolist()):
                if int(frame_idx) >= render_frame_count:
                    continue
                per_frame.setdefault(int(frame_idx), {})[int(hand_idx)] = {
                    "joints": joints[local_idx],
                    "root_rotmat": root_rotmats[local_idx],
                }

    render_out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(render_out), cv2.VideoWriter_fourcc(*"mp4v"), float(render_fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {render_out}")

    frames_with_prediction = 0
    try:
        for frame_idx in range(render_frame_count):
            image = cv2.imread(str(image_paths[frame_idx]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read frame: {image_paths[frame_idx]}")
            hand_preds = per_frame.get(frame_idx, {})
            counts = []
            if hand_preds:
                frames_with_prediction += 1
            for hand_idx, payload in sorted(hand_preds.items()):
                visible = _draw_cam_hand_overlay(image, payload["joints"], payload["root_rotmat"], hand_idx, focal, center_xy)
                counts.append(f"{_hand_name(hand_idx)}={visible}/21")
            _annotate_frame(image, [f"motion frame={frame_idx}", " ".join(counts) if counts else "no prediction"])
            writer.write(image)
    finally:
        writer.release()

    return {
        "render_out": str(render_out),
        "rendered_frames": int(render_frame_count),
        "frames_with_prediction": int(frames_with_prediction),
        "focal": float(focal),
    }


def _load_world_prediction(seq_folder: Path) -> dict | None:
    if not (seq_folder / "world_space_res.pth").is_file():
        return None
    return _load_world_space_prediction({"episode_id": seq_folder.name}, str(seq_folder / "world_space_res.pth"))


def _summarize_world_prediction(prediction: dict | None) -> dict | None:
    if prediction is None:
        print("\n=== World summary ===")
        print("world_space_res.pth missing; skip world/build checks")
        return None

    print("\n=== World summary ===")
    pred_trans = prediction["pred_trans"].cpu().numpy()
    pred_rot = prediction["pred_rot"].cpu().numpy()
    pred_hand_pose = prediction["pred_hand_pose"].cpu().numpy()
    pred_valid = np.asarray(prediction["pred_valid"])
    summary = {"num_frames": int(pred_trans.shape[1]), "hands": {}}
    print(f"num_frames={pred_trans.shape[1]}")
    for hand_idx in (0, 1):
        hand_name = _hand_name(hand_idx)
        valid = pred_valid[hand_idx] > 0.5
        valid_count = int(valid.sum())
        print(f"[{hand_name}] valid={valid_count}/{pred_valid.shape[1]}")
        hand_summary = {"valid": valid_count}
        if valid_count > 0:
            hand_trans = pred_trans[hand_idx, valid]
            hand_rot = pred_rot[hand_idx, valid]
            hand_pose = pred_hand_pose[hand_idx, valid]
            print(f"  trans xyz range={_format_range(hand_trans)}")
            print(f"  trans norm range={_format_norm_range(hand_trans)}")
            print(f"  rot aa range={_format_range(hand_rot)}")
            print(f"  hand_pose aa range={_format_range(hand_pose)}")
            hand_summary.update(
                {
                    "trans_range": _format_range(hand_trans),
                    "trans_norm_range": _format_norm_range(hand_trans),
                    "rot_range": _format_range(hand_rot),
                    "hand_pose_range": _format_range(hand_pose),
                }
            )
        summary["hands"][hand_name] = hand_summary
    return summary


def _render_world_video(
    seq_folder: Path,
    prediction: dict,
    *,
    image_paths: list[Path],
    mano_dir: str | None,
    device: str,
    render_out: Path,
    render_max_frames: int,
    render_fps: int,
) -> dict:
    import torch

    pred_trans = prediction["pred_trans"].float()
    pred_rot = prediction["pred_rot"].float()
    pred_hand_pose = prediction["pred_hand_pose"].float()
    pred_betas = prediction["pred_betas"].float()
    pred_valid = np.asarray(prediction["pred_valid"])
    source_frame_count = int(pred_trans.shape[1])
    render_frame_count = min(source_frame_count, len(image_paths), max(1, int(render_max_frames)))
    ep = {"crop_dir": str(seq_folder), "episode_id": seq_folder.name}
    extrinsics, intrinsic = _load_episode_camera_features(ep, source_frame_count)

    first_frame = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read frame: {image_paths[0]}")
    height, width = first_frame.shape[:2]
    torch_device = torch.device(device)
    mano_right, mano_left = build_mano_models(torch_device, mano_dir=mano_dir)

    with torch.inference_mode():
        outputs_left = run_mano_left(
            pred_trans[0:1].to(torch_device),
            pred_rot[0:1].to(torch_device),
            pred_hand_pose[0:1].reshape(1, source_frame_count, 15, 3).to(torch_device),
            betas=pred_betas[0:1].to(torch_device),
            mano_model=mano_left,
        )
        outputs_right = run_mano(
            pred_trans[1:2].to(torch_device),
            pred_rot[1:2].to(torch_device),
            pred_hand_pose[1:2].reshape(1, source_frame_count, 15, 3).to(torch_device),
            betas=pred_betas[1:2].to(torch_device),
            mano_model=mano_right,
        )
    left_joints = outputs_left["joints"][0].detach().cpu().numpy().astype(np.float32)
    right_joints = outputs_right["joints"][0].detach().cpu().numpy().astype(np.float32)
    root_rotmats = angle_axis_to_rotation_matrix(pred_rot.reshape(-1, 3)).reshape(2, source_frame_count, 3, 3).cpu().numpy().astype(np.float32)

    render_out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(render_out), cv2.VideoWriter_fourcc(*"mp4v"), float(render_fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {render_out}")

    frames_with_prediction = 0
    try:
        for frame_idx in range(render_frame_count):
            image = cv2.imread(str(image_paths[frame_idx]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read frame: {image_paths[frame_idx]}")
            counts = []
            for hand_idx, joints_world in ((0, left_joints), (1, right_joints)):
                if pred_valid[hand_idx, frame_idx] <= 0.5:
                    continue
                visible = _draw_world_hand_overlay(
                    image,
                    joints_world[frame_idx],
                    root_rotmats[hand_idx, frame_idx],
                    hand_idx,
                    extrinsics[frame_idx],
                    intrinsic,
                )
                counts.append(f"{_hand_name(hand_idx)}={visible}/21")
            if counts:
                frames_with_prediction += 1
            _annotate_frame(image, [f"world frame={frame_idx}", " ".join(counts) if counts else "no valid prediction"])
            writer.write(image)
    finally:
        writer.release()

    return {
        "render_out": str(render_out),
        "rendered_frames": int(render_frame_count),
        "frames_with_prediction": int(frames_with_prediction),
        "intrinsic": intrinsic.astype(np.float32).tolist(),
    }


def _action_consistency_stats(lowdim_all: np.ndarray) -> dict:
    if lowdim_all.shape[0] <= 1:
        return {"wrist_max_abs_diff": 0.0, "hand_max_abs_diff": 0.0}
    wrist_diff = np.abs(lowdim_all[:-1, WRIST_ACTION_SLICE] - lowdim_all[1:, WRIST_STATE_SLICE])
    hand_diff = np.abs(lowdim_all[:-1, HAND_ACTION_SLICE] - lowdim_all[1:, HAND_STATE_SLICE])
    return {
        "wrist_max_abs_diff": float(wrist_diff.max()),
        "hand_max_abs_diff": float(hand_diff.max()),
    }


def _summarize_lowdim(label: str, lowdim_all: np.ndarray) -> dict:
    print(f"\n=== {label} lowdim summary ===")
    invalid_frames = []
    for frame_idx, lowdim in enumerate(lowdim_all):
        sanity = validate_lowdim_numeric_sanity(lowdim)
        if not sanity["valid"]:
            invalid_frames.append({"frame": int(frame_idx), "issues": list(sanity["issues"])})
    action_stats = _action_consistency_stats(lowdim_all)
    print(f"frames={lowdim_all.shape[0]}")
    print(f"invalid_lowdim_frames={len(invalid_frames)}")
    print(f"wrist_action_next_state_max_abs_diff={action_stats['wrist_max_abs_diff']:.6g}")
    print(f"hand_action_next_state_max_abs_diff={action_stats['hand_max_abs_diff']:.6g}")
    if invalid_frames:
        for item in invalid_frames[:10]:
            print(f"  frame={item['frame']} issues={','.join(item['issues'])}")
    return {
        "frames": int(lowdim_all.shape[0]),
        "invalid_lowdim_frames": invalid_frames,
        "action_consistency": action_stats,
    }


def _build_current_export_lowdim(
    seq_folder: Path,
    prediction: dict,
    *,
    mano_dir: str | None,
    device: str,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
) -> dict:
    import torch

    pred_trans = prediction["pred_trans"].float()
    pred_rot = prediction["pred_rot"].float()
    pred_hand_pose = prediction["pred_hand_pose"].float()
    pred_betas = prediction["pred_betas"].float()
    pred_valid = np.asarray(prediction["pred_valid"])
    source_frame_count = int(pred_trans.shape[1])
    torch_device = torch.device(device)
    mano_right, mano_left = build_mano_models(torch_device, mano_dir=mano_dir)
    wrist_state, hand_state = _compute_joint_states(pred_trans, pred_rot, pred_hand_pose, pred_betas, mano_right, mano_left, torch_device)
    ep = {"crop_dir": str(seq_folder), "episode_id": seq_folder.name}
    extrinsics, intrinsic = _load_episode_camera_features(ep, source_frame_count)
    presence_per_frame = _compute_presence_per_frame(pred_valid, source_frame_count)
    target_count = source_frame_count
    if interpolate_labels and source_fps > 0 and target_fps > 0 and source_frame_count > 1:
        duration = float(source_frame_count - 1) / float(source_fps)
        target_count = int(round(duration * float(target_fps))) + 1

    wrist_state, hand_state, pred_rot_resampled, pred_hand_pose_resampled, pred_betas_resampled, extrinsics_resampled, presence_resampled = resample_episode_features(
        wrist_state[:source_frame_count],
        hand_state[:source_frame_count],
        pred_rot[:, :source_frame_count],
        pred_hand_pose[:, :source_frame_count],
        pred_betas[:, :source_frame_count],
        extrinsics[:source_frame_count],
        presence_per_frame[:source_frame_count],
        target_count,
        source_fps=source_fps,
        target_fps=target_fps,
        interpolate_labels=interpolate_labels,
    )
    lowdim_all = _build_lowdim_features(wrist_state, hand_state, extrinsics_resampled[:target_count], intrinsic)
    return {
        "lowdim_all": lowdim_all.astype(np.float32),
        "presence_per_frame": np.asarray(presence_resampled),
        "pred_rot": pred_rot_resampled,
        "pred_hand_pose": pred_hand_pose_resampled,
        "pred_betas": pred_betas_resampled,
        "extrinsics": extrinsics_resampled.astype(np.float32),
        "intrinsic": np.asarray(intrinsic, dtype=np.float32),
    }


def _decode_image_bgr(image_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode JPEG/PNG image bytes")
    return image


def _nearest_source_indices(source_count: int, target_count: int, source_fps: float, target_fps: float) -> np.ndarray:
    if target_count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if source_count <= 1:
        return np.zeros((target_count,), dtype=np.int64)
    source_times, target_times = build_source_target_times(source_count, target_count, source_fps, target_fps)
    float_indices = np.interp(target_times, source_times, np.arange(source_count, dtype=np.float64))
    return np.clip(np.rint(float_indices).astype(np.int64), 0, source_count - 1)


def _render_lowdim_video(
    *,
    label: str,
    lowdim_all: np.ndarray,
    image_reader,
    image_count: int,
    image_index_map: np.ndarray,
    render_out: Path,
    render_max_frames: int,
    render_fps: int,
) -> dict:
    render_frame_count = min(int(lowdim_all.shape[0]), int(render_max_frames))
    first_image = image_reader(int(image_index_map[0]))
    height, width = first_image.shape[:2]
    writer = cv2.VideoWriter(str(render_out), cv2.VideoWriter_fourcc(*"mp4v"), float(render_fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {render_out}")
    frames_with_projection = 0
    try:
        for frame_idx in range(render_frame_count):
            image = image_reader(int(image_index_map[frame_idx])).copy()
            lowdim = np.asarray(lowdim_all[frame_idx], dtype=np.float32)
            extrinsic = lowdim[EXTRINSIC_SLICE].reshape(4, 4)
            intrinsic = lowdim[INTRINSIC_SLICE]
            left_wrist = lowdim[0:3]
            right_wrist = lowdim[3:6]
            left_rotmat = rot6d_to_rotmat(lowdim[6:12])[0].astype(np.float32)
            right_rotmat = rot6d_to_rotmat(lowdim[12:18])[0].astype(np.float32)
            left_tips = lowdim[18:33].reshape(5, 3)
            right_tips = lowdim[33:48].reshape(5, 3)
            counts = []
            left_visible = _draw_world_hand_overlay(
                image,
                np.concatenate([left_wrist[None, :], left_tips], axis=0),
                left_rotmat,
                0,
                extrinsic,
                intrinsic,
            )
            right_visible = _draw_world_hand_overlay(
                image,
                np.concatenate([right_wrist[None, :], right_tips], axis=0),
                right_rotmat,
                1,
                extrinsic,
                intrinsic,
            )
            if left_visible > 0:
                counts.append(f"left={left_visible}/6")
            if right_visible > 0:
                counts.append(f"right={right_visible}/6")
            if counts:
                frames_with_projection += 1
            _annotate_frame(image, [f"{label} frame={frame_idx}", " ".join(counts) if counts else "no projected points"])
            writer.write(image)
    finally:
        writer.release()
    return {
        "render_out": str(render_out),
        "rendered_frames": int(render_frame_count),
        "frames_with_projection": int(frames_with_projection),
        "image_count": int(image_count),
    }


def _load_wds_clip_samples(shard_path: Path, clip_id: str) -> dict | None:
    if not shard_path.is_file():
        raise FileNotFoundError(f"WDS shard not found: {shard_path}")

    samples: dict[str, dict] = {}
    with tarfile.open(shard_path, "r") as tar_reader:
        for member in tar_reader:
            if not member.isfile():
                continue
            for suffix, field_name in WDS_MEMBER_SUFFIXES.items():
                if not member.name.endswith(suffix):
                    continue
                sample_key = member.name[: -len(suffix)]
                if not sample_key.startswith(f"{clip_id}_f"):
                    break
                member_file = tar_reader.extractfile(member)
                if member_file is None:
                    break
                payload = member_file.read()
                sample = samples.setdefault(sample_key, {"key": sample_key})
                sample[field_name] = payload
                break

    if not samples:
        return None

    ordered = sorted(samples.values(), key=lambda item: _parse_frame_index_from_key(item["key"]))
    lowdim_all = np.stack([decode_lowdim(item["lowdim_bytes"]) for item in ordered], axis=0).astype(np.float32)
    image_bytes = [item.get("image_bytes") for item in ordered]
    metas = []
    for item in ordered:
        meta_bytes = item.get("meta_bytes")
        if meta_bytes is None:
            metas.append(None)
        else:
            metas.append(json.loads(meta_bytes.decode("utf-8")))
    return {
        "clip_id": clip_id,
        "sample_keys": [item["key"] for item in ordered],
        "lowdim_all": lowdim_all,
        "image_bytes": image_bytes,
        "metas": metas,
    }


def _compare_lowdim_arrays(label_a: str, a: np.ndarray, label_b: str, b: np.ndarray) -> dict:
    compare_count = min(int(a.shape[0]), int(b.shape[0]))
    if compare_count <= 0:
        return {"compare_frames": 0}
    diff = np.abs(np.asarray(a[:compare_count], dtype=np.float32) - np.asarray(b[:compare_count], dtype=np.float32))
    max_per_dim = diff.max(axis=0)
    top_indices = np.argsort(max_per_dim)[::-1][:10]
    summary = {
        "compare_frames": compare_count,
        "frame_count_a": int(a.shape[0]),
        "frame_count_b": int(b.shape[0]),
        "overall_max_abs_diff": float(diff.max()),
        "state_max_abs_diff": float(diff[:, :48].max()),
        "action_max_abs_diff": float(diff[:, 48:96].max()),
        "extrinsic_max_abs_diff": float(diff[:, 96:112].max()),
        "intrinsic_max_abs_diff": float(diff[:, 112:116].max()),
        "top_dims": [
            {
                "index": int(idx),
                "name": LOWDIM_DIMENSION_NAMES[int(idx)],
                "max_abs_diff": float(max_per_dim[int(idx)]),
            }
            for idx in top_indices
        ],
    }
    print(f"\n=== Compare {label_a} vs {label_b} ===")
    print(f"compare_frames={summary['compare_frames']} frame_count_a={summary['frame_count_a']} frame_count_b={summary['frame_count_b']}")
    print(f"overall_max_abs_diff={summary['overall_max_abs_diff']:.6g}")
    print(f"state_max_abs_diff={summary['state_max_abs_diff']:.6g}")
    print(f"action_max_abs_diff={summary['action_max_abs_diff']:.6g}")
    print(f"extrinsic_max_abs_diff={summary['extrinsic_max_abs_diff']:.6g}")
    print(f"intrinsic_max_abs_diff={summary['intrinsic_max_abs_diff']:.6g}")
    for item in summary["top_dims"]:
        print(f"  dim[{item['index']}] {item['name']} max_abs_diff={item['max_abs_diff']:.6g}")
    return summary


def _analyze_seq_folder(args, seq_folder: Path, *, render_dir_override: str | None = None) -> dict:
    if not seq_folder.is_dir():
        raise FileNotFoundError(f"seq_folder not found: {seq_folder}")

    print(f"seq_folder: {seq_folder}")
    report = {"seq_folder": str(seq_folder)}

    image_paths = None
    try:
        image_paths = _resolve_image_paths(seq_folder, args.frame_dir)
        resolved_frame_dir = image_paths[0].parent
        print(f"resolved_frame_dir: {resolved_frame_dir}")
        report["frame_dir"] = str(resolved_frame_dir)
    except Exception as error:
        print(f"Warning: failed to resolve frame_dir: {error}")
        report["frame_dir_error"] = str(error)

    render_dir = _resolve_render_dir(seq_folder, render_dir_override if render_dir_override is not None else args.render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    report["render_dir"] = str(render_dir)

    cam_chunks = _load_cam_space_chunks(seq_folder)
    report["motion_cam_space"] = _summarize_cam_space(cam_chunks, args.chunk_limit)

    if image_paths is not None:
        try:
            report["motion_render"] = _render_motion_video(
                seq_folder,
                cam_chunks,
                image_paths=image_paths,
                mano_dir=args.mano_dir,
                device=args.device,
                render_out=render_dir / "motion_cam_space.mp4",
                render_max_frames=args.render_max_frames,
                render_fps=args.render_fps,
            )
            print(f"motion render: {report['motion_render']['render_out']}")
        except Exception as error:
            print(f"Warning: failed to render motion video: {error}")
            report["motion_render_error"] = str(error)

    prediction = _load_world_prediction(seq_folder)
    world_summary = _summarize_world_prediction(prediction)
    if world_summary is not None:
        report["world_summary"] = world_summary

    if prediction is not None and image_paths is not None:
        try:
            report["world_render"] = _render_world_video(
                seq_folder,
                prediction,
                image_paths=image_paths,
                mano_dir=args.mano_dir,
                device=args.device,
                render_out=render_dir / "world_space.mp4",
                render_max_frames=args.render_max_frames,
                render_fps=args.render_fps,
            )
            print(f"world render: {report['world_render']['render_out']}")
        except Exception as error:
            print(f"Warning: failed to render world video: {error}")
            report["world_render_error"] = str(error)

    current_export = None
    if prediction is not None:
        try:
            current_export = _build_current_export_lowdim(
                seq_folder,
                prediction,
                mano_dir=args.mano_dir,
                device=args.device,
                source_fps=args.source_fps,
                target_fps=args.target_fps,
                interpolate_labels=bool(args.interpolate_labels),
            )
            report["current_export_lowdim"] = _summarize_lowdim("Current export", current_export["lowdim_all"])
        except Exception as error:
            print(f"Warning: failed to build current-export lowdim: {error}")
            report["current_export_lowdim_error"] = str(error)

    if current_export is not None and image_paths is not None:
        try:
            image_index_map = _nearest_source_indices(
                len(image_paths),
                int(current_export["lowdim_all"].shape[0]),
                float(args.source_fps),
                float(args.target_fps),
            )
            report["current_export_render"] = _render_lowdim_video(
                label="current_export",
                lowdim_all=current_export["lowdim_all"],
                image_reader=lambda idx: cv2.imread(str(image_paths[int(idx)]), cv2.IMREAD_COLOR),
                image_count=len(image_paths),
                image_index_map=image_index_map,
                render_out=render_dir / "current_export_lowdim.mp4",
                render_max_frames=args.render_max_frames,
                render_fps=args.render_fps,
            )
            print(f"current export render: {report['current_export_render']['render_out']}")
        except Exception as error:
            print(f"Warning: failed to render current-export lowdim video: {error}")
            report["current_export_render_error"] = str(error)

    wds_clip_id = args.wds_clip_id or seq_folder.name
    if args.wds_shard:
        try:
            wds_payload = _load_wds_clip_samples(Path(args.wds_shard).expanduser().resolve(), wds_clip_id)
            if wds_payload is None:
                print(f"\n=== Final WDS summary ===\nclip_id={wds_clip_id} not found in shard")
                report["wds_error"] = f"clip_id {wds_clip_id} not found"
            else:
                report["wds_lowdim"] = _summarize_lowdim("Final WDS", wds_payload["lowdim_all"])
                if current_export is not None:
                    report["current_vs_wds"] = _compare_lowdim_arrays(
                        "current_export",
                        current_export["lowdim_all"],
                        "final_wds",
                        wds_payload["lowdim_all"],
                    )
                if any(payload is not None for payload in wds_payload["image_bytes"]):
                    image_index_map = np.arange(len(wds_payload["image_bytes"]), dtype=np.int64)
                    report["wds_render"] = _render_lowdim_video(
                        label="final_wds",
                        lowdim_all=wds_payload["lowdim_all"],
                        image_reader=lambda idx: _decode_image_bgr(wds_payload["image_bytes"][int(idx)]),
                        image_count=len(wds_payload["image_bytes"]),
                        image_index_map=image_index_map,
                        render_out=render_dir / "final_wds_lowdim.mp4",
                        render_max_frames=args.render_max_frames,
                        render_fps=args.render_fps,
                    )
                    print(f"final WDS render: {report['wds_render']['render_out']}")
        except Exception as error:
            print(f"Warning: failed WDS comparison: {error}")
            report["wds_error"] = str(error)

    return report


def _multi_report_path(base_path: Path, clip_id: str) -> Path:
    suffix = base_path.suffix or ".json"
    stem = base_path.stem if base_path.suffix else base_path.name
    return base_path.with_name(f"{stem}.{clip_id}{suffix}")


def main() -> None:
    args = build_parser().parse_args()
    seq_folders = [Path(item).expanduser().resolve() for item in args.seq_folders]
    multi_clip = len(seq_folders) > 1
    reports = []

    for index, seq_folder in enumerate(seq_folders):
        if multi_clip:
            if index > 0:
                print()
            print("=" * 120)

        render_dir_override = None
        if args.render_dir:
            render_root = Path(args.render_dir).expanduser().resolve()
            render_dir_override = str(render_root / seq_folder.name) if multi_clip else str(render_root)

        report = _analyze_seq_folder(args, seq_folder, render_dir_override=render_dir_override)
        reports.append(report)

    if args.report_out:
        report_out = Path(args.report_out).expanduser().resolve()
        report_out.parent.mkdir(parents=True, exist_ok=True)
        if multi_clip:
            aggregate = {"clips": reports}
            report_out.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nreport_out: {report_out}")
            for item in reports:
                per_clip_path = _multi_report_path(report_out, Path(item["seq_folder"]).name)
                per_clip_path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"per_clip_report: {per_clip_path}")
        else:
            report_out.write_text(json.dumps(reports[0], ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nreport_out: {report_out}")


if __name__ == "__main__":
    main()
