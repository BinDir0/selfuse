#!/usr/bin/env python3
"""Render motion-stage cam_space hand predictions directly on source frames."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hawor.utils.process import get_mano_cfg, run_mano, run_mano_left
from hawor.utils.rotation import rotation_matrix_to_angle_axis
from lib.pipeline.clip_manifest import load_clip_manifest
from lib.models.mano_wrapper import MANO
from lib.pipeline.exporters.webdataset_discovery import load_or_build_frame_index
from lib.pipeline.frame_sources import build_frame_source_from_descriptor


HAND_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
LEFT_COLOR = (255, 0, 255)
RIGHT_COLOR = (0, 255, 0)
TRACK_DIR_RE = re.compile(r"tracks_(\d+)_(\d+)$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay motion-stage cam_space MANO joints directly on source RGB frames."
    )
    parser.add_argument("--seq-folder", required=True, help="Processed seq_folder, e.g. factory*/outputs/<clip_id>")
    parser.add_argument(
        "--descriptor-manifest",
        default=None,
        help="Optional clip manifest used to resolve tar-backed source frames for BuildAI shard layouts.",
    )
    parser.add_argument("--output", required=True, help="Output mp4 path")
    parser.add_argument("--mano-device", default="cpu", help="Device for MANO forward pass, e.g. cpu or cuda:0")
    parser.add_argument("--fps", type=int, default=30, help="FPS for the output video")
    parser.add_argument("--frame-start", type=int, default=0, help="Inclusive starting frame index")
    parser.add_argument("--frame-end", type=int, default=None, help="Exclusive ending frame index")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on rendered frame count")
    parser.add_argument("--render-every", type=int, default=1, help="Render every Nth frame")
    parser.add_argument("--line-thickness", type=int, default=2, help="Polyline thickness")
    parser.add_argument("--joint-radius", type=int, default=3, help="Joint point radius")
    return parser


def _find_track_dir(seq_folder: Path) -> Path:
    candidates: list[tuple[int, int, int, Path]] = []
    for path in seq_folder.glob("tracks_*_*"):
        if not (path / "frame_chunks_all.npy").is_file():
            continue
        match = TRACK_DIR_RE.fullmatch(path.name)
        if match is None:
            continue
        start_idx = int(match.group(1))
        end_idx = int(match.group(2))
        candidates.append((end_idx - start_idx, end_idx, start_idx, path))
    if not candidates:
        raise FileNotFoundError(f"No tracks_*_* directory with frame_chunks_all.npy found under {seq_folder}")
    candidates.sort()
    return candidates[-1][-1]


def _load_frame_index(extracted_dir: Path) -> dict[int, str]:
    frame_index = load_or_build_frame_index(str(extracted_dir), rescan=False)
    if not frame_index:
        raise RuntimeError(f"No frames found in {extracted_dir}")
    return frame_index


def _infer_num_frames(seq_folder: Path, frame_index: dict[int, str]) -> int:
    world_path = seq_folder / "world_space_res.pth"
    if world_path.is_file():
        pred_trans, *_ = joblib.load(world_path)
        return int(np.asarray(pred_trans).shape[1])
    return int(max(frame_index.keys()) + 1)


def _load_descriptor(seq_folder: Path, manifest_path: Path):
    clip_id = seq_folder.name
    records = load_clip_manifest(manifest_path)
    for record in records:
        if record.clip_id == clip_id:
            return record.descriptor
    raise KeyError(f"clip_id {clip_id!r} not found in descriptor manifest {manifest_path}")


def _load_intrinsic(seq_folder: Path, image_shape: tuple[int, int, int]) -> np.ndarray:
    slam_dir = seq_folder / "SLAM"
    slam_files = sorted(slam_dir.glob("hawor_slam_w_scale_*.npz")) if slam_dir.is_dir() else []
    if slam_files:
        with np.load(str(slam_files[0]), allow_pickle=False) as payload:
            if "img_focal" in payload.files and "img_center" in payload.files:
                center = np.asarray(payload["img_center"], dtype=np.float32).reshape(-1)
                if center.shape[0] >= 2:
                    focal = float(payload["img_focal"])
                    return np.array([focal, focal, float(center[0]), float(center[1])], dtype=np.float32)

    height, width = image_shape[:2]
    focal = float(max(height, width))
    return np.array([focal, focal, width / 2.0, height / 2.0], dtype=np.float32)


def _build_mano_models(device: torch.device):
    mano_right = MANO(**get_mano_cfg(is_right=True)).to(device)
    mano_left = MANO(**get_mano_cfg(is_right=False)).to(device)
    mano_left.shapedirs[:, 0, :] *= -1
    return mano_right, mano_left


def _ensure_batched_feature(value: np.ndarray, *, feature_dim: int, num_frames: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1 and array.shape[0] == feature_dim:
        array = array[None, None, :]
        return np.repeat(array, num_frames, axis=1)
    if array.ndim == 2:
        if array.shape == (1, feature_dim):
            array = array[:, None, :]
            return np.repeat(array, num_frames, axis=1)
        if array.shape == (num_frames, feature_dim):
            return array[None, :, :]
    if array.ndim == 3 and array.shape[0] == 1 and array.shape[2] == feature_dim:
        if array.shape[1] == num_frames:
            return array
        if array.shape[1] == 1:
            return np.repeat(array, num_frames, axis=1)
    raise ValueError(f"Unexpected feature shape={array.shape}, expected last dim {feature_dim} for {num_frames} frames")


def _load_chunk_json(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {key: np.asarray(value, dtype=np.float32) for key, value in payload.items()}


def _project_points(points_cam: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intrinsic.tolist()
    points = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        uv_valid = uv[valid]
        uv_valid[:, 0] = fx * points[valid, 0] / z[valid] + cx
        uv_valid[:, 1] = fy * points[valid, 1] / z[valid] + cy
        uv[valid] = uv_valid
    return uv, valid


def _draw_hand(
    image_bgr: np.ndarray,
    joints_cam: np.ndarray,
    intrinsic: np.ndarray,
    *,
    color,
    line_thickness: int,
    joint_radius: int,
    label: str,
) -> None:
    uv, valid = _project_points(joints_cam, intrinsic)
    image_h, image_w = image_bgr.shape[:2]

    def inframe(idx: int) -> bool:
        if not valid[idx]:
            return False
        x, y = uv[idx]
        return (-0.25 * image_w) <= x <= (1.25 * image_w) and (-0.25 * image_h) <= y <= (1.25 * image_h)

    for chain in HAND_CHAINS:
        for start_idx, end_idx in zip(chain[:-1], chain[1:]):
            if not inframe(start_idx) or not inframe(end_idx):
                continue
            p0 = tuple(np.round(uv[start_idx]).astype(np.int32))
            p1 = tuple(np.round(uv[end_idx]).astype(np.int32))
            cv2.line(image_bgr, p0, p1, color, line_thickness, cv2.LINE_AA)

    for joint_idx in range(joints_cam.shape[0]):
        if not inframe(joint_idx):
            continue
        point = tuple(np.round(uv[joint_idx]).astype(np.int32))
        cv2.circle(image_bgr, point, joint_radius, color, -1, cv2.LINE_AA)

    if inframe(0):
        wrist_uv = tuple(np.round(uv[0]).astype(np.int32))
        cv2.putText(
            image_bgr,
            label,
            (wrist_uv[0] + 8, wrist_uv[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def _decode_rotation_mats(payload: dict[str, np.ndarray], num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    root_rotmat = np.asarray(payload["init_root_orient"], dtype=np.float32)
    hand_rotmat = np.asarray(payload["init_hand_pose"], dtype=np.float32)

    if root_rotmat.ndim == 4 and root_rotmat.shape[0] == 1:
        pass
    elif root_rotmat.ndim == 3 and root_rotmat.shape[0] == num_frames:
        root_rotmat = root_rotmat[None, ...]
    else:
        raise ValueError(f"Unexpected init_root_orient shape: {root_rotmat.shape}")

    if hand_rotmat.ndim == 5 and hand_rotmat.shape[0] == 1:
        pass
    elif hand_rotmat.ndim == 4 and hand_rotmat.shape[0] == num_frames:
        hand_rotmat = hand_rotmat[None, ...]
    else:
        raise ValueError(f"Unexpected init_hand_pose shape: {hand_rotmat.shape}")

    return root_rotmat, hand_rotmat


def _render_chunk_joints(
    payload: dict[str, np.ndarray],
    frame_ck: np.ndarray,
    *,
    hand_idx: int,
    device: torch.device,
    mano_left,
    mano_right,
) -> np.ndarray:
    num_frames = int(frame_ck.shape[0])
    trans = _ensure_batched_feature(payload["init_trans"], feature_dim=3, num_frames=num_frames)
    betas = _ensure_batched_feature(payload["init_betas"], feature_dim=10, num_frames=num_frames)
    root_rotmat, hand_rotmat = _decode_rotation_mats(payload, num_frames)

    root_aa = rotation_matrix_to_angle_axis(torch.from_numpy(root_rotmat).float().reshape(-1, 3, 3)).reshape(1, num_frames, 3)
    hand_aa = rotation_matrix_to_angle_axis(torch.from_numpy(hand_rotmat).float().reshape(-1, 3, 3)).reshape(1, num_frames, 15, 3)

    trans_t = torch.from_numpy(trans).float().to(device)
    betas_t = torch.from_numpy(betas).float().to(device)
    root_aa_t = root_aa.float().to(device)
    hand_aa_t = hand_aa.float().to(device)

    with torch.no_grad():
        if hand_idx == 0:
            outputs = run_mano_left(
                trans_t,
                root_aa_t,
                hand_aa_t,
                betas=betas_t,
                use_cuda=device.type == "cuda",
                mano_model=mano_left,
            )
        else:
            outputs = run_mano(
                trans_t,
                root_aa_t,
                hand_aa_t,
                betas=betas_t,
                use_cuda=device.type == "cuda",
                mano_model=mano_right,
            )
    return outputs["joints"][0].detach().cpu().numpy().astype(np.float32)


def _select_frame_ids(
    available_frame_ids: list[int],
    *,
    num_frames: int,
    frame_start: int,
    frame_end: int | None,
    render_every: int,
    max_frames: int | None,
) -> list[int]:
    frame_ids = sorted(frame_id for frame_id in available_frame_ids if 0 <= frame_id < num_frames and frame_id >= frame_start)
    if frame_end is not None:
        frame_ids = [frame_id for frame_id in frame_ids if frame_id < frame_end]
    if render_every > 1:
        frame_ids = frame_ids[::render_every]
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    return frame_ids


def main() -> None:
    args = build_parser().parse_args()
    seq_folder = Path(args.seq_folder).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not seq_folder.is_dir():
        raise FileNotFoundError(f"seq_folder not found: {seq_folder}")
    if args.render_every < 1:
        raise ValueError("--render-every must be >= 1")

    descriptor = None
    frame_source = None
    frame_index = None

    if args.descriptor_manifest:
        manifest_path = Path(args.descriptor_manifest).expanduser().resolve()
        descriptor = _load_descriptor(seq_folder, manifest_path)
        frame_source = build_frame_source_from_descriptor(descriptor)
        num_frames = int(min(int(descriptor.frame_count), _infer_num_frames(seq_folder, {0: ""})))
        available_frame_ids = list(range(int(descriptor.frame_count)))
    else:
        extracted_dir = seq_folder / "extracted_images"
        frame_index = _load_frame_index(extracted_dir)
        num_frames = _infer_num_frames(seq_folder, frame_index)
        available_frame_ids = sorted(frame_index.keys())

    track_dir = _find_track_dir(seq_folder)
    frame_chunks_all = joblib.load(track_dir / "frame_chunks_all.npy")

    if frame_source is not None:
        first_frame = frame_source.get_frame(0, rgb=False)
    else:
        first_frame_path = Path(frame_index[min(frame_index.keys())])
        first_frame = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
        if first_frame is None:
            raise RuntimeError(f"Failed to read first frame: {first_frame_path}")
    intrinsic = _load_intrinsic(seq_folder, first_frame.shape)

    device = torch.device(args.mano_device)
    mano_right, mano_left = _build_mano_models(device)

    all_joints = {
        0: np.full((num_frames, 21, 3), np.nan, dtype=np.float32),
        1: np.full((num_frames, 21, 3), np.nan, dtype=np.float32),
    }

    for hand_idx in (0, 1):
        chunks = frame_chunks_all.get(hand_idx, [])
        print(f"hand={hand_idx} chunks={len(chunks)}", flush=True)
        for chunk in chunks:
            frame_ck = np.asarray(chunk, dtype=np.int64).reshape(-1)
            if frame_ck.size == 0:
                continue
            key = f"{int(frame_ck[0])}_{int(frame_ck[-1])}"
            json_path = seq_folder / "cam_space" / str(hand_idx) / f"{key}.json"
            if not json_path.is_file():
                print(f"warning: missing {json_path}", flush=True)
                continue
            payload = _load_chunk_json(json_path)
            joints = _render_chunk_joints(
                payload,
                frame_ck,
                hand_idx=hand_idx,
                device=device,
                mano_left=mano_left,
                mano_right=mano_right,
            )
            valid_frames = frame_ck[: min(len(frame_ck), joints.shape[0])]
            valid_mask = (valid_frames >= 0) & (valid_frames < num_frames)
            all_joints[hand_idx][valid_frames[valid_mask]] = joints[: len(valid_frames)][valid_mask]

    frame_ids = _select_frame_ids(
        available_frame_ids,
        num_frames=num_frames,
        frame_start=int(args.frame_start),
        frame_end=None if args.frame_end is None else int(args.frame_end),
        render_every=int(args.render_every),
        max_frames=None if args.max_frames is None else int(args.max_frames),
    )
    if not frame_ids:
        raise RuntimeError("No frames selected for rendering")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (first_frame.shape[1], first_frame.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    rendered_mid_frame = None
    try:
        total = len(frame_ids)
        for render_idx, frame_id in enumerate(frame_ids, start=1):
            if frame_source is not None:
                image = frame_source.get_frame(int(frame_id), rgb=False)
            else:
                image = cv2.imread(frame_index[frame_id], cv2.IMREAD_COLOR)
                if image is None:
                    continue

            left_joints = all_joints[0][frame_id]
            right_joints = all_joints[1][frame_id]
            if np.isfinite(left_joints).any():
                _draw_hand(
                    image,
                    left_joints,
                    intrinsic,
                    color=LEFT_COLOR,
                    line_thickness=int(args.line_thickness),
                    joint_radius=int(args.joint_radius),
                    label="L-cam",
                )
            if np.isfinite(right_joints).any():
                _draw_hand(
                    image,
                    right_joints,
                    intrinsic,
                    color=RIGHT_COLOR,
                    line_thickness=int(args.line_thickness),
                    joint_radius=int(args.joint_radius),
                    label="R-cam",
                )

            cv2.putText(
                image,
                f"frame={frame_id} ({render_idx}/{total})",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(image)
            if render_idx == (total // 2) + 1:
                rendered_mid_frame = image.copy()
    finally:
        writer.release()

    poster_path = output_path.with_suffix(".poster.jpg")
    if rendered_mid_frame is not None:
        cv2.imwrite(str(poster_path), rendered_mid_frame)

    print(f"wrote video: {output_path}", flush=True)
    print(f"wrote poster: {poster_path}", flush=True)


if __name__ == "__main__":
    main()
