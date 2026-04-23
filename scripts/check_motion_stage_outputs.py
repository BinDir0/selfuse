#!/usr/bin/env python3
"""Smoke-test one processed clip's motion-stage cam_space outputs and render overlays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hawor.utils.process import run_mano, run_mano_left
from hawor.utils.rotation import rotation_matrix_to_angle_axis
from lib.pipeline.exporters.webdataset_features import build_mano_models
from lib.pipeline.stage_api import get_track_range


MANO_JOINT_TREE = [
    [(0, 1), (1, 2), (2, 3), (3, 4)],
    [(0, 5), (5, 6), (6, 7), (7, 8)],
    [(0, 9), (9, 10), (10, 11), (11, 12)],
    [(0, 13), (13, 14), (14, 15), (15, 16)],
    [(0, 17), (17, 18), (18, 19), (19, 20)],
]
HAND_COLORS = {
    0: (80, 200, 255),   # left: orange-ish
    1: (120, 255, 120),  # right: green-ish
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test one clip's motion-stage cam_space outputs")
    parser.add_argument("--seq-folder", required=True, help="Processed clip folder containing tracks_*, cam_space, and extracted_images")
    parser.add_argument("--mano-dir", default=None, help="Optional MANO model directory override")
    parser.add_argument("--device", default="cpu", help="Torch device for MANO forward, e.g. cpu or cuda:0")
    parser.add_argument("--chunk-limit", type=int, default=8, help="How many cam_space chunks per hand to print detailed examples for")
    parser.add_argument("--render-out", default=None, help="Optional output video path; defaults to <seq_folder>/motion_smoke_test.mp4")
    parser.add_argument("--render-max-frames", type=int, default=240, help="Maximum number of frames to render")
    parser.add_argument("--render-fps", type=int, default=15, help="FPS for the output video")
    return parser


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


def _hand_name(hand_idx: int) -> str:
    return "left" if int(hand_idx) == 0 else "right"


def _resolve_image_paths(seq_folder: Path) -> list[Path]:
    extracted_dir = seq_folder / "extracted_images"
    if not extracted_dir.is_dir():
        raise FileNotFoundError(f"extracted_images not found under {seq_folder}")
    image_paths = sorted(extracted_dir.glob("*.jpg"))
    if not image_paths:
        image_paths = sorted(extracted_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg/.png frames found under {extracted_dir}")
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
            chunk = {
                "key": key,
                "frame_chunk": frame_chunk,
                "frame_start": int(frame_chunk[0]),
                "frame_end": int(frame_chunk[-1]),
                "num_frames": int(frame_chunk.size),
                "init_trans": np.asarray(payload["init_trans"], dtype=np.float32),
                "init_root_orient": np.asarray(payload["init_root_orient"], dtype=np.float32),
                "init_hand_pose": np.asarray(payload["init_hand_pose"], dtype=np.float32),
                "init_betas": np.asarray(payload["init_betas"], dtype=np.float32),
            }
            results[hand_idx].append(chunk)
    return results


def _summarize_cam_space(cam_chunks: dict[int, list[dict]], chunk_limit: int) -> None:
    print("\n=== Motion cam_space summary ===")
    for hand_idx in (0, 1):
        hand_name = _hand_name(hand_idx)
        chunks = cam_chunks[hand_idx]
        print(f"[{hand_name}] chunks={len(chunks)}")
        if not chunks:
            continue

        all_trans = np.concatenate([chunk["init_trans"].reshape(-1, 3) for chunk in chunks], axis=0)
        all_root = np.concatenate([chunk["init_root_orient"].reshape(-1, 3, 3) for chunk in chunks], axis=0)
        root_dets = np.linalg.det(all_root.astype(np.float64))
        ortho_err = np.linalg.norm(
            np.matmul(np.transpose(all_root, (0, 2, 1)), all_root) - np.eye(3, dtype=np.float32),
            axis=(1, 2),
        )
        print(f"  init_trans xyz range={_format_range(all_trans)}")
        print(f"  init_trans norm range={_format_norm_range(all_trans)}")
        print(f"  root_rot det range={_format_range(root_dets)}")
        print(f"  root_rot ortho_err range={_format_range(ortho_err)}")

        for example in chunks[: max(0, int(chunk_limit))]:
            example_trans = example["init_trans"].reshape(-1, 3)
            print(
                "  "
                f"chunk={example['key']} frames={example['num_frames']} "
                f"trans_norm={_format_norm_range(example_trans)}"
            )


def _project_cam_points(points_cam: np.ndarray, focal: float, center_xy: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_cam, dtype=np.float32)
    z = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (z > 1e-6)
    uv = np.zeros((pts.shape[0], 2), dtype=np.float32)
    uv[valid, 0] = pts[valid, 0] / z[valid] * float(focal) + float(center_xy[0])
    uv[valid, 1] = pts[valid, 1] / z[valid] * float(focal) + float(center_xy[1])
    return uv, valid


def _draw_hand_overlay(image_bgr: np.ndarray, joints_cam: np.ndarray, hand_idx: int, focal: float, center_xy: tuple[float, float]) -> None:
    color = HAND_COLORS[int(hand_idx)]
    uv, valid = _project_cam_points(joints_cam, focal, center_xy)
    h, w = image_bgr.shape[:2]
    in_frame = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)

    for chain in MANO_JOINT_TREE:
        for j1, j2 in chain:
            if in_frame[j1] and in_frame[j2]:
                cv2.line(
                    image_bgr,
                    tuple(uv[j1].astype(np.int32)),
                    tuple(uv[j2].astype(np.int32)),
                    color,
                    2,
                )
    for joint_idx, pt in enumerate(uv.astype(np.int32)):
        if in_frame[joint_idx]:
            radius = 5 if joint_idx == 0 else 3
            cv2.circle(image_bgr, tuple(pt), radius, color, -1)

    visible_points = int(in_frame.sum())
    cv2.putText(
        image_bgr,
        f"{_hand_name(hand_idx)} visible_joints={visible_points}/21",
        (12, 30 + 28 * int(hand_idx)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def _render_motion_video(
    seq_folder: Path,
    cam_chunks: dict[int, list[dict]],
    *,
    mano_dir: str | None,
    device: str,
    render_out: Path,
    render_max_frames: int,
    render_fps: int,
) -> dict:
    import torch

    image_paths = _resolve_image_paths(seq_folder)
    render_frame_count = min(len(image_paths), max(1, int(render_max_frames)))
    first_frame = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read frame: {image_paths[0]}")
    height, width = first_frame.shape[:2]
    focal = _resolve_focal(seq_folder)
    center_xy = (width / 2.0, height / 2.0)

    torch_device = torch.device(device)
    mano_right, mano_left = build_mano_models(torch_device, mano_dir=mano_dir)

    frame_predictions: dict[int, dict[int, np.ndarray]] = {}
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
                    outputs = run_mano_left(
                        init_trans,
                        root_aa,
                        hand_aa,
                        betas=init_betas,
                        mano_model=mano_left,
                    )
                else:
                    outputs = run_mano(
                        init_trans,
                        root_aa,
                        hand_aa,
                        betas=init_betas,
                        mano_model=mano_right,
                    )
            joints = outputs["joints"][0].detach().cpu().numpy().astype(np.float32)
            for local_idx, frame_idx in enumerate(frame_chunk.tolist()):
                if int(frame_idx) >= render_frame_count:
                    continue
                frame_predictions.setdefault(int(frame_idx), {})[int(hand_idx)] = joints[local_idx]

    render_out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(render_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(render_fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {render_out}")

    frames_with_prediction = 0
    try:
        for frame_idx in range(render_frame_count):
            image = cv2.imread(str(image_paths[frame_idx]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read frame: {image_paths[frame_idx]}")
            hand_preds = frame_predictions.get(frame_idx, {})
            if hand_preds:
                frames_with_prediction += 1
            for hand_idx, joints_cam in sorted(hand_preds.items()):
                _draw_hand_overlay(image, joints_cam, hand_idx, focal, center_xy)
            cv2.putText(
                image,
                f"frame={frame_idx}",
                (12, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(image)
    finally:
        writer.release()

    return {
        "render_out": str(render_out),
        "rendered_frames": int(render_frame_count),
        "frames_with_prediction": int(frames_with_prediction),
        "focal": float(focal),
        "image_size": [int(width), int(height)],
    }


def main() -> None:
    args = build_parser().parse_args()
    seq_folder = Path(args.seq_folder).expanduser().resolve()
    if not seq_folder.is_dir():
        raise FileNotFoundError(f"seq_folder not found: {seq_folder}")

    print(f"seq_folder: {seq_folder}")
    cam_chunks = _load_cam_space_chunks(seq_folder)
    _summarize_cam_space(cam_chunks, args.chunk_limit)

    render_out = Path(args.render_out).expanduser().resolve() if args.render_out else (seq_folder / "motion_smoke_test.mp4")
    render_summary = _render_motion_video(
        seq_folder,
        cam_chunks,
        mano_dir=args.mano_dir,
        device=args.device,
        render_out=render_out,
        render_max_frames=args.render_max_frames,
        render_fps=args.render_fps,
    )

    print("\n=== Motion visualization ===")
    print(f"render_out={render_summary['render_out']}")
    print(
        f"rendered_frames={render_summary['rendered_frames']} "
        f"frames_with_prediction={render_summary['frames_with_prediction']}"
    )
    print(f"focal={render_summary['focal']:.6g} image_size={render_summary['image_size']}")


if __name__ == "__main__":
    main()
