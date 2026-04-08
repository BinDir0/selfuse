#!/usr/bin/env python3
"""Rerun viewer for teleop-style zarr datasets with MANO PCA hand states."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.mano_codec import MANO_CENTER_IDX
from lib.pipeline.exporters.mano_codec import build_manopth_models, rot6d_to_axis_angle, run_manopth_mano
from tools.ops import webdataset_visualizer as wv
from tools.ops.rerun_webdataset_visualizer import (
    LEFT_IMAGE_ROOT,
    LEFT_WORLD_ROOT,
    MANO_FACE_EXTRA,
    RIGHT_IMAGE_ROOT,
    RIGHT_WORLD_ROOT,
    _log_camera,
    _log_empty_hand,
    _log_keypoint_hand,
    _log_mesh_hand,
    _log_skeleton_hand,
    configure_recording,
    load_rerun,
    log_static_scene,
    send_default_blueprint,
)


@dataclass
class ZarrEpisodeFrame:
    frame_idx: int
    image_rgb: np.ndarray
    intrinsic: np.ndarray
    c2w: np.ndarray
    presence: int
    instruction: str
    wrist: np.ndarray
    fingertips: np.ndarray
    mano: np.ndarray
    shape: np.ndarray


def _decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _decode_instruction_row(row, instruction_num: int) -> str:
    if isinstance(row, np.ndarray):
        values = [_decode_text(item).strip() for item in row.tolist()]
    elif isinstance(row, (list, tuple)):
        values = [_decode_text(item).strip() for item in row]
    else:
        return _decode_text(row).strip()
    count = max(0, min(int(instruction_num), len(values)))
    values = [item for item in values[:count] if item]
    return " | ".join(values)


def _zarr_open(path: str):
    try:
        import zarr
    except ImportError as error:
        raise SystemExit(
            "zarr is not installed. Install runtime dependencies with:\n"
            "  pip install zarr"
        ) from error
    return zarr.open(path, mode="r")


def _group_get(root, path: str):
    node = root
    for part in path.split("/"):
        node = node[part]
    return node


def _load_episode_catalog(root) -> tuple[list[str], np.ndarray]:
    raw_names = _group_get(root, "meta/episode_names")[:]
    episode_names = [_decode_text(item) for item in raw_names.tolist()]
    episode_ends = np.asarray(_group_get(root, "meta/episode_ends")[:], dtype=np.int64).reshape(-1)
    return episode_names, episode_ends


def _select_episode(
    episode_names: list[str],
    *,
    filter_key: str,
    episode_name: Optional[str],
    episode_index: Optional[int],
) -> tuple[int, str]:
    indexed = list(enumerate(episode_names))
    if filter_key:
        needle = filter_key.lower()
        indexed = [(idx, name) for idx, name in indexed if needle in name.lower()]
    if episode_name is not None:
        indexed = [(idx, name) for idx, name in indexed if name == episode_name]
    if not indexed:
        raise SystemExit("No episodes matched the current filters.")
    if episode_index is not None:
        if episode_index < 1 or episode_index > len(indexed):
            raise SystemExit(f"episode-index must be in [1, {len(indexed)}], got {episode_index}")
        return indexed[episode_index - 1]
    if len(indexed) == 1:
        return indexed[0]
    if not sys.stdin.isatty():
        names = ", ".join(name for _, name in indexed[:8])
        raise SystemExit(
            f"Matched {len(indexed)} episodes. Pass --episode-name or --episode-index. First matches: {names}"
        )
    print("Matched multiple episodes; choose one:", flush=True)
    for local_idx, (global_idx, name) in enumerate(indexed, start=1):
        print(f"  {local_idx}. episode={name} global_index={global_idx}", flush=True)
    while True:
        raw = input(f"Select episode [1-{len(indexed)}]: ").strip()
        if not raw:
            continue
        try:
            selected = int(raw)
        except ValueError:
            print("Please enter an integer index.", flush=True)
            continue
        if 1 <= selected <= len(indexed):
            return indexed[selected - 1]
        print(f"Index out of range: {selected}", flush=True)


def _episode_bounds(episode_ends: np.ndarray, episode_idx: int) -> tuple[int, int]:
    end = int(episode_ends[episode_idx])
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    return start, end


def _resolve_mano_runtime(mano_dir: Optional[str], mano_device: str):
    import torch

    requested = mano_device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"MANO device {requested} requested but CUDA is unavailable; falling back to cpu.", flush=True)
        requested = "cpu"
    device = torch.device(requested)
    mano_right, mano_left = build_manopth_models(
        device,
        mano_dir=mano_dir,
        center_idx=MANO_CENTER_IDX,
        flat_hand_mean=True,
        ncomps=45,
    )
    return {
        "device": device,
        "mano_right": mano_right,
        "mano_left": mano_left,
    }


def _compute_mano_frame(frame: ZarrEpisodeFrame, runtime: dict) -> dict:
    wrist = np.asarray(frame.wrist, dtype=np.float32).reshape(-1)
    if wrist.shape[0] != 18:
        raise ValueError(f"Expected wrist shape (18,), got {wrist.shape}")
    hand_pose = np.asarray(frame.mano, dtype=np.float32).reshape(-1)
    if hand_pose.shape[0] != 90:
        raise ValueError(f"Expected mano shape (90,), got {hand_pose.shape}")
    betas = np.asarray(frame.shape, dtype=np.float32).reshape(-1)
    if betas.shape[0] != 20:
        raise ValueError(f"Expected shape shape (20,), got {betas.shape}")

    left_trans = wrist[0:3]
    right_trans = wrist[3:6]
    left_root = rot6d_to_axis_angle(wrist[6:12])
    right_root = rot6d_to_axis_angle(wrist[12:18])
    left_pose_pca = hand_pose[0:45]
    right_pose_pca = hand_pose[45:90]
    left_betas = betas[0:10]
    right_betas = betas[10:20]

    left_verts, left_joints = run_manopth_mano(
        runtime["mano_left"],
        wrist_world=left_trans[None, :],
        root_rot_axis_angle=left_root[None, :],
        hand_pose_pca=left_pose_pca[None, :],
        betas=left_betas[None, :],
        device=runtime["device"],
    )
    right_verts, right_joints = run_manopth_mano(
        runtime["mano_right"],
        wrist_world=right_trans[None, :],
        root_rot_axis_angle=right_root[None, :],
        hand_pose_pca=right_pose_pca[None, :],
        betas=right_betas[None, :],
        device=runtime["device"],
    )
    return {
        "c2w": frame.c2w,
        "intrinsic": frame.intrinsic,
        "left_verts": left_verts[0].astype(np.float32),
        "left_joints": left_joints[0].astype(np.float32),
        "right_verts": right_verts[0].astype(np.float32),
        "right_joints": right_joints[0].astype(np.float32),
    }


def _build_keypoint_frame(frame: ZarrEpisodeFrame) -> dict:
    wrist = np.asarray(frame.wrist, dtype=np.float32).reshape(-1)
    fingertips = np.asarray(frame.fingertips, dtype=np.float32).reshape(-1)
    if wrist.shape[0] != 18:
        raise ValueError(f"Expected wrist shape (18,), got {wrist.shape}")
    if fingertips.shape[0] != 30:
        raise ValueError(f"Expected fingertips shape (30,), got {fingertips.shape}")
    return {
        "c2w": frame.c2w,
        "intrinsic": frame.intrinsic,
        "left_wrist": wrist[0:3].astype(np.float32),
        "right_wrist": wrist[3:6].astype(np.float32),
        "left_tips": fingertips[0:15].reshape(5, 3).astype(np.float32),
        "right_tips": fingertips[15:30].reshape(5, 3).astype(np.float32),
    }


def _load_episode_frames(
    root,
    *,
    episode_idx: int,
    pose_source: str,
    frame_step: int,
    max_frames: Optional[int],
) -> tuple[str, list[ZarrEpisodeFrame]]:
    episode_names, episode_ends = _load_episode_catalog(root)
    episode_name = episode_names[episode_idx]
    start, end = _episode_bounds(episode_ends, episode_idx)

    indices = np.arange(start, end, max(1, int(frame_step)), dtype=np.int64)
    if max_frames is not None:
        indices = indices[: max(0, int(max_frames))]
    if indices.size == 0:
        return episode_name, []

    image = np.asarray(_group_get(root, "data/image")[indices], dtype=np.uint8)
    extrinsic = np.asarray(_group_get(root, "data/extrinsic")[indices], dtype=np.float32)
    intrinsic = np.asarray(_group_get(root, "data/intrinsic")[indices], dtype=np.float32)
    presence = np.asarray(_group_get(root, "data/presence")[indices], dtype=np.int64)
    instruction = _group_get(root, "data/instruction")[indices]
    instruction_num = np.asarray(_group_get(root, "data/instruction_num")[indices], dtype=np.int64)
    wrist = np.asarray(_group_get(root, f"data/{pose_source}/wrist")[indices], dtype=np.float32)
    fingertips = np.asarray(_group_get(root, f"data/{pose_source}/fingertips")[indices], dtype=np.float32)
    mano = np.asarray(_group_get(root, f"data/{pose_source}/mano")[indices], dtype=np.float32)
    shape = np.asarray(_group_get(root, f"data/{pose_source}/shape")[indices], dtype=np.float32)

    frames: list[ZarrEpisodeFrame] = []
    for local_idx, global_idx in enumerate(indices.tolist()):
        c2w, _ = wv._resolve_camera_c2w(extrinsic[local_idx])
        frames.append(
            ZarrEpisodeFrame(
                frame_idx=int(global_idx - start),
                image_rgb=image[local_idx],
                intrinsic=intrinsic[local_idx].reshape(4).astype(np.float32),
                c2w=np.asarray(c2w, dtype=np.float32),
                presence=int(presence[local_idx]),
                instruction=_decode_instruction_row(instruction[local_idx], int(instruction_num[local_idx])),
                wrist=wrist[local_idx].reshape(-1).astype(np.float32),
                fingertips=fingertips[local_idx].reshape(-1).astype(np.float32),
                mano=mano[local_idx].reshape(-1).astype(np.float32),
                shape=shape[local_idx].reshape(-1).astype(np.float32),
            )
        )
    return episode_name, frames


def log_episode(rr, frames: list[ZarrEpisodeFrame], render_mode: str, runtime: Optional[dict]):
    faces_right = None
    faces_left = None
    if render_mode == "mesh":
        from hawor.utils.process import get_mano_faces

        faces_right = np.concatenate([get_mano_faces(), MANO_FACE_EXTRA], axis=0).astype(np.uint32)
        faces_left = faces_right[:, [0, 2, 1]]

    for index, frame in enumerate(frames):
        rr.set_time("frame_idx", sequence=index)
        rr.set_time("source_frame_idx", sequence=int(frame.frame_idx))
        _log_camera(rr, frame.image_rgb, frame.c2w, frame.intrinsic)

        left_present, right_present = wv._presence_flags(frame.presence)

        if render_mode == "keypoint":
            keypoint_frame = _build_keypoint_frame(frame)
            left_points = np.concatenate([keypoint_frame["left_wrist"][None, :], keypoint_frame["left_tips"]], axis=0)
            right_points = np.concatenate([keypoint_frame["right_wrist"][None, :], keypoint_frame["right_tips"]], axis=0)
            if left_present:
                _log_keypoint_hand(rr, "left", LEFT_WORLD_ROOT, LEFT_IMAGE_ROOT, left_points, frame.c2w, frame.intrinsic, frame.image_rgb.shape)
            else:
                _log_empty_hand(rr, LEFT_WORLD_ROOT, LEFT_IMAGE_ROOT, with_mesh=False)
            if right_present:
                _log_keypoint_hand(rr, "right", RIGHT_WORLD_ROOT, RIGHT_IMAGE_ROOT, right_points, frame.c2w, frame.intrinsic, frame.image_rgb.shape)
            else:
                _log_empty_hand(rr, RIGHT_WORLD_ROOT, RIGHT_IMAGE_ROOT, with_mesh=False)
            continue

        if runtime is None:
            raise ValueError("MANO runtime is required for skeleton/mesh modes")
        mano_frame = _compute_mano_frame(frame, runtime)
        if render_mode == "skeleton":
            if left_present:
                _log_skeleton_hand(rr, "left", LEFT_WORLD_ROOT, LEFT_IMAGE_ROOT, mano_frame["left_joints"], frame.c2w, frame.intrinsic, frame.image_rgb.shape)
            else:
                _log_empty_hand(rr, LEFT_WORLD_ROOT, LEFT_IMAGE_ROOT, with_mesh=False)
            if right_present:
                _log_skeleton_hand(rr, "right", RIGHT_WORLD_ROOT, RIGHT_IMAGE_ROOT, mano_frame["right_joints"], frame.c2w, frame.intrinsic, frame.image_rgb.shape)
            else:
                _log_empty_hand(rr, RIGHT_WORLD_ROOT, RIGHT_IMAGE_ROOT, with_mesh=False)
            continue

        if left_present:
            _log_mesh_hand(
                rr,
                "left",
                LEFT_WORLD_ROOT,
                LEFT_IMAGE_ROOT,
                mano_frame["left_verts"],
                mano_frame["left_joints"],
                faces_left,
                frame.c2w,
                frame.intrinsic,
                frame.image_rgb.shape,
            )
        else:
            _log_empty_hand(rr, LEFT_WORLD_ROOT, LEFT_IMAGE_ROOT, with_mesh=True)
        if right_present:
            _log_mesh_hand(
                rr,
                "right",
                RIGHT_WORLD_ROOT,
                RIGHT_IMAGE_ROOT,
                mano_frame["right_verts"],
                mano_frame["right_joints"],
                faces_right,
                frame.c2w,
                frame.intrinsic,
                frame.image_rgb.shape,
            )
        else:
            _log_empty_hand(rr, RIGHT_WORLD_ROOT, RIGHT_IMAGE_ROOT, with_mesh=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Rerun viewer for teleop-style zarr episodes")
    parser.add_argument("--input", required=True, help="Path to the zarr root directory")
    parser.add_argument("--filter-key", default="", help="Substring filter on episode_names")
    parser.add_argument("--episode-name", type=str, default=None, help="Exact episode name to open")
    parser.add_argument("--episode-index", type=int, default=None, help="1-based episode index after filters are applied")
    parser.add_argument("--pose-source", default="state", choices=["state", "action"], help="Whether to replay current state or next-step action tensors")
    parser.add_argument("--frame-step", type=int, default=1, help="Replay every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on loaded frames after frame-step")
    parser.add_argument("--render-mode", default="keypoint", choices=["keypoint", "skeleton", "mesh"], help="Visualization mode")
    parser.add_argument("--output-mode", default="online", choices=["online", "offline", "both"], help="Send recording to a live viewer, offline .rrd, or both")
    parser.add_argument("--rrd-out", type=str, default=None, help="Offline .rrd output path")
    parser.add_argument("--mano-dir", type=str, default=None, help="Optional MANO model directory override")
    parser.add_argument("--mano-device", type=str, default="cpu", help="Device for MANO replay, e.g. cpu or cuda:0")
    return parser


def main():
    args = build_parser().parse_args()
    root = _zarr_open(args.input)
    episode_names, _episode_ends = _load_episode_catalog(root)
    selected_episode_idx, selected_episode_name = _select_episode(
        episode_names,
        filter_key=args.filter_key,
        episode_name=args.episode_name,
        episode_index=args.episode_index,
    )
    selected_episode_name, frames = _load_episode_frames(
        root,
        episode_idx=selected_episode_idx,
        pose_source=args.pose_source,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
    )
    if not frames:
        raise SystemExit(f"No frames found for episode {selected_episode_name}")

    rr = load_rerun()
    rrd_path = configure_recording(
        rr,
        output_mode=args.output_mode,
        rrd_out=args.rrd_out,
        episode_key=selected_episode_name,
        render_mode=args.render_mode,
    )
    send_default_blueprint(rr)
    log_static_scene(rr)
    runtime = None if args.render_mode == "keypoint" else _resolve_mano_runtime(args.mano_dir, args.mano_device)
    log_episode(rr, frames, args.render_mode, runtime)

    print(
        f"Loaded zarr={Path(args.input).expanduser().resolve()} "
        f"episode={selected_episode_name} frames={len(frames)} "
        f"pose_source={args.pose_source} render_mode={args.render_mode}",
        flush=True,
    )
    if frames[0].instruction:
        print(f"Instruction preview: {frames[0].instruction}", flush=True)
    if rrd_path is not None:
        print(f"Saved Rerun recording to: {rrd_path}", flush=True)


if __name__ == "__main__":
    main()
