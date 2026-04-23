#!/usr/bin/env python3
"""Smoke-test one processed clip across motion, infiller, and lowdim export semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from lib.pipeline.exporters.webdataset_features import (
    _build_lowdim_features,
    _compute_joint_states,
    _load_episode_camera_features,
    _load_world_space_prediction,
    build_mano_models,
)
from lib.pipeline.quality_metrics import (
    LEFT_HAND_TRANSLATION_SLICE,
    RIGHT_HAND_TRANSLATION_SLICE,
    LEFT_ROOT_ROT6D_SLICE,
    RIGHT_ROOT_ROT6D_SLICE,
    validate_lowdim_numeric_sanity,
)
from lib.pipeline.stage_api import get_track_range

WRIST_STATE_SLICE = slice(0, 18)
HAND_STATE_SLICE = slice(18, 48)
WRIST_ACTION_SLICE = slice(48, 66)
HAND_ACTION_SLICE = slice(66, 96)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test one clip's motion/cam-space/world-space/lowdim outputs")
    parser.add_argument("--seq-folder", required=True, help="Processed clip folder containing tracks_*, cam_space, SLAM, world_space_res.pth")
    parser.add_argument("--mano-dir", default=None, help="Optional MANO model directory override")
    parser.add_argument("--device", default="cpu", help="Torch device for MANO forward, e.g. cpu or cuda:0")
    parser.add_argument("--chunk-limit", type=int, default=8, help="How many cam_space chunks per hand to print detailed examples for")
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


def _summarize_world_space(seq_folder: Path) -> dict:
    print("\n=== Infiller world_space_res summary ===")
    prediction = _load_world_space_prediction({"episode_id": seq_folder.name}, str(seq_folder / "world_space_res.pth"))
    if prediction is None:
        raise RuntimeError(f"Failed to load world_space_res.pth under {seq_folder}")

    pred_trans = prediction["pred_trans"].cpu().numpy()
    pred_rot = prediction["pred_rot"].cpu().numpy()
    pred_hand_pose = prediction["pred_hand_pose"].cpu().numpy()
    pred_betas = prediction["pred_betas"].cpu().numpy()
    pred_valid = np.asarray(prediction["pred_valid"])
    print(f"num_frames={pred_trans.shape[1]}")
    for hand_idx in (0, 1):
        hand_name = _hand_name(hand_idx)
        valid = pred_valid[hand_idx] > 0.5
        valid_count = int(valid.sum())
        print(f"[{hand_name}] valid={valid_count}/{pred_valid.shape[1]}")
        if valid_count <= 0:
            continue
        hand_trans = pred_trans[hand_idx, valid]
        hand_rot = pred_rot[hand_idx, valid]
        hand_pose = pred_hand_pose[hand_idx, valid]
        print(f"  trans xyz range={_format_range(hand_trans)}")
        print(f"  trans norm range={_format_norm_range(hand_trans)}")
        print(f"  rot aa range={_format_range(hand_rot)}")
        print(f"  hand_pose aa range={_format_range(hand_pose)}")
        if valid_count > 1:
            steps = np.linalg.norm(hand_trans[1:] - hand_trans[:-1], axis=1)
            print(f"  trans step range={_format_range(steps)}")

    return {
        "pred_trans": pred_trans,
        "pred_rot": pred_rot,
        "pred_hand_pose": pred_hand_pose,
        "pred_betas": pred_betas,
        "pred_valid": pred_valid,
    }


def _summarize_lowdim(seq_folder: Path, prediction: dict, mano_dir: str | None, device: str) -> None:
    print("\n=== Export lowdim summary ===")
    import torch

    torch_device = torch.device(device)
    mano_right, mano_left = build_mano_models(torch_device, mano_dir=mano_dir)
    wrist_state, hand_state = _compute_joint_states(
        prediction["pred_trans"],
        prediction["pred_rot"],
        prediction["pred_hand_pose"],
        prediction["pred_betas"],
        mano_right,
        mano_left,
        torch_device,
    )
    num_frames = int(prediction["pred_trans"].shape[1])
    ep = {"crop_dir": str(seq_folder), "episode_id": seq_folder.name}
    extrinsics, intrinsic = _load_episode_camera_features(ep, num_frames)
    lowdim_all = _build_lowdim_features(wrist_state, hand_state, extrinsics, intrinsic)

    invalid_indices = []
    for frame_idx, lowdim in enumerate(lowdim_all):
        sanity = validate_lowdim_numeric_sanity(lowdim)
        if not sanity["valid"]:
            invalid_indices.append((frame_idx, sanity["issues"]))

    print(f"lowdim shape={lowdim_all.shape}")
    print(f"left wrist xyz range={_format_range(lowdim_all[:, LEFT_HAND_TRANSLATION_SLICE])}")
    print(f"right wrist xyz range={_format_range(lowdim_all[:, RIGHT_HAND_TRANSLATION_SLICE])}")
    print(f"left rot6d range={_format_range(lowdim_all[:, LEFT_ROOT_ROT6D_SLICE])}")
    print(f"right rot6d range={_format_range(lowdim_all[:, RIGHT_ROOT_ROT6D_SLICE])}")
    print(f"invalid lowdim frames={len(invalid_indices)}")
    if invalid_indices:
        for frame_idx, issues in invalid_indices[:10]:
            print(f"  frame={frame_idx} issues={','.join(issues)}")

    if lowdim_all.shape[0] > 1:
        wrist_action = lowdim_all[:-1, WRIST_ACTION_SLICE]
        next_wrist_state = lowdim_all[1:, WRIST_STATE_SLICE]
        hand_action = lowdim_all[:-1, HAND_ACTION_SLICE]
        next_hand_state = lowdim_all[1:, HAND_STATE_SLICE]
        wrist_diff = np.abs(wrist_action - next_wrist_state)
        hand_diff = np.abs(hand_action - next_hand_state)
        print(f"wrist action-next_state max_abs_diff={float(wrist_diff.max()):.6g}")
        print(f"hand action-next_state max_abs_diff={float(hand_diff.max()):.6g}")


def main() -> None:
    args = build_parser().parse_args()
    seq_folder = Path(args.seq_folder).expanduser().resolve()
    if not seq_folder.is_dir():
        raise FileNotFoundError(f"seq_folder not found: {seq_folder}")

    print(f"seq_folder: {seq_folder}")
    cam_chunks = _load_cam_space_chunks(seq_folder)
    _summarize_cam_space(cam_chunks, args.chunk_limit)
    world_numpy = _summarize_world_space(seq_folder)

    import torch

    prediction = {
        "pred_trans": torch.from_numpy(world_numpy["pred_trans"]).float(),
        "pred_rot": torch.from_numpy(world_numpy["pred_rot"]).float(),
        "pred_hand_pose": torch.from_numpy(world_numpy["pred_hand_pose"]).float(),
        "pred_betas": torch.from_numpy(world_numpy["pred_betas"]).float(),
        "pred_valid": world_numpy["pred_valid"],
    }
    _summarize_lowdim(seq_folder, prediction, args.mano_dir, args.device)


if __name__ == "__main__":
    main()
