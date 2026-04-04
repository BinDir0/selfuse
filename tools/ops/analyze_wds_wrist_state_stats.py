#!/usr/bin/env python3
"""Compute wrist-state statistics from WebDataset lowdim payloads."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.webdataset_rewriter import iter_shard_paths, iter_shard_samples, validate_sample_record
from lib.pipeline.quality_metrics import parse_frame_index

LOWDIM_SIZE = 116
WORLD_STATE_SLICE = slice(0, 18)
EXTRINSIC_SLICE = slice(96, 112)

STATE_DIM_NAMES = [
    "left_wrist_x",
    "left_wrist_y",
    "left_wrist_z",
    "right_wrist_x",
    "right_wrist_y",
    "right_wrist_z",
    "left_root_rot6d_0",
    "left_root_rot6d_1",
    "left_root_rot6d_2",
    "left_root_rot6d_3",
    "left_root_rot6d_4",
    "left_root_rot6d_5",
    "right_root_rot6d_0",
    "right_root_rot6d_1",
    "right_root_rot6d_2",
    "right_root_rot6d_3",
    "right_root_rot6d_4",
    "right_root_rot6d_5",
]


def build_parser():
    parser = argparse.ArgumentParser(description="Analyze 18D wrist-state statistics from WebDataset lowdim.npy payloads")
    parser.add_argument("--input", required=True, help="Input shard tar file or directory containing shard tar files")
    parser.add_argument("--sample-offset", type=int, default=0, help="Skip the first N valid samples before collecting stats")
    parser.add_argument("--sample-limit", type=int, default=None, help="Analyze at most N valid samples after offset")
    parser.add_argument("--shard-start", type=int, default=0, help="Inclusive shard index in sorted shard order")
    parser.add_argument("--shard-end", type=int, default=None, help="Exclusive shard index in sorted shard order")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    return parser


def resolve_tar_paths(input_path: str) -> list[str]:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        return [str(path)]
    return list(iter_shard_paths(str(path)))


def rot6d_to_rotmat_np(rot6d: np.ndarray) -> np.ndarray:
    array = np.asarray(rot6d, dtype=np.float32).reshape(-1, 6)
    view = array.reshape(-1, 3, 2)
    a1 = view[:, :, 0]
    a2 = view[:, :, 1]
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)
    proj = np.sum(b1 * a2, axis=1, keepdims=True)
    b2 = a2 - proj * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def rotmat_to_rot6d_np(rotmat: np.ndarray) -> np.ndarray:
    array = np.asarray(rotmat, dtype=np.float32).reshape(-1, 3, 3)
    return array[:, :, :2].reshape(-1, 6).astype(np.float32)


def transform_wrist_state_to_camera(world_state: np.ndarray, extrinsic_w2c: np.ndarray) -> np.ndarray:
    state = np.asarray(world_state, dtype=np.float32).reshape(18)
    w2c = np.asarray(extrinsic_w2c, dtype=np.float32).reshape(4, 4)
    rot_w2c = w2c[:3, :3]
    trans_w2c = w2c[:3, 3]

    left_pos_world = state[0:3]
    right_pos_world = state[3:6]
    left_rot_world = rot6d_to_rotmat_np(state[6:12])[0]
    right_rot_world = rot6d_to_rotmat_np(state[12:18])[0]

    left_pos_cam = (rot_w2c @ left_pos_world) + trans_w2c
    right_pos_cam = (rot_w2c @ right_pos_world) + trans_w2c
    left_rot_cam = rot_w2c @ left_rot_world
    right_rot_cam = rot_w2c @ right_rot_world

    return np.concatenate(
        [
            left_pos_cam.astype(np.float32),
            right_pos_cam.astype(np.float32),
            rotmat_to_rot6d_np(left_rot_cam)[0],
            rotmat_to_rot6d_np(right_rot_cam)[0],
        ],
        axis=0,
    ).astype(np.float32)


def load_lowdim_from_sample(sample: dict) -> np.ndarray:
    if sample.get("lowdim_bytes") is None:
        raise ValueError(f"Sample {sample['key']} is missing lowdim.npy")
    array = np.load(io.BytesIO(sample["lowdim_bytes"]), allow_pickle=False).astype(np.float32).reshape(-1)
    if array.shape[0] != LOWDIM_SIZE:
        raise ValueError(f"Sample {sample['key']} expected lowdim[116], got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Sample {sample['key']} has non-finite lowdim values")
    return array


def compute_stats(values: np.ndarray) -> dict:
    return {
        "count": int(values.shape[0]),
        "dim_names": STATE_DIM_NAMES,
        "min": values.min(axis=0).astype(np.float32).tolist(),
        "max": values.max(axis=0).astype(np.float32).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(np.float32).tolist(),
    }


def format_stats_lines(name: str, stats: dict) -> list[str]:
    lines = [f"{name}:"]
    for field in ("min", "max", "q01", "q99"):
        lines.append(f"  {field}:")
        for dim_name, value in zip(stats["dim_names"], stats[field]):
            lines.append(f"    {dim_name:20s} {value: .6f}")
    return lines


def main():
    args = build_parser().parse_args()
    tar_paths = resolve_tar_paths(args.input)
    tar_paths = tar_paths[args.shard_start:args.shard_end]
    if not tar_paths:
        raise SystemExit("No shard tar files matched the requested shard range.")

    world_states = []
    camera_states = []
    total_seen = 0
    total_valid = 0

    for shard_path in tar_paths:
        for sample in iter_shard_samples(shard_path):
            total_seen += 1
            validate_sample_record(sample)
            lowdim = load_lowdim_from_sample(sample)

            if total_valid < args.sample_offset:
                total_valid += 1
                continue
            if args.sample_limit is not None and len(world_states) >= args.sample_limit:
                break

            world_state = lowdim[WORLD_STATE_SLICE].astype(np.float32)
            extrinsic = lowdim[EXTRINSIC_SLICE].reshape(4, 4).astype(np.float32)
            camera_state = transform_wrist_state_to_camera(world_state, extrinsic)
            world_states.append(world_state)
            camera_states.append(camera_state)
            total_valid += 1
        if args.sample_limit is not None and len(world_states) >= args.sample_limit:
            break

    if not world_states:
        raise SystemExit("No valid samples were collected for statistics.")

    world_array = np.stack(world_states, axis=0)
    camera_array = np.stack(camera_states, axis=0)

    payload = {
        "input": str(Path(args.input).expanduser().resolve()),
        "shards_analyzed": len(tar_paths),
        "samples_seen": total_seen,
        "samples_used": int(world_array.shape[0]),
        "sample_offset": int(args.sample_offset),
        "sample_limit": None if args.sample_limit is None else int(args.sample_limit),
        "state_definition": {
            "world_18d": "left_wrist_xyz + right_wrist_xyz + left_root_rot6d + right_root_rot6d",
            "camera_18d": "same 18D state transformed by lowdim camera_w2c",
        },
        "world_18d": compute_stats(world_array),
        "camera_18d": compute_stats(camera_array),
    }

    lines = [
        f"Input: {payload['input']}",
        f"Shards analyzed: {payload['shards_analyzed']}",
        f"Samples seen: {payload['samples_seen']}",
        f"Samples used: {payload['samples_used']}",
        f"Sample offset: {payload['sample_offset']}",
        f"Sample limit: {payload['sample_limit']}",
        "",
    ]
    lines.extend(format_stats_lines("world_18d", payload["world_18d"]))
    lines.append("")
    lines.extend(format_stats_lines("camera_18d", payload["camera_18d"]))
    print("\n".join(lines), flush=True)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
