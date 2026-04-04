#!/usr/bin/env python3
"""Compute wrist-state statistics from one or more WebDataset collections."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.webdataset_rewriter import iter_shard_paths, iter_shard_samples, validate_sample_record

LOWDIM_SIZE = 116
WORLD_STATE_SLICE = slice(0, 18)
NEXT_WORLD_STATE_SLICE = slice(48, 66)
EXTRINSIC_SLICE = slice(96, 112)
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))
DEPTH_EPS = 1e-6
ACTION_PRESENT_EPS = 1e-8

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
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input shard tar file or directory containing shard tar files. Repeat this flag for multiple WDS roots.",
    )
    parser.add_argument("--input-list", type=str, default=None, help="Optional text file with one input path per line")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10000,
        help="Total random sampling budget across all matched shards.",
    )
    parser.add_argument("--shard-start", type=int, default=0, help="Inclusive shard index in sorted shard order")
    parser.add_argument("--shard-end", type=int, default=None, help="Exclusive shard index in sorted shard order")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shard-budget allocation and within-shard sampling")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel shard workers for random mode")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    return parser


def resolve_requested_inputs(cli_inputs: list[str], input_list: str | None) -> list[str]:
    inputs = [str(Path(item).expanduser().resolve()) for item in (cli_inputs or [])]
    if input_list:
        list_path = Path(input_list).expanduser().resolve()
        for line in list_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            inputs.append(str(Path(stripped).expanduser().resolve()))
    deduped = []
    seen = set()
    for item in inputs:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def resolve_tar_paths(input_paths: list[str]) -> list[str]:
    tar_paths = []
    seen = set()
    for input_path in input_paths:
        path = Path(input_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        if path.is_file() and path.suffix != ".tar":
            raise ValueError(f"Expected a .tar shard file, got: {path}")
        candidates = [str(path)] if path.is_file() else list(iter_shard_paths(str(path)))
        for tar_path in candidates:
            if tar_path not in seen:
                seen.add(tar_path)
                tar_paths.append(tar_path)
    return sorted(tar_paths)


def rot6d_to_rotmat_np(rot6d: np.ndarray) -> np.ndarray:
    array = np.asarray(rot6d, dtype=np.float32).reshape(-1, 6)
    a1 = array[:, 0:3]
    a2 = array[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)
    proj = np.sum(b1 * a2, axis=1, keepdims=True)
    b2 = a2 - proj * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def rotmat_to_rot6d_np(rotmat: np.ndarray) -> np.ndarray:
    array = np.asarray(rotmat, dtype=np.float32).reshape(-1, 3, 3)
    return array[:, :, :2].transpose(0, 2, 1).reshape(-1, 6).astype(np.float32)


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


def sample_to_states_and_action(sample: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    validate_sample_record(sample)
    lowdim = load_lowdim_from_sample(sample)
    world_state = lowdim[WORLD_STATE_SLICE].astype(np.float32)
    next_world_state = lowdim[NEXT_WORLD_STATE_SLICE].astype(np.float32)
    extrinsic = lowdim[EXTRINSIC_SLICE].reshape(4, 4).astype(np.float32)
    camera_state = transform_wrist_state_to_camera(world_state, extrinsic)
    if np.max(np.abs(next_world_state)) <= ACTION_PRESENT_EPS:
        return world_state, camera_state, None, None

    next_camera_state = transform_wrist_state_to_camera(next_world_state, extrinsic)
    world_action = (next_world_state - world_state).astype(np.float32)
    camera_action = (next_camera_state - camera_state).astype(np.float32)
    return world_state, camera_state, world_action, camera_action


def allocate_shard_budgets(num_shards: int, total_samples: int, seed: int) -> np.ndarray:
    if num_shards <= 0:
        return np.zeros((0,), dtype=np.int64)
    if total_samples <= 0:
        raise ValueError("--sample-limit must be > 0")
    rng = np.random.default_rng(seed)
    return rng.multinomial(total_samples, np.full(num_shards, 1.0 / num_shards, dtype=np.float64))


def sample_shard(task: tuple[str, int, int]) -> dict:
    shard_path, budget, seed = task
    rng = np.random.default_rng(seed)
    world_candidates: list[np.ndarray] = []
    camera_candidates: list[np.ndarray] = []
    world_action_candidates: list[np.ndarray | None] = []
    camera_action_candidates: list[np.ndarray | None] = []
    samples_seen = 0
    valid_samples = 0
    invalid_samples = 0
    action_valid_samples = 0

    for sample in iter_shard_samples(shard_path):
        samples_seen += 1
        try:
            world_state, camera_state, world_action, camera_action = sample_to_states_and_action(sample)
        except Exception:
            invalid_samples += 1
            continue

        valid_samples += 1
        world_candidates.append(world_state)
        camera_candidates.append(camera_state)
        world_action_candidates.append(world_action)
        camera_action_candidates.append(camera_action)
        if world_action is not None:
            action_valid_samples += 1

    world_samples: list[np.ndarray] = []
    camera_samples: list[np.ndarray] = []
    world_action_samples: list[np.ndarray] = []
    camera_action_samples: list[np.ndarray] = []
    if budget > 0 and world_candidates:
        # Sample with replacement so each shard can honor its pre-allocated budget.
        sample_indices = rng.integers(0, len(world_candidates), size=budget)
        world_samples = [world_candidates[idx] for idx in sample_indices]
        camera_samples = [camera_candidates[idx] for idx in sample_indices]
        world_action_samples = [world_action_candidates[idx] for idx in sample_indices if world_action_candidates[idx] is not None]
        camera_action_samples = [camera_action_candidates[idx] for idx in sample_indices if camera_action_candidates[idx] is not None]

    return {
        "shard_path": shard_path,
        "budget": int(budget),
        "samples_seen": int(samples_seen),
        "valid_samples": int(valid_samples),
        "action_valid_samples": int(action_valid_samples),
        "invalid_samples": int(invalid_samples),
        "world_states": [state.tolist() for state in world_samples],
        "camera_states": [state.tolist() for state in camera_samples],
        "world_actions": [action.tolist() for action in world_action_samples],
        "camera_actions": [action.tolist() for action in camera_action_samples],
    }


def collect_states_random(
    tar_paths: list[str], *, sample_limit: int, seed: int, workers: int, show_progress: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    budgets = allocate_shard_budgets(len(tar_paths), sample_limit, seed)
    tasks = [(tar_paths[idx], int(budget), seed + idx + 1) for idx, budget in enumerate(budgets) if int(budget) > 0]

    world_states = []
    camera_states = []
    world_actions = []
    camera_actions = []
    samples_seen = 0
    valid_samples = 0
    action_valid_samples = 0
    invalid_samples = 0
    shard_reports = []

    if tqdm is None:
        show_progress = False

    progress = tqdm(
        total=len(tasks),
        desc="Sampling shards",
        unit="shard",
        disable=(not show_progress) or len(tasks) == 0,
    ) if tqdm is not None else None
    if workers <= 1 or len(tasks) <= 1:
        results_iter = (sample_shard(task) for task in tasks)
        for result in results_iter:
            samples_seen += result["samples_seen"]
            valid_samples += result["valid_samples"]
            action_valid_samples += result["action_valid_samples"]
            invalid_samples += result["invalid_samples"]
            shard_reports.append(
                {
                    "shard_path": result["shard_path"],
                    "budget": result["budget"],
                    "samples_seen": result["samples_seen"],
                    "valid_samples": result["valid_samples"],
                    "action_valid_samples": result["action_valid_samples"],
                    "invalid_samples": result["invalid_samples"],
                    "samples_selected": len(result["world_states"]),
                    "action_samples_selected": len(result["world_actions"]),
                }
            )
            world_states.extend(result["world_states"])
            camera_states.extend(result["camera_states"])
            world_actions.extend(result["world_actions"])
            camera_actions.extend(result["camera_actions"])
            if progress is not None:
                progress.update(1)
                progress.set_postfix(samples=len(world_states), actions=len(world_actions))
    else:
        with get_context("spawn").Pool(processes=min(workers, len(tasks))) as pool:
            for result in pool.imap_unordered(sample_shard, tasks):
                samples_seen += result["samples_seen"]
                valid_samples += result["valid_samples"]
                action_valid_samples += result["action_valid_samples"]
                invalid_samples += result["invalid_samples"]
                shard_reports.append(
                    {
                        "shard_path": result["shard_path"],
                        "budget": result["budget"],
                        "samples_seen": result["samples_seen"],
                        "valid_samples": result["valid_samples"],
                        "action_valid_samples": result["action_valid_samples"],
                        "invalid_samples": result["invalid_samples"],
                        "samples_selected": len(result["world_states"]),
                        "action_samples_selected": len(result["world_actions"]),
                    }
                )
                world_states.extend(result["world_states"])
                camera_states.extend(result["camera_states"])
                world_actions.extend(result["world_actions"])
                camera_actions.extend(result["camera_actions"])
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(samples=len(world_states), actions=len(world_actions))
    if progress is not None:
        progress.close()

    if not world_states:
        raise SystemExit("No valid samples were collected for statistics.")

    return (
        np.asarray(world_states, dtype=np.float32).reshape(-1, 18),
        np.asarray(camera_states, dtype=np.float32).reshape(-1, 18),
        np.asarray(world_actions, dtype=np.float32).reshape(-1, 18),
        np.asarray(camera_actions, dtype=np.float32).reshape(-1, 18),
        {
            "samples_seen": int(samples_seen),
            "valid_samples": int(valid_samples),
            "action_valid_samples": int(action_valid_samples),
            "invalid_samples": int(invalid_samples),
            "shard_reports": shard_reports,
            "requested_samples": int(sample_limit),
            "shards_sampled": len(tasks),
        },
    )


def compute_stats(values: np.ndarray) -> dict:
    return {
        "count": int(values.shape[0]),
        "dim_names": STATE_DIM_NAMES,
        "min": values.min(axis=0).astype(np.float32).tolist(),
        "max": values.max(axis=0).astype(np.float32).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(np.float32).tolist(),
    }


def compute_camera_depth_summary(camera_states: np.ndarray) -> dict:
    left_z = camera_states[:, 2]
    right_z = camera_states[:, 5]

    def summarize(z_values: np.ndarray) -> dict:
        return {
            "positive_ratio": float(np.mean(z_values > DEPTH_EPS)),
            "negative_ratio": float(np.mean(z_values < -DEPTH_EPS)),
            "near_zero_ratio": float(np.mean(np.abs(z_values) <= DEPTH_EPS)),
            "q01": float(np.quantile(z_values, 0.01)),
            "q50": float(np.quantile(z_values, 0.50)),
            "q99": float(np.quantile(z_values, 0.99)),
        }

    combined = np.concatenate([left_z, right_z], axis=0)
    return {
        "left_wrist_z": summarize(left_z),
        "right_wrist_z": summarize(right_z),
        "combined_wrist_z": summarize(combined),
    }


def format_stats_lines(name: str, stats: dict) -> list[str]:
    lines = [f"{name}:"]
    for field in ("min", "max", "q01", "q99"):
        lines.append(f"  {field}:")
        for dim_name, value in zip(stats["dim_names"], stats[field]):
            lines.append(f"    {dim_name:20s} {value: .6f}")
    return lines


def format_depth_summary_lines(summary: dict) -> list[str]:
    lines = ["camera_depth_summary:"]
    for name, item in summary.items():
        lines.append(f"  {name}:")
        for key in ("positive_ratio", "negative_ratio", "near_zero_ratio", "q01", "q50", "q99"):
            lines.append(f"    {key:16s} {item[key]: .6f}")
    return lines


def main():
    args = build_parser().parse_args()
    requested_inputs = resolve_requested_inputs(args.input, args.input_list)
    if not requested_inputs:
        raise SystemExit("Provide at least one --input or --input-list.")
    tar_paths = resolve_tar_paths(requested_inputs)
    tar_paths = tar_paths[args.shard_start:args.shard_end]
    if not tar_paths:
        raise SystemExit("No shard tar files matched the requested shard range.")

    world_array, camera_array, world_action_array, camera_action_array, collection_report = collect_states_random(
        tar_paths,
        sample_limit=args.sample_limit,
        seed=args.seed,
        workers=args.workers,
        show_progress=not args.no_progress,
    )

    payload = {
        "inputs": requested_inputs,
        "shards_analyzed": len(tar_paths),
        "sampling_mode": "random",
        "seed": int(args.seed),
        "workers": int(args.workers),
        "samples_seen": collection_report["samples_seen"],
        "valid_samples_seen": collection_report["valid_samples"],
        "action_valid_samples_seen": collection_report["action_valid_samples"],
        "invalid_samples_skipped": collection_report["invalid_samples"],
        "state_samples_used": int(world_array.shape[0]),
        "action_samples_used": int(world_action_array.shape[0]),
        "samples_used": int(world_array.shape[0]),
        "shards_sampled": int(collection_report.get("shards_sampled", len(tar_paths))),
        "sample_limit": int(args.sample_limit),
        "camera_z_convention": "camera-space z > 0 is in front of the camera; z < 0 is behind the camera.",
        "state_definition": {
            "world_18d": "left_wrist_xyz + right_wrist_xyz + left_root_rot6d + right_root_rot6d",
            "camera_18d": "same 18D state transformed by lowdim camera_w2c",
            "world_action_18d": "lowdim next-frame wrist state minus current wrist state in world coordinates",
            "camera_action_18d": "transform current and next wrist state with the current frame lowdim camera_w2c, then subtract",
        },
        "world_18d": compute_stats(world_array),
        "camera_18d": compute_stats(camera_array),
        "camera_depth_summary": compute_camera_depth_summary(camera_array),
    }
    if world_action_array.shape[0] > 0:
        payload["world_action_18d"] = compute_stats(world_action_array)
        payload["camera_action_18d"] = compute_stats(camera_action_array)
    if "shard_reports" in collection_report:
        payload["shard_sampling"] = collection_report["shard_reports"]

    lines = [
        f"Inputs: {len(requested_inputs)}",
        f"Shards analyzed: {payload['shards_analyzed']}",
        f"Samples seen: {payload['samples_seen']}",
        f"Valid samples seen: {payload['valid_samples_seen']}",
        f"Action-valid samples seen: {payload['action_valid_samples_seen']}",
        f"Invalid samples skipped: {payload['invalid_samples_skipped']}",
        f"State samples used: {payload['state_samples_used']}",
        f"Action samples used: {payload['action_samples_used']}",
        f"Shards sampled: {payload['shards_sampled']}",
        f"Sample limit: {payload['sample_limit']}",
        "",
    ]
    lines.extend(format_depth_summary_lines(payload["camera_depth_summary"]))
    lines.append("")
    lines.extend(format_stats_lines("world_18d", payload["world_18d"]))
    lines.append("")
    lines.extend(format_stats_lines("camera_18d", payload["camera_18d"]))
    if "world_action_18d" in payload:
        lines.append("")
        lines.extend(format_stats_lines("world_action_18d", payload["world_action_18d"]))
        lines.append("")
        lines.extend(format_stats_lines("camera_action_18d", payload["camera_action_18d"]))
    print("\n".join(lines), flush=True)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
