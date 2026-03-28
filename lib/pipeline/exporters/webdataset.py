#!/usr/bin/env python3
"""Build VLA WebDataset from BuildAI 10K + HaWoR outputs."""

import argparse
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .webdataset_discovery import (  # noqa: E402
    discover_episode_stats,
    discover_episodes,
    get_episode_feature_cache_path,
    load_episode_stats,
    load_or_build_frame_index,
    parse_factory_range,
    repeat_episode_stats,
)
from .webdataset_annotation import (  # noqa: E402
    DEFAULT_ANNOTATION_SUFFIX,
    attach_or_filter_episode_instructions,
)
from .webdataset_features import (  # noqa: E402
    DEFAULT_INTRINSIC,
    FINGERTIP_INDICES,
    LOWDIM_SIZE,
    build_mano_models,
    load_episode_features,
    run_infill_for_episode,
    run_mano_forward,
)
from .webdataset_geometry import (  # noqa: E402
    axis_angle_to_rot6d,
    interpolate_extrinsics,
    normalize_slam_keyframes,
    quat_to_4x4,
)
from .webdataset_workers import (  # noqa: E402
    _worker_init,
    _worker_process_shard,
    normalize_mano_devices,
)
from .webdataset_writer import add_sample_to_tar, iter_episode_samples, plan_shards  # noqa: E402

__all__ = [
    "DEFAULT_INTRINSIC",
    "DEFAULT_ANNOTATION_SUFFIX",
    "FINGERTIP_INDICES",
    "LOWDIM_SIZE",
    "_worker_init",
    "_worker_process_shard",
    "add_sample_to_tar",
    "attach_or_filter_episode_instructions",
    "axis_angle_to_rot6d",
    "build_mano_models",
    "discover_episode_stats",
    "discover_episodes",
    "get_episode_feature_cache_path",
    "interpolate_extrinsics",
    "iter_episode_samples",
    "load_episode_features",
    "load_episode_stats",
    "load_or_build_frame_index",
    "main",
    "normalize_mano_devices",
    "parse_factory_range",
    "normalize_slam_keyframes",
    "plan_shards",
    "quat_to_4x4",
    "repeat_episode_stats",
    "run_infill_for_episode",
    "run_mano_forward",
]


def build_parser():
    """Build CLI parser for WebDataset export."""
    parser = argparse.ArgumentParser(description="Build VLA WebDataset from BuildAI + HaWoR")
    parser.add_argument("--input_dir", default="/share_data/lvjianan/datasets/BuildAI-processed/")
    parser.add_argument("--output_dir", default="/share_data/guantianrui/datasets/BuildAI-VLA/")
    parser.add_argument("--episode_list", default=None, help="Text file with one episode path per line")
    parser.add_argument("--frames_per_shard", type=int, default=10000)
    parser.add_argument("--factory_range", default=None, help="Inclusive factory range like 1-50 for 10K BuildAI layout")
    parser.add_argument("--repeat_episodes", type=int, default=1, help="Repeat the full episode list this many times in order")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes for testing")
    parser.add_argument("--mano_device", default="cuda:0", help="Device for MANO forward pass")
    parser.add_argument("--mano_gpus", default=None, help="Comma-separated GPU ids for parallel MANO workers, e.g. 0,1,2,3")
    parser.add_argument("--mano_dir", default=None, help="Directory containing MANO_RIGHT.pkl and MANO_LEFT.pkl")
    parser.add_argument("--rescan", action="store_true", help="Force rescan episodes and frame indexes")
    parser.add_argument("--writer_workers", type=int, default=8, help="Number of parallel shard writers")
    parser.add_argument("--shard_manifest_out", default=None, help="Optional JSON manifest of planned shards")
    parser.add_argument("--auto_infill", action="store_true", help="Run infill for missing world_space_res.pth")
    parser.add_argument(
        "--checkpoint",
        default="/share_data/guantianrui/webhaworset/facHaWoRy/weights/hawor/checkpoints/hawor.ckpt",
        help="HaWoR checkpoint path (required if --auto_infill)",
    )
    parser.add_argument(
        "--infiller_weight",
        default="/share_data/guantianrui/webhaworset/facHaWoRy/weights/hawor/checkpoints/infiller.pt",
        help="Infiller weight path (required if --auto_infill)",
    )
    parser.add_argument(
        "--annotation_suffix",
        default=DEFAULT_ANNOTATION_SUFFIX,
        help="Suffix for per-episode annotation JSON files",
    )
    parser.add_argument(
        "--allow_missing_annotation",
        action="store_true",
        help="Keep episodes even when annotation is missing or invalid",
    )
    return parser


def normalize_args(args):
    """Normalize deprecated aliases and validate simple invariants."""
    if args.repeat_episodes < 1:
        raise ValueError("--repeat_episodes must be >= 1")
    parse_factory_range(args.factory_range)
    return args.writer_workers


def validate_auto_infill_args(args):
    """Validate auto-infill dependencies."""
    if not args.auto_infill:
        return True
    if not args.checkpoint or not args.infiller_weight:
        print("Error: --auto_infill requires --checkpoint and --infiller_weight")
        return False
    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}")
        return False
    if not os.path.exists(args.infiller_weight):
        print(f"Error: infiller_weight not found: {args.infiller_weight}")
        return False
    return True


def maybe_rescan_episode_cache(cache_file, rescan):
    """Drop stale episode cache when requested."""
    if rescan and os.path.exists(cache_file):
        os.remove(cache_file)


def maybe_run_auto_infill(args, cache_file, writer_workers):
    """Run infiller for episodes missing world-space results."""
    if not args.auto_infill:
        return

    all_episodes = discover_episodes(
        args.input_dir,
        episode_list=args.episode_list,
        max_episodes=args.max_episodes,
        require_world_res=False,
        factory_range=args.factory_range,
    )
    missing_infill = [
        ep
        for ep in all_episodes
        if not os.path.exists(os.path.join(ep["crop_dir"], "world_space_res.pth"))
    ]
    if missing_infill:
        if len(missing_infill) > max(8, writer_workers):
            print(
                "Warning: many episodes need infill. For large runs, prefer "
                "`scripts/batch_infer.py --stages infiller` before building WebDataset."
            )
        print(f"Running infill for {len(missing_infill)} episodes...")
        for ep in tqdm(missing_infill, desc="Infill"):
            run_infill_for_episode(ep["crop_dir"], args.checkpoint, args.infiller_weight, args.mano_device)

    if os.path.exists(cache_file):
        os.remove(cache_file)


def prepare_episode_stats(args, cache_file):
    """Discover, validate, and expand episodes before shard planning."""
    episodes = discover_episodes(
        args.input_dir,
        args.episode_list,
        args.max_episodes,
        cache_file=cache_file,
        factory_range=args.factory_range,
    )
    print(f"Found {len(episodes)} episodes with world_space_res.pth")
    if not episodes:
        return None, None

    print("Collecting episode stats...")
    episode_stats = discover_episode_stats(episodes, rescan_frame_index=args.rescan)
    if not episode_stats:
        return episodes, None

    episode_stats, annotation_stats = attach_or_filter_episode_instructions(
        episode_stats,
        annotation_suffix=args.annotation_suffix,
        allow_missing_annotation=args.allow_missing_annotation,
    )
    print(
        "Annotation filter:"
        f" kept={annotation_stats['kept']}"
        f" filtered={annotation_stats['filtered']}"
        f" missing={annotation_stats['missing_annotation']}"
        f" invalid_json={annotation_stats['invalid_json']}"
        f" invalid_status={annotation_stats['invalid_status']}"
        f" empty_instruction={annotation_stats['empty_instruction']}"
    )
    if not episode_stats:
        return episodes, None

    filtered_episodes = episode_stats
    episode_stats = repeat_episode_stats(filtered_episodes, args.repeat_episodes)
    return filtered_episodes, episode_stats


def prepare_feature_cache_dir(args, episodes, episode_stats):
    """Create feature cache directory when repeating episodes."""
    feature_cache_dir = None
    if args.repeat_episodes > 1:
        print(
            f"Expanded dataset by repeating {len(episodes)} episodes x{args.repeat_episodes} "
            f"-> {len(episode_stats)} episode entries"
        )
        feature_cache_dir = os.path.join(args.output_dir, "_episode_feature_cache")
        os.makedirs(feature_cache_dir, exist_ok=True)
        print(f"Episode feature cache enabled: {feature_cache_dir}")
    return feature_cache_dir


def write_shard_manifest(shard_tasks, shard_manifest_out):
    """Write planned shard manifest when requested."""
    if not shard_manifest_out:
        return
    with open(shard_manifest_out, "w") as f:
        json.dump(shard_tasks, f, ensure_ascii=False, indent=2)
    print(f"Wrote shard manifest to {shard_manifest_out}")


def resolve_mano_runtime(args, writer_workers):
    """Resolve MANO device placement and worker caps."""
    mano_device = torch.device(args.mano_device if torch.cuda.is_available() else "cpu")
    mano_device_specs = normalize_mano_devices(str(mano_device), args.mano_gpus if mano_device.type == "cuda" else None)

    if mano_device.type == "cuda":
        if len(mano_device_specs) > 1:
            if writer_workers > len(mano_device_specs):
                print(
                    f"Capping shard workers from {writer_workers} to {len(mano_device_specs)} "
                    f"to match MANO GPU workers: {', '.join(mano_device_specs)}"
                )
                writer_workers = len(mano_device_specs)
        elif writer_workers > 1:
            print(
                f"MANO device {mano_device} is CUDA with a single GPU worker; capping shard workers "
                f"from {writer_workers} to 1 to avoid GPU contention. Use --mano_gpus for multi-GPU writing."
            )
            writer_workers = 1

    return mano_device, mano_device_specs, writer_workers


def print_writer_config(writer_workers, mano_device_specs):
    """Print worker allocation summary."""
    if len(mano_device_specs) > 1:
        print(
            f"Writing shards with {writer_workers} worker(s) across MANO GPUs: "
            f"{', '.join(mano_device_specs)}"
        )
    else:
        print(f"Writing shards with {writer_workers} worker(s) on MANO device {mano_device_specs[0]}...")


def run_shard_writers(shard_tasks, writer_workers, mano_device, mano_device_specs, args, feature_cache_dir):
    """Execute shard writers and aggregate progress."""
    totals = {
        "total_frames": 0,
        "total_shards": 0,
        "total_episodes_written": 0,
        "total_skipped": 0,
    }
    pool = None

    if writer_workers <= 1:
        _worker_init(mano_device_specs, args.mano_dir, args.rescan, feature_cache_dir)
        results_iter = (_worker_process_shard(task) for task in shard_tasks)
    else:
        mp_context = get_context("spawn") if mano_device.type == "cuda" else get_context()
        pool = mp_context.Pool(
            writer_workers,
            initializer=_worker_init,
            initargs=(mano_device_specs, args.mano_dir, args.rescan, feature_cache_dir),
        )
        results_iter = pool.imap_unordered(_worker_process_shard, shard_tasks)

    try:
        for result in tqdm(results_iter, total=len(shard_tasks), desc="Shards"):
            totals["total_frames"] += result["frames_written"]
            totals["total_shards"] += 1 if result["frames_written"] > 0 else 0
            totals["total_episodes_written"] += result["episodes_written"]
            totals["total_skipped"] += result["skipped_episodes"]
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return totals


def print_summary(output_dir, totals):
    """Print final export summary."""
    print("\nDone!")
    print(f"  Episodes touched: {totals['total_episodes_written']}")
    print(f"  Skipped episode slices: {totals['total_skipped']}")
    print(f"  Frames: {totals['total_frames']}")
    print(f"  Shards: {totals['total_shards']}")
    print(f"  Output: {output_dir}")


def main():
    args = build_parser().parse_args()
    writer_workers = normalize_args(args)
    os.makedirs(args.output_dir, exist_ok=True)

    if not validate_auto_infill_args(args):
        return

    cache_file = os.path.join(args.input_dir, "_vla_episodes_cache.json")
    maybe_rescan_episode_cache(cache_file, args.rescan)
    maybe_run_auto_infill(args, cache_file, writer_workers)

    episodes, episode_stats = prepare_episode_stats(args, cache_file)
    if not episodes:
        print("No episodes found!")
        return
    if not episode_stats:
        print("No valid episodes with extracted frames found!")
        return

    feature_cache_dir = prepare_feature_cache_dir(args, episodes, episode_stats)
    shard_tasks = plan_shards(episode_stats, args.frames_per_shard, args.output_dir)
    print(f"Planned {len(shard_tasks)} shards from {sum(ep['num_valid_frames'] for ep in episode_stats)} frames")
    write_shard_manifest(shard_tasks, args.shard_manifest_out)

    mano_device, mano_device_specs, writer_workers = resolve_mano_runtime(args, writer_workers)
    print_writer_config(writer_workers, mano_device_specs)
    totals = run_shard_writers(
        shard_tasks,
        writer_workers,
        mano_device,
        mano_device_specs,
        args,
        feature_cache_dir,
    )
    print_summary(args.output_dir, totals)


if __name__ == "__main__":
    main()
