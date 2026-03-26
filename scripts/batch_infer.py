#!/usr/bin/env python3
"""
Multi-GPU batch inference scheduler for HaWoR.

This entrypoint keeps CLI parsing and input collection thin. Batch scheduling,
state tracking, and worker orchestration live under lib.pipeline.batch.
"""
import argparse
import os
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.batch.config import BatchRunConfig


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")

SHARED_TMP_DIR = Path("/share_data/guantianrui/tmp")
SHARED_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(SHARED_TMP_DIR)
os.environ["TEMP"] = str(SHARED_TMP_DIR)
os.environ["TMP"] = str(SHARED_TMP_DIR)
tempfile.tempdir = str(SHARED_TMP_DIR)


def collect_videos(video_dir: Path, extensions=(".mp4", ".avi", ".mov")) -> List[str]:
    videos = []
    for ext in extensions:
        videos.extend(str(path) for path in video_dir.rglob(f"*{ext}"))
    return sorted(videos)


def resolve_inputs(args):
    from lib.pipeline.video_index import collect_videos_from_factory, collect_videos_from_factories

    descriptors = None

    if args.video_list:
        with open(args.video_list) as handle:
            video_paths = [line.strip() for line in handle if line.strip()]
    elif args.video_dir:
        video_paths = collect_videos(Path(args.video_dir))
    elif args.factory_dir:
        descriptors = collect_videos_from_factory(args.factory_dir)
        video_paths = [descriptor.video_key for descriptor in descriptors]
        print(f"Factory mode: {args.factory_dir}")
        print(f"Discovered {len(descriptors)} videos from factory")
    elif args.factory_list:
        with open(args.factory_list) as handle:
            factory_dirs = [line.strip() for line in handle if line.strip()]
        descriptors = collect_videos_from_factories(factory_dirs)
        video_paths = [descriptor.video_key for descriptor in descriptors]
        print(f"Factory mode: {len(factory_dirs)} factories")
        print(f"Discovered {len(descriptors)} videos total")
    elif args.factory_range:
        start_factory = int(args.factory_range[0])
        end_factory = int(args.factory_range[1])
        factory_dirs = [
            os.path.join(args.factory_base, f"factory{factory_id:03d}")
            for factory_id in range(start_factory, end_factory + 1)
        ]
        missing = [path for path in factory_dirs if not os.path.isdir(path)]
        if missing:
            print(f"Warning: {len(missing)} factory dirs not found, skipping", file=sys.stderr)
            factory_dirs = [path for path in factory_dirs if os.path.isdir(path)]
        if not factory_dirs:
            raise ValueError("No valid factory directories found")

        descriptors = collect_videos_from_factories(factory_dirs)
        video_paths = [descriptor.video_key for descriptor in descriptors]
        print(
            f"Factory range: factory{start_factory:03d} ~ factory{end_factory:03d} "
            f"({len(factory_dirs)} factories)"
        )
        print(f"Discovered {len(descriptors)} videos total")
    else:
        video_paths = []

    if not video_paths:
        raise ValueError("No videos found")

    total_videos = len(video_paths)
    start_idx = args.start
    end_idx = args.end if args.end is not None else total_videos

    if start_idx < 0 or start_idx >= total_videos:
        raise ValueError(f"--start {start_idx} is out of range [0, {total_videos})")
    if end_idx < start_idx or end_idx > total_videos:
        raise ValueError(f"--end {end_idx} is out of range [{start_idx}, {total_videos}]")

    video_paths = video_paths[start_idx:end_idx]
    if descriptors is not None:
        descriptors = descriptors[start_idx:end_idx]

    if not video_paths:
        raise ValueError(f"No videos in range [{start_idx}, {end_idx})")

    return video_paths, descriptors, total_videos, start_idx, end_idx


def build_run_dir(args) -> Path:
    if args.run_dir:
        return Path(args.run_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "batch_runs" / timestamp


def get_parser():
    parser = argparse.ArgumentParser(description="Multi-GPU batch inference scheduler for HaWoR")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_list", type=str, help="Path to text file with one video path per line")
    input_group.add_argument("--video_dir", type=str, help="Directory to recursively search for video files")
    input_group.add_argument("--factory_dir", type=str, help="WebDataset factory directory containing tar shards")
    input_group.add_argument("--factory_list", type=str, help="Path to text file with one factory directory per line")
    input_group.add_argument(
        "--factory_range",
        type=str,
        nargs=2,
        metavar=("START", "END"),
        help="Factory ID range (inclusive), e.g. --factory_range 1 10",
    )

    parser.add_argument(
        "--factory_base",
        type=str,
        default="/share_data/guantianrui/datasets/Egocentric-100K/processed_v9_test_jpg",
        help="Base directory for --factory_range",
    )
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs (e.g. 0,1,2,3)")
    parser.add_argument(
        "--stages",
        type=str,
        default="detect_track,motion,slam,infiller",
        help="Comma-separated stage names",
    )
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Resume from existing outputs")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore existing outputs and rerun all stages")
    parser.add_argument("--run_dir", type=str, help="Custom run directory (default: batch_runs/<timestamp>)")
    parser.add_argument("--checkpoint", type=str, default="./weights/hawor/checkpoints/hawor.ckpt", help="Path to HaWoR checkpoint")
    parser.add_argument(
        "--infiller_weight",
        type=str,
        default="./weights/hawor/checkpoints/infiller.pt",
        help="Path to infiller weights",
    )
    parser.add_argument("--img_focal", type=float, help="Image focal length (optional)")
    parser.add_argument(
        "--chunk_batch_size",
        type=int,
        default=64,
        help="Number of 16-frame chunks processed per forward in HAWOR motion stage",
    )
    parser.add_argument("--num_workers", type=int, default=16, help="Number of DataLoader workers for motion stage frame loading")
    parser.add_argument(
        "--metric3d_batch_size",
        type=int,
        default=32,
        help="Batch size for Metric3D depth estimation in SLAM stage",
    )
    parser.add_argument(
        "--slam_backend",
        type=str,
        default="dpvo",
        choices=["droid", "dpvo"],
        help="SLAM backend for stage3",
    )
    parser.add_argument(
        "--depth_backend",
        type=str,
        default="metric3d",
        choices=["metric3d", "any4d"],
        help="Metric depth backend used for stage3 scale estimation",
    )
    parser.add_argument(
        "--depth_predict_all_frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Predict dense depth for all frames and cache it under SLAM/",
    )
    parser.add_argument("--any4d_repo_root", type=str, default=None, help="Optional Any4D repo root override")
    parser.add_argument("--any4d_checkpoint_path", type=str, default=None, help="Optional Any4D checkpoint override")
    parser.add_argument("--any4d_resolution_set", type=int, default=None, help="Optional Any4D inference resolution override")
    parser.add_argument(
        "--any4d_use_amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override Any4D AMP usage when depth_backend=any4d",
    )
    parser.add_argument(
        "--render_batch_size",
        type=int,
        default=8,
        help="Batch size for rendering phase in motion stage",
    )
    parser.add_argument(
        "--infiller_window_batch_size",
        type=int,
        default=64,
        help="Number of infiller windows to batch per forward pass",
    )
    parser.add_argument("--detect_batch_size", type=int, default=128, help="Batch size for YOLO detection in detect_track stage")
    parser.add_argument("--detect_io_workers", type=int, default=8, help="Number of DataLoader workers for detect_track stage")
    parser.add_argument("--detect_device", type=str, default="cuda:0", help="Device for YOLO detector in detect_track stage")
    parser.add_argument("--detect_half_precision", action="store_true", default=True, help="Use FP16 for YOLO detector")
    parser.add_argument("--no-detect_half_precision", dest="detect_half_precision", action="store_false", help="Disable FP16 for YOLO detector")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="Default number of parallel worker processes per GPU in wave mode")
    parser.add_argument("--detect_track_workers_per_gpu", type=int, help="Override workers per GPU for detect_track stage")
    parser.add_argument("--motion_workers_per_gpu", type=int, help="Override workers per GPU for motion stage")
    parser.add_argument("--slam_workers_per_gpu", type=int, help="Override workers per GPU for slam stage")
    parser.add_argument("--infiller_workers_per_gpu", type=int, help="Override workers per GPU for infiller stage")
    parser.add_argument("--enable_profiler", action="store_true", help="Enable torch profiler to diagnose performance bottlenecks")
    parser.add_argument("--rebuild_cam_space_cache", action="store_true", help="Rebuild cached cam_space tensors before running infiller")
    parser.add_argument("--start", type=int, default=0, help="Start index of video list (inclusive, 0-based)")
    parser.add_argument("--end", type=int, default=None, help="End index of video list (exclusive, None means process all)")
    parser.add_argument(
        "--max_stage_retries",
        type=int,
        default=2,
        help="Max retries per stage wave.",
    )
    parser.add_argument(
        "--wave_stall_timeout_sec",
        type=int,
        default=1800,
        help="Fail the current wave if no worker reports a result for this many seconds.",
    )
    return parser


def main():
    args = get_parser().parse_args()

    try:
        video_paths, descriptors, total_videos, start_idx, end_idx = resolve_inputs(args)
        run_dir = build_run_dir(args)
        run_dir.mkdir(parents=True, exist_ok=True)
        config = BatchRunConfig.from_args(
            args,
            video_paths=video_paths,
            descriptors=descriptors,
            run_dir=run_dir,
        )
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    print("=== Batch Inference Configuration ===")
    print(f"Total videos in list: {total_videos}")
    print(f"Processing range: [{start_idx}, {end_idx})")
    print(f"Videos to process: {len(video_paths)}")
    print(f"GPUs: {config.gpus}")
    print(f"Stages: {config.stages}")
    print("Scheduler mode: wave")
    print(f"Max stage retries: {config.max_stage_retries}")
    print(f"Wave stall timeout (sec): {config.wave_stall_timeout_sec}")
    print(f"Detect batch size (detect_track): {config.detect_batch_size}")
    print(f"Detect I/O workers: {config.detect_io_workers}")
    print(f"Chunk batch size (motion): {config.chunk_batch_size}")
    print(f"Metric3D batch size (slam): {config.metric3d_batch_size}")
    print(
        "Stage3 backends: "
        f"slam={config.slam_backend}, "
        f"depth={config.depth_backend}, "
        f"depth_predict_all_frames={config.depth_predict_all_frames}"
    )
    print(f"Infiller window batch size: {config.infiller_window_batch_size}")
    print(
        "Workers per GPU: "
        f"default={config.workers_per_gpu}, "
        f"detect_track={config.worker_count_for_stage('detect_track')}, "
        f"motion={config.worker_count_for_stage('motion')}, "
        f"slam={config.worker_count_for_stage('slam')}, "
        f"infiller={config.worker_count_for_stage('infiller')}"
    )
    print(f"Resume: {config.resume}")
    print(f"Run directory: {run_dir}")
    print()

    from lib.pipeline.batch.scheduler import BatchScheduler

    success = BatchScheduler(config).run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
