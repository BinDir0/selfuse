from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from lib.pipeline.clip_manifest import load_clip_manifest


DEFAULT_BATCH_STAGES = "detect_track,motion,slam,infiller"
DEFAULT_ANY4D_BATCH_SIZE = 32
DEFAULT_INFILLER_WINDOW_BATCH_SIZE = 64
DEFAULT_WAVE_STALL_TIMEOUT_SEC = 3600


@dataclass(frozen=True)
class BatchInputSelection:
    input_mode: str
    input_path: str
    total_items: int
    start_idx: int
    end_idx: int
    video_paths: list[str]
    descriptors: list | None


def collect_videos(video_dir: Path, extensions=(".mp4", ".avi", ".mov")) -> list[str]:
    videos = []
    for ext in extensions:
        videos.extend(str(path) for path in video_dir.rglob(f"*{ext}"))
    return sorted(videos)


def build_batch_infer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-GPU batch inference scheduler for HaWoR")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--descriptor_manifest",
        type=str,
        help="Frozen clip manifest JSONL path. Preferred input mode.",
    )
    input_group.add_argument(
        "--video_list",
        type=str,
        help="Compatibility input mode: path to text file with one video path per line.",
    )
    input_group.add_argument(
        "--video_dir",
        type=str,
        help="Compatibility input mode: directory to recursively search for video files.",
    )

    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs (e.g. '0,1,2,3').")
    parser.add_argument("--stages", type=str, default=DEFAULT_BATCH_STAGES, help="Comma-separated stage names.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing outputs. Default: enabled.",
    )
    parser.add_argument("--run_dir", type=str, help="Custom run directory. Default: batch_runs/<timestamp>.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./weights/hawor/checkpoints/hawor.ckpt",
        help="Path to HaWoR checkpoint.",
    )
    parser.add_argument(
        "--infiller_weight",
        type=str,
        default="./weights/hawor/checkpoints/infiller.pt",
        help="Path to infiller weights.",
    )
    parser.add_argument("--img_focal", type=float, help="Image focal length.")
    parser.add_argument(
        "--chunk_batch_size",
        type=int,
        default=64,
        help="Number of 16-frame chunks processed per forward in the motion stage.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of DataLoader workers for frame loading in the motion stage.",
    )
    parser.add_argument(
        "--any4d_batch_size",
        type=int,
        default=None,
        help="Batch size for Any4D depth in the SLAM stage.",
    )
    parser.add_argument(
        "--metric3d_batch_size",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--render_batch_size",
        type=int,
        default=8,
        help="Batch size for the rendering phase in the motion stage.",
    )
    parser.add_argument(
        "--infiller_window_batch_size",
        type=int,
        default=DEFAULT_INFILLER_WINDOW_BATCH_SIZE,
        help="Number of infiller windows processed per batch.",
    )
    parser.add_argument(
        "--detect_batch_size",
        type=int,
        default=128,
        help="Batch size for YOLO detection in the detect_track stage.",
    )
    parser.add_argument(
        "--detect_io_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers for frame loading in the detect_track stage.",
    )
    parser.add_argument(
        "--detect_device",
        type=str,
        default="cuda:0",
        help="Device for YOLO detection in the detect_track stage.",
    )
    parser.add_argument(
        "--detect_half_precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use FP16 for YOLO detection. Default: disabled.",
    )
    parser.add_argument(
        "--enable_profiler",
        action="store_true",
        help="Enable torch profiler to diagnose performance bottlenecks.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index of the selected input list (inclusive).")
    parser.add_argument("--end", type=int, default=None, help="End index of the selected input list (exclusive).")
    parser.add_argument(
        "--scheduler_mode",
        type=str,
        default="legacy",
        choices=["legacy", "wave"],
        help="Compatibility flag. Unified scheduler always uses stage-wave execution.",
    )
    parser.add_argument(
        "--persistent_worker",
        action="store_true",
        help="Compatibility flag retained for older launch scripts.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help="Deprecated alias for --max_stage_retries.",
    )
    parser.add_argument(
        "--max_stage_retries",
        type=int,
        default=1,
        help="Max retries per stage wave.",
    )
    parser.add_argument(
        "--wave_stall_timeout_sec",
        type=int,
        default=DEFAULT_WAVE_STALL_TIMEOUT_SEC,
        help="Terminate and retry a stage wave if workers emit no results for this many seconds.",
    )
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=1,
        help="Default worker slots per GPU for stage-wave execution.",
    )
    parser.add_argument(
        "--detect_track_workers_per_gpu",
        type=int,
        default=None,
        help="Optional per-stage override for detect_track worker slots per GPU.",
    )
    parser.add_argument(
        "--motion_workers_per_gpu",
        type=int,
        default=None,
        help="Optional per-stage override for motion worker slots per GPU.",
    )
    parser.add_argument(
        "--slam_workers_per_gpu",
        type=int,
        default=None,
        help="Optional per-stage override for slam worker slots per GPU.",
    )
    parser.add_argument(
        "--infiller_workers_per_gpu",
        type=int,
        default=None,
        help="Optional per-stage override for infiller worker slots per GPU.",
    )
    parser.add_argument(
        "--slam_backend",
        type=str,
        default="droid",
        choices=["droid", "dpvo"],
        help="SLAM backend to use in the slam stage.",
    )
    parser.add_argument(
        "--depth_backend",
        type=str,
        default=None,
        choices=["metric3d", "any4d"],
        help="Depth backend for SLAM scale estimation.",
    )
    parser.add_argument("--any4d", action="store_true", help="Deprecated shorthand for --depth_backend any4d.")
    parser.add_argument(
        "--depth_predict_all_frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Predict dense depth for all frames in the SLAM stage. Default: enabled.",
    )
    parser.add_argument(
        "--rebuild_cam_space_cache",
        action="store_true",
        help="Rebuild cached camera-space infiller inputs before running infiller.",
    )
    parser.add_argument("--any4d_repo_root", type=str, default=None, help="Optional Any4D repository root.")
    parser.add_argument("--any4d_checkpoint_path", type=str, default=None, help="Optional Any4D checkpoint path.")
    parser.add_argument("--any4d_resolution_set", type=int, default=None, help="Optional Any4D resolution set.")
    parser.add_argument(
        "--any4d_use_amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override AMP usage for Any4D inference.",
    )
    parser.add_argument("--stage3_tmp_root", type=str, default=None, help="Temporary workspace root for Any4D SLAM.")
    return parser


def normalize_batch_infer_args(args, *, raw_argv: list[str] | None = None) -> list[str]:
    raw_argv = list(sys.argv[1:] if raw_argv is None else raw_argv)
    notes = []

    if getattr(args, "any4d", False):
        args.depth_backend = "any4d"

    if args.any4d_batch_size is None:
        if args.metric3d_batch_size is not None:
            args.any4d_batch_size = int(args.metric3d_batch_size)
            notes.append("`--metric3d_batch_size` is deprecated; using it as `--any4d_batch_size`.")
        else:
            args.any4d_batch_size = DEFAULT_ANY4D_BATCH_SIZE
    elif args.metric3d_batch_size is not None and int(args.metric3d_batch_size) != int(args.any4d_batch_size):
        notes.append("Ignoring deprecated `--metric3d_batch_size` because `--any4d_batch_size` is set.")

    if args.retries is not None and "--max_stage_retries" not in raw_argv:
        args.max_stage_retries = int(args.retries)
        notes.append("`--retries` is deprecated; treating it as `--max_stage_retries`.")

    if args.scheduler_mode != "wave":
        notes.append(
            f"`--scheduler_mode {args.scheduler_mode}` is retained for compatibility; the unified scheduler uses wave mode."
        )
    if args.persistent_worker:
        notes.append("`--persistent_worker` is retained for compatibility and has no effect in the unified scheduler.")

    return notes


def load_batch_inputs(args) -> BatchInputSelection:
    descriptors = None
    if args.descriptor_manifest:
        records = load_clip_manifest(args.descriptor_manifest)
        descriptors = [record.descriptor for record in records]
        video_paths = [descriptor.video_key for descriptor in descriptors]
        input_mode = "descriptor_manifest"
        input_path = args.descriptor_manifest
    elif args.video_list:
        with open(args.video_list, "r", encoding="utf-8") as handle:
            video_paths = [line.strip() for line in handle if line.strip()]
        input_mode = "video_list"
        input_path = args.video_list
    else:
        video_paths = collect_videos(Path(args.video_dir))
        input_mode = "video_dir"
        input_path = args.video_dir

    total_items = len(video_paths)
    if total_items <= 0:
        raise ValueError("No videos found for batch inference")

    start_idx = int(args.start)
    end_idx = total_items if args.end is None else int(args.end)
    if start_idx < 0 or start_idx >= total_items:
        raise ValueError(f"--start {start_idx} is out of range [0, {total_items})")
    if end_idx < start_idx or end_idx > total_items:
        raise ValueError(f"--end {end_idx} is out of range [{start_idx}, {total_items}]")

    selected_video_paths = video_paths[start_idx:end_idx]
    selected_descriptors = descriptors[start_idx:end_idx] if descriptors is not None else None
    if not selected_video_paths:
        raise ValueError(f"No videos in range [{start_idx}, {end_idx})")

    return BatchInputSelection(
        input_mode=input_mode,
        input_path=input_path,
        total_items=total_items,
        start_idx=start_idx,
        end_idx=end_idx,
        video_paths=selected_video_paths,
        descriptors=selected_descriptors,
    )
