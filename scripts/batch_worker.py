import argparse
import json
import os
import sys
import tempfile
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import torch

# Suppress common warnings to reduce output noise
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', message='.*timm.models.layers.*')
warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.stage_api import (
    STAGES,
    PipelineVideoTask,
    StageExecutionConfig,
    run_pipeline_stage,
)
from lib.pipeline.runtime import WorkerRuntime, set_determinism
from lib.pipeline.video_index import VideoDescriptor

# Set temporary directory to shared storage instead of local /tmp
# IMPORTANT: Set this AFTER importing torch to avoid library loading issues
SHARED_TMP_DIR = Path("/share_data/guantianrui/tmp")
SHARED_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(SHARED_TMP_DIR)
os.environ["TEMP"] = str(SHARED_TMP_DIR)
os.environ["TMP"] = str(SHARED_TMP_DIR)
tempfile.tempdir = str(SHARED_TMP_DIR)

# Suppress verbose output from stage scripts
os.environ["HAWOR_QUIET"] = "1"

def run_stage_with_runtime(runtime: WorkerRuntime, ns, prefetched_data=None):
    task = PipelineVideoTask.from_namespace(ns)
    return run_pipeline_stage(
        ns.stage,
        task,
        runtime.stage_config,
        runtime=runtime,
        prefetched_data=prefetched_data,
        resume=ns.resume,
        force=ns.force,
    )


def worker_runtime_loop(ns):
    set_determinism(ns.seed)
    runtime = WorkerRuntime(
        gpu=ns.gpu,
        checkpoint=ns.checkpoint,
        infiller_weight=ns.infiller_weight,
        img_focal=ns.img_focal,
        input_type=ns.input_type,
        chunk_batch_size=ns.chunk_batch_size,
        num_workers=getattr(ns, 'num_workers', 16),
        render_batch_size=getattr(ns, 'render_batch_size', 8),
        metric3d_batch_size=getattr(ns, 'metric3d_batch_size', 32),
        detect_batch_size=getattr(ns, 'detect_batch_size', 128),
        detect_io_workers=getattr(ns, 'detect_io_workers', 8),
        detect_device=getattr(ns, 'detect_device', "cuda:0"),
        detect_half_precision=bool(getattr(ns, 'detect_half_precision', True)),
        infiller_window_batch_size=getattr(ns, 'infiller_window_batch_size', 64),
        rebuild_cam_space_cache=getattr(ns, 'rebuild_cam_space_cache', False),
    )

    with open(ns.video_list) as f:
        lines = [line.strip() for line in f if line.strip()]

    # Detect format: JSON Lines (descriptor) or plain paths
    descriptors = []
    for line in lines:
        if line.startswith('{'):
            descriptors.append(VideoDescriptor.from_json(line))
        else:
            descriptors.append(None)  # plain video_path mode

    overall_success = True
    for i, line in enumerate(lines):
        task_ns = argparse.Namespace(**vars(ns))
        desc = descriptors[i]
        if desc is not None:
            task_ns._descriptor = desc
            task_ns.video_path = desc.video_key
            video_label = desc.video_key
        else:
            task_ns._descriptor = None
            task_ns.video_path = line
            video_label = line

        common_fields = {
            "video": video_label,
            "stage": task_ns.stage,
            "gpu": task_ns.gpu,
        }
        started_at = time.time()
        emit_event("stage_start", **common_fields)
        try:
            result = run_stage_with_runtime(runtime, task_ns)
            emit_event(
                "stage_end",
                **common_fields,
                status=result.get("status", "success"),
                elapsed_sec=round(time.time() - started_at, 3),
                reason=result.get("reason"),
                start_idx=result.get("start_idx"),
                end_idx=result.get("end_idx"),
            )
        except Exception as err:
            overall_success = False
            emit_event(
                "stage_end",
                **common_fields,
                status="failed",
                elapsed_sec=round(time.time() - started_at, 3),
                error=str(err),
            )
            traceback.print_exc()

        # Free GPU memory between videos to prevent fragmentation
        torch.cuda.empty_cache()

    return overall_success


def emit_event(event: str, **kwargs):
    payload = {
        "time": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **kwargs,
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def run_stage(ns):
    if ns.gpu is not None and ns.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ns.gpu)

    set_determinism(ns.seed)
    task = PipelineVideoTask.from_namespace(ns)
    config = StageExecutionConfig.from_namespace(ns)

    profiler = None
    if ns.stage == "motion" and getattr(ns, 'enable_profiler', False):
        from torch.profiler import profile, ProfilerActivity, schedule

        if getattr(ns, 'run_dir', None):
            profiler_output_dir = Path(ns.run_dir) / "profiler_traces"
        else:
            profiler_output_dir = task.seq_folder.parent / "profiler_traces"
        profiler_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[PROFILER] Enabled. Output dir: {profiler_output_dir}")
        print(f"[PROFILER] Video: {Path(ns.video_path).stem}")

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=1, active=3, repeat=1),
            on_trace_ready=lambda p: (
                print(f"[PROFILER] Trace ready, exporting to {profiler_output_dir / f'motion_trace_{Path(ns.video_path).stem}.json'}"),
                p.export_chrome_trace(str(profiler_output_dir / f"motion_trace_{Path(ns.video_path).stem}.json"))
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            return run_pipeline_stage(
                ns.stage,
                task,
                config,
                profiler=prof,
                resume=ns.resume,
                force=ns.force,
            )

    return run_pipeline_stage(
        ns.stage,
        task,
        config,
        profiler=profiler,
        resume=ns.resume,
        force=ns.force,
    )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--gpu", default="", type=str)
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--input_type", type=str, default="file")
    parser.add_argument("--checkpoint", type=str, default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", type=str, default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16, help="Number of DataLoader workers for parallel frame loading")
    parser.add_argument("--render_batch_size", type=int, default=8, help="Batch size for rendering phase")
    parser.add_argument("--metric3d_batch_size", type=int, default=32, help="Batch size for Metric3D depth estimation")
    parser.add_argument("--detect_batch_size", type=int, default=128, help="Batch size for YOLO detection (default 128)")
    parser.add_argument("--detect_io_workers", type=int, default=8, help="Number of DataLoader workers for parallel frame loading")
    parser.add_argument("--infiller_window_batch_size", type=int, default=64, help="Number of infiller windows to batch per forward pass")
    parser.add_argument("--rebuild_cam_space_cache", action="store_true", help="Rebuild cached cam_space tensors before running infiller")
    parser.add_argument("--detect_device", type=str, default="cuda:0", help="Device for YOLO detector (e.g., cuda:0)")
    parser.add_argument("--detect_half_precision", action="store_true", default=True, help="Use FP16 for YOLO detector (2x faster)")
    parser.add_argument("--no-detect_half_precision", dest="detect_half_precision", action="store_false", help="Disable FP16 for YOLO")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force", action="store_true", help="Ignore existing outputs and rerun this stage")
    parser.add_argument("--video_list", type=str, help="Optional file with one video path per line (or JSON Lines for WebDataset) for persistent worker mode")
    parser.add_argument("--persistent_worker", action="store_true", help="Run as long-lived stage worker for multiple videos")
    parser.add_argument("--video_descriptor", type=str, help="JSON VideoDescriptor for WebDataset mode (alternative to --video_path)")
    parser.add_argument("--enable_profiler", action="store_true", help="Enable torch profiler to diagnose performance bottlenecks")
    parser.add_argument("--run_dir", type=str, help="Batch run directory for output organization")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    # Parse video_descriptor if provided
    if args.video_descriptor:
        args._descriptor = VideoDescriptor.from_json(args.video_descriptor)
        if not args.video_path:
            args.video_path = args._descriptor.video_key
    else:
        args._descriptor = None

    if args.persistent_worker:
        if not args.video_list:
            raise ValueError("--video_list is required when --persistent_worker is set")
        success = worker_runtime_loop(args)
        sys.exit(0 if success else 1)

    started_at = time.time()
    common_fields = {
        "video": args.video_path,
        "stage": args.stage,
        "gpu": args.gpu,
    }

    emit_event("stage_start", **common_fields)
    try:
        result = run_stage(args)
        emit_event(
            "stage_end",
            **common_fields,
            status=result.get("status", "success"),
            elapsed_sec=round(time.time() - started_at, 3),
            reason=result.get("reason"),
            start_idx=result.get("start_idx"),
            end_idx=result.get("end_idx"),
        )
        sys.exit(0)
    except Exception as err:
        emit_event(
            "stage_end",
            **common_fields,
            status="failed",
            elapsed_sec=round(time.time() - started_at, 3),
            error=str(err),
        )
        traceback.print_exc()
        sys.exit(1)
