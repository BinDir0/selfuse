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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _configure_process_environment():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*pkg_resources.*")
    warnings.filterwarnings("ignore", message=".*timm.models.layers.*")
    warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

    shared_tmp_dir = Path("/share_data/guantianrui/tmp")
    shared_tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(shared_tmp_dir)
    os.environ["TEMP"] = str(shared_tmp_dir)
    os.environ["TMP"] = str(shared_tmp_dir)
    tempfile.tempdir = str(shared_tmp_dir)
    os.environ["HAWOR_QUIET"] = "1"


_configure_process_environment()

from lib.pipeline.runtime import WorkerRuntime, set_determinism  # noqa: E402
from lib.pipeline.stage_api import STAGES, PipelineVideoTask, run_pipeline_stage  # noqa: E402
from lib.pipeline.video_index import VideoDescriptor  # noqa: E402


def emit_event(event: str, **kwargs):
    payload = {
        "time": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **kwargs,
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _build_runtime_from_args(ns):
    set_determinism(ns.seed)
    return WorkerRuntime(
        gpu=ns.gpu,
        checkpoint=ns.checkpoint,
        infiller_weight=ns.infiller_weight,
        img_focal=ns.img_focal,
        chunk_batch_size=ns.chunk_batch_size,
        num_workers=ns.num_workers,
        render_batch_size=ns.render_batch_size,
        any4d_batch_size=ns.any4d_batch_size,
        detect_batch_size=ns.detect_batch_size,
        detect_io_workers=ns.detect_io_workers,
        detect_device=ns.detect_device,
        detect_half_precision=ns.detect_half_precision,
        infiller_window_batch_size=ns.infiller_window_batch_size,
        rebuild_cam_space_cache=ns.rebuild_cam_space_cache,
        depth_predict_all_frames=ns.depth_predict_all_frames,
        any4d_repo_root=ns.any4d_repo_root,
        any4d_checkpoint_path=ns.any4d_checkpoint_path,
        any4d_resolution_set=ns.any4d_resolution_set,
        any4d_use_amp=ns.any4d_use_amp,
        stage3_tmp_root=ns.stage3_tmp_root,
    )


def _parse_video_list_entry(line: str):
    if line.startswith("{"):
        descriptor = VideoDescriptor.from_json(line)
        return {
            "descriptor": descriptor,
            "video_path": descriptor.video_key,
            "video_label": descriptor.video_key,
        }

    return {
        "descriptor": None,
        "video_path": line,
        "video_label": line,
    }


def _load_video_tasks(video_list_path: str):
    with open(video_list_path) as handle:
        lines = [line.strip() for line in handle if line.strip()]
    return [_parse_video_list_entry(line) for line in lines]


def _build_task_namespace(base_ns, task_entry):
    task_ns = argparse.Namespace(**vars(base_ns))
    task_ns._descriptor = task_entry["descriptor"]
    task_ns.video_path = task_entry["video_path"]
    return task_ns


def _build_common_event_fields(task_ns, video_label):
    return {
        "video": video_label,
        "stage": task_ns.stage,
        "gpu": task_ns.gpu,
    }


def _build_profiler_output_dir(task_ns, task: PipelineVideoTask):
    if getattr(task_ns, "run_dir", None):
        profiler_output_dir = Path(task_ns.run_dir) / "profiler_traces"
    else:
        profiler_output_dir = task.seq_folder.parent / "profiler_traces"
    profiler_output_dir.mkdir(parents=True, exist_ok=True)
    return profiler_output_dir


def _run_stage_with_profiler(runtime: WorkerRuntime, task_ns, task: PipelineVideoTask):
    from torch.profiler import ProfilerActivity, profile, schedule

    profiler_output_dir = _build_profiler_output_dir(task_ns, task)
    trace_path = profiler_output_dir / f"motion_trace_{Path(task_ns.video_path).stem}.json"

    print(f"[PROFILER] Enabled. Output dir: {profiler_output_dir}")
    print(f"[PROFILER] Video: {Path(task_ns.video_path).stem}")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=1, active=3, repeat=1),
        on_trace_ready=lambda profiler: (
            print(f"[PROFILER] Trace ready, exporting to {trace_path}"),
            profiler.export_chrome_trace(str(trace_path)),
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        return run_pipeline_stage(
            task_ns.stage,
            task,
            runtime.stage_config,
            runtime=runtime,
            profiler=profiler,
            resume=task_ns.resume,
            force=task_ns.force,
        )


def _run_stage_with_runtime(runtime: WorkerRuntime, task_ns):
    task = PipelineVideoTask.from_namespace(task_ns)
    if task_ns.stage == "motion" and getattr(task_ns, "enable_profiler", False):
        return _run_stage_with_profiler(runtime, task_ns, task)

    return run_pipeline_stage(
        task_ns.stage,
        task,
        runtime.stage_config,
        runtime=runtime,
        resume=task_ns.resume,
        force=task_ns.force,
    )


def _execute_task(runtime: WorkerRuntime, task_ns, video_label: str):
    common_fields = _build_common_event_fields(task_ns, video_label)
    started_at = time.time()
    emit_event("stage_start", **common_fields)

    try:
        result = _run_stage_with_runtime(runtime, task_ns)
        emit_event(
            "stage_end",
            **common_fields,
            status=result.get("status", "success"),
            elapsed_sec=round(time.time() - started_at, 3),
            reason=result.get("reason"),
            start_idx=result.get("start_idx"),
            end_idx=result.get("end_idx"),
        )
        return True
    except Exception as error:
        emit_event(
            "stage_end",
            **common_fields,
            status="failed",
            elapsed_sec=round(time.time() - started_at, 3),
            error=str(error),
        )
        traceback.print_exc()
        return False
    finally:
        torch.cuda.empty_cache()


def worker_runtime_loop(ns):
    runtime = _build_runtime_from_args(ns)
    runtime.ensure_runner(ns.stage)

    overall_success = True
    for task_entry in _load_video_tasks(ns.video_list):
        task_ns = _build_task_namespace(ns, task_entry)
        if not _execute_task(runtime, task_ns, task_entry["video_label"]):
            overall_success = False

    return overall_success


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--video_list", required=True, help="Path to text file with one video path per line or descriptor JSON Lines")
    parser.add_argument("--gpu", default="", type=str)
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--checkpoint", type=str, default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", type=str, default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16, help="Number of DataLoader workers for parallel frame loading")
    parser.add_argument("--render_batch_size", type=int, default=8, help="Batch size for rendering phase")
    parser.add_argument("--any4d_batch_size", type=int, default=32, help="Batch size for Any4D depth estimation")
    parser.add_argument(
        "--depth_predict_all_frames",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Predict dense depth for all frames in SLAM; defaults to env HAWOR_DEPTH_PREDICT_ALL_FRAMES or on.",
    )
    parser.add_argument("--any4d_repo_root", type=str, default=None)
    parser.add_argument("--any4d_checkpoint_path", type=str, default=None)
    parser.add_argument("--any4d_resolution_set", type=int, default=None)
    parser.add_argument("--any4d_use_amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stage3_tmp_root", type=str, default=None)
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
    parser.add_argument("--enable_profiler", action="store_true", help="Enable torch profiler to diagnose performance bottlenecks")
    parser.add_argument("--run_dir", type=str, help="Batch run directory for output organization")
    return parser


def main():
    args = get_parser().parse_args()
    success = worker_runtime_loop(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
