#!/usr/bin/env python3
"""Benchmark end-to-end dataset-pipeline stages on a sampled descriptor manifest."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.batch.cli import (
    add_any4d_runtime_args,
    add_infer_profile_arg,
    add_local_cache_args,
    normalize_batch_infer_args,
)
from lib.pipeline.batch.config import BatchRunConfig
from lib.pipeline.batch.scheduler import BatchScheduler
from lib.pipeline.clip_manifest import load_clip_manifest
from tools.ops.benchmark_slam_stage import GPUSampler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark detect_track/motion/slam/infiller throughput from a descriptor manifest."
    )
    parser.add_argument("--descriptor_manifest", type=str, required=True, help="Clip manifest JSONL to benchmark.")
    parser.add_argument("--stages", type=str, default="detect_track,motion,slam,infiller")
    parser.add_argument("--gpus", type=str, default="0")
    add_infer_profile_arg(parser)
    parser.add_argument("--workers_per_gpu", type=int, default=1)
    parser.add_argument("--detect_track_workers_per_gpu", type=int, default=None)
    parser.add_argument("--motion_workers_per_gpu", type=int, default=None)
    parser.add_argument("--slam_workers_per_gpu", type=int, default=None)
    parser.add_argument("--infiller_workers_per_gpu", type=int, default=None)
    parser.add_argument("--num_videos", type=int, default=32)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--sample_mode", choices=["random", "first", "longest"], default="random")
    parser.add_argument("--resume", dest="resume", action="store_true", default=False)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--run_dir", type=str, default=None)

    parser.add_argument("--checkpoint", type=str, default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", type=str, default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--img_focal", type=float, default=None)
    parser.add_argument("--chunk_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--any4d_batch_size", type=int, default=32)
    parser.add_argument("--render_batch_size", type=int, default=8)
    parser.add_argument("--infiller_window_batch_size", type=int, default=64)
    parser.add_argument("--detect_batch_size", type=int, default=128)
    parser.add_argument("--detect_device", type=str, default="cuda:0")
    parser.add_argument("--detect_half_precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detect_io_workers", type=int, default=8)
    parser.add_argument("--rebuild_cam_space_cache", action="store_true")
    add_any4d_runtime_args(parser)
    add_local_cache_args(parser)
    parser.add_argument("--max_stage_retries", type=int, default=0)
    parser.add_argument("--wave_stall_timeout_sec", type=int, default=3600)
    parser.add_argument("--sample_interval_sec", type=float, default=1.0)
    args = parser.parse_args()
    normalize_batch_infer_args(args)
    return args


def build_run_dir(args) -> Path:
    if args.run_dir:
        return Path(args.run_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "batch_runs" / f"pipeline_benchmark_{timestamp}"


def sample_descriptors(descriptors, args):
    end = args.end if args.end is not None else len(descriptors)
    sliced = descriptors[args.start:end]
    if not sliced:
        raise ValueError("No descriptors available after slicing")
    if args.num_videos > len(sliced):
        raise ValueError(f"Requested {args.num_videos} videos, but only {len(sliced)} are available")
    if args.sample_mode == "first":
        return sliced[: args.num_videos]
    if args.sample_mode == "longest":
        return sorted(sliced, key=lambda desc: int(desc.frame_count), reverse=True)[: args.num_videos]
    rng = random.Random(args.sample_seed)
    return rng.sample(sliced, args.num_videos)


def remap_descriptor(descriptor, benchmark_inputs_dir: Path):
    dst_seq = benchmark_inputs_dir / descriptor.video_key
    dst_seq.mkdir(parents=True, exist_ok=True)
    return replace(descriptor, seq_folder=str(dst_seq))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    run_dir = build_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    records = load_clip_manifest(args.descriptor_manifest)
    descriptors = [record.descriptor for record in records]
    sampled = sample_descriptors(descriptors, args)

    benchmark_inputs_dir = run_dir / "benchmark_inputs"
    benchmark_inputs_dir.mkdir(parents=True, exist_ok=True)
    benchmark_descriptors = [remap_descriptor(descriptor, benchmark_inputs_dir) for descriptor in sampled]

    manifest = {
        "descriptor_manifest": str(Path(args.descriptor_manifest).resolve()),
        "sample_mode": args.sample_mode,
        "num_videos": len(benchmark_descriptors),
        "stages": args.stages.split(","),
        "sampled": [asdict(descriptor) for descriptor in benchmark_descriptors],
    }
    write_json(run_dir / "benchmark_manifest.json", manifest)

    config = BatchRunConfig.from_args(
        args,
        video_paths=[descriptor.video_key for descriptor in benchmark_descriptors],
        descriptors=benchmark_descriptors,
        run_dir=run_dir / "batch_run",
    )

    print("=" * 72)
    print("Pipeline Benchmark Configuration")
    print("=" * 72)
    print(f"Source manifest       : {args.descriptor_manifest}")
    print(f"Sampled videos        : {len(benchmark_descriptors)}")
    print(f"Stages                : {config.stages}")
    print(f"GPUs                  : {config.gpus}")
    print(f"Workers per GPU       : default={config.workers_per_gpu}")
    print(f"Per-stage workers     : detect={config.worker_count_for_stage('detect_track')} motion={config.worker_count_for_stage('motion')} slam={config.worker_count_for_stage('slam')} infiller={config.worker_count_for_stage('infiller')}")
    print(f"Infer profile         : {config.infer_profile}")
    print(f"Local cache           : mode={config.local_cache_mode} root={config.local_cache_root} quota_gb={config.local_cache_quota_gb}")
    print(f"Stage3 tmp root       : {config.stage3_tmp_root}")
    print(f"Run dir               : {run_dir}")
    print("=" * 72)

    sampler = GPUSampler(config.gpus, interval_sec=args.sample_interval_sec)
    scheduler = BatchScheduler(config)
    sampler.start()
    start_time = time.time()
    success = False
    try:
        success = scheduler.run()
    finally:
        sampler.stop()
    elapsed = time.time() - start_time

    telemetry = sampler.summary()
    completed = {
        stage: sum(1 for task in scheduler.state.tasks.values() if task.stage_status.get(stage) == "completed")
        for stage in config.stages
    }
    report = {
        "success": success,
        "elapsed_sec": elapsed,
        "videos_total": len(config.video_paths),
        "avg_sec_per_video": elapsed / max(len(config.video_paths), 1),
        "stages": config.stages,
        "completed_per_stage": completed,
        "gpu_summary": telemetry,
        "config": {
            "gpus": config.gpus,
            "workers_per_gpu": config.workers_per_gpu,
            "detect_track_workers_per_gpu": config.detect_track_workers_per_gpu,
            "motion_workers_per_gpu": config.motion_workers_per_gpu,
            "slam_workers_per_gpu": config.slam_workers_per_gpu,
            "infiller_workers_per_gpu": config.infiller_workers_per_gpu,
            "infer_profile": config.infer_profile,
            "local_cache_root": config.local_cache_root,
            "local_cache_quota_gb": config.local_cache_quota_gb,
            "local_cache_mode": config.local_cache_mode,
            "local_cache_min_frames": config.local_cache_min_frames,
            "stage3_tmp_root": config.stage3_tmp_root,
        },
    }
    write_json(run_dir / "benchmark_report.json", report)

    print("\nBenchmark summary")
    print("-" * 72)
    print(f"Success             : {success}")
    print(f"Elapsed             : {elapsed:.2f}s")
    print(f"Average sec/video   : {report['avg_sec_per_video']:.2f}")
    print(f"Manifest            : {run_dir / 'benchmark_manifest.json'}")
    print(f"Report              : {run_dir / 'benchmark_report.json'}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
