#!/usr/bin/env python3
"""Benchmark real Stage3/SLAM throughput on WebDataset factories.

This script runs the existing HaWoR Stage3 pipeline on a sampled subset of
videos using the real decode + DPVO + Any4D path, while isolating outputs under
the benchmark run directory so production outputs are not touched.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from argparse import Namespace
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.batch.config import BatchRunConfig
from lib.pipeline.batch.cli import (
    add_any4d_runtime_args,
    add_infer_profile_arg,
    add_local_cache_args,
    normalize_batch_infer_args,
)
from lib.pipeline.batch.scheduler import BatchScheduler
from lib.pipeline.stage_api import get_track_range
from lib.pipeline.video_index import VideoDescriptor, collect_videos_from_factories, collect_videos_from_factory


class GPUSampler:
    def __init__(self, gpu_ids: List[int], interval_sec: float = 1.0):
        self.gpu_ids = gpu_ids
        self.interval_sec = interval_sec
        self.samples: Dict[int, List[dict]] = {gpu_id: [] for gpu_id in gpu_ids}
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self):
        gpu_arg = ",".join(str(gpu_id) for gpu_id in self.gpu_ids)
        cmd = [
            "nvidia-smi",
            f"--id={gpu_arg}",
            "--query-gpu=index,name,utilization.gpu,memory.used,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
                if result.returncode == 0:
                    now = time.time()
                    for line in result.stdout.strip().splitlines():
                        parts = [part.strip() for part in line.split(",")]
                        if len(parts) != 6:
                            continue
                        gpu_id = int(parts[0])
                        if gpu_id not in self.samples:
                            continue
                        self.samples[gpu_id].append(
                            {
                                "time": now,
                                "name": parts[1],
                                "gpu_util": float(parts[2]),
                                "mem_used_mb": float(parts[3]),
                                "temp_c": float(parts[4]),
                                "power_w": float(parts[5]),
                            }
                        )
            except Exception:
                pass
            self._stop.wait(self.interval_sec)

    def summary(self) -> Dict[int, dict]:
        out = {}
        for gpu_id, samples in self.samples.items():
            if not samples:
                out[gpu_id] = {"samples": 0}
                continue
            utils = sorted(sample["gpu_util"] for sample in samples)
            out[gpu_id] = {
                "samples": len(samples),
                "name": samples[0]["name"],
                "gpu_util_mean": sum(sample["gpu_util"] for sample in samples) / len(samples),
                "gpu_util_p50": utils[len(utils) // 2],
                "gpu_util_max": max(sample["gpu_util"] for sample in samples),
                "mem_used_mean_mb": sum(sample["mem_used_mb"] for sample in samples) / len(samples),
                "mem_used_max_mb": max(sample["mem_used_mb"] for sample in samples),
                "temp_mean_c": sum(sample["temp_c"] for sample in samples) / len(samples),
                "temp_max_c": max(sample["temp_c"] for sample in samples),
                "power_mean_w": sum(sample["power_w"] for sample in samples) / len(samples),
                "power_max_w": max(sample["power_w"] for sample in samples),
            }
        return out


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark real HaWoR slam stage on WebDataset factories")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--factory_dir", type=str, help="One factory directory")
    input_group.add_argument("--factory_list", type=str, help="Text file with one factory directory per line")
    input_group.add_argument("--shard_dir", type=str, help="Alias of --factory_dir for shard-style layout")
    input_group.add_argument("--shard_dir_list", type=str, help="Alias of --factory_list for shard-style layout")
    input_group.add_argument(
        "--factory_range",
        type=str,
        nargs=2,
        metavar=("START", "END"),
        help="Factory id range, inclusive",
    )

    parser.add_argument(
        "--factory_base",
        type=str,
        default="/share_data/guantianrui/datasets/Egocentric-100K/processed_v9_test_jpg",
        help="Base dir for --factory_range",
    )
    parser.add_argument("--gpus", type=str, default="0")
    add_infer_profile_arg(parser)
    parser.add_argument("--workers_per_gpu", type=int, default=1)
    parser.add_argument("--slam_workers_per_gpu", type=int, default=None)
    parser.add_argument("--num_videos", type=int, default=64, help="How many sampled videos to benchmark")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0, help="Slice discovered descriptors before sampling")
    parser.add_argument("--end", type=int, default=None, help="Slice discovered descriptors before sampling")
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
    parser.add_argument("--depth_predict_all_frames", action=argparse.BooleanOptionalAction, default=None)
    add_any4d_runtime_args(parser, include_depth_predict_all_frames=False)
    add_local_cache_args(parser)
    parser.add_argument("--max_stage_retries", type=int, default=0)
    parser.add_argument("--wave_stall_timeout_sec", type=int, default=3600)
    parser.add_argument("--enable_profiler", action="store_true")
    parser.add_argument("--sample_interval_sec", type=float, default=1.0)
    args = parser.parse_args()
    normalize_batch_infer_args(args)
    return args


def build_run_dir(args) -> Path:
    if args.run_dir:
        return Path(args.run_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "batch_runs" / f"slam_benchmark_{timestamp}"


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def collect_descriptors(args) -> List[VideoDescriptor]:
    if args.factory_dir:
        descriptors = collect_videos_from_factory(args.factory_dir)
    elif args.shard_dir:
        descriptors = collect_videos_from_factory(args.shard_dir)
    elif args.factory_list:
        descriptors = collect_videos_from_factories(_read_lines(args.factory_list))
    elif args.shard_dir_list:
        descriptors = collect_videos_from_factories(_read_lines(args.shard_dir_list))
    elif args.factory_range:
        start_factory = int(args.factory_range[0])
        end_factory = int(args.factory_range[1])
        factory_dirs = [
            os.path.join(args.factory_base, f"factory{factory_id:03d}")
            for factory_id in range(start_factory, end_factory + 1)
            if os.path.isdir(os.path.join(args.factory_base, f"factory{factory_id:03d}"))
        ]
        descriptors = collect_videos_from_factories(factory_dirs)
    else:
        raise ValueError("No input source provided")

    end = args.end if args.end is not None else len(descriptors)
    descriptors = descriptors[args.start:end]
    if not descriptors:
        raise ValueError("No descriptors available after slicing")
    return descriptors


def sample_descriptors(descriptors: List[VideoDescriptor], args) -> List[VideoDescriptor]:
    if args.num_videos < 1:
        raise ValueError("--num_videos must be >= 1")
    if args.num_videos > len(descriptors):
        raise ValueError(f"Requested {args.num_videos} videos, but only {len(descriptors)} are available")

    if args.sample_mode == "first":
        return descriptors[: args.num_videos]
    if args.sample_mode == "longest":
        ranked = sorted(descriptors, key=lambda desc: len(desc.frame_names), reverse=True)
        return ranked[: args.num_videos]

    rng = random.Random(args.sample_seed)
    return rng.sample(descriptors, args.num_videos)


def prepare_benchmark_descriptor(src_desc: VideoDescriptor, benchmark_inputs_dir: Path) -> dict:
    src_seq = Path(src_desc.seq_folder)
    start_idx, end_idx = get_track_range(src_seq, fast=True)
    src_tracks_dir = src_seq / f"tracks_{start_idx}_{end_idx}"
    src_masks = src_tracks_dir / "model_masks.npy"
    if not src_masks.exists():
        raise FileNotFoundError(f"Missing model_masks.npy for {src_desc.video_key}: {src_masks}")

    dst_seq = benchmark_inputs_dir / src_desc.video_key
    dst_tracks_dir = dst_seq / f"tracks_{start_idx}_{end_idx}"
    dst_tracks_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_masks, dst_tracks_dir / "model_masks.npy")
    (dst_seq / ".track_range").write_text(f"{start_idx},{end_idx}", encoding="utf-8")

    src_focal = src_seq / "est_focal.txt"
    if src_focal.exists():
        shutil.copy2(src_focal, dst_seq / "est_focal.txt")

    cloned = replace(src_desc, seq_folder=str(dst_seq))
    return {
        "descriptor": cloned,
        "video_key": src_desc.video_key,
        "source_seq_folder": str(src_seq),
        "benchmark_seq_folder": str(dst_seq),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "num_frames": len(src_desc.frame_names),
    }


def build_config(args, run_dir: Path, benchmark_descriptors: List[VideoDescriptor]) -> BatchRunConfig:
    config_ns = Namespace(**vars(args), stages="slam")
    return BatchRunConfig.from_namespace(
        config_ns,
        video_paths=[desc.video_key for desc in benchmark_descriptors],
        descriptors=benchmark_descriptors,
        run_dir=run_dir,
    )


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    run_dir = build_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    all_descriptors = collect_descriptors(args)
    sampled = sample_descriptors(all_descriptors, args)

    benchmark_inputs_dir = run_dir / "benchmark_inputs"
    benchmark_inputs_dir.mkdir(parents=True, exist_ok=True)

    prepared_records = []
    benchmark_descriptors = []
    for desc in sampled:
        record = prepare_benchmark_descriptor(desc, benchmark_inputs_dir)
        prepared_records.append(record)
        benchmark_descriptors.append(record["descriptor"])

    manifest = {
        "run_dir": str(run_dir),
        "num_discovered": len(all_descriptors),
        "num_sampled": len(sampled),
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed,
        "videos": [
            {key: value for key, value in record.items() if key != "descriptor"}
            for record in prepared_records
        ],
    }
    write_json(run_dir / "benchmark_manifest.json", manifest)

    config = build_config(args, run_dir / "batch_run", benchmark_descriptors)

    print("=" * 72)
    print("SLAM Benchmark Configuration")
    print("=" * 72)
    print(f"Discovered videos    : {len(all_descriptors)}")
    print(f"Sampled videos       : {len(sampled)}")
    print(f"GPUs                 : {config.gpus}")
    print(f"Workers per GPU      : default={config.workers_per_gpu}, slam={config.worker_count_for_stage('slam')}")
    print(f"Any4D batch size     : {config.any4d_batch_size}")
    print(f"Depth all frames     : {config.depth_predict_all_frames}")
    print(f"Stage3 tmp root      : {config.stage3_tmp_root}")
    print(f"Resume               : {config.resume}")
    print(f"Run dir              : {run_dir}")
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
    success_count = sum(1 for task in scheduler.state.tasks.values() if task.stage_status.get("slam") == "completed")
    failed = [
        video_path
        for video_path, task in scheduler.state.tasks.items()
        if task.stage_status.get("slam") != "completed"
    ]
    report = {
        "success": success,
        "elapsed_sec": elapsed,
        "videos_total": len(config.video_paths),
        "videos_completed": success_count,
        "videos_failed": len(failed),
        "avg_sec_per_video": elapsed / max(len(config.video_paths), 1),
        "failed_videos": failed,
        "gpu_summary": telemetry,
        "config": {
            "gpus": config.gpus,
            "workers_per_gpu": config.workers_per_gpu,
            "slam_workers_per_gpu": config.worker_count_for_stage("slam"),
            "any4d_batch_size": config.any4d_batch_size,
            "depth_predict_all_frames": config.depth_predict_all_frames,
            "stage3_tmp_root": config.stage3_tmp_root,
        },
    }
    write_json(run_dir / "benchmark_report.json", report)

    print("\nGPU telemetry")
    print("-" * 72)
    for gpu_id in config.gpus:
        info = telemetry.get(gpu_id, {})
        if info.get("samples", 0) == 0:
            print(f"GPU {gpu_id}: no samples")
            continue
        print(
            f"GPU {gpu_id} {info['name']}: "
            f"util mean={info['gpu_util_mean']:.1f}% "
            f"p50={info['gpu_util_p50']:.1f}% "
            f"max={info['gpu_util_max']:.1f}% | "
            f"mem mean={info['mem_used_mean_mb']:.0f}MB "
            f"max={info['mem_used_max_mb']:.0f}MB | "
            f"temp mean={info['temp_mean_c']:.1f}C "
            f"max={info['temp_max_c']:.1f}C | "
            f"power mean={info['power_mean_w']:.1f}W "
            f"max={info['power_max_w']:.1f}W"
        )

    print("\nBenchmark summary")
    print("-" * 72)
    print(f"Success            : {success}")
    print(f"Elapsed            : {elapsed:.2f}s")
    print(f"Videos completed   : {success_count}/{len(config.video_paths)}")
    print(f"Average sec/video  : {report['avg_sec_per_video']:.2f}")
    print(f"Manifest           : {run_dir / 'benchmark_manifest.json'}")
    print(f"Report             : {run_dir / 'benchmark_report.json'}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
