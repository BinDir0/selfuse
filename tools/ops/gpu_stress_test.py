#!/usr/bin/env python3
"""Real GPU stress/benchmark script for capacity validation.

This script intentionally runs heavy CUDA matmul workloads to measure:
  - sustained GPU utilization
  - approximate GEMM throughput
  - memory usage / temperature / power draw via nvidia-smi

It is intended for machine qualification and scheduling experiments, not for
pipeline execution.

Example:
  python tools/ops/gpu_stress_test.py --gpus 0,1,2,3 --duration-sec 180

More aggressive:
  python tools/ops/gpu_stress_test.py \
    --gpus 0,1,2,3 \
    --workers-per-gpu 2 \
    --matrix-size 8192 \
    --dtype bf16 \
    --duration-sec 300 \
    --allow-tf32
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, List

import torch


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass
class WorkerResult:
    gpu_id: int
    worker_id: int
    matrix_size: int
    dtype: str
    elapsed_sec: float
    iterations: int
    approx_tflops: float
    error: str = ""


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
        query = ",".join(
            [
                "index",
                "name",
                "utilization.gpu",
                "utilization.memory",
                "memory.used",
                "memory.total",
                "temperature.gpu",
                "power.draw",
            ]
        )
        cmd = [
            "nvidia-smi",
            f"--id={gpu_arg}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]

        while not self._stop.is_set():
            timestamp = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().splitlines():
                        parts = [part.strip() for part in line.split(",")]
                        if len(parts) != 8:
                            continue
                        gpu_id = int(parts[0])
                        if gpu_id not in self.samples:
                            continue
                        self.samples[gpu_id].append(
                            {
                                "time": timestamp,
                                "name": parts[1],
                                "gpu_util": float(parts[2]),
                                "mem_util": float(parts[3]),
                                "mem_used_mb": float(parts[4]),
                                "mem_total_mb": float(parts[5]),
                                "temp_c": float(parts[6]),
                                "power_w": float(parts[7]),
                            }
                        )
            except Exception:
                pass
            self._stop.wait(self.interval_sec)

    def summary(self) -> Dict[int, dict]:
        output = {}
        for gpu_id, samples in self.samples.items():
            if not samples:
                output[gpu_id] = {"samples": 0}
                continue

            def mean(key: str) -> float:
                return sum(sample[key] for sample in samples) / len(samples)

            def peak(key: str) -> float:
                return max(sample[key] for sample in samples)

            utils = sorted(sample["gpu_util"] for sample in samples)
            output[gpu_id] = {
                "samples": len(samples),
                "name": samples[0]["name"],
                "gpu_util_mean": mean("gpu_util"),
                "gpu_util_p50": utils[len(utils) // 2],
                "gpu_util_max": peak("gpu_util"),
                "mem_used_mean_mb": mean("mem_used_mb"),
                "mem_used_max_mb": peak("mem_used_mb"),
                "temp_mean_c": mean("temp_c"),
                "temp_max_c": peak("temp_c"),
                "power_mean_w": mean("power_w"),
                "power_max_w": peak("power_w"),
            }
        return output


def _format_bytes_gb(num_bytes: float) -> float:
    return num_bytes / (1024 ** 3)


def _estimate_tensor_bytes(matrix_size: int, dtype: torch.dtype) -> int:
    element_size = torch.tensor([], dtype=dtype).element_size()
    return matrix_size * matrix_size * element_size


def _worker_main(
    gpu_id: int,
    worker_id: int,
    matrix_size: int,
    dtype_name: str,
    warmup_sec: float,
    duration_sec: float,
    sync_every: int,
    allow_tf32: bool,
    result_queue: mp.Queue,
):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    start_time = time.time()

    try:
        dtype = DTYPE_MAP[dtype_name]
        torch.cuda.set_device(gpu_id)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.set_grad_enabled(False)

        device = torch.device(f"cuda:{gpu_id}")
        a = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
        b = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
        c = torch.empty_like(a)

        warmup_end = start_time + warmup_sec
        while time.time() < warmup_end:
            torch.matmul(a, b, out=c)
            a, b, c = b, c, a
        torch.cuda.synchronize(device)

        measured_start = time.time()
        measured_end = measured_start + duration_sec
        iterations = 0

        while time.time() < measured_end:
            for _ in range(sync_every):
                torch.matmul(a, b, out=c)
                a, b, c = b, c, a
            torch.cuda.synchronize(device)
            iterations += sync_every

        torch.cuda.synchronize(device)
        elapsed = time.time() - measured_start
        flops_per_iter = 2.0 * (matrix_size ** 3)
        approx_tflops = (iterations * flops_per_iter) / max(elapsed, 1e-6) / 1e12
        checksum = float(a[0, 0].item()) + float(b[0, 0].item())
        if math.isnan(checksum):
            raise RuntimeError("checksum became NaN during stress test")

        result_queue.put(
            asdict(
                WorkerResult(
                    gpu_id=gpu_id,
                    worker_id=worker_id,
                    matrix_size=matrix_size,
                    dtype=dtype_name,
                    elapsed_sec=elapsed,
                    iterations=iterations,
                    approx_tflops=approx_tflops,
                )
            )
        )
    except Exception as exc:
        result_queue.put(
            asdict(
                WorkerResult(
                    gpu_id=gpu_id,
                    worker_id=worker_id,
                    matrix_size=matrix_size,
                    dtype=dtype_name,
                    elapsed_sec=max(time.time() - start_time, 0.0),
                    iterations=0,
                    approx_tflops=0.0,
                    error=str(exc),
                )
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Real GPU stress/benchmark tool")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1, help="Worker processes to launch per GPU")
    parser.add_argument("--matrix-size", type=int, default=8192, help="Square GEMM size N for N x N matmul")
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP.keys()), default="bf16")
    parser.add_argument("--duration-sec", type=float, default=120.0, help="Measured run time per worker")
    parser.add_argument("--warmup-sec", type=float, default=10.0, help="Warmup time before measuring")
    parser.add_argument("--sync-every", type=int, default=8, help="CUDA sync interval in matmul iterations")
    parser.add_argument("--sample-interval-sec", type=float, default=1.0, help="nvidia-smi sampling interval")
    parser.add_argument("--allow-tf32", action="store_true", help="Enable TF32 matmul on supported GPUs")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save summary JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    gpu_ids = [int(part) for part in args.gpus.split(",") if part.strip()]
    if not gpu_ids:
        raise SystemExit("No GPU ids provided")
    if args.workers_per_gpu < 1:
        raise SystemExit("--workers-per-gpu must be >= 1")
    if args.matrix_size < 512:
        raise SystemExit("--matrix-size must be >= 512")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    visible_count = torch.cuda.device_count()
    bad_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= visible_count]
    if bad_gpu_ids:
        raise SystemExit(f"Invalid GPU ids {bad_gpu_ids}; visible CUDA device count is {visible_count}")

    dtype = DTYPE_MAP[args.dtype]
    per_tensor_bytes = _estimate_tensor_bytes(args.matrix_size, dtype)
    approx_working_set_bytes = per_tensor_bytes * 3 * args.workers_per_gpu

    print("=" * 72)
    print("GPU Stress Test")
    print("=" * 72)
    print(f"GPUs              : {gpu_ids}")
    print(f"Workers per GPU   : {args.workers_per_gpu}")
    print(f"Matrix size       : {args.matrix_size} x {args.matrix_size}")
    print(f"Dtype             : {args.dtype}")
    print(f"Warmup            : {args.warmup_sec:.1f}s")
    print(f"Duration          : {args.duration_sec:.1f}s")
    print(f"Sync every        : {args.sync_every} iterations")
    print(f"TF32 allowed      : {args.allow_tf32}")
    print(f"Approx tensor mem : { _format_bytes_gb(per_tensor_bytes):.2f} GiB per matrix")
    print(f"Approx working set: { _format_bytes_gb(approx_working_set_bytes):.2f} GiB per GPU")
    print("=" * 72)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    sampler = GPUSampler(gpu_ids=gpu_ids, interval_sec=args.sample_interval_sec)

    sampler.start()
    launch_time = time.time()
    try:
        for gpu_id in gpu_ids:
            for worker_id in range(args.workers_per_gpu):
                proc = ctx.Process(
                    target=_worker_main,
                    args=(
                        gpu_id,
                        worker_id,
                        args.matrix_size,
                        args.dtype,
                        args.warmup_sec,
                        args.duration_sec,
                        args.sync_every,
                        args.allow_tf32,
                        result_queue,
                    ),
                )
                proc.start()
                processes.append(proc)

        deadline = launch_time + args.warmup_sec + args.duration_sec + 120.0
        while any(proc.is_alive() for proc in processes):
            if time.time() > deadline:
                raise TimeoutError("Timed out waiting for GPU stress workers to finish")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nInterrupted, terminating workers...")
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        raise
    finally:
        for proc in processes:
            proc.join(timeout=5.0)
        sampler.stop()

    worker_results = []
    while True:
        try:
            worker_results.append(result_queue.get_nowait())
        except queue.Empty:
            break

    worker_results.sort(key=lambda item: (item["gpu_id"], item["worker_id"]))
    sampler_summary = sampler.summary()

    print("\nWorker results")
    print("-" * 72)
    for item in worker_results:
        if item["error"]:
            print(
                f"GPU {item['gpu_id']} worker {item['worker_id']}: ERROR | {item['error']}"
            )
            continue
        print(
            f"GPU {item['gpu_id']} worker {item['worker_id']}: "
            f"{item['approx_tflops']:.2f} TFLOP/s | "
            f"{item['iterations']} iterations | "
            f"{item['elapsed_sec']:.1f}s"
        )

    print("\nGPU telemetry")
    print("-" * 72)
    for gpu_id in gpu_ids:
        info = sampler_summary.get(gpu_id, {})
        if info.get("samples", 0) == 0:
            print(f"GPU {gpu_id}: no nvidia-smi samples collected")
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

    aggregate_by_gpu = {}
    for item in worker_results:
        if item["error"]:
            continue
        aggregate_by_gpu.setdefault(item["gpu_id"], 0.0)
        aggregate_by_gpu[item["gpu_id"]] += item["approx_tflops"]

    if aggregate_by_gpu:
        print("\nApprox aggregate throughput")
        print("-" * 72)
        for gpu_id in gpu_ids:
            if gpu_id in aggregate_by_gpu:
                print(f"GPU {gpu_id}: {aggregate_by_gpu[gpu_id]:.2f} TFLOP/s total across workers")

    output = {
        "args": vars(args),
        "worker_results": worker_results,
        "gpu_summary": sampler_summary,
        "aggregate_tflops_by_gpu": aggregate_by_gpu,
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved summary JSON to {args.output_json}")


if __name__ == "__main__":
    main()
