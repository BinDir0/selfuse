"""Layer 1 - GEMM TFLOPS sweep.

Measures achieved matmul throughput across dtypes and sizes. This is the
single most informative microbench: it tells you how close the PPU gets to
its spec sheet, and whether bf16/fp16/tf32 paths are actually accelerated or
silently demoted to fp32.
"""

from __future__ import annotations

import torch

from common import DTYPES, IS_ACCEL, PEAK_TFLOPS, banner, device_time, guard

# (M, N, K) - square-ish GEMMs spanning small to large.
SHAPES = [
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]


def _tflops(m: int, n: int, k: int, secs: float) -> float:
    return (2.0 * m * n * k) / secs / 1e12


def run() -> None:
    banner("Layer 1: GEMM TFLOPS")
    if not IS_ACCEL:
        print("  (CPU only - skipping)")
        return

    # tf32 is a separate 'dtype' here: fp32 inputs with tf32 matmul enabled.
    for label in ("bf16", "fp16", "tf32", "fp32"):
        for (m, n, k) in SHAPES:
            name = f"gemm {label} {m}x{k}x{n}"
            with guard(name) as r:
                if label == "tf32":
                    torch.backends.cuda.matmul.allow_tf32 = True
                    dtype = torch.float32
                else:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    dtype = DTYPES[label]
                a = torch.randn(m, k, device="cuda", dtype=dtype)
                b = torch.randn(k, n, device="cuda", dtype=dtype)
                secs = device_time(lambda: torch.mm(a, b))
                tf = _tflops(m, n, k, secs)
                r.metrics = {"tflops": round(tf, 1), "ms": round(secs * 1e3, 3)}
                r.detail = f"{tf:6.1f} TFLOPS  ({tf / PEAK_TFLOPS * 100:4.1f}% of {PEAK_TFLOPS:g} peak)"
                # Flag if a 'fast' dtype is no faster than fp32 baseline intent.
                if label in ("bf16", "fp16") and tf < 0.15 * PEAK_TFLOPS:
                    r.status = "SLOW"
            torch.backends.cuda.matmul.allow_tf32 = False


if __name__ == "__main__":
    run()
