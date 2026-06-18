"""Tiny self-contained helpers so each thirdparty probe can be scp'd alone."""

from __future__ import annotations

import time

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_ACCEL = DEVICE.type == "cuda"


def sync() -> None:
    if IS_ACCEL:
        torch.cuda.synchronize()


def timed(fn, iters: int = 5, warmup: int = 2) -> float:
    """Median seconds per call."""
    for _ in range(warmup):
        fn()
    sync()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        out.append(time.perf_counter() - t0)
    out.sort()
    return out[len(out) // 2]


def section(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * max(4, 64 - len(title))}", flush=True)


def ok(msg: str) -> None:
    print(f"  ✅ {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"  ❌ {msg}", flush=True)


def skip(msg: str) -> None:
    print(f"  ➖ SKIP: {msg}", flush=True)
