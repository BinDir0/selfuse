"""Shared infrastructure for the PPU bench toolkit.

Vendor-agnostic: imports only torch + stdlib, never EgoVLA. The PPU presents
itself as a CUDA device through Alibaba's compatibility layer, so everything
here uses the normal ``torch.cuda`` API and works unchanged on NVIDIA too
(handy for running the same harness on an A100/H20 baseline).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import time
from typing import Any, Callable

import torch


# --------------------------------------------------------------------------- #
# Device / dtype
# --------------------------------------------------------------------------- #
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = pick_device()
IS_ACCEL = DEVICE.type == "cuda"

# Peak *dense bf16* TFLOPS used for MFU%. H20-class is ~148 TF dense bf16;
# CONFIRM against the PPU vendor spec sheet and override via env.
PEAK_TFLOPS = float(os.environ.get("PPU_PEAK_TFLOPS", "148"))

DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def sync() -> None:
    if IS_ACCEL:
        torch.cuda.synchronize()


def device_time(fn: Callable[[], Any], iters: int = 30, warmup: int = 8) -> float:
    """Median wall-seconds per call, measured with CUDA events when on device."""
    for _ in range(warmup):
        fn()
    sync()
    samples: list[float] = []
    if IS_ACCEL:
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 1e3)  # ms -> s
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Max relative error of ``out`` against an fp32 reference ``ref``."""
    out = out.detach().float()
    ref = ref.detach().float()
    denom = ref.abs().max().clamp_min(1e-6)
    return ((out - ref).abs().max() / denom).item()


# --------------------------------------------------------------------------- #
# Result recording
# --------------------------------------------------------------------------- #
# PASS    feature works and is numerically correct
# FAIL    raised, or produced wrong numbers — the loud signal we care about
# SKIP    dependency missing / not applicable (e.g. flash-attn not installed)
# SLOW    works but no/negative speedup — suspect silent eager fallback
# FALLBACK ran, but degraded (graph breaks, recompiles, partial support)
_STATUS_ICON = {
    "PASS": "✅",
    "FAIL": "❌",
    "SKIP": "➖",
    "SLOW": "\U0001f422",
    "FALLBACK": "⚠️ ",
}

_RESULTS: list["Result"] = []


@dataclasses.dataclass
class Result:
    name: str
    status: str = "PASS"
    detail: str = ""
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)
    error: str | None = None

    def line(self) -> str:
        icon = _STATUS_ICON.get(self.status, "  ")
        tail = self.detail or self.error or ""
        return f"  {icon} [{self.status:<8}] {self.name:<42} {tail}"


def record(r: Result) -> Result:
    _RESULTS.append(r)
    print(r.line(), flush=True)
    return r


def results() -> list[Result]:
    return _RESULTS


@contextlib.contextmanager
def guard(name: str):
    """Context manager that records a Result, turning any exception into FAIL.

    Inside the block, mutate the yielded Result (``.status``, ``.detail``,
    ``.metrics``) to report richer outcomes than the default PASS.
    """
    r = Result(name=name)
    try:
        yield r
    except Exception as exc:  # noqa: BLE001 - we want to capture *anything*
        r.status = "FAIL"
        r.error = f"{type(exc).__name__}: {exc}"
    record(r)


def dump(path: str, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "device": str(DEVICE),
        "peak_tflops": PEAK_TFLOPS,
        "results": [dataclasses.asdict(r) for r in _RESULTS],
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nWrote {path} ({len(_RESULTS)} results)", flush=True)


def banner(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * max(4, 72 - len(title))}", flush=True)
