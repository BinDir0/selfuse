"""Layer 1 - HBM memory bandwidth (copy / triad).

A model that is memory-bound (attention, layernorm, elementwise) is gated by
this number, not by GEMM TFLOPS. Worth knowing before you blame the model.
"""

from __future__ import annotations

import torch

from common import IS_ACCEL, banner, device_time, guard

# ~256 M fp32 elements = 1 GiB per buffer.
N = 256 * 1024 * 1024


def run() -> None:
    banner("Layer 1: memory bandwidth")
    if not IS_ACCEL:
        print("  (CPU only - skipping)")
        return

    a = torch.randn(N, device="cuda")
    b = torch.empty_like(a)
    bytes_moved = a.numel() * a.element_size()

    with guard("membw copy (read+write)") as r:
        secs = device_time(lambda: b.copy_(a))
        gbs = 2 * bytes_moved / secs / 1e9  # read a + write b
        r.metrics = {"GB/s": round(gbs, 1)}
        r.detail = f"{gbs:7.1f} GB/s"

    c = torch.randn(N, device="cuda")
    with guard("membw triad (a = b + 2*c)") as r:
        secs = device_time(lambda: torch.add(b, c, alpha=2.0, out=a))
        gbs = 3 * bytes_moved / secs / 1e9  # read b + read c + write a
        r.metrics = {"GB/s": round(gbs, 1)}
        r.detail = f"{gbs:7.1f} GB/s"


if __name__ == "__main__":
    run()
