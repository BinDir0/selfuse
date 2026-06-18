"""Layer 1 - torch.compile / Inductor probe (THE highest-risk feature).

EgoVLA compiles 5 component stacks with backend=inductor. On the PPU that
lowers through the vendor's Triton port. The dangerous outcome is not a crash
but a *silent eager fallback*: it "works" but gives zero speedup. This probe
makes that visible by counting dynamo graph breaks and comparing timings.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common import IS_ACCEL, banner, device_time, guard, rel_err


class Block(nn.Module):
    """A small transformer-ish block - enough to exercise fusion."""

    def __init__(self, d=2048):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        h = self.ln(x)
        h = self.fc2(torch.nn.functional.gelu(self.fc1(h)))
        return x + h


def _graph_breaks() -> int:
    try:
        counters = torch._dynamo.utils.counters
        return int(sum(counters.get("graph_break", {}).values()))
    except Exception:  # noqa: BLE001
        return -1


def _bench_mode(model, x, mode_name, backend="inductor", mode=None) -> None:
    with guard(f"torch.compile [{mode_name}]") as r:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        eager_secs = device_time(lambda: model(x))
        eager_out = model(x)

        kwargs = {"backend": backend}
        if mode is not None:
            kwargs["mode"] = mode
        compiled = torch.compile(model, **kwargs)
        comp_out = compiled(x)  # triggers compilation
        comp_secs = device_time(lambda: compiled(x))

        breaks = _graph_breaks()
        speedup = eager_secs / comp_secs
        err = rel_err(comp_out, eager_out)
        r.metrics = {
            "eager_ms": round(eager_secs * 1e3, 3),
            "compiled_ms": round(comp_secs * 1e3, 3),
            "speedup": round(speedup, 2),
            "graph_breaks": breaks,
            "rel_err_vs_eager": round(err, 5),
        }
        r.detail = f"{speedup:.2f}x  breaks={breaks}  err={err:.4f}"
        if err > 5e-2:
            r.status, r.detail = "FAIL", f"compiled output diverged err={err:.4f}"
        elif breaks > 0:
            r.status = "FALLBACK"  # partial - some subgraphs ran eager
        elif speedup < 1.05:
            r.status = "SLOW"      # compiled but no win => suspect eager lowering


def run() -> None:
    banner("Layer 1: torch.compile / Inductor")
    if not IS_ACCEL:
        print("  (CPU only - skipping)")
        return
    model = Block().cuda().to(torch.bfloat16).eval()
    x = torch.randn(8, 512, 2048, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        _bench_mode(model, x, "default")
        _bench_mode(model, x, "max-autotune", mode="max-autotune")
        # reduce-overhead exercises CUDA graphs on top of Inductor.
        _bench_mode(model, x, "reduce-overhead (cudagraphs)", mode="reduce-overhead")


if __name__ == "__main__":
    run()
