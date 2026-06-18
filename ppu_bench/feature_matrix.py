"""Layer 2 - feature compatibility matrix.

Targeted on/off probes for the "latest DL features" that the GEMM/ops/compile
probes don't already cover. Each lands as PASS / FAIL / SKIP so you get a
single table answering "which features can I actually use on the PPU".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common import IS_ACCEL, banner, guard, rel_err


def run() -> None:
    banner("Layer 2: feature compatibility matrix")
    if not IS_ACCEL:
        print("  (CPU only - skipping)")
        return

    # ---- AMP autocast (bf16) ----
    with guard("amp autocast bf16") as r:
        lin = nn.Linear(1024, 1024).cuda()
        x = torch.randn(64, 1024, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = lin(x)
        r.detail = f"out dtype={out.dtype}"
        if out.dtype != torch.bfloat16:
            r.status, r.detail = "FALLBACK", f"autocast did not cast (dtype={out.dtype})"

    # ---- fused AdamW ----
    with guard("fused AdamW") as r:
        p = nn.Parameter(torch.randn(4096, 4096, device="cuda"))
        opt = torch.optim.AdamW([p], lr=1e-3, fused=True)
        p.grad = torch.randn_like(p)
        opt.step()
        r.detail = "fused=True accepted + stepped"

    # ---- gradient checkpointing ----
    with guard("gradient checkpointing") as r:
        from torch.utils.checkpoint import checkpoint

        block = nn.Sequential(nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 1024)).cuda()
        x = torch.randn(32, 1024, device="cuda", requires_grad=True)
        out = checkpoint(block, x, use_reentrant=False)
        out.sum().backward()
        r.detail = f"grad finite={bool(torch.isfinite(x.grad).all())}"

    # ---- CUDA graphs (raw capture, not via compile) ----
    with guard("raw CUDA graph capture") as r:
        try:
            s = torch.cuda.Stream()
            x = torch.randn(1024, 1024, device="cuda")
            w = torch.randn(1024, 1024, device="cuda")
            # warmup on side stream
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    y = x @ w
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y = x @ w  # noqa: F841
            g.replay()
            torch.cuda.synchronize()
            r.detail = "capture + replay ok"
        except Exception as exc:  # noqa: BLE001
            r.status, r.error = "FAIL", f"{type(exc).__name__}: {exc}"

    # ---- torch.profiler + FLOP counting (EgoVLA's MFU path uses this) ----
    with guard("torch.profiler CUDA + with_flops") as r:
        from torch.profiler import ProfilerActivity, profile

        a = torch.randn(2048, 2048, device="cuda")
        b = torch.randn(2048, 2048, device="cuda")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True) as prof:
            (a @ b).sum().item()
        events = prof.key_averages()
        total_flops = sum(e.flops for e in events if e.flops > 0)
        has_cuda = any(getattr(e, "self_device_time_total", 0) > 0 for e in events)
        r.metrics = {"flops_reported": total_flops, "cuda_events": has_cuda}
        r.detail = f"flops={total_flops}  cuda_timing={has_cuda}"
        if not has_cuda:
            r.status, r.detail = "FALLBACK", "no CUDA timing (CUPTI?) - MFU% will be blank"

    # ---- bitsandbytes 8-bit (EgoVLA: optional 4-bit load path only) ----
    with guard("bitsandbytes 8-bit optimizer") as r:
        try:
            import bitsandbytes as bnb

            p = nn.Parameter(torch.randn(2048, 2048, device="cuda"))
            opt = bnb.optim.Adam8bit([p], lr=1e-3)
            p.grad = torch.randn_like(p)
            opt.step()
            r.detail = "Adam8bit stepped (off EgoVLA critical path anyway)"
        except ImportError:
            r.status, r.detail = "SKIP", "bitsandbytes absent - fine, disable in EgoVLA"
        except Exception as exc:  # noqa: BLE001
            r.status, r.detail = "FAIL", f"{type(exc).__name__}: {exc} - disable in EgoVLA"


if __name__ == "__main__":
    run()
