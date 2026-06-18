"""Worry #2 - do the GPU optimizations still pay off on the PPU?

Builds a representative transformer and times one fwd+bwd step under each
optimization toggle, reporting the speedup vs a naive fp32/eager/sdpa baseline.
If torch.compile lowers to eager or flash/flex fall back, the "optimized" rows
collapse toward the baseline - that's the answer you're looking for.
"""

from __future__ import annotations

import gc

import torch

from common import IS_ACCEL, banner, device_time, guard
from repmodel import RepModel, make_block_mask

LAYERS, D, HEADS, FFN = 8, 2048, 16, 6144
BATCH, SEQ = 4, 1024

# name -> (dtype, attn_backend, compile_mode | None, grad_ckpt)
CONFIGS = [
    ("baseline  fp32 / eager / sdpa", "fp32", "sdpa", None, False),
    ("tf32  / eager / sdpa",          "tf32", "sdpa", None, False),
    ("bf16  / eager / sdpa",          "bf16", "sdpa", None, False),
    ("bf16  / eager / flash",         "bf16", "flash", None, False),
    ("bf16  / compile / sdpa",        "bf16", "sdpa", "default", False),
    ("bf16  / compile / flex",        "bf16", "flex", "default", False),
    ("bf16  / cudagraphs / sdpa",     "bf16", "sdpa", "reduce-overhead", False),
    ("bf16  / eager / sdpa + ckpt",   "bf16", "sdpa", None, True),
]


def _make(dtype_label, backend, grad_ckpt):
    if dtype_label == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        dtype = torch.float32
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype_label]

    model = RepModel(LAYERS, D, HEADS, FFN).cuda().to(dtype).train()
    model.use_ckpt = grad_ckpt
    bm = make_block_mask(SEQ, torch.device("cuda")) if backend == "flex" else None
    model.set_backend(backend, bm)
    x = torch.randn(BATCH, SEQ, D, device="cuda", dtype=dtype)
    return model, x


def run() -> None:
    banner("opt ablation: do GPU optimizations survive on PPU?")
    if not IS_ACCEL:
        print("  (CPU only - skipping)")
        return

    baseline_ms = None
    for name, dtype, backend, cmode, ckpt in CONFIGS:
        with guard(name) as r:
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            model, x = _make(dtype, backend, ckpt)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            fwd = torch.compile(model, mode=cmode) if cmode else model

            def step():
                opt.zero_grad(set_to_none=True)
                loss = fwd(x).float().pow(2).mean()
                loss.backward()
                opt.step()

            step()  # warmup / trigger compile
            secs = device_time(step, iters=15, warmup=4)
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            toks_s = BATCH * SEQ / secs

            if baseline_ms is None:
                baseline_ms = secs * 1e3
            speedup = baseline_ms / (secs * 1e3)
            r.metrics = {"step_ms": round(secs * 1e3, 1), "tokens_s": round(toks_s),
                         "peak_gb": round(peak_gb, 2), "speedup_vs_baseline": round(speedup, 2)}
            r.detail = f"{secs * 1e3:7.1f} ms  {toks_s:8.0f} tok/s  {peak_gb:5.2f} GB  {speedup:5.2f}x"

    print("\n  (speedup is vs the naive fp32/eager/sdpa baseline on THIS device.)")
    print("  If 'compile' rows ~= eager rows, torch.compile isn't paying off on the PPU.")


if __name__ == "__main__":
    run()
