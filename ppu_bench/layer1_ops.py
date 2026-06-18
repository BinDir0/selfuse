"""Layer 1 - attention backends + common ops, correctness *and* speed.

These are the exact accelerated paths EgoVLA depends on:
  * vision attention  -> flash_attn (flash_attn_varlen_func)
  * text attention    -> flex_attention
plus the SDPA fallback the repo drops to. Every op is checked numerically
against an fp32 reference, so a kernel that *runs but returns garbage* on the
PPU compat layer shows up as FAIL rather than a silent wrong-result.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from common import IS_ACCEL, banner, device_time, guard, rel_err

B, H, S, D = 4, 16, 2048, 64
BF16_TOL = 3e-2  # bf16 attention vs fp32 reference


def _ref_causal_attention(q, k, v) -> torch.Tensor:
    """fp32 reference: softmax(QK^T/sqrt(d) + causal) V. Shapes (B,H,S,D)."""
    qf, kf, vf = q.float(), k.float(), v.float()
    scores = (qf @ kf.transpose(-1, -2)) / math.sqrt(D)
    mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ vf


def run() -> None:
    banner("Layer 1: ops (attention backends + elementwise)")
    if not IS_ACCEL:
        print("  (CPU only - skipping)")
        return

    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    ref = _ref_causal_attention(q, k, v)

    # ---- SDPA (the safe fallback EgoVLA can switch to) ----
    sdpa_secs = None
    with guard("attn sdpa (causal, bf16)") as r:
        fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = fn()
        err = rel_err(out, ref)
        sdpa_secs = device_time(fn)
        r.metrics = {"rel_err": round(err, 5), "ms": round(sdpa_secs * 1e3, 3)}
        r.detail = f"rel_err={err:.4f}  {sdpa_secs * 1e3:.2f} ms"
        if err > BF16_TOL:
            r.status, r.detail = "FAIL", f"WRONG NUMBERS rel_err={err:.4f}"

    # ---- FlashAttention (EgoVLA vision path) ----
    with guard("attn flash_attn (causal, bf16)") as r:
        try:
            from flash_attn import flash_attn_func
        except Exception as exc:  # noqa: BLE001
            r.status = "SKIP"
            r.detail = f"flash_attn not importable ({type(exc).__name__}) - use PPU vendor build"
        else:
            # flash_attn wants (B, S, H, D)
            qf = q.transpose(1, 2).contiguous()
            kf = k.transpose(1, 2).contiguous()
            vf = v.transpose(1, 2).contiguous()
            fn = lambda: flash_attn_func(qf, kf, vf, causal=True)
            out = fn().transpose(1, 2)  # back to (B,H,S,D)
            err = rel_err(out, ref)
            secs = device_time(fn)
            r.metrics = {"rel_err": round(err, 5), "ms": round(secs * 1e3, 3),
                         "speedup_vs_sdpa": round(sdpa_secs / secs, 2) if sdpa_secs else None}
            r.detail = f"rel_err={err:.4f}  {secs * 1e3:.2f} ms  ({sdpa_secs / secs:.2f}x sdpa)"
            if err > BF16_TOL:
                r.status, r.detail = "FAIL", f"WRONG NUMBERS rel_err={err:.4f}"

    # ---- flex_attention (EgoVLA text path) ----
    with guard("attn flex_attention (causal, bf16, compiled)") as r:
        try:
            from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        except Exception as exc:  # noqa: BLE001
            r.status = "SKIP"
            r.detail = f"flex_attention unavailable ({type(exc).__name__})"
        else:
            def causal(b, h, qi, ki):
                return qi >= ki

            block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=S, KV_LEN=S)
            flex = torch.compile(flex_attention)
            fn = lambda: flex(q, k, v, block_mask=block_mask)
            out = fn()
            err = rel_err(out, ref)
            secs = device_time(fn)
            r.metrics = {"rel_err": round(err, 5), "ms": round(secs * 1e3, 3),
                         "speedup_vs_sdpa": round(sdpa_secs / secs, 2) if sdpa_secs else None}
            r.detail = f"rel_err={err:.4f}  {secs * 1e3:.2f} ms  ({sdpa_secs / secs:.2f}x sdpa)"
            if err > BF16_TOL:
                r.status, r.detail = "FAIL", f"WRONG NUMBERS rel_err={err:.4f}"

    # ---- elementwise / normalization ----
    x = torch.randn(B * S, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    with guard("layer_norm (bf16)") as r:
        fn = lambda: F.layer_norm(x, (4096,), w, bias)
        out = fn()
        err = rel_err(out, F.layer_norm(x.float(), (4096,), w.float(), bias.float()))
        secs = device_time(fn)
        r.metrics = {"rel_err": round(err, 5), "ms": round(secs * 1e3, 3)}
        r.detail = f"rel_err={err:.4f}  {secs * 1e3:.3f} ms"
        if err > 5e-2:
            r.status = "FAIL"

    with guard("softmax (bf16)") as r:
        fn = lambda: F.softmax(x, dim=-1)
        out = fn()
        err = rel_err(out, F.softmax(x.float(), dim=-1))
        secs = device_time(fn)
        r.metrics = {"rel_err": round(err, 5), "ms": round(secs * 1e3, 3)}
        r.detail = f"rel_err={err:.4f}  {secs * 1e3:.3f} ms"
        if err > 5e-2:
            r.status = "FAIL"


if __name__ == "__main__":
    run()
