"""Layer 1 - public-model smoke (diverse, NOT EgoVLA).

Confirms the general DL stack trains end-to-end on the PPU using well-known
models, so a later EgoVLA failure can be attributed to the repo rather than
the platform. Each model is independent and skips cleanly if its dep is absent.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common import IS_ACCEL, banner, device_time, guard


def run() -> None:
    banner("Layer 1: public-model smoke")
    if not IS_ACCEL:
        print("  (CPU only - skipping)")
        return

    # ---- torchvision ResNet-18 train step ----
    with guard("torchvision resnet18 fwd+bwd") as r:
        try:
            import torchvision
        except ImportError:
            r.status, r.detail = "SKIP", "torchvision absent"
        else:
            net = torchvision.models.resnet18(weights=None).cuda()
            opt = torch.optim.SGD(net.parameters(), lr=0.01)
            x = torch.randn(32, 3, 224, 224, device="cuda")
            y = torch.randint(0, 1000, (32,), device="cuda")

            def step():
                opt.zero_grad()
                loss = nn.functional.cross_entropy(net(x), y)
                loss.backward()
                opt.step()
                return loss

            loss = step()
            secs = device_time(step, iters=10, warmup=3)
            r.metrics = {"loss": round(loss.item(), 4), "step_ms": round(secs * 1e3, 1)}
            r.detail = f"loss={loss.item():.3f}  {secs * 1e3:.0f} ms/step"

    # ---- HuggingFace tiny causal LM train step ----
    with guard("HF tiny GPT2 fwd+bwd (bf16)") as r:
        try:
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError:
            r.status, r.detail = "SKIP", "transformers absent"
        else:
            cfg = GPT2Config(n_layer=4, n_head=8, n_embd=512, vocab_size=50257, n_positions=512)
            model = GPT2LMHeadModel(cfg).cuda().to(torch.bfloat16)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            ids = torch.randint(0, 50257, (4, 256), device="cuda")

            def step():
                opt.zero_grad()
                loss = model(input_ids=ids, labels=ids).loss
                loss.backward()
                opt.step()
                return loss

            loss = step()
            secs = device_time(step, iters=10, warmup=3)
            r.metrics = {"loss": round(loss.item(), 4), "step_ms": round(secs * 1e3, 1)}
            r.detail = f"loss={loss.item():.3f}  {secs * 1e3:.0f} ms/step"
            if not torch.isfinite(loss).item():
                r.status = "FAIL"

    # ---- nn.Transformer (zero-dependency fallback) ----
    with guard("nn.TransformerEncoder fwd+bwd (bf16)") as r:
        layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        enc = nn.TransformerEncoder(layer, num_layers=4).cuda().to(torch.bfloat16)
        x = torch.randn(8, 256, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        out = enc(x)
        out.float().pow(2).mean().backward()
        r.detail = f"out={tuple(out.shape)}  grad_finite={bool(torch.isfinite(x.grad).all())}"


if __name__ == "__main__":
    run()
