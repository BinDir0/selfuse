"""A small, representative transformer stack shared by the precision-parity and
optimization-ablation harnesses.

Dims default to Qwen3-VL-2B-ish text proportions (d=2048, 16 heads, ffn=6144)
so timing/precision behavior is more representative than a toy MLP. The
attention backend is switchable so the ablation can compare sdpa/flash/flex.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_block_mask(seq: int, device: torch.device):
    from torch.nn.attention.flex_attention import create_block_mask

    def causal(b, h, qi, ki):
        return qi >= ki

    return create_block_mask(causal, B=None, H=None, Q_LEN=seq, KV_LEN=seq, device=str(device))


class Block(nn.Module):
    def __init__(self, d: int = 2048, heads: int = 16, ffn: int = 6144):
        super().__init__()
        self.heads = heads
        self.hd = d // heads
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)
        self.backend = "sdpa"          # sdpa | flash | flex
        self.block_mask = None         # set by RepModel.set_backend for flex

    def _attn(self, x):
        B, S, _ = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, S, hd)
        if self.backend == "flash":
            from flash_attn import flash_attn_func

            o = flash_attn_func(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
                causal=True,
            ).transpose(1, 2)
        elif self.backend == "flex":
            from torch.nn.attention.flex_attention import flex_attention

            o = flex_attention(q, k, v, block_mask=self.block_mask)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, S, -1)
        return self.proj(o)

    def forward(self, x):
        x = x + self._attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class RepModel(nn.Module):
    def __init__(self, layers: int = 8, d: int = 2048, heads: int = 16, ffn: int = 6144):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d, heads, ffn) for _ in range(layers)])
        self.use_ckpt = False

    def set_backend(self, backend: str, block_mask=None) -> None:
        for b in self.blocks:
            b.backend = backend
            b.block_mask = block_mask

    def forward(self, x):
        for b in self.blocks:
            if self.use_ckpt and self.training:
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(b, x, use_reentrant=False)
            else:
                x = b(x)
        return x
