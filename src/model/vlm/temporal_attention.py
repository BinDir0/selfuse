"""MEM Temporal Causal Attention for Qwen3-VL Vision Transformer.

Reference: MEM (Torne et al., 2025), arxiv 2603.03596
  - Section 3: factorized temporal-then-spatial attention
  - Appendix C: sinusoidal temporal PE with boundary condition e(0) = 0

Inserted BEFORE spatial attention in selected ViT blocks via MEMVisionBlock
wrapper. Temporal attention shares the spatial block's norm1, QKV, and
output projection — zero new learnable parameters.

Critical assumption (owned by the dataset / collator layer): every entry
in the batch has been padded / resized to the same temporal length T and
spatial token count N. Under this assumption MEM does pure
view + SDPA + (optional) element-wise mask multiply.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def build_sinusoidal_temporal_pe(
    max_len: int, dim: int, base: float = 10000.0,
) -> torch.Tensor:
    """Sinusoidal positional encoding with boundary condition e(0) = 0.

    Reference: MEM (Torne et al., 2025), Appendix C
    sin dimensions use standard sin(pos * freq).
    cos dimensions use cos(pos * freq) - 1.0 so that pe[0] = zeros.
    """
    pos = torch.arange(max_len).unsqueeze(1).float()
    freq = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(base) / dim))
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq) - 1.0
    return pe


class TemporalCausalAttention(nn.Module):
    """Per-patch temporal causal attention; weights shared with spatial block.

    Reference: MEM (Torne et al., 2025), Section 3.

    norm / qkv_proj / out_proj are passed in at forward time so this module
    registers no parameters of its own (FSDP wraps each spatial block once).

    Input is the flat patch sequence (B*T*N, D) from Qwen3-VL's ViT.
    `grid_thw.shape[0]` gives B (Python int); T is `self.T`; N is recovered
    via view(-1). Dataset / collator must pad VLM to the same (T, N) as VLA
    — assertions are intentionally omitted to keep the path sync-free.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        num_frames: int,
        mask_non_vla: bool = False,
        max_temporal_len: int = 32,
        base: float = 10000.0,
    ):
        super().__init__()
        pe = build_sinusoidal_temporal_pe(max_temporal_len, hidden_size, base)
        self.register_buffer("temporal_pe", pe, persistent=False)
        self.num_heads = num_heads
        self.T = num_frames
        self.mask_non_vla = mask_non_vla

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        norm: nn.Module,
        qkv_proj: nn.Linear,
        out_proj: nn.Linear,
        is_vla_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        T = self.T
        D = hidden_states.shape[-1]
        H = self.num_heads
        head_dim = D // H
        B = int(grid_thw.shape[0])

        # (B*T*N, D) -> (B, T, N, D) -> (B, N, T, D)
        x = hidden_states.view(B, T, -1, D).permute(0, 2, 1, 3)

        # Add temporal PE before norm so qkv_proj sees LN(z + e(t)).
        # Boundary condition e(0)=0 keeps t=0 frames identical to spatial path.
        x = x + self.temporal_pe[:T].to(dtype=x.dtype)
        x = norm(x)

        # qkv: (B, N, T, 3, H, head_dim)
        qkv = qkv_proj(x).view(B, -1, T, 3, H, head_dim)
        # each: (B, N, T, H, head_dim) -> (B, N, H, T, head_dim)
        q, k, v = (t.transpose(-3, -2) for t in qkv.unbind(3))

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # (B, N, H, T, head_dim) -> (B, N, T, D) -> (B, T, N, D) -> (B*T*N, D)
        attn_out = attn_out.transpose(-3, -2).reshape(B, -1, T, D)
        attn_out = out_proj(attn_out).permute(0, 2, 1, 3).reshape(-1, D)

        # Optional: zero out non-VLA entries' contribution to preserve pretrained
        # VLM behavior. Element-wise multiply only — no scatter, no sync.
        if self.mask_non_vla and is_vla_mask is not None:
            tokens_per_entry = attn_out.shape[0] // B
            mask_per_token = is_vla_mask.repeat_interleave(tokens_per_entry).to(
                dtype=attn_out.dtype
            )
            attn_out = attn_out * mask_per_token.unsqueeze(-1)

        return attn_out


class MEMVisionBlock(nn.Module):
    """Wraps a Qwen3VLVisionBlock with temporal-then-spatial attention.

    Reference: MEM (Torne et al., 2025).

    Temporal runs first so the shared norm1/QKV/proj see the same input
    distribution as in pretraining. `mem_grid_thw` and `is_vla_mask` are
    routed in via HF's **kwargs forwarding chain (get_video_features →
    visual.forward → block); spatial Qwen3VLVisionBlock ignores them.
    `mem_grid_thw` (not `grid_thw`) is used as the kwarg name to avoid
    colliding with visual.forward's positional `grid_thw`.

    Per-layer checkpointing toggles must unwrap to `spatial_block`; HF's
    apply()-based enable already finds it via the module tree.
    """

    def __init__(
        self,
        spatial_block: nn.Module,
        temporal_attn: TemporalCausalAttention,
    ):
        super().__init__()
        self.spatial_block = spatial_block
        self.temporal_attn = temporal_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        mem_grid_thw: torch.Tensor | None = None,
        is_vla_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.temporal_attn(
            hidden_states,
            grid_thw=mem_grid_thw,
            norm=self.spatial_block.norm1,
            qkv_proj=self.spatial_block.attn.qkv,
            out_proj=self.spatial_block.attn.proj,
            is_vla_mask=is_vla_mask,
        )
        hidden_states = self.spatial_block(hidden_states, cu_seqlens, **kwargs)
        return hidden_states
