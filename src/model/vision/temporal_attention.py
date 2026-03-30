"""MEM Temporal Causal Attention for Qwen3-VL Vision Transformer.

Reference: MEM (Torne et al., 2025), arxiv 2603.03596
  - Section 3: factorized spatial-then-temporal attention
  - Appendix C: sinusoidal temporal PE with boundary condition e(0) = 0

Inserted after spatial attention in selected ViT blocks via MEMVisionBlock wrapper.
For T=1 (single image), temporal PE is all zeros and causal attention on one token
is identity, preserving pretrained single-image behavior.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from flash_attn import flash_attn_func


def build_sinusoidal_temporal_pe(max_len: int, dim: int) -> torch.Tensor:
    """Sinusoidal positional encoding with boundary condition e(0) = 0.

    Reference: MEM (Torne et al., 2025), Appendix C
    sin dimensions use standard sin(pos * freq).
    cos dimensions use cos(pos * freq) - 1.0 so that pe[0] = zeros.
    """
    pos = torch.arange(max_len).unsqueeze(1).float()
    freq = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq) - 1.0
    return pe


class TemporalCausalAttentionPass(nn.Module):
    """Per-patch temporal causal attention reusing vision block's QKV weights.

    Reference: MEM (Torne et al., 2025), Section 3
    Zero new learnable parameters. grid_thw is set as transient state
    before each ViT forward pass to determine entry structure (T, H, W).
    """

    def __init__(self, hidden_size: int, max_temporal_len: int = 32):
        super().__init__()
        pe = build_sinusoidal_temporal_pe(max_temporal_len, hidden_size)
        self.register_buffer("temporal_pe", pe, persistent=False)
        self.current_grid_thw: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        qkv_proj: nn.Linear,
        out_proj: nn.Linear,
        num_heads: int,
    ) -> torch.Tensor:
        """Compute temporal causal attention residual over all vision entries.

        Args:
            hidden_states: (total_tokens, hidden) flat across batch entries.
            qkv_proj: shared fused QKV projection from the spatial block.
            out_proj: shared output projection from the spatial block.
            num_heads: number of attention heads.

        Returns:
            Residual (total_tokens, hidden) to add to hidden_states.
        """
        grid_thw = self.current_grid_thw
        if grid_thw is None:
            return torch.zeros_like(hidden_states)

        # All single-frame entries → skip entirely
        if grid_thw[:, 0].max().item() <= 1:
            return torch.zeros_like(hidden_states)

        # Entry boundaries in the flattened token sequence
        entry_lengths = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        entry_offsets = torch.zeros(
            len(entry_lengths) + 1, dtype=torch.long, device=hidden_states.device,
        )
        entry_offsets[1:] = entry_lengths.cumsum(0)

        residuals = []
        for i in range(len(grid_thw)):
            T = int(grid_thw[i, 0].item())
            N = int(grid_thw[i, 1].item()) * int(grid_thw[i, 2].item())
            start = int(entry_offsets[i].item())
            end = int(entry_offsets[i + 1].item())
            entry_hidden = hidden_states[start:end]

            if T == 1:
                residuals.append(torch.zeros_like(entry_hidden))
                continue

            # (T*N, D) → (N, T, D) for per-patch temporal causal attention
            x = entry_hidden.reshape(T, N, -1).permute(1, 0, 2).contiguous()
            x = x + self.temporal_pe[:T].to(dtype=x.dtype, device=x.device)

            # Reuse spatial block's fused QKV projection
            qkv = qkv_proj(x)  # (N, T, 3*D)
            head_dim = qkv.shape[-1] // (3 * num_heads)
            q, k, v = (
                qkv.reshape(N, T, 3, num_heads, head_dim)
                .permute(2, 0, 1, 3, 4)
                .unbind(0)
            )  # each: (N, T, heads, head_dim)

            # flash_attn_func: (batch, seqlen, nheads, headdim) with causal mask
            attn_out = flash_attn_func(q, k, v, causal=True)
            attn_out = attn_out.reshape(N, T, -1)
            attn_out = out_proj(attn_out)  # (N, T, D)

            # (N, T, D) → (T*N, D)
            residuals.append(attn_out.permute(1, 0, 2).reshape(T * N, -1))

        return torch.cat(residuals, dim=0)


class MEMVisionBlock(nn.Module):
    """Wraps a Qwen3VLVisionBlock, appending temporal causal attention.

    Reference: MEM (Torne et al., 2025), factorized spatial-then-temporal.

    Replaces the original block in visual.blocks[i]. The original block is
    preserved as self.spatial_block, keeping gradient checkpointing and
    serialization working transparently.

    Gradient checkpointing flags are proxied to the wrapped spatial block
    so HF's selective enable/disable logic works unchanged.
    """

    def __init__(
        self,
        spatial_block: nn.Module,
        temporal_attn: TemporalCausalAttentionPass,
    ):
        super().__init__()
        self.spatial_block = spatial_block
        self.temporal_attn = temporal_attn

    # -- Proxy gradient checkpointing state to the wrapped spatial block --

    @property
    def gradient_checkpointing(self) -> bool:
        return getattr(self.spatial_block, "gradient_checkpointing", False)

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value: bool) -> None:
        self.spatial_block.gradient_checkpointing = value

    @property
    def _gradient_checkpointing_func(self):
        return getattr(self.spatial_block, "_gradient_checkpointing_func", None)

    @_gradient_checkpointing_func.setter
    def _gradient_checkpointing_func(self, value) -> None:
        self.spatial_block._gradient_checkpointing_func = value

    # -- Forward --

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # 1. Standard spatial attention + MLP (original block, unchanged)
        hidden_states = self.spatial_block(hidden_states, cu_seqlens, **kwargs)
        # 2. Temporal causal attention residual (reusing spatial block's QKV)
        hidden_states = hidden_states + self.temporal_attn(
            hidden_states,
            qkv_proj=self.spatial_block.attn.qkv,
            out_proj=self.spatial_block.attn.proj,
            num_heads=self.spatial_block.attn.num_heads,
        )
        return hidden_states
