"""MEM Temporal Causal Attention for Qwen3-VL Vision Transformer.

Reference: MEM (Torne et al., 2025), arxiv 2603.03596
  - Section 3: factorized temporal-then-spatial attention
  - Appendix C: sinusoidal temporal PE with boundary condition e(0) = 0

Inserted BEFORE spatial attention in selected ViT blocks via MEMVisionBlock
wrapper. Temporal attention shares the spatial block's norm1, QKV, and
output projection — zero new learnable parameters.

For T=1 (single image), temporal PE is all zeros and causal attention on one
token is identity, preserving pretrained single-image behavior.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from flash_attn import flash_attn_varlen_func


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


class TemporalCausalAttentionPass(nn.Module):
    """Per-patch temporal causal attention sharing spatial block's weights.

    Reference: MEM (Torne et al., 2025), Section 3
    Zero new learnable parameters. Binds norm1, QKV, output projection, and
    num_heads from the spatial block at construction time.

    grid_thw is set as transient state before each ViT forward pass to
    determine entry structure (T, H, W).
    """

    def __init__(
        self,
        spatial_block: nn.Module,
        hidden_size: int,
        max_temporal_len: int = 32,
        base: float = 10000.0,
    ):
        super().__init__()
        pe = build_sinusoidal_temporal_pe(max_temporal_len, hidden_size, base)
        self.register_buffer("temporal_pe", pe, persistent=False)
        self.current_grid_thw: torch.Tensor | None = None

        # Share spatial block's weights (reference, not copy)
        self.norm = spatial_block.norm1
        self.qkv_proj = spatial_block.attn.qkv
        self.out_proj = spatial_block.attn.proj
        self.num_heads = spatial_block.attn.num_heads

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute temporal causal attention residual over all vision entries.

        Fully vectorized — no Python loops. All multi-frame entries are
        gathered, transposed, and processed in a single flash_attn_varlen_func
        kernel launch, then scattered back.

        Requires all multi-frame entries to share the same T and N (guaranteed
        when all images are resized to the same target size and the observation
        horizon is fixed).

        Args:
            hidden_states: (total_tokens, hidden) flat across batch entries.

        Returns:
            Residual (total_tokens, hidden) to add to hidden_states.
        """
        grid_thw = self.current_grid_thw
        if grid_thw is None:
            return torch.zeros_like(hidden_states)

        # All single-frame entries -> skip entirely
        if grid_thw[:, 0].max().item() <= 1:
            return torch.zeros_like(hidden_states)

        # Identify multi-frame entries (T > 1)
        multi_mask = grid_thw[:, 0] > 1
        if not multi_mask.any():
            return torch.zeros_like(hidden_states)

        multi_indices = multi_mask.nonzero(as_tuple=False).squeeze(-1)
        multi_grid = grid_thw[multi_indices]  # (M, 3)
        M = multi_indices.shape[0]
        T = multi_grid[0, 0].item()
        N = (multi_grid[0, 1] * multi_grid[0, 2]).item()

        assert (multi_grid[:, 0] == T).all() and (
            (multi_grid[:, 1] * multi_grid[:, 2]) == N
        ).all(), (
            f"All multi-frame entries must share the same T={T} and N={N} "
            "for batched temporal attention"
        )

        # Entry offsets in the flat token sequence
        entry_lengths = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        entry_offsets = torch.zeros(
            len(entry_lengths) + 1, dtype=torch.long, device=hidden_states.device,
        )
        entry_offsets[1:] = entry_lengths.cumsum(0)
        starts = entry_offsets[multi_indices]  # (M,)

        # Gather all multi-frame tokens via flat index broadcast
        token_offsets = torch.arange(T * N, device=hidden_states.device)  # (T*N,)
        flat_indices = (starts.unsqueeze(1) + token_offsets.unsqueeze(0)).reshape(-1)

        # (M*T*N, D) -> (M, T, N, D) -> (M, N, T, D)
        D = hidden_states.shape[-1]
        x = hidden_states[flat_indices].reshape(M, T, N, D).permute(0, 2, 1, 3).contiguous()

        # Apply spatial block's norm1, then add temporal PE
        x = self.norm(x)
        x = x + self.temporal_pe[:T].to(dtype=x.dtype, device=x.device)

        # (M, N, T, D) -> (M*N*T, D) for varlen attention
        x = x.reshape(M * N * T, D)

        # Shared QKV projection
        head_dim = self.qkv_proj.out_features // (3 * self.num_heads)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.reshape(-1, 3, self.num_heads, head_dim).unbind(1)

        # cu_seqlens: M*N sequences each of length T
        cu_seqlens = torch.arange(
            0, (M * N + 1) * T, T,
            dtype=torch.int32, device=x.device,
        )

        # Single kernel launch for all per-patch temporal sequences
        attn_out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=T,
            max_seqlen_k=T,
            causal=True,
        )
        attn_out = self.out_proj(attn_out.reshape(M * N * T, -1))

        # (M*N*T, D) -> (M, N, T, D) -> (M, T, N, D) -> (M*T*N, D)
        attn_out = attn_out.reshape(M, N, T, D).permute(0, 2, 1, 3).reshape(-1, D)

        # Scatter back into full-size residual
        residual = torch.zeros_like(hidden_states)
        residual[flat_indices] = attn_out
        return residual


class MEMVisionBlock(nn.Module):
    """Wraps a Qwen3VLVisionBlock, prepending temporal causal attention.

    Reference: MEM (Torne et al., 2025), factorized temporal-then-spatial.

    Temporal attention runs BEFORE spatial attention so that the shared QKV
    projection receives input from the same representation space it was
    pretrained on (norm1 applied to h_{l-1}).

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
        # 1. Temporal causal attention FIRST (shares spatial block's norm1 + QKV)
        hidden_states = hidden_states + self.temporal_attn(hidden_states)
        # 2. Standard spatial attention + MLP (original block, unchanged)
        hidden_states = self.spatial_block(hidden_states, cu_seqlens, **kwargs)
        return hidden_states
