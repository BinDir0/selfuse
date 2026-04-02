"""(B,T,S,D) token grid：展平为 (B,T*S,D) 做 TransformerEncoder，时空可学习偏置。"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        num_spatial_tokens: int,
        max_temporal_length: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_spatial_tokens = num_spatial_tokens
        self.max_temporal_length = max_temporal_length

        self.temporal_embed = nn.Embedding(max_temporal_length, dim)
        self.spatial_embed = nn.Parameter(torch.zeros(1, num_spatial_tokens, dim))
        nn.init.trunc_normal_(self.spatial_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x / return: (B, T, S, D)."""
        b, t, s, d = x.shape
        if s != self.num_spatial_tokens:
            raise ValueError(
                f"Spatial length {s} != num_spatial_tokens {self.num_spatial_tokens}"
            )
        if t > self.max_temporal_length:
            raise ValueError(
                f"T={t} exceeds max_temporal_length={self.max_temporal_length}"
            )

        device = x.device
        time_ids = torch.arange(t, device=device)
        t_emb = self.temporal_embed(time_ids).view(1, t, 1, d)
        s_emb = self.spatial_embed.view(1, 1, s, d)
        h = x + t_emb + s_emb
        flat = h.reshape(b, t * s, d)
        out = self.encoder(flat)
        out = self.norm(out)
        return out.view(b, t, s, d)
