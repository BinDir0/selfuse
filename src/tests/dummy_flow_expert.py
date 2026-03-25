from __future__ import annotations

import torch
from torch import nn

from src.model.vlm.prefix_cache import PrefixKVCache


class DummyFlowExpert(nn.Module):
    def __init__(self, hidden_size: int, time_hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_proj = nn.Linear(hidden_size, hidden_size)
        self.time_proj = nn.Linear(time_hidden_size, hidden_size, bias=False)
        self.mode_embedding = nn.Embedding(2, hidden_size)
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            ]
        )

    def forward(
        self,
        action_embeds: torch.Tensor,
        prefix_cache: PrefixKVCache,
        action_position_ids: torch.Tensor,
        time_cond: torch.Tensor,
        action_mask: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        if prefix_cache is None:
            raise ValueError("DummyFlowExpert requires a prefix cache.")

        if time_cond.ndim == 2:
            time_cond = time_cond.unsqueeze(1).expand(-1, action_embeds.shape[1], -1)

        if action_position_ids.ndim == 3:
            action_position_ids = action_position_ids[0]
        if action_position_ids.ndim == 1:
            action_position_ids = action_position_ids.unsqueeze(0).expand(action_embeds.shape[0], -1)

        mode_index = {"flow": 0, "ar": 1}.get(mode)
        if mode_index is None:
            raise ValueError(f"Unsupported mode: {mode}")

        hidden_states = self.action_proj(action_embeds)
        # Aggregate prefix signal from stacked KV tensors
        # keys/values: [num_layers, B, num_kv_heads, prefix_len, head_dim]
        prefix_signal = (
            prefix_cache.keys.to(dtype=hidden_states.dtype).mean(dim=(0, 2, 3, 4))
            + prefix_cache.values.to(dtype=hidden_states.dtype).mean(dim=(0, 2, 3, 4))
        )
        hidden_states = hidden_states + prefix_signal.view(-1, 1, 1)
        hidden_states = hidden_states + self.time_proj(time_cond)
        hidden_states = hidden_states + self.mode_embedding.weight[mode_index].view(1, 1, -1)
        hidden_states = hidden_states + action_position_ids.unsqueeze(-1).to(hidden_states.dtype) * 0.01
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states * action_mask.unsqueeze(-1).to(hidden_states.dtype)
