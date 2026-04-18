from __future__ import annotations

import torch
from torch import nn

from src.model.vlm.prefix_cache import PrefixKVCache


class DummyFlowExpert(nn.Module):
    def __init__(self, hidden_size: int, time_hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.last_action_position_ids: torch.Tensor | None = None
        self.last_num_parallel_chunks: int | None = None
        self.action_proj = nn.Linear(hidden_size, hidden_size)
        self.time_proj = nn.Linear(time_hidden_size, hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            ]
        )

    def forward(
        self,
        suffix_embeds: torch.Tensor,
        prefix_cache: PrefixKVCache,
        suffix_position_ids: torch.Tensor,
        cond: torch.Tensor | None = None,
        suffix_mask: torch.Tensor | None = None,
        num_parallel_chunks: int = 1,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        # Production Qwen3Expert renamed action_* → suffix_* when action + world-model
        # tokens started sharing the expert; dummy mirrors that contract.
        if prefix_cache is None:
            raise ValueError("DummyFlowExpert requires a prefix cache.")

        if suffix_position_ids.ndim == 3:
            suffix_position_ids = suffix_position_ids[0]
        if suffix_position_ids.ndim == 1:
            suffix_position_ids = suffix_position_ids.unsqueeze(0).expand(suffix_embeds.shape[0], -1)
        self.last_action_position_ids = suffix_position_ids.detach().clone()
        self.last_num_parallel_chunks = num_parallel_chunks

        if suffix_mask is None:
            suffix_mask = torch.ones(
                suffix_embeds.shape[0], suffix_embeds.shape[1],
                device=suffix_embeds.device, dtype=torch.bool,
            )
        hidden_states = self.action_proj(suffix_embeds)
        # Aggregate prefix signal from stacked KV tensors
        # keys/values: [num_layers, B, num_kv_heads, prefix_len, head_dim]
        prefix_signal = (
            prefix_cache.keys.to(dtype=hidden_states.dtype).mean(dim=(0, 2, 3, 4))
            + prefix_cache.values.to(dtype=hidden_states.dtype).mean(dim=(0, 2, 3, 4))
        )
        hidden_states = hidden_states + prefix_signal.view(-1, 1, 1)
        if cond is not None:
            hidden_states = hidden_states + self.time_proj(cond)
        hidden_states = hidden_states + suffix_position_ids.unsqueeze(-1).to(hidden_states.dtype) * 0.01
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states * suffix_mask.unsqueeze(-1).to(hidden_states.dtype)
