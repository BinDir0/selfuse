from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.model.common.modules import AdaLNZero
from src.model.common.utils import apply_rotary_pos_emb, repeat_kv
from src.model.vlm.prefix_cache import PrefixKVCache


class RotaryEmbedding1D(nn.Module):
    def __init__(self, head_dim: int, rope_theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 3:
            position_ids = position_ids[0]
        freqs = torch.einsum("bl,d->bld", position_ids.to(self.inv_freq.dtype), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


class SharedPrefixAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.attention_bias = attention_bias
        self.rope_theta = rope_theta
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=attention_bias)
        self.rope = RotaryEmbedding1D(head_dim, rope_theta)

    def build_attention_mask(
        self,
        prefix_mask: torch.Tensor,
        action_mask: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        batch_size, action_len = action_mask.shape
        prefix_len = prefix_mask.shape[1]
        device = action_mask.device

        prefix_visible = prefix_mask[:, None, None, :].expand(batch_size, 1, action_len, prefix_len)
        valid_action_key = action_mask[:, None, None, :].expand(batch_size, 1, action_len, action_len)

        if mode == "ar":
            causal = torch.tril(torch.ones(action_len, action_len, dtype=torch.bool, device=device))
            valid_action_key = valid_action_key & causal[None, None, :, :]

        full_mask = torch.cat([prefix_visible, valid_action_key], dim=-1)
        query_mask = action_mask[:, None, :, None]
        return full_mask & query_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        prefix_mask: torch.Tensor,
        action_mask: torch.Tensor,
        action_position_ids: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        batch_size, action_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(batch_size, action_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch_size, action_len, self.num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, action_len, self.num_kv_heads, self.head_dim)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = self.rope(action_position_ids, dtype=query_states.dtype)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=1)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=1)

        key_states = torch.cat([prefix_k.to(dtype=key_states.dtype), key_states], dim=2)
        value_states = torch.cat([prefix_v.to(dtype=value_states.dtype), value_states], dim=2)

        num_key_value_groups = self.num_heads // self.num_kv_heads
        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)

        attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale
        attn_mask = self.build_attention_mask(prefix_mask, action_mask, mode)
        valid_rows = attn_mask.any(dim=-1, keepdim=True)
        fill_value = torch.finfo(attn_scores.dtype).min
        attn_scores = torch.where(attn_mask, attn_scores, torch.full_like(attn_scores, fill_value))
        attn_scores = torch.where(valid_rows, attn_scores, torch.zeros_like(attn_scores))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.where(valid_rows, attn_weights, torch.zeros_like(attn_weights))

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, action_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output * action_mask.unsqueeze(-1).to(dtype=attn_output.dtype)


class QwenAdaLNZeroDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        time_hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.time_hidden_size = time_hidden_size

        self.self_attn = SharedPrefixAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
        )
        self.attn_modulation = AdaLNZero(hidden_size, time_hidden_size)
        self.mlp_modulation = AdaLNZero(hidden_size, time_hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.SiLU(),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        prefix_mask: torch.Tensor,
        action_mask: torch.Tensor,
        action_position_ids: torch.Tensor,
        time_cond: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        normed_hidden, attn_gate = self.attn_modulation(hidden_states, time_cond)
        attn_output = self.self_attn(
            hidden_states=normed_hidden,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            prefix_mask=prefix_mask,
            action_mask=action_mask,
            action_position_ids=action_position_ids,
            mode=mode,
        )
        hidden_states = hidden_states + attn_gate * attn_output

        normed_hidden, mlp_gate = self.mlp_modulation(hidden_states, time_cond)
        mlp_output = self.mlp(normed_hidden)
        hidden_states = hidden_states + mlp_gate * mlp_output
        return hidden_states * action_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)


class ActionExpertDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        time_hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            QwenAdaLNZeroDecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                time_hidden_size=time_hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rope_theta=rope_theta,
                attention_bias=attention_bias,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)

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
            raise ValueError("Action expert requires a prefix cache.")
        if len(prefix_cache.layers) < len(self.layers):
            raise ValueError(
                f"Prefix cache has {len(prefix_cache.layers)} layers, but expert expects {len(self.layers)}."
            )

        hidden_states = action_embeds
        for layer, layer_cache in zip(self.layers, prefix_cache.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                prefix_k=layer_cache.key,
                prefix_v=layer_cache.value,
                prefix_mask=prefix_cache.mask,
                action_mask=action_mask,
                action_position_ids=action_position_ids,
                time_cond=time_cond,
                mode=mode,
            )
        hidden_states = self.final_norm(hidden_states)
        return hidden_states * action_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
