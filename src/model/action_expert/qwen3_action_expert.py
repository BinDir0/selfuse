"""Action expert using Qwen3VLTextModel layers with AdaLN-Zero time conditioning.

Reuses HF Qwen3VLTextModel layers (preserving M-RoPE, QK-Norm, flash attention)
with per-layer AdaLN-Zero modulation for flow matching time conditioning.
Uses a custom forward loop to inject modulation between norm and attn/mlp.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.model.common.modules import AdaLNZero
from src.model.vlm.prefix_cache import PrefixKVCache


class Qwen3ActionExpert(nn.Module):
    """Action expert that reuses Qwen3VLTextModel layers with AdaLN-Zero.

    Automatically inherits ``num_key_value_heads``, ``head_dim``, and
    ``rope_parameters`` from the backbone's ``text_config`` so the prefix
    KV cache is natively compatible.  Only ``num_layers``, ``hidden_size``,
    ``intermediate_size``, and ``num_heads`` can be overridden.
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_layers: int,
        time_hidden_size: int,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        num_heads: int | None = None,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        try:
            from transformers import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLTextModel
        except ImportError as exc:
            raise ImportError(
                "Qwen3ActionExpert requires transformers with Qwen3-VL support."
            ) from exc

        # Load backbone text_config to inherit KV-critical params
        base = Qwen3VLConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code,
        ).text_config

        hidden_size = hidden_size or base.hidden_size
        intermediate_size = intermediate_size or base.intermediate_size
        num_heads = num_heads or base.num_attention_heads

        config = Qwen3VLTextConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=base.num_key_value_heads,
            head_dim=base.head_dim,
            rope_parameters=base.rope_parameters,
            intermediate_size=intermediate_size,
            attention_bias=base.attention_bias,
            rms_norm_eps=base.rms_norm_eps,
            hidden_act=base.hidden_act,
            max_position_embeddings=base.max_position_embeddings,
            vocab_size=32,
        )
        self.model = Qwen3VLTextModel(config)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Per-layer AdaLN-Zero: reuse existing module (norm + scale/shift/gate)
        self.attn_adalns = nn.ModuleList([
            AdaLNZero(hidden_size, time_hidden_size, eps=config.rms_norm_eps)
            for _ in range(num_layers)
        ])
        self.mlp_adalns = nn.ModuleList([
            AdaLNZero(hidden_size, time_hidden_size, eps=config.rms_norm_eps)
            for _ in range(num_layers)
        ])

    # ------------------------------------------------------------------
    # Prefix cache → HF DynamicCache
    # ------------------------------------------------------------------
    @staticmethod
    def _prefix_cache_to_dynamic_cache(prefix_cache: PrefixKVCache, num_layers: int) -> Any:
        from transformers import DynamicCache

        cache = DynamicCache()
        for i, layer_kv in enumerate(prefix_cache.layers[:num_layers]):
            cache.update(layer_kv.key, layer_kv.value, i)
        return cache

    # ------------------------------------------------------------------
    # 4D attention mask (bypasses HF create_causal_mask)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_4d_attention_mask(
        prefix_mask: torch.Tensor,
        action_mask: torch.Tensor,
        mode: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build [B, 1, action_len, prefix_len + action_len] float attention bias."""
        batch_size, action_len = action_mask.shape
        prefix_len = prefix_mask.shape[1]
        device = action_mask.device

        prefix_visible = prefix_mask[:, None, :].expand(batch_size, action_len, prefix_len)

        valid_action_key = action_mask[:, None, :].expand(batch_size, action_len, action_len)
        if mode == "ar":
            causal = torch.tril(torch.ones(action_len, action_len, dtype=torch.bool, device=device))
            valid_action_key = valid_action_key & causal[None, :, :]

        full_mask = torch.cat([prefix_visible, valid_action_key], dim=-1)
        full_mask = full_mask & action_mask[:, :, None]

        attn_bias = torch.zeros(
            batch_size, 1, action_len, prefix_len + action_len,
            dtype=dtype, device=device,
        )
        attn_bias.masked_fill_(~full_mask.unsqueeze(1), torch.finfo(dtype).min)
        return attn_bias

    # ------------------------------------------------------------------
    # Forward with AdaLN-Zero
    # ------------------------------------------------------------------
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
        if len(prefix_cache.layers) < self.num_layers:
            raise ValueError(
                f"Prefix cache has {len(prefix_cache.layers)} layers, "
                f"but expert expects {self.num_layers}."
            )

        hidden_states = action_embeds

        # Convert prefix cache → HF DynamicCache
        past_key_values = self._prefix_cache_to_dynamic_cache(prefix_cache, self.num_layers)

        # 4D attention mask
        attention_mask = self._build_4d_attention_mask(
            prefix_cache.mask, action_mask, mode, dtype=hidden_states.dtype,
        )

        # Position IDs: expand [B, L] → [4, B, L] for M-RoPE
        if action_position_ids.ndim == 3:
            action_position_ids = action_position_ids[0]
        if action_position_ids.ndim == 2:
            action_position_ids = action_position_ids.unsqueeze(0).expand(4, -1, -1)

        # Split: text_position_ids=[B,L], rotary uses [3,B,L] (temporal/height/width)
        text_position_ids = action_position_ids[0]
        position_embeddings = self.model.rotary_emb(hidden_states, action_position_ids[1:])

        # Cache position for attention
        prefix_len = prefix_cache.layers[0].key.shape[2]
        action_len = action_embeds.shape[1]
        cache_position = torch.arange(
            prefix_len, prefix_len + action_len, device=action_embeds.device,
        )

        # Custom layer loop with AdaLN-Zero
        for i, layer in enumerate(self.model.layers):
            # Attention with AdaLN-Zero
            residual = hidden_states
            h, attn_gate = self.attn_adalns[i](hidden_states, time_cond)
            h, _ = layer.self_attn(
                hidden_states=h,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=False,
                cache_position=cache_position,
            )
            hidden_states = residual + attn_gate * h

            # MLP with AdaLN-Zero
            residual = hidden_states
            h, mlp_gate = self.mlp_adalns[i](hidden_states, time_cond)
            h = layer.mlp(h)
            hidden_states = residual + mlp_gate * h

        hidden_states = self.model.norm(hidden_states)
        return hidden_states * action_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
