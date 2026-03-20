"""Qwen3-based action expert with DiT-style AdaLN-Zero modulation.

This module keeps the Qwen3 text attention / MLP / rotary embedding stack
structurally aligned with the backbone LLM while replacing the two per-layer
pre-norms with AdaLN-Zero conditioning.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.model.common.modules import AdaLNZero
from src.model.vlm.prefix_cache import PrefixKVCache


class DiTQwen3DecoderLayer(nn.Module):
    """Qwen3 decoder block with AdaLN-Zero in place of the two pre-norms."""

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        time_hidden_size: int,
        attention_cls: type[nn.Module],
        mlp_cls: type[nn.Module],
    ):
        super().__init__()
        self.self_attn = attention_cls(config=config, layer_idx=layer_idx)
        self.mlp = mlp_cls(config)
        self.attn_adaln = AdaLNZero(config.hidden_size, time_hidden_size, eps=config.rms_norm_eps)
        self.mlp_adaln = AdaLNZero(config.hidden_size, time_hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | Any,
        past_key_values: Any,
        time_cond: torch.Tensor,
        is_causal: bool,
        text_position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        attn_inputs, attn_gate = self.attn_adaln(hidden_states, time_cond)
        attn_output, _ = self.self_attn(
            hidden_states=attn_inputs,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            cache_position=cache_position,
            is_causal=is_causal,
        )
        hidden_states = residual + attn_gate * attn_output

        residual = hidden_states
        mlp_inputs, mlp_gate = self.mlp_adaln(hidden_states, time_cond)
        mlp_output = self.mlp(mlp_inputs)
        hidden_states = residual + mlp_gate * mlp_output
        return hidden_states


class Qwen3ActionExpert(nn.Module):
    """Action expert that reuses Qwen3 text primitives with DiT-style AdaLN-Zero.

    The expert always inherits the backbone text layer count so that each action
    expert layer consumes the prefix KV cache from the matching backbone layer.
    ``hidden_size`` and ``intermediate_size`` may differ from the backbone,
    while KV-critical dimensions remain aligned.
    """

    def __init__(
        self,
        model_name_or_path: str,
        time_hidden_size: int,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        num_heads: int | None = None,
        attn_implementation: str | None = None,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        try:
            from transformers import DynamicCache, Qwen3VLConfig, Qwen3VLTextConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLTextAttention,
                Qwen3VLTextMLP,
                Qwen3VLTextRMSNorm,
                Qwen3VLTextRotaryEmbedding,
            )
        except ImportError as exc:
            raise ImportError(
                "Qwen3ActionExpert requires transformers with Qwen3-VL support."
            ) from exc

        self.dynamic_cache_cls = DynamicCache
        base = Qwen3VLConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        ).text_config

        hidden_size = hidden_size or base.hidden_size
        intermediate_size = intermediate_size or base.intermediate_size
        num_heads = num_heads or base.num_attention_heads

        config = Qwen3VLTextConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=base.num_hidden_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=base.num_key_value_heads,
            head_dim=base.head_dim,
            hidden_act=base.hidden_act,
            max_position_embeddings=base.max_position_embeddings,
            rms_norm_eps=base.rms_norm_eps,
            rope_parameters=base.rope_parameters,
            attention_bias=base.attention_bias,
            attention_dropout=getattr(base, "attention_dropout", 0.0),
            vocab_size=32,
        )
        if attn_implementation is not None:
            config._attn_implementation = attn_implementation
        elif hasattr(base, "_attn_implementation"):
            config._attn_implementation = base._attn_implementation

        self.config = config
        self.hidden_size = hidden_size
        self.num_layers = config.num_hidden_layers
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [
                DiTQwen3DecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                    time_hidden_size=time_hidden_size,
                    attention_cls=Qwen3VLTextAttention,
                    mlp_cls=Qwen3VLTextMLP,
                )
                for layer_idx in range(self.num_layers)
            ]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        past_key_values = self.dynamic_cache_cls()
        for layer_idx, layer_kv in enumerate(prefix_cache.layers[:self.num_layers]):
            past_key_values.update(layer_kv.key, layer_kv.value, layer_idx)

        full_attention_mask = torch.cat(
            [
                prefix_cache.mask.to(device=action_mask.device, dtype=torch.bool),
                action_mask.to(dtype=torch.bool),
            ],
            dim=-1,
        )
        from transformers.masking_utils import create_bidirectional_mask, create_causal_mask

        if mode == "ar":
            attention_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=full_attention_mask,
                past_key_values=past_key_values,
            )
            is_causal = True
        elif mode == "flow":
            attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=full_attention_mask,
                past_key_values=past_key_values,
            )
            is_causal = False
        else:
            raise ValueError(f"Unsupported action expert mode: {mode}")

        if action_position_ids.ndim == 3:
            action_position_ids = action_position_ids[0]
        if action_position_ids.ndim == 2:
            action_position_ids = action_position_ids.unsqueeze(0).expand(4, -1, -1)

        text_position_ids = action_position_ids[0]
        position_embeddings = self.rotary_emb(hidden_states, action_position_ids[1:])

        prefix_len = prefix_cache.layers[0].key.shape[2]
        action_len = action_embeds.shape[1]
        cache_position = torch.arange(
            prefix_len,
            prefix_len + action_len,
            device=action_embeds.device,
        )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                time_cond=time_cond,
                is_causal=is_causal,
                text_position_ids=text_position_ids,
                cache_position=cache_position,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states * action_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
