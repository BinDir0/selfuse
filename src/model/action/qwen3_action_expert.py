"""Qwen3-based action expert with DiT-style AdaLN-Zero modulation.

This module keeps the Qwen3 text attention / MLP / rotary embedding stack
structurally aligned with the backbone LLM while replacing the two per-layer
pre-norms with AdaLN-Zero conditioning.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.model.common.modules import AdaLNZero
from src.model.vlm.prefix_cache import PrefixKVCache


class StaticPrefixCache:
    """Compile-friendly single-layer KV cache for action expert.

    Mimics the subset of HF DynamicCache interface used by
    Qwen3VLTextAttention.forward (only ``update`` is called when
    past_key_values is not None). Stores a fixed prefix and prepends it
    to incoming key/value states — pure tensor ops, no graph breaks.

    # Source: transformers Qwen3VLTextAttention.forward calls
    #   past_key_values.update(key_states, value_states, self.layer_idx)
    """

    def __init__(self, key: torch.Tensor, value: torch.Tensor):
        # key/value: [B, num_kv_heads, prefix_len, head_dim]
        self.key = key
        self.value = value

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_states = torch.cat([self.key.to(key_states.dtype), key_states], dim=2)
        value_states = torch.cat([self.value.to(value_states.dtype), value_states], dim=2)
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.key.shape[2]


class DiTQwen3DecoderLayer(nn.Module):
    """Qwen3 decoder block with AdaLN-Zero in place of the two pre-norms."""

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        time_hidden_size: int,
        attention_cls: type[nn.Module],
        mlp_cls: type[nn.Module],
        use_kv_projection: bool = False,
    ):
        super().__init__()
        self.self_attn = attention_cls(config=config, layer_idx=layer_idx)
        self.mlp = mlp_cls(config)
        self.attn_adaln = AdaLNZero(config.hidden_size, time_hidden_size, eps=config.rms_norm_eps)
        self.mlp_adaln = AdaLNZero(config.hidden_size, time_hidden_size, eps=config.rms_norm_eps)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        if use_kv_projection:
            self.prefix_key_proj = nn.Linear(head_dim, head_dim, bias=False)
            self.prefix_value_proj = nn.Linear(head_dim, head_dim, bias=False)
            nn.init.eye_(self.prefix_key_proj.weight)
            nn.init.eye_(self.prefix_value_proj.weight)
        else:
            self.prefix_key_proj = nn.Identity()
            self.prefix_value_proj = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | Any,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
        time_cond: torch.Tensor,
        text_position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        prefix_key = self.prefix_key_proj(prefix_key)
        prefix_value = self.prefix_value_proj(prefix_value)
        prefix_cache = StaticPrefixCache(prefix_key, prefix_value)

        residual = hidden_states
        attn_inputs, attn_gate = self.attn_adaln(hidden_states, time_cond)
        # Second return is attn_weights (eager), LSE (flex), or None (sdpa).
        # We only collect it when output_attentions=True + eager backend.
        attn_output, attn_weights = self.self_attn(
            hidden_states=attn_inputs,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=prefix_cache,
            use_cache=False,
            is_causal=False,
        )
        hidden_states = residual + attn_gate * attn_output

        residual = hidden_states
        mlp_inputs, mlp_gate = self.mlp_adaln(hidden_states, time_cond)
        mlp_output = self.mlp(mlp_inputs)
        hidden_states = residual + mlp_gate * mlp_output
        return hidden_states, attn_weights


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
        use_kv_projection: bool = False,
    ):
        super().__init__()
        try:
            from transformers import Qwen3VLConfig, Qwen3VLTextConfig
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
        if getattr(config, "_attn_implementation", None) != "flex_attention":
            raise ValueError(
                "Qwen3ActionExpert requires attn_implementation='flex_attention' "
                "because action attention always uses a BlockMask."
            )

        self.config = config
        self.hidden_size = hidden_size
        self.num_layers = config.num_hidden_layers
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)
        # All layers use layer_idx=0 because each layer gets its own
        # single-entry cache at runtime.  Using the same index everywhere
        # prevents torch.compile from recompiling for every distinct layer_idx.
        self.layers = nn.ModuleList(
            [
                DiTQwen3DecoderLayer(
                    config=config,
                    layer_idx=0,
                    time_hidden_size=time_hidden_size,
                    attention_cls=Qwen3VLTextAttention,
                    mlp_cls=Qwen3VLTextMLP,
                    use_kv_projection=use_kv_projection,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.checkpoint_every_n = 1
        self._gradient_checkpointing_func = partial(
            checkpoint, use_reentrant=False, preserve_rng_state=False,
        )

    def enable_gradient_checkpointing(self, every_n: int = 1) -> None:
        self.gradient_checkpointing = True
        self.checkpoint_every_n = every_n
        self._gradient_checkpointing_func = partial(
            checkpoint, use_reentrant=False, preserve_rng_state=False,
        )

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.checkpoint_every_n = 1

    @staticmethod
    def build_4d_attention_mask(
        full_attention_mask_bool: torch.Tensor,
        prefix_len: int,
        action_len: int,
        chunk_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build 4D additive attention mask equivalent to the flex_attention BlockMask.

        Returns [B, 1, action_len, prefix_len + action_len] with 0 for visible
        positions and a large negative value for masked positions.
        """
        device = full_attention_mask_bool.device
        kv_len = prefix_len + action_len

        # Per-KV validity: [B, 1, 1, KV]
        valid = full_attention_mask_bool[:, None, None, :]

        kv_indices = torch.arange(kv_len, device=device)
        q_indices = torch.arange(action_len, device=device)

        # Prefix positions are always visible: [1, 1, 1, KV]
        is_prefix = (kv_indices < prefix_len).view(1, 1, 1, -1)

        # Same-chunk logic for the action part: [1, 1, Q, KV]
        q_chunks = q_indices // chunk_size
        kv_action_offset = (kv_indices - prefix_len).clamp(min=0)
        kv_chunks = kv_action_offset // chunk_size
        same_chunk = q_chunks.view(1, 1, -1, 1) == kv_chunks.view(1, 1, 1, -1)
        is_action_kv = kv_indices >= prefix_len
        is_action_kv = is_action_kv.view(1, 1, 1, -1)

        # Reproduce mask_mod: valid & (is_prefix | (is_action_kv & same_chunk))
        bool_mask = valid & (is_prefix | (is_action_kv & same_chunk))

        min_val = torch.finfo(dtype).min
        return torch.where(bool_mask, torch.zeros((), dtype=dtype, device=device),
                           torch.full((), min_val, dtype=dtype, device=device))

    def forward(
        self,
        action_embeds: torch.Tensor,
        prefix_cache: PrefixKVCache,
        action_position_ids: torch.Tensor,
        time_cond: torch.Tensor,
        action_mask: torch.Tensor,
        num_parallel_chunks: int,
        output_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor | None]]:
        if prefix_cache is None:
            raise ValueError("Action expert requires a prefix cache.")
        if prefix_cache.num_layers < self.num_layers:
            raise ValueError(
                f"Prefix cache has {prefix_cache.num_layers} layers, "
                f"but expert expects {self.num_layers}."
            )
        if num_parallel_chunks < 1:
            raise ValueError(f"num_parallel_chunks must be >= 1, got {num_parallel_chunks}.")
        hidden_states = action_embeds

        # Build full attention mask: [prefix_mask | action_mask]
        full_attention_mask = torch.cat(
            [
                prefix_cache.mask.to(device=action_mask.device, dtype=torch.bool),
                action_mask.to(dtype=torch.bool),
            ],
            dim=-1,
        )

        if action_position_ids.ndim == 3:
            action_position_ids = action_position_ids[0]
        if action_position_ids.ndim != 2:
            raise ValueError(f"Expected action_position_ids to be 2D after squeeze, got {action_position_ids.shape}.")
        if action_position_ids.shape[1] != hidden_states.shape[1]:
            raise ValueError(
                "Action position ids length "
                f"{action_position_ids.shape[1]} does not match hidden sequence length {hidden_states.shape[1]}."
            )
        # Qwen3-VL position_ids: 4 dims = [text_pos, height, width, temporal]
        action_position_ids = action_position_ids.unsqueeze(0).expand(4, -1, -1)

        text_position_ids = action_position_ids[0]
        position_embeddings = self.rotary_emb(hidden_states, action_position_ids[1:])

        prefix_len = prefix_cache.kv_seq_len
        action_len = action_embeds.shape[1]
        if action_len % num_parallel_chunks != 0:
            raise ValueError(
                f"Action length {action_len} must be divisible by num_parallel_chunks={num_parallel_chunks}."
            )

        batch_size = hidden_states.shape[0]
        chunk_size = action_len // num_parallel_chunks
        full_attention_mask_bool = full_attention_mask.to(device=action_embeds.device)

        use_flex = self.config._attn_implementation == "flex_attention"
        if use_flex:
            from torch.nn.attention.flex_attention import create_block_mask

            def mask_mod(b, h, q_idx, kv_idx):
                del h
                valid = full_attention_mask_bool[b, kv_idx]
                is_prefix = kv_idx < prefix_len
                same_chunk = (q_idx // chunk_size) == ((kv_idx - prefix_len) // chunk_size)
                return valid & (is_prefix | same_chunk)

            # T=1 is the same code path with a single chunk, which degenerates to
            # fully bidirectional action attention plus always-visible prefix KV.
            attention_mask = create_block_mask(
                mask_mod=mask_mod,
                B=batch_size,
                H=None,
                Q_LEN=action_len,
                KV_LEN=prefix_len + action_len,
                device=action_embeds.device,
            )
        else:
            # Eager / sdpa: build a standard 4D additive attention mask
            # that reproduces the same masking logic.
            attention_mask = self.build_4d_attention_mask(
                full_attention_mask_bool,
                prefix_len,
                action_len,
                chunk_size,
                dtype=hidden_states.dtype,
            )

        all_attn_weights: list[torch.Tensor | None] = []
        for layer_idx, layer in enumerate(self.layers):
            # Compile-friendly: pure tensor indexing on stacked cache
            prefix_key = prefix_cache.keys[layer_idx]
            prefix_value = prefix_cache.values[layer_idx]

            if (
                self.gradient_checkpointing
                and self.training
                and layer_idx % self.checkpoint_every_n == 0
            ):
                hidden_states, layer_attn = self._gradient_checkpointing_func(
                    layer,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    prefix_key,
                    prefix_value,
                    time_cond,
                    text_position_ids,
                    output_attentions,
                )
            else:
                hidden_states, layer_attn = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    prefix_key=prefix_key,
                    prefix_value=prefix_value,
                    time_cond=time_cond,
                    text_position_ids=text_position_ids,
                    output_attentions=output_attentions,
                )
            if output_attentions:
                all_attn_weights.append(layer_attn)

        hidden_states = self.norm(hidden_states)
        result = hidden_states * action_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        if output_attentions:
            return result, all_attn_weights
        return result
