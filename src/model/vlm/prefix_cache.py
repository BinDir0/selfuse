from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LayerKV:
    """Per-layer key/value pair. Used internally by get_hf_cache_layers."""
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class PrefixKVCache:
    """Stacked prefix KV cache for compile-friendly tensor indexing.

    keys:    [num_layers, B, num_kv_heads, kv_seq_len, head_dim]
    values:  [num_layers, B, num_kv_heads, kv_seq_len, head_dim]
    mask:    [B, kv_seq_len]  — True for valid prefix positions
    lengths: [B]
    """
    keys: torch.Tensor
    values: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor

    @property
    def num_layers(self) -> int:
        return int(self.keys.shape[0])

    @property
    def kv_seq_len(self) -> int:
        return int(self.mask.shape[-1])

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> "PrefixKVCache":
        return PrefixKVCache(
            keys=self.keys.to(device=device, dtype=dtype if self.keys.is_floating_point() else None),
            values=self.values.to(device=device, dtype=dtype if self.values.is_floating_point() else None),
            mask=self.mask.to(device=device),
            lengths=self.lengths.to(device=device),
        )

    def detach(self) -> "PrefixKVCache":
        return PrefixKVCache(
            keys=self.keys.detach(),
            values=self.values.detach(),
            mask=self.mask.detach(),
            lengths=self.lengths.detach(),
        )


@dataclass
class BackboneStreamOutput:
    last_hidden_states: torch.Tensor
    position_ids: torch.Tensor | None
    past_key_values_hf: Any
    prefix_cache: PrefixKVCache | None = None


def build_prefix_mask(
    prefix_lengths: torch.Tensor,
    max_prefix_len: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    prefix_lengths = prefix_lengths.to(dtype=torch.long)
    if device is None:
        device = prefix_lengths.device
    if max_prefix_len is None:
        max_prefix_len = int(prefix_lengths.max().item()) if prefix_lengths.numel() > 0 else 0
    if max_prefix_len == 0:
        return torch.zeros(prefix_lengths.shape[0], 0, dtype=torch.bool, device=device)
    positions = torch.arange(max_prefix_len, device=device).unsqueeze(0)
    return positions < prefix_lengths.unsqueeze(1)


def get_hf_cache_layers(full_kv: Any) -> list[LayerKV]:
    if full_kv is None:
        return []

    if hasattr(full_kv, "key_cache") and hasattr(full_kv, "value_cache"):
        return [
            LayerKV(key=key_layer, value=value_layer)
            for key_layer, value_layer in zip(full_kv.key_cache, full_kv.value_cache)
        ]

    if hasattr(full_kv, "layers"):
        layers: list[LayerKV] = []
        for layer in full_kv.layers:
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                layers.append(LayerKV(key=layer.keys, value=layer.values))
                continue
            if hasattr(layer, "key") and hasattr(layer, "value"):
                layers.append(LayerKV(key=layer.key, value=layer.value))
                continue
            raise TypeError(f"Unsupported dynamic cache layer type: {type(layer)!r}")
        return layers

    if isinstance(full_kv, (list, tuple)):
        layers: list[LayerKV] = []
        for layer in full_kv:
            if isinstance(layer, LayerKV):
                layers.append(layer)
                continue
            if isinstance(layer, (list, tuple)) and len(layer) >= 2:
                layers.append(LayerKV(key=layer[0], value=layer[1]))
                continue
            if hasattr(layer, "key") and hasattr(layer, "value"):
                layers.append(LayerKV(key=layer.key, value=layer.value))
                continue
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                layers.append(LayerKV(key=layer.keys, value=layer.values))
                continue
            raise TypeError(f"Unsupported cache layer type: {type(layer)!r}")
        return layers

    raise TypeError(f"Unsupported HF cache type: {type(full_kv)!r}")


def slice_prefix_cache_from_full_kv(full_kv: Any, prefix_lengths: torch.Tensor) -> PrefixKVCache:
    layers = get_hf_cache_layers(full_kv)
    prefix_lengths = prefix_lengths.to(dtype=torch.long)
    batch_size = int(prefix_lengths.shape[0])

    if not layers:
        empty_keys = torch.zeros(0, batch_size, 0, 0, 0, device=prefix_lengths.device)
        empty_values = torch.zeros(0, batch_size, 0, 0, 0, device=prefix_lengths.device)
        mask = build_prefix_mask(prefix_lengths, max_prefix_len=0, device=prefix_lengths.device)
        return PrefixKVCache(keys=empty_keys, values=empty_values, mask=mask, lengths=prefix_lengths)

    # Use full cache seq_len as mask length (no truncation, shape stays fixed)
    full_seq_len = layers[0].key.shape[2]
    mask = build_prefix_mask(prefix_lengths, max_prefix_len=full_seq_len, device=prefix_lengths.device)

    all_keys: list[torch.Tensor] = []
    all_values: list[torch.Tensor] = []
    for layer in layers:
        if layer.key.shape[0] != batch_size:
            raise ValueError(
                f"Prefix cache batch mismatch: cache={layer.key.shape[0]} prefix_lengths={batch_size}"
            )
        all_keys.append(layer.key)
        all_values.append(layer.value)

    # keys/values: [num_layers, B, num_kv_heads, prefix_len, head_dim]
    return PrefixKVCache(
        keys=torch.stack(all_keys, dim=0),
        values=torch.stack(all_values, dim=0),
        mask=mask,
        lengths=prefix_lengths,
    )


def gather_action_position_ids(
    input_ids: torch.Tensor,
    action_token_id: int,
    position_ids: torch.Tensor | None,
    n_actions: torch.Tensor,
    action_len: int | None = None,
) -> torch.Tensor:
    n_actions = n_actions.to(dtype=torch.long, device=input_ids.device)
    max_actions = int(n_actions.max().item()) if n_actions.numel() > 0 else 0
    if action_len is not None:
        max_actions = max(max_actions, action_len)
    if max_actions == 0:
        return torch.zeros(input_ids.shape[0], 0, dtype=torch.long, device=input_ids.device)

    if position_ids is None:
        fallback = torch.arange(max_actions, device=input_ids.device, dtype=torch.long).unsqueeze(0)
        return fallback.expand(input_ids.shape[0], -1)

    if position_ids.ndim == 3:
        position_ids = position_ids[0]

    gathered = torch.zeros(input_ids.shape[0], max_actions, dtype=torch.long, device=input_ids.device)
    action_mask = input_ids == action_token_id

    for batch_idx in range(input_ids.shape[0]):
        valid_positions = position_ids[batch_idx][action_mask[batch_idx]]
        limit = min(int(n_actions[batch_idx].item()), int(valid_positions.numel()), max_actions)
        if limit > 0:
            gathered[batch_idx, :limit] = valid_positions[:limit].to(dtype=torch.long)

    return gathered
