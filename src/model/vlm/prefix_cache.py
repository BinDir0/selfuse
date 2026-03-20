from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class PrefixKVCache:
    layers: list[LayerKV]
    mask: torch.Tensor
    lengths: torch.Tensor

    @property
    def max_prefix_len(self) -> int:
        return int(self.mask.shape[-1])

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> "PrefixKVCache":
        cast_layers = []
        for layer in self.layers:
            cast_layers.append(
                LayerKV(
                    key=layer.key.to(device=device, dtype=dtype if layer.key.is_floating_point() else None),
                    value=layer.value.to(device=device, dtype=dtype if layer.value.is_floating_point() else None),
                )
            )
        return PrefixKVCache(
            layers=cast_layers,
            mask=self.mask.to(device=device),
            lengths=self.lengths.to(device=device),
        )

    def detach(self) -> "PrefixKVCache":
        detached_layers = []
        for layer in self.layers:
            detached_layers.append(
                LayerKV(
                    key=layer.key.detach(),
                    value=layer.value.detach(),
                )
            )
        return PrefixKVCache(
            layers=detached_layers,
            mask=self.mask.detach(),
            lengths=self.lengths.detach(),
        )


@dataclass
class BackboneStreamOutput:
    last_hidden_states: torch.Tensor
    all_hidden_states: tuple[torch.Tensor, ...] | None
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
    max_prefix_len = int(prefix_lengths.max().item()) if prefix_lengths.numel() > 0 else 0
    mask = build_prefix_mask(prefix_lengths, max_prefix_len=max_prefix_len, device=prefix_lengths.device)

    if not layers:
        return PrefixKVCache(layers=[], mask=mask, lengths=prefix_lengths)

    sliced_layers: list[LayerKV] = []
    for layer in layers:
        key = layer.key[:, :, :max_prefix_len, :]
        value = layer.value[:, :, :max_prefix_len, :]
        if key.shape[0] != batch_size:
            raise ValueError(
                f"Prefix cache batch mismatch: cache={key.shape[0]} prefix_lengths={batch_size}"
            )
        if max_prefix_len > 0:
            broadcast_mask = mask[:, None, :, None].to(dtype=key.dtype)
            key = key * broadcast_mask
            value = value * broadcast_mask
        sliced_layers.append(LayerKV(key=key, value=value))

    return PrefixKVCache(layers=sliced_layers, mask=mask, lengths=prefix_lengths)


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
