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

    The KV tensors keep the full backbone sequence length (no truncation)
    for shape stability under torch.compile. ``mask`` marks which positions
    belong to the prompt prefix (True) — action-token and padding positions
    in the backbone sequence are False and will never be attended to.

    keys:    [num_layers, B, num_kv_heads, kv_seq_len, head_dim]
    values:  [num_layers, B, num_kv_heads, kv_seq_len, head_dim]
    mask:    [B, kv_seq_len]  — True only for prompt-prefix positions
    lengths: [B]              — per-sample prefix length (== answer_start_idx)
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

    def partial_detach(self, depth: int) -> "PrefixKVCache":
        """Detach layers [0, depth), keep gradient for layers [depth, num_layers).

        Useful for partial knowledge insulation: early backbone layers are
        protected from action-expert gradients while late layers can adapt.
        """
        if depth <= 0:
            return self
        if depth >= self.num_layers:
            return self.detach()
        return PrefixKVCache(
            keys=torch.cat([self.keys[:depth].detach(), self.keys[depth:]], dim=0),
            values=torch.cat([self.values[:depth].detach(), self.values[depth:]], dim=0),
            mask=self.mask,
            lengths=self.lengths,
        )


@dataclass
class BackboneStreamOutput:
    last_hidden_states: torch.Tensor
    position_ids: torch.Tensor | None
    past_key_values_hf: Any
    prefix_cache: PrefixKVCache | None = None
    attention_weights: tuple[torch.Tensor | None, ...] | None = None


def build_prefix_mask(
    prefix_lengths: torch.Tensor,
    max_prefix_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a boolean mask [B, max_prefix_len] where position i is True iff i < prefix_lengths[b]."""
    prefix_lengths = prefix_lengths.to(dtype=torch.long)
    if device is None:
        device = prefix_lengths.device
    positions = torch.arange(max_prefix_len, device=device).unsqueeze(0)
    return positions < prefix_lengths.unsqueeze(1)


def get_hf_cache_layers(full_kv: Any) -> list[LayerKV]:
    """Extract per-layer KV pairs from Qwen3VLTextModelWithKV output.

    Expected input: tuple of (key_tensor, value_tensor) per layer,
    produced by Qwen3VLTextModelWithKV.forward as ``full_layer_kv``.
    """
    if not isinstance(full_kv, (list, tuple)):
        raise TypeError(f"Expected list/tuple of (key, value) pairs, got {type(full_kv)!r}")
    return [LayerKV(key=kv[0], value=kv[1]) for kv in full_kv]


def slice_prefix_cache_from_full_kv(full_kv: Any, prefix_lengths: torch.Tensor) -> PrefixKVCache:
    """Build a PrefixKVCache from the backbone's full-sequence KV output.

    The KV tensors are kept at full backbone sequence length for compile
    friendliness. The returned mask is True only for prompt-prefix
    positions (i < answer_start_idx); action-token and padding KV
    entries remain in the tensor but are masked out by the action expert.
    """
    if full_kv is None:
        raise ValueError("full_kv must not be None; backbone must return layer KV pairs.")
    layers = get_hf_cache_layers(full_kv)

    prefix_lengths = prefix_lengths.to(dtype=torch.long)
    batch_size = int(prefix_lengths.shape[0])

    full_seq_len = layers[0].key.shape[2]
    mask = build_prefix_mask(prefix_lengths, full_seq_len, device=prefix_lengths.device)

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
