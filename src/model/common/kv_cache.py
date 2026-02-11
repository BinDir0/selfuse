from typing import List, Tuple

import torch


class KVCache:
    def __init__(self) -> None:
        """list for layers"""
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def has_item(self, layer_idx) -> bool:
        return len(self.key_cache) > layer_idx

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def get(self, layer_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        # ... and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class StaticKVCache(KVCache):
    """KV cache preallocated with fixed maximum sizes."""

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        # Track current sequence length per layer.
        self._seq_lens: List[int] = [0 for _ in range(num_layers)]

        for _ in range(num_layers):
            # Shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            k = torch.zeros(
                (max_batch_size, num_heads, max_seq_len, head_dim),
                device=device,
                dtype=dtype,
            )
            v = torch.zeros(
                (max_batch_size, num_heads, max_seq_len, head_dim),
                device=device,
                dtype=dtype,
            )
            self.key_cache.append(k)
            self.value_cache.append(v)

    def has_item(self, layer_idx) -> bool:
        return 0 <= layer_idx < self.num_layers

    def num_items(self, layer_idx: int | None = None) -> int:
        if layer_idx is None:
            return max(self._seq_lens) if self._seq_lens else 0
        return self._seq_lens[layer_idx]

    def get(self, layer_idx, static=True) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = self._seq_lens[layer_idx]
        if static:
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            return (
                self.key_cache[layer_idx][:, :, :seq_len, :], 
                self.value_cache[layer_idx][:, :, :seq_len, :],
            )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        start_pos: int | None = None,
        static=True, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # key_states/value_states shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        if start_pos is None:
            start_pos = self._seq_lens[layer_idx]
        seq_len = key_states.shape[-2]
        end_pos = start_pos + seq_len

        if end_pos > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: end_pos={end_pos} exceeds max_seq_len={self.max_seq_len}"
            )

        self.key_cache[layer_idx][:, :, start_pos:end_pos, :] = key_states
        self.value_cache[layer_idx][:, :, start_pos:end_pos, :] = value_states
        self._seq_lens[layer_idx] = max(self._seq_lens[layer_idx], end_pos)

        if static:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            total_seq_len = self._seq_lens[layer_idx]
            return (
                self.key_cache[layer_idx][:, :, :total_seq_len, :], 
                self.value_cache[layer_idx][:, :, :total_seq_len, :]
            )

    def reset(self) -> None:
        self._seq_lens = [0 for _ in range(self.num_layers)]