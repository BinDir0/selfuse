"""
Unit tests for Qwen3-VL migration components.

Covers: UnifiedVLACollator, prefix cache utilities, SharedPrefixAttention
mask behavior, and processor prompt building.

These tests do NOT require downloading the Qwen3-VL model weights.
"""

import numpy as np
import pytest
import torch

from src.dataset.unified_vla_collator import UnifiedVLACollator
from src.model.vlm.prefix_cache import (
    LayerKV,
    PrefixKVCache,
    build_prefix_mask,
    gather_action_position_ids,
    get_hf_cache_layers,
    slice_prefix_cache_from_full_kv,
)
from src.model.action_expert.qwen_shared_kv_expert import SharedPrefixAttention


# ======================================================================
# Module 1: UnifiedVLACollator
# ======================================================================


def make_vla_sample(seq_len, n_states, n_actions, action_horizon=4, action_dim=48):
    """Build a minimal VLA sample dict like VLAWdsDataset.sample_to_data() returns."""
    actions_valid_mask = np.zeros((action_horizon, action_dim), dtype=bool)
    actions_valid_mask[:n_actions] = True
    return {
        "input_ids": torch.randint(1, 100, (seq_len,)),
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
        "labels": torch.randint(-100, 100, (seq_len,)),
        "mm_token_type_ids": torch.zeros(seq_len, dtype=torch.long),
        "pixel_values": torch.randn(1, 3, 16, 16),
        "image_grid_thw": torch.tensor([1, 4, 4], dtype=torch.long),
        "states": torch.randn(18, action_dim),
        "actions": torch.randn(action_horizon, action_dim),
        "actions_valid_mask": torch.from_numpy(actions_valid_mask),
        "n_states": torch.tensor(n_states, dtype=torch.int32),
        "n_actions": torch.tensor(n_actions, dtype=torch.int32),
        "answer_start_idx": torch.tensor(seq_len - n_actions, dtype=torch.int64),
        "is_vla_data": torch.tensor(True, dtype=torch.bool),
    }


def make_vlm_sample(seq_len, action_horizon=4, action_dim=48):
    """Build a minimal VLM sample dict like UnifiedWdsDataset.pad_vlm_sample()."""
    return {
        "input_ids": torch.randint(1, 100, (seq_len,)),
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
        "labels": torch.randint(-100, 100, (seq_len,)),
        "mm_token_type_ids": torch.zeros(seq_len, dtype=torch.long),
        "pixel_values": torch.randn(1, 3, 16, 16),
        "image_grid_thw": torch.tensor([1, 4, 4], dtype=torch.long),
        "states": torch.zeros(18, action_dim),
        "actions": torch.zeros(action_horizon, action_dim),
        "actions_valid_mask": torch.zeros(action_horizon, action_dim, dtype=torch.bool),
        "n_states": torch.tensor(0, dtype=torch.int32),
        "n_actions": torch.tensor(0, dtype=torch.int32),
        "answer_start_idx": torch.tensor(seq_len, dtype=torch.int64),
        "is_vla_data": torch.tensor(False, dtype=torch.bool),
    }


class TestUnifiedVLACollator:
    """Verify padding and batching of VLA/VLM samples."""

    def test_uniform_length_no_padding_needed(self):
        """Same-length samples should not require any padding."""
        collator = UnifiedVLACollator(pad_token_id=0, ignore_index=-100)
        samples = [make_vla_sample(10, 2, 4), make_vla_sample(10, 3, 4)]
        batch = collator(samples)
        assert batch["input_ids"].shape == (2, 10)
        assert batch["attention_mask"].shape == (2, 10)
        assert batch["labels"].shape == (2, 10)

    def test_variable_length_padding(self):
        """Different-length samples should be padded to the max length."""
        collator = UnifiedVLACollator(pad_token_id=0, ignore_index=-100)
        samples = [make_vla_sample(8, 2, 4), make_vla_sample(12, 3, 4)]
        batch = collator(samples)
        assert batch["input_ids"].shape == (2, 12)
        assert batch["attention_mask"].shape == (2, 12)
        # Shorter sample should be padded with pad_token_id
        assert batch["input_ids"][0, 8:].sum() == 0
        # Shorter sample attention_mask should be 0 in padded region
        assert batch["attention_mask"][0, 8:].sum() == 0
        # Labels should be padded with ignore_index
        assert (batch["labels"][0, 8:] == -100).all()

    def test_mixed_vla_vlm_batch(self):
        """VLA and VLM samples should be collated together."""
        collator = UnifiedVLACollator(pad_token_id=0, ignore_index=-100)
        samples = [make_vla_sample(10, 2, 4), make_vlm_sample(8)]
        batch = collator(samples)
        assert batch["input_ids"].shape == (2, 10)
        assert batch["is_vla_data"].tolist() == [True, False]
        assert batch["n_states"].tolist() == [2, 0]
        assert batch["n_actions"].tolist() == [4, 0]

    def test_pixel_values_batching(self):
        """pixel_values with batch dim 1 should be cat'd, not stacked."""
        collator = UnifiedVLACollator(pad_token_id=0, ignore_index=-100)
        samples = [make_vla_sample(10, 2, 4), make_vla_sample(10, 2, 4)]
        batch = collator(samples)
        assert batch["pixel_values"].shape[0] == 2

    def test_image_grid_thw_batching(self):
        """1D image_grid_thw should be stacked into [B, 3]."""
        collator = UnifiedVLACollator(pad_token_id=0, ignore_index=-100)
        samples = [make_vla_sample(10, 2, 4), make_vla_sample(10, 2, 4)]
        batch = collator(samples)
        assert batch["image_grid_thw"].shape == (2, 3)

    def test_left_padding_side(self):
        """Left padding should prepend pad tokens instead of appending."""
        collator = UnifiedVLACollator(pad_token_id=0, ignore_index=-100, padding_side="left")
        s1 = make_vla_sample(8, 2, 4)
        s1["input_ids"][0] = 42
        s2 = make_vla_sample(10, 2, 4)
        batch = collator([s1, s2])
        # s1 is padded on the left: first 2 tokens should be padding
        assert batch["input_ids"][0, 0].item() == 0
        assert batch["input_ids"][0, 1].item() == 0
        assert batch["input_ids"][0, 2].item() == 42


# ======================================================================
# Module 2: Prefix cache utilities
# ======================================================================


class TestBuildPrefixMask:
    """Verify build_prefix_mask produces correct boolean masks."""

    def test_basic_prefix_mask(self):
        lengths = torch.tensor([3, 5, 1])
        mask = build_prefix_mask(lengths)
        assert mask.shape == (3, 5)
        assert mask[0].tolist() == [True, True, True, False, False]
        assert mask[1].tolist() == [True, True, True, True, True]
        assert mask[2].tolist() == [True, False, False, False, False]

    def test_explicit_max_prefix_len(self):
        lengths = torch.tensor([2, 3])
        mask = build_prefix_mask(lengths, max_prefix_len=6)
        assert mask.shape == (2, 6)
        assert mask[0].sum().item() == 2
        assert mask[1].sum().item() == 3

    def test_zero_length_prefix(self):
        lengths = torch.tensor([0, 0])
        mask = build_prefix_mask(lengths)
        assert mask.shape == (2, 0)


class TestGetHfCacheLayers:
    """Verify HF cache format normalization."""

    def test_list_of_tuples(self):
        key = torch.randn(2, 4, 10, 8)
        value = torch.randn(2, 4, 10, 8)
        cache = [(key, value), (key.clone(), value.clone())]
        layers = get_hf_cache_layers(cache)
        assert len(layers) == 2
        assert isinstance(layers[0], LayerKV)
        torch.testing.assert_close(layers[0].key, key)

    def test_none_returns_empty(self):
        layers = get_hf_cache_layers(None)
        assert layers == []


class TestSlicePrefixCache:
    """Verify prefix cache slicing from full KV."""

    def test_slice_shapes(self):
        """Sliced cache should have shape [B, H_kv, max_prefix_len, Dh]."""
        batch_size, num_kv_heads, seq_len, head_dim = 2, 4, 20, 8
        num_layers = 3
        cache = [
            (
                torch.randn(batch_size, num_kv_heads, seq_len, head_dim),
                torch.randn(batch_size, num_kv_heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        ]
        prefix_lengths = torch.tensor([5, 10])
        prefix_cache = slice_prefix_cache_from_full_kv(cache, prefix_lengths)

        assert len(prefix_cache.layers) == num_layers
        assert prefix_cache.mask.shape == (2, 10)
        for layer in prefix_cache.layers:
            assert layer.key.shape == (2, 4, 10, 8)
            assert layer.value.shape == (2, 4, 10, 8)

    def test_masked_positions_are_zeroed(self):
        """KV at positions beyond prefix_length should be zeroed out."""
        cache = [(
            torch.ones(2, 2, 8, 4),
            torch.ones(2, 2, 8, 4),
        )]
        prefix_lengths = torch.tensor([3, 5])
        prefix_cache = slice_prefix_cache_from_full_kv(cache, prefix_lengths)

        # Batch 0: positions [3, 4] should be zero (max_prefix_len = 5)
        assert (prefix_cache.layers[0].key[0, :, 3:, :] == 0).all()
        # Batch 1: all positions [0:5] should be ones
        assert (prefix_cache.layers[0].key[1, :, :5, :] == 1).all()

    def test_uniform_prefix_lengths(self):
        """When all prefix lengths are the same, no masking needed."""
        cache = [(
            torch.ones(3, 2, 10, 4),
            torch.ones(3, 2, 10, 4),
        )]
        prefix_lengths = torch.tensor([6, 6, 6])
        prefix_cache = slice_prefix_cache_from_full_kv(cache, prefix_lengths)
        assert prefix_cache.layers[0].key.shape == (3, 2, 6, 4)
        assert prefix_cache.mask.all()

    def test_empty_cache_returns_empty_layers(self):
        prefix_lengths = torch.tensor([5])
        prefix_cache = slice_prefix_cache_from_full_kv(None, prefix_lengths)
        assert prefix_cache.layers == []
        assert prefix_cache.mask.shape == (1, 5)


class TestGatherActionPositionIds:
    """Verify action position ID gathering from backbone position IDs."""

    def test_basic_gather(self):
        input_ids = torch.tensor([[1, 2, 102, 102, 0]])
        position_ids = torch.tensor([[[0, 1, 2, 3, 0]]])
        n_actions = torch.tensor([2])
        gathered = gather_action_position_ids(input_ids, 102, position_ids, n_actions)
        assert gathered.shape == (1, 2)
        assert gathered[0].tolist() == [2, 3]

    def test_no_action_tokens(self):
        input_ids = torch.tensor([[1, 2, 3]])
        position_ids = torch.tensor([[[0, 1, 2]]])
        n_actions = torch.tensor([0])
        gathered = gather_action_position_ids(input_ids, 102, position_ids, n_actions)
        assert gathered.shape == (1, 0)

    def test_none_position_ids_fallback(self):
        input_ids = torch.tensor([[1, 102, 102]])
        n_actions = torch.tensor([2])
        gathered = gather_action_position_ids(input_ids, 102, None, n_actions)
        assert gathered.shape == (1, 2)
        assert gathered[0].tolist() == [0, 1]

    def test_batch_with_different_action_counts(self):
        input_ids = torch.tensor([
            [1, 102, 102, 102, 0],
            [1, 2, 102, 102, 0],
        ])
        position_ids = torch.tensor([[[0, 1, 2, 3, 0], [0, 1, 2, 3, 0]]])
        n_actions = torch.tensor([3, 2])
        gathered = gather_action_position_ids(input_ids, 102, position_ids, n_actions)
        assert gathered.shape == (2, 3)
        assert gathered[0].tolist() == [1, 2, 3]
        assert gathered[1, :2].tolist() == [2, 3]
        assert gathered[1, 2].item() == 0  # padded


# ======================================================================
# Module 3: SharedPrefixAttention mask behavior
# ======================================================================


class TestSharedPrefixAttentionMask:
    """Verify attention mask construction for flow and ar modes."""

    @pytest.fixture()
    def attn_module(self):
        return SharedPrefixAttention(
            hidden_size=32,
            num_heads=4,
            num_kv_heads=4,
            head_dim=8,
            rope_theta=10000.0,
            attention_bias=False,
        )

    def test_flow_mask_is_bidirectional_in_suffix(self, attn_module):
        """In flow mode, all suffix tokens should attend to each other."""
        prefix_mask = torch.tensor([[True, True, True, False]])
        action_mask = torch.tensor([[True, True, True]])
        mask = attn_module.build_attention_mask(prefix_mask, action_mask, mode="flow")

        # mask shape: [B=1, 1, A=3, Lp+A=4+3=7]
        assert mask.shape == (1, 1, 3, 7)
        suffix_part = mask[0, 0, :, 4:]  # suffix-to-suffix part
        # All valid suffix tokens should see each other
        expected_suffix = torch.ones(3, 3, dtype=torch.bool)
        assert suffix_part.equal(expected_suffix)

    def test_ar_mask_is_causal_in_suffix(self, attn_module):
        """In ar mode, suffix tokens should attend causally."""
        prefix_mask = torch.tensor([[True, True]])
        action_mask = torch.tensor([[True, True, True]])
        mask = attn_module.build_attention_mask(prefix_mask, action_mask, mode="ar")

        suffix_part = mask[0, 0, :, 2:]  # suffix-to-suffix
        expected_suffix = torch.tensor([
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ])
        assert suffix_part.equal(expected_suffix)

    def test_prefix_always_visible(self, attn_module):
        """All suffix tokens should see all valid prefix positions."""
        prefix_mask = torch.tensor([[True, True, True, False, False]])
        action_mask = torch.tensor([[True, True]])

        for mode in ["flow", "ar"]:
            mask = attn_module.build_attention_mask(prefix_mask, action_mask, mode=mode)
            prefix_part = mask[0, 0, :, :5]  # suffix-to-prefix
            expected_prefix = torch.tensor([
                [True, True, True, False, False],
                [True, True, True, False, False],
            ])
            assert prefix_part.equal(expected_prefix), f"Failed for mode={mode}"

    def test_invalid_action_tokens_masked_out(self, attn_module):
        """Action tokens with mask=False should not attend or be attended to."""
        prefix_mask = torch.tensor([[True, True]])
        action_mask = torch.tensor([[True, False, True]])
        mask = attn_module.build_attention_mask(prefix_mask, action_mask, mode="flow")

        # Row for invalid token (index 1) should be all False
        assert not mask[0, 0, 1, :].any()


class TestSharedPrefixAttentionForward:
    """Verify SharedPrefixAttention forward pass shape correctness."""

    def test_output_shape(self):
        attn = SharedPrefixAttention(
            hidden_size=32,
            num_heads=4,
            num_kv_heads=4,
            head_dim=8,
            rope_theta=10000.0,
            attention_bias=False,
        )
        batch_size, prefix_len, action_len = 2, 5, 3
        hidden = torch.randn(batch_size, action_len, 32)
        prefix_k = torch.randn(batch_size, 4, prefix_len, 8)
        prefix_v = torch.randn(batch_size, 4, prefix_len, 8)
        prefix_mask = torch.ones(batch_size, prefix_len, dtype=torch.bool)
        action_mask = torch.ones(batch_size, action_len, dtype=torch.bool)
        position_ids = torch.arange(action_len).unsqueeze(0).expand(batch_size, -1)

        for mode in ["flow", "ar"]:
            output = attn(
                hidden_states=hidden,
                prefix_k=prefix_k,
                prefix_v=prefix_v,
                prefix_mask=prefix_mask,
                action_mask=action_mask,
                action_position_ids=position_ids,
                mode=mode,
            )
            assert output.shape == (batch_size, action_len, 32)


# ======================================================================
# Module 4: Processor prompt building (no model download required)
# ======================================================================


class TestProcessorPromptBuilding:
    """Verify VLA prompt string construction logic."""

    def test_vla_prefix_format(self):
        """Verify the prompt format matches the plan specification."""
        # Simulate the build_vla_prefix logic without instantiating the processor
        STATE_TOKEN = "<state>"
        text = "Pick up the cup."
        intrinsic = np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float32)
        states = np.zeros((3, 48))
        image_prefix = "<|vision_start|><|image_pad|><|vision_end|>"

        clean_text = text.replace(".", "").lower()
        intrinsic_str = (
            f"fx:{intrinsic[0]:.2f} fy:{intrinsic[1]:.2f} "
            f"cx:{intrinsic[2]:.2f} cy:{intrinsic[3]:.2f}"
        )
        prompt = (
            f"{image_prefix}Task: {clean_text}, Camera intrinsic: {intrinsic_str}, "
            f"States: {STATE_TOKEN * len(states)} "
            f"Actions: "
        )

        # Verify key format properties
        assert "Task: pick up the cup" in prompt
        assert "Camera intrinsic: fx:500.00 fy:500.00 cx:320.00 cy:240.00" in prompt
        assert "<state><state><state>" in prompt
        assert "States: <state><state><state> Actions:" in prompt
        # Verify spacing between States block and Actions
        assert "<state> Actions:" in prompt

    def test_vla_suffix_format(self):
        ACTION_TOKEN = "<action>"
        actions = np.zeros((4, 48))
        suffix = f"{ACTION_TOKEN * len(actions)}"
        assert suffix == "<action><action><action><action>"

    def test_prompt_lowercase(self):
        """VLA text should be lowercased by default."""
        text = "Pick Up The CUP"
        clean_text = text.replace(".", "").lower()
        assert clean_text == "Pick Up The CUP".lower()


# ======================================================================
# Module 5: PrefixKVCache device/dtype casting
# ======================================================================


class TestPrefixKVCacheCast:
    """Verify PrefixKVCache.to() preserves semantics."""

    def test_dtype_cast(self):
        layers = [LayerKV(key=torch.randn(2, 4, 5, 8), value=torch.randn(2, 4, 5, 8))]
        mask = torch.ones(2, 5, dtype=torch.bool)
        lengths = torch.tensor([5, 5])
        cache = PrefixKVCache(layers=layers, mask=mask, lengths=lengths)

        cast = cache.to(dtype=torch.float16)
        assert cast.layers[0].key.dtype == torch.float16
        assert cast.layers[0].value.dtype == torch.float16
        # mask is not floating point, should stay bool
        assert cast.mask.dtype == torch.bool

    def test_max_prefix_len_property(self):
        mask = torch.ones(2, 7, dtype=torch.bool)
        cache = PrefixKVCache(layers=[], mask=mask, lengths=torch.tensor([7, 7]))
        assert cache.max_prefix_len == 7
