"""
Unit tests for Qwen3-VL migration components.

Covers: UnifiedVLACollator, prefix cache utilities, and processor prompt building.

These tests do NOT require downloading the Qwen3-VL model weights.
"""

import numpy as np
import pytest
import torch

from src.dataset.qwen3_vl_batching import Qwen3VLBatchProcessor, Qwen3VLChatFormatter
from src.dataset.unified_vla_collator import UnifiedVLACollator
from src.model.vlm.prefix_cache import (
    LayerKV,
    PrefixKVCache,
    build_prefix_mask,
    get_hf_cache_layers,
    slice_prefix_cache_from_full_kv,
)


# ======================================================================
# Module 1: UnifiedVLACollator
# ======================================================================


class DummyBatchProcessor:
    def __init__(self):
        self.ignore_index = -100
        self.padding_side = "right"
        self.state_token = "<state>"
        self.action_token = "<action>"
        self.action_token_id = 102

    def encode_messages(
        self,
        messages_batch,
        batch_samples,
        add_generation_prompt,
        return_rendered_texts: bool = False,
    ):
        del messages_batch
        if add_generation_prompt:
            batch_size = len(batch_samples)
            batch = {
                "input_ids": torch.tensor([[10, 11]] * batch_size, dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]] * batch_size, dtype=torch.long),
                "pixel_values": torch.randn(batch_size, 3, 8, 8),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * batch_size, dtype=torch.long),
                "pixel_values_videos": None,
                "video_grid_thw": None,
                "mm_token_type_ids": torch.zeros(batch_size, 2, dtype=torch.long),
            }
        else:
            batch = {
                "input_ids": torch.tensor([
                    [1, 2, 102, 102, 0],
                    [10, 11, 12, 13, 14],
                ], dtype=torch.long),
                "attention_mask": torch.tensor([
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                ], dtype=torch.long),
                "pixel_values": torch.randn(2, 3, 8, 8),
                "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long),
                "pixel_values_videos": None,
                "video_grid_thw": None,
                "mm_token_type_ids": torch.zeros(2, 5, dtype=torch.long),
            }
        if return_rendered_texts:
            batch["rendered_texts"] = ["dummy" for _ in range(len(batch_samples))]
        return batch

    def build_labels(self, input_ids, attention_mask, answer_start_idx):
        labels = input_ids.clone()
        labels = labels.masked_fill(attention_mask == 0, self.ignore_index)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        labels = labels.masked_fill(positions < answer_start_idx.unsqueeze(1), self.ignore_index)
        return labels

    def find_first_token_positions(self, input_ids, token_id, fallback):
        token_mask = input_ids == token_id
        first_positions = token_mask.to(dtype=torch.long).argmax(dim=1)
        has_token = token_mask.any(dim=1)
        return torch.where(has_token, first_positions, fallback)


class DummyTokenizerForBatchProcessor:
    def add_special_tokens(self, _tokens):
        return 0

    def convert_tokens_to_ids(self, token):
        if token == "<action>":
            return 102
        return 0


class DummyProcessorForBatchProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizerForBatchProcessor()


class TestQwen3VLBatchProcessor:
    def test_build_vision_inputs_mixes_image_and_video(self):
        batch_processor = Qwen3VLBatchProcessor(
            model_name_or_path="demo",
            processor_call_kwargs={"padding": "longest", "max_length": 32},
            processor=DummyProcessorForBatchProcessor(),
        )
        batch_inputs = batch_processor.build_vision_inputs([
            {
                "vision_type": "video",
                "images": torch.zeros(3, 8, 8, 3, dtype=torch.uint8),
                "video_fps": torch.tensor(15.0),
            },
            {
                "vision_type": "image",
                "images": torch.zeros(2, 8, 8, 3, dtype=torch.uint8),
            },
        ])

        assert batch_inputs["do_sample_frames"] is False
        assert len(batch_inputs["videos"]) == 1
        assert batch_inputs["videos"][0].shape == (3, 8, 8, 3)
        assert len(batch_inputs["video_metadata"]) == 1
        assert batch_inputs["video_metadata"][0]["fps"] == 15.0
        assert len(batch_inputs["images"]) == 1
        assert len(batch_inputs["images"][0]) == 2

    def test_build_vision_inputs_normalizes_single_and_multi_image_samples(self):
        batch_processor = Qwen3VLBatchProcessor(
            model_name_or_path="demo",
            processor_call_kwargs={"padding": "longest", "max_length": 32},
            processor=DummyProcessorForBatchProcessor(),
        )
        batch_inputs = batch_processor.build_vision_inputs([
            {
                "vision_type": "image",
                "images": torch.zeros(8, 8, 3, dtype=torch.uint8),
            },
            {
                "vision_type": "video",
                "images": torch.zeros(3, 8, 8, 3, dtype=torch.uint8),
                "video_fps": torch.tensor(15.0),
            },
            {
                "vision_type": "image",
                "images": torch.zeros(2, 8, 8, 3, dtype=torch.uint8),
            },
        ])

        assert len(batch_inputs["images"]) == 2
        assert all(isinstance(sample_images, list) for sample_images in batch_inputs["images"])
        assert len(batch_inputs["images"][0]) == 1
        assert len(batch_inputs["images"][1]) == 2
        assert len(batch_inputs["videos"]) == 1
        assert len(batch_inputs["video_metadata"]) == 1

    def test_build_vision_inputs_rejects_non_uint8_images(self):
        batch_processor = Qwen3VLBatchProcessor(
            model_name_or_path="demo",
            processor_call_kwargs={"padding": "longest", "max_length": 32},
            processor=DummyProcessorForBatchProcessor(),
        )

        with pytest.raises(ValueError, match=r"images must use dtype torch\.uint8"):
            batch_processor.build_vision_inputs([
                {
                    "vision_type": "image",
                    "images": torch.rand(2, 8, 8, 3, dtype=torch.float32),
                }
            ])

    def test_build_vision_inputs_rejects_non_uint8_videos(self):
        batch_processor = Qwen3VLBatchProcessor(
            model_name_or_path="demo",
            processor_call_kwargs={"padding": "longest", "max_length": 32},
            processor=DummyProcessorForBatchProcessor(),
        )

        with pytest.raises(ValueError, match=r"video must use dtype torch\.uint8"):
            batch_processor.build_vision_inputs([
                {
                    "vision_type": "video",
                    "images": torch.rand(3, 8, 8, 3, dtype=torch.float32),
                    "video_fps": torch.tensor(15.0),
                }
            ])


class TestUnifiedVLACollatorRaw:
    @staticmethod
    def make_raw_vla_sample():
        return {
            "images": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
            "instruction": "Pick the cup",
            "intrinsic": torch.tensor([1.0, 1.0, 0.5, 0.5]),
            "vision_type": "video",
            "video_fps": torch.tensor(15.0),
            "states": torch.randn(4, 48),
            "actions": torch.randn(4, 48),
            "actions_valid_mask": torch.ones(4, 48, dtype=torch.bool),
            "n_states": torch.tensor(4, dtype=torch.int32),
            "n_actions": torch.tensor(2, dtype=torch.int32),
            "is_vla_data": torch.tensor(True),
            "has_depth_values": torch.tensor(False),
        }

    @staticmethod
    def make_raw_vlm_sample():
        return {
            "images": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
            "question": "What is in the image?",
            "answer": "A cup.",
            "vision_type": "image",
            "states": torch.zeros(4, 48),
            "actions": torch.zeros(4, 48),
            "actions_valid_mask": torch.zeros(4, 48, dtype=torch.bool),
            "n_states": torch.tensor(0, dtype=torch.int32),
            "n_actions": torch.tensor(0, dtype=torch.int32),
            "is_vla_data": torch.tensor(False),
            "has_depth_values": torch.tensor(False),
        }

    def test_raw_batch_uses_formatter_and_batch_processor(self):
        collator = UnifiedVLACollator(
            formatter=Qwen3VLChatFormatter(),
            batch_processor=DummyBatchProcessor(),
        )
        batch = collator([self.make_raw_vla_sample(), self.make_raw_vlm_sample()])

        assert batch["input_ids"].shape == (2, 5)
        assert batch["pixel_values"].shape[0] == 2
        assert batch["answer_start_idx"].tolist() == [2, 2]
        assert (batch["labels"][0] == -100).all()
        assert batch["labels"][1, :2].tolist() == [-100, -100]
        assert batch["labels"][1, 2:].tolist() == [12, 13, 14]
        assert batch["is_vla_data"].tolist() == [True, False]
        assert batch["states"].shape == (2, 4, 48)
        assert batch["actions"].shape == (2, 4, 48)

    def test_mode_sets_padding_side_and_prompt_only(self):
        train_collator = UnifiedVLACollator(
            formatter=Qwen3VLChatFormatter(),
            batch_processor=DummyBatchProcessor(),
        )
        infer_ar_collator = UnifiedVLACollator(
            formatter=Qwen3VLChatFormatter(),
            batch_processor=DummyBatchProcessor(),
            mode="infer-ar",
        )

        assert train_collator.batch_processor.padding_side == "right"
        assert train_collator.prompt_only_input is False
        assert infer_ar_collator.batch_processor.padding_side == "left"
        assert infer_ar_collator.prompt_only_input is True



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
        """Cache keeps full KV seq_len; mask marks valid prefix positions."""
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

        assert prefix_cache.num_layers == num_layers
        assert prefix_cache.mask.shape == (2, seq_len)
        assert prefix_cache.keys.shape == (num_layers, 2, 4, seq_len, 8)
        assert prefix_cache.values.shape == (num_layers, 2, 4, seq_len, 8)

    def test_mask_marks_valid_prefix_positions(self):
        """Mask is True only for positions < prefix_length."""
        cache = [(
            torch.ones(2, 2, 8, 4),
            torch.ones(2, 2, 8, 4),
        )]
        prefix_lengths = torch.tensor([3, 5])
        prefix_cache = slice_prefix_cache_from_full_kv(cache, prefix_lengths)

        assert prefix_cache.mask.shape == (2, 8)
        assert prefix_cache.mask[0].sum() == 3
        assert prefix_cache.mask[1].sum() == 5

    def test_uniform_prefix_lengths(self):
        """When all prefix lengths are the same, mask is uniform."""
        cache = [(
            torch.ones(3, 2, 10, 4),
            torch.ones(3, 2, 10, 4),
        )]
        prefix_lengths = torch.tensor([6, 6, 6])
        prefix_cache = slice_prefix_cache_from_full_kv(cache, prefix_lengths)
        assert prefix_cache.keys[0].shape == (3, 2, 10, 4)
        assert prefix_cache.mask[:, :6].all()
        assert not prefix_cache.mask[:, 6:].any()

    def test_empty_cache_returns_empty_tensors(self):
        prefix_lengths = torch.tensor([5])
        prefix_cache = slice_prefix_cache_from_full_kv(None, prefix_lengths)
        assert prefix_cache.num_layers == 0


# ======================================================================
# Module 3: Processor prompt building (no model download required)
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

    def test_future_frame_tokens_follow_temporal_patch_count(self):
        # 64x64 frames, patch_size=16, spatial_merge_size=2 → 2x2=4 tokens per temporal patch
        # 4 frames, tps=2 → T_future=2 → 2*4=8 tokens
        formatter = Qwen3VLChatFormatter(future_frame_token="<future_frame>")
        formatter.patch_size = 16
        formatter.spatial_merge_size = 2
        formatter.temporal_patch_size = 2
        sample = {
            "is_vla_data": torch.tensor(True),
            "instruction": "Open drawer",
            "intrinsic": torch.tensor([1.0, 1.0, 0.5, 0.5]),
            "n_states": torch.tensor(2, dtype=torch.int32),
            "n_actions": torch.tensor(2, dtype=torch.int32),
            "vision_type": "video",
            "images": torch.zeros(2, 64, 64, 3, dtype=torch.uint8),
            "video_fps": torch.tensor(15.0),
            "future_frames": torch.zeros(4, 64, 64, 3, dtype=torch.uint8),
        }

        messages = formatter.build_messages(sample, prompt_only=False)
        assistant_text = messages[1]["content"][0]["text"]
        user_text = messages[0]["content"][-1]["text"]

        assert assistant_text.startswith("<action><action>")
        assert assistant_text.count("<future_frame>") == 8
        assert "future visual observations" in user_text

    def test_future_frame_tokens_zero_when_truncated_to_zero(self):
        """K < tps → T_future = 0 → no future_frame tokens, no prompt hint."""
        formatter = Qwen3VLChatFormatter(future_frame_token="<future_frame>")
        formatter.patch_size = 16
        formatter.spatial_merge_size = 2
        formatter.temporal_patch_size = 2
        sample = {
            "is_vla_data": torch.tensor(True),
            "instruction": "Open drawer",
            "intrinsic": torch.tensor([1.0, 1.0, 0.5, 0.5]),
            "n_states": torch.tensor(2, dtype=torch.int32),
            "n_actions": torch.tensor(2, dtype=torch.int32),
            "vision_type": "video",
            "images": torch.zeros(2, 64, 64, 3, dtype=torch.uint8),
            "video_fps": torch.tensor(15.0),
            "future_frames": torch.zeros(1, 64, 64, 3, dtype=torch.uint8),
        }

        messages = formatter.build_messages(sample, prompt_only=False)
        assistant_text = messages[1]["content"][0]["text"]
        user_text = messages[0]["content"][-1]["text"]

        assert "<future_frame>" not in assistant_text
        assert "future visual observations" not in user_text

    def test_future_frame_tokens_odd_frame_count(self):
        """K=3, tps=2 → T_future=1 → 1*4=4 tokens emitted."""
        formatter = Qwen3VLChatFormatter(future_frame_token="<future_frame>")
        formatter.patch_size = 16
        formatter.spatial_merge_size = 2
        formatter.temporal_patch_size = 2
        sample = {
            "is_vla_data": torch.tensor(True),
            "instruction": "Open drawer",
            "intrinsic": torch.tensor([1.0, 1.0, 0.5, 0.5]),
            "n_states": torch.tensor(2, dtype=torch.int32),
            "n_actions": torch.tensor(2, dtype=torch.int32),
            "vision_type": "video",
            "images": torch.zeros(2, 64, 64, 3, dtype=torch.uint8),
            "video_fps": torch.tensor(15.0),
            "future_frames": torch.zeros(3, 64, 64, 3, dtype=torch.uint8),
        }

        messages = formatter.build_messages(sample, prompt_only=False)
        assistant_text = messages[1]["content"][0]["text"]

        assert assistant_text.count("<future_frame>") == 4


# ======================================================================
# Module 5: PrefixKVCache device/dtype casting
# ======================================================================


class TestPrefixKVCacheCast:
    """Verify PrefixKVCache.to() preserves semantics."""

    def test_dtype_cast(self):
        keys = torch.randn(1, 2, 4, 5, 8)
        values = torch.randn(1, 2, 4, 5, 8)
        mask = torch.ones(2, 5, dtype=torch.bool)
        lengths = torch.tensor([5, 5])
        cache = PrefixKVCache(keys=keys, values=values, mask=mask, lengths=lengths)

        cast = cache.to(dtype=torch.float16)
        assert cast.keys.dtype == torch.float16
        assert cast.values.dtype == torch.float16
        # mask is not floating point, should stay bool
        assert cast.mask.dtype == torch.bool

    def test_kv_seq_len_property(self):
        mask = torch.ones(2, 7, dtype=torch.bool)
        empty_kv = torch.zeros(0, 2, 0, 7, 0)
        cache = PrefixKVCache(keys=empty_kv, values=empty_kv, mask=mask, lengths=torch.tensor([7, 7]))
        assert cache.kv_seq_len == 7


# ======================================================================
# Module 6: process_future_frames ff_video_indices tracking
# ======================================================================


class MockVideoProcessor:
    """Minimal mock that returns dummy tensors with correct grid shapes."""

    temporal_patch_size = 2
    patch_size = 16
    merge_size = 2

    def __call__(self, videos, return_tensors="pt", do_sample_frames=False):
        grid_list = []
        total_patches = 0
        for v in videos:
            t_patches = v.shape[0]
            grid_list.append([t_patches, 1, 1])
            total_patches += t_patches
        return {
            "pixel_values_videos": torch.randn(total_patches, 4),
            "video_grid_thw": torch.tensor(grid_list, dtype=torch.long),
        }


class MockProcessorWithVideo:
    def __init__(self):
        self.tokenizer = DummyTokenizerForBatchProcessor()
        self.video_processor = MockVideoProcessor()


class TestProcessFutureFramesFfVideoIndices:
    """Verify ff_video_indices tracking in Qwen3VLBatchProcessor.process_future_frames."""

    def make_batch_processor(self):
        return Qwen3VLBatchProcessor(
            model_name_or_path="demo",
            processor=MockProcessorWithVideo(),
        )

    def test_ff_video_indices_maps_to_video_only_positions(self):
        """Image samples are not counted in video_idx; ff_video_indices reflects video-only ordering."""
        bp = self.make_batch_processor()
        samples = [
            {"vision_type": "video", "future_frames": torch.zeros(4, 8, 8, 3, dtype=torch.uint8)},
            {"vision_type": "image"},
            {"vision_type": "video", "future_frames": torch.zeros(2, 8, 8, 3, dtype=torch.uint8)},
        ]

        result = bp.process_future_frames(samples)

        # video_idx: sample 0 → 0, sample 1 → skipped (image), sample 2 → 1
        assert result["ff_video_indices"].tolist() == [0, 1]
        assert result["ff_grid_thw"].shape[0] == 2

    def test_ff_video_indices_skips_zero_truncated_entries(self):
        """Samples with K < tps frames produce 0 valid patches and are excluded."""
        bp = self.make_batch_processor()
        samples = [
            {"vision_type": "video", "future_frames": torch.zeros(4, 8, 8, 3, dtype=torch.uint8)},
            {"vision_type": "video", "future_frames": torch.zeros(1, 8, 8, 3, dtype=torch.uint8)},
            {"vision_type": "video", "future_frames": torch.zeros(2, 8, 8, 3, dtype=torch.uint8)},
        ]

        result = bp.process_future_frames(samples)

        # Sample 1: K=1 < tps=2 → truncated to 0, skipped.
        # video_idx: sample 0 → 0, sample 1 → 1 (skipped in output), sample 2 → 2
        assert result["ff_video_indices"].tolist() == [0, 2]

    def test_ff_video_indices_empty_when_no_future_frames(self):
        """Batch with no future frames returns empty dict."""
        bp = self.make_batch_processor()
        samples = [
            {"vision_type": "video"},
            {"vision_type": "image"},
        ]

        result = bp.process_future_frames(samples)

        assert result == {}

    def test_ff_video_indices_all_video_samples_with_future_frames(self):
        """When all video samples have future frames, indices are 0..N-1."""
        bp = self.make_batch_processor()
        samples = [
            {"vision_type": "video", "future_frames": torch.zeros(2, 8, 8, 3, dtype=torch.uint8)},
            {"vision_type": "video", "future_frames": torch.zeros(4, 8, 8, 3, dtype=torch.uint8)},
            {"vision_type": "video", "future_frames": torch.zeros(6, 8, 8, 3, dtype=torch.uint8)},
        ]

        result = bp.process_future_frames(samples)

        assert result["ff_video_indices"].tolist() == [0, 1, 2]
        assert result["ff_grid_thw"].shape[0] == 3
