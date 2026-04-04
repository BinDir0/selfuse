from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from src.dataset.qwen3_vl_batching import Qwen3VLChatFormatter
from src.dataset.unified_vla_collator import UnifiedVLACollator
from src.policy.legendvla_inference_wrapper import LegendVLAInference


class DummyBatchProcessor:
    def __init__(self):
        self.ignore_index = -100
        self.padding_side = "right"
        self.state_token = "<state>"
        self.action_token = "<action>"
        self.action_token_id = 102
        self.processor_call_kwargs = {"text_kwargs": {"padding": "max_length", "max_length": 32}}

    def encode_messages(
        self,
        messages_batch,
        batch_samples,
        add_generation_prompt,
        return_rendered_texts: bool = False,
    ):
        del messages_batch
        batch_size = len(batch_samples)
        if add_generation_prompt:
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
            input_ids = []
            attention_mask = []
            for sample in batch_samples:
                if bool(sample["is_vla_data"].item()):
                    ids = [1, 2]
                    ids.extend([101] * int(sample["n_states"].item()))
                    ids.extend([102] * int(sample["n_actions"].item()))
                else:
                    ids = [10, 11, 12, 13, 14]
                input_ids.append(ids)

            max_len = max(len(ids) for ids in input_ids)
            for row in input_ids:
                pad_len = max_len - len(row)
                attention_mask.append([1] * len(row) + [0] * pad_len)
                row.extend([0] * pad_len)

            batch = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "pixel_values": torch.randn(batch_size, 3, 8, 8),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * batch_size, dtype=torch.long),
                "pixel_values_videos": None,
                "video_grid_thw": None,
                "mm_token_type_ids": torch.zeros(batch_size, max_len, dtype=torch.long),
            }

        if return_rendered_texts:
            batch["rendered_texts"] = ["dummy" for _ in range(batch_size)]
        return batch

    def build_labels(self, input_ids, attention_mask, answer_start_idx):
        labels = input_ids.clone()
        labels = labels.masked_fill(attention_mask == 0, self.ignore_index)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        labels = labels.masked_fill(positions < answer_start_idx.unsqueeze(1), self.ignore_index)
        return labels


class DummyTokenizerForBatchProcessor:
    def add_special_tokens(self, _tokens):
        return 0

    def convert_tokens_to_ids(self, token):
        if token == "<state>":
            return 101
        if token == "<action>":
            return 102
        return 0


class DummyProcessorForBatchProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizerForBatchProcessor()


def build_dummy_collator() -> UnifiedVLACollator:
    return UnifiedVLACollator(
        formatter=Qwen3VLChatFormatter(),
        batch_processor=DummyBatchProcessor(),
    )


def build_wrapper_stub(
    *,
    history_pad_mode: str = "truncate",
    mode: str = "flow",
    data_collator: Any | None = None,
) -> LegendVLAInference:
    wrapper = LegendVLAInference.__new__(LegendVLAInference)
    nn.Module.__init__(wrapper)
    wrapper.dtype = torch.float32
    wrapper.model = None
    wrapper.data_collator = build_dummy_collator() if data_collator is None else data_collator
    wrapper.normalizer = None
    wrapper.use_relative_action = True
    wrapper.mode = mode
    wrapper.default_instruction = "pick the cup"
    wrapper.action_horizon = 4
    wrapper.action_dim = 2
    wrapper.state_horizon = 4
    wrapper.state_dim = 2
    wrapper.image_horizon = 1
    wrapper.image_stride = 1
    wrapper.video_base_fps = 30.0
    wrapper.target_image_size = None
    wrapper.history_pad_mode = history_pad_mode
    wrapper.ar_max_new_tokens = wrapper.action_horizon
    wrapper.ar_temperature = 1.0
    wrapper.ar_cfg = 1.0
    wrapper.metadata = {"mode": mode}
    wrapper._model_compiled = False
    wrapper.compile_cfg = None
    return wrapper


def make_obs(states: list[list[float]] | torch.Tensor, instruction: str | None = None) -> dict[str, Any]:
    if torch.is_tensor(states):
        state_array = states.detach().cpu().numpy()
    else:
        state_array = torch.tensor(states, dtype=torch.float32).numpy()

    return {
        "image": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
        "states": state_array,
        "intrinsic": torch.tensor([1.0, 1.0, 0.5, 0.5], dtype=torch.float32).numpy(),
        "instruction": instruction,
    }


def make_expected_vla_sample(
    *,
    states: torch.Tensor,
    n_states: int,
    instruction: str = "pick the cup",
    action_horizon: int = 4,
) -> dict[str, Any]:
    return {
        "images": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
        "instruction": instruction,
        "intrinsic": torch.tensor([1.0, 1.0, 0.5, 0.5], dtype=torch.float32),
        "vision_type": "video",
        "video_fps": torch.tensor(30.0, dtype=torch.float32),
        "states": states,
        "n_states": torch.tensor(n_states, dtype=torch.long),
        "n_actions": torch.tensor(action_horizon, dtype=torch.long),
        "is_vla_data": torch.tensor(True, dtype=torch.bool),
    }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def make_debug_args(**kwargs: Any) -> SimpleNamespace:
    defaults = {
        "config": str(repo_root() / "src" / "config" / "experiment" / "legendvla_qwen3_vl.yaml"),
        "output_dir": str(repo_root() / "tmp" / "codex_debug"),
        "sample_count": 4,
        "split": "val",
        "dataset_kind": "vla",
        "num_workers": 0,
        "normalizer_path": None,
        "seed": 0,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)

