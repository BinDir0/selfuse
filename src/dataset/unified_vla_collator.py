from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from src.dataset.qwen3_vl_batching import Qwen3VLBatchProcessor, Qwen3VLChatFormatter


class UnifiedVLACollator:
    """Collate raw VLA/VLM samples into one multimodal batch.

    This is the top-level batching contract for the project.

    Input sample contract:
    - VLA samples provide `instruction`, `images`, `intrinsic`, `states`, `actions`, `n_states`, `n_actions`,
      `vision_type`, `video_fps`, and `is_vla_data=True`.
    - VLM samples provide `question`, `answer`, `images`, `vision_type`, and `is_vla_data=False`.

    Output contract:
    - The returned batch always includes HF multimodal fields such as `input_ids`, `attention_mask`,
      `pixel_values`/`pixel_values_videos`, grid metadata, `mm_token_type_ids`, and `answer_start_idx`.
    - For VLM samples, `labels` supervise the assistant text after the prompt boundary.
    - For VLA samples, `labels` stay masked because action supervision is handled by slot tokens and action heads.
    """

    def __init__(
        self,
        formatter: Qwen3VLChatFormatter,
        batch_processor: Qwen3VLBatchProcessor,
    ):
        self.formatter = formatter
        self.batch_processor = batch_processor
        self.ignore_index = batch_processor.ignore_index

        if self.formatter.state_token != self.batch_processor.state_token:
            raise ValueError("Formatter and batch processor must share the same state token.")
        if self.formatter.action_token != self.batch_processor.action_token:
            raise ValueError("Formatter and batch processor must share the same action token.")

    def for_mode(self, mode: str) -> "UnifiedVLACollator":
        collator = deepcopy(self)
        collator.batch_processor.padding_side = "left" if mode == "infer-ar" else "right"
        return collator

    def collate_values(self, values: list[Any]) -> Any:
        if isinstance(values[0], torch.Tensor):
            return torch.stack(values)
        return values

    def collate_raw(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Build one batch from raw dataset samples.

        The collator runs two related encodes:
        - A full-conversation encode for the actual model inputs.
        - A prompt-only encode for all samples to locate where assistant generation starts.

        `add_generation_prompt` is intentionally different across the two passes:
        - Full conversations use `False` because assistant targets are already present in the rendered messages.
        - Prompt-only inputs use `True` so the rendered prompt ends exactly at the assistant prefix where
          generation should begin.
        """
        full_messages = [self.formatter.build_messages(sample, prompt_only=False) for sample in samples]
        # Keep the original assistant turn only. Appending a fresh assistant prefix here would shift labels.
        full_batch = self.batch_processor.encode_messages(
            messages_batch=full_messages,
            batch_samples=samples,
            add_generation_prompt=False,
        )

        prompt_messages = [self.formatter.build_messages(sample, prompt_only=True) for sample in samples]
        # Append the assistant prefix so prompt length matches the generation start for both VLM and VLA samples.
        prompt_batch = self.batch_processor.encode_messages(
            messages_batch=prompt_messages,
            batch_samples=samples,
            add_generation_prompt=True,
        )

        input_ids = full_batch["input_ids"].to(dtype=torch.long)
        attention_mask = full_batch["attention_mask"].to(dtype=torch.long)
        mm_token_type_ids = full_batch["mm_token_type_ids"].to(dtype=torch.long)
        image_grid_thw = full_batch["image_grid_thw"]
        video_grid_thw = full_batch["video_grid_thw"]

        is_vla_mask = torch.stack([sample["is_vla_data"] for sample in samples]).to(
            device=input_ids.device,
            dtype=torch.bool,
        )
        answer_start_idx = prompt_batch["attention_mask"].sum(dim=1).to(device=input_ids.device, dtype=torch.long)
        labels = torch.full_like(input_ids, self.ignore_index)

        vlm_indices = (~is_vla_mask).nonzero(as_tuple=False).flatten().tolist()
        if vlm_indices:
            vlm_answer_start_idx = answer_start_idx[vlm_indices]
            labels[vlm_indices] = self.batch_processor.build_labels(
                input_ids=input_ids[vlm_indices],
                attention_mask=attention_mask[vlm_indices],
                answer_start_idx=vlm_answer_start_idx,
            ).to(device=input_ids.device)

        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": full_batch["pixel_values"],
            "image_grid_thw": image_grid_thw.to(dtype=torch.long) if image_grid_thw is not None else None,
            "pixel_values_videos": full_batch["pixel_values_videos"],
            "video_grid_thw": video_grid_thw.to(dtype=torch.long) if video_grid_thw is not None else None,
            "mm_token_type_ids": mm_token_type_ids,
            "answer_start_idx": answer_start_idx,
        }

        reserved_keys = {
            "images",
            "instruction",
            "question",
            "answer",
            "intrinsic",
            "vision_type",
            "video_fps",
        }
        common_keys = set(samples[0])
        for sample in samples[1:]:
            common_keys &= set(sample)

        for key in sorted(common_keys - reserved_keys):
            batch[key] = self.collate_values([sample[key] for sample in samples])

        return batch

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        return self.collate_raw(samples)
