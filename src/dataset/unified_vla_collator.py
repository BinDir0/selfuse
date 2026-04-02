from __future__ import annotations

from copy import deepcopy
import time
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
        debug_capture_texts: bool = False,
        debug_profile_timing: bool = False,
    ):
        self.formatter = formatter
        self.batch_processor = batch_processor
        self.ignore_index = batch_processor.ignore_index
        self.debug_capture_texts = bool(debug_capture_texts)
        self.debug_profile_timing = bool(debug_profile_timing)

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
        collate_start = time.perf_counter()
        full_messages = [self.formatter.build_messages(sample, prompt_only=False) for sample in samples]
        # Keep the original assistant turn only. Appending a fresh assistant prefix here would shift labels.
        full_batch = self.batch_processor.encode_messages(
            messages_batch=full_messages,
            batch_samples=samples,
            add_generation_prompt=False,
            return_rendered_texts=self.debug_capture_texts,
        )

        prompt_messages = [self.formatter.build_messages(sample, prompt_only=True) for sample in samples]
        # Append the assistant prefix so prompt length matches the generation start for both VLM and VLA samples.
        prompt_batch = self.batch_processor.encode_messages(
            messages_batch=prompt_messages,
            batch_samples=samples,
            add_generation_prompt=True,
            return_rendered_texts=self.debug_capture_texts,
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

        # When camera_intrinsic_mode=token, pass intrinsic tensor for camera_encoder.
        # Shape: [B, 1, 4] — one <camera> token per sample, 4D intrinsic [fx, fy, cx, cy].
        # VLM samples carry zero-filled intrinsic; their input_ids have no <camera> token
        # so masked_scatter is a no-op for those rows.
        if self.formatter.camera_intrinsic_mode == "token":
            intrinsics = [s["intrinsic"] for s in samples]
            batch["camera_intrinsic"] = torch.stack(intrinsics).unsqueeze(1)



        if self.debug_capture_texts:
            batch["debug_full_messages"] = full_messages
            batch["debug_prompt_messages"] = prompt_messages
            batch["debug_full_texts"] = full_batch["rendered_texts"]
            batch["debug_prompt_texts"] = prompt_batch["rendered_texts"]

        if self.debug_profile_timing:
            sample_profiles = [
                sample.get("debug_sample_profile")
                for sample in samples
                if sample.get("debug_sample_profile") is not None
            ]
            worker_ids = sorted({int(profile["worker_id"]) for profile in sample_profiles})
            sample_to_data_total_s = sum(float(profile["sample_to_data_s"]) for profile in sample_profiles)
            preprocess_total_s = sum(float(profile["preprocess_total_s"]) for profile in sample_profiles)
            profiled_samples = len(sample_profiles)
            collate_total_s = time.perf_counter() - collate_start
            batch["debug_collate_profile"] = {
                "worker_ids": worker_ids,
                "profiled_samples": profiled_samples,
                "collator_s": collate_total_s,
                "sample_to_data_total_s": sample_to_data_total_s,
                "sample_to_data_avg_s": sample_to_data_total_s / profiled_samples if profiled_samples else 0.0,
                "preprocess_total_s": preprocess_total_s,
                "preprocess_avg_s": preprocess_total_s / profiled_samples if profiled_samples else 0.0,
            }

        return batch

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        return self.collate_raw(samples)
