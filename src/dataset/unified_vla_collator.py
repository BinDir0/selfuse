from __future__ import annotations

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
        mode: str = "train",
        debug_capture_texts: bool = False,
        debug_profile_timing: bool = False,
    ):
        self.formatter = formatter
        self.batch_processor = batch_processor
        self.ignore_index = batch_processor.ignore_index
        self.debug_capture_texts = bool(debug_capture_texts)
        self.debug_profile_timing = bool(debug_profile_timing)
        self.mode = mode
        self.prompt_only_input = "infer" in mode
        self.batch_processor.padding_side = "left" if mode == "infer-ar" else "right"

        if self.formatter.state_token != self.batch_processor.state_token:
            raise ValueError("Formatter and batch processor must share the same state token.")
        if self.formatter.action_token != self.batch_processor.action_token:
            raise ValueError("Formatter and batch processor must share the same action token.")

        # Derive temporal_patch_size from the processor so the formatter
        # can compute the correct <future_frame> token count when obs+future
        # are packed into one temporal sequence for the target encoder.
        if self.formatter.ff_tokens_per_frame > 0:
            video_proc = getattr(self.batch_processor.processor, "video_processor", None)
            tps = getattr(video_proc, "temporal_patch_size", 1) if video_proc else 1
            self.formatter.ff_temporal_patch_size = tps

    def collate_values(self, values: list[Any]) -> Any:
        if isinstance(values[0], torch.Tensor):
            return torch.stack(values)
        return values

    def collate_raw(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Build one batch from raw dataset samples.

        Two encoding strategies depending on ``prompt_only_input``:

        - **Training** (``prompt_only_input=False``):
          Full-conversation encode (with assistant content) for model inputs,
          plus a prompt-only encode to locate ``answer_start_idx``.
        - **Inference** (``prompt_only_input=True``):
          Prompt-only encode with ``add_generation_prompt=True`` only.
          No assistant content in ``input_ids``; downstream modules use
          ``answer_start_idx`` to locate the generation boundary.
        """
        collate_start = time.perf_counter()

        # Prompt-only encode: always needed for answer_start_idx.
        prompt_messages = [self.formatter.build_messages(sample, prompt_only=True) for sample in samples]
        prompt_batch = self.batch_processor.encode_messages(
            messages_batch=prompt_messages,
            batch_samples=samples,
            add_generation_prompt=True,
            return_rendered_texts=self.debug_capture_texts,
        )

        if self.prompt_only_input:
            main_batch = prompt_batch
            full_messages = None
        else:
            full_messages = [self.formatter.build_messages(sample, prompt_only=False) for sample in samples]
            main_batch = self.batch_processor.encode_messages(
                messages_batch=full_messages,
                batch_samples=samples,
                add_generation_prompt=False,
                return_rendered_texts=self.debug_capture_texts,
            )

        input_ids = main_batch["input_ids"].to(dtype=torch.long)
        attention_mask = main_batch["attention_mask"].to(dtype=torch.long)
        mm_token_type_ids = main_batch["mm_token_type_ids"].to(dtype=torch.long)
        image_grid_thw = main_batch["image_grid_thw"]
        video_grid_thw = main_batch["video_grid_thw"]

        is_vla_mask = torch.stack([sample["is_vla_data"] for sample in samples]).to(
            device=input_ids.device,
            dtype=torch.bool,
        )
        answer_start_idx = prompt_batch["attention_mask"].sum(dim=1).to(device=input_ids.device, dtype=torch.long)
        labels = torch.full_like(input_ids, self.ignore_index)

        if not self.prompt_only_input:
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
            "pixel_values": main_batch["pixel_values"],
            "image_grid_thw": image_grid_thw.to(dtype=torch.long) if image_grid_thw is not None else None,
            "pixel_values_videos": main_batch["pixel_values_videos"],
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
            # future_frames is only present in VLA samples; handled below.
            "future_frames",
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

        # Preprocess future frames for the target encoder.
        # VLM samples never carry future_frames, so B_vla-sized batching
        # is correct — masked_scatter only targets <future_frame> tokens.
        ff_samples_raw = [s for s in samples if "future_frames" in s]
        if ff_samples_raw:
            processor = self.batch_processor.processor
            tps = self.formatter.ff_temporal_patch_size

            if tps > 1:
                # Combined obs+future as a temporal sequence (video).
                # This enables MEM temporal causal attention across
                # observation and future frames in the self_vit target
                # encoder. The HF video processor handles temporal packing.
                videos = []
                n_obs = None
                for s in ff_samples_raw:
                    obs = s["images"] if s["images"].ndim == 4 else s["images"].unsqueeze(0)
                    n_obs_cur = obs.shape[0]
                    assert n_obs_cur % tps == 0, (
                        f"Observation frame count ({n_obs_cur}) must be divisible by "
                        f"temporal_patch_size ({tps}) so obs/future features split "
                        f"cleanly after the ViT spatial merger."
                    )
                    assert s["future_frames"].shape[0] % tps == 0, (
                        f"Future frame count ({s['future_frames'].shape[0]}) must be "
                        f"divisible by temporal_patch_size ({tps})."
                    )
                    if n_obs is None:
                        n_obs = n_obs_cur
                    videos.append(torch.cat([obs, s["future_frames"]], dim=0))
                processed = processor.video_processor(videos=videos, return_tensors="pt")
                pv = processed["pixel_values_videos"]
                if pv.ndim == 3:
                    pv = pv.reshape(-1, pv.shape[-1])
                batch["ff_pixel_values"] = pv
                batch["ff_grid_thw"] = processed["video_grid_thw"]
                batch["ff_n_obs_frames"] = torch.tensor(
                    [n_obs] * len(ff_samples_raw), dtype=torch.long,
                )
            else:
                # Independent per-frame processing (no temporal packing).
                ff_values = [s["future_frames"] for s in ff_samples_raw]
                ff_stacked = self.collate_values(ff_values)
                if not isinstance(ff_stacked, torch.Tensor):
                    import numpy as np
                    ff_stacked = torch.from_numpy(np.stack(ff_values))
                B_vla, K = ff_stacked.shape[:2]
                image_list = [ff_stacked[b, k] for b in range(B_vla) for k in range(K)]
                processed = processor.image_processor(images=image_list, return_tensors="pt")
                pv = processed["pixel_values"]
                if pv.ndim == 3:
                    pv = pv.reshape(-1, pv.shape[-1])
                batch["ff_pixel_values"] = pv
                batch["ff_grid_thw"] = processed["image_grid_thw"]

        if self.debug_capture_texts:
            batch["debug_prompt_messages"] = prompt_messages
            batch["debug_prompt_texts"] = prompt_batch["rendered_texts"]
            if full_messages is not None:
                batch["debug_full_messages"] = full_messages
                batch["debug_full_texts"] = main_batch["rendered_texts"]

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
