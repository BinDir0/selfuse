from __future__ import annotations

from typing import Any

import torch


def ensure_uint8_vision_tensor(images: torch.Tensor, field_name: str) -> None:
    """Reject normalized float inputs because this batching path expects raw uint8 pixels."""
    if images.dtype != torch.uint8:
        raise ValueError(f"{field_name} must use dtype torch.uint8 before Qwen3-VL batching, got {images.dtype}")


def count_images(images: torch.Tensor) -> int:
    """Count image placeholders for one sample.

    Contract:
    - `images` is `[H, W, C]` for one image or `[T, H, W, C]` for multiple images.
    """
    if images.ndim == 3:
        return 1
    if images.ndim == 4:
        return int(images.shape[0])
    raise ValueError(f"Unsupported image container shape: {tuple(images.shape)}")


def build_sample_images(images: torch.Tensor) -> list[torch.Tensor]:
    """Convert one image sample into a normalized image batch for the HF processor.

    The returned value is always a list of image frames so mixed image batches do not need
    placeholder empty lists or mixed tensor/list top-level containers.
    Raw visual inputs must stay in `torch.uint8`; normalized float images should fail here.
    """
    ensure_uint8_vision_tensor(images, field_name="images")
    if images.ndim == 3:
        return [images]
    if images.ndim == 4:
        return [frame for frame in images]
    raise ValueError(f"Unsupported image container shape: {tuple(images.shape)}")


def build_sample_video(video: torch.Tensor) -> torch.Tensor:
    """Convert one video sample into the structure expected by the HF processor.

    Raw video frames must stay in `torch.uint8`; normalized float videos should fail here.
    """
    ensure_uint8_vision_tensor(video, field_name="video")
    if video.ndim != 4:
        raise ValueError(f"Unsupported video container shape: {tuple(video.shape)}")
    return video


def build_video_metadata(video: torch.Tensor, video_fps: float) -> dict[str, Any]:
    """Build video metadata for already-sampled frames.

    Contract:
    - `video` is `[T, H, W, C]`.
    - `video_fps` is the effective FPS after dataset stride sampling.
    """
    frame_count = int(video.shape[0])
    return {
        "total_num_frames": frame_count,
        "fps": float(video_fps),
        "frames_indices": list(range(frame_count)),
    }


class Qwen3VLChatFormatter:
    """Convert one project sample into the HF chat message structure expected by Qwen3-VL."""

    def __init__(
        self,
        state_token: str = "<state>",
        action_token: str = "<action>",
        camera_token: str = "<camera>",
        camera_intrinsic_mode: str = "text",
        lowercase_vla_text: bool = True,
        predict_future_frames: bool = False,
    ):
        self.state_token = state_token
        self.action_token = action_token
        self.camera_token = camera_token
        self.camera_intrinsic_mode = camera_intrinsic_mode
        self.lowercase_vla_text = lowercase_vla_text
        self.predict_future_frames = predict_future_frames

    def build_visual_content(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        vision_type = sample["vision_type"]
        if vision_type == "video":
            return [{"type": "video"}]
        if vision_type == "image":
            return [{"type": "image"} for _ in range(count_images(sample["images"]))]
        raise ValueError(f"Unsupported vision_type: {vision_type}")

    def build_vla_user_text(self, instruction: str, intrinsic: torch.Tensor, n_states: torch.Tensor) -> str:
        clean_text = str(instruction).replace(".", "").strip()
        if self.lowercase_vla_text:
            clean_text = clean_text.lower()
        state_slots = self.state_token * int(n_states.item())
        if self.camera_intrinsic_mode == "token":
            camera_part = f"Camera intrinsic: {self.camera_token}."
        else:
            intrinsic_values = intrinsic.tolist()
            intrinsic_str = (
                f"fx:{intrinsic_values[0]:.2f} fy:{intrinsic_values[1]:.2f} "
                f"cx:{intrinsic_values[2]:.2f} cy:{intrinsic_values[3]:.2f}"
            )
            camera_part = f"Camera intrinsic: {intrinsic_str}."
        return (
            f"Task: {clean_text}. {camera_part} "
            f"States: {state_slots}."
        )

    def build_messages(self, sample: dict[str, Any], prompt_only: bool = False) -> list[dict[str, Any]]:
        """Build one chat conversation from a raw sample.

        Contract:
        - VLA samples are converted into a user turn with visual placeholders plus task text, followed by an
          assistant turn made of repeated `<action>` slot tokens.
        - VLM samples are converted into a standard user/assistant question-answer pair.
        - `prompt_only=True` keeps only the user turn so callers can measure the prompt boundary before assistant
          generation starts.
        """
        is_vla = bool(sample["is_vla_data"].item())

        if is_vla:
            user_text = self.build_vla_user_text(
                instruction=sample["instruction"],
                intrinsic=sample["intrinsic"],
                n_states=sample["n_states"],
            )
            assistant_text = self.action_token * int(sample["n_actions"].item())
            if self.predict_future_frames:
                user_text += " Predict the next action sequence and future frames."
            else:
                user_text += " Predict the next action sequence."
        else:
            user_text = str(sample["question"]).strip()
            assistant_text = str(sample["answer"]).strip()

        messages = [
            {
                "role": "user",
                "content": [*self.build_visual_content(sample), {"type": "text", "text": user_text}],
            }
        ]
        if not prompt_only:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                }
            )
        return messages


class Qwen3VLBatchProcessor:
    """Bridge project batches to the native HF Qwen3-VL processor.

    Kwarg routing contract:
    - `processor_init_kwargs` are forwarded to `AutoProcessor.from_pretrained(...)`.
      Loader kwargs such as `trust_remote_code` are consumed by `transformers` at load time, while recognized
      modality kwargs such as `size` continue to the underlying tokenizer/image/video sub-processors.
    - `processor_call_kwargs` are forwarded to `processor(...)` at encode time.
      `transformers` then routes recognized flat kwargs to text/image/video branches and ignores unsupported keys.
    """

    def __init__(
        self,
        model_name_or_path: str,
        processor_init_kwargs: dict[str, Any] | None = None,
        processor_call_kwargs: dict[str, Any] | None = None,
        ignore_index: int = -100,
        padding_side: str = "right",
        state_token: str = "<state>",
        action_token: str = "<action>",
        processor: Any = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.processor_init_kwargs = dict(processor_init_kwargs or {})
        self.processor_call_kwargs = dict(processor_call_kwargs or {})
        self.ignore_index = int(ignore_index)
        self.padding_side = padding_side
        self.state_token = state_token
        self.action_token = action_token
        self.processor = processor if processor is not None else self.init_processor()
        self.tokenizer = self.processor.tokenizer
        special_tokens = [self.state_token, self.action_token]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.action_token_id = int(self.tokenizer.convert_tokens_to_ids(self.action_token))

    def init_processor(self):
        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Qwen3VLBatchProcessor requires transformers with Qwen3-VL support."
            ) from exc

        return AutoProcessor.from_pretrained(
            self.model_name_or_path,
            **self.processor_init_kwargs,
        )

    def render_chat_texts(
        self,
        messages_batch: list[list[dict[str, Any]]],
        add_generation_prompt: bool,
    ) -> list[str]:
        rendered = self.processor.apply_chat_template(
            messages_batch,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if isinstance(rendered, str):
            return [rendered]
        return list(rendered)

    def build_vision_inputs(self, batch_samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Build modality inputs using only real image/video samples.

        Contract:
        - `images` contains only image samples, ordered by their image placeholder appearance in the
          rendered batch texts.
        - `videos` contains only video samples, ordered by their video placeholder appearance.
        - No batch-sized empty placeholder lists are inserted for missing modalities.
        """
        images: list[Any] = []
        videos: list[Any] = []
        video_metadata: list[dict[str, Any]] = []

        for sample in batch_samples:
            vision_type = sample["vision_type"]
            if vision_type == "image":
                images.append(build_sample_images(sample["images"]))
            elif vision_type == "video":
                video = build_sample_video(sample["images"])
                video_fps = float(sample["video_fps"].item())
                videos.append(video)
                video_metadata.append(build_video_metadata(video, video_fps))
            else:
                raise ValueError(f"Unsupported vision_type: {vision_type}")

        processor_inputs: dict[str, Any] = {}
        if images:
            processor_inputs["images"] = images
        if videos:
            processor_inputs["videos"] = videos
            processor_inputs["video_metadata"] = video_metadata
            processor_inputs["do_sample_frames"] = False
        return processor_inputs

    def encode_messages(
        self,
        messages_batch: list[list[dict[str, Any]]],
        batch_samples: list[dict[str, Any]],
        add_generation_prompt: bool,
        return_rendered_texts: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run the native Qwen3-VL processor on one batch.

        Contract:
        - Image samples use `vision_type="image"` and `images` shaped `[H, W, C]` or `[T, H, W, C]`.
        - Video samples use `vision_type="video"`, `images` shaped `[T, H, W, C]`, and `video_fps`.
        - Processor kwargs are passed directly from `processor_call_kwargs`.
        - Do not set text truncation here because it can break multimodal token alignment.
        """
        self.processor.tokenizer.padding_side = self.padding_side

        rendered_texts = self.render_chat_texts(
            messages_batch=messages_batch,
            add_generation_prompt=add_generation_prompt,
        )
        encoded = self.processor(
            text=rendered_texts,
            **self.processor_call_kwargs,
            **self.build_vision_inputs(batch_samples),
        )
        batch = dict(encoded)
        batch.setdefault("pixel_values", None)
        batch.setdefault("image_grid_thw", None)
        batch.setdefault("pixel_values_videos", None)
        batch.setdefault("video_grid_thw", None)
        if "mm_token_type_ids" not in batch:
            batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"])
        if return_rendered_texts:
            batch["rendered_texts"] = rendered_texts
        return batch

    def build_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        answer_start_idx: torch.Tensor,
    ) -> torch.Tensor:
        labels = input_ids.clone()
        labels = labels.masked_fill(attention_mask == 0, self.ignore_index)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        labels = labels.masked_fill(positions < answer_start_idx.unsqueeze(1), self.ignore_index)
        return labels

