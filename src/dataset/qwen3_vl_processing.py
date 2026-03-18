from __future__ import annotations

from typing import Any

import numpy as np
import torch


def get_cfg_value(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def to_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def unwrap_batch_dimension(encoded: dict[str, Any]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for key, value in encoded.items():
        array = to_numpy_array(value)
        if array.ndim > 0 and array.shape[0] == 1:
            output[key] = array[0]
        else:
            output[key] = array
    return output


class Qwen3VLProcessor:
    STATE_TOKEN = "<state>"
    ACTION_TOKEN = "<action>"

    def __init__(self, cfg: Any = None, **kwargs):
        if cfg is None:
            cfg = kwargs
        self.cfg = cfg
        self.model_name_or_path = get_cfg_value(cfg, "model_name_or_path")
        self.max_seq_len = int(get_cfg_value(cfg, "max_seq_len"))
        self.ignore_index = int(get_cfg_value(cfg, "ignore_index", -100))
        self.tokenizer_padding = get_cfg_value(cfg, "tokenizer_padding", "max_length")
        self.trust_remote_code = bool(get_cfg_value(cfg, "trust_remote_code", False))
        self.single_image_only = bool(get_cfg_value(cfg, "single_image_only", True))
        self.lowercase_vla_text = bool(get_cfg_value(cfg, "lowercase_vla_text", True))

        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Qwen3-VL processing requires a recent transformers installation."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [self.STATE_TOKEN, self.ACTION_TOKEN],
        })
        self.state_token_id = self.tokenizer.convert_tokens_to_ids(self.STATE_TOKEN)
        self.action_token_id = self.tokenizer.convert_tokens_to_ids(self.ACTION_TOKEN)
        self.image_token = getattr(self.processor, "image_token", "<|image_pad|>")
        self.vision_start_token = getattr(self.processor, "vision_start_token", "<|vision_start|>")
        self.vision_end_token = getattr(self.processor, "vision_end_token", "<|vision_end|>")

    def select_single_image(self, images: Any) -> Any:
        images = to_numpy_array(images)
        if images.ndim == 4:
            image = images[-1]
        elif images.ndim == 3:
            image = images
        else:
            raise ValueError(f"Unsupported image shape: {images.shape}")

        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        return image

    def build_image_prefix(self) -> str:
        return f"{self.vision_start_token}{self.image_token}{self.vision_end_token}"

    def tokenize_text(self, text: str, image: Any, mode: str) -> dict[str, np.ndarray]:
        if mode == "infer-ar":
            self.tokenizer.padding_side = "left"
        else:
            self.tokenizer.padding_side = "right"

        encoded = self.processor(
            images=image,
            text=text,
            padding=self.tokenizer_padding,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return unwrap_batch_dimension(encoded)

    def build_labels(self, input_ids: np.ndarray, attention_mask: np.ndarray, answer_start_idx: int) -> np.ndarray:
        labels = input_ids.copy()
        labels[attention_mask == 0] = self.ignore_index
        labels[:answer_start_idx] = self.ignore_index
        return labels

    def decode(self, output_ids: torch.Tensor | np.ndarray | list[int], skip_special_tokens: bool = True):
        if isinstance(output_ids, torch.Tensor):
            output_ids = output_ids.detach().cpu().tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens)

    def postprocess(self, generated_ids, pixel_values, input_ids):
        return {
            "pred_texts": self.decode(generated_ids, True),
            "instructions": self.decode(input_ids, True),
            "pixel_values": pixel_values,
        }

    def __call__(self, images: np.ndarray, text: str, target: str, mode: str = "train") -> dict[str, np.ndarray]:
        image = self.select_single_image(images)
        prompt = f"{self.build_image_prefix()}{text.strip()}"
        full_text = prompt if (target is None or "infer" in mode) else f"{prompt}\n{target.strip()}"

        prompt_inputs = self.tokenize_text(prompt, image=image, mode=mode)
        full_inputs = self.tokenize_text(full_text, image=image, mode=mode)
        answer_start_idx = int(prompt_inputs["attention_mask"].sum())
        labels = self.build_labels(full_inputs["input_ids"], full_inputs["attention_mask"], answer_start_idx)

        return {
            "input_ids": full_inputs["input_ids"].astype(np.int64),
            "attention_mask": full_inputs["attention_mask"].astype(np.int64),
            "labels": labels.astype(np.int64),
            "pixel_values": full_inputs["pixel_values"],
            "image_grid_thw": full_inputs["image_grid_thw"].astype(np.int64),
            "mm_token_type_ids": full_inputs["mm_token_type_ids"].astype(np.int64),
            "answer_start_idx": np.array(answer_start_idx, dtype=np.int64),
        }


class Qwen3VLVLAProcessor(Qwen3VLProcessor):
    def build_vla_prefix(
        self,
        text: str,
        intrinsic: np.ndarray,
        states: np.ndarray,
    ) -> str:
        clean_text = text.replace(".", "")
        if self.lowercase_vla_text:
            clean_text = clean_text.lower()
        intrinsic = np.asarray(intrinsic, dtype=np.float32)
        intrinsic_str = (
            f"fx:{intrinsic[0]:.2f} fy:{intrinsic[1]:.2f} "
            f"cx:{intrinsic[2]:.2f} cy:{intrinsic[3]:.2f}"
        )
        image_prefix = self.build_image_prefix()
        return (
            f"{image_prefix}Task: {clean_text}, Camera intrinsic: {intrinsic_str}, "
            f"States: {self.STATE_TOKEN * len(states)} "
            f"Actions: "
        )

    def build_vla_suffix(self, actions: np.ndarray) -> str:
        return f"{self.ACTION_TOKEN * len(actions)}"

    def __call__(
        self,
        text: str,
        images: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        intrinsic: np.ndarray,
        objective: str | None = None,
        truncation: bool = True,
        depth_images: np.ndarray | None = None,
        mode: str = "train",
    ) -> dict[str, np.ndarray]:
        del truncation, depth_images
        image = self.select_single_image(images)
        prefix = self.build_vla_prefix(text, intrinsic, states)
        suffix = self.build_vla_suffix(actions)
        prompt_inputs = self.tokenize_text(prefix, image=image, mode=mode)
        full_inputs = self.tokenize_text(f"{prefix}{suffix}", image=image, mode=mode)
        answer_start_idx = int(prompt_inputs["attention_mask"].sum())
        del objective
        labels = np.full_like(full_inputs["input_ids"], self.ignore_index, dtype=np.int64)

        return {
            "input_ids": full_inputs["input_ids"].astype(np.int64),
            "attention_mask": full_inputs["attention_mask"].astype(np.int64),
            "labels": labels.astype(np.int64),
            "pixel_values": full_inputs["pixel_values"],
            "image_grid_thw": full_inputs["image_grid_thw"].astype(np.int64),
            "mm_token_type_ids": full_inputs["mm_token_type_ids"].astype(np.int64),
            "answer_start_idx": np.array(answer_start_idx, dtype=np.int64),
        }
