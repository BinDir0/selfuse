from __future__ import annotations

import logging
import pathlib
import pickle
from typing import Any, Dict

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from src.utils.checkpoint_util import load_checkpoint


log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval, replace=True)


class LegendVLAInference(nn.Module):
    """Inference wrapper for LegendVLA policy serving.

    Bridges raw observations from the robot environment to model forward pass,
    implementing the interface expected by RuntimeEngine:
        prepare_process(obs) -> build_model_inputs(batch) -> forward(inputs) -> post_process(actions)
    """

    def __init__(
        self,
        model_config_path: str,
        checkpoint_path: str = None,
        mode: str = "flow",
        use_mixed_precision: bool = True,
        tokenizer_padding: str = "longest",
        max_length: int | None = None,
        default_instruction: str | None = None,
        flow_sampling_steps: int = None,
        ar_max_new_tokens: int | None = None,
        ar_temperature: float = 1.0,
        ar_cfg: float = 1.0,
        normalizer_path: str = None,
        use_relative_action: bool = False,
        compile: Any = None,
    ) -> None:
        super().__init__()
        self.dtype = torch.bfloat16 if use_mixed_precision else torch.float32

        model_cfg = OmegaConf.load(model_config_path)

        self.model: nn.Module = hydra.utils.instantiate(model_cfg.policy)
        if checkpoint_path:
            load_checkpoint(self.model, checkpoint_path)
        if self.dtype != torch.float32:
            self.model.to(dtype=self.dtype)
            log.info("Cast model weights to %s", self.dtype)
        self.model.eval()

        if flow_sampling_steps:
            self.model.num_inference_steps = flow_sampling_steps
            if self.model.diffloss is not None:
                self.model.diffloss.num_inference_steps = flow_sampling_steps

        collator_mode = "infer" if mode == "flow" else "infer-ar"
        self.data_collator = hydra.utils.instantiate(model_cfg.data_collator, mode=collator_mode)
        self.configure_batch_processor_text_kwargs(tokenizer_padding=tokenizer_padding, max_length=max_length)

        self.normalizer = self.load_normalizer(normalizer_path) if normalizer_path else None
        self.use_relative_action = use_relative_action

        self.mode = mode
        self.default_instruction = default_instruction
        self.action_horizon = int(self.model.shape_meta["action"]["horizon"])
        self.action_dim = int(self.model.shape_meta["action"]["shape"][0])
        self.state_horizon = int(self.model.shape_meta["obs"]["state"]["horizon"])
        self.state_dim = int(self.model.shape_meta["obs"]["state"]["shape"][0])
        self.image_horizon = int(self.model.shape_meta["obs"]["rgb"]["horizon"])
        self.image_stride = int(self.model.shape_meta["obs"]["rgb"].get("stride", 1))
        dataset_cfg = getattr(model_cfg, "dataset", None)
        vla_dataset_cfg = getattr(dataset_cfg, "vla_dataset", None)
        self.history_pad_mode = str(getattr(vla_dataset_cfg, "history_pad_mode", "truncate"))
        data_cfg = getattr(model_cfg, "data", None)
        self.video_base_fps = float(getattr(data_cfg, "video_base_fps", getattr(model_cfg, "video_base_fps", 30.0)))
        self.ar_max_new_tokens = ar_max_new_tokens or self.action_horizon
        self.ar_temperature = ar_temperature
        self.ar_cfg = ar_cfg

        self.metadata = {
            "mode": mode,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
        }
        self._model_compiled = False
        self.compile_cfg = None
        if compile is not None:
            self.compile_cfg = (
                OmegaConf.to_container(compile, resolve=True)
                if OmegaConf.is_config(compile)
                else compile
            )

    @property
    def shape_meta(self) -> dict:
        return self.get_model_core().shape_meta

    def get_model_core(self) -> nn.Module:
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        if hasattr(model, "module"):
            model = model.module
        return model

    def maybe_compile_model(self) -> None:
        if self.compile_cfg is None or not self.compile_cfg.get("enabled", False) or self._model_compiled:
            return

        compile_kwargs = {
            key: value
            for key, value in self.compile_cfg.items()
            if key != "enabled" and value is not None
        }
        log.info("Compiling model with kwargs=%s", compile_kwargs)
        self.model = torch.compile(self.model, **compile_kwargs)
        self._model_compiled = True

    def configure_batch_processor_text_kwargs(
        self,
        tokenizer_padding: str,
        max_length: int | None,
    ) -> None:
        processor_call_kwargs = self.data_collator.batch_processor.processor_call_kwargs
        text_kwargs = processor_call_kwargs.get("text_kwargs")

        if isinstance(text_kwargs, dict):
            text_kwargs["padding"] = tokenizer_padding
            if max_length is not None:
                text_kwargs["max_length"] = max_length
            return

        processor_call_kwargs["padding"] = tokenizer_padding
        if max_length is not None:
            processor_call_kwargs["max_length"] = max_length

    def load_normalizer(self, path: str) -> dict:
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Normalizer not found: {path}")
        with open(path, "rb") as f:
            normalizer = pickle.load(f)
        log.info("Loaded normalizer from %s", path)
        return normalizer

    def extract_intrinsic(self, intrinsic: np.ndarray) -> np.ndarray:
        intrinsic = np.asarray(intrinsic, dtype=np.float32)
        if intrinsic.shape == (3, 3):
            intrinsic = np.array([
                intrinsic[0, 0], intrinsic[1, 1],
                intrinsic[0, 2], intrinsic[1, 2],
            ])
        return intrinsic.reshape(-1)

    def prepare_history(self, data: np.ndarray, horizon: int) -> tuple[np.ndarray, int]:
        """Pad or truncate observation history to target horizon.

        Matches gather_history_frames in wds_dataset.py:
        - repeat: left-pad by repeating the earliest entry, length = horizon
        - truncate: keep as-is if shorter, length <= horizon
        If input exceeds horizon, takes the last ``horizon`` entries.
        Returns (data, valid_count).
        """
        current = data.shape[0]
        if current >= horizon:
            return data[-horizon:], horizon

        if self.history_pad_mode == "repeat":
            pad_count = horizon - current
            if current == 0:
                return np.zeros((horizon, *data.shape[1:]), dtype=data.dtype), 0
            padding = np.repeat(data[:1], pad_count, axis=0)
            return np.concatenate([padding, data], axis=0), horizon

        # truncate: return as-is, valid count is actual length
        return data, current

    def prepare_process(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert one runtime observation into the single-sample schema expected by the collator.

        Input contract:
        - `obs["image"]` is the already stacked visual history used as one Qwen video sample.
        - `obs["states"]` is shaped `[T, D]` before padding/truncation.
        - `obs["intrinsic"]` is either a flattened camera intrinsic vector or a matrix that can be reduced to
          `[fx, fy, cx, cy]`.
        - `obs["instruction"]` is optional when `default_instruction` is configured.

        Output contract:
        - The returned `sample` matches the VLA-side raw sample schema that `UnifiedVLACollator` consumes during
          training, so inference reuses the exact same batching path.
        """
        instruction = obs.get("instruction") or self.default_instruction
        if instruction is None:
            raise ValueError("Inference requires an instruction.")

        images, _ = self.prepare_history(np.asarray(obs["image"]), self.image_horizon)

        states = np.asarray(obs["states"], dtype=np.float32)
        if self.normalizer is not None:
            key = "states" if self.use_relative_action else "motions"
            states = self.normalizer[key](states)
        states, n_states = self.prepare_history(states, self.state_horizon)
        # States require fixed-length tensor for batch collation;
        # zero-pad to state_horizon when truncate mode returns fewer entries.
        if states.shape[0] < self.state_horizon:
            padded = np.zeros((self.state_horizon, self.state_dim), dtype=np.float32)
            padded[:states.shape[0]] = states
            states = padded

        sample = {
            "images": torch.as_tensor(images),
            "instruction": instruction,
            "intrinsic": torch.from_numpy(self.extract_intrinsic(obs["intrinsic"])),
            "vision_type": "video",
            "video_fps": torch.tensor(self.video_base_fps / self.image_stride, dtype=torch.float32),
            "states": torch.from_numpy(states),
            "n_states": torch.tensor(n_states, dtype=torch.int32),
            "actions": torch.zeros((self.action_horizon, self.action_dim), dtype=torch.float32),
            "actions_valid_mask": torch.ones((self.action_horizon, self.action_dim), dtype=torch.bool),
            "n_actions": torch.tensor(self.action_horizon, dtype=torch.int32),
            "is_vla_data": torch.tensor(True, dtype=torch.bool),
        }
        batch = self.data_collator([sample])

        # Normalize executed action prefix for RTC condition (same space as model actions)
        prev_action_chunk = obs.get("prev_action_chunk")
        if prev_action_chunk is not None:
            prev_action_chunk = np.asarray(prev_action_chunk, dtype=np.float32)
            if self.normalizer is not None:
                key = "actions" if self.use_relative_action else "motions"
                prev_action_chunk = self.normalizer[key](prev_action_chunk)
            batch["prev_action_chunk"] = prev_action_chunk

        return batch

    def build_model_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Construct model input tensors from one collated batch."""
        inputs: dict[str, Any] = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"].to(self.dtype)
            if batch["pixel_values"] is not None else None,
            "image_grid_thw": batch["image_grid_thw"],
            "pixel_values_videos": batch["pixel_values_videos"].to(self.dtype)
            if batch["pixel_values_videos"] is not None else None,
            "video_grid_thw": batch["video_grid_thw"],
            "mm_token_type_ids": batch["mm_token_type_ids"],
            "states": batch["states"].to(self.dtype),
            "n_states": batch["n_states"],
            "n_actions": batch["n_actions"].to(dtype=torch.long),
            "is_vla_data": batch["is_vla_data"],
        }

        if "camera_intrinsic" in batch:
            inputs["camera_intrinsic"] = batch["camera_intrinsic"].to(self.dtype)

        if self.mode == "flow":
            inputs["answer_start_idx"] = batch["answer_start_idx"]
            inputs["actions"] = batch["actions"].to(self.dtype)
            inputs["actions_valid_mask"] = batch["actions_valid_mask"].to(dtype=torch.bool)

            # RTC condition: normalized prefix from prepare_process → pad to action_horizon
            prev_action_chunk = batch.get("prev_action_chunk")
            if prev_action_chunk is not None:
                if not isinstance(prev_action_chunk, torch.Tensor):
                    prev_action_chunk = torch.as_tensor(prev_action_chunk)
                prev_action_chunk = prev_action_chunk.to(dtype=self.dtype)
                if prev_action_chunk.ndim == 2:
                    prev_action_chunk = prev_action_chunk.unsqueeze(0)
                inference_delay = prev_action_chunk.shape[1]
                if inference_delay < self.action_horizon:
                    pad = prev_action_chunk.new_zeros(
                        prev_action_chunk.shape[0],
                        self.action_horizon - inference_delay,
                        prev_action_chunk.shape[2],
                    )
                    prev_action_chunk = torch.cat([prev_action_chunk, pad], dim=1)
                inputs["prev_action_chunk"] = prev_action_chunk
                inputs["inference_delay"] = inference_delay

        return inputs

    def post_process(self, actions: torch.Tensor) -> torch.Tensor:
        """Unnormalize predicted actions back to physical space."""
        if self.normalizer is None:
            return actions
        key = "actions" if self.use_relative_action else "motions"
        return self.normalizer[key].unnormalize(actions)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run model inference. RTC fields (prev_action_chunk, inference_delay)
        are carried inside ``inputs`` when present."""
        self.maybe_compile_model()
        if self.mode == "flow":
            return self.model("infer_action", inputs)
        return self.model(
            "infer_vla", inputs,
            max_new_tokens=self.ar_max_new_tokens,
            temperature=self.ar_temperature,
            cfg=self.ar_cfg,
        )["generated_actions"]
