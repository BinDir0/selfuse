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

from src.dataset.data_transforms import process_image
from src.utils.checkpoint_util import load_checkpoint
from src.utils.visual_attention import (
    aggregate_visual_attention,
    compute_middle_layer_range,
    renormalize_visual_subset,
    reshape_visual_to_grid,
    stack_visual_attention,
)


log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval, replace=True)


class LegendVLAInference(nn.Module):
    """Inference wrapper for LegendVLA policy serving.

    Bridges raw observations from the robot environment to model forward pass,
    implementing the interface expected by RuntimeEngine:
        prepare_process(obs) -> forward(inputs) -> post_process(actions)
    """

    def __init__(
        self,
        model_config_path: str,
        checkpoint_path: str = None,
        pretrained_vlm_path: str = None,
        teacher_path: str | None = None,
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
        attention_recording: bool = False,
        compile: Any = None,
    ) -> None:
        super().__init__()
        self.dtype = torch.bfloat16 if use_mixed_precision else torch.float32

        model_cfg = OmegaConf.load(model_config_path)
        if pretrained_vlm_path:
            OmegaConf.update(model_cfg, "pretrained.model_name_or_path", pretrained_vlm_path)
            log.info("Overriding pretrained VLM path to %s", pretrained_vlm_path)
        if teacher_path:
            OmegaConf.update(model_cfg, "policy.frozen_teacher.model_name_or_path", teacher_path)
            log.info("Overriding frozen_teacher path to %s", teacher_path)
        if attention_recording:
            OmegaConf.update(model_cfg, "policy.flow_expert.attn_implementation", "eager")
            log.info("Attention recording: set flow_expert attn_implementation to eager")

        self.model: nn.Module = hydra.utils.instantiate(model_cfg.policy)
        if checkpoint_path:
            load_checkpoint(self.model, checkpoint_path)
        else:
            log.warning("No checkpoint_path provided — model uses initial pretrained weights only.")
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
        self.action_horizon = self.model.shape_meta["action"]["horizon"]
        self.action_dim = self.model.shape_meta["action"]["shape"][0]
        self.state_horizon = self.model.shape_meta["obs"]["state"]["horizon"]
        self.state_dim = self.model.shape_meta["obs"]["state"]["shape"][0]
        self.image_horizon = self.model.shape_meta["obs"]["rgb"]["horizon"]
        self.image_stride = self.model.shape_meta["obs"]["rgb"].get("stride", 1)
        self.history_pad_mode = self.model.shape_meta.get("history_pad_mode", "repeat")
        data_cfg = model_cfg.data
        self.video_base_fps = float(getattr(data_cfg, "video_base_fps", 30.0))
        target_image_size = getattr(data_cfg, "target_image_size", None)
        self.target_image_size = tuple(target_image_size) if target_image_size else None
        self.ar_max_new_tokens = ar_max_new_tokens or self.action_horizon
        self.ar_temperature = ar_temperature
        self.ar_cfg = ar_cfg

        self.metadata = {
            "mode": mode,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
        }
        self._model_compiled = False
        self._attention_recording = attention_recording
        self._middle_layer_range = (
            compute_middle_layer_range(self.model.flow_expert.num_layers)
            if attention_recording else None
        )
        self._last_attention_grid = None
        self.compile_cfg = None
        if compile is not None:
            self.compile_cfg = (
                OmegaConf.to_container(compile, resolve=True)
                if OmegaConf.is_config(compile)
                else compile
            )
        self.maybe_compile_model()

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

        if OmegaConf.is_config(text_kwargs):
            text_kwargs = OmegaConf.to_container(text_kwargs, resolve=False)
            processor_call_kwargs["text_kwargs"] = text_kwargs
        elif not isinstance(text_kwargs, dict):
            text_kwargs = dict(text_kwargs)
            processor_call_kwargs["text_kwargs"] = text_kwargs

        text_kwargs["padding"] = tokenizer_padding
        if max_length is not None:
            text_kwargs["max_length"] = max_length

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
            assert current > 0, "Cannot repeat empty history"
            padding = np.repeat(data[:1], pad_count, axis=0)
            return np.concatenate([padding, data], axis=0), horizon

        # truncate: return as-is, valid count is actual length
        return data, current

    def prepare_process(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert one runtime observation into the collated batch ready for model input.

        Input contract:
        - `obs["image"]`: stacked visual history, used as one Qwen video sample.
        - `obs["states"]`: shaped `[T, D]` before padding/truncation.
        - `obs["intrinsic"]`: flattened `[fx, fy, cx, cy]` or 3x3 matrix.
        - `obs["instruction"]`: optional when `default_instruction` is configured.
        - `obs["prev_action_chunk"]`: optional RTC action prefix from previous step.
        """
        instruction = obs.get("instruction") or self.default_instruction
        if instruction is None:
            raise ValueError("Inference requires an instruction.")

        images, _ = self.prepare_history(np.asarray(obs["image"]), self.image_horizon)
        intrinsic = self.extract_intrinsic(obs["intrinsic"])
        if self.target_image_size is not None:
            images, _, intrinsic = process_image(images, intrinsic=intrinsic, target_size=self.target_image_size)

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
            "intrinsic": torch.from_numpy(intrinsic),
            "vision_type": "video",
            "video_fps": torch.tensor(self.video_base_fps / self.image_stride, dtype=torch.float32),
            "states": torch.from_numpy(states),
            "n_states": torch.tensor(n_states, dtype=torch.long),
            "n_actions": torch.tensor(self.action_horizon, dtype=torch.long),
            "is_vla_data": torch.tensor(True, dtype=torch.bool),
        }
        batch = self.data_collator([sample])

        # Drop labels — only used for VLM text supervision during training.
        batch.pop("labels", None)

        # Normalize executed action prefix for RTC condition (same space as model actions).
        prev_action_chunk = obs.get("prev_action_chunk")
        if prev_action_chunk is not None:
            prev_action_chunk = np.asarray(prev_action_chunk, dtype=np.float32)
            if self.normalizer is not None:
                key = "actions" if self.use_relative_action else "motions"
                prev_action_chunk = self.normalizer[key](prev_action_chunk)
            prev_action_chunk = torch.as_tensor(prev_action_chunk)
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
            batch["prev_action_chunk"] = prev_action_chunk
            batch["inference_delay"] = inference_delay

        # Cast floating-point tensors to model dtype.
        for key, value in batch.items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                batch[key] = value.to(self.dtype)

        return batch

    def extract_attention_grid(
        self, batch: Dict[str, torch.Tensor], expert_attn: list,
    ) -> np.ndarray:
        """Aggregate expert attention into a spatial grid [T_g, tH, tW].

        Uses the stack -> reshape -> aggregate -> renormalize pipeline from
        src.utils.visual_attention. Returns a numpy float32 array.
        """
        input_ids = batch["input_ids"][0]
        visual_indices = (input_ids == self.model.backbone.video_token_id).nonzero(as_tuple=True)[0].cpu()
        T_g, H_g, W_g = (int(v) for v in batch["video_grid_thw"][0])

        vis_flat = stack_visual_attention(expert_attn, visual_indices)
        vis_grid, _, _ = reshape_visual_to_grid(vis_flat, T_g, H_g, W_g)
        agg = aggregate_visual_attention(
            vis_grid,
            strategy="middle_layers_mean_heads",
            ode_step=0,
            layer_range=self._middle_layer_range,
        )
        return renormalize_visual_subset(agg).numpy()

    def post_process(self, actions: torch.Tensor) -> torch.Tensor:
        """Unnormalize predicted actions back to physical space."""
        if self.normalizer is None:
            return actions
        key = "actions" if self.use_relative_action else "motions"
        return self.normalizer[key].unnormalize(actions)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run model inference. RTC fields (prev_action_chunk, inference_delay)
        are carried inside ``inputs`` when present."""
        if self.mode == "flow":
            if self._attention_recording:
                result = self.model("infer_action", inputs, output_attentions=True)
                # Detach and move to CPU immediately to free GPU memory
                # and prevent compile graph extension.
                expert_attn = [
                    [w.detach().cpu() if w is not None else None for w in step]
                    for step in result["expert_attention_weights"]
                ]
                self._last_attention_grid = self.extract_attention_grid(inputs, expert_attn)
                return result["generated_actions"]
            return self.model("infer_action", inputs)
        return self.model(
            "infer_vla", inputs,
            max_new_tokens=self.ar_max_new_tokens,
            temperature=self.ar_temperature,
            cfg=self.ar_cfg,
        )["generated_actions"]