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

log = logging.getLogger(__name__)

_CONFIG_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "config")


class LegendVLAInference(nn.Module):
    """Inference wrapper for LegendVLA policy serving.

    Bridges raw observations from the robot environment to model forward pass,
    implementing the interface expected by RuntimeEngine:
        prepare_process(obs) -> build_model_inputs(prepared) -> forward(inputs) -> post_process(actions)
    """

    def __init__(
        self,
        model_config_name: str = "legendvla_qwen3_vl",
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

        model_cfg = self.compose_model_config(model_config_name)

        self.model: nn.Module = hydra.utils.instantiate(model_cfg.policy)
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        if self.dtype != torch.float32:
            self.model.to(dtype=self.dtype)
            log.info("Cast model weights to %s", self.dtype)
        self.model.eval()

        if flow_sampling_steps:
            self.model.num_inference_steps = flow_sampling_steps
            if self.model.diffloss is not None:
                self.model.diffloss.num_inference_steps = flow_sampling_steps

        collator_mode = "infer" if mode == "flow" else "infer-ar"
        self.data_collator = hydra.utils.instantiate(model_cfg.data_collator).for_mode(collator_mode)
        self.data_collator.batch_processor.processor_call_kwargs["padding"] = tokenizer_padding
        if max_length is not None:
            self.data_collator.batch_processor.processor_call_kwargs["max_length"] = max_length

        self.normalizer = self.load_normalizer(normalizer_path) if normalizer_path else None
        self.use_relative_action = use_relative_action

        self.mode = mode
        self.default_instruction = default_instruction
        self.action_horizon = int(self.model.shape_meta["action"]["horizon"])
        self.action_dim = int(self.model.shape_meta["action"]["shape"][0])
        self.state_horizon = int(self.model.shape_meta["obs"]["state"]["horizon"])
        self.state_dim = int(self.model.shape_meta["obs"]["state"]["shape"][0])
        self.image_stride = int(self.model.shape_meta["obs"]["rgb"].get("stride", 1))
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
        self.compile_kwargs = None
        if compile is not None:
            self.compile_kwargs = (
                OmegaConf.to_container(compile, resolve=True)
                if OmegaConf.is_config(compile)
                else compile
            )

    @staticmethod
    def compose_model_config(experiment_name: str) -> OmegaConf:
        """Compose a full training config using Hydra compose API.

        This properly resolves defaults, interpolations, and cross-file references,
        unlike raw OmegaConf.load() which skips defaults composition.
        """
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="train_config",
                overrides=[f"experiment={experiment_name}"],
            )
        return cfg

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
        if self.compile_kwargs is None or self._model_compiled:
            return
        log.info("Compiling model with kwargs=%s", self.compile_kwargs)
        self.model = torch.compile(self.model, **self.compile_kwargs)
        self._model_compiled = True

    def load_checkpoint(self, path: str) -> None:
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state_dict = torch.load(path, map_location="cpu")
        for key in ["model", "module", "model_state_dict"]:
            if key in state_dict:
                state_dict = state_dict[key]
                break
        self.model.load_state_dict(state_dict)
        log.info("Loaded checkpoint from %s", path)

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

    def pad_states(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2:
            raise ValueError(f"Expected states shape [T, D], got {states.shape}")
        current = states.shape[0]
        if current >= self.state_horizon:
            return states[-self.state_horizon:]
        pad_count = self.state_horizon - current
        padding = (
            np.zeros((pad_count, self.state_dim), dtype=states.dtype)
            if current == 0
            else np.repeat(states[-1:], pad_count, axis=0)
        )
        return np.concatenate([states, padding], axis=0)

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

        states = np.asarray(obs["states"], dtype=np.float32)
        if self.normalizer is not None:
            key = "states" if self.use_relative_action else "motions"
            states = self.normalizer[key](states)

        sample = {
            "images": torch.as_tensor(obs["image"]),
            "instruction": instruction,
            "intrinsic": torch.from_numpy(self.extract_intrinsic(obs["intrinsic"])),
            "vision_type": "video",
            "video_fps": torch.tensor(self.video_base_fps / self.image_stride, dtype=torch.float32),
            "states": torch.from_numpy(self.pad_states(states)),
            "n_states": torch.tensor(min(states.shape[0], self.state_horizon), dtype=torch.int32),
            "actions": torch.zeros((self.action_horizon, self.action_dim), dtype=torch.float32),
            "actions_valid_mask": torch.ones((self.action_horizon, self.action_dim), dtype=torch.bool),
            "n_actions": torch.tensor(self.action_horizon, dtype=torch.int32),
            "is_vla_data": torch.tensor(True, dtype=torch.bool),
        }
        batch = self.data_collator([sample])
        return {"batch": batch}

    def build_model_inputs(self, prepared: Dict[str, Any]) -> Dict[str, Any]:
        """Construct model input tensors from one collated sample."""
        batch = prepared["batch"]

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

        if self.mode == "flow":
            inputs["answer_start_idx"] = batch["answer_start_idx"]
            inputs["actions"] = batch["actions"].to(self.dtype)
            inputs["actions_valid_mask"] = batch["actions_valid_mask"].to(dtype=torch.bool)

        return inputs

    def post_process(self, actions: torch.Tensor) -> torch.Tensor:
        """Unnormalize predicted actions back to physical space."""
        if self.normalizer is None:
            return actions
        key = "actions" if self.use_relative_action else "motions"
        return self.normalizer[key].unnormalize(actions)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        prev_action_chunk: torch.Tensor = None,
        inference_delay: int = 0,
    ) -> torch.Tensor:
        self.maybe_compile_model()
        if self.mode == "flow":
            return self.model(
                "infer_action", inputs,
                prev_action_chunk=prev_action_chunk,
                inference_delay=inference_delay,
            )
        return self.model(
            "infer_vla", inputs,
            max_new_tokens=self.ar_max_new_tokens,
            temperature=self.ar_temperature,
            cfg=self.ar_cfg,
        )["generated_actions"]
