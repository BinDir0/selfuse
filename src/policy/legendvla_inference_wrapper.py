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

from src.utils.pytorch_util import dict_apply

log = logging.getLogger(__name__)

# Resolved once at import time
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

        # Use Hydra compose API to properly resolve defaults and interpolations
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

        self.processor = hydra.utils.instantiate(model_cfg.vla_processor)
        if hasattr(self.processor, "tokenizer_padding"):
            self.processor.tokenizer_padding = tokenizer_padding
        if max_length is not None and hasattr(self.processor, "max_seq_len"):
            self.processor.max_seq_len = max_length

        self.normalizer = self.load_normalizer(normalizer_path) if normalizer_path else None
        self.use_relative_action = use_relative_action

        self.mode = mode
        self.default_instruction = default_instruction
        self.action_horizon = int(self.model.shape_meta["action"]["horizon"])
        self.action_dim = int(self.model.shape_meta["action"]["shape"][0])
        self.state_horizon = int(self.model.shape_meta["obs"]["state"]["horizon"])
        self.state_dim = int(self.model.shape_meta["obs"]["state"]["shape"][0])
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
        """Convert raw observation into processor inputs."""
        instruction = obs.get("instruction") or self.default_instruction
        images = obs["image"]
        states = obs["states"]
        n_states = torch.full((1,), states.shape[0], dtype=torch.int32)
        intrinsic = self.extract_intrinsic(obs["intrinsic"])

        processor_mode = "infer" if self.mode == "flow" else "infer-ar"
        processed = self.processor(
            text=instruction,
            images=images,
            states=states,
            actions=np.zeros((self.action_horizon, self.action_dim), dtype=np.float32),
            intrinsic=intrinsic,
            depth_images=obs.get("depth"),
            mode=processor_mode,
        )
        processed = dict_apply(processed, lambda x: torch.from_numpy(x)[None, ...])

        if self.normalizer is not None:
            key = "states" if self.use_relative_action else "motions"
            states = self.normalizer[key](states)
        states = self.pad_states(states)
        states = torch.from_numpy(states)[None, ...]
        return {"processed": processed, "states": states, "n_states": n_states}

    def build_model_inputs(self, prepared: Dict[str, Any]) -> Dict[str, Any]:
        """Construct model input tensors from preprocessed data."""
        processed = prepared["processed"]
        input_ids = processed["input_ids"]
        batch_size = input_ids.shape[0]

        inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": processed["attention_mask"],
            "pixel_values": processed["pixel_values"].to(self.dtype),
            "image_grid_thw": processed["image_grid_thw"],
            "mm_token_type_ids": processed["mm_token_type_ids"],
            "states": prepared["states"].to(self.dtype),
            "n_states": prepared["n_states"],
            "is_vla_data": torch.ones(batch_size, dtype=torch.bool),
        }

        if self.mode == "flow":
            inputs["n_actions"] = torch.full((batch_size,), self.action_horizon, dtype=torch.long)
            inputs["answer_start_idx"] = processed["answer_start_idx"]
            inputs["actions"] = torch.zeros(
                batch_size, self.action_horizon, self.action_dim, dtype=self.dtype,
            )
            inputs["actions_valid_mask"] = torch.ones(
                batch_size, self.action_horizon, self.action_dim, dtype=torch.bool,
            )

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
        else:
            return self.model(
                "infer_vla", inputs,
                max_new_tokens=self.ar_max_new_tokens,
                temperature=self.ar_temperature,
                cfg=self.ar_cfg,
            )["generated_actions"]
