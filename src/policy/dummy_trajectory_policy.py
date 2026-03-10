import logging
import pathlib
from typing import Any, Dict

import numpy as np
import torch
from torch import nn


log = logging.getLogger(__name__)


class DummyTrajectoryPolicy(nn.Module):
    """Temporary dummy policy for websocket replay testing.

    It replays precomputed dataset action chunks instead of running model inference.
    Chunks follow the same websocket response format as the real policy:
    RuntimeEngine -> `pred_actions: [H, D]`.
    """

    def __init__(
        self,
        model_config_path: str,
        dataset_index: int = 0,
        episode_index: int = 0,
        start_t: int = 0,
        step_hop: int = 6,
        loop: bool = True,
        mode: str = "dummy",
        use_mixed_precision: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.model_config_path = pathlib.Path(model_config_path).expanduser()
        self.dataset_index = int(dataset_index)
        self.episode_index = int(episode_index)
        self.start_t = int(start_t)
        self.step_hop = int(step_hop)
        self.loop = bool(loop)
        self.mode = mode
        self.dtype = torch.bfloat16 if use_mixed_precision else torch.float32
        self._cursor = 0
        self._model_compiled = False

        from omegaconf import OmegaConf

        OmegaConf.register_new_resolver("eval", eval, replace=True)
        cfg = OmegaConf.load(self.model_config_path)
        OmegaConf.resolve(cfg)
        self.shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
        self.use_relative_action = bool(
            OmegaConf.select(cfg, "dataset.vla_dataset.use_relative_action", default=False)
        )
        self.action_horizon = int(self.shape_meta["action"]["horizon"])
        self.action_dim = int(self.shape_meta["action"]["shape"][0])

        import hydra

        dataset = hydra.utils.instantiate(cfg.dataset.vla_dataset)
        chunks, anchors = self._build_action_chunks(dataset)
        if not chunks:
            raise ValueError("No valid action chunks were built for dummy trajectory policy")

        self.register_buffer(
            "action_chunks",
            torch.from_numpy(np.stack(chunks, axis=0)).to(torch.float32),
            persistent=False,
        )
        self.anchor_steps = anchors
        self.metadata = {
            "mode": mode,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "source": "dummy_dataset_trajectory",
            "dataset_index": self.dataset_index,
            "episode_index": self.episode_index,
            "start_t": self.start_t,
            "step_hop": self.step_hop,
            "num_chunks": len(chunks),
        }
        log.info(
            "Loaded %d dummy action chunks from %s (dataset=%d, episode=%d, start_t=%d, hop=%d)",
            len(chunks),
            self.model_config_path,
            self.dataset_index,
            self.episode_index,
            self.start_t,
            self.step_hop,
        )

    def _build_action_chunks(self, dataset) -> tuple[list[np.ndarray], list[int]]:
        from src.dataset.data_transforms import process_state_action
        from src.dataset.sampler import SequenceSampler

        if self.dataset_index < 0 or self.dataset_index >= len(dataset.replay_buffers):
            raise IndexError(
                f"dataset_index={self.dataset_index} is out of range for {len(dataset.replay_buffers)} datasets"
            )

        replay_buffer = dataset.replay_buffers[self.dataset_index]
        if self.episode_index < 0 or self.episode_index >= replay_buffer.n_episodes:
            raise IndexError(
                f"episode_index={self.episode_index} is out of range for {replay_buffer.n_episodes} episodes"
            )

        episode_mask = np.zeros(replay_buffer.n_episodes, dtype=bool)
        episode_mask[self.episode_index] = True
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            episode_mask=episode_mask,
            **dataset.sampler_cfg,
        )

        episode_ends = replay_buffer.episode_ends[:]
        episode_start = 0 if self.episode_index == 0 else int(episode_ends[self.episode_index - 1])
        episode_end = int(episode_ends[self.episode_index])
        action_steps, action_stride, _ = dataset.sampler_cfg["action"]
        full_span = (action_steps - 1) * action_stride
        max_anchor = episode_end - 1 - full_span
        anchor_start = episode_start + self.start_t
        if anchor_start > max_anchor:
            raise ValueError(
                f"start_t={self.start_t} leaves no full action chunk in episode of length {episode_end - episode_start}"
            )

        chunks: list[np.ndarray] = []
        anchors: list[int] = []
        for anchor in range(anchor_start, max_anchor + 1, self.step_hop):
            sampler_index = anchor - episode_start
            sample = sampler.sample_sequence(sampler_index)
            _, action = process_state_action(
                wrist_state=sample["wrist_state"].astype(np.float32),
                hand_state=sample["hand_state"].astype(np.float32),
                wrist_action=sample["wrist_action"].astype(np.float32),
                hand_action=sample["hand_action"].astype(np.float32),
                extrinsic=sample["extrinsic"].astype(np.float32).reshape(4, 4),
                hand_ndim=dataset.hand_ndim,
                motion_type=dataset.motion_type,
                use_relative_action=self.use_relative_action,
                normalizer=None,
            )
            if action.shape != (self.action_horizon, self.action_dim):
                raise ValueError(
                    f"Expected full action chunk shape {(self.action_horizon, self.action_dim)}, got {action.shape} at anchor {anchor}"
                )
            chunks.append(action.astype(np.float32))
            anchors.append(anchor)
        return chunks, anchors

    def maybe_compile_model(self) -> None:
        return None

    def prepare_process(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return {"obs": obs}

    def build_model_inputs(self, prepared: Dict[str, Any]) -> Dict[str, Any]:
        return prepared

    def post_process(self, actions: torch.Tensor) -> torch.Tensor:
        return actions

    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        del inputs
        chunk = self.action_chunks[self._cursor].unsqueeze(0)
        if self.loop:
            self._cursor = (self._cursor + 1) % len(self.action_chunks)
        else:
            self._cursor = min(self._cursor + 1, len(self.action_chunks) - 1)
        return chunk
