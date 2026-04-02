"""Frozen target encoder for future frame dense feature supervision.

Provides per-patch merged features as prediction targets for the world model
DiffLoss. Currently only the "self_vit" backend is supported:

  - "self_vit" with use_ema=True: EMA copy of the backbone Qwen3-VL ViT +
    merger. Zero extra model parameters; EMA updated once per optimizer step.
  - "self_vit" with use_ema=False (freeze): frozen snapshot of the backbone
    at init time. No EMA updates; parameters stay at initial checkpoint values.

For self_vit mode:
  - When use_ema=True, call init_ema() after backbone construction.
  - When use_ema=False, call init_frozen() after backbone construction.
  - forward receives preprocessed pixel_values (same format as backbone ViT
    input, produced by the HF Qwen3-VL processor) and grid_thw.
  - Returns flat (total_merged_tokens, feature_dim) per-patch features.

EMA implementation uses torch.optim.swa_utils.AveragedModel, consistent with
the existing ModelAveraging class in src/model/common/model_average.py.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from src.model.common.model_average import _unshard_params


class FutureFrameTargetEncoder(nn.Module):

    def __init__(
        self,
        encoder_type: str,
        feature_dim: int,
        use_ema: bool = True,
        model_name_or_path: str = "",
    ):
        super().__init__()
        assert encoder_type == "self_vit", (
            f"Only 'self_vit' encoder is supported, got '{encoder_type}'"
        )
        self.encoder_type = encoder_type
        self.feature_dim = feature_dim
        self.use_ema = use_ema
        self.model_name_or_path = model_name_or_path
        # Lazily initialized via init_ema() or init_frozen().
        self.ema: AveragedModel | None = None
        self.frozen_visual: nn.Module | None = None

    # -- EMA lifecycle (use_ema=True) --

    def init_ema(self, source_visual: nn.Module, momentum: float = 0.996) -> None:
        """Deep-copy the backbone visual module as the EMA target encoder.

        Uses torch.optim.swa_utils.AveragedModel with EMA averaging function,
        consistent with src/model/common/model_average.py.
        """
        assert self.use_ema, "init_ema() requires use_ema=True"
        # Source: torch.optim.swa_utils.AveragedModel
        self.ema = AveragedModel(
            source_visual,
            multi_avg_fn=get_ema_multi_avg_fn(momentum),
        )
        # AveragedModel registers n_averaged as a persistent torch.long buffer.
        # FSDP2's fully_shard() does not convert integer buffers to DTensor,
        # but accelerate's fsdp2_load_full_state_dict assumes every state_dict
        # entry has .device_mesh. Re-register as non-persistent so it is
        # excluded from state_dict() and avoids the crash.
        if hasattr(self.ema, 'n_averaged'):
            n_avg = self.ema.n_averaged.clone()
            del self.ema._buffers['n_averaged']
            self.ema.register_buffer('n_averaged', n_avg, persistent=False)
        self.ema.requires_grad_(False)

    @torch.no_grad()
    def update_ema(self, source_visual: nn.Module) -> None:
        if self.ema is not None:
            # Under FSDP2 both the live backbone and the EMA copy have DTensor
            # parameters after accelerator.prepare(). AveragedModel.update_parameters
            # does in-place mul_/add_ across both param sets, so both must be
            # temporarily unsharded to avoid Tensor/DTensor copy_ mismatches.
            with _unshard_params(source_visual), _unshard_params(self.ema.module):
                self.ema.update_parameters(source_visual)

    # -- Frozen lifecycle (use_ema=False) --

    def init_frozen(self, source_visual: nn.Module) -> None:
        """Deep-copy the backbone visual module as a frozen target encoder.

        No EMA updates; the copy stays at the initial checkpoint values.
        """
        assert not self.use_ema, "init_frozen() requires use_ema=False"
        self.frozen_visual = copy.deepcopy(source_visual)
        self.frozen_visual.requires_grad_(False)

    # -- Forward --

    def visual_module(self) -> nn.Module:
        """Return the underlying visual module (EMA or frozen)."""
        if self.use_ema:
            assert self.ema is not None, "Call init_ema() before forward."
            return self.ema.module
        assert self.frozen_visual is not None, "Call init_frozen() before forward."
        return self.frozen_visual

    def set_mem_grid_thw(self, grid_thw: torch.Tensor) -> None:
        """Set grid_thw on MEM temporal attention blocks inside the target ViT.

        Must be called before forward so that MEM temporal causal attention
        knows the per-entry (T, H, W) structure of the combined
        observation + future frame sequence.
        """
        try:
            from src.model.vision.temporal_attention import MEMVisionBlock
        except ImportError:
            return
        for block in self.visual_module().blocks:
            if isinstance(block, MEMVisionBlock):
                block.temporal_attn.current_grid_thw = grid_thw

    @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode future frames into per-patch merged features.

        Args:
            pixel_values: preprocessed pixels, same flat format as backbone ViT
                input (total_3d_patches, C * temporal_patch_size * patch_size^2).
            grid_thw: (num_entries, 3) temporal / height / width per entry.

        Returns:
            features: (total_merged_tokens, feature_dim) detached.
            grid_thw: passthrough for downstream token-count bookkeeping.
        """
        self.set_mem_grid_thw(grid_thw)
        visual = self.visual_module()
        # Source: huggingface/transformers, Qwen3VLVisionModel.forward
        # Returns pooler_output as flat (total_merged_tokens, out_hidden_size)
        # after the spatial merger.
        output = visual(pixel_values, grid_thw=grid_thw, return_dict=True)
        return output.pooler_output.detach(), grid_thw
