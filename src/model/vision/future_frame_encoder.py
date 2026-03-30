"""Frozen target encoder for future frame dense feature supervision.

Provides per-patch merged features as prediction targets for the world model
DiffLoss. Three encoder backends are planned:

  - "self_vit": EMA copy of the backbone Qwen3-VL ViT + merger.
    Zero extra model parameters; EMA updated once per optimizer step.
  - "dinov2": frozen DINOv2 ViT (not yet implemented).
  - "vae": frozen video VAE encoder (not yet implemented).

For self_vit mode:
  - EMA must be initialized after backbone construction via init_ema().
  - forward receives preprocessed pixel_values (same format as backbone ViT
    input, produced by the HF Qwen3-VL processor) and grid_thw.
  - Returns flat (total_merged_tokens, feature_dim) per-patch features.

EMA implementation uses torch.optim.swa_utils.AveragedModel, consistent with
the existing ModelAveraging class in src/model/common/model_average.py.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


class FutureFrameTargetEncoder(nn.Module):

    def __init__(
        self,
        encoder_type: str,
        feature_dim: int,
        model_name_or_path: str = "",
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.feature_dim = feature_dim
        self.model_name_or_path = model_name_or_path
        # EMA is created lazily via init_ema() after the backbone is ready.
        # Stored as a registered submodule so .to(device) propagates automatically.
        self.ema: AveragedModel | None = None

        if encoder_type == "dinov2":
            raise NotImplementedError("DINOv2 target encoder not yet implemented")
        if encoder_type == "vae":
            raise NotImplementedError("VAE target encoder not yet implemented")

    # -- EMA lifecycle (self_vit mode) --

    def init_ema(self, source_visual: nn.Module, momentum: float = 0.996) -> None:
        """Deep-copy the backbone visual module as the EMA target encoder.

        Uses torch.optim.swa_utils.AveragedModel with EMA averaging function,
        consistent with src/model/common/model_average.py.
        """
        # Source: torch.optim.swa_utils.AveragedModel
        self.ema = AveragedModel(
            source_visual,
            multi_avg_fn=get_ema_multi_avg_fn(momentum),
        )
        self.ema.requires_grad_(False)

    @torch.no_grad()
    def update_ema(self, source_visual: nn.Module) -> None:
        if self.ema is not None:
            self.ema.update_parameters(source_visual)

    # -- Forward --

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
        if self.encoder_type == "self_vit":
            return self.forward_self_vit(pixel_values, grid_thw)
        raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

    def forward_self_vit(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EMA ViT + merger -> flat per-patch features.

        Reference: Qwen3VLVisionModel.forward returns pooler_output as a flat
        (total_merged_tokens, out_hidden_size) tensor after the spatial merger.
        """
        assert self.ema is not None, "Call init_ema() before forward."
        # Source: huggingface/transformers, Qwen3VLVisionModel.forward
        output = self.ema.module(pixel_values, grid_thw=grid_thw, return_dict=True)
        return output.pooler_output.detach(), grid_thw
