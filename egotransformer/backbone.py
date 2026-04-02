"""timm ViT/DINO 骨干，输出 patch tokens（可选保留 CLS+register）。"""

from __future__ import annotations

import torch
import torch.nn as nn


class DinoVisionBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
        use_cls_token: bool = False,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        import timm

        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        self.use_cls_token = use_cls_token
        self.embed_dim = self.vit.num_features

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

    def _prefix_len_without_patches(self) -> int:
        if self.use_cls_token:
            return 0
        n_prefix = getattr(self.vit, "num_prefix_tokens", None)
        if n_prefix is not None:
            return int(n_prefix)
        reg = int(getattr(self.vit, "num_register_tokens", 0))
        return 1 + reg

    @property
    def num_patches(self) -> int:
        return int(self.vit.patch_embed.num_patches)

    @torch.no_grad()
    def num_patch_tokens(self, height: int, width: int) -> int:
        dev = next(self.vit.parameters()).device
        x = torch.zeros(1, 3, height, width, device=dev, dtype=torch.float32)
        return int(self.forward(x).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) -> (B, N, C). N: patch 数（默认无 CLS/register）；C: embed_dim。"""
        feat = self.vit.forward_features(x)
        if feat.dim() != 3:
            raise RuntimeError(f"Expected (B, L, C), got shape {tuple(feat.shape)}")
        if not self.use_cls_token:
            start = self._prefix_len_without_patches()
            if feat.shape[1] <= start:
                raise RuntimeError(
                    "No patch tokens left after dropping prefix; "
                    f"L={feat.shape[1]}, prefix={start}"
                )
            feat = feat[:, start:, :]
        return feat
