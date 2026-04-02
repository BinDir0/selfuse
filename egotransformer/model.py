"""EgoHandSTModel：DINOv3 骨干、ST-Transformer、双手 cross-attn、存在性与 RoWaH 风格 MANO 头。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .backbone import DinoVisionBackbone
from .st_transformer import SpatioTemporalTransformer


@dataclass
class EgoHandSTConfig:
    backbone_name: str = "vit_base_patch16_dinov3.lvd1689m"
    pretrained_backbone: bool = True
    freeze_backbone: bool = True
    use_cls_token: bool = False
    image_size: int = 224

    st_depth: int = 4
    st_num_heads: int = 12
    st_mlp_ratio: float = 4.0
    st_dropout: float = 0.0
    max_temporal_length: int = 64

    use_hand_query_cross_attn: bool = True
    cross_attn_num_heads: int = 8
    cross_attn_dropout: float = 0.0

    mano_trans_dim: int = 3
    mano_root_orient_dim: int = 3
    mano_hand_pose_dim: int = 45
    mano_betas_dim: int = 10


class ManoParamHeadsRoWaH(nn.Module):
    def __init__(
        self,
        dim: int,
        trans_dim: int,
        root_dim: int,
        pose_dim: int,
        betas_dim: int,
    ) -> None:
        super().__init__()
        self.trans = nn.Linear(dim, trans_dim)
        self.root_orient = nn.Linear(dim, root_dim)
        self.hand_pose = nn.Linear(dim, pose_dim)
        self.betas = nn.Linear(dim, betas_dim)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """h (B,T,D)；键 trans/root_orient/hand_pose/betas 为 (B,T,3)/(B,T,3)/(B,T,45)/(B,T,10)。"""
        return {
            "trans": self.trans(h),
            "root_orient": self.root_orient(h),
            "hand_pose": self.hand_pose(h),
            "betas": self.betas(h),
        }


class EgoHandSTModel(nn.Module):
    """
    I/O:
      video: (B, T, 3, H, W)
      hand_existence_logits: (B, T, 2)  # [...,0] left, [...,1] right
      mano_left / mano_right: dict, each value (B, T, d) with d in {3,3,45,10}
      mano_concat: (B, T, 2, K), K = 3+3+45+10
      per_frame_tokens: (B, T, S, D)
      hand_features: (B, T, 2, D)
    ViT-B/16 @224: S=196, D=768（以 num_patch_tokens 为准）。
    """

    def __init__(self, config: EgoHandSTConfig | None = None) -> None:
        super().__init__()
        self.config = config or EgoHandSTConfig()

        self.backbone = DinoVisionBackbone(
            model_name=self.config.backbone_name,
            pretrained=self.config.pretrained_backbone,
            use_cls_token=self.config.use_cls_token,
            freeze=self.config.freeze_backbone,
        )
        dim = self.backbone.embed_dim
        img = self.config.image_size
        num_patches = self.backbone.num_patch_tokens(img, img)

        if dim % self.config.cross_attn_num_heads != 0:
            raise ValueError(
                f"embed_dim {dim} 必须能整除 cross_attn_num_heads "
                f"{self.config.cross_attn_num_heads}"
            )

        self.st = SpatioTemporalTransformer(
            dim=dim,
            depth=self.config.st_depth,
            num_heads=self.config.st_num_heads,
            num_spatial_tokens=num_patches,
            max_temporal_length=self.config.max_temporal_length,
            mlp_ratio=self.config.st_mlp_ratio,
            dropout=self.config.st_dropout,
        )

        self.hand_queries = nn.Parameter(torch.empty(2, dim))
        nn.init.trunc_normal_(self.hand_queries, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=self.config.cross_attn_num_heads,
            dropout=self.config.cross_attn_dropout,
            batch_first=True,
        )

        self.hand_norm = nn.LayerNorm(dim)
        self.existence_head_left = nn.Linear(dim, 1)
        self.existence_head_right = nn.Linear(dim, 1)

        c = self.config
        self.mano_heads_left = ManoParamHeadsRoWaH(
            dim,
            c.mano_trans_dim,
            c.mano_root_orient_dim,
            c.mano_hand_pose_dim,
            c.mano_betas_dim,
        )
        self.mano_heads_right = ManoParamHeadsRoWaH(
            dim,
            c.mano_trans_dim,
            c.mano_root_orient_dim,
            c.mano_hand_pose_dim,
            c.mano_betas_dim,
        )

    def _hand_tokens_from_st(
        self, st_out: torch.Tensor, *, batch: int, time_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, s, d = st_out.shape
        assert b == batch and t == time_len
        if self.config.use_hand_query_cross_attn:
            mem = st_out.reshape(b * t, s, d)
            q = self.hand_queries.unsqueeze(0).expand(b * t, -1, -1)
            out, _ = self.cross_attn(q, mem, mem, need_weights=False)
            h = out.view(b, t, 2, d)
            h_l = self.hand_norm(h[:, :, 0])
            h_r = self.hand_norm(h[:, :, 1])
        else:
            pooled = self.hand_norm(st_out.mean(dim=2))
            h_l, h_r = pooled, pooled
        return h_l, h_r

    @staticmethod
    def _concat_mano(d: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [d["trans"], d["root_orient"], d["hand_pose"], d["betas"]], dim=-1
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        if video.dim() != 5:
            raise ValueError(f"Expected video (B,T,3,H,W), got {tuple(video.shape)}")
        b, t, c, h, w = video.shape
        if c != 3:
            raise ValueError(f"Expected 3 input channels, got {c}")

        x = video.reshape(b * t, c, h, w)
        tokens = self.backbone(x)
        _, s, d = tokens.shape
        feats = tokens.reshape(b, t, s, d)

        st_out = self.st(feats)
        h_left, h_right = self._hand_tokens_from_st(st_out, batch=b, time_len=t)

        logit_l = self.existence_head_left(h_left)
        logit_r = self.existence_head_right(h_right)
        existence_logits = torch.cat([logit_l, logit_r], dim=-1)

        mano_left = self.mano_heads_left(h_left)
        mano_right = self.mano_heads_right(h_right)
        mano_concat = torch.stack(
            [self._concat_mano(mano_left), self._concat_mano(mano_right)], dim=2
        )

        return {
            "hand_existence_logits": existence_logits,
            "mano_left": mano_left,
            "mano_right": mano_right,
            "mano_concat": mano_concat,
            "per_frame_tokens": st_out,
            "hand_features": torch.stack([h_left, h_right], dim=2),
        }

    @torch.no_grad()
    def predict_proba_hands(self, video: torch.Tensor) -> torch.Tensor:
        """video (B,T,3,H,W) -> (B,T,2) sigmoid probabilities."""
        logits = self.forward(video)["hand_existence_logits"]
        return torch.sigmoid(logits)
