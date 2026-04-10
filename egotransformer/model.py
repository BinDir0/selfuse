"""EgoHandSTModel：DINOv3 、ST-Transformer、双手 cross-attn、存在性与  MANO 头。"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn

from dataloader.lowdim_to_mano import rot6d_to_rotmat, rotmat_to_axis_angle

from .backbone import DinoVisionBackbone
from .st_transformer import SpatioTemporalTransformer

_MANO_NHANDJOINTS = 15
_ROOT_ROT6D_DIM = 6


@dataclass
class EgoHandSTConfig:
    backbone_name: str = "vit_base_patch16_dinov3.lvd1689m"
    pretrained_backbone: bool = True
    freeze_backbone: bool = True
    use_cls_token: bool = False
    image_size: int = 384

    st_depth: int = 4
    st_num_heads: int = 12
    st_mlp_ratio: float = 4.0
    st_dropout: float = 0.0
    max_temporal_length: int = 64

    use_hand_query_cross_attn: bool = True
    cross_attn_num_heads: int = 8
    cross_attn_dropout: float = 0.0

    # Reduce left/right attention collapse: slot embedding + optional per-patch mem bias (two MHA passes)
    use_hand_side_embedding: bool = True
    use_hand_role_mem_bias: bool = True

    mano_trans_dim: int = 3
    # Global root: head regresses Zhou rot6d, then maps to 3-D axis-angle for MANO / refine.
    mano_root_rot6d_dim: int = _ROOT_ROT6D_DIM
    mano_root_orient_dim: int = 3
    # Hand pose: head regresses 15×6 Zhou rot6d, then maps to 45-D axis-angle for MANO / refine.
    mano_hand_pose_rot6d_dim: int = _MANO_NHANDJOINTS * 6
    mano_hand_pose_dim: int = _MANO_NHANDJOINTS * 3
    mano_betas_dim: int = 10
    camera_intrinsics_dim: int = 4
    camera_init_fx: float = 384.0
    camera_init_fy: float = 384.0
    camera_init_cx: float = 192.0
    camera_init_cy: float = 192.0

    # cross-attn decoder: refine hand token against per-frame ST patch tokens (HMR2-style)
    use_mano_cross_decoder: bool = True
    mano_decoder_depth: int = 2
    mano_decoder_heads: int = 8
    mano_decoder_dropout: float = 0.0
    mano_decoder_ff_mult: float = 4.0

    # temporal Transformer on stacked MANO vector (61-D), residual delta (HaWoR motion_module-ish)
    use_mano_temporal_refine: bool = True
    mano_refine_hdim: int = 512
    mano_refine_layers: int = 2
    mano_refine_heads: int = 8
    mano_refine_dropout: float = 0.0


class ManoCrossAttnDecoder(nn.Module):
    """Hand token cross-attends to spatial ST tokens per (batch, time)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        depth: int,
        dropout: float,
        ff_mult: float,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must divide num_heads {num_heads}")
        ff = int(dim * ff_mult)
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)

    def forward(self, h_bt_d: torch.Tensor, memory_bts_d: torch.Tensor) -> torch.Tensor:
        b, t, d = h_bt_d.shape
        b2, t2, s, d_m = memory_bts_d.shape
        if (b2, t2, d_m) != (b, t, d):
            raise ValueError(
                f"h {tuple(h_bt_d.shape)} vs mem {tuple(memory_bts_d.shape)}"
            )
        q = h_bt_d.reshape(b * t, 1, d)
        mem = memory_bts_d.reshape(b * t, s, d)
        out = self.decoder(q, mem)
        return out.reshape(b, t, d)


class ManoTemporalRefine(nn.Module):
    """(B,T,M) MANO vector -> same shape, x + delta(enc(proj(x)+pos))."""

    def __init__(
        self,
        mano_dim: int,
        hdim: int,
        nlayer: int,
        nhead: int,
        max_temporal_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hdim % nhead != 0:
            raise ValueError(f"mano_refine_hdim {hdim} must divide mano_refine_heads {nhead}")
        self.proj_in = nn.Linear(mano_dim, hdim)
        self.pos_emb = nn.Embedding(max_temporal_length, hdim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hdim,
            nhead=nhead,
            dim_feedforward=hdim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        enc_kw: dict = {"num_layers": nlayer}
        if "enable_nested_tensor" in inspect.signature(nn.TransformerEncoder).parameters:
            enc_kw["enable_nested_tensor"] = False
        self.encoder = nn.TransformerEncoder(enc_layer, **enc_kw)
        self.proj_out = nn.Linear(hdim, mano_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.proj_in(x)
        if t > self.pos_emb.num_embeddings:
            raise ValueError(
                f"T={t} exceeds ManoTemporalRefine pos_emb {self.pos_emb.num_embeddings}"
            )
        pos = torch.arange(t, device=x.device).view(1, t).expand(b, t)
        h = h + self.pos_emb(pos)
        h = self.encoder(h)
        return x + self.proj_out(h)


class ManoParamHeadsRoWaH(nn.Module):
    def __init__(
        self,
        dim: int,
        trans_dim: int,
        root_rot6d_dim: int,
        hand_pose_rot6d_dim: int,
        betas_dim: int,
    ) -> None:
        super().__init__()
        self.trans = nn.Linear(dim, trans_dim)
        self.root_orient = nn.Linear(dim, root_rot6d_dim)
        self.hand_pose = nn.Linear(dim, hand_pose_rot6d_dim)
        self.betas = nn.Linear(dim, betas_dim)
        nn.init.xavier_uniform_(self.root_orient.weight, gain=0.01)
        nn.init.zeros_(self.root_orient.bias)
        nn.init.xavier_uniform_(self.hand_pose.weight, gain=0.01)
        nn.init.zeros_(self.hand_pose.bias)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """h (B,T,D)；root_orient (B,T,6) 与 hand_pose (B,T,90) 为 rot6d，模外转 axis-angle。"""
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
      mano_left / mano_right: dict; trans/root_orient/hand_pose/betas = (B,T,3)/(B,T,3)/(B,T,45)/(B,T,10).
        root_orient & hand_pose: head outputs rot6d then axis-angle for MANO.
      mano_concat: (B, T, 2, K), K = 61 = 3+3+45+10
      per_frame_tokens: (B, T, S, D)
      hand_features: (B, T, 2, D)  # after cross-decoder if enabled, else raw hand tokens
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
        c = self.config
        self.hand_side_embed = (
            nn.Embedding(2, dim) if c.use_hand_side_embedding else None
        )
        if self.hand_side_embed is not None:
            nn.init.trunc_normal_(self.hand_side_embed.weight, std=0.02)
        if c.use_hand_role_mem_bias:
            self.left_hand_mem_bias = nn.Parameter(torch.zeros(num_patches, dim))
            self.right_hand_mem_bias = nn.Parameter(torch.zeros(num_patches, dim))
            nn.init.trunc_normal_(self.left_hand_mem_bias, std=0.02)
            nn.init.trunc_normal_(self.right_hand_mem_bias, std=0.02)
        else:
            self.register_parameter("left_hand_mem_bias", None)
            self.register_parameter("right_hand_mem_bias", None)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=c.cross_attn_num_heads,
            dropout=c.cross_attn_dropout,
            batch_first=True,
        )

        self.hand_norm = nn.LayerNorm(dim)
        self.existence_head_left = nn.Linear(dim, 1)
        self.existence_head_right = nn.Linear(dim, 1)

        self.mano_heads_left = ManoParamHeadsRoWaH(
            dim,
            c.mano_trans_dim,
            c.mano_root_rot6d_dim,
            c.mano_hand_pose_rot6d_dim,
            c.mano_betas_dim,
        )
        self.mano_heads_right = ManoParamHeadsRoWaH(
            dim,
            c.mano_trans_dim,
            c.mano_root_rot6d_dim,
            c.mano_hand_pose_rot6d_dim,
            c.mano_betas_dim,
        )
        self.cam_head_left = nn.Linear(dim * 2, c.camera_intrinsics_dim)
        self.cam_head_right = nn.Linear(dim * 2, c.camera_intrinsics_dim)
        nn.init.zeros_(self.cam_head_left.weight)
        nn.init.zeros_(self.cam_head_right.weight)
        with torch.no_grad():
            self.cam_head_left.bias.zero_()
            self.cam_head_right.bias.zero_()
            self.cam_head_left.bias[0] = c.camera_init_fx
            self.cam_head_left.bias[1] = c.camera_init_fy
            self.cam_head_left.bias[2] = c.camera_init_cx
            self.cam_head_left.bias[3] = c.camera_init_cy
            self.cam_head_right.bias[0] = c.camera_init_fx
            self.cam_head_right.bias[1] = c.camera_init_fy
            self.cam_head_right.bias[2] = c.camera_init_cx
            self.cam_head_right.bias[3] = c.camera_init_cy

        self._mano_vec_dim = (
            c.mano_trans_dim
            + c.mano_root_orient_dim
            + c.mano_hand_pose_dim
            + c.mano_betas_dim
        )

        if c.use_mano_cross_decoder:
            if dim % c.mano_decoder_heads != 0:
                raise ValueError(
                    f"embed_dim {dim} must divide mano_decoder_heads {c.mano_decoder_heads}"
                )
            dec_kw = dict(
                dim=dim,
                num_heads=c.mano_decoder_heads,
                depth=c.mano_decoder_depth,
                dropout=c.mano_decoder_dropout,
                ff_mult=c.mano_decoder_ff_mult,
            )
            self.mano_cross_dec_left = ManoCrossAttnDecoder(**dec_kw)
            self.mano_cross_dec_right = ManoCrossAttnDecoder(**dec_kw)
        else:
            self.mano_cross_dec_left = None
            self.mano_cross_dec_right = None

        if c.use_mano_temporal_refine:
            if c.mano_refine_hdim % c.mano_refine_heads != 0:
                raise ValueError(
                    f"mano_refine_hdim {c.mano_refine_hdim} must divide "
                    f"mano_refine_heads {c.mano_refine_heads}"
                )
            ref_kw = dict(
                mano_dim=self._mano_vec_dim,
                hdim=c.mano_refine_hdim,
                nlayer=c.mano_refine_layers,
                nhead=c.mano_refine_heads,
                max_temporal_length=c.max_temporal_length,
                dropout=c.mano_refine_dropout,
            )
            self.mano_temporal_refine_left = ManoTemporalRefine(**ref_kw)
            self.mano_temporal_refine_right = ManoTemporalRefine(**ref_kw)
        else:
            self.mano_temporal_refine_left = None
            self.mano_temporal_refine_right = None

    @staticmethod
    def _root_rot6d_to_axis_angle(root_rot6d: torch.Tensor) -> torch.Tensor:
        """(B,T,6) rot6d -> (B,T,3) axis-angle."""
        if root_rot6d.shape[-1] != _ROOT_ROT6D_DIM:
            raise ValueError(
                f"root rot6d last dim must be {_ROOT_ROT6D_DIM}, got {root_rot6d.shape[-1]}"
            )
        x = root_rot6d.unsqueeze(-2)
        R = rot6d_to_rotmat(x)
        aa = rotmat_to_axis_angle(R)
        return aa.squeeze(-2)

    @staticmethod
    def _hand_pose_rot6d_to_axis_angle(hand_pose_rot6d: torch.Tensor) -> torch.Tensor:
        """(B,T,90) rot6d -> (B,T,45) axis-angle (15 joints)."""
        if hand_pose_rot6d.shape[-1] != _MANO_NHANDJOINTS * 6:
            raise ValueError(
                f"hand_pose rot6d last dim must be {_MANO_NHANDJOINTS * 6}, got {hand_pose_rot6d.shape[-1]}"
            )
        x = hand_pose_rot6d.reshape(
            *hand_pose_rot6d.shape[:-1], _MANO_NHANDJOINTS, 6
        )
        R = rot6d_to_rotmat(x)
        aa = rotmat_to_axis_angle(R)
        return aa.reshape(*hand_pose_rot6d.shape[:-1], _MANO_NHANDJOINTS * 3)

    def _dict_mano_rot6d_to_axis_angle(self, d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            **d,
            "root_orient": self._root_rot6d_to_axis_angle(d["root_orient"]),
            "hand_pose": self._hand_pose_rot6d_to_axis_angle(d["hand_pose"]),
        }

    def _hand_tokens_from_st(
        self, st_out: torch.Tensor, *, batch: int, time_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, s, d = st_out.shape
        assert b == batch and t == time_len
        if self.config.use_hand_query_cross_attn:
            mem = st_out.reshape(b * t, s, d)
            bt = b * t
            q_slot = self.hand_queries
            if self.hand_side_embed is not None:
                q_slot = q_slot + self.hand_side_embed.weight

            if self.left_hand_mem_bias is not None:
                mem_l = mem + self.left_hand_mem_bias.unsqueeze(0)
                mem_r = mem + self.right_hand_mem_bias.unsqueeze(0)
                ql = q_slot[0:1].expand(bt, 1, -1)
                qr = q_slot[1:2].expand(bt, 1, -1)
                out_l, _ = self.cross_attn(ql, mem_l, mem_l, need_weights=False)
                out_r, _ = self.cross_attn(qr, mem_r, mem_r, need_weights=False)
                out = torch.cat([out_l, out_r], dim=1)
            else:
                q = q_slot.unsqueeze(0).expand(bt, -1, -1)
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

    def _split_mano_vec(self, v: torch.Tensor) -> dict[str, torch.Tensor]:
        c = self.config
        i = 0
        trans = v[..., i : i + c.mano_trans_dim]
        i += c.mano_trans_dim
        root = v[..., i : i + c.mano_root_orient_dim]
        i += c.mano_root_orient_dim
        pose = v[..., i : i + c.mano_hand_pose_dim]
        i += c.mano_hand_pose_dim
        betas = v[..., i : i + c.mano_betas_dim]
        return {
            "trans": trans,
            "root_orient": root,
            "hand_pose": pose,
            "betas": betas,
        }

    @staticmethod
    def _sanitize_intrinsics(pred_cam: torch.Tensor) -> torch.Tensor:
        """Force positive focal lengths while keeping principal point unconstrained."""
        if pred_cam.shape[-1] != 4:
            raise ValueError(f"expected pred_cam (...,4) for fx,fy,cx,cy, got {tuple(pred_cam.shape)}")
        fx = torch.clamp_min(pred_cam[..., 0:1], 1e-3)
        fy = torch.clamp_min(pred_cam[..., 1:2], 1e-3)
        cx = pred_cam[..., 2:3]
        cy = pred_cam[..., 3:4]
        return torch.cat([fx, fy, cx, cy], dim=-1)

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        if video.dim() != 5:
            raise ValueError(f"Expected video (B,T,3,H,W), got {tuple(video.shape)}")
        b, t, c, h, w = video.shape
        if c != 3:
            raise ValueError(f"Expected 3 input channels, got {c}")

        x = video.reshape(b * t, c, h, w)
        patches, cls_bt = self.backbone.forward_patches_and_cls(x)
        _, s, d = patches.shape
        feats = patches.reshape(b, t, s, d)
        cls_btd = cls_bt.reshape(b, t, d)

        st_out = self.st(feats)
        h_left, h_right = self._hand_tokens_from_st(st_out, batch=b, time_len=t)

        logit_l = self.existence_head_left(h_left)
        logit_r = self.existence_head_right(h_right)
        existence_logits = torch.cat([logit_l, logit_r], dim=-1)

        hl = h_left
        hr = h_right
        if self.mano_cross_dec_left is not None:
            hl = self.mano_cross_dec_left(h_left, st_out)
            hr = self.mano_cross_dec_right(h_right, st_out)

        mano_left = self._dict_mano_rot6d_to_axis_angle(self.mano_heads_left(hl))
        mano_right = self._dict_mano_rot6d_to_axis_angle(self.mano_heads_right(hr))
        if self.mano_temporal_refine_left is not None:
            assert self.mano_temporal_refine_right is not None
            v_l = self._concat_mano(mano_left)
            v_r = self._concat_mano(mano_right)
            mano_left = self._split_mano_vec(self.mano_temporal_refine_left(v_l))
            mano_right = self._split_mano_vec(self.mano_temporal_refine_right(v_r))

        mano_concat = torch.stack(
            [self._concat_mano(mano_left), self._concat_mano(mano_right)], dim=2
        )
        cam_in_left = torch.cat([hl, cls_btd], dim=-1)
        cam_in_right = torch.cat([hr, cls_btd], dim=-1)
        cam_left = self._sanitize_intrinsics(self.cam_head_left(cam_in_left))
        cam_right = self._sanitize_intrinsics(self.cam_head_right(cam_in_right))
        pred_cam = torch.stack([cam_left, cam_right], dim=2)

        return {
            "hand_existence_logits": existence_logits,
            "mano_left": mano_left,
            "mano_right": mano_right,
            "mano_concat": mano_concat,
            "pred_cam": pred_cam,
            "per_frame_tokens": st_out,
            "hand_features": torch.stack([hl, hr], dim=2),
        }

    @torch.no_grad()
    def predict_proba_hands(self, video: torch.Tensor) -> torch.Tensor:
        """video (B,T,3,H,W) -> (B,T,2) sigmoid probabilities."""
        logits = self.forward(video)["hand_existence_logits"]
        return torch.sigmoid(logits)
