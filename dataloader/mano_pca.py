"""MANO 手指姿态：PCA 与轴角互转（manopth ManoLayer；读 pkl 走 mano/chumpy，版本见 requirements.txt）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import torch

__all__ = [
    "build_mano_pca_layer",
    "get_cached_mano_pca_decode_layers",
    "hand_pca_bt_to_axisang45",
    "mano_axisang45_to_pca",
    "mano_pca_to_axisang45",
    "mano_pose48_to_pca",
    "mano_pose48_to_pca_pose",
]


_DEFAULT_MANO_ROOT = str(Path(__file__).resolve().parent.parent / "data" / "mano")
_MANO_PCA_DECODE_CACHE: tuple[torch.nn.Module, torch.nn.Module] | None = None


def _resolve_manopth_import(manopth_root: str | None = None):
    if manopth_root:
        root = str(Path(manopth_root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)

    from manopth.manolayer import ManoLayer

    return ManoLayer


def build_mano_pca_layer(
    hand_side: str,
    *,
    pca_dim: int = 45,
    mano_root: str | None = None,
    manopth_root: str | None = None,
    flat_hand_mean: bool = True,
    center_idx: int = 0,
) -> torch.nn.Module:
    if hand_side not in {"left", "right"}:
        raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side!r}")
    if not (1 <= int(pca_dim) <= 45):
        raise ValueError(f"pca_dim must be in [1, 45], got {pca_dim}")

    ManoLayer = _resolve_manopth_import(manopth_root)

    if mano_root is None:
        if manopth_root is not None:
            cand = str(Path(manopth_root).expanduser().resolve() / "mano" / "model")
            if os.path.isdir(cand):
                mano_root = cand
        if mano_root is None:
            mano_root = os.environ.get("MANO_ROOT", _DEFAULT_MANO_ROOT)

    return ManoLayer(
        mano_root=mano_root,
        use_pca=True,
        ncomps=int(pca_dim),
        flat_hand_mean=flat_hand_mean,
        side=hand_side,
        center_idx=center_idx,
        root_rot_mode="axisang",
        joint_rot_mode="axisang",
        robust_rot=False,
    )


def get_cached_mano_pca_decode_layers(
    *,
    mano_root: str | None = None,
    manopth_root: str | None = None,
    pca_dim: int = 45,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """返回 (left, right) ManoLayer，供 ``hand_pca_bt_to_axisang45`` / ``mano_pca_to_axisang45`` 使用。"""
    global _MANO_PCA_DECODE_CACHE
    if _MANO_PCA_DECODE_CACHE is not None:
        return _MANO_PCA_DECODE_CACHE
    left = build_mano_pca_layer(
        "left",
        pca_dim=pca_dim,
        mano_root=mano_root,
        manopth_root=manopth_root,
        flat_hand_mean=True,
        center_idx=0,
    )
    right = build_mano_pca_layer(
        "right",
        pca_dim=pca_dim,
        mano_root=mano_root,
        manopth_root=manopth_root,
        flat_hand_mean=True,
        center_idx=0,
    )
    _MANO_PCA_DECODE_CACHE = (left, right)
    return _MANO_PCA_DECODE_CACHE


def _as_batch_pose(tensor: torch.Tensor, last_dim: int, name: str) -> Tuple[torch.Tensor, bool]:
    if tensor.ndim == 1:
        if tensor.shape[0] != last_dim:
            raise ValueError(f"{name} must have shape [{last_dim}], got {tuple(tensor.shape)}")
        return tensor.unsqueeze(0), True
    if tensor.ndim != 2 or tensor.shape[1] != last_dim:
        raise ValueError(f"{name} must have shape [B, {last_dim}], got {tuple(tensor.shape)}")
    return tensor, False


def mano_axisang45_to_pca(
    joint_axisang45: torch.Tensor,
    mano_layer: torch.nn.Module,
) -> torch.Tensor:
    if not getattr(mano_layer, "use_pca", False):
        raise ValueError("mano_layer.use_pca must be True")

    joint_axisang45, squeeze = _as_batch_pose(joint_axisang45, 45, "joint_axisang45")

    dtype = joint_axisang45.dtype
    device = joint_axisang45.device

    hands_mean = mano_layer.th_hands_mean.to(device=device, dtype=dtype)
    selected_comps = mano_layer.th_selected_comps.to(device=device, dtype=dtype)

    demeaned_pose = joint_axisang45 - hands_mean
    basis_pinv = torch.linalg.pinv(selected_comps)
    hand_pca = demeaned_pose.mm(basis_pinv)

    return hand_pca.squeeze(0) if squeeze else hand_pca


def mano_pca_to_axisang45(
    hand_pca: torch.Tensor,
    mano_layer: torch.nn.Module,
) -> torch.Tensor:
    if not getattr(mano_layer, "use_pca", False):
        raise ValueError("mano_layer.use_pca must be True")

    ncomps = int(mano_layer.ncomps)
    hand_pca, squeeze = _as_batch_pose(hand_pca, ncomps, "hand_pca")

    dtype = hand_pca.dtype
    device = hand_pca.device

    hands_mean = mano_layer.th_hands_mean.to(device=device, dtype=dtype)
    selected_comps = mano_layer.th_selected_comps.to(device=device, dtype=dtype)
    joint_axisang45 = hand_pca.mm(selected_comps) + hands_mean

    return joint_axisang45.squeeze(0) if squeeze else joint_axisang45


def hand_pca_bt_to_axisang45(
    hand_pca_bt: torch.Tensor,
    mano_layer: torch.nn.Module,
) -> torch.Tensor:
    """lowdim 手指 PCA ``(B, T, ncomps)`` -> 关节轴角 ``(B, T, 45)``。"""
    if hand_pca_bt.ndim != 3:
        raise ValueError(f"hand_pca_bt must be (B,T,K), got {tuple(hand_pca_bt.shape)}")
    b, t, k = hand_pca_bt.shape
    if k != int(mano_layer.ncomps):
        raise ValueError(f"last dim {k} != mano_layer.ncomps {mano_layer.ncomps}")
    flat = hand_pca_bt.reshape(b * t, k)
    aa = mano_pca_to_axisang45(flat, mano_layer)
    return aa.view(b, t, 45)


def mano_pose48_to_pca(
    raw_pose48: torch.Tensor,
    mano_layer: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw_pose48, squeeze = _as_batch_pose(raw_pose48, 48, "raw_pose48")

    global_rot = raw_pose48[:, :3]
    joint_axisang45 = raw_pose48[:, 3:]
    hand_pca = mano_axisang45_to_pca(joint_axisang45, mano_layer)

    if squeeze:
        return global_rot.squeeze(0), hand_pca
    return global_rot, hand_pca


def mano_pose48_to_pca_pose(
    raw_pose48: torch.Tensor,
    mano_layer: torch.nn.Module,
) -> torch.Tensor:
    global_rot, hand_pca = mano_pose48_to_pca(raw_pose48, mano_layer)
    if global_rot.ndim == 1:
        return torch.cat([global_rot, hand_pca], dim=0)
    return torch.cat([global_rot, hand_pca], dim=1)
