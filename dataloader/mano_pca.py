"""MANO 手指姿态：PCA 与轴角互转（manopth ManoLayer；读 pkl 走 mano/chumpy，版本见 requirements.txt）。"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

__all__ = [
    "build_mano_pca_layer",
    "get_cached_mano_pca_decode_layers",
    "hand_pca_bt_to_axisang45",
    "mano_axisang45_to_pca",
    "mano_pca_to_axisang45",
    "mano_pose48_to_pca",
    "mano_pose48_to_pca_pose",
    "mano_parameter_dict_to_joints_bt",
    "mano_parameter_dict_to_verts_bt",
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

    # manopth loads read-only numpy from pickle; torch warns but buffers are not written in-place here.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*given NumPy array is not writable.*",
            category=UserWarning,
        )
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


def mano_parameter_dict_to_joints_bt(
    trans_bt: torch.Tensor,
    root_orient_bt: torch.Tensor,
    hand_pose_aa_bt: torch.Tensor,
    betas_bt: torch.Tensor,
    mano_layer: nn.Module,
    *,
    chunk: int = 512,
) -> torch.Tensor:
    """Training-space MANO dict -> joints ``(B,T,J,3)`` in millimeters.

    Important: in this codebase ``trans_bt`` is treated as an external wrist/camera-space
    translation in meters. To make ``center_idx=0`` effective, we decode MANO without
    ``th_trans`` so manopth returns wrist-centered outputs, then add ``trans_bt`` after
    converting it to millimeters.
    """
    if trans_bt.ndim != 3 or root_orient_bt.shape[-1] != 3 or betas_bt.shape[-1] != 10:
        raise ValueError(
            f"expected trans (B,T,3), root (B,T,3), betas (B,T,10); got "
            f"{tuple(trans_bt.shape)}, {tuple(root_orient_bt.shape)}, {tuple(betas_bt.shape)}"
        )
    b, t, _ = trans_bt.shape
    n = b * t
    trans = trans_bt.reshape(n, 3)
    root = root_orient_bt.reshape(n, 3)
    hand = hand_pose_aa_bt.reshape(n, 45)
    betas = betas_bt.reshape(n, 10)
    pca = mano_axisang45_to_pca(hand, mano_layer)
    pose = torch.cat([root, pca], dim=-1)

    parts: list[torch.Tensor] = []
    for s in range(0, n, max(1, int(chunk))):
        e = min(n, s + max(1, int(chunk)))
        _, jtr = mano_layer(
            pose[s:e],
            th_betas=betas[s:e],
        )
        parts.append(jtr + trans[s:e].unsqueeze(1) * 1000.0)
    j = torch.cat(parts, dim=0)
    return j.view(b, t, j.shape[1], 3)


def mano_parameter_dict_to_verts_bt(
    trans_bt: torch.Tensor,
    root_orient_bt: torch.Tensor,
    hand_pose_aa_bt: torch.Tensor,
    betas_bt: torch.Tensor,
    mano_layer: nn.Module,
    *,
    chunk: int = 512,
) -> torch.Tensor:
    """MANO dict -> mesh vertices ``(B,T,V,3)`` in millimeters.

    Mirrors ``mano_parameter_dict_to_joints_bt``: decode with centered MANO outputs and
    apply the external translation after converting meters -> millimeters.
    """
    if trans_bt.ndim != 3 or root_orient_bt.shape[-1] != 3 or betas_bt.shape[-1] != 10:
        raise ValueError(
            f"expected trans (B,T,3), root (B,T,3), betas (B,T,10); got "
            f"{tuple(trans_bt.shape)}, {tuple(root_orient_bt.shape)}, {tuple(betas_bt.shape)}"
        )
    b, t, _ = trans_bt.shape
    n = b * t
    trans = trans_bt.reshape(n, 3)
    root = root_orient_bt.reshape(n, 3)
    hand = hand_pose_aa_bt.reshape(n, 45)
    betas = betas_bt.reshape(n, 10)
    pca = mano_axisang45_to_pca(hand, mano_layer)
    pose = torch.cat([root, pca], dim=-1)

    parts: list[torch.Tensor] = []
    for s in range(0, n, max(1, int(chunk))):
        e = min(n, s + max(1, int(chunk)))
        v, _ = mano_layer(
            pose[s:e],
            th_betas=betas[s:e],
        )
        parts.append(v + trans[s:e].unsqueeze(1) * 1000.0)
    v_all = torch.cat(parts, dim=0)
    return v_all.view(b, t, v_all.shape[1], 3)


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
