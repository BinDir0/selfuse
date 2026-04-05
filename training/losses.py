from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader.mano_pca import (
    mano_axisang45_to_pca,
    mano_parameter_dict_to_joints_bt,
    mano_parameter_dict_to_verts_bt,
)

MANO_PARAM_KEY_CHOICES = frozenset({"trans", "root_orient", "hand_pose", "betas"})
DEFAULT_MANO_PARAM_KEYS = frozenset({"trans", "root_orient"})

# 21 joints in manopth order; wrist 0, five chains ending at tips 4,8,12,16,20.
MANO_FINGERTIP_JOINT_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)

# Edges (child, parent), same as vis skeleton.
_MANO_HAND_BONE_EDGES: tuple[tuple[int, int], ...] = (
    (1, 0),
    (2, 1),
    (3, 2),
    (4, 3),
    (5, 0),
    (6, 5),
    (7, 6),
    (8, 7),
    (9, 0),
    (10, 9),
    (11, 10),
    (12, 11),
    (13, 0),
    (14, 13),
    (15, 14),
    (16, 15),
    (17, 0),
    (18, 17),
    (19, 18),
    (20, 19),
)


def bce_existence(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


def masked_param_tensor_loss(
    pred_v: torch.Tensor,
    tgt_v: torch.Tensor,
    mask_exp: torch.Tensor,
    *,
    kind: Literal["l1", "l2", "huber"],
    huber_delta: float,
) -> torch.Tensor:
    if kind == "l1":
        elem = (pred_v - tgt_v).abs()
    elif kind == "l2":
        d = pred_v - tgt_v
        elem = d * d
    else:
        elem = F.smooth_l1_loss(pred_v, tgt_v, reduction="none", beta=huber_delta)
    denom = mask_exp.sum().clamp_min(1.0)
    return (elem * mask_exp).sum() / denom


def mano_regression_loss(
    pred: dict[str, torch.Tensor],
    tgt: dict[str, torch.Tensor],
    mask_bt: torch.Tensor,
    *,
    param_keys: frozenset[str],
    hand_pose_weight: float = 1.0,
    root_orient_weight: float = 1.0,
    param_loss: Literal["l1", "l2", "huber"] = "l2",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    if mask_bt.sum() < 1e-6:
        return pred["trans"].new_tensor(0.0)
    m = mask_bt.unsqueeze(-1)
    acc = pred["trans"].new_tensor(0.0)
    w_sum = 0.0

    def add_param(key: str, weight: float) -> None:
        nonlocal acc, w_sum
        if weight <= 0.0 or key not in param_keys:
            return
        acc = acc + weight * masked_param_tensor_loss(
            pred[key], tgt[key], m, kind=param_loss, huber_delta=huber_delta
        )
        w_sum += weight

    add_param("trans", 1.0)
    add_param("root_orient", root_orient_weight)
    add_param("hand_pose", hand_pose_weight)
    add_param("betas", 1.0)
    if w_sum <= 0.0:
        return pred["trans"].new_tensor(0.0)
    return acc / w_sum


def mano_hand_pose_pca_mse(
    pred: dict[str, torch.Tensor],
    tgt: dict[str, torch.Tensor],
    mask_bt: torch.Tensor,
    mano_layer: nn.Module,
) -> torch.Tensor:
    """Mean squared error on MANO finger PCA coefficients (from axis-angle via layer basis)."""
    if mask_bt.sum() < 1e-6:
        return pred["trans"].new_tensor(0.0)
    hp_p = pred["hand_pose"]
    hp_t = tgt["hand_pose"]
    b, t, _ = hp_p.shape
    p_p = mano_axisang45_to_pca(hp_p.reshape(b * t, 45), mano_layer).view(b, t, -1)
    with torch.no_grad():
        p_t = mano_axisang45_to_pca(hp_t.reshape(b * t, 45), mano_layer).view(b, t, -1)
    per_bt = (p_p - p_t).pow(2).mean(-1)
    return (per_bt * mask_bt).sum() / mask_bt.sum().clamp_min(1.0)


def mano_joint_weight_vector_21(
    preset: Literal["uniform", "fingertip"],
    fingertip_scale: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Per-joint nonnegative weights for joint MSE; None means uniform mean (same as all-ones / 21)."""
    if preset == "uniform":
        return None
    w = torch.ones(21, device=device, dtype=dtype)
    for i in MANO_FINGERTIP_JOINT_INDICES:
        w[i] = w[i] * fingertip_scale
    return w


def mano_masked_joint_mse_m2(
    pred: dict[str, torch.Tensor],
    tgt: dict[str, torch.Tensor],
    mask_bt: torch.Tensor,
    mano_layer: nn.Module,
    *,
    chunk: int,
    joint_weight_21: torch.Tensor | None = None,
    smooth_max_weight: float = 0.0,
    smooth_max_tau: float = 0.002,
) -> torch.Tensor:
    if mask_bt.sum() < 1e-6:
        return pred["trans"].new_tensor(0.0)
    mm_to_m = 0.001
    with torch.no_grad():
        j_t = mano_parameter_dict_to_joints_bt(
            tgt["trans"],
            tgt["root_orient"],
            tgt["hand_pose"],
            tgt["betas"],
            mano_layer,
            chunk=chunk,
        )
        j_t = j_t * mm_to_m
    j_p = mano_parameter_dict_to_joints_bt(
        pred["trans"],
        pred["root_orient"],
        pred["hand_pose"],
        pred["betas"],
        mano_layer,
        chunk=chunk,
    ) * mm_to_m
    e = (j_p - j_t).pow(2).sum(-1)
    if joint_weight_21 is None:
        l_avg = e.mean(dim=-1)
    else:
        w = joint_weight_21.to(device=e.device, dtype=e.dtype).view(1, 1, -1)
        l_avg = (e * w).sum(dim=-1) / w.sum().clamp_min(1e-8)
    if smooth_max_weight > 0.0:
        tau = max(float(smooth_max_tau), 1e-8)
        t = e.new_tensor(tau)
        l_smax = t * torch.logsumexp(e / t, dim=-1)
        per_bt = l_avg + smooth_max_weight * l_smax
    else:
        per_bt = l_avg
    return (per_bt * mask_bt).sum() / mask_bt.sum().clamp_min(1.0)


def mano_masked_bone_length_mse_m2(
    pred: dict[str, torch.Tensor],
    tgt: dict[str, torch.Tensor],
    mask_bt: torch.Tensor,
    mano_layer: nn.Module,
    *,
    chunk: int,
) -> torch.Tensor:
    """MSE of parent-child bone lengths (m) vs GT; mean over 20 edges, then mask-mean over B,T."""
    if mask_bt.sum() < 1e-6:
        return pred["trans"].new_tensor(0.0)
    mm_to_m = 0.001
    with torch.no_grad():
        j_t = mano_parameter_dict_to_joints_bt(
            tgt["trans"],
            tgt["root_orient"],
            tgt["hand_pose"],
            tgt["betas"],
            mano_layer,
            chunk=chunk,
        )
        j_t = j_t * mm_to_m
    j_p = (
        mano_parameter_dict_to_joints_bt(
            pred["trans"],
            pred["root_orient"],
            pred["hand_pose"],
            pred["betas"],
            mano_layer,
            chunk=chunk,
        )
        * mm_to_m
    )
    errs: list[torch.Tensor] = []
    for c, p in _MANO_HAND_BONE_EDGES:
        len_p = (j_p[..., c, :] - j_p[..., p, :]).norm(dim=-1)
        len_t = (j_t[..., c, :] - j_t[..., p, :]).norm(dim=-1)
        errs.append((len_p - len_t).pow(2))
    err_bt = torch.stack(errs, dim=-1).mean(dim=-1)
    return (err_bt * mask_bt).sum() / mask_bt.sum().clamp_min(1.0)


def mano_masked_bone_direction_mse(
    pred: dict[str, torch.Tensor],
    tgt: dict[str, torch.Tensor],
    mask_bt: torch.Tensor,
    mano_layer: nn.Module,
    *,
    chunk: int,
) -> torch.Tensor:
    """MSE of unit bone directions vs GT (dimensionless); 20 edges; same graph as bone-length loss."""
    if mask_bt.sum() < 1e-6:
        return pred["trans"].new_tensor(0.0)
    mm_to_m = 0.001
    with torch.no_grad():
        j_t = mano_parameter_dict_to_joints_bt(
            tgt["trans"],
            tgt["root_orient"],
            tgt["hand_pose"],
            tgt["betas"],
            mano_layer,
            chunk=chunk,
        )
        j_t = j_t * mm_to_m
    j_p = (
        mano_parameter_dict_to_joints_bt(
            pred["trans"],
            pred["root_orient"],
            pred["hand_pose"],
            pred["betas"],
            mano_layer,
            chunk=chunk,
        )
        * mm_to_m
    )
    errs: list[torch.Tensor] = []
    eps = 1e-6
    for c, p in _MANO_HAND_BONE_EDGES:
        d_p = j_p[..., c, :] - j_p[..., p, :]
        d_t = j_t[..., c, :] - j_t[..., p, :]
        u_p = d_p / d_p.norm(dim=-1, keepdim=True).clamp_min(eps)
        u_t = d_t / d_t.norm(dim=-1, keepdim=True).clamp_min(eps)
        errs.append((u_p - u_t).pow(2).sum(dim=-1))
    err_bt = torch.stack(errs, dim=-1).mean(dim=-1)
    return (err_bt * mask_bt).sum() / mask_bt.sum().clamp_min(1.0)


def mano_masked_vert_mse_m2(
    pred: dict[str, torch.Tensor],
    tgt: dict[str, torch.Tensor],
    mask_bt: torch.Tensor,
    mano_layer: nn.Module,
    *,
    chunk: int,
) -> torch.Tensor:
    if mask_bt.sum() < 1e-6:
        return pred["trans"].new_tensor(0.0)
    mm_to_m = 0.001
    with torch.no_grad():
        v_t = mano_parameter_dict_to_verts_bt(
            tgt["trans"],
            tgt["root_orient"],
            tgt["hand_pose"],
            tgt["betas"],
            mano_layer,
            chunk=chunk,
        )
        v_t = v_t * mm_to_m
    v_p = (
        mano_parameter_dict_to_verts_bt(
            pred["trans"],
            pred["root_orient"],
            pred["hand_pose"],
            pred["betas"],
            mano_layer,
            chunk=chunk,
        )
        * mm_to_m
    )
    per_bt = (v_p - v_t).pow(2).sum(-1).mean(-1)
    return (per_bt * mask_bt).sum() / mask_bt.sum().clamp_min(1.0)
