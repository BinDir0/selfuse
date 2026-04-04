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
DEFAULT_MANO_PARAM_KEYS = frozenset({"trans", "betas"})


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
    add_param("root_orient", 1.0)
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


def mano_masked_joint_mse_m2(
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
    per_bt = (j_p - j_t).pow(2).sum(-1).mean(-1)
    return (per_bt * mask_bt).sum() / mask_bt.sum().clamp_min(1.0)


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
