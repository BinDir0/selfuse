from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader.mano_pca import mano_parameter_dict_to_joints_bt

MANO_PARAM_KEY_CHOICES = frozenset({"trans", "root_orient", "hand_pose", "betas"})
DEFAULT_MANO_PARAM_KEYS = frozenset({"trans", "betas"})

# 21 joints in manopth order; wrist 0, five chains ending at tips 4,8,12,16,20.
MANO_FINGERTIP_JOINT_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)


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


def _skew_from_vec3(v: torch.Tensor) -> torch.Tensor:
    """v (...,3) -> skew-symmetric matrix (...,3,3)."""
    z = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    row0 = torch.stack([z, -vz, vy], dim=-1)
    row1 = torch.stack([vz, z, -vx], dim=-1)
    row2 = torch.stack([-vy, vx, z], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _axis_angle_to_rotmat(axis_angle: torch.Tensor) -> torch.Tensor:
    """axis-angle (...,3) -> rotation matrix (...,3,3)."""
    eps = 1e-6
    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp_min(eps)
    axis = axis_angle / theta
    k = _skew_from_vec3(axis)
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    eye = eye.view(*([1] * (axis_angle.ndim - 1)), 3, 3)
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    return eye + sin_t * k + (1.0 - cos_t) * (k @ k)


def masked_root_orient_geodesic_loss(
    pred_root: torch.Tensor,
    tgt_root: torch.Tensor,
    mask_bt: torch.Tensor,
    *,
    kind: Literal["l1", "l2", "huber"],
    huber_delta: float,
) -> torch.Tensor:
    """Geodesic loss on SO(3) for root_orient (axis-angle inputs)."""
    if mask_bt.sum() < 1e-6:
        return pred_root.new_tensor(0.0)
    rp = _axis_angle_to_rotmat(pred_root)
    rt = _axis_angle_to_rotmat(tgt_root)
    rel = rp.transpose(-1, -2) @ rt
    tr = rel.diagonal(dim1=-1, dim2=-2).sum(-1)
    cos = ((tr - 1.0) * 0.5).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
    ang = torch.acos(cos)
    if kind == "l1":
        elem = ang.abs()
    elif kind == "l2":
        elem = ang * ang
    else:
        elem = F.smooth_l1_loss(ang, torch.zeros_like(ang), reduction="none", beta=huber_delta)
    denom = mask_bt.sum().clamp_min(1.0)
    return (elem * mask_bt).sum() / denom


def masked_hand_pose_geodesic_loss(
    pred_pose: torch.Tensor,
    tgt_pose: torch.Tensor,
    mask_bt: torch.Tensor,
    *,
    kind: Literal["l1", "l2", "huber"],
    huber_delta: float,
) -> torch.Tensor:
    """Per-joint geodesic loss on SO(3) for hand_pose axis-angle (B,T,45)."""
    if mask_bt.sum() < 1e-6:
        return pred_pose.new_tensor(0.0)
    if pred_pose.shape[-1] % 3 != 0 or tgt_pose.shape[-1] % 3 != 0:
        raise ValueError(
            f"hand_pose last dim must be multiple of 3, got {pred_pose.shape[-1]} and {tgt_pose.shape[-1]}"
        )
    j = pred_pose.shape[-1] // 3
    pp = pred_pose.view(*pred_pose.shape[:-1], j, 3)
    pt = tgt_pose.view(*tgt_pose.shape[:-1], j, 3)
    rp = _axis_angle_to_rotmat(pp)
    rt = _axis_angle_to_rotmat(pt)
    rel = rp.transpose(-1, -2) @ rt
    tr = rel.diagonal(dim1=-1, dim2=-2).sum(-1)
    cos = ((tr - 1.0) * 0.5).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
    ang = torch.acos(cos)  # (B,T,J)
    if kind == "l1":
        elem = ang.abs()
    elif kind == "l2":
        elem = ang * ang
    else:
        elem = F.smooth_l1_loss(ang, torch.zeros_like(ang), reduction="none", beta=huber_delta)
    per_bt = elem.mean(dim=-1)
    denom = mask_bt.sum().clamp_min(1.0)
    return (per_bt * mask_bt).sum() / denom


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
        if key == "root_orient":
            val = masked_root_orient_geodesic_loss(
                pred["root_orient"],
                tgt["root_orient"],
                mask_bt,
                kind=param_loss,
                huber_delta=huber_delta,
            )
        elif key == "hand_pose":
            val = masked_hand_pose_geodesic_loss(
                pred["hand_pose"],
                tgt["hand_pose"],
                mask_bt,
                kind=param_loss,
                huber_delta=huber_delta,
            )
        else:
            val = masked_param_tensor_loss(
                pred[key], tgt[key], m, kind=param_loss, huber_delta=huber_delta
            )
        acc = acc + weight * val
        w_sum += weight

    add_param("trans", 1.0)
    add_param("root_orient", root_orient_weight)
    add_param("hand_pose", hand_pose_weight)
    add_param("betas", 1.0)
    if w_sum <= 0.0:
        return pred["trans"].new_tensor(0.0)
    return acc / w_sum


def mano_regression_per_key_raw(
    pred: dict[str, torch.Tensor],
    tgt: dict[str, torch.Tensor],
    mask_bt: torch.Tensor,
    *,
    param_keys: frozenset[str],
    param_loss: Literal["l1", "l2", "huber"] = "l2",
    huber_delta: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Per-parameter masked mean (same as ``masked_param_tensor_loss``), for TensorBoard per-key curves."""
    if mask_bt.sum() < 1e-6:
        z = pred["trans"].new_tensor(0.0)
        return {k: z for k in sorted(param_keys & MANO_PARAM_KEY_CHOICES)}
    m = mask_bt.unsqueeze(-1)
    out: dict[str, torch.Tensor] = {}
    for key in ("trans", "root_orient", "hand_pose", "betas"):
        if key not in param_keys:
            continue
        if key == "root_orient":
            out[key] = masked_root_orient_geodesic_loss(
                pred["root_orient"],
                tgt["root_orient"],
                mask_bt,
                kind=param_loss,
                huber_delta=huber_delta,
            )
        elif key == "hand_pose":
            out[key] = masked_hand_pose_geodesic_loss(
                pred["hand_pose"],
                tgt["hand_pose"],
                mask_bt,
                kind=param_loss,
                huber_delta=huber_delta,
            )
        else:
            out[key] = masked_param_tensor_loss(
                pred[key], tgt[key], m, kind=param_loss, huber_delta=huber_delta
            )
    return out


def mano_joint_weight_vector_21(
    preset: Literal["uniform", "fingertip"],
    fingertip_scale: float,
    wrist_scale: float,
    chain_scale: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Per-joint nonnegative weights for joint MSE.

    ``chain_scale``: base weight for non-wrist joints before fingertip / wrist multipliers
    (typically 0.5 so "middle" joints are down-weighted vs previous defaults).

    None only when preset is uniform and wrist_scale==1 (plain mean over 21).
    """
    if preset == "uniform" and abs(float(wrist_scale) - 1.0) < 1e-6:
        return None
    w = torch.full((21,), float(chain_scale), device=device, dtype=dtype)
    if preset == "fingertip":
        for i in MANO_FINGERTIP_JOINT_INDICES:
            w[i] = w[i] * float(fingertip_scale)
    w[0] = w[0] * float(wrist_scale)
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


def mano_masked_mano_reproj_normalized_plane_m2(
    pred_mano: dict[str, torch.Tensor],
    tgt_mano: dict[str, torch.Tensor],
    mask_bt: torch.Tensor,
    mano_layer: nn.Module,
    intr_bt4: torch.Tensor,
    *,
    chunk: int,
    joint_weight_21: torch.Tensor | None,
    z_min: float = 0.05,
) -> torch.Tensor:
    """(X/Z, Y/Z) vs GT joints from MANO; same as former kp-head path but joints come from ``pred_mano``."""
    del intr_bt4
    if mask_bt.sum() < 1e-6:
        return pred_mano["trans"].new_tensor(0.0)
    mm_to_m = 0.001
    zm = max(float(z_min), 1e-6)
    with torch.no_grad():
        j_t = mano_parameter_dict_to_joints_bt(
            tgt_mano["trans"],
            tgt_mano["root_orient"],
            tgt_mano["hand_pose"],
            tgt_mano["betas"],
            mano_layer,
            chunk=chunk,
        )
        j_t = j_t * mm_to_m
    j_p = (
        mano_parameter_dict_to_joints_bt(
            pred_mano["trans"],
            pred_mano["root_orient"],
            pred_mano["hand_pose"],
            pred_mano["betas"],
            mano_layer,
            chunk=chunk,
        )
        * mm_to_m
    )
    zp = j_p[..., 2].clamp_min(zm)
    zt = j_t[..., 2].clamp_min(zm)
    plane_p = torch.stack([j_p[..., 0] / zp, j_p[..., 1] / zp], dim=-1)
    plane_t = torch.stack([j_t[..., 0] / zt, j_t[..., 1] / zt], dim=-1)
    valid = (j_p[..., 2] > zm) & (j_t[..., 2] > zm)
    valid = valid & torch.isfinite(plane_p).all(dim=-1) & torch.isfinite(plane_t).all(dim=-1)
    e = (plane_p - plane_t).pow(2).sum(dim=-1)
    m = valid & mask_bt.unsqueeze(-1).bool()
    if joint_weight_21 is None:
        w = e.new_ones(e.shape[-1])
    else:
        w = joint_weight_21.to(device=e.device, dtype=e.dtype).view(1, 1, -1).expand_as(e)
    num = (e * w * m.float()).sum(dim=-1)
    den = (w * m.float()).sum(dim=-1).clamp_min(1e-8)
    l_avg = num / den
    frame_ok = den > 1e-7
    per_bt = torch.where(frame_ok, l_avg, l_avg.new_zeros(()))
    return (per_bt * mask_bt * frame_ok.float()).sum() / mask_bt.sum().clamp_min(1.0)


def weakcam_reg_loss(pred_cam_bt3: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
    """Regularize weak camera to identity-ish: s->1, tx/ty->0."""
    if mask_bt.sum() < 1e-6:
        return pred_cam_bt3.new_tensor(0.0)
    s = pred_cam_bt3[..., 0]
    tx = pred_cam_bt3[..., 1]
    ty = pred_cam_bt3[..., 2]
    reg = (s - 1.0).pow(2) + tx.pow(2) + ty.pow(2)
    return (reg * mask_bt).sum() / mask_bt.sum().clamp_min(1.0)


def mano_masked_weakcam_reproj_normalized_plane_m2(
    pred_mano: dict[str, torch.Tensor],
    tgt_mano: dict[str, torch.Tensor],
    pred_cam_bt3: torch.Tensor,
    mask_bt: torch.Tensor,
    mano_layer: nn.Module,
    *,
    chunk: int,
    joint_weight_21: torch.Tensor | None,
    z_min: float = 0.05,
) -> torch.Tensor:
    """Weak-cam reprojection on normalized plane: (s*x/z+tx, s*y/z+ty) vs GT (x/z,y/z)."""
    if mask_bt.sum() < 1e-6:
        return pred_mano["trans"].new_tensor(0.0)
    mm_to_m = 0.001
    zm = max(float(z_min), 1e-6)
    with torch.no_grad():
        j_t = mano_parameter_dict_to_joints_bt(
            tgt_mano["trans"],
            tgt_mano["root_orient"],
            tgt_mano["hand_pose"],
            tgt_mano["betas"],
            mano_layer,
            chunk=chunk,
        )
        j_t = j_t * mm_to_m
    j_p = (
        mano_parameter_dict_to_joints_bt(
            pred_mano["trans"],
            pred_mano["root_orient"],
            pred_mano["hand_pose"],
            pred_mano["betas"],
            mano_layer,
            chunk=chunk,
        )
        * mm_to_m
    )
    zp = j_p[..., 2].clamp_min(zm)
    zt = j_t[..., 2].clamp_min(zm)
    plane_p = torch.stack([j_p[..., 0] / zp, j_p[..., 1] / zp], dim=-1)
    plane_t = torch.stack([j_t[..., 0] / zt, j_t[..., 1] / zt], dim=-1)

    s = pred_cam_bt3[..., 0].unsqueeze(-1).unsqueeze(-1)
    tx = pred_cam_bt3[..., 1].unsqueeze(-1).unsqueeze(-1)
    ty = pred_cam_bt3[..., 2].unsqueeze(-1).unsqueeze(-1)
    plane_pw = torch.cat([s * plane_p[..., 0:1] + tx, s * plane_p[..., 1:2] + ty], dim=-1)

    valid = (j_p[..., 2] > zm) & (j_t[..., 2] > zm)
    valid = valid & torch.isfinite(plane_pw).all(dim=-1) & torch.isfinite(plane_t).all(dim=-1)
    e = (plane_pw - plane_t).pow(2).sum(dim=-1)
    m = valid & mask_bt.unsqueeze(-1).bool()
    if joint_weight_21 is None:
        w = e.new_ones(e.shape[-1])
    else:
        w = joint_weight_21.to(device=e.device, dtype=e.dtype).view(1, 1, -1).expand_as(e)
    num = (e * w * m.float()).sum(dim=-1)
    den = (w * m.float()).sum(dim=-1).clamp_min(1e-8)
    l_avg = num / den
    frame_ok = den > 1e-7
    per_bt = torch.where(frame_ok, l_avg, l_avg.new_zeros(()))
    return (per_bt * mask_bt * frame_ok.float()).sum() / mask_bt.sum().clamp_min(1.0)


def mano_masked_weakcam_reproj_pixel_m2(
    pred_mano: dict[str, torch.Tensor],
    tgt_mano: dict[str, torch.Tensor],
    pred_cam_bt3: torch.Tensor,
    intr_bt4: torch.Tensor,
    mask_bt: torch.Tensor,
    mano_layer: nn.Module,
    *,
    chunk: int,
    joint_weight_21: torch.Tensor | None,
    z_min: float = 0.05,
) -> torch.Tensor:
    """Weak-cam reprojection in pixel space using intrinsics (fx,fy,cx,cy) and pred_cam(s,tx,ty)."""
    if mask_bt.sum() < 1e-6:
        return pred_mano["trans"].new_tensor(0.0)
    if intr_bt4.shape[-1] != 4:
        raise ValueError(f"expected intr_bt4 (...,4) fx,fy,cx,cy, got {tuple(intr_bt4.shape)}")
    mm_to_m = 0.001
    zm = max(float(z_min), 1e-6)

    with torch.no_grad():
        j_t = mano_parameter_dict_to_joints_bt(
            tgt_mano["trans"],
            tgt_mano["root_orient"],
            tgt_mano["hand_pose"],
            tgt_mano["betas"],
            mano_layer,
            chunk=chunk,
        ) * mm_to_m
    j_p = (
        mano_parameter_dict_to_joints_bt(
            pred_mano["trans"],
            pred_mano["root_orient"],
            pred_mano["hand_pose"],
            pred_mano["betas"],
            mano_layer,
            chunk=chunk,
        )
        * mm_to_m
    )

    zp = j_p[..., 2].clamp_min(zm)
    zt = j_t[..., 2].clamp_min(zm)
    plane_p = torch.stack([j_p[..., 0] / zp, j_p[..., 1] / zp], dim=-1)
    plane_t = torch.stack([j_t[..., 0] / zt, j_t[..., 1] / zt], dim=-1)

    s = pred_cam_bt3[..., 0].unsqueeze(-1).unsqueeze(-1)
    tx = pred_cam_bt3[..., 1].unsqueeze(-1).unsqueeze(-1)
    ty = pred_cam_bt3[..., 2].unsqueeze(-1).unsqueeze(-1)
    plane_pw = torch.cat([s * plane_p[..., 0:1] + tx, s * plane_p[..., 1:2] + ty], dim=-1)

    fx = intr_bt4[..., 0].unsqueeze(-1).unsqueeze(-1)
    fy = intr_bt4[..., 1].unsqueeze(-1).unsqueeze(-1)
    cx = intr_bt4[..., 2].unsqueeze(-1).unsqueeze(-1)
    cy = intr_bt4[..., 3].unsqueeze(-1).unsqueeze(-1)
    pix_p = torch.cat([fx * plane_pw[..., 0:1] + cx, fy * plane_pw[..., 1:2] + cy], dim=-1)
    pix_t = torch.cat([fx * plane_t[..., 0:1] + cx, fy * plane_t[..., 1:2] + cy], dim=-1)

    valid = (j_p[..., 2] > zm) & (j_t[..., 2] > zm)
    valid = valid & torch.isfinite(pix_p).all(dim=-1) & torch.isfinite(pix_t).all(dim=-1)
    e = (pix_p - pix_t).pow(2).sum(dim=-1)
    m = valid & mask_bt.unsqueeze(-1).bool()
    if joint_weight_21 is None:
        w = e.new_ones(e.shape[-1])
    else:
        w = joint_weight_21.to(device=e.device, dtype=e.dtype).view(1, 1, -1).expand_as(e)
    num = (e * w * m.float()).sum(dim=-1)
    den = (w * m.float()).sum(dim=-1).clamp_min(1e-8)
    l_avg = num / den
    frame_ok = den > 1e-7
    per_bt = torch.where(frame_ok, l_avg, l_avg.new_zeros(()))
    return (per_bt * mask_bt * frame_ok.float()).sum() / mask_bt.sum().clamp_min(1.0)
