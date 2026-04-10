"""Training debug: GT | Pred PNG via vis.mano_render.render_hand_on_frame (same pipeline as zarr viewer)."""

from __future__ import annotations

import os
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[misc, assignment]

from dataloader.mano_pca import mano_axisang45_to_pca


def _mano_layer_device(layer: nn.Module) -> torch.device:
    """ManoLayer has buffers only (no Parameters); use buffers() for device."""
    for p in layer.parameters():
        return p.device
    for b in layer.buffers():
        return b.device
    return torch.device("cpu")


def _as_numpy_intrinsic(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _scale_intrinsics(
    fx: float, fy: float, cx: float, cy: float, w0: int, h0: int, w1: int, h1: int
) -> tuple[float, float, float, float]:
    sx = w1 / max(w0, 1)
    sy = h1 / max(h0, 1)
    return fx * sx, fy * sy, cx * sx, cy * sy


def _aa_to_rotmat_np(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(aa))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    k = aa / theta
    x, y, z = float(k[0]), float(k[1]), float(k[2])
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    r = np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return r.astype(np.float32)


def _aa_to_rot6_np(aa: np.ndarray) -> np.ndarray:
    """First two columns of R(aa); matches vis rot6_to_rotmat convention for wrist_params."""
    r = _aa_to_rotmat_np(aa)
    return np.concatenate([r[:, 0], r[:, 1]], axis=0).astype(np.float32)


def _presence_int(left_on: bool, right_on: bool) -> int:
    if left_on and right_on:
        return 3
    if left_on:
        return 1
    if right_on:
        return 2
    return 0


def _pack_mano_render_args(
    mano_l: Mapping[str, torch.Tensor],
    mano_r: Mapping[str, torch.Tensor],
    ml: nn.Module,
    mr: nn.Module,
    bi: int,
    ti: int,
    device: torch.device,
    *,
    left_on: bool,
    right_on: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    mano_params: dict[str, Any] = {}
    wrist_params: dict[str, Any] = {}
    shape_params: dict[str, Any] = {}
    if left_on:
        hp = mano_l["hand_pose"][bi : bi + 1, ti : ti + 1].to(device)
        pca = mano_axisang45_to_pca(hp.squeeze(0).squeeze(0), ml)
        mano_params["left"] = pca.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        wrist_params["left_translation"] = (
            mano_l["trans"][bi, ti].detach().float().cpu().numpy().astype(np.float32)
        )
        wrist_params["left_rotation"] = _aa_to_rot6_np(
            mano_l["root_orient"][bi, ti].detach().float().cpu().numpy()
        )
        shape_params["left"] = mano_l["betas"][bi, ti].detach().float().cpu().numpy().astype(np.float32)
    if right_on:
        hp = mano_r["hand_pose"][bi : bi + 1, ti : ti + 1].to(device)
        pca = mano_axisang45_to_pca(hp.squeeze(0).squeeze(0), mr)
        mano_params["right"] = pca.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        wrist_params["right_translation"] = (
            mano_r["trans"][bi, ti].detach().float().cpu().numpy().astype(np.float32)
        )
        wrist_params["right_rotation"] = _aa_to_rot6_np(
            mano_r["root_orient"][bi, ti].detach().float().cpu().numpy()
        )
        shape_params["right"] = mano_r["betas"][bi, ti].detach().float().cpu().numpy().astype(np.float32)
    return mano_params, wrist_params, shape_params, _presence_int(left_on, right_on)


def _frame_rgb_uint8(video_btchw: torch.Tensor, bi: int, ti: int) -> np.ndarray:
    fr = video_btchw[bi, ti].detach().float().cpu().clamp(0.0, 1.0).numpy()
    fr = np.transpose(fr, (1, 2, 0))
    return (fr * 255.0).round().clip(0, 255).astype(np.uint8)


def maybe_save_train_mano_compare_png(
    *,
    out_path: str,
    video_btchw: torch.Tensor,
    intr_bt: torch.Tensor,
    mano_l_gt: Mapping[str, torch.Tensor],
    mano_r_gt: Mapping[str, torch.Tensor],
    mano_l_pr: Mapping[str, torch.Tensor],
    mano_r_pr: Mapping[str, torch.Tensor],
    pred_intr_bt: torch.Tensor,
    exist_bt2: torch.Tensor,
    mano_pca_layers: tuple[nn.Module, nn.Module],
    joint_chunk: int,
    device: torch.device,
    bi: int = 0,
    ti: int | None = None,
) -> bool:
    """3 panels: GT | GT intr + Pred MANO | Pred intr + Pred MANO."""
    del joint_chunk  # API compatibility with train_loop; render uses ManoLayer forward like vis.
    if cv2 is None:
        return False

    from vis.mano_render import MANO_AVAILABLE, render_hand_on_frame

    if not MANO_AVAILABLE:
        return False

    b, t, _, h, w = video_btchw.shape
    if ti is None:
        ti = max(0, t // 2)
    if bi < 0 or bi >= b or ti < 0 or ti >= t:
        return False

    left_on = float(exist_bt2[bi, ti, 0].item()) > 0.5
    right_on = float(exist_bt2[bi, ti, 1].item()) > 0.5
    if not left_on and not right_on:
        return False

    intr = _as_numpy_intrinsic(intr_bt[bi, ti])
    if intr.size < 4:
        return False
    fx, fy, cx, cy = float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3])
    intrinsic_gt_np = np.array([fx, fy, cx, cy], dtype=np.float32)

    pred_intr = _as_numpy_intrinsic(pred_intr_bt[bi, ti])
    if pred_intr.size < 4:
        return False
    pfx = max(float(pred_intr[0]), 1e-3)
    pfy = max(float(pred_intr[1]), 1e-3)
    pcx = float(pred_intr[2])
    pcy = float(pred_intr[3])
    intrinsic_pred_np = np.array([pfx, pfy, pcx, pcy], dtype=np.float32)

    ml, mr = mano_pca_layers
    mano_layers_dict = {"left": ml, "right": mr}

    # train.py moves ManoLayer to CUDA; vis generate_mano_mesh uses CPU torch.from_numpy inputs.
    orig_dev = _mano_layer_device(ml)
    need_restore_mano = orig_dev.type == "cuda"
    if need_restore_mano:
        ml.cpu()
        mr.cpu()

    frame_rgb = _frame_rgb_uint8(video_btchw, bi, ti)

    mp_g, wp_g, sp_g, pr_g = _pack_mano_render_args(
        mano_l_gt, mano_r_gt, ml, mr, bi, ti, device, left_on=left_on, right_on=right_on
    )
    mp_p, wp_p, sp_p, pr_p = _pack_mano_render_args(
        mano_l_pr, mano_r_pr, ml, mr, bi, ti, device, left_on=left_on, right_on=right_on
    )

    try:
        # Supervision is already in camera frame; vis uses world + extrinsic -> cam, so use identity.
        img_gt = render_hand_on_frame(
            frame_rgb.copy(),
            mano_params=mp_g,
            wrist_params=wp_g,
            extrinsic=None,
            intrinsic=intrinsic_gt_np,
            presence=pr_g,
            shape_params=sp_g,
            mano_layers=mano_layers_dict,
            fingertips=None,
            auto_reframe=False,
        )
        img_pr_gt_intr = render_hand_on_frame(
            frame_rgb.copy(),
            mano_params=mp_p,
            wrist_params=wp_p,
            extrinsic=None,
            intrinsic=intrinsic_gt_np,
            presence=pr_p,
            shape_params=sp_p,
            mano_layers=mano_layers_dict,
            fingertips=None,
            auto_reframe=False,
        )
        img_pr_pred_intr = render_hand_on_frame(
            frame_rgb.copy(),
            mano_params=mp_p,
            wrist_params=wp_p,
            extrinsic=None,
            intrinsic=intrinsic_pred_np,
            presence=pr_p,
            shape_params=sp_p,
            mano_layers=mano_layers_dict,
            fingertips=None,
            auto_reframe=False,
        )
    finally:
        if need_restore_mano:
            ml.to(orig_dev)
            mr.to(orig_dev)

    combo_rgb = np.concatenate([img_gt, img_pr_gt_intr, img_pr_pred_intr], axis=1)
    combo_bgr = cv2.cvtColor(combo_rgb, cv2.COLOR_RGB2BGR)
    w_img = img_gt.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combo_bgr, "GT", (8, 24), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combo_bgr, "GT intr + Pred", (w_img + 8, 24), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combo_bgr, "Pred intr + Pred", (2 * w_img + 8, 24), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, combo_bgr)
    return True
