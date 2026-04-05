"""训练调试：将相机系 MANO 关节投影到当前帧，拼 GT | Pred 存图。"""

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

from dataloader.mano_pca import mano_parameter_dict_to_joints_bt

# 与 vis/mano_render.py 中手指骨架一致（MANO 21 关节）
_MANO_JOINT_TREE: list[list[tuple[int, int]]] = [
    [(0, 1), (1, 2), (2, 3), (3, 4)],
    [(0, 5), (5, 6), (6, 7), (7, 8)],
    [(0, 9), (9, 10), (10, 11), (11, 12)],
    [(0, 13), (13, 14), (14, 15), (15, 16)],
    [(0, 17), (17, 18), (18, 19), (19, 20)],
]


def _project_points(pts_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    zs = pts_cam[:, 2] + 1e-8
    us = fx * (pts_cam[:, 0] / zs) + cx
    vs = fy * (pts_cam[:, 1] / zs) + cy
    return np.stack([us, vs], axis=1)


def _scale_intrinsics(
    fx: float, fy: float, cx: float, cy: float, w0: int, h0: int, w1: int, h1: int
) -> tuple[float, float, float, float]:
    sx = w1 / max(w0, 1)
    sy = h1 / max(h0, 1)
    return fx * sx, fy * sy, cx * sx, cy * sy


def _as_numpy_intrinsic(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _video_slice_to_bgr_uint8(video_btchw: torch.Tensor, bi: int, ti: int) -> np.ndarray:
    """video: (B,T,3,H,W) float, 值域约 [0,1]。"""
    fr = video_btchw[bi, ti].detach().float().cpu().clamp(0.0, 1.0).numpy()
    fr = np.transpose(fr, (1, 2, 0))
    fr_u8 = (fr * 255.0).round().clip(0, 255).astype(np.uint8)
    return fr_u8[:, :, ::-1].copy()


def _draw_hand_skeleton(
    img_bgr: np.ndarray,
    joints_cam_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    line_bgr: tuple[int, int, int],
    point_bgr: tuple[int, int, int],
    line_thickness: int = 2,
) -> None:
    if cv2 is None:
        return
    h, w = img_bgr.shape[:2]
    uv = _project_points(joints_cam_m, fx, fy, cx, cy).astype(np.int32)
    z_ok = joints_cam_m[:, 2] > 1e-5
    in_img = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )
    mask = z_ok & in_img
    for chain in _MANO_JOINT_TREE:
        for j1, j2 in chain:
            if mask[j1] and mask[j2]:
                cv2.line(
                    img_bgr,
                    tuple(uv[j1]),
                    tuple(uv[j2]),
                    line_bgr,
                    line_thickness,
                    lineType=cv2.LINE_AA,
                )
    for i in range(uv.shape[0]):
        if mask[i]:
            cv2.circle(img_bgr, tuple(uv[i]), 3, point_bgr, -1, lineType=cv2.LINE_AA)


def _joints_cam_m_for_hand(
    mano: Mapping[str, torch.Tensor],
    mano_layer: nn.Module,
    bi: int,
    ti: int,
    *,
    joint_chunk: int,
    device: torch.device,
) -> np.ndarray:
    sl = slice(bi, bi + 1), slice(ti, ti + 1)
    trans = mano["trans"][sl].to(device)
    root = mano["root_orient"][sl].to(device)
    hand = mano["hand_pose"][sl].to(device)
    betas = mano["betas"][sl].to(device)
    with torch.no_grad():
        j = mano_parameter_dict_to_joints_bt(
            trans, root, hand, betas, mano_layer, chunk=joint_chunk
        )
    # 与 loss 一致：manopth 关节单位为 mm，投影用米
    jm = (j.squeeze(0).squeeze(0) * 0.001).float().cpu().numpy()
    return jm


def maybe_save_train_mano_compare_png(
    *,
    out_path: str,
    video_btchw: torch.Tensor,
    batch: dict[str, Any],
    mano_l_gt: Mapping[str, torch.Tensor],
    mano_r_gt: Mapping[str, torch.Tensor],
    mano_l_pr: Mapping[str, torch.Tensor],
    mano_r_pr: Mapping[str, torch.Tensor],
    exist_bt2: torch.Tensor,
    mano_pca_layers: tuple[nn.Module, nn.Module],
    joint_chunk: int,
    device: torch.device,
    bi: int = 0,
    ti: int | None = None,
) -> bool:
    """GT 在左、Pred 在右；仅当可导入 cv2 且 MANO 层可用时写盘。成功返回 True。"""
    if cv2 is None:
        return False
    b, t, _, h, w = video_btchw.shape
    if ti is None:
        ti = max(0, t // 2)
    if bi < 0 or bi >= b or ti < 0 or ti >= t:
        return False

    vid_np = batch["video"]
    if torch.is_tensor(vid_np):
        h0, w0 = int(vid_np.shape[-2]), int(vid_np.shape[-1])
    else:
        h0, w0 = int(np.asarray(vid_np).shape[-2]), int(np.asarray(vid_np).shape[-1])

    intr = _as_numpy_intrinsic(batch["intrinsic"][bi, ti])
    if intr.size < 4:
        return False
    fx, fy, cx, cy = float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3])
    fx, fy, cx, cy = _scale_intrinsics(fx, fy, cx, cy, w0, h0, w, h)

    ml, mr = mano_pca_layers
    left_on = float(exist_bt2[bi, ti, 0].item()) > 0.5
    right_on = float(exist_bt2[bi, ti, 1].item()) > 0.5

    img_gt = _video_slice_to_bgr_uint8(video_btchw, bi, ti)
    img_pr = img_gt.copy()

    c_l_line, c_l_pt = (255, 100, 0), (200, 50, 0)
    c_r_line, c_r_pt = (0, 100, 255), (0, 50, 200)
    if left_on:
        jl_gt = _joints_cam_m_for_hand(mano_l_gt, ml, bi, ti, joint_chunk=joint_chunk, device=device)
        jl_pr = _joints_cam_m_for_hand(mano_l_pr, ml, bi, ti, joint_chunk=joint_chunk, device=device)
        _draw_hand_skeleton(img_gt, jl_gt, fx, fy, cx, cy, line_bgr=c_l_line, point_bgr=c_l_pt)
        _draw_hand_skeleton(img_pr, jl_pr, fx, fy, cx, cy, line_bgr=c_l_line, point_bgr=c_l_pt)
    if right_on:
        jr_gt = _joints_cam_m_for_hand(mano_r_gt, mr, bi, ti, joint_chunk=joint_chunk, device=device)
        jr_pr = _joints_cam_m_for_hand(mano_r_pr, mr, bi, ti, joint_chunk=joint_chunk, device=device)
        _draw_hand_skeleton(img_gt, jr_gt, fx, fy, cx, cy, line_bgr=c_r_line, point_bgr=c_r_pt)
        _draw_hand_skeleton(img_pr, jr_pr, fx, fy, cx, cy, line_bgr=c_r_line, point_bgr=c_r_pt)

    _, w_img = img_gt.shape[:2]
    combo = np.concatenate([img_gt, img_pr], axis=1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combo, "GT", (8, 24), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combo, "Pred", (w_img + 8, 24), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, combo)
    return True
