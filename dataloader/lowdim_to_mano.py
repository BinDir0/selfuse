"""由 lowdim 的腕部 world 位姿、外参与 shape 构造相机系 MANO 监督 dict。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "lowdim_wrist_to_mano_cam",
    "rot6d_to_rotmat",
    "rotmat_to_axis_angle",
]


def rot6d_to_rotmat(d6: torch.Tensor) -> torch.Tensor:
    """Zhou 6D，列向量组正交基。(..., 6) -> (..., 3, 3)。"""
    x_raw = d6[..., 0:3]
    y_raw = d6[..., 3:6]
    x = F.normalize(x_raw, dim=-1, eps=1e-8)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1, eps=1e-8)
    y = torch.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def rotmat_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """(…,3,3) -> (…,3) so(3) 向量。"""
    *batch_shape, _, _ = R.shape
    Rf = R.reshape(-1, 3, 3)
    m00, m01, m02 = Rf[:, 0, 0], Rf[:, 0, 1], Rf[:, 0, 2]
    m10, m11, m12 = Rf[:, 1, 0], Rf[:, 1, 1], Rf[:, 1, 2]
    m20, m21, m22 = Rf[:, 2, 0], Rf[:, 2, 1], Rf[:, 2, 2]
    tr = m00 + m11 + m22
    cos = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    sin = torch.sqrt((1.0 - cos * cos).clamp(min=0.0))
    theta = torch.atan2(sin, cos)
    rx = m21 - m12
    ry = m02 - m20
    rz = m10 - m01
    v = torch.stack((rx, ry, rz), dim=-1)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    aa = v / v_norm * theta.unsqueeze(-1)

    skew_small = torch.stack(
        (
            torch.zeros_like(m00),
            m01 - m10,
            m02 - m20,
            m10 - m01,
            torch.zeros_like(m00),
            m12 - m21,
            m20 - m02,
            m21 - m12,
            torch.zeros_like(m00),
        ),
        dim=-1,
    ).view(-1, 3, 3)
    aa_small = torch.stack((skew_small[:, 2, 1], skew_small[:, 0, 2], skew_small[:, 1, 0]), dim=-1)
    small = (sin.unsqueeze(-1) < 1e-4).squeeze(-1)
    aa = torch.where(small.unsqueeze(-1), aa_small, aa)
    return aa.view(*batch_shape, 3)


def _left_mano_root_fix(R: torch.Tensor) -> torch.Tensor:
    """左手全局旋转：R @ diag(-1, 1, 1)。"""
    *b, _, _ = R.shape
    fix = R.new_tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).view(1, 3, 3).expand(*b, 3, 3)
    return R @ fix


def lowdim_wrist_to_mano_cam(
    trans_w: torch.Tensor,
    rot6: torch.Tensor,
    extrinsic_4x4: torch.Tensor,
    betas: torch.Tensor,
    *,
    is_left: bool,
    apply_left_root_fix: bool = True,
    hand_pose_fill: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Batch 维 (B,T,·)、外参 (B,T,4,4)；返回 trans、root_orient、hand_pose、betas。"""
    R_w2c = extrinsic_4x4[..., :3, :3]
    t_w2c = extrinsic_4x4[..., :3, 3]
    Rw2world = rot6d_to_rotmat(rot6)

    trans_c = torch.einsum("btij,btj->bti", R_w2c, trans_w) + t_w2c

    R_in_cam = torch.matmul(R_w2c, Rw2world)
    if is_left and apply_left_root_fix:
        R_in_cam = _left_mano_root_fix(R_in_cam)
    root_aa = rotmat_to_axis_angle(R_in_cam)

    if hand_pose_fill is None:
        b, t, _ = trans_w.shape
        hand_pose_fill = trans_w.new_zeros(b, t, 45)

    return {
        "trans": trans_c,
        "root_orient": root_aa,
        "hand_pose": hand_pose_fill,
        "betas": betas,
    }
