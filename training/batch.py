from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from dataloader import lowdim_wrist_to_mano_cam
from dataloader.mano_pca import hand_pca_bt_to_axisang45


def to_float_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.float().to(device, non_blocking=True)
    else:
        t = torch.from_numpy(x).float().to(device, non_blocking=True)
    return t


def wds_batch_to_training_batch(
    batch: dict[str, Any],
    *,
    device: torch.device,
    image_size: int,
    image_scale: float = 1.0 / 255.0,
    apply_left_root_fix: bool = False,
    mano_pca_layers: tuple[torch.nn.Module, torch.nn.Module] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
]:
    video = to_float_tensor(batch["video"], device)
    if video.dim() == 5 and video.max() > 1.5:
        video = video * image_scale
    b, t, c, h, w = video.shape
    intr_bt = to_float_tensor(batch["intrinsic"], device)
    if intr_bt.dim() != 3 or intr_bt.shape[-1] != 4:
        raise ValueError(f"expected batch['intrinsic'] (B,T,4) fx,fy,cx,cy, got {tuple(intr_bt.shape)}")
    if (h, w) != (image_size, image_size):
        sx = image_size / max(w, 1)
        sy = image_size / max(h, 1)
        intr_bt = intr_bt.clone()
        intr_bt[..., 0] *= sx
        intr_bt[..., 1] *= sy
        intr_bt[..., 2] *= sx
        intr_bt[..., 3] *= sy
        video = F.interpolate(
            video.flatten(0, 1),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).view(b, t, c, image_size, image_size)

    existence = to_float_tensor(batch["existence"], device)
    e4 = to_float_tensor(batch["extrinsic_4x4"], device)

    hand_l_fill = None
    hand_r_fill = None
    if mano_pca_layers is not None:
        ml, mr = mano_pca_layers
        hand_l_fill = hand_pca_bt_to_axisang45(
            to_float_tensor(batch["left_hand_pose45"], device), ml
        )
        hand_r_fill = hand_pca_bt_to_axisang45(
            to_float_tensor(batch["right_hand_pose45"], device), mr
        )

    mano_l = lowdim_wrist_to_mano_cam(
        to_float_tensor(batch["left_translation"], device),
        to_float_tensor(batch["left_rot6"], device),
        e4,
        to_float_tensor(batch["left_shape"], device),
        is_left=True,
        apply_left_root_fix=apply_left_root_fix,
        hand_pose_fill=hand_l_fill,
    )
    mano_r = lowdim_wrist_to_mano_cam(
        to_float_tensor(batch["right_translation"], device),
        to_float_tensor(batch["right_rot6"], device),
        e4,
        to_float_tensor(batch["right_shape"], device),
        is_left=False,
        apply_left_root_fix=False,
        hand_pose_fill=hand_r_fill,
    )
    return video, existence, mano_l, mano_r, intr_bt
