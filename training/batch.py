from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from dataloader import lowdim_wrist_to_mano_cam
from dataloader.mano_pca import hand_pca_bt_to_axisang45
from dataloader.mano_pca import mano_parameter_dict_to_joints_bt

try:
    import kornia.color as kcolor
except Exception:
    kcolor = None


def to_float_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.float().to(device, non_blocking=True)
    else:
        t = torch.from_numpy(x).float().to(device, non_blocking=True)
    return t


def _sample_uniform(
    shape: tuple[int, ...],
    low: float,
    high: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(shape, device=device, dtype=dtype).uniform_(float(low), float(high))


def _apply_color_augmentations(
    video: torch.Tensor,
    *,
    enable: bool,
    color_temp_prob: float,
    color_temp_strength: float,
    contrast_prob: float,
    contrast_range: tuple[float, float],
    saturation_prob: float,
    saturation_range: tuple[float, float],
    grayscale_prob: float,
) -> torch.Tensor:
    if not enable:
        return video
    if kcolor is None:
        raise RuntimeError("kornia is required for augmentation. Install it before training.")
    if video.dim() != 5:
        return video

    b, t, c, h, w = video.shape
    x = video

    # Per-window factors (shared across temporal dimension) reduce temporal flicker.
    fac_shape = (b, 1, 1, 1, 1)

    if color_temp_prob > 0.0 and color_temp_strength > 0.0:
        apply_m = (torch.rand((b, 1, 1, 1, 1), device=x.device) < float(color_temp_prob)).to(x.dtype)
        delta = _sample_uniform(fac_shape, -float(color_temp_strength), float(color_temp_strength), device=x.device, dtype=x.dtype)
        r_gain = 1.0 + delta
        g_gain = torch.ones_like(delta)
        b_gain = 1.0 - delta
        gains = torch.cat([r_gain, g_gain, b_gain], dim=2)
        x = x * (1.0 + apply_m * (gains - 1.0))

    if contrast_prob > 0.0:
        apply_m = (torch.rand((b, 1, 1, 1, 1), device=x.device) < float(contrast_prob)).to(x.dtype)
        cmin, cmax = float(contrast_range[0]), float(contrast_range[1])
        fac = _sample_uniform(fac_shape, cmin, cmax, device=x.device, dtype=x.dtype)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        x_contrast = (x - mean) * fac + mean
        x = x + apply_m * (x_contrast - x)

    if saturation_prob > 0.0:
        apply_m = (torch.rand((b, 1, 1, 1, 1), device=x.device) < float(saturation_prob)).to(x.dtype)
        smin, smax = float(saturation_range[0]), float(saturation_range[1])
        fac = _sample_uniform(fac_shape, smin, smax, device=x.device, dtype=x.dtype)
        x_btchw = x.flatten(0, 1)
        gray = kcolor.rgb_to_grayscale(x_btchw)
        gray = gray.view(b, t, 1, h, w)
        x_sat = gray + fac * (x - gray)
        x = x + apply_m * (x_sat - x)

    if grayscale_prob > 0.0:
        apply_m = (torch.rand((b, 1, 1, 1, 1), device=x.device) < float(grayscale_prob)).to(x.dtype)
        x_btchw = x.flatten(0, 1)
        gray = kcolor.rgb_to_grayscale(x_btchw).view(b, t, 1, h, w)
        gray3 = gray.repeat(1, 1, 3, 1, 1)
        x = x + apply_m * (gray3 - x)

    return x.clamp(0.0, 1.0)


def _apply_isotropic_scale(
    video: torch.Tensor,
    intr_bt: torch.Tensor,
    *,
    enable: bool,
    prob: float,
    scale_min: float,
    scale_max: float,
    pad_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not enable or prob <= 0.0 or video.dim() != 5:
        return video, intr_bt

    b, t, c, h, w = video.shape
    out = video
    intr = intr_bt.clone()
    for bi in range(b):
        if float(torch.rand((), device=video.device)) >= float(prob):
            continue
        s = float(torch.empty((), device=video.device).uniform_(float(scale_min), float(scale_max)).item())
        sh = max(1, int(round(h * s)))
        sw = max(1, int(round(w * s)))

        x = out[bi : bi + 1].flatten(0, 1)
        x = F.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
        x = x.view(1, t, c, sh, sw)

        if s >= 1.0:
            top = max(0, (sh - h) // 2)
            left = max(0, (sw - w) // 2)
            x = x[..., top : top + h, left : left + w]
            shift_x = -float(left)
            shift_y = -float(top)
        else:
            pad_h = max(0, h - sh)
            pad_w = max(0, w - sw)
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            if pad_mode == "reflect" and (sh > 1 and sw > 1):
                x = F.pad(x.flatten(0, 1), (left, right, top, bottom), mode="reflect").view(1, t, c, h, w)
            elif pad_mode == "constant_zero":
                x = F.pad(x.flatten(0, 1), (left, right, top, bottom), mode="constant", value=0.0).view(1, t, c, h, w)
            else:
                fill = float(video[bi].mean().item())
                x = F.pad(x.flatten(0, 1), (left, right, top, bottom), mode="constant", value=fill).view(1, t, c, h, w)
            shift_x = float(left)
            shift_y = float(top)

        out[bi] = x[0]
        intr[bi, :, 0] = intr[bi, :, 0] * s
        intr[bi, :, 1] = intr[bi, :, 1] * s
        intr[bi, :, 2] = intr[bi, :, 2] * s + shift_x
        intr[bi, :, 3] = intr[bi, :, 3] * s + shift_y

    return out, intr


def _update_existence_from_visibility(
    existence: torch.Tensor,
    mano_l: dict[str, torch.Tensor],
    mano_r: dict[str, torch.Tensor],
    intr_bt: torch.Tensor,
    *,
    image_h: int,
    image_w: int,
    mano_pca_layers: tuple[torch.nn.Module, torch.nn.Module] | None,
    invisible_joint_threshold: int,
    z_min: float = 1e-6,
    chunk: int = 512,
) -> torch.Tensor:
    if mano_pca_layers is None:
        return existence
    thr = max(0, min(21, int(invisible_joint_threshold)))
    if thr <= 0:
        return existence

    ml, mr = mano_pca_layers
    jl = mano_parameter_dict_to_joints_bt(
        mano_l["trans"],
        mano_l["root_orient"],
        mano_l["hand_pose"],
        mano_l["betas"],
        ml,
        chunk=chunk,
    )
    jr = mano_parameter_dict_to_joints_bt(
        mano_r["trans"],
        mano_r["root_orient"],
        mano_r["hand_pose"],
        mano_r["betas"],
        mr,
        chunk=chunk,
    )

    def _vis_count(j: torch.Tensor) -> torch.Tensor:
        z = j[..., 2]
        valid_z = z > float(z_min)
        fx = intr_bt[..., 0].unsqueeze(-1)
        fy = intr_bt[..., 1].unsqueeze(-1)
        cx = intr_bt[..., 2].unsqueeze(-1)
        cy = intr_bt[..., 3].unsqueeze(-1)
        u = fx * (j[..., 0] / z.clamp_min(float(z_min))) + cx
        v = fy * (j[..., 1] / z.clamp_min(float(z_min))) + cy
        in_frame = (u >= 0.0) & (u <= float(image_w - 1)) & (v >= 0.0) & (v <= float(image_h - 1))
        vis = valid_z & in_frame & torch.isfinite(u) & torch.isfinite(v)
        return vis.sum(dim=-1)

    vis_l = _vis_count(jl)
    vis_r = _vis_count(jr)
    inv_l = 21 - vis_l
    inv_r = 21 - vis_r

    ex = existence.clone()
    ex_l = ex[..., 0] > 0.5
    ex_r = ex[..., 1] > 0.5
    ex[..., 0] = torch.where(ex_l & (inv_l >= thr), ex[..., 0].new_zeros(()), ex[..., 0])
    ex[..., 1] = torch.where(ex_r & (inv_r >= thr), ex[..., 1].new_zeros(()), ex[..., 1])
    return ex


def wds_batch_to_training_batch(
    batch: dict[str, Any],
    *,
    device: torch.device,
    image_size: int,
    image_scale: float = 1.0 / 255.0,
    apply_left_root_fix: bool = False,
    mano_pca_layers: tuple[torch.nn.Module, torch.nn.Module] | None = None,
    augment_config: dict[str, Any] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
]:
    aug = dict(augment_config or {})
    aug_enable = bool(aug.get("enable", False))

    video = to_float_tensor(batch["video"], device)
    if video.dim() == 5 and video.max() > 1.5:
        video = video * image_scale
    b, t, c, h, w = video.shape
    intr_bt = to_float_tensor(batch["intrinsic"], device)
    if intr_bt.dim() != 3 or intr_bt.shape[-1] != 4:
        raise ValueError(f"expected batch['intrinsic'] (B,T,4) fx,fy,cx,cy, got {tuple(intr_bt.shape)}")

    video = _apply_color_augmentations(
        video,
        enable=aug_enable,
        color_temp_prob=float(aug.get("color_temp_prob", 0.0)),
        color_temp_strength=float(aug.get("color_temp_strength", 0.0)),
        contrast_prob=float(aug.get("contrast_prob", 0.0)),
        contrast_range=(float(aug.get("contrast_min", 1.0)), float(aug.get("contrast_max", 1.0))),
        saturation_prob=float(aug.get("saturation_prob", 0.0)),
        saturation_range=(float(aug.get("saturation_min", 1.0)), float(aug.get("saturation_max", 1.0))),
        grayscale_prob=float(aug.get("grayscale_prob", 0.0)),
    )
    video, intr_bt = _apply_isotropic_scale(
        video,
        intr_bt,
        enable=aug_enable,
        prob=float(aug.get("scale_prob", 0.0)),
        scale_min=float(aug.get("scale_min", 1.0)),
        scale_max=float(aug.get("scale_max", 1.0)),
        pad_mode=str(aug.get("scale_pad_mode", "constant_mean")),
    )

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

    if aug_enable and bool(aug.get("update_existence_from_visibility", True)):
        existence = _update_existence_from_visibility(
            existence,
            mano_l,
            mano_r,
            intr_bt,
            image_h=video.shape[-2],
            image_w=video.shape[-1],
            mano_pca_layers=mano_pca_layers,
            invisible_joint_threshold=int(aug.get("invisible_joint_threshold", 20)),
            chunk=int(aug.get("visibility_joint_chunk", 512)),
        )

    return video, existence, mano_l, mano_r, intr_bt
