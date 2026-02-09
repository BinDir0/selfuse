"""
Image feature visualization utilities.
"""

from typing import Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def _infer_grid_size(num_patches: int) -> Tuple[int, int]:
    grid_h = int(round(num_patches ** 0.5))
    if grid_h * grid_h == num_patches:
        return grid_h, grid_h
    # Try common rectangular grids (e.g., 14x16, 16x24)
    for grid_h in range(1, int(num_patches ** 0.5) + 1):
        if num_patches % grid_h == 0:
            return grid_h, num_patches // grid_h
    raise ValueError(f"Cannot infer grid size from num_patches={num_patches}")


def _pca_project_top3(x: torch.Tensor) -> torch.Tensor:
    """
    PCA projection to top-3 components for a batch of images.

    Args:
        x: [B, P, D] tensor

    Returns:
        [B, P, 3] tensor
    """
    if x.ndim != 3:
        raise ValueError(f"Expected [B, P, D] tensor, got shape={x.shape}")
    bsz, p, d = x.shape
    if d < 3:
        out = torch.zeros(bsz, p, 3, device=x.device, dtype=x.dtype)
        out[..., :d] = x
        return out
    x = x - x.mean(dim=1, keepdim=True)
    # Batched SVD. Vh rows are principal directions in feature space.
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    components_t = vh[:, :3, :].transpose(-1, -2)  # [B, D, 3]
    proj = torch.matmul(x, components_t)  # [B, P, 3]
    return proj


def visualize_pca_dino_style(
    image_features: Union[torch.Tensor, np.ndarray],
    grid_size: Optional[Union[int, Tuple[int, int]]] = None,
    normalize: bool = True,
    output_uint8: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    DINOv2-style visualization of image features with PCA.

    Args:
        image_features: [B, P, D] or [P, D] image features.
        grid_size: (H, W) patch grid size. If None, infer from P.
        normalize: min-max normalize each RGB channel.
        output_uint8: return uint8 (0-255) if True, else float in [0, 1].

    Returns:
        Visualized RGB image(s) with shape [B, H, W, 3] or [H, W, 3].
        Type matches input: numpy if input is numpy, else torch.
    """
    is_numpy = isinstance(image_features, np.ndarray)
    feats = torch.from_numpy(image_features) if is_numpy else image_features
    if feats.ndim == 2:
        feats = feats.unsqueeze(0)
    if feats.ndim != 3:
        raise ValueError(f"Expected [B, P, D] or [P, D], got shape={feats.shape}")

    bsz, num_patches, _ = feats.shape
    if grid_size is None:
        grid_h, grid_w = _infer_grid_size(num_patches)
    else:
        if isinstance(grid_size, int):
            grid_h, grid_w = grid_size, grid_size
        else:
            grid_h, grid_w = grid_size
        if grid_h * grid_w != num_patches:
            raise ValueError(
                f"grid_size {grid_size} does not match num_patches={num_patches}"
            )

    proj = _pca_project_top3(feats)  # [B, P, 3]
    if normalize:
        mins = proj.amin(dim=1, keepdim=True)
        maxs = proj.amax(dim=1, keepdim=True)
        proj = (proj - mins) / (maxs - mins + 1e-6)
    proj = proj.reshape(bsz, grid_h, grid_w, 3)
    if output_uint8:
        proj = (proj * 255.0).clamp(0, 255).to(torch.uint8)
    vis = proj
    if vis.shape[0] == 1:
        vis = vis[0]

    if is_numpy:
        return vis.cpu().numpy()
    return vis


def save_pca_dino_style_images(
    pixel_values: Union[torch.Tensor, np.ndarray],
    image_features: Union[torch.Tensor, np.ndarray],
    save_path: Union[str, Path],
    grid_size: Optional[Union[int, Tuple[int, int]]] = None,
    normalize: bool = True,
    output_uint8: bool = True,
):
    """
    Helper: visualize PCA (DINOv2 style) and save image(s) to disk.

    Args:
        pixel_values: [B, C, H, W] or [B, T, C, H, W] pixel values.
        image_features: [B, P, D] or [P, D] image features.
        save_path: Output file path (for single image) or directory (for batch).
        grid_size: (H, W) patch grid size. If None, infer from P.
        normalize: min-max normalize each RGB channel.
        output_uint8: save uint8 (0-255) if True, else float in [0, 1].
    """
    vis = visualize_pca_dino_style(
        image_features=image_features,
        grid_size=grid_size,
        normalize=normalize,
        output_uint8=output_uint8,
    )
    save_path = Path(save_path)
    if isinstance(vis, torch.Tensor):
        vis = vis.cpu().numpy()
    if vis.ndim == 3:
        vis = vis[None, ...]

    pixels = torch.from_numpy(pixel_values) if isinstance(pixel_values, np.ndarray) else pixel_values
    assert pixels.ndim == 4, ValueError(f"Expected pixel_values [B, C, H, W], got {pixels.shape}")
    assert pixels.shape[0] == vis.shape[0], ValueError(f"Batch size mismatch: pixel_values {pixels.shape[0]} vs vis {vis.shape[0]}")

    # Convert pixel_values to [B, H, W, 3] float in [0, 1]
    if pixels.shape[1] == 1:
        pixels = pixels.repeat(1, 3, 1, 1)
    if pixels.shape[1] > 3:
        pixels = pixels[:, :3]
    pixels = pixels.float()
    if pixels.max() > 1.5:
        pixels = pixels / 255.0
    pixels = pixels.clamp(0.0, 1.0).permute(0, 2, 3, 1)
    pixels_np = pixels.cpu().numpy()

    # Ensure vis is float [0, 1] and resize to match pixel size if needed
    vis_float = vis.astype(np.float32)
    if output_uint8:
        vis_float = vis_float / 255.0
    if vis_float.shape[1:3] != pixels_np.shape[1:3]:
        vis_t = torch.from_numpy(vis_float).permute(0, 3, 1, 2)
        vis_t = F.interpolate(vis_t, size=pixels_np.shape[1:3], mode="bilinear", align_corners=False)
        vis_float = vis_t.permute(0, 2, 3, 1).cpu().numpy()

    combined = np.concatenate([pixels_np, vis_float], axis=2)

    if combined.shape[0] == 1 and save_path.suffix:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(str(save_path), combined[0])
        return

    save_path.mkdir(parents=True, exist_ok=True)
    for i in range(combined.shape[0]):
        out_file = save_path / f"pca_vis_{i:04d}.png"
        plt.imsave(str(out_file), combined[i])
