"""
World Model Visualization: compare predicted DINO features against the frozen
teacher features on future frames.

For each sample we run:
  pred = WorldModelExpert(backbone_kv, action_cond_embeds, future_queries)
  target = FrozenDINOv3Teacher(future_frame_images)

`pred` and `target` share shape `[B, K=1, 576, 1024]` = 24x24 spatial grid of
1024-d DINOv3 features per future frame. We visualize alignment with:

  1. PCA RGB maps (fit on GT, project both) - direct semantic comparison
  2. Per-patch cosine similarity overlay on the future frame
  3. Per-patch MSE overlay on the future frame

and aggregate stats (global MSE, mean cosine, "better than mean" baseline).

References for feature-map visualization:
  - DINOv2 (Oquab et al., 2023) "Learning Robust Visual Features without
    Supervision" uses PCA on spatial features for qualitative figures.
  - DINO (Caron et al., 2021) visualizes the [CLS] attention map for the same
    purpose, but for a patch-level feature map PCA is the standard approach.

Usage:
    python -m src.tests.full_chain_verification.visual_world_model \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml \
        --train-config-path /efs-exp/.../config.yaml \
        --vla-shard '/efs-exp/.../real_world-test/shard-*.tar' \
        --normalizer-path /efs-exp/.../normalizer.pkl \
        --checkpoint-path /efs-exp/.../checkpoints/update_step=30000 \
        --n-per-shard 1 \
        --output-subdir visual_world_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from src.tests.full_chain_verification.part9_input_ablation import (
    build_vla_dataset,
    collate_to_device,
    collect_samples,
)
from src.tests.full_chain_verification.part3_backbone_prefix_cache import (
    build_model_and_collator,
    load_hydra_config,
)
from src.tests.full_chain_verification.utils import get_output_dir, safe_import_plt
from src.tests.full_chain_verification.visual_attention_spatial import (
    collect_diverse_samples,
)
from src.utils.checkpoint_util import (
    load_checkpoint,
    load_model_and_collator_from_saved_config,
)
from src.utils.eval_visualizer import recover_all_frames
from src.utils.visual_attention import overlay_attention

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sample selection: prefer dynamic (motion-rich) samples
# ─────────────────────────────────────────────────────────────────────────────

def score_candidate(collator, sample: dict, device: str) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Score a candidate by pixel L2 between last obs frame and future frame.

    Larger score = more visual change between current observation and the
    future frame that the world model has to predict.

    Returns (score, last_obs_frame, future_frame) or None if sample has no
    future frame.
    """
    batch = collate_to_device(collator, [sample], device)
    if int(batch["n_future_frames"][0].item()) == 0:
        return None
    if batch.get("pixel_values_videos") is None:
        return None
    obs_frames, _ = recover_all_frames(
        batch["pixel_values_videos"],
        batch["video_grid_thw"],
        sample_idx=0,
    )
    last_obs = obs_frames[-1]  # [384, 384, 3] uint8
    future = batch["future_frames"][0, 0].cpu().numpy()  # [384, 384, 3] uint8
    diff = last_obs.astype(np.float32) - future.astype(np.float32)
    pixel_l2 = float(np.sqrt((diff ** 2).mean()))  # RMSE over all pixels
    return pixel_l2, last_obs, future


def collect_dynamic_samples(
    cfg,
    normalizer_path: str,
    shard_pattern: str,
    collator,
    device: str,
    n_candidates_per_shard: int = 10,
    max_shards: int | None = None,
) -> list[tuple[str, dict, float, np.ndarray, np.ndarray]]:
    """Sample from each shard, pick the single most-dynamic sample per shard.

    Returns list of (shard_name, sample, score, last_obs_frame, future_frame).
    """
    from src.dataset.wds_dataset import expand_shard_patterns
    urls, _ = expand_shard_patterns(shard_pattern)
    if max_shards is not None:
        urls = urls[:max_shards]
    print(f"Found {len(urls)} shards; scoring {n_candidates_per_shard} candidates per shard")

    selected: list[tuple[str, dict, float, np.ndarray, np.ndarray]] = []
    for shard_url in urls:
        shard_name = Path(shard_url).stem
        try:
            ds = build_vla_dataset(cfg, normalizer_path, shard_url)
            candidates = collect_samples(ds, n_candidates_per_shard)
        except Exception as e:
            print(f"  {shard_name}: FAILED to load ({e})")
            continue

        scored: list[tuple[dict, float, np.ndarray, np.ndarray]] = []
        for s in candidates:
            try:
                result = score_candidate(collator, s, device)
            except Exception as e:
                print(f"    score failed: {e}")
                continue
            if result is None:
                continue
            score, last_obs, future = result
            scored.append((s, score, last_obs, future))

        if not scored:
            print(f"  {shard_name}: no valid candidates")
            continue

        scored.sort(key=lambda x: x[1], reverse=True)
        best_sample, best_score, best_obs, best_future = scored[0]
        selected.append((shard_name, best_sample, best_score, best_obs, best_future))
        scores_summary = ", ".join(f"{s[1]:.1f}" for s in scored[:5])
        print(
            f"  {shard_name}: best score={best_score:.2f} "
            f"(top 5 of {len(scored)}: [{scores_summary}])"
        )
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Frozen teacher on arbitrary frame (for copy baseline)
# ─────────────────────────────────────────────────────────────────────────────

def run_teacher_on_frame(model, frame_rgb_uint8: np.ndarray) -> np.ndarray:
    """Run the model's frozen DINOv3 teacher on a single RGB frame.

    Args:
        frame_rgb_uint8: [H, W, 3] uint8 numpy array.
    Returns:
        DINO features [N_patches, D] as float32 numpy.
    """
    device = next(model.parameters()).device
    frame_t = torch.from_numpy(frame_rgb_uint8).to(device=device)
    frame_t = frame_t.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, 3]
    with torch.no_grad():
        feats = model.frozen_teacher(frame_t)  # [1, 1, N, D]
    return feats[0, 0].float().cpu().numpy()


def run_world_model(model, batch: dict) -> dict:
    """Forward backbone + world model and return pred/target feature maps.

    Returns a dict with:
      - pred:    [B, K, N, D]  world model predicted DINO features
      - target:  [B, K, N, D]  frozen teacher DINO features
      - future_frames: [B, K, H, W, 3] uint8 original future frames
      - n_future_frames: [B] int
    """
    if not getattr(model, "use_world_model", False):
        raise RuntimeError(
            "Model has no world_model. Cannot run this visualization."
        )

    slot_embeds = model.build_slot_embeddings(batch, add_action_noise=False)
    with torch.no_grad():
        backbone_output = model.forward_backbone_stream(batch, slot_embeds)
        action_cond_embeds = None
        if (
            model.world_model_config.action_conditioning
            and "actions" in batch
        ):
            action_cond_embeds = model.action_encoder(batch["actions"])
        wm_output = model.forward_world_model_stream(
            batch, backbone_output, action_cond_embeds,
        )
    return {
        "pred": wm_output["pred"].float().cpu().numpy(),
        "target": wm_output["target"].float().cpu().numpy(),
        "future_frames": batch["future_frames"].cpu().numpy(),
        "n_future_frames": batch["n_future_frames"].cpu().numpy(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature map processing
# ─────────────────────────────────────────────────────────────────────────────

def features_to_grid(features: np.ndarray, grid_hw: int = 24) -> np.ndarray:
    """Reshape [N=grid_hw^2, D] to [grid_hw, grid_hw, D]."""
    assert features.shape[0] == grid_hw * grid_hw, (
        f"flat dim {features.shape[0]} != {grid_hw}^2={grid_hw*grid_hw}"
    )
    return features.reshape(grid_hw, grid_hw, features.shape[-1])


def pca_np(X: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple PCA via SVD.

    Args:
        X: [N, D] features
    Returns:
        (transformed [N, n_components], components [n_components, D], mean [D])
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    transformed = Xc @ components.T
    return transformed, components, mean


def apply_pca(X: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return (X - mean) @ components.T


def pca_to_rgb(
    features_grid: np.ndarray,
    components: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    norm_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Map a [H, W, D] feature map to a [H, W, 3] RGB image via PCA.

    If `components` / `mean` are provided, they are reused (so pred and GT
    land in the same color space). Otherwise PCA is fit on this feature map.
    Returns (rgb, components, mean, (lo, hi)) so the caller can reuse.
    """
    H, W, D = features_grid.shape
    flat = features_grid.reshape(-1, D)
    if components is None or mean is None:
        proj, components, mean = pca_np(flat, n_components=3)
    else:
        proj = apply_pca(flat, components, mean)
    # Normalize each component independently over the full range of this map.
    if norm_range is None:
        lo, hi = proj.min(axis=0), proj.max(axis=0)
    else:
        lo, hi = norm_range
    rgb = (proj - lo) / (hi - lo + 1e-8)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb.reshape(H, W, 3), components, mean, (lo, hi)


def cosine_map(pred_grid: np.ndarray, target_grid: np.ndarray) -> np.ndarray:
    """Per-patch cosine similarity between pred and target feature maps.

    Args:
        pred_grid, target_grid: [H, W, D]
    Returns:
        [H, W] cosine similarity in [-1, 1]
    """
    pred_n = pred_grid / (np.linalg.norm(pred_grid, axis=-1, keepdims=True) + 1e-8)
    tgt_n = target_grid / (np.linalg.norm(target_grid, axis=-1, keepdims=True) + 1e-8)
    return (pred_n * tgt_n).sum(axis=-1)


def mse_map(pred_grid: np.ndarray, target_grid: np.ndarray) -> np.ndarray:
    """Per-patch MSE between pred and target feature maps."""
    return ((pred_grid - target_grid) ** 2).mean(axis=-1)


def feature_rank_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute aggregate metrics over flat features [N, D].

    - mean_cosine:  average per-patch cosine similarity
    - mse:          mean squared error over all entries
    - mean_baseline_cosine:
        avg cosine between pred and the GT mean feature (should be lower if
        pred carries sample-specific information, not just average)
    """
    pred_n = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
    tgt_n = target / (np.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
    cos_per_patch = (pred_n * tgt_n).sum(axis=-1)  # [N]
    mse = float(((pred - target) ** 2).mean())

    tgt_mean = target.mean(axis=0, keepdims=True)
    tgt_mean_n = tgt_mean / (np.linalg.norm(tgt_mean, axis=-1, keepdims=True) + 1e-8)
    baseline_cos = float((pred_n * tgt_mean_n).sum(axis=-1).mean())

    return {
        "mean_cosine": float(cos_per_patch.mean()),
        "median_cosine": float(np.median(cos_per_patch)),
        "mse": mse,
        "mean_cosine_vs_target_mean": baseline_cos,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def upsample_grid(grid: np.ndarray, target_hw: tuple[int, int] = (384, 384)) -> np.ndarray:
    """Bilinear upsample a small 2D / 3D array to a target spatial size."""
    H, W = target_hw
    if grid.ndim == 2:
        return cv2.resize(grid.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    if grid.ndim == 3:
        out = np.zeros((H, W, grid.shape[-1]), dtype=np.float32)
        for c in range(grid.shape[-1]):
            out[..., c] = cv2.resize(
                grid[..., c].astype(np.float32), (W, H),
                interpolation=cv2.INTER_LINEAR,
            )
        return out
    raise ValueError(f"bad ndim {grid.ndim}")


def plot_sample(
    last_obs_frame: np.ndarray,
    future_frame: np.ndarray,
    pred_grid: np.ndarray,
    target_grid: np.ndarray,
    current_grid: np.ndarray,
    pixel_l2: float,
    out_path: Path,
    title: str,
) -> dict:
    """Generate the per-sample comparison plot with copy-baseline analysis.

    Args:
        last_obs_frame: [H, W, 3] uint8, the most recent observation frame
        future_frame:   [H, W, 3] uint8, the GT future frame (~1s later)
        pred_grid:      [gh, gw, D] WM predicted DINO features
        target_grid:    [gh, gw, D] GT DINO features on future frame
        current_grid:   [gh, gw, D] GT DINO features on *current* (last obs) frame
        pixel_l2:       raw pixel RMSE between last_obs and future (selection score)
    """
    plt = safe_import_plt()
    if plt is None:
        raise RuntimeError("matplotlib not available")

    # Fit PCA on GT target features, reuse basis for pred and current.
    # This makes all three feature maps directly comparable in color space.
    gt_rgb, components, mean, norm_range = pca_to_rgb(target_grid)
    pred_rgb, _, _, _ = pca_to_rgb(
        pred_grid, components=components, mean=mean, norm_range=norm_range,
    )
    current_rgb, _, _, _ = pca_to_rgb(
        current_grid, components=components, mean=mean, norm_range=norm_range,
    )

    # Cosine / MSE maps
    cos_pred_target = cosine_map(pred_grid, target_grid)
    cos_current_target = cosine_map(current_grid, target_grid)
    mse_pred = mse_map(pred_grid, target_grid)

    # Upsample for display
    target_hw = future_frame.shape[:2]
    gt_rgb_up = upsample_grid(gt_rgb, target_hw)
    pred_rgb_up = upsample_grid(pred_rgb, target_hw)
    current_rgb_up = upsample_grid(current_rgb, target_hw)
    cos_pred_up = upsample_grid(cos_pred_target, target_hw)
    cos_current_up = upsample_grid(cos_current_target, target_hw)
    mse_up = upsample_grid(mse_pred, target_hw)

    gt_rgb_u8 = np.clip(gt_rgb_up * 255.0, 0, 255).astype(np.uint8)
    pred_rgb_u8 = np.clip(pred_rgb_up * 255.0, 0, 255).astype(np.uint8)
    current_rgb_u8 = np.clip(current_rgb_up * 255.0, 0, 255).astype(np.uint8)

    # Overlay cos/MSE on the future frame.
    cos_pred_overlay = overlay_attention(future_frame, cos_pred_up, alpha=0.55, cmap_name="jet")
    cos_current_overlay = overlay_attention(future_frame, cos_current_up, alpha=0.55, cmap_name="jet")
    mse_overlay = overlay_attention(future_frame, mse_up, alpha=0.55, cmap_name="jet")

    # Flattened stats
    pred_flat = pred_grid.reshape(-1, pred_grid.shape[-1])
    target_flat = target_grid.reshape(-1, target_grid.shape[-1])
    current_flat = current_grid.reshape(-1, current_grid.shape[-1])

    stats = feature_rank_metrics(pred_flat, target_flat)

    # Copy baseline: how well does just copying current-frame DINO do?
    copy_metrics = feature_rank_metrics(current_flat, target_flat)
    stats["copy_baseline_mean_cosine"] = copy_metrics["mean_cosine"]
    stats["copy_baseline_mse"] = copy_metrics["mse"]

    # How similar is pred to the current frame itself?
    # If this is very high, pred is essentially copying current.
    pred_vs_current = feature_rank_metrics(pred_flat, current_flat)
    stats["pred_vs_current_cosine"] = pred_vs_current["mean_cosine"]

    # Gain of the model over the copy baseline
    stats["gain_over_copy"] = stats["mean_cosine"] - stats["copy_baseline_mean_cosine"]
    stats["pixel_l2_obs_vs_future"] = pixel_l2

    # Layout: 2 rows × 4 cols
    # Row 0:  last_obs | future | current DINO PCA | GT DINO PCA
    # Row 1:  pred DINO PCA (GT basis) | cos(pred,target) | cos(current,target) | mse(pred,target)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(last_obs_frame)
    axes[0, 0].set_title(
        f"last obs frame\n(pixel L2={pixel_l2:.1f})", fontsize=10,
    )
    axes[0, 0].axis("off")

    axes[0, 1].imshow(future_frame)
    axes[0, 1].set_title("future frame (~1s later)", fontsize=10)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(current_rgb_u8)
    axes[0, 2].set_title(
        f"CURRENT DINO (GT basis)\ncos(curr,tgt)={stats['copy_baseline_mean_cosine']:.3f}",
        fontsize=10,
    )
    axes[0, 2].axis("off")

    axes[0, 3].imshow(gt_rgb_u8)
    axes[0, 3].set_title("GT DINO (target)\n— PCA basis origin —", fontsize=10)
    axes[0, 3].axis("off")

    axes[1, 0].imshow(pred_rgb_u8)
    gain = stats["gain_over_copy"]
    axes[1, 0].set_title(
        f"PRED DINO (GT basis)\ncos(pred,tgt)={stats['mean_cosine']:.3f}  "
        f"gain={gain:+.3f}",
        fontsize=10,
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cos_pred_overlay)
    axes[1, 1].set_title(
        f"cos(pred, target) overlay\nμ={stats['mean_cosine']:.3f}", fontsize=10,
    )
    axes[1, 1].axis("off")

    axes[1, 2].imshow(cos_current_overlay)
    axes[1, 2].set_title(
        f"cos(current, target) overlay\n(copy baseline) μ={stats['copy_baseline_mean_cosine']:.3f}",
        fontsize=10,
    )
    axes[1, 2].axis("off")

    axes[1, 3].imshow(mse_overlay)
    axes[1, 3].set_title(
        f"MSE(pred, target)\nμ={stats['mse']:.4f}", fontsize=10,
    )
    axes[1, 3].axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return stats


def plot_summary(
    per_sample: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]],
    out_path: Path,
) -> None:
    """Grid summary: one row per sample, showing current vs pred vs GT DINO PCA.

    Columns: [last_obs | future | current DINO PCA | GT DINO PCA | Pred DINO PCA]
    """
    plt = safe_import_plt()
    if plt is None:
        return
    n = len(per_sample)
    ncols = 5
    fig, axes = plt.subplots(n, ncols, figsize=(17, 3.2 * n), squeeze=False)
    for i, entry in enumerate(per_sample):
        (title, last_obs, future_frame, pred_grid, target_grid, current_grid, stats) = entry
        # Use GT basis for all three feature maps (comparable colors).
        gt_rgb, components, mean, norm_range = pca_to_rgb(target_grid)
        pred_rgb, _, _, _ = pca_to_rgb(
            pred_grid, components=components, mean=mean, norm_range=norm_range,
        )
        current_rgb, _, _, _ = pca_to_rgb(
            current_grid, components=components, mean=mean, norm_range=norm_range,
        )
        target_hw = future_frame.shape[:2]
        gt_rgb_u8 = np.clip(upsample_grid(gt_rgb, target_hw) * 255.0, 0, 255).astype(np.uint8)
        pred_rgb_u8 = np.clip(upsample_grid(pred_rgb, target_hw) * 255.0, 0, 255).astype(np.uint8)
        current_rgb_u8 = np.clip(upsample_grid(current_rgb, target_hw) * 255.0, 0, 255).astype(np.uint8)

        axes[i, 0].imshow(last_obs)
        axes[i, 0].set_title(f"{title[:50]}\npx L2={stats['pixel_l2_obs_vs_future']:.1f}", fontsize=8)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(future_frame)
        axes[i, 1].set_title("future (~1s)", fontsize=8)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(current_rgb_u8)
        axes[i, 2].set_title(
            f"Current DINO\ncopy baseline={stats['copy_baseline_mean_cosine']:.3f}",
            fontsize=8,
        )
        axes[i, 2].axis("off")
        axes[i, 3].imshow(gt_rgb_u8)
        axes[i, 3].set_title("GT DINO (target)", fontsize=8)
        axes[i, 3].axis("off")
        axes[i, 4].imshow(pred_rgb_u8)
        axes[i, 4].set_title(
            f"Pred DINO\ncos={stats['mean_cosine']:.3f}  gain={stats['gain_over_copy']:+.3f}",
            fontsize=8,
        )
        axes[i, 4].axis("off")
    fig.suptitle(
        "World Model: current vs predicted vs GT DINO features (high-motion samples)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

def run(
    config_path: str,
    vla_shard: str,
    normalizer_path: str,
    checkpoint_path: str,
    n_candidates_per_shard: int = 10,
    max_shards: int | None = None,
    output_subdir: str = "visual_world_model",
    train_config_path: str | None = None,
) -> None:
    out_dir = get_output_dir(output_subdir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if train_config_path is not None:
        model, collator, cfg = load_model_and_collator_from_saved_config(
            train_config_path, device,
        )
    else:
        cfg = load_hydra_config(config_path)
        model, collator = build_model_and_collator(config_path, device)

    if not getattr(model, "use_world_model", False):
        raise RuntimeError(
            "This checkpoint's model has no world_model — nothing to visualize. "
            "Use a frozen-future-regression run."
        )

    print(f"Loading checkpoint: {checkpoint_path}")
    load_checkpoint(model, checkpoint_path)
    model.eval()
    print(
        f"World model: action_conditioning={model.world_model_config.action_conditioning}, "
        f"num_future_frames={model.wm_num_future_frames}, "
        f"grid={model.wm_grid_h}x{model.wm_grid_w}, "
        f"upsample={model.wm_upsample_factor}"
    )

    # Selection phase: score candidates by pixel L2, pick top-1 per shard.
    print(f"\n=== Selection phase: picking high-motion samples ===")
    print(f"Shard pattern: {vla_shard}")
    selected = collect_dynamic_samples(
        cfg, normalizer_path, vla_shard, collator, device,
        n_candidates_per_shard=n_candidates_per_shard,
        max_shards=max_shards,
    )
    print(f"\nSelected {len(selected)} samples total.")

    all_stats = []
    per_sample_viz = []
    grid_hw = model.wm_grid_h * model.wm_upsample_factor  # 12 * 2 = 24

    print(f"\n=== Prediction phase ===")
    for i, (shard_name, sample, pixel_l2, last_obs, future_frame) in enumerate(selected):
        instruction = sample.get("instruction", "")
        print(
            f"\n[sample {i}] shard={shard_name} pixel_l2={pixel_l2:.1f} "
            f"instruction={instruction[:70]}"
        )
        batch = collate_to_device(collator, [sample], device)

        out = run_world_model(model, batch)
        pred = out["pred"][0, 0]      # [N=576, D=1024]
        target = out["target"][0, 0]  # [N=576, D=1024]

        # Run frozen teacher on the last observation frame to get the copy
        # baseline DINO features (same teacher as target features).
        current_features = run_teacher_on_frame(model, last_obs)  # [576, 1024]

        pred_grid = features_to_grid(pred, grid_hw=grid_hw)
        target_grid = features_to_grid(target, grid_hw=grid_hw)
        current_grid = features_to_grid(current_features, grid_hw=grid_hw)

        sample_dir = out_dir / f"sample_{i:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        title = f"sample {i} | {shard_name} | {instruction[:50]}"
        stats = plot_sample(
            last_obs_frame=last_obs,
            future_frame=future_frame,
            pred_grid=pred_grid,
            target_grid=target_grid,
            current_grid=current_grid,
            pixel_l2=pixel_l2,
            out_path=sample_dir / "panel.png",
            title=title,
        )
        stats["sample_idx"] = i
        stats["shard"] = shard_name
        stats["instruction"] = instruction
        with open(sample_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        all_stats.append(stats)
        per_sample_viz.append(
            (
                f"[{i:02d}] {shard_name[:40]}",
                last_obs,
                future_frame,
                pred_grid,
                target_grid,
                current_grid,
                stats,
            )
        )

        print(
            f"  cos(pred,tgt) ={stats['mean_cosine']:.4f}  "
            f"cos(curr,tgt)={stats['copy_baseline_mean_cosine']:.4f}  "
            f"gain         ={stats['gain_over_copy']:+.4f}  "
            f"cos(pred,curr)={stats['pred_vs_current_cosine']:.4f}"
        )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    # Cross-sample summary plot
    plot_summary(per_sample_viz, out_dir / "summary.png")

    # Global stats
    def m(key):
        return float(np.mean([s[key] for s in all_stats]))
    print(
        f"\n=== Aggregate ({len(all_stats)} high-motion samples) ===\n"
        f"  cos(pred, target)         = {m('mean_cosine'):.4f}   ← what the model achieves\n"
        f"  cos(current, target)      = {m('copy_baseline_mean_cosine'):.4f}   ← copy-current baseline\n"
        f"  gain over copy baseline   = {m('gain_over_copy'):+.4f}   ← **key metric**\n"
        f"  cos(pred, current)        = {m('pred_vs_current_cosine'):.4f}   ← pred vs current (high = copying)\n"
        f"  cos(pred, target_mean)    = {m('mean_cosine_vs_target_mean'):.4f}   ← naive mean baseline\n"
        f"  mean MSE                  = {m('mse'):.4f}\n"
        f"  mean pixel L2 (obs→future)= {m('pixel_l2_obs_vs_future'):.2f}\n"
        f"Done. Outputs in {out_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--vla-shard", type=str, required=True)
    parser.add_argument("--normalizer-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--train-config-path", type=str, default=None)
    parser.add_argument("--n-candidates-per-shard", type=int, default=10,
                        help="Number of candidates scored per shard; top-1 is selected.")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--output-subdir", type=str, default="visual_world_model_dynamic")
    args = parser.parse_args()
    run(
        config_path=args.config_path,
        vla_shard=args.vla_shard,
        normalizer_path=args.normalizer_path,
        checkpoint_path=args.checkpoint_path,
        train_config_path=args.train_config_path,
        n_candidates_per_shard=args.n_candidates_per_shard,
        max_shards=args.max_shards,
        output_subdir=args.output_subdir,
    )
