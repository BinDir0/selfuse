"""
Visual Attention Spatial Projection.

Overlay action expert visual attention back onto the original image frames to
see which image regions the policy attends to when generating actions.

Aggregation strategy (based on literature review):
  - Query dimension (32 action positions): mean
  - Head dimension: mean (primary), max (secondary, for comparison)
  - Layer dimension (14 action expert layers): mean (primary), per-layer (secondary)
  - ODE step dimension (10 steps): step 0 (t~=0.1, highest visual attention);
    progression visualization covers steps 0/3/6/9
  - Attention sink handling: renormalize within the visual-token subset so
    the prominent BOS/<|im_start|> sink does not wash out spatial contrast.
    We also report the absolute visual attention share alongside.

References:
  - Abnar & Zuidema, 2020 "Quantifying Attention Flow in Transformers"
  - Caron et al., 2021 "Emerging Properties in Self-Supervised Vision
    Transformers" (DINO)  https://github.com/facebookresearch/dino
  - jacobgil vit-explain https://jacobgil.github.io/deeplearning/vision-transformer-explainability
  - Kim et al., ICLR 2025 "See What You Are Told: Visual Attention Sink in
    Large Multimodal Models"

Qwen3-VL patchification (from src/utils/eval_visualizer.py):
  - temporal_patch_size = 2, patch_size = 16, spatial merge_size = 2
  - video_grid_thw = [T_g, H_g, W_g]; for 6 frames at 384x384 this is
    [3, 12, 12] -> 432 visual tokens total
  - Flat token layout matches the reshape:
      reshape(T_g, H_g//m, W_g//m, m, m, ...)
      .permute(0, 1, 3, 2, 4, ...)
      .reshape(T_g, H_g, W_g, ...)
    i.e. block-major within each temporal group, sub-patch inside each block.

Usage:
  python -m src.tests.full_chain_verification.visual_attention_spatial \
      --config-path src/config/experiment/legendvla_qwen3_vl.yaml \
      --vla-shard '/efs-exp/zengfanlian/datasets/Webdataset/real_world-test/shard-*.tar' \
      --normalizer-path /efs-exp/.../normalizer.pkl \
      --checkpoint-path /efs-exp/.../checkpoints/update_step=30000 \
      --n-samples 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.policy.legendvla_inference import infer_flow_action
from src.tests.full_chain_verification.part3_backbone_prefix_cache import (
    build_model_and_collator,
    load_hydra_config,
)
from src.tests.full_chain_verification.part9_input_ablation import (
    build_vla_dataset,
    clone_batch,
    collate_to_device,
    collect_samples,
)
from src.tests.full_chain_verification.utils import get_output_dir, safe_import_plt
from src.utils.checkpoint_util import (
    load_checkpoint,
    load_model_and_collator_from_saved_config,
)
from src.utils.eval_visualizer import TEMPORAL_PATCH_SIZE, recover_all_frames
from src.utils.visual_attention import (
    aggregate_visual_attention,
    compute_middle_layer_range,
    overlay_attention,
    renormalize_visual_subset,
    reshape_visual_to_grid,
    spatial_entropy,
    stack_visual_attention,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ─────────────────────────────────────────────────────────────────────────────
# Attention extraction (kept here; tightly bound to model.flow_expert)
# ─────────────────────────────────────────────────────────────────────────────

def extract_expert_attention(
    model,
    batch: dict,
    seed: int = 42,
) -> dict:
    """Run one inference pass in eager mode, capturing expert attention.

    Returns a dict with keys:
      - expert_attn: list[list[Tensor|None]]   shape [n_steps][n_layers] -> [B, H, A, kv]
      - generated_actions: Tensor
    """
    expert = model.flow_expert
    original_impl = expert.config._attn_implementation
    expert.config._attn_implementation = "eager"
    try:
        model.eval()
        with torch.no_grad():
            torch.manual_seed(seed)
            result = infer_flow_action(
                model, clone_batch(batch), output_attentions=True,
            )
    finally:
        expert.config._attn_implementation = original_impl
    return {
        "expert_attn": result["expert_attention_weights"],
        "generated_actions": result["generated_actions"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────


def plot_overview(
    frames: list[np.ndarray],
    attn_grid_3x12x12: np.ndarray,
    stats: dict,
    out_path: Path,
    title: str = "",
) -> None:
    """Primary per-sample visualization: original / overlay / renorm overlay.

    Layout: rows = 6 frames, cols = [original, abs overlay, renorm overlay]
    """
    plt = safe_import_plt()
    if plt is None:
        return
    T_g = attn_grid_3x12x12.shape[0]
    n_frames = len(frames)
    abs_grid = attn_grid_3x12x12  # already aggregated
    renorm_grid = renormalize_visual_subset(abs_grid)

    # Compute common dynamic range per column so overlays share contrast.
    fig, axes = plt.subplots(
        n_frames, 3, figsize=(9, 2.5 * n_frames), squeeze=False,
    )
    for i in range(n_frames):
        # Each temporal group covers temporal_patch_size consecutive frames.
        t_group = min(i // TEMPORAL_PATCH_SIZE, T_g - 1)
        attn_abs = abs_grid[t_group]
        attn_re = renorm_grid[t_group]
        orig = frames[i]

        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f"frame {i} (t_group={t_group})", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay_attention(orig, attn_abs))
        axes[i, 1].set_title(f"abs attn (share={stats['visual_share']*100:.2f}%)", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay_attention(orig, attn_re))
        axes[i, 2].set_title("renorm (within visual)", fontsize=9)
        axes[i, 2].axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_layer_progression(
    frame_rgb: np.ndarray,
    vis_attn_grid: torch.Tensor,  # [S, L, H, A, T_g, H_g, W_g]
    t_group: int,
    ode_step: int,
    out_path: Path,
    title: str = "",
) -> None:
    """Per-layer visualization (mean heads, mean action positions), single frame."""
    plt = safe_import_plt()
    if plt is None:
        return
    n_layers = vis_attn_grid.shape[1]
    # Mean over H and A: [L, T_g, H_g, W_g]. Input is a torch.Tensor, so we
    # use torch.nanmean here too (not np.nanmean).
    per_layer = torch.nanmean(vis_attn_grid[ode_step], dim=2)  # mean A
    per_layer = torch.nanmean(per_layer, dim=1)                # mean H
    # per_layer: [L, T_g, H_g, W_g]
    ncols = 5
    nrows = (n_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for l in range(n_layers):
        ax = axes[l // ncols][l % ncols]
        grid = per_layer[l, t_group]
        if torch.isnan(grid).all():
            ax.text(0.5, 0.5, "None", ha="center", va="center")
            ax.axis("off")
            continue
        ax.imshow(overlay_attention(frame_rgb, grid))
        ax.set_title(f"layer {l}", fontsize=9)
        ax.axis("off")
    for l in range(n_layers, nrows * ncols):
        axes[l // ncols][l % ncols].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_ode_progression(
    frame_rgb: np.ndarray,
    vis_attn_grid: torch.Tensor,
    t_group: int,
    out_path: Path,
    title: str = "",
) -> None:
    """Per-ODE-step visualization (mean layers, heads, action positions)."""
    plt = safe_import_plt()
    if plt is None:
        return
    n_steps = vis_attn_grid.shape[0]
    per_step = torch.nanmean(vis_attn_grid, dim=3)  # mean A -> [S, L, H, T_g, H_g, W_g]
    per_step = torch.nanmean(per_step, dim=2)       # mean H -> [S, L, T_g, H_g, W_g]
    per_step = torch.nanmean(per_step, dim=1)       # mean L -> [S, T_g, H_g, W_g]
    ncols = 5
    nrows = (n_steps + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for s in range(n_steps):
        ax = axes[s // ncols][s % ncols]
        grid = per_step[s, t_group]
        ax.imshow(overlay_attention(frame_rgb, grid))
        ax.set_title(f"step {s} (t={(s+1)/n_steps:.2f})", fontsize=9)
        ax.axis("off")
    for s in range(n_steps, nrows * ncols):
        axes[s // ncols][s % ncols].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_head_specialization(
    frame_rgb: np.ndarray,
    vis_attn_grid: np.ndarray,
    t_group: int,
    ode_step: int,
    out_path: Path,
    title: str = "",
) -> None:
    """Per-head visualization at the last non-NaN layer (DINO-style)."""
    plt = safe_import_plt()
    if plt is None:
        return
    # Pick the last non-NaN layer at the chosen ODE step.
    # vis_attn_grid is now a torch.Tensor (from reshape_visual_to_grid), so use
    # torch ops rather than numpy's (deprecated) torch-compat shim.
    step_slice = vis_attn_grid[ode_step]  # [L, H, A, T_g, H_g, W_g]
    L = step_slice.shape[0]
    picked_l = None
    for l in range(L - 1, -1, -1):
        if not torch.isnan(step_slice[l]).all():
            picked_l = l
            break
    if picked_l is None:
        return
    layer_attn = step_slice[picked_l]  # [H, A, T_g, H_g, W_g]
    per_head = torch.nanmean(layer_attn, dim=1)  # [H, T_g, H_g, W_g]
    n_heads = per_head.shape[0]
    ncols = 4
    nrows = (n_heads + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for h in range(n_heads):
        ax = axes[h // ncols][h % ncols]
        grid = per_head[h, t_group]
        ax.imshow(overlay_attention(frame_rgb, grid))
        ax.set_title(f"head {h} (layer {picked_l})", fontsize=9)
        ax.axis("off")
    for h in range(n_heads, nrows * ncols):
        axes[h // ncols][h % ncols].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_agg_comparison(
    frame_rgb: np.ndarray,
    vis_attn_grid: np.ndarray,
    t_group: int,
    ode_step: int,
    out_path: Path,
    *,
    layer_range: tuple[int, int],
    title: str = "",
) -> None:
    """Compare four aggregation strategies on a single frame.

    Args:
        layer_range: (start, end) half-open layer range for the
            middle-layers strategy; required keyword argument.
    """
    plt = safe_import_plt()
    if plt is None:
        return
    strategies = [
        (f"mid layers {layer_range[0]}-{layer_range[1]-1} + mean heads", "middle_layers_mean_heads"),
        ("all layers + mean heads", "mean_layers_heads"),
        ("all layers + max heads", "max_heads"),
        ("last layer + mean heads", "last_layer_mean_heads"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), squeeze=False)
    axes[0, 0].imshow(frame_rgb)
    axes[0, 0].set_title("original", fontsize=10)
    axes[0, 0].axis("off")
    for i, (name, strat) in enumerate(strategies, start=1):
        grid_tg = aggregate_visual_attention(
            vis_attn_grid,
            strategy=strat,
            ode_step=ode_step,
            layer_range=layer_range if strat == "middle_layers_mean_heads" else None,
        )
        grid = renormalize_visual_subset(grid_tg)[t_group]
        axes[0, i].imshow(overlay_attention(frame_rgb, grid))
        axes[0, i].set_title(name, fontsize=10)
        axes[0, i].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main per-sample routine
# ─────────────────────────────────────────────────────────────────────────────

def analyze_sample(
    model,
    collator,
    sample: dict,
    sample_idx: int,
    out_dir: Path,
    middle_layer_range: tuple[int, int],
    instruction: str = "",
) -> dict:
    """Run attention capture and generate all plots for one sample.

    Args:
        middle_layer_range: (start, end) half-open layer range used for the
            primary "middle_layers_mean_heads" aggregation strategy. Derive
            this via `compute_middle_layer_range(num_expert_layers)` once at
            `run()` entry and pass it through.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = collate_to_device(collator, [sample], device)

    # Recover original frames (all 6 assuming T_g=3 temporal groups).
    frames, (T_g, H_g, W_g) = recover_all_frames(
        batch["pixel_values_videos"], batch["video_grid_thw"], sample_idx=0,
    )
    assert T_g == 3 and H_g == 24 and W_g == 24, (
        f"Unexpected grid {(T_g, H_g, W_g)} — script assumes 6 frames at 384x384 "
        f"(24x24 raw patches with patch_size=16, merge_size=2 -> 12x12 tokens)"
    )

    # Capture expert attention.
    cap = extract_expert_attention(model, batch)
    expert_attn = cap["expert_attn"]

    # Identify visual positions in the prefix.
    answer_start_idx = int(batch["answer_start_idx"][0].item())
    input_ids = batch["input_ids"][0, :answer_start_idx]
    backbone = model.backbone
    visual_mask = (input_ids == backbone.video_token_id).cpu().numpy()
    state_mask = (input_ids == backbone.state_token_id).cpu().numpy()
    text_mask = ~(visual_mask | state_mask)
    visual_indices = np.where(visual_mask)[0]
    prefix_len = answer_start_idx

    # Dense tensor [S, L, H, A, n_visual]
    vis_flat = stack_visual_attention(
        expert_attn, prefix_len, visual_indices, sample_idx=0,
    )
    # Reshape visual axis -> [S, L, H, A, T_g, token_H, token_W]
    vis_grid, token_H, token_W = reshape_visual_to_grid(vis_flat, T_g, H_g, W_g)

    # Absolute visual/state/text share (mean across steps, layers, heads, A)
    def _share(mask):
        # Compute per-step/layer total attention share to mask columns.
        # expert_attn structure: list[list[[B,H,A,kv]]]
        totals = []
        for s_step in expert_attn:
            for w in s_step:
                if w is None:
                    continue
                w0 = w[0].float()  # [H, A, kv]
                totals.append(w0[:, :, :prefix_len][:, :, mask].sum(dim=-1).mean().item())
        return float(np.mean(totals)) if totals else 0.0

    visual_share = _share(visual_mask)
    state_share = _share(state_mask)
    text_share = _share(text_mask)

    stats = {
        "sample_idx": sample_idx,
        "instruction": instruction,
        "prefix_len": prefix_len,
        "n_visual": int(visual_mask.sum()),
        "n_state": int(state_mask.sum()),
        "n_text": int(text_mask.sum()),
        "visual_share": visual_share,
        "state_share": state_share,
        "text_share": text_share,
    }

    # Primary aggregated grid for overview plot: middle layers, mean heads,
    # ODE step 0. Middle layers empirically show the clearest spatial focus.
    primary = aggregate_visual_attention(
        vis_grid,
        strategy="middle_layers_mean_heads",
        ode_step=0,
        layer_range=middle_layer_range,
    )  # [T_g, token_H, token_W]
    stats["spatial_entropy_step0_middle"] = spatial_entropy(primary)
    stats["spatial_entropy_step0_middle_per_group"] = [
        spatial_entropy(primary[tg]) for tg in range(T_g)
    ]
    # Also report all-layer aggregation entropy for comparison.
    primary_all = aggregate_visual_attention(
        vis_grid, strategy="mean_layers_heads", ode_step=0,
    )
    stats["spatial_entropy_step0_all_layers"] = spatial_entropy(primary_all)

    # --- Plots ---
    sample_dir = out_dir / f"sample_{sample_idx:02d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    inst_trunc = (instruction[:60] + "...") if len(instruction) > 60 else instruction
    title_prefix = f"sample {sample_idx} | visual_share={visual_share*100:.2f}% | {inst_trunc}"

    plot_overview(
        frames, primary, stats,
        sample_dir / "overview.png",
        title=f"{title_prefix}\n"
              f"(primary: middle layers {middle_layer_range[0]}-{middle_layer_range[1]-1} + mean heads, ODE step 0)",
    )

    # Representative frame for progression plots: frame 2 (middle of T_g=1 group).
    rep_t_group = 1 if T_g >= 2 else 0
    rep_frame_idx = rep_t_group * TEMPORAL_PATCH_SIZE
    rep_frame = frames[rep_frame_idx]

    plot_layer_progression(
        rep_frame, vis_grid, t_group=rep_t_group, ode_step=0,
        out_path=sample_dir / "layer_progression.png",
        title=f"{title_prefix}\nPer-layer attention @ ODE step 0, temporal group {rep_t_group}",
    )
    plot_ode_progression(
        rep_frame, vis_grid, t_group=rep_t_group,
        out_path=sample_dir / "ode_progression.png",
        title=f"{title_prefix}\nAttention evolution across ODE steps (mean over layers/heads), group {rep_t_group}",
    )
    plot_head_specialization(
        rep_frame, vis_grid, t_group=rep_t_group, ode_step=0,
        out_path=sample_dir / "per_head_last_layer.png",
        title=f"{title_prefix}\nPer-head attention @ last non-NaN layer, ODE step 0, group {rep_t_group}",
    )
    plot_agg_comparison(
        rep_frame, vis_grid, t_group=rep_t_group, ode_step=0,
        out_path=sample_dir / "agg_comparison.png",
        layer_range=middle_layer_range,
        title=f"{title_prefix}\nAggregation strategy comparison",
    )

    with open(sample_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats, primary, frames, rep_t_group, rep_frame_idx


# ─────────────────────────────────────────────────────────────────────────────
# Multi-shard sampling
# ─────────────────────────────────────────────────────────────────────────────

def collect_diverse_samples(
    cfg,
    normalizer_path: str,
    shard_pattern: str,
    n_per_shard: int = 1,
    max_shards: int | None = None,
) -> list[tuple[str, dict]]:
    """Sample from each shard separately to get task diversity.

    Returns:
        list of (shard_name, sample_dict) tuples.
    """
    from src.dataset.wds_dataset import expand_shard_patterns
    urls, _ = expand_shard_patterns(shard_pattern)
    if max_shards is not None:
        urls = urls[:max_shards]
    print(f"Found {len(urls)} shards")

    samples: list[tuple[str, dict]] = []
    for shard_url in urls:
        shard_name = Path(shard_url).stem
        try:
            ds = build_vla_dataset(cfg, normalizer_path, shard_url)
            shard_samples = collect_samples(ds, n_per_shard)
            for s in shard_samples:
                samples.append((shard_name, s))
            print(f"  {shard_name}: collected {len(shard_samples)}")
        except Exception as e:
            print(f"  {shard_name}: FAILED ({e})")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Cross-sample summary plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_sample_summary(
    per_sample_data: list[tuple[str, np.ndarray, list[np.ndarray], int]],
    out_path: Path,
) -> None:
    """Grid plot: each row is a sample, cols = [representative frame, middle-layer attn overlay].

    Args:
        per_sample_data: list of (title, primary_grid [T_g,tH,tW], frames_list, rep_frame_idx, rep_t_group)
    """
    plt = safe_import_plt()
    if plt is None:
        return
    n = len(per_sample_data)
    ncols = 3  # original, abs overlay, renorm overlay
    fig, axes = plt.subplots(n, ncols, figsize=(9, 2.8 * n), squeeze=False)
    for i, (title, primary_grid, frames, rep_frame_idx, rep_t_group) in enumerate(per_sample_data):
        frame = frames[rep_frame_idx]
        abs_attn = primary_grid[rep_t_group]
        renorm = renormalize_visual_subset(primary_grid)[rep_t_group]
        axes[i, 0].imshow(frame)
        axes[i, 0].set_title(title[:60], fontsize=9)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(overlay_attention(frame, abs_attn))
        axes[i, 1].set_title("abs (middle layers)", fontsize=9)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(overlay_attention(frame, renorm))
        axes[i, 2].set_title("renorm (within visual)", fontsize=9)
        axes[i, 2].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    config_path: str,
    vla_shard: str,
    normalizer_path: str,
    checkpoint_path: str,
    n_per_shard: int = 1,
    max_shards: int | None = None,
    output_subdir: str = "visual_attention_spatial",
    train_config_path: str | None = None,
) -> None:
    out_dir = get_output_dir(output_subdir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if train_config_path is not None:
        # Load model from the saved run config (matches evaluate.py behavior).
        # Use this when the checkpoint was trained with options that differ
        # from the current code's default experiment config.
        model, collator, cfg = load_model_and_collator_from_saved_config(
            train_config_path, device,
        )
    else:
        cfg = load_hydra_config(config_path)
        model, collator = build_model_and_collator(config_path, device)
    print(f"Loading checkpoint: {checkpoint_path}")
    load_checkpoint(model, checkpoint_path)
    model.eval()

    # Auto-detect action expert depth and derive the middle-layer range used
    # for the primary aggregation strategy. Threaded through analyze_sample
    # as an explicit parameter — no mutable module-level global here.
    try:
        num_expert_layers = int(model.flow_expert.num_layers)
    except AttributeError:
        num_expert_layers = int(len(model.flow_expert.layers))
    middle_layer_range = compute_middle_layer_range(num_expert_layers)
    print(
        f"Action expert: {num_expert_layers} layers, "
        f"middle range {middle_layer_range} (layers "
        f"{middle_layer_range[0]}..{middle_layer_range[1]-1} used for primary aggregation)"
    )

    print(f"Building datasets from shard pattern: {vla_shard}")
    tagged_samples = collect_diverse_samples(
        cfg, normalizer_path, vla_shard,
        n_per_shard=n_per_shard, max_shards=max_shards,
    )
    print(f"Collected {len(tagged_samples)} samples across shards.")

    all_stats = []
    per_sample_viz = []
    for i, (shard_name, sample) in enumerate(tagged_samples):
        instruction = sample.get("instruction", "")
        short_title = f"[{i:02d}] {shard_name}\n{instruction[:60]}"
        print(f"\n[sample {i}] shard={shard_name} instruction={instruction[:80]}")
        stats, primary, frames, rep_t_group, rep_frame_idx = analyze_sample(
            model, collator, sample, sample_idx=i,
            out_dir=out_dir,
            middle_layer_range=middle_layer_range,
            instruction=instruction,
        )
        stats["shard"] = shard_name
        print(
            f"  visual_share={stats['visual_share']*100:.2f}%, "
            f"state_share={stats['state_share']*100:.2f}%, "
            f"text_share={stats['text_share']*100:.2f}%, "
            f"entropy_middle={stats['spatial_entropy_step0_middle']:.4f}, "
            f"entropy_all={stats['spatial_entropy_step0_all_layers']:.4f}"
        )
        all_stats.append(stats)
        per_sample_viz.append((short_title, primary, frames, rep_frame_idx, rep_t_group))

    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    plot_cross_sample_summary(per_sample_viz, out_dir / "summary.png")
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--vla-shard", type=str, required=True)
    parser.add_argument("--normalizer-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--n-per-shard", type=int, default=1,
                        help="Number of samples per shard (for task diversity)")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Limit number of shards to sample from")
    parser.add_argument("--output-subdir", type=str, default="visual_attention_spatial",
                        help="Output subdir under outputs/full_chain_verification/")
    parser.add_argument("--train-config-path", type=str, default=None,
                        help="Path to the training run's .hydra/config.yaml. "
                             "Required when checkpoint was trained with options "
                             "that differ from the code's current default config.")
    args = parser.parse_args()
    run(
        config_path=args.config_path,
        vla_shard=args.vla_shard,
        normalizer_path=args.normalizer_path,
        checkpoint_path=args.checkpoint_path,
        n_per_shard=args.n_per_shard,
        max_shards=args.max_shards,
        output_subdir=args.output_subdir,
        train_config_path=args.train_config_path,
    )
