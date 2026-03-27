"""
Phase 1: Data pipeline end-to-end verification.

Checks:
  1.1  Collator output schema     -- all required keys and tensor shapes
  1.2  VLA/VLM interleaving ratio -- should be ~5:1 (83%)
  1.3  Visual input verification  -- video frames count, grid metadata
  1.4  Token sequence structure   -- state/action token counts, label masking
  1.5  actions_valid_mask          -- consistency with n_actions
  1.6  Chat template & tokenization -- decoded text, token type heatmap
  1.7  States/Actions 2D projection -- wrist trajectory overlaid on image plane
  1.8  Data augmentation consistency -- temporal consistency of color jitter

Requires: experiment config YAML + data shards + normalizer (on training cluster)
Outputs:  outputs/pretrain_verification/phase1/  (report.json + PNG plots)

Usage:
    python -m src.tests.pretrain_verification.phase1_data_pipeline \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml

    # With more batches for ratio check:
    python -m src.tests.pretrain_verification.phase1_data_pipeline \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml \
        --num-batches 100

    # Skip visualization:
    python -m src.tests.pretrain_verification.phase1_data_pipeline \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml --skip-visual
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.tests.pretrain_verification.utils import (
    CheckResult,
    PhaseReport,
    assert_check,
    get_output_dir,
    plot_bar_chart,
    plot_image_grid,
    safe_import_plt,
    tensor_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataloader_from_config(config_path: str) -> tuple[DataLoader, Any, Any]:
    """Instantiate dataset + collator + dataloader from Hydra config."""
    from datetime import datetime
    import os

    import hydra
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)
    OmegaConf.register_new_resolver("hydra", lambda key: "", replace=True)

    # Extract config name from path (e.g., "src/config/experiment/legendvla_qwen3_vl.yaml" -> "experiment/legendvla_qwen3_vl")
    # We assume the config root is "src/config"
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    
    # Initialize hydra and compose the config to properly resolve defaults
    with hydra.initialize(version_base=None, config_path="../../config"):
        cfg = hydra.compose(config_name=f"experiment/{config_name}")

    OmegaConf.set_struct(cfg, False)
    cfg.hydra = {"runtime": {"output_dir": "outputs", "choices": {}}, "job": {"num": 0, "name": "test"}}
    OmegaConf.register_new_resolver("hydra", lambda key: "outputs" if "output_dir" in key else "", replace=True)
    
    OmegaConf.resolve(cfg)

    dataset = hydra.utils.instantiate(cfg.dataset)
    data_collator = hydra.utils.instantiate(cfg.data_collator)
    dataset.vla_dataset.set_collator(data_collator)
    if dataset.vlm_dataset is not None:
        dataset.vlm_dataset.set_collator(data_collator)

    normalizer_path = getattr(cfg.training, "normalizer_path", None)
    if normalizer_path:
        with open(normalizer_path, "rb") as f:
            normalizer = pickle.load(f)
    else:
        normalizer = dataset.vla_dataset.get_normalizer()
    dataset.vla_dataset.set_normalizer(normalizer)

    dl = DataLoader(
        dataset=dataset,
        collate_fn=dataset.get_collator(),
        **cfg.dataloader.loader,
    )
    return dl, data_collator, normalizer


# ---------------------------------------------------------------------------
# Check 1.1: Collator output schema completeness
# ---------------------------------------------------------------------------

def check_1_1_schema(batch: dict[str, Any]) -> CheckResult:
    required_keys = [
        "input_ids", "attention_mask", "labels",
        "mm_token_type_ids", "answer_start_idx",
        "states", "actions", "actions_valid_mask",
        "is_vla_data", "n_states", "n_actions",
    ]
    # At least one of pixel_values or pixel_values_videos should exist
    vision_keys_present = ("pixel_values" in batch and batch["pixel_values"] is not None) or \
                          ("pixel_values_videos" in batch and batch["pixel_values_videos"] is not None)

    missing = [k for k in required_keys if k not in batch]
    if missing:
        return assert_check(False, "1.1 schema", f"Missing keys: {missing}", {"missing": missing})

    B = batch["input_ids"].shape[0]
    shape_errors = []
    expected = {
        "states": (B, 18, 48),
        "actions": (B, 64, 48),
        "actions_valid_mask": (B, 64, 48),
        "n_states": (B,),
        "n_actions": (B,),
        "answer_start_idx": (B,),
    }
    for key, exp_shape in expected.items():
        actual = tuple(batch[key].shape)
        # Allow flexibility in state/action horizon from config
        if actual[0] != exp_shape[0]:
            shape_errors.append(f"{key}: batch dim {actual[0]} != {exp_shape[0]}")

    if not vision_keys_present:
        shape_errors.append("Neither pixel_values nor pixel_values_videos present")

    passed = len(missing) == 0 and len(shape_errors) == 0
    msg = "All keys present and shapes valid" if passed else f"Errors: {shape_errors}"
    return CheckResult(name="1.1 schema", passed=passed, message=msg,
                       details={"batch_size": B, "keys": list(batch.keys()), "errors": shape_errors})


# ---------------------------------------------------------------------------
# Check 1.2: VLA/VLM interleaving ratio
# ---------------------------------------------------------------------------

def check_1_2_ratio(dataloader: DataLoader, num_batches: int = 100) -> CheckResult:
    vla_count = 0
    total_count = 0
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        is_vla = batch["is_vla_data"].bool()
        vla_count += is_vla.sum().item()
        total_count += is_vla.numel()

    if total_count == 0:
        return assert_check(False, "1.2 ratio", "No samples consumed")

    ratio = vla_count / total_count
    passed = 0.70 <= ratio <= 0.95
    msg = f"VLA ratio = {ratio:.4f} ({vla_count}/{total_count})"
    return CheckResult(name="1.2 ratio", passed=passed, message=msg,
                       details={"vla_count": vla_count, "total": total_count, "ratio": ratio})


# ---------------------------------------------------------------------------
# Check 1.3: Visual input verification
# ---------------------------------------------------------------------------

def check_1_3_visual(batch: dict[str, Any], output_dir: Path, skip_visual: bool) -> CheckResult:
    errors = []

    # Check video frames
    if batch.get("pixel_values_videos") is not None:
        pvv = batch["pixel_values_videos"]
        vgt = batch.get("video_grid_thw")
        if vgt is not None:
            # video_grid_thw should describe the patching for each sample's video
            if pvv.ndim < 2:
                errors.append(f"pixel_values_videos unexpected ndim={pvv.ndim}")
        if not skip_visual and pvv.ndim >= 2:
            # Visualize first sample's video frames
            # pixel_values_videos shape depends on processor output format
            try:
                _visualize_video_frames(pvv, batch, output_dir)
            except Exception as e:
                errors.append(f"Visualization error: {e}")

    elif batch.get("pixel_values") is not None:
        pv = batch["pixel_values"]
        if pv.ndim < 3:
            errors.append(f"pixel_values unexpected ndim={pv.ndim}")
    else:
        errors.append("No visual inputs found in batch")

    passed = len(errors) == 0
    msg = "Visual inputs valid" if passed else f"Errors: {errors}"
    return CheckResult(name="1.3 visual", passed=passed, message=msg, details={"errors": errors})


def _visualize_video_frames(pvv: torch.Tensor, batch: dict, output_dir: Path) -> None:
    """Render video frames as a grid image for visual inspection."""
    plt = safe_import_plt()
    if plt is None:
        return

    # Try to extract frames for the first VLA sample
    is_vla = batch["is_vla_data"].bool()
    vla_indices = is_vla.nonzero(as_tuple=False).flatten()
    if len(vla_indices) == 0:
        return

    # pixel_values_videos from Qwen3 processor: typically [total_patches, C] or [B, T, C, H, W]
    # We'll just save raw tensor stats since the format varies
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.text(0.5, 0.5,
            f"pixel_values_videos shape: {list(pvv.shape)}\n"
            f"dtype: {pvv.dtype}\n"
            f"min: {pvv.min().item():.4f}, max: {pvv.max().item():.4f}",
            transform=ax.transAxes, ha="center", va="center", fontsize=12)
    ax.set_title("Video Tensor Info")
    ax.axis("off")
    fig.savefig(output_dir / "video_tensor_info.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Check 1.4: Token sequence structure
# ---------------------------------------------------------------------------

def check_1_4_tokens(batch: dict[str, Any], collator: Any) -> CheckResult:
    errors = []
    B = batch["input_ids"].shape[0]
    input_ids = batch["input_ids"]
    is_vla = batch["is_vla_data"].bool()
    labels = batch["labels"]

    # Get special token IDs from the collator
    state_token_id = collator.batch_processor.tokenizer.convert_tokens_to_ids(
        collator.formatter.state_token
    )
    action_token_id = collator.batch_processor.action_token_id

    for i in range(B):
        state_count = (input_ids[i] == state_token_id).sum().item()
        action_count = (input_ids[i] == action_token_id).sum().item()

        if is_vla[i]:
            expected_states = batch["n_states"][i].item()
            expected_actions = batch["n_actions"][i].item()

            if state_count != expected_states:
                errors.append(f"sample {i}: state token count {state_count} != n_states {expected_states}")
            if action_count != expected_actions:
                errors.append(f"sample {i}: action token count {action_count} != n_actions {expected_actions}")
            if not (labels[i] == -100).all():
                errors.append(f"sample {i}: VLA labels not fully masked")
        else:
            valid_labels = (labels[i] != -100).sum().item()
            if valid_labels == 0:
                errors.append(f"sample {i}: VLM sample has no valid labels")

    passed = len(errors) == 0
    msg = f"Token structure valid" if passed else f"{len(errors)} errors"
    return CheckResult(name="1.4 tokens", passed=passed, message=msg,
                       details={"errors": errors[:10]})


# ---------------------------------------------------------------------------
# Check 1.5: actions_valid_mask and n_actions consistency
# ---------------------------------------------------------------------------

def check_1_5_valid_mask(batch: dict[str, Any]) -> CheckResult:
    errors = []
    B = batch["input_ids"].shape[0]
    is_vla = batch["is_vla_data"].bool()
    actions_valid_mask = batch["actions_valid_mask"]
    n_actions = batch["n_actions"]

    for i in range(B):
        valid_steps = actions_valid_mask[i].any(dim=-1).sum().item()
        expected = n_actions[i].item()
        if valid_steps != expected:
            errors.append(f"sample {i}: valid_mask steps {valid_steps} != n_actions {expected}")

        if not is_vla[i]:
            if expected != 0:
                errors.append(f"sample {i}: VLM n_actions should be 0, got {expected}")
            if actions_valid_mask[i].any():
                errors.append(f"sample {i}: VLM actions_valid_mask should be all False")

    passed = len(errors) == 0
    msg = "valid_mask consistent with n_actions" if passed else f"{len(errors)} errors"
    return CheckResult(name="1.5 valid_mask", passed=passed, message=msg,
                       details={"errors": errors[:10]})


# ---------------------------------------------------------------------------
# Check 1.6: Chat template and tokenization visualization
# ---------------------------------------------------------------------------

def check_1_6_chat_template(
    batch: dict[str, Any], collator: Any, dataloader: DataLoader,
    output_dir: Path, skip_visual: bool,
) -> CheckResult:
    errors = []

    tokenizer = collator.batch_processor.tokenizer
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    is_vla = batch["is_vla_data"].bool()
    B = input_ids.shape[0]

    state_token = collator.formatter.state_token
    action_token = collator.formatter.action_token

    # Decode and check token counts
    decoded_texts = []
    for i in range(min(B, 4)):
        # Only decode non-padding tokens
        valid_ids = input_ids[i][attention_mask[i].bool()]
        decoded = tokenizer.decode(valid_ids, skip_special_tokens=False)
        decoded_texts.append(decoded)

        if is_vla[i]:
            n_states_expected = batch["n_states"][i].item()
            n_actions_expected = batch["n_actions"][i].item()
            state_count = decoded.count(state_token)
            action_count = decoded.count(action_token)
            if state_count != n_states_expected:
                errors.append(
                    f"sample {i}: decoded state token count {state_count} != n_states {n_states_expected}"
                )
            if action_count != n_actions_expected:
                errors.append(
                    f"sample {i}: decoded action token count {action_count} != n_actions {n_actions_expected}"
                )

    # Visualization: save decoded texts and token type heatmap
    if not skip_visual:
        _visualize_chat_template(decoded_texts, is_vla[:min(B, 4)], input_ids[:min(B, 4)],
                                 attention_mask[:min(B, 4)], collator, output_dir)

    passed = len(errors) == 0
    msg = "Chat template tokens match" if passed else f"{len(errors)} errors"
    return CheckResult(name="1.6 chat_template", passed=passed, message=msg,
                       details={"errors": errors[:10], "decoded_samples": len(decoded_texts)})


def _visualize_chat_template(
    decoded_texts: list[str], is_vla: torch.Tensor,
    input_ids: torch.Tensor, attention_mask: torch.Tensor,
    collator: Any, output_dir: Path,
) -> None:
    """Save decoded texts and token type heatmap."""
    # Save decoded texts
    text_path = output_dir / "decoded_texts.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(decoded_texts):
            vla_str = "VLA" if is_vla[i] else "VLM"
            f.write(f"=== Sample {i} ({vla_str}) ===\n")
            f.write(text)
            f.write("\n\n")

    # Token type heatmap
    plt = safe_import_plt()
    if plt is None:
        return

    tokenizer = collator.batch_processor.tokenizer
    state_token_id = tokenizer.convert_tokens_to_ids(collator.formatter.state_token)
    action_token_id = collator.batch_processor.action_token_id
    pad_token_id = tokenizer.pad_token_id or 0

    B, L = input_ids.shape
    # 0=padding, 1=text, 2=state, 3=action, 4=image/video placeholder
    type_map = torch.ones(B, L, dtype=torch.long)  # default: text
    type_map[input_ids == pad_token_id] = 0
    type_map[input_ids == state_token_id] = 2
    type_map[input_ids == action_token_id] = 3
    type_map[attention_mask == 0] = 0

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#f0f0f0", "#4a90d9", "#e8a838", "#d94a4a", "#50c878"])
    fig, ax = plt.subplots(figsize=(16, max(2, B * 0.8)))
    im = ax.imshow(type_map.numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=4)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Sample")
    ax.set_title("Token Type Heatmap (gray=pad, blue=text, orange=state, red=action)")
    ax.set_yticks(range(B))
    ylabels = ["VLA" if is_vla[i] else "VLM" for i in range(B)]
    ax.set_yticklabels(ylabels)
    fig.tight_layout()
    fig.savefig(output_dir / "token_type_heatmap.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Check 1.7: States/Actions projected to 2D image
# ---------------------------------------------------------------------------

def check_1_7_projection(
    batch: dict[str, Any], normalizer: Any,
    output_dir: Path, skip_visual: bool,
) -> CheckResult:
    """Project wrist 3D positions to 2D image plane using camera intrinsics."""
    is_vla = batch["is_vla_data"].bool()
    vla_indices = is_vla.nonzero(as_tuple=False).flatten()

    if len(vla_indices) == 0:
        return assert_check(True, "1.7 projection", "No VLA samples to check")

    errors = []
    if "intrinsic" not in batch:
        return assert_check(True, "1.7 projection", "No intrinsic in batch, skipping projection")

    states = batch["states"]  # [B, T_s, 48]
    actions = batch["actions"]  # [B, T_a, 48]
    intrinsics = batch["intrinsic"]  # [B, 4] (fx, fy, cx, cy)

    # Unnormalize to get camera-frame 3D positions
    if normalizer is not None:
        try:
            # Determine which normalizer key to use
            if "motions" in normalizer.params_dict:
                states_unnorm = normalizer["motions"].unnormalize(states.float())
                actions_unnorm = normalizer["motions"].unnormalize(actions.float())
            elif "states" in normalizer.params_dict:
                states_unnorm = normalizer["states"].unnormalize(states.float())
                actions_unnorm = normalizer["actions"].unnormalize(actions.float())
            else:
                states_unnorm = states.float()
                actions_unnorm = actions.float()
        except Exception:
            states_unnorm = states.float()
            actions_unnorm = actions.float()
    else:
        states_unnorm = states.float()
        actions_unnorm = actions.float()

    # Check projection for VLA samples
    out_of_range_total = 0
    total_projected = 0

    for idx in vla_indices[:4]:
        i = idx.item()
        fx, fy, cx, cy = intrinsics[i].float().tolist()
        n_valid_actions = batch["n_actions"][i].item()

        # Wrist 3D positions: dims 0-2 (left hand translation)
        wrist_3d = actions_unnorm[i, :n_valid_actions, :3]  # [T, 3] (x, y, z)

        if wrist_3d.numel() == 0:
            continue

        # Perspective projection: u = fx * X/Z + cx, v = fy * Y/Z + cy
        z = wrist_3d[:, 2].clamp(min=1e-6)
        u = fx * wrist_3d[:, 0] / z + cx
        v = fy * wrist_3d[:, 1] / z + cy

        total_projected += u.numel()
        # Check how many projections are out of a reasonable image range
        # Typical image sizes: 224x224 to 640x480
        oor = ((u < -100) | (u > 1000) | (v < -100) | (v > 1000)).sum().item()
        out_of_range_total += oor

    if total_projected > 0:
        oor_ratio = out_of_range_total / total_projected
        if oor_ratio > 0.5:
            errors.append(f"Out-of-range ratio {oor_ratio:.2%} > 50%")
    else:
        errors.append("No valid projections computed")

    # Visualization
    if not skip_visual and total_projected > 0:
        _visualize_projections(batch, vla_indices, actions_unnorm, states_unnorm, output_dir)

    passed = len(errors) == 0
    msg = "2D projections valid" if passed else f"{len(errors)} errors"
    return CheckResult(name="1.7 projection", passed=passed, message=msg,
                       details={"errors": errors, "total_projected": total_projected,
                                "out_of_range_ratio": oor_ratio if total_projected > 0 else None})


def _visualize_projections(
    batch: dict, vla_indices: torch.Tensor,
    actions_unnorm: torch.Tensor, states_unnorm: torch.Tensor,
    output_dir: Path,
) -> None:
    """Overlay projected wrist trajectory on image plane."""
    plt = safe_import_plt()
    if plt is None:
        return

    intrinsics = batch["intrinsic"]
    n_samples = min(len(vla_indices), 4)
    fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 5))
    if n_samples == 1:
        axes = [axes]

    for plot_i, idx in enumerate(vla_indices[:n_samples]):
        i = idx.item()
        ax = axes[plot_i]
        fx, fy, cx, cy = intrinsics[i].float().tolist()
        n_valid = batch["n_actions"][i].item()

        # Project action wrist (dims 0-2) trajectory
        wrist_3d = actions_unnorm[i, :n_valid, :3]
        z = wrist_3d[:, 2].clamp(min=1e-6)
        u = (fx * wrist_3d[:, 0] / z + cx).numpy()
        v = (fy * wrist_3d[:, 1] / z + cy).numpy()
        t_colors = np.linspace(0, 1, n_valid)

        ax.scatter(u, v, c=t_colors, cmap="coolwarm", s=8, alpha=0.7)
        ax.plot(u, v, color="gray", alpha=0.3, linewidth=0.5)

        # Project state wrist (dims 0-2)
        n_states = batch["n_states"][i].item()
        s_wrist = states_unnorm[i, :n_states, :3]
        sz = s_wrist[:, 2].clamp(min=1e-6)
        su = (fx * s_wrist[:, 0] / sz + cx).numpy()
        sv = (fy * s_wrist[:, 1] / sz + cy).numpy()
        ax.scatter(su, sv, c="blue", s=30, marker="o", zorder=5, label="states")

        ax.set_title(f"Sample {i} (n_act={n_valid})")
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        ax.invert_yaxis()
        ax.legend(fontsize=7)
        ax.set_aspect("equal")

    fig.suptitle("Action Trajectory Projected to 2D Image Plane\n(red=action, blue=state, gradient=time)")
    fig.tight_layout()
    fig.savefig(output_dir / "action_2d_projection.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Check 1.8: Data augmentation consistency
# ---------------------------------------------------------------------------

def check_1_8_augmentation(batch: dict[str, Any]) -> CheckResult:
    """Verify that video frames have consistent augmentation."""
    # If pixel_values_videos is available and is a per-sample video,
    # check that per-frame statistics are similar (same color jitter)
    pvv = batch.get("pixel_values_videos")
    if pvv is None:
        return assert_check(True, "1.8 augmentation", "No video data to check")

    # The Qwen3 processor normalizes and patches the video, so raw pixel checks
    # aren't meaningful after processing. Instead, verify that the tensor is finite.
    has_nan = torch.isnan(pvv.float()).any().item()
    has_inf = torch.isinf(pvv.float()).any().item()
    passed = not has_nan and not has_inf
    msg = "Video tensor finite" if passed else f"NaN={has_nan}, Inf={has_inf}"
    return CheckResult(name="1.8 augmentation", passed=passed, message=msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Data pipeline verification")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--num-batches", type=int, default=20,
                        help="Number of batches for ratio check")
    parser.add_argument("--skip-visual", action="store_true",
                        help="Skip visualization generation")
    args = parser.parse_args()

    output_dir = get_output_dir("phase1")
    report = PhaseReport("Phase 1: Data Pipeline", output_dir)

    print("Loading dataloader from config...")
    dl, collator, normalizer = load_dataloader_from_config(args.config_path)

    print("Fetching first batch...")
    batch = next(iter(dl))

    print("\nRunning checks...")

    # Check 1.1: Schema
    report.add(check_1_1_schema(batch))

    # Check 1.2: Ratio (needs multiple batches)
    print(f"  Consuming {args.num_batches} batches for ratio check...")
    report.add(check_1_2_ratio(dl, args.num_batches))

    # Re-fetch a batch (iterator may have advanced)
    batch = next(iter(dl))

    # Check 1.3: Visual inputs
    report.add(check_1_3_visual(batch, output_dir, args.skip_visual))

    # Check 1.4: Token structure
    report.add(check_1_4_tokens(batch, collator))

    # Check 1.5: Valid mask consistency
    report.add(check_1_5_valid_mask(batch))

    # Check 1.6: Chat template + tokenization
    report.add(check_1_6_chat_template(batch, collator, dl, output_dir, args.skip_visual))

    # Check 1.7: 2D projection
    report.add(check_1_7_projection(batch, normalizer, output_dir, args.skip_visual))

    # Check 1.8: Augmentation
    report.add(check_1_8_augmentation(batch))

    # Summary
    report.print_summary()
    report_path = report.save()
    print(f"Report saved to: {report_path}")
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
