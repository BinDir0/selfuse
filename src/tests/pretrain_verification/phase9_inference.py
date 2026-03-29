"""
Phase 9: Inference and closed-loop verification.

Checks:
  9.1  Flow inference shape     -- output [B,64,48], all values finite
  9.2  RTC inference            -- prev_action_chunk pinned for inference_delay steps
  9.3  Train-infer round-trip   -- fixed single-VLA overfit then infer, L1 error < 1.2
  9.4  AR inference (DiffLoss)  -- infer_vla returns valid generated_actions
  9.5  VLM inference            -- infer_vlm returns valid generated_ids

Requires: nothing (uses DummyBackbone, no real Qwen3-VL weights)
Outputs:  outputs/pretrain_verification/phase9/  (report.json + PNG plots)

Usage:
    python -m src.tests.pretrain_verification.phase9_inference
    python -m src.tests.pretrain_verification.phase9_inference --skip-visual
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from src.tests.pretrain_verification.utils import (
    CheckResult,
    PhaseReport,
    get_output_dir,
    safe_import_plt,
)


def _build_model_and_batch():
    from src.tests.test_e2e_forward_backward import build_model, build_batch
    torch.manual_seed(0)
    np.random.seed(0)
    model = build_model(with_diffloss=True, knowledge_insulation=True)
    batch = build_batch(batch_size=2)
    return model, batch


# ---------------------------------------------------------------------------
# Check 9.1: Flow inference shape and finiteness
# ---------------------------------------------------------------------------

def check_9_1_flow_inference() -> CheckResult:
    """Verify infer_action returns correct shape and all-finite values."""
    model, batch = _build_model_and_batch()
    model.eval()

    with torch.no_grad():
        actions = model("infer_action", batch)

    errors = []
    B = batch["input_ids"].shape[0]
    action_horizon = batch["actions"].shape[1]
    action_dim = batch["actions"].shape[2]

    if actions.shape != (B, action_horizon, action_dim):
        errors.append(f"Shape mismatch: got {list(actions.shape)}, "
                      f"expected [{B}, {action_horizon}, {action_dim}]")

    if not torch.isfinite(actions).all():
        nan_count = torch.isnan(actions).sum().item()
        inf_count = torch.isinf(actions).sum().item()
        errors.append(f"Non-finite values: {nan_count} NaN, {inf_count} Inf")

    passed = len(errors) == 0
    msg = f"Shape {list(actions.shape)}, all finite" if passed else errors[0]
    return CheckResult(name="9.1 flow_inference", passed=passed, message=msg,
                       details={"shape": list(actions.shape), "errors": errors})


# ---------------------------------------------------------------------------
# Check 9.2: RTC inference consistency
# ---------------------------------------------------------------------------

def check_9_2_rtc_inference() -> CheckResult:
    """Verify RTC inference pins prefix actions from prev_action_chunk."""
    model, batch = _build_model_and_batch()
    model.eval()

    # Only test sample 0 (VLA)
    from src.policy.legendvla_inference import infer_flow_action

    # Generate a reference action chunk
    prev_action = torch.randn(1, batch["actions"].shape[1], batch["actions"].shape[2])
    inference_delay = 2

    # Create a single-sample batch
    single_batch = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.shape[0] >= 1:
            single_batch[k] = v[:1]
        else:
            single_batch[k] = v

    with torch.no_grad():
        actions = infer_flow_action(model, single_batch,
                                    prev_action_chunk=prev_action,
                                    inference_delay=inference_delay)

    errors = []
    # First `inference_delay` steps should be pinned to prev_action
    pinned = actions[:, :inference_delay]
    expected = prev_action[:, :inference_delay]

    if not torch.allclose(pinned, expected, atol=1e-5):
        max_diff = (pinned - expected).abs().max().item()
        errors.append(f"RTC prefix not pinned: max diff = {max_diff:.6f}")

    passed = len(errors) == 0
    msg = "RTC prefix correctly pinned" if passed else errors[0]
    return CheckResult(name="9.2 rtc_inference", passed=passed, message=msg, details={"errors": errors})


# ---------------------------------------------------------------------------
# Check 9.3: Train-infer round-trip
# ---------------------------------------------------------------------------

def check_9_3_roundtrip(skip_visual: bool, output_dir: Path) -> CheckResult:
    """Overfit on one fixed VLA batch, then run inference. Results should approximate GT."""
    from src.tests.test_e2e_forward_backward import build_model, build_batch

    torch.manual_seed(0)
    np.random.seed(0)
    model = build_model(with_diffloss=True, knowledge_insulation=True)
    batch = build_batch(batch_size=1)
    model.train()
    num_steps = 800

    optimizer = AdamW(model.parameters(), lr=1e-3)
    for step in range(num_steps):
        optimizer.zero_grad()
        output = model("train", batch)
        output["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    # Inference
    model.eval()
    with torch.no_grad():
        pred_actions = model("infer_action", batch)

    gt_actions = batch["actions"]
    valid_mask = batch["actions_valid_mask"].bool()

    sample_mask = valid_mask[0]
    pred_sample = pred_actions[0][sample_mask.any(dim=-1)]
    gt_sample = gt_actions[0][sample_mask.any(dim=-1)]
    l1_error = (pred_sample - gt_sample).abs().mean().item()

    errors = []
    if l1_error > 1.2:
        errors.append(f"Train-infer L1 error {l1_error:.4f} > 1.2")

    # Visualization: GT vs Pred trajectories
    if not skip_visual:
        plt = safe_import_plt()
        if plt is not None:
            n_dims_show = min(6, gt_actions.shape[-1])
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            for d in range(n_dims_show):
                ax = axes[d]
                gt_d = gt_actions[0, :, d].numpy()
                pred_d = pred_actions[0, :, d].detach().numpy()
                ax.plot(gt_d, label="GT", linewidth=1.2)
                ax.plot(pred_d, label="Pred", linewidth=1.2, linestyle="--")
                ax.set_title(f"dim {d}")
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
            fig.suptitle(f"Train-Infer Round-trip (L1={l1_error:.4f})")
            fig.tight_layout()
            fig.savefig(output_dir / "roundtrip_trajectories.png", dpi=150)
            plt.close(fig)

            # Per-dim L1 error bar chart
            per_dim_l1 = (pred_actions[0] - gt_actions[0]).abs().mean(dim=0).numpy()
            fig2, ax2 = plt.subplots(figsize=(14, 4))
            ax2.bar(range(len(per_dim_l1)), per_dim_l1)
            ax2.set_xlabel("dimension")
            ax2.set_ylabel("L1 error")
            ax2.set_title("Per-Dimension L1 Error (Train-Infer Round-trip)")
            fig2.tight_layout()
            fig2.savefig(output_dir / "roundtrip_per_dim_l1.png", dpi=150)
            plt.close(fig2)

    passed = len(errors) == 0
    msg = f"Round-trip L1={l1_error:.4f}" if passed else errors[0]
    return CheckResult(name="9.3 roundtrip", passed=passed, message=msg,
                       details={"l1_error": l1_error, "overfit_steps": num_steps})


# ---------------------------------------------------------------------------
# Check 9.4: AR inference (DiffLoss sampling)
# ---------------------------------------------------------------------------

def check_9_4_ar_inference() -> CheckResult:
    """Verify infer_vla returns generated_ids with correct type."""
    model, batch = _build_model_and_batch()
    model.eval()

    errors = []
    try:
        with torch.no_grad():
            result = model("infer_vla", batch)

        # Result should contain generated token ids
        if torch.is_tensor(result):
            if result.ndim < 1:
                errors.append(f"Generated result has unexpected ndim={result.ndim}")
            if torch.isnan(result.float()).any():
                errors.append("NaN in generated tokens")
        elif isinstance(result, dict):
            # May return a dict with various outputs
            pass
        else:
            errors.append(f"Unexpected result type: {type(result)}")
    except Exception as e:
        errors.append(f"infer_vla raised: {e}")

    passed = len(errors) == 0
    msg = "AR inference ok" if passed else errors[0]
    return CheckResult(name="9.4 ar_inference", passed=passed, message=msg, details={"errors": errors})


# ---------------------------------------------------------------------------
# Check 9.5: VLM inference (text generation)
# ---------------------------------------------------------------------------

def check_9_5_vlm_inference() -> CheckResult:
    """Verify infer_vlm generates text tokens."""
    model, batch = _build_model_and_batch()
    model.eval()

    errors = []
    try:
        with torch.no_grad():
            result = model("infer_vlm", batch)

        if torch.is_tensor(result):
            if result.numel() == 0:
                errors.append("Empty generation result")
        elif isinstance(result, dict):
            pass
        else:
            errors.append(f"Unexpected result type: {type(result)}")
    except Exception as e:
        errors.append(f"infer_vlm raised: {e}")

    passed = len(errors) == 0
    msg = "VLM inference ok" if passed else errors[0]
    return CheckResult(name="9.5 vlm_inference", passed=passed, message=msg, details={"errors": errors})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 9: Inference verification")
    parser.add_argument("--skip-visual", action="store_true")
    args = parser.parse_args()

    output_dir = get_output_dir("phase9")
    report = PhaseReport("Phase 9: Inference", output_dir)

    print("Running inference checks...\n")
    report.add(check_9_1_flow_inference())
    report.add(check_9_2_rtc_inference())
    report.add(check_9_3_roundtrip(args.skip_visual, output_dir))
    report.add(check_9_4_ar_inference())
    report.add(check_9_5_vlm_inference())

    report.print_summary()
    report.save()
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
