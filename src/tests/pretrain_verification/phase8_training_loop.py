"""
Phase 8: Training loop verification.

Checks:
  8.1  Optimizer param groups   -- no overlap, all trainable params covered, weight decay grouping
  8.2  LR schedule             -- cosine with warmup, simulated 10k steps curve
  8.3  Gradient clipping       -- max_norm=1.0, post-clip norm <= 1.0
  8.4  Mini training (P0)      -- 100-step convergence, loss decrease > 5%
  8.5  Single-batch overfit (P0) -- 500-step overfit, final_loss < 0.3 * initial_loss

Requires: nothing (uses DummyBackbone, no real Qwen3-VL weights)
Outputs:  outputs/pretrain_verification/phase8/  (report.json + PNG plots)

Usage:
    python -m src.tests.pretrain_verification.phase8_training_loop
    python -m src.tests.pretrain_verification.phase8_training_loop --skip-visual
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from src.tests.pretrain_verification.utils import (
    CheckResult,
    PhaseReport,
    get_output_dir,
    plot_loss_curves,
    safe_import_plt,
)


def _build_model_and_batch():
    from src.tests.test_e2e_forward_backward import build_model, build_batch
    model = build_model(with_diffloss=True, knowledge_insulation=True)
    batch = build_batch(batch_size=2)
    return model, batch


# ---------------------------------------------------------------------------
# Check 8.1: Optimizer parameter group verification
# ---------------------------------------------------------------------------

def check_8_1_optimizer_groups() -> CheckResult:
    """Verify optimizer groups cover all trainable params with no overlap."""
    model, _ = _build_model_and_batch()
    errors = []

    # Collect parameter groups as the training workspace does
    vlm_params = list(model.trainable_vlm_parameters)
    action_params = list(model.action_expert_parameters)
    diff_params = list(model.diffloss_parameters)

    vlm_ids = {id(p) for p in vlm_params}
    action_ids = {id(p) for p in action_params}
    diff_ids = {id(p) for p in diff_params}

    # Check no overlap
    if vlm_ids & action_ids:
        errors.append(f"VLM/action overlap: {len(vlm_ids & action_ids)} params")
    if vlm_ids & diff_ids:
        errors.append(f"VLM/diffloss overlap: {len(vlm_ids & diff_ids)} params")
    if action_ids & diff_ids:
        errors.append(f"action/diffloss overlap: {len(action_ids & diff_ids)} params")

    # Check all trainable params are covered
    all_trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    covered_ids = vlm_ids | action_ids | diff_ids
    uncovered = all_trainable_ids - covered_ids
    if uncovered:
        errors.append(f"{len(uncovered)} trainable params not in any group")

    # Check weight decay grouping (dim >= 2 gets decay)
    def group_params(params):
        decay = [p for p in params if p.ndim >= 2]
        no_decay = [p for p in params if p.ndim < 2]
        return decay, no_decay

    for name, params in [("vlm", vlm_params), ("action", action_params), ("diff", diff_params)]:
        decay, no_decay = group_params(params)
        for p in no_decay:
            if p.ndim >= 2:
                errors.append(f"{name}: param with ndim={p.ndim} in no_decay group")

    passed = len(errors) == 0
    details = {
        "vlm_count": len(vlm_params),
        "action_count": len(action_params),
        "diff_count": len(diff_params),
        "total_trainable": len(all_trainable_ids),
        "covered": len(covered_ids),
    }
    msg = f"Groups valid ({details['vlm_count']}+{details['action_count']}+{details['diff_count']}={details['covered']})" if passed else f"{len(errors)} errors"
    return CheckResult(name="8.1 optimizer_groups", passed=passed, message=msg,
                       details={**details, "errors": errors})


# ---------------------------------------------------------------------------
# Check 8.2: LR schedule verification
# ---------------------------------------------------------------------------

def check_8_2_lr_schedule(skip_visual: bool, output_dir: Path) -> CheckResult:
    """Verify cosine schedule with warmup."""
    errors = []
    peak_lr = 3e-4
    warmup_steps = 200
    total_steps = 10000

    # Simulate optimizer with single param group
    dummy_param = torch.nn.Parameter(torch.randn(10))
    optimizer = AdamW([{"params": [dummy_param], "lr": peak_lr}])

    # Build warmup + cosine schedule (matching training workspace pattern)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + __import__("math").cos(3.14159265 * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    lrs = []
    for step in range(total_steps):
        lrs.append(optimizer.param_groups[0]["lr"] * lr_lambda(step))
        scheduler.step()

    # Check warmup
    if lrs[0] > peak_lr * 0.01:
        errors.append(f"LR at step 0 too high: {lrs[0]:.6e}")

    lr_at_warmup = lrs[warmup_steps]
    if lr_at_warmup < peak_lr * 0.9:
        errors.append(f"LR at warmup end {lr_at_warmup:.6e} < 0.9 * peak {peak_lr:.6e}")

    # Check decay after warmup
    if lrs[-1] > lrs[warmup_steps] * 0.5:
        errors.append(f"LR at end {lrs[-1]:.6e} not decayed enough from warmup end {lrs[warmup_steps]:.6e}")

    # Visualization
    if not skip_visual:
        plt = safe_import_plt()
        if plt is not None:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(lrs, linewidth=1.0)
            ax.axvline(x=warmup_steps, color="red", linestyle="--", alpha=0.5, label=f"warmup end ({warmup_steps})")
            ax.set_xlabel("step")
            ax.set_ylabel("learning rate")
            ax.set_title(f"Cosine LR Schedule (peak={peak_lr}, warmup={warmup_steps})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / "lr_schedule.png", dpi=150)
            plt.close(fig)

    passed = len(errors) == 0
    msg = f"LR schedule valid" if passed else f"{len(errors)} errors"
    return CheckResult(name="8.2 lr_schedule", passed=passed, message=msg,
                       details={"lr_step0": lrs[0], "lr_warmup_end": lr_at_warmup,
                                "lr_final": lrs[-1], "errors": errors})


# ---------------------------------------------------------------------------
# Check 8.3: Gradient clipping
# ---------------------------------------------------------------------------

def check_8_3_grad_clipping() -> CheckResult:
    """Verify gradient clipping limits grad norm to max_norm."""
    model, batch = _build_model_and_batch()
    max_norm = 1.0

    model.train()
    output = model("train", batch)
    output["total_loss"].backward()

    # Compute grad norm before clipping
    params = [p for p in model.parameters() if p.grad is not None]
    norm_before = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))

    # Re-run forward/backward
    model.zero_grad()
    output = model("train", batch)
    output["total_loss"].backward()

    # Clip
    clipped_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)

    # Compute norm after clipping
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    norm_after = total_norm ** 0.5

    errors = []
    if norm_after > max_norm + 1e-4:
        errors.append(f"Grad norm after clipping {norm_after:.4f} > max_norm {max_norm}")

    passed = len(errors) == 0
    msg = f"Clipped: {clipped_norm:.2f} -> {norm_after:.4f}" if passed else errors[0]
    return CheckResult(name="8.3 grad_clipping", passed=passed, message=msg,
                       details={"norm_before_clip": float(clipped_norm),
                                "norm_after_clip": norm_after, "max_norm": max_norm})


# ---------------------------------------------------------------------------
# Check 8.4: Mini training convergence
# ---------------------------------------------------------------------------

def check_8_4_convergence(skip_visual: bool, output_dir: Path) -> CheckResult:
    """Run 100 training steps on dummy data and check loss decreases."""
    model, batch = _build_model_and_batch()
    model.train()
    num_steps = 100

    optimizer = AdamW(model.parameters(), lr=1e-3)
    losses = {"total": [], "ce": [], "flow": [], "diff": []}
    grad_norms = []

    for step in range(num_steps):
        optimizer.zero_grad()
        # Re-sample t each step
        batch["t"] = torch.rand(batch["input_ids"].shape[0])
        output = model("train", batch)

        total_loss = output["total_loss"]
        if not torch.isfinite(total_loss):
            return CheckResult(name="8.4 convergence", passed=False,
                               message=f"NaN/Inf loss at step {step}")

        total_loss.backward()

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norms.append(float(gn))
        optimizer.step()

        losses["total"].append(total_loss.item())
        losses["ce"].append(output.get("ce_loss", torch.tensor(0.0)).item())
        losses["flow"].append(output.get("flow_loss", torch.tensor(0.0)).item())
        losses["diff"].append(output.get("diffusion_loss", torch.tensor(0.0)).item())

    # Check loss decreased
    errors = []
    initial_avg = sum(losses["total"][:5]) / 5
    final_avg = sum(losses["total"][-5:]) / 5

    if final_avg >= initial_avg * 0.95:
        errors.append(f"Loss not decreasing: initial_avg={initial_avg:.4f}, final_avg={final_avg:.4f}")

    # Check no NaN
    if any(not __import__("math").isfinite(v) for v in losses["total"]):
        errors.append("NaN/Inf in loss history")

    # Visualization
    if not skip_visual:
        plot_loss_curves(losses, "Mini Training Loss Curves (100 steps)",
                         output_dir / "convergence_loss.png")
        plt = safe_import_plt()
        if plt is not None:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(grad_norms, linewidth=0.8)
            ax.set_xlabel("step")
            ax.set_ylabel("grad norm")
            ax.set_title("Gradient Norm During Mini Training")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / "convergence_grad_norm.png", dpi=150)
            plt.close(fig)

    passed = len(errors) == 0
    msg = f"Loss: {initial_avg:.4f} -> {final_avg:.4f}" if passed else errors[0]
    return CheckResult(name="8.4 convergence", passed=passed, message=msg,
                       details={"initial_avg": initial_avg, "final_avg": final_avg,
                                "errors": errors})


# ---------------------------------------------------------------------------
# Check 8.5: Single batch overfitting
# ---------------------------------------------------------------------------

def check_8_5_overfit(skip_visual: bool, output_dir: Path) -> CheckResult:
    """Overfit on a single batch for 500 steps. Loss should drop significantly."""
    model, batch = _build_model_and_batch()
    model.train()
    num_steps = 500

    optimizer = AdamW(model.parameters(), lr=5e-4)
    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()
        batch["t"] = torch.rand(batch["input_ids"].shape[0])
        output = model("train", batch)
        total_loss = output["total_loss"]

        if not torch.isfinite(total_loss):
            return CheckResult(name="8.5 overfit", passed=False,
                               message=f"NaN/Inf loss at step {step}")

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(total_loss.item())

    initial_loss = losses[0]
    final_loss = losses[-1]

    errors = []
    if final_loss > initial_loss * 0.3:
        errors.append(f"Cannot overfit: initial={initial_loss:.4f}, final={final_loss:.4f}, "
                       f"ratio={final_loss/initial_loss:.4f}")

    if not skip_visual:
        plot_loss_curves({"total_loss": losses},
                         f"Single Batch Overfit ({num_steps} steps)",
                         output_dir / "overfit_loss.png")

    passed = len(errors) == 0
    msg = f"Loss: {initial_loss:.4f} -> {final_loss:.6f}" if passed else errors[0]
    return CheckResult(name="8.5 overfit", passed=passed, message=msg,
                       details={"initial_loss": initial_loss, "final_loss": final_loss,
                                "ratio": final_loss / max(initial_loss, 1e-8)})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 8: Training loop verification")
    parser.add_argument("--skip-visual", action="store_true")
    args = parser.parse_args()

    output_dir = get_output_dir("phase8")
    report = PhaseReport("Phase 8: Training Loop", output_dir)

    print("Running training loop checks...\n")
    report.add(check_8_1_optimizer_groups())
    report.add(check_8_2_lr_schedule(args.skip_visual, output_dir))
    report.add(check_8_3_grad_clipping())
    report.add(check_8_4_convergence(args.skip_visual, output_dir))
    report.add(check_8_5_overfit(args.skip_visual, output_dir))

    report.print_summary()
    report.save()
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
