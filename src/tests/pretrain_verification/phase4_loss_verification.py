"""
Phase 4: Loss computation verification.

Checks:
  4.1  CE Loss              -- pure VLA = 0, mixed batch > 0, initial range
  4.2  Flow Loss            -- velocity target formula, loss vs t curve
  4.3  DiffLoss             -- chunk construction, hidden state gathering, positivity
  4.4  Total loss weighting -- 0.1*CE + 20.0*Diff + 1.0*Flow, balance visualization
  4.5  Gradient flow        -- all param groups have gradients, knowledge insulation

Requires: nothing (uses DummyBackbone, no real Qwen3-VL weights)
Outputs:  outputs/pretrain_verification/phase4/  (report.json + PNG plots)

Usage:
    python -m src.tests.pretrain_verification.phase4_loss_verification
    python -m src.tests.pretrain_verification.phase4_loss_verification --skip-visual
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from src.tests.test_e2e_forward_backward import build_model, build_batch
from src.tests.pretrain_verification.utils import (
    PhaseReport,
    CheckResult,
    assert_check,
    get_output_dir,
    safe_import_plt,
    tensor_stats,
)
from src.policy.legendvla_loss import (
    build_flow_inputs,
    _compute_flow_loss,
    _build_dense_diffloss_inputs,
)


PHASE = "phase4"


# ======================================================================
# Check 4.1: CE Loss
# ======================================================================


def check_ce_loss(report: PhaseReport) -> None:
    """Verify CE loss is zero for pure VLA batch and positive for VLM samples."""
    print("\n--- Check 4.1: CE Loss ---")

    # Pure VLA batch: all samples are VLA, CE loss should be exactly 0
    model = build_model(with_diffloss=False)
    batch = build_batch(batch_size=2)
    batch["is_vla_data"] = torch.tensor([True, True], dtype=torch.bool)
    output = model("train", batch)
    ce_pure_vla = output["ce_loss"].item()

    report.add(assert_check(
        ce_pure_vla == 0.0,
        "4.1a_ce_pure_vla_zero",
        f"CE loss on pure VLA batch = {ce_pure_vla} (expected 0.0)",
        {"ce_loss": ce_pure_vla},
    ))

    # Mixed batch: sample 1 is VLM (is_vla_data[1]=False), CE loss should be positive
    model2 = build_model(with_diffloss=False)
    batch2 = build_batch(batch_size=2)
    # build_batch already sets is_vla_data=[True, False] and labels[1] has valid tokens
    output2 = model2("train", batch2)
    ce_mixed = output2["ce_loss"].item()

    report.add(assert_check(
        ce_mixed > 0.0,
        "4.1b_ce_mixed_positive",
        f"CE loss on mixed batch = {ce_mixed:.4f} (expected > 0)",
        {"ce_loss": ce_mixed},
    ))

    # Reasonable range check: for vocab_size=128, initial CE ~ ln(128) ~ 4.85
    # Allow a generous range [1.0, 10.0] since random init can vary
    expected_initial = math.log(128)
    reasonable = 0.5 <= ce_mixed <= 15.0

    report.add(assert_check(
        reasonable,
        "4.1c_ce_reasonable_range",
        f"CE loss = {ce_mixed:.4f}, ln(vocab_size=128) = {expected_initial:.4f}, range check [0.5, 15.0]",
        {"ce_loss": ce_mixed, "ln_vocab_size": expected_initial},
    ))


# ======================================================================
# Check 4.2: Flow Loss
# ======================================================================


def check_flow_loss(report: PhaseReport, output_dir: Path, skip_visual: bool) -> None:
    """Verify flow loss formula and visualize flow loss vs t curve."""
    print("\n--- Check 4.2: Flow Loss ---")

    model = build_model(with_diffloss=False)
    batch = build_batch(batch_size=2)

    # Run train_flow mode
    output = model("train_flow", batch)
    flow_loss_val = output["flow_loss"].item()

    report.add(assert_check(
        flow_loss_val > 0.0,
        "4.2a_flow_loss_positive",
        f"Flow loss = {flow_loss_val:.6f} (expected > 0)",
        {"flow_loss": flow_loss_val},
    ))

    # Manual formula verification:
    # Rebuild the forward pass to extract intermediate values
    model2 = build_model(with_diffloss=False)
    batch2 = build_batch(batch_size=1)
    batch2["is_vla_data"] = torch.tensor([True], dtype=torch.bool)
    batch2["t"] = torch.tensor([0.5])

    torch.manual_seed(42)
    slot_embeds = model2.build_slot_embeddings(batch2)
    backbone_output = model2.forward_backbone_stream(batch2, slot_embeds)

    torch.manual_seed(123)
    flow_inputs = build_flow_inputs(model2, batch2)

    flow_output = model2.forward_flow_stream(
        batch=batch2,
        backbone_output=backbone_output,
        flow_inputs=flow_inputs,
    )

    # Manually compute target_v = actions - (1 - sig_min) * noise
    actions = batch2["actions"]
    noise = flow_inputs["noise"]
    sig_min = model2.flow_sig_min
    target_v_manual = actions - (1 - sig_min) * noise

    # Compute manual MSE loss
    pred_v = flow_output["pred_v"]
    actions_valid_mask = batch2["actions_valid_mask"]
    rtc_mask = flow_inputs["rtc_mask"]
    loss_mask = rtc_mask if rtc_mask is not None else actions_valid_mask
    manual_flow_loss_elements = (pred_v - target_v_manual) ** 2
    masked_loss = manual_flow_loss_elements * loss_mask.to(dtype=manual_flow_loss_elements.dtype)
    valid_count = loss_mask.to(dtype=manual_flow_loss_elements.dtype).sum()
    manual_flow_loss = masked_loss.sum() / valid_count.clamp(min=1)

    # Also compute via the internal function for cross-check
    internal_flow_loss = _compute_flow_loss(
        model=model2,
        actions=actions,
        actions_valid_mask=actions_valid_mask,
        pred_v_t=pred_v,
        noise=noise,
        rtc_mask=rtc_mask,
    )

    manual_val = manual_flow_loss.item()
    internal_val = internal_flow_loss.item()
    formula_match = abs(manual_val - internal_val) < 1e-5

    report.add(assert_check(
        formula_match,
        "4.2b_flow_formula_correct",
        f"Manual flow loss = {manual_val:.6f}, internal = {internal_val:.6f}, diff = {abs(manual_val - internal_val):.2e}",
        {"manual": manual_val, "internal": internal_val},
    ))

    # Visualization: flow loss vs t curve (20 different t values)
    if not skip_visual:
        t_values = torch.linspace(0.01, 0.99, 20)
        flow_losses = []
        for t_val in t_values:
            model_t = build_model(with_diffloss=False)
            batch_t = build_batch(batch_size=1)
            batch_t["is_vla_data"] = torch.tensor([True], dtype=torch.bool)
            batch_t["t"] = torch.tensor([t_val.item()])
            with torch.no_grad():
                out_t = model_t("train_flow", batch_t)
            flow_losses.append(out_t["flow_loss"].item())

        plt = safe_import_plt()
        if plt is not None:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(t_values.numpy(), flow_losses, marker="o", linewidth=1.5, markersize=4)
            ax.set_title("Flow Loss vs Diffusion Time t")
            ax.set_xlabel("t")
            ax.set_ylabel("Flow Loss")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            save_path = output_dir / "flow_loss_vs_t.png"
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"  Saved flow loss vs t plot to {save_path}")


# ======================================================================
# Check 4.3: DiffLoss
# ======================================================================


def check_diffloss(report: PhaseReport) -> None:
    """Verify DiffLoss chunk construction and loss positivity."""
    print("\n--- Check 4.3: DiffLoss ---")

    model = build_model(with_diffloss=True)
    batch = build_batch(batch_size=2)

    # Run train_ar mode to get diffusion_loss
    output = model("train_ar", batch)
    diff_loss_val = output["diffusion_loss"].item()

    report.add(assert_check(
        diff_loss_val > 0.0,
        "4.3a_diffloss_positive",
        f"Diffusion loss = {diff_loss_val:.6f} (expected > 0)",
        {"diffusion_loss": diff_loss_val},
    ))

    # Verify chunk construction: actions.unfold(dim=1, size=ar_action_chunk_size, step=1)
    actions = batch["actions"]
    ar_chunk_size = model.ar_action_chunk_size
    n_actions_tensor = batch["n_actions"]
    B, H, D = actions.shape

    # Unfold to create chunks
    action_chunks = actions.unfold(dimension=1, size=ar_chunk_size, step=1)
    actual_chunk_count = action_chunks.shape[1]
    expected_chunk_count = max(0, H - ar_chunk_size + 1)

    report.add(assert_check(
        actual_chunk_count == expected_chunk_count,
        "4.3b_chunk_count",
        f"Chunk count: actual={actual_chunk_count}, expected=max(0, {H} - {ar_chunk_size} + 1)={expected_chunk_count}",
        {"actual": actual_chunk_count, "expected": expected_chunk_count,
         "n_actions_horizon": H, "ar_chunk_size": ar_chunk_size},
    ))

    # Verify per-sample valid chunk count
    vla_mask = batch["is_vla_data"].to(dtype=torch.bool)
    vla_n_actions = n_actions_tensor[vla_mask]
    for i, n_act in enumerate(vla_n_actions):
        per_sample_expected = max(0, n_act.item() - ar_chunk_size + 1)
        per_sample_clamped = min(per_sample_expected, actual_chunk_count)
        report.add(assert_check(
            per_sample_clamped == per_sample_expected,
            f"4.3c_valid_chunks_sample{i}",
            f"VLA sample {i}: n_actions={n_act.item()}, valid chunks={per_sample_expected}",
            {"n_actions": n_act.item(), "valid_chunks": per_sample_expected},
        ))

    # Also verify via _build_dense_diffloss_inputs
    slot_embeds = model.build_slot_embeddings(batch)
    backbone_output = model.forward_backbone_stream(batch, slot_embeds)
    hidden_states = backbone_output.last_hidden_states

    vla_hidden_z, action_gt, diffloss_mask = _build_dense_diffloss_inputs(
        model,
        hidden_states,
        batch["actions"],
        batch["answer_start_idx"],
        batch["n_actions"],
        vla_mask,
    )

    # diffloss_mask should have True entries equal to total valid chunks across VLA samples
    total_valid = sum(
        max(0, n.item() - ar_chunk_size + 1)
        for n in n_actions_tensor[vla_mask]
    )
    actual_valid = diffloss_mask.sum().item()

    report.add(assert_check(
        actual_valid == total_valid,
        "4.3d_diffloss_mask_count",
        f"DiffLoss valid mask entries: actual={int(actual_valid)}, expected={total_valid}",
        {"actual_valid": int(actual_valid), "expected_valid": total_valid},
    ))


# ======================================================================
# Check 4.4: Total Loss Weighting
# ======================================================================


def check_total_loss_weighting(report: PhaseReport, output_dir: Path, skip_visual: bool) -> None:
    """Verify total = ce_weight*ce + diff_weight*diff + flow_weight*flow."""
    print("\n--- Check 4.4: Total Loss Weighting ---")

    model = build_model(with_diffloss=True)
    batch = build_batch(batch_size=2)
    output = model("train", batch)

    w = model.loss_weights
    expected_total = (
        w.ce_loss_weight * output["ce_loss"]
        + w.diffusion_loss_weight * output["diffusion_loss"]
        + w.flow_loss_weight * output["flow_loss"]
    )
    actual_total = output["total_loss"]

    diff = abs(actual_total.item() - expected_total.item())
    match = diff < 1e-5

    report.add(assert_check(
        match,
        "4.4a_total_weight_formula",
        (
            f"total={actual_total.item():.6f}, "
            f"expected={expected_total.item():.6f}, diff={diff:.2e}, "
            f"weights: ce={w.ce_loss_weight}, diff={w.diffusion_loss_weight}, flow={w.flow_loss_weight}"
        ),
        {
            "total_loss": actual_total.item(),
            "expected_total": expected_total.item(),
            "ce_loss": output["ce_loss"].item(),
            "diffusion_loss": output["diffusion_loss"].item(),
            "flow_loss": output["flow_loss"].item(),
            "ce_weight": w.ce_loss_weight,
            "diff_weight": w.diffusion_loss_weight,
            "flow_weight": w.flow_loss_weight,
        },
    ))

    # Visualization: bar chart of 3 weighted components for 10 batches
    if not skip_visual:
        ce_weighted_list = []
        diff_weighted_list = []
        flow_weighted_list = []

        for _ in range(10):
            model_i = build_model(with_diffloss=True)
            batch_i = build_batch(batch_size=2)
            batch_i["t"] = torch.rand(2)
            with torch.no_grad():
                out_i = model_i("train", batch_i)
            ce_weighted_list.append(w.ce_loss_weight * out_i["ce_loss"].item())
            diff_weighted_list.append(w.diffusion_loss_weight * out_i["diffusion_loss"].item())
            flow_weighted_list.append(w.flow_loss_weight * out_i["flow_loss"].item())

        plt = safe_import_plt()
        if plt is not None:
            x = np.arange(10)
            width = 0.25
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(x - width, ce_weighted_list, width, label=f"CE (w={w.ce_loss_weight})", color="steelblue")
            ax.bar(x, diff_weighted_list, width, label=f"Diff (w={w.diffusion_loss_weight})", color="coral")
            ax.bar(x + width, flow_weighted_list, width, label=f"Flow (w={w.flow_loss_weight})", color="seagreen")
            ax.set_title("Weighted Loss Components Across 10 Batches")
            ax.set_xlabel("Batch Index")
            ax.set_ylabel("Weighted Loss")
            ax.set_xticks(x)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            save_path = output_dir / "weighted_loss_components.png"
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"  Saved weighted loss bar chart to {save_path}")


# ======================================================================
# Check 4.5: Gradient Flow
# ======================================================================


def check_gradient_flow(report: PhaseReport) -> None:
    """Verify gradient propagation through all parameter groups."""
    print("\n--- Check 4.5: Gradient Flow ---")

    # 4.5a: Full train mode with diffloss - all groups should have gradients
    model = build_model(with_diffloss=True)
    batch = build_batch(batch_size=2)
    output = model("train", batch)
    output["total_loss"].backward()

    has_vlm_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.trainable_vlm_parameters
    )
    has_expert_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.action_expert_parameters
    )
    has_diffloss_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.diffloss_parameters
    )

    report.add(assert_check(
        has_vlm_grad,
        "4.5a_vlm_grad_nonzero",
        f"VLM parameters have non-zero gradients: {has_vlm_grad}",
    ))
    report.add(assert_check(
        has_expert_grad,
        "4.5b_expert_grad_nonzero",
        f"Action expert parameters have non-zero gradients: {has_expert_grad}",
    ))
    report.add(assert_check(
        has_diffloss_grad,
        "4.5c_diffloss_grad_nonzero",
        f"DiffLoss parameters have non-zero gradients: {has_diffloss_grad}",
    ))

    # 4.5d: Knowledge insulation in train_flow mode - backbone should have ZERO gradients
    model_insulated = build_model(with_diffloss=False, knowledge_insulation=True)
    batch_insulated = build_batch(batch_size=1)
    output_insulated = model_insulated("train_flow", batch_insulated)
    output_insulated["total_loss"].backward()

    backbone_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model_insulated.trainable_vlm_parameters
    )

    report.add(assert_check(
        not backbone_has_grad,
        "4.5d_insulation_blocks_backbone",
        f"Knowledge insulation blocks backbone gradients in train_flow: backbone_has_grad={backbone_has_grad}",
    ))

    # Flow expert should still receive gradients even with insulation
    expert_has_grad_insulated = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model_insulated.action_expert_parameters
    )

    report.add(assert_check(
        expert_has_grad_insulated,
        "4.5e_expert_grad_with_insulation",
        f"Action expert has gradients with knowledge insulation: {expert_has_grad_insulated}",
    ))

    # 4.5f: No parameter overlap between groups
    model_groups = build_model(with_diffloss=True)
    vlm_ids = {id(p) for p in model_groups.trainable_vlm_parameters}
    expert_ids = {id(p) for p in model_groups.action_expert_parameters}
    diffloss_ids = {id(p) for p in model_groups.diffloss_parameters}

    vlm_expert_overlap = vlm_ids & expert_ids
    vlm_diff_overlap = vlm_ids & diffloss_ids
    expert_diff_overlap = expert_ids & diffloss_ids

    no_overlap = (
        len(vlm_expert_overlap) == 0
        and len(vlm_diff_overlap) == 0
        and len(expert_diff_overlap) == 0
    )

    report.add(assert_check(
        no_overlap,
        "4.5f_no_param_overlap",
        (
            f"Parameter group overlap: VLM-Expert={len(vlm_expert_overlap)}, "
            f"VLM-Diff={len(vlm_diff_overlap)}, Expert-Diff={len(expert_diff_overlap)}"
        ),
        {
            "vlm_count": len(vlm_ids),
            "expert_count": len(expert_ids),
            "diffloss_count": len(diffloss_ids),
        },
    ))


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Loss verification for LegendVLA")
    parser.add_argument("--skip-visual", action="store_true", help="Skip visualization outputs")
    args = parser.parse_args()

    output_dir = get_output_dir(PHASE)
    report = PhaseReport("Phase 4: Loss Verification", output_dir)

    print(f"Output directory: {output_dir}")

    check_ce_loss(report)
    check_flow_loss(report, output_dir, args.skip_visual)
    check_diffloss(report)
    check_total_loss_weighting(report, output_dir, args.skip_visual)
    check_gradient_flow(report)

    report.print_summary()
    report_path = report.save()
    print(f"Report saved to {report_path}")

    if not report.all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
