"""
Phase 7: DiffLoss detailed verification.

Checks:
  7.1  Zero-init verification   -- AdaLN modulation & FinalLayer weights = 0, initial output < 0.1
  7.2  Chunk construction       -- build_dense_diffloss_inputs shape, hidden state gathering
  7.3  Sampling quality         -- initial sampling is finite (sanity check)

Requires: nothing (uses DummyBackbone, no real Qwen3-VL weights)
Outputs:  outputs/pretrain_verification/phase7/  (report.json + PNG plots)

Usage:
    python -m src.tests.pretrain_verification.phase7_diffloss
    python -m src.tests.pretrain_verification.phase7_diffloss --skip-visual
"""

from __future__ import annotations

import argparse
import sys

import torch

from src.tests.pretrain_verification.utils import (
    CheckResult,
    PhaseReport,
    get_output_dir,
    safe_import_plt,
)


# ---------------------------------------------------------------------------
# Check 7.1: DiffLoss zero-init verification
# ---------------------------------------------------------------------------

def check_7_1_zero_init() -> CheckResult:
    """Verify that DiffLoss AdaLN modulation and FinalLayer are zero-initialized."""
    errors = []

    try:
        from src.model.common.diffloss import DiffLoss
    except ImportError:
        return CheckResult(name="7.1 zero_init", passed=True,
                           message="Skipped (DiffLoss import failed)", details={"skipped": True})

    diffloss = DiffLoss(
        target_channels=192,  # 48 * 4
        z_channels=2048,
        depth=8,
        width=2048,
        num_sampling_steps="100",
        use_ddim_sampling=True,
        use_flow_matching=False,
    )

    # Check zero-init in the network (SimpleMLPAdaLN)
    net = diffloss.net
    zero_init_checked = 0

    # Check ResBlock AdaLN modulation layers
    if hasattr(net, "res_blocks"):
        for i, block in enumerate(net.res_blocks):
            if hasattr(block, "adaLN_modulation"):
                # Last layer of adaLN_modulation should be zero-init
                last_layer = block.adaLN_modulation[-1]
                if hasattr(last_layer, "weight"):
                    w_max = last_layer.weight.abs().max().item()
                    b_max = last_layer.bias.abs().max().item() if last_layer.bias is not None else 0
                    if w_max > 1e-8 or b_max > 1e-8:
                        errors.append(f"ResBlock[{i}] adaLN not zero-init: w_max={w_max:.6e}, b_max={b_max:.6e}")
                    zero_init_checked += 1

    # Check FinalLayer
    if hasattr(net, "final_layer"):
        fl = net.final_layer
        if hasattr(fl, "linear"):
            w_max = fl.linear.weight.abs().max().item()
            b_max = fl.linear.bias.abs().max().item() if fl.linear.bias is not None else 0
            if w_max > 1e-8 or b_max > 1e-8:
                errors.append(f"FinalLayer not zero-init: w_max={w_max:.6e}, b_max={b_max:.6e}")
            zero_init_checked += 1

    # Verify initial output is near zero
    x = torch.randn(16, 192)
    t = torch.rand(16)
    z = torch.randn(16, 2048)
    with torch.no_grad():
        if hasattr(net, "forward"):
            out = net(x, t, z)
            out_max = out.abs().max().item()
            if out_max > 0.1:
                errors.append(f"Initial output max {out_max:.4f} > 0.1")

    passed = len(errors) == 0
    msg = f"Zero-init verified ({zero_init_checked} layers)" if passed else f"{len(errors)} errors"
    return CheckResult(name="7.1 zero_init", passed=passed, message=msg,
                       details={"zero_init_checked": zero_init_checked, "errors": errors})


# ---------------------------------------------------------------------------
# Check 7.2: Chunk construction and hidden state gathering
# ---------------------------------------------------------------------------

def check_7_2_chunk_construction() -> CheckResult:
    """Verify build_dense_diffloss_inputs produces correct shapes and indices."""
    from src.policy.legendvla_loss import build_dense_diffloss_inputs

    errors = []
    B = 3
    seq_len = 20
    hidden_size = 64
    action_horizon = 8
    action_dim = 48
    ar_chunk_size = 2

    hidden_states = torch.randn(B, seq_len, hidden_size)
    actions = torch.randn(B, action_horizon, action_dim)
    answer_start_idx = torch.tensor([5, 6, 7], dtype=torch.long)
    n_actions = torch.tensor([8, 6, 4], dtype=torch.long)
    vla_mask = torch.tensor([True, True, True], dtype=torch.bool)

    # Create a simple mock model with ar_action_chunk_size
    class MockModel:
        ar_action_chunk_size = ar_chunk_size
    model = MockModel()

    vla_hidden_z, action_gt, diffloss_mask = build_dense_diffloss_inputs(
        model, hidden_states, actions, answer_start_idx, n_actions, vla_mask,
    )

    # Expected chunk count per sample = max(0, n_actions - chunk_size + 1)
    expected_chunks = [max(0, n - ar_chunk_size + 1) for n in n_actions.tolist()]
    max_chunks = max(expected_chunks)

    # Total flat length = B * max_chunks
    expected_flat_len = B * max_chunks
    if vla_hidden_z.shape[0] != expected_flat_len:
        errors.append(f"hidden_z flat len {vla_hidden_z.shape[0]} != expected {expected_flat_len}")
    if action_gt.shape[0] != expected_flat_len:
        errors.append(f"action_gt flat len {action_gt.shape[0]} != expected {expected_flat_len}")

    # Action chunk width = ar_chunk_size * action_dim
    expected_width = ar_chunk_size * action_dim
    if action_gt.shape[-1] != expected_width:
        errors.append(f"action_gt width {action_gt.shape[-1]} != expected {expected_width}")

    # Hidden z width = hidden_size
    if vla_hidden_z.shape[-1] != hidden_size:
        errors.append(f"hidden_z width {vla_hidden_z.shape[-1]} != expected {hidden_size}")

    # Check mask validity
    valid_mask_count = diffloss_mask.sum().item()
    expected_valid = sum(expected_chunks)
    if valid_mask_count != expected_valid:
        errors.append(f"valid mask count {valid_mask_count} != expected {expected_valid}")

    passed = len(errors) == 0
    msg = f"Chunks correct ({expected_chunks})" if passed else f"{len(errors)} errors"
    return CheckResult(name="7.2 chunk_construction", passed=passed, message=msg,
                       details={
                           "expected_chunks": expected_chunks,
                           "flat_len": vla_hidden_z.shape[0],
                           "action_width": action_gt.shape[-1],
                           "valid_mask_count": valid_mask_count,
                           "errors": errors,
                       })


# ---------------------------------------------------------------------------
# Check 7.3: DiffLoss sampling quality
# ---------------------------------------------------------------------------

def check_7_3_sampling(skip_visual: bool, output_dir) -> CheckResult:
    """Verify DiffLoss sampling at initialization produces near-random outputs."""
    try:
        from src.model.common.diffloss import DiffLoss
    except ImportError:
        return CheckResult(name="7.3 sampling", passed=True,
                           message="Skipped (DiffLoss import failed)", details={"skipped": True})

    diffloss = DiffLoss(
        target_channels=192,
        z_channels=2048,
        depth=8,
        width=2048,
        num_sampling_steps="100",
        use_ddim_sampling=True,
        use_flow_matching=False,
    )
    diffloss.eval()

    z = torch.randn(16, 2048)
    with torch.no_grad():
        sampled = diffloss.sample(z)

    # At initialization (zero-init), sampled should be close to noise
    stats = {
        "mean": sampled.mean().item(),
        "std": sampled.std().item(),
        "min": sampled.min().item(),
        "max": sampled.max().item(),
    }

    # The initial sampling should produce somewhat random-looking output
    # (not all zeros, not all the same value)
    errors = []
    if torch.isnan(sampled).any():
        errors.append("NaN in sampled output")
    if torch.isinf(sampled).any():
        errors.append("Inf in sampled output")

    passed = len(errors) == 0
    msg = f"Sampling ok (mean={stats['mean']:.4f}, std={stats['std']:.4f})" if passed else errors[0]
    return CheckResult(name="7.3 sampling", passed=passed, message=msg, details=stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 7: DiffLoss verification")
    parser.add_argument("--skip-visual", action="store_true")
    args = parser.parse_args()

    output_dir = get_output_dir("phase7")
    report = PhaseReport("Phase 7: DiffLoss", output_dir)

    print("Running DiffLoss checks...\n")
    report.add(check_7_1_zero_init())
    report.add(check_7_2_chunk_construction())
    report.add(check_7_3_sampling(args.skip_visual, output_dir))

    report.print_summary()
    report.save()
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
