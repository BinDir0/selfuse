"""
Phase 5 & 6: RTC (Recurrent Time Conditioning) and Flow Matching verification.

Checks:
  5.1  RTC delay sampling distribution -- exp strategy, P(delay=0) > 2*P(delay=7)
  5.2  Prefix/Postfix mask correctness -- forced_delay deterministic check
  5.3  RTC noisy actions               -- prefix = clean actions, postfix = psi_t interpolated
  6.1  psi_t interpolation             -- t=0 -> noise, t=1 -> target, t=0.5 -> midpoint
  6.2  Velocity target                 -- target_v = actions - (1-sig_min)*noise
  6.3  Beta time sampling              -- E[t] ~ 0.40 (biased toward small t)
  6.4  Euler integration closed-loop   -- oracle velocity, 10-step recovery, error < 0.05

Requires: nothing (pure math checks, no model weights needed)
Outputs:  outputs/pretrain_verification/phase56/  (report.json + PNG plots)

Usage:
    python -m src.tests.pretrain_verification.phase5_rtc_and_phase6_flow
    python -m src.tests.pretrain_verification.phase5_rtc_and_phase6_flow --skip-visual
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.policy.legendvla_loss import (
    build_rtc_flow_inputs,
    compute_packed_flow_loss,
    build_flow_inputs,
    psi_t,
)
from src.utils.sample_utils import sample_rtc_delay
from src.tests.pretrain_verification.utils import (
    CheckResult,
    PhaseReport,
    assert_check,
    get_output_dir,
    safe_import_plt,
)

# Re-implement sample_fm_time locally so we don't import the heavy hooks module.
# Source: src/tests/test_legendvla_hooks.py L441-L448
def sample_fm_time(bsz: int) -> torch.FloatTensor:
    flow_alpha = 1.5
    flow_beta = 1
    flow_t_max = 1 - 0.001  # 1 - sig_min
    flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)
    z = flow_beta_dist.sample((bsz,))
    t = flow_t_max * (1 - z)  # flip and shift
    return t


# ---------------------------------------------------------------------------
# Phase 5: RTC Verification
# ---------------------------------------------------------------------------

def check_5_1_delay_sampling_distribution(report: PhaseReport, skip_visual: bool) -> None:
    """Check 5.1: Delay Sampling Distribution (exponential bias)."""
    print("\n--- Check 5.1: Delay Sampling Distribution ---")

    n_samples = 100_000
    max_delay = 8
    valid_action_len = torch.full((n_samples,), 64, dtype=torch.long)

    delays = sample_rtc_delay(
        valid_action_len, strategy="exp", max_delay=max_delay,
    )

    # All delays in [0, max_delay)
    in_range = bool((delays >= 0).all() and (delays < max_delay).all())
    report.add(assert_check(
        in_range,
        "5.1a_delay_range",
        f"All delays in [0, {max_delay}): min={delays.min().item()}, max={delays.max().item()}",
        {"min": int(delays.min().item()), "max": int(delays.max().item())},
    ))

    # P(delay=0) > 2 * P(delay=7) -- exponential bias toward small delays
    counts = torch.bincount(delays, minlength=max_delay).float()
    probs = counts / counts.sum()
    p0 = probs[0].item()
    p7 = probs[7].item()
    exp_bias = p0 > 2 * p7
    report.add(assert_check(
        exp_bias,
        "5.1b_exponential_bias",
        f"P(delay=0)={p0:.4f} > 2*P(delay=7)={2*p7:.4f}: {exp_bias}",
        {"P_delay_0": p0, "P_delay_7": p7, "ratio": p0 / max(p7, 1e-9)},
    ))

    # Visualization: histogram of delay distribution + uniform reference
    if not skip_visual:
        plt = safe_import_plt()
        if plt is not None:
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(max_delay)
            bar_width = 0.35
            ax.bar(x - bar_width / 2, probs.numpy(), bar_width, label="exp strategy", color="steelblue")
            uniform_prob = 1.0 / max_delay
            ax.bar(x + bar_width / 2, [uniform_prob] * max_delay, bar_width, label="uniform ref", color="salmon", alpha=0.7)
            ax.set_xlabel("delay")
            ax.set_ylabel("probability")
            ax.set_title("RTC Delay Distribution: exp vs uniform")
            ax.set_xticks(x)
            ax.legend()
            fig.tight_layout()
            fig.savefig(report.output_dir / "check_5_1_delay_distribution.png", dpi=150)
            plt.close(fig)


def check_5_2_prefix_postfix_mask(report: PhaseReport) -> None:
    """Check 5.2: Prefix/Postfix Mask Correctness with forced delay."""
    print("\n--- Check 5.2: Prefix/Postfix Mask Correctness ---")

    batch_size, horizon, action_dim = 2, 64, 48
    actions = torch.randn(batch_size, horizon, action_dim)
    actions_valid_mask = torch.ones(batch_size, horizon, action_dim, dtype=torch.bool)
    postfix_time = torch.tensor([0.5, 0.7])
    n_actions = torch.tensor([64, 64], dtype=torch.long)
    forced_delay = torch.tensor([3, 5], dtype=torch.long)

    token_t, postfix_valid_mask, prefix_mask, delay = build_rtc_flow_inputs(
        actions=actions,
        actions_valid_mask=actions_valid_mask,
        postfix_time=postfix_time,
        n_actions=n_actions,
        forced_delay=forced_delay,
    )

    # prefix_mask[0, :3] all True, prefix_mask[0, 3:] all False
    prefix_0_correct = bool(prefix_mask[0, :3].all() and (~prefix_mask[0, 3:]).all())
    report.add(assert_check(
        prefix_0_correct,
        "5.2a_prefix_mask_sample0",
        f"prefix_mask[0,:3] all True, [0,3:] all False: {prefix_0_correct}",
    ))

    # prefix_mask[1, :5] all True, prefix_mask[1, 5:] all False
    prefix_1_correct = bool(prefix_mask[1, :5].all() and (~prefix_mask[1, 5:]).all())
    report.add(assert_check(
        prefix_1_correct,
        "5.2b_prefix_mask_sample1",
        f"prefix_mask[1,:5] all True, [1,5:] all False: {prefix_1_correct}",
    ))

    # token_t[0, :3] == 1.0, token_t[0, 3:] == 0.5
    t_prefix_ok = bool(torch.allclose(token_t[0, :3], torch.ones(3)))
    t_postfix_ok = bool(torch.allclose(token_t[0, 3:], torch.full((61,), 0.5)))
    report.add(assert_check(
        t_prefix_ok and t_postfix_ok,
        "5.2c_token_t_values",
        f"token_t[0,:3]==1.0: {t_prefix_ok}, token_t[0,3:]==0.5: {t_postfix_ok}",
    ))

    # postfix_valid_mask[0, :3] all False (prefix excluded from loss)
    # postfix_valid_mask has shape [B, H, D] due to broadcasting
    postfix_prefix_excluded = bool((~postfix_valid_mask[0, :3]).all())
    report.add(assert_check(
        postfix_prefix_excluded,
        "5.2d_postfix_excludes_prefix",
        f"postfix_valid_mask[0,:3] all False (prefix excluded): {postfix_prefix_excluded}",
    ))

    # delay=0 should be a valid RTC training case: no prefix, all valid tokens
    # remain supervised as postfix tokens.
    zero_delay = torch.zeros(batch_size, dtype=torch.long)
    token_t_zero, postfix_valid_mask_zero, prefix_mask_zero, delay_zero = build_rtc_flow_inputs(
        actions=actions,
        actions_valid_mask=actions_valid_mask,
        postfix_time=postfix_time,
        n_actions=n_actions,
        forced_delay=zero_delay,
    )

    zero_delay_ok = bool(torch.equal(delay_zero, zero_delay))
    prefix_empty = bool((~prefix_mask_zero).all())
    token_t_matches_postfix = bool(torch.allclose(
        token_t_zero,
        postfix_time[:, None].expand(-1, horizon),
    ))
    postfix_covers_all_valid = bool(torch.equal(
        postfix_valid_mask_zero,
        actions_valid_mask,
    ))

    report.add(assert_check(
        zero_delay_ok and prefix_empty and token_t_matches_postfix and postfix_covers_all_valid,
        "5.2e_zero_delay_is_valid_training_case",
        "forced_delay=0 -> empty prefix, token_t==postfix_time, postfix_valid_mask==actions_valid_mask",
        {
            "delay_zero_ok": zero_delay_ok,
            "prefix_empty": prefix_empty,
            "token_t_matches_postfix": token_t_matches_postfix,
            "postfix_covers_all_valid": postfix_covers_all_valid,
        },
    ))


def check_5_3_rtc_noisy_actions(report: PhaseReport) -> None:
    """Check 5.3: RTC noisy actions preserve original actions at prefix positions."""
    print("\n--- Check 5.3: RTC Noisy Actions ---")

    batch_size, horizon, action_dim = 2, 64, 48
    torch.manual_seed(42)
    actions = torch.randn(batch_size, horizon, action_dim)
    actions_valid_mask = torch.ones(batch_size, horizon, action_dim, dtype=torch.bool)
    t = torch.tensor([0.5, 0.7])
    n_actions = torch.tensor([64, 64], dtype=torch.long)

    # Create a mock model with required attributes
    mock_model = SimpleNamespace(
        flow_config=SimpleNamespace(
            sig_min=0.001, num_parallel_t=1,
            sampling="beta", alpha=1.5, beta=1.0,
        ),
        rtc_config=SimpleNamespace(enabled=True, delay_strategy="uniform", max_delay=8),
    )

    # Run build_flow_inputs multiple times to find a case with nonzero delay.
    # With uniform strategy and max_delay=8, P(delay>0) = 7/8 per sample.
    found_nonzero = False
    for seed in range(50):
        torch.manual_seed(seed)
        batch = {
            "actions": actions.clone(),
            "actions_valid_mask": actions_valid_mask.clone(),
            "n_actions": n_actions.clone(),
        }
        result = build_flow_inputs(mock_model, batch, num_parallel_t=1, sampled_t=t[:batch_size].clone())
        noisy_actions = result["noisy_actions"]
        time_for_model = result["time_for_model"]

        # Identify prefix positions from time_for_model (token_t == 1.0)
        prefix_mask = (time_for_model == 1.0)

        if prefix_mask.any():
            found_nonzero = True
            # At prefix positions, noisy_actions should equal original actions
            prefix_expanded = prefix_mask.unsqueeze(-1).expand_as(actions)
            prefix_match = bool(torch.allclose(
                noisy_actions[prefix_expanded],
                actions[prefix_expanded],
                atol=1e-6,
            ))
            report.add(assert_check(
                prefix_match,
                "5.3_rtc_noisy_prefix_unchanged",
                f"noisy_actions at prefix == actions at prefix: {prefix_match} (seed={seed})",
                {"seed": seed, "prefix_count": int(prefix_mask.sum().item())},
            ))
            break

    if not found_nonzero:
        report.add(assert_check(
            False,
            "5.3_rtc_noisy_prefix_unchanged",
            "Could not find a seed with nonzero RTC delay in 50 attempts",
        ))


# ---------------------------------------------------------------------------
# Phase 6: Flow Matching Verification
# ---------------------------------------------------------------------------

def check_6_1_psi_t_interpolation(report: PhaseReport) -> None:
    """Check 6.1: psi_t interpolation at t=0, t=1, and t=0.5."""
    print("\n--- Check 6.1: psi_t Interpolation ---")

    torch.manual_seed(0)
    B, H, D = 4, 16, 48
    noise = torch.randn(B, H, D)
    target = torch.randn(B, H, D)
    sig_min = 0.001

    # t=0: psi_t should return noise
    t0 = torch.zeros(B)
    out_t0 = psi_t(noise, target, t0, sig_min)
    err_t0 = (out_t0 - noise).abs().max().item()
    report.add(assert_check(
        err_t0 < 1e-6,
        "6.1a_psi_t_at_t0",
        f"psi_t(noise, target, t=0) ~= noise, max_err={err_t0:.2e}",
    ))

    # t=1: psi_t should return target + sig_min * noise
    t1 = torch.ones(B)
    out_t1 = psi_t(noise, target, t1, sig_min)
    expected_t1 = target + sig_min * noise
    err_t1 = (out_t1 - expected_t1).abs().max().item()
    report.add(assert_check(
        err_t1 < 1e-5,
        "6.1b_psi_t_at_t1",
        f"psi_t(noise, target, t=1) ~= target + sig_min*noise, max_err={err_t1:.2e}",
    ))

    # t=0.5: psi_t = (1 - 0.999*0.5)*noise + 0.5*target = 0.5005*noise + 0.5*target
    t05 = torch.full((B,), 0.5)
    out_t05 = psi_t(noise, target, t05, sig_min)
    expected_t05 = 0.5005 * noise + 0.5 * target
    err_t05 = (out_t05 - expected_t05).abs().max().item()
    report.add(assert_check(
        err_t05 < 1e-5,
        "6.1c_psi_t_at_t05",
        f"psi_t at t=0.5 = 0.5005*noise + 0.5*target, max_err={err_t05:.2e}",
    ))


def check_6_2_velocity_target(report: PhaseReport) -> None:
    """Check 6.2: Velocity target formula: target_v = actions - (1-sig_min)*noise."""
    print("\n--- Check 6.2: Velocity Target ---")

    torch.manual_seed(1)
    B, H, D = 4, 16, 48
    sig_min = 0.001
    actions = torch.randn(B, H, D)
    noise = torch.randn(B, H, D)

    # Manually compute target velocity
    target_v_manual = actions - (1 - sig_min) * noise

    # Compare with compute_flow_loss internals by constructing a mock
    mock_model = SimpleNamespace(flow_config=SimpleNamespace(sig_min=sig_min))

    # Use pred_v = target_v (perfect prediction) -- loss should be zero
    pred_v = target_v_manual.clone()
    loss_mask = torch.ones(B, H, D, dtype=torch.bool)

    loss = compute_packed_flow_loss(
        model=mock_model,
        actions=actions,
        pred_v_t=pred_v,
        noise=noise,
        loss_mask=loss_mask,
    )
    loss_val = loss.item()
    report.add(assert_check(
        loss_val < 1e-10,
        "6.2_velocity_target_formula",
        f"Flow loss with perfect velocity = {loss_val:.2e} (should be ~0)",
        {"loss": loss_val},
    ))


def check_6_3_beta_time_sampling(report: PhaseReport, skip_visual: bool) -> None:
    """Check 6.3: Beta time sampling distribution statistics."""
    print("\n--- Check 6.3: Beta Time Sampling Distribution ---")

    n_samples = 100_000
    t_samples = sample_fm_time(n_samples)

    mean_t = t_samples.mean().item()
    # z ~ Beta(1.5, 1.0), E[z] = 0.6.  t = (1 - sig_min) * (1 - z).
    # z ~ Beta(1.5, 1.0), E[z] = 1.5/2.5 = 0.6
    # t = 0.999 * (1 - z), so E[t] = 0.999 * 0.4 ~ 0.40
    # The distribution biases toward *smaller* t (more denoising needed),
    # which is the intended design (pi0 paper: more training near t=0).
    mean_check = 0.3 < mean_t < 0.5
    report.add(assert_check(
        mean_check,
        "6.3_beta_time_mean",
        f"E[t] = {mean_t:.4f} (expected ~0.40, check in [0.3, 0.5])",
        {"mean": mean_t, "std": t_samples.std().item(),
         "min": t_samples.min().item(), "max": t_samples.max().item()},
    ))

    # Visualization: histogram of t samples
    if not skip_visual:
        plt = safe_import_plt()
        if plt is not None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(t_samples.numpy(), bins=80, density=True, color="steelblue", alpha=0.8, edgecolor="white", linewidth=0.3)
            ax.axvline(mean_t, color="red", linestyle="--", label=f"mean={mean_t:.4f}")
            ax.set_xlabel("t")
            ax.set_ylabel("density")
            ax.set_title("Flow Matching Time Sampling: Beta(1.5, 1.0) flipped")
            ax.legend()
            fig.tight_layout()
            fig.savefig(report.output_dir / "check_6_3_beta_time_distribution.png", dpi=150)
            plt.close(fig)


def check_6_4_euler_integration(report: PhaseReport, skip_visual: bool) -> None:
    """Check 6.4: Euler integration with perfect velocity oracle recovers target actions."""
    print("\n--- Check 6.4: Euler Integration Closed-loop ---")

    torch.manual_seed(7)
    B, H, D = 2, 16, 48
    sig_min = 0.001

    # Ground truth actions (x1) and initial noise (x0)
    x1 = torch.randn(B, H, D)
    x0 = torch.randn(B, H, D)

    # Perfect velocity oracle: v(x_t, t) = x1 - (1 - sig_min) * x0
    # This is the conditional vector field for the OT path: psi_t(x0, x1, t).
    # The ODE is dx/dt = v, where v = x1 - (1 - sig_min) * x0 (constant in t).
    target_v = x1 - (1 - sig_min) * x0

    # 10-step Euler integration from x0
    n_steps = 10
    delta_t = 1.0 / n_steps
    x = x0.clone()
    t = 0.0

    # Track trajectory for 3 selected dims for visualization
    trajectory = []  # list of (t, x_snapshot)
    trajectory.append((t, x[:, 0, :3].clone()))

    for step in range(n_steps):
        # Oracle velocity at time t: v(x_t, t) = x1 - (1 - sig_min) * x0
        v = target_v
        x = x + delta_t * v
        t += delta_t
        trajectory.append((t, x[:, 0, :3].clone()))

    # At t=1, the exact solution is psi_t(x0, x1, t=1) = sig_min*x0 + x1
    expected_final = psi_t(x0, x1, torch.ones(B), sig_min)
    error = (x - expected_final).abs().max().item()

    report.add(assert_check(
        error < 0.05,
        "6.4_euler_integration",
        f"Euler 10-step final error = {error:.6f} (threshold < 0.05)",
        {"max_error": error, "n_steps": n_steps},
    ))

    # Visualization: 3 selected dims trajectory during integration
    if not skip_visual:
        plt = safe_import_plt()
        if plt is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            times = [tr[0] for tr in trajectory]
            for dim_idx in range(3):
                ax = axes[dim_idx]
                # Plot trajectory for batch element 0
                values = [tr[1][0, dim_idx].item() for tr in trajectory]
                ax.plot(times, values, "o-", color="steelblue", markersize=4, label="Euler trajectory")
                # Plot start and target
                ax.axhline(x0[0, 0, dim_idx].item(), color="gray", linestyle=":", alpha=0.6, label="x0 (noise)")
                ax.axhline(expected_final[0, 0, dim_idx].item(), color="red", linestyle="--", alpha=0.8, label="target")
                ax.set_xlabel("t")
                ax.set_ylabel("value")
                ax.set_title(f"dim {dim_idx}")
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
            fig.suptitle("Euler Integration: 3 dims, batch 0, token 0", fontsize=12)
            fig.tight_layout()
            fig.savefig(report.output_dir / "check_6_4_euler_trajectory.png", dpi=150)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 & 6: RTC and Flow Matching verification")
    parser.add_argument("--skip-visual", action="store_true", help="Skip visualization outputs")
    args = parser.parse_args()

    phase5_dir = get_output_dir("phase5_rtc")
    phase6_dir = get_output_dir("phase6_flow")

    # Phase 5
    print("=" * 60)
    print("  Phase 5: RTC Verification")
    print("=" * 60)
    report5 = PhaseReport("Phase 5: RTC Verification", phase5_dir)
    check_5_1_delay_sampling_distribution(report5, args.skip_visual)
    check_5_2_prefix_postfix_mask(report5)
    check_5_3_rtc_noisy_actions(report5)
    report5.save()
    report5.print_summary()

    # Phase 6
    print("=" * 60)
    print("  Phase 6: Flow Matching Verification")
    print("=" * 60)
    report6 = PhaseReport("Phase 6: Flow Matching Verification", phase6_dir)
    check_6_1_psi_t_interpolation(report6)
    check_6_2_velocity_target(report6)
    check_6_3_beta_time_sampling(report6, args.skip_visual)
    check_6_4_euler_integration(report6, args.skip_visual)
    report6.save()
    report6.print_summary()

    # Exit code
    if report5.all_passed and report6.all_passed:
        print("All Phase 5 and Phase 6 checks passed.")
        sys.exit(0)
    else:
        print("Some checks failed -- see reports above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
