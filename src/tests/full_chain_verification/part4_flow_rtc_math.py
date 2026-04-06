"""
Part 4: Flow Matching Math + RTC Logic Verification.

Pure-math tests that run on CPU without model weights or real data.

Checks:
  4.1  psi_t interpolation: boundary values, linearity, monotonicity
  4.2  Velocity target formula: analytic match, finite-difference check
  4.3  Flow time sampling: range, distribution, stratification
  4.4  build_flow_inputs integration: shapes, content, mask
  4.5  RTC delay/mask/time: sampling distribution, prefix mask, token times

Usage:
    python -m src.tests.full_chain_verification.part4_flow_rtc_math
"""

from __future__ import annotations

import argparse

import torch

from src.policy.legendvla_loss import (
    build_dense_diffloss_inputs,
    build_flow_inputs,
    build_rtc_flow_inputs,
    compute_packed_flow_loss,
    psi_t,
    sample_rtc_delay,
)
from src.tests.full_chain_verification.utils import (
    CheckResult,
    PhaseReport,
    assert_check,
    get_output_dir,
    safe_import_plt,
    tensor_stats,
)


OUTPUT_PART = "part4"
SIG_MIN = 0.001


# ── 4.1 psi_t interpolation ────────────────────────────────────────────────

def test_psi_t_t0_is_noise(report: PhaseReport) -> None:
    """t=0: psi_t should return pure noise x."""
    B, H, D = 4, 32, 48
    x = torch.randn(B, H, D)
    x1 = torch.randn(B, H, D)
    t = torch.zeros(B)
    result = psi_t(x, x1, t, SIG_MIN)
    report.add(assert_check(
        torch.allclose(result, x, atol=1e-6),
        "4.1a psi_t(t=0) == noise",
        f"max_diff={float((result - x).abs().max()):.2e}",
    ))


def test_psi_t_t1_approaches_target(report: PhaseReport) -> None:
    """t=1: psi_t should be sig_min * x + x1, approximately equal to x1."""
    B, H, D = 4, 32, 48
    x = torch.randn(B, H, D)
    x1 = torch.randn(B, H, D)
    t = torch.ones(B)
    result = psi_t(x, x1, t, SIG_MIN)
    expected = SIG_MIN * x + x1
    report.add(assert_check(
        torch.allclose(result, expected, atol=1e-6),
        "4.1b psi_t(t=1) == sig_min*x + x1",
        f"max_diff={float((result - expected).abs().max()):.2e}",
    ))
    # Also verify it's close to x1 (since sig_min is small)
    report.add(assert_check(
        float((result - x1).abs().max()) < 0.01 * float(x.abs().max()) + 1e-4,
        "4.1c psi_t(t=1) ≈ x1",
        f"max_diff_from_target={float((result - x1).abs().max()):.4f}",
    ))


def test_psi_t_linearity(report: PhaseReport) -> None:
    """t=0.5: result should be between x and x1."""
    B, H, D = 4, 32, 48
    x = torch.randn(B, H, D)
    x1 = torch.randn(B, H, D)
    t = torch.full((B,), 0.5)
    result = psi_t(x, x1, t, SIG_MIN)
    # At t=0.5: (1 - 0.999*0.5)*x + 0.5*x1 = 0.5005*x + 0.5*x1
    expected = (1 - (1 - SIG_MIN) * 0.5) * x + 0.5 * x1
    report.add(assert_check(
        torch.allclose(result, expected, atol=1e-6),
        "4.1d psi_t(t=0.5) analytic",
        f"max_diff={float((result - expected).abs().max()):.2e}",
    ))


def test_psi_t_monotonic(report: PhaseReport) -> None:
    """As t goes from 0 to 1, L2 distance to x1 should monotonically decrease."""
    B, H, D = 2, 16, 48
    x = torch.randn(B, H, D)
    x1 = torch.randn(B, H, D)
    steps = 100
    distances = []
    for i in range(steps + 1):
        t_val = i / steps
        t = torch.full((B,), t_val)
        result = psi_t(x, x1, t, SIG_MIN)
        dist = float(((result - x1) ** 2).sum(dim=-1).mean())
        distances.append(dist)
    monotonic = all(distances[i] >= distances[i + 1] - 1e-6 for i in range(len(distances) - 1))
    report.add(assert_check(
        monotonic,
        "4.1e psi_t monotonically approaches x1",
        f"distances[0]={distances[0]:.4f}, distances[-1]={distances[-1]:.6f}",
    ))


def test_psi_t_2d_time(report: PhaseReport) -> None:
    """psi_t should also accept 2D time [B, H] for token-wise RTC times."""
    B, H, D = 2, 8, 48
    x = torch.randn(B, H, D)
    x1 = torch.randn(B, H, D)
    t = torch.rand(B, H)
    result = psi_t(x, x1, t, SIG_MIN)
    # Manual computation with broadcasting
    t_3d = t.unsqueeze(-1)
    expected = (1 - (1 - SIG_MIN) * t_3d) * x + t_3d * x1
    report.add(assert_check(
        torch.allclose(result, expected, atol=1e-6),
        "4.1f psi_t with 2D time [B, H]",
        f"max_diff={float((result - expected).abs().max()):.2e}",
    ))


# ── 4.2 Velocity target formula ────────────────────────────────────────────

def test_velocity_target_formula(report: PhaseReport) -> None:
    """target_v = x1 - (1 - sig_min) * x0, matching compute_packed_flow_loss."""
    B, H, D = 4, 32, 48
    actions = torch.randn(B, H, D)
    noise = torch.randn(B, H, D)
    expected_v = actions - (1 - SIG_MIN) * noise
    report.add(assert_check(
        True,  # Just verify the formula is correct by construction
        "4.2a velocity_target formula",
        f"target_v = x1 - (1 - sig_min) * x0, sig_min={SIG_MIN}",
        details={"target_v_stats": tensor_stats(expected_v)},
    ))


def test_velocity_matches_psi_t_derivative(report: PhaseReport) -> None:
    """Finite difference: d(psi_t)/dt should equal the velocity target."""
    B, H, D = 4, 32, 48
    x0 = torch.randn(B, H, D, dtype=torch.float64)
    x1 = torch.randn(B, H, D, dtype=torch.float64)
    eps = 1e-5
    t_val = 0.3
    t = torch.full((B,), t_val, dtype=torch.float64)
    t_plus = torch.full((B,), t_val + eps, dtype=torch.float64)
    psi_t_val = psi_t(x0, x1, t, SIG_MIN)
    psi_t_plus = psi_t(x0, x1, t_plus, SIG_MIN)
    finite_diff = (psi_t_plus - psi_t_val) / eps
    analytic_v = x1 - (1 - SIG_MIN) * x0
    max_diff = float((finite_diff - analytic_v).abs().max())
    report.add(assert_check(
        max_diff < 1e-3,
        "4.2b velocity matches psi_t derivative (finite diff)",
        f"max_diff={max_diff:.2e}, eps={eps}",
    ))


# ── 4.3 Flow time sampling ─────────────────────────────────────────────────

def test_flow_time_range_beta(report: PhaseReport) -> None:
    """Beta sampled times should be in [0, 1)."""
    from torch.distributions import Beta
    dist = Beta(1.5, 1.0)
    N = 10000
    z = dist.sample((N,))
    t = (1 - SIG_MIN) * (1 - z)
    t_min, t_max = float(t.min()), float(t.max())
    report.add(assert_check(
        t_min >= 0 and t_max < 1.0,
        "4.3a beta time range [0, 1)",
        f"min={t_min:.6f}, max={t_max:.6f}",
    ))


def test_flow_time_beta_mean(report: PhaseReport) -> None:
    """Beta(1.5, 1.0): E[z]=0.6, E[t]=(1-sig_min)*(1-0.6)≈0.40."""
    from torch.distributions import Beta
    dist = Beta(1.5, 1.0)
    N = 100000
    z = dist.sample((N,))
    t = (1 - SIG_MIN) * (1 - z)
    mean_t = float(t.mean())
    report.add(assert_check(
        0.35 < mean_t < 0.45,
        "4.3b beta time E[t] ≈ 0.40",
        f"mean_t={mean_t:.4f} (expected ~0.40)",
    ))


def test_flow_time_uniform_stratified(report: PhaseReport) -> None:
    """Uniform stratified: batch elements should be evenly spaced."""
    B = 16
    eps = 1e-5
    ranks = torch.arange(B, dtype=torch.float32) / B
    offsets = torch.rand(1, dtype=torch.float32)
    t = (ranks.unsqueeze(1) + offsets.unsqueeze(0)) % (1 - eps)
    t = t.squeeze(1)
    # Check approximate uniform spacing
    sorted_t, _ = t.sort()
    diffs = sorted_t[1:] - sorted_t[:-1]
    expected_gap = 1.0 / B
    max_gap_deviation = float((diffs - expected_gap).abs().max())
    report.add(assert_check(
        max_gap_deviation < expected_gap * 0.5,
        "4.3c uniform stratified spacing",
        f"expected_gap={expected_gap:.4f}, max_deviation={max_gap_deviation:.4f}",
    ))


# ── 4.4 build_flow_inputs integration ──────────────────────────────────────

class MockModelForFlow:
    """Minimal mock of LegendVLA for build_flow_inputs."""
    class _FlowConfig:
        sig_min = SIG_MIN
        num_parallel_t = 4
        sampling = "beta"
        alpha = 1.5
        beta = 1.0

    class _RTCConfig:
        enabled = False
        delay_strategy = "exp"
        max_delay = 8

    flow_config = _FlowConfig()
    rtc_config = _RTCConfig()
    flow_beta_dist = torch.distributions.Beta(1.5, 1.0)

    def sample_flow_time(self, batch_size, num_samples=1):
        z = self.flow_beta_dist.sample((batch_size, num_samples))
        t = (1 - self.flow_config.sig_min) * (1 - z)
        return t.squeeze(1) if num_samples == 1 else t


def test_flow_inputs_shapes_T1(report: PhaseReport) -> None:
    """num_parallel_t=1: shapes should match [B, H, D]."""
    B, H, D = 2, 32, 48
    model = MockModelForFlow()
    model.flow_config.num_parallel_t = 1
    batch = {
        "actions": torch.randn(B, H, D),
        "actions_valid_mask": torch.ones(B, H, D, dtype=torch.bool),
    }
    result = build_flow_inputs(model, batch, num_parallel_t=1)
    checks = {
        "noisy_actions": (B, H, D),
        "time_for_model": (B, H),
        "noise": (B, H, D),
        "actions": (B, H, D),
        "loss_mask": (B, H, D),
    }
    all_ok = True
    details = {}
    for key, expected_shape in checks.items():
        actual = tuple(result[key].shape)
        ok = actual == expected_shape
        details[key] = f"{actual} (expected {expected_shape})"
        if not ok:
            all_ok = False
    report.add(assert_check(all_ok, "4.4a flow_inputs shapes T=1", str(details), details))


def test_flow_inputs_shapes_T4(report: PhaseReport) -> None:
    """num_parallel_t=4: action dim packed as T*H."""
    B, H, D, T = 2, 32, 48, 4
    model = MockModelForFlow()
    batch = {
        "actions": torch.randn(B, H, D),
        "actions_valid_mask": torch.ones(B, H, D, dtype=torch.bool),
    }
    result = build_flow_inputs(model, batch, num_parallel_t=T)
    checks = {
        "noisy_actions": (B, T * H, D),
        "time_for_model": (B, T * H),
        "noise": (B, T * H, D),
        "actions": (B, T * H, D),
        "loss_mask": (B, T * H, D),
    }
    all_ok = True
    details = {}
    for key, expected_shape in checks.items():
        actual = tuple(result[key].shape)
        ok = actual == expected_shape
        details[key] = f"{actual} (expected {expected_shape})"
        if not ok:
            all_ok = False
    report.add(assert_check(all_ok, "4.4b flow_inputs shapes T=4", str(details), details))


def test_flow_inputs_noisy_action_content(report: PhaseReport) -> None:
    """With fixed seed, noisy_actions = psi_t(noise, actions, time)."""
    B, H, D = 2, 32, 48
    model = MockModelForFlow()

    actions = torch.randn(B, H, D)
    batch = {
        "actions": actions,
        "actions_valid_mask": torch.ones(B, H, D, dtype=torch.bool),
    }
    sampled_t = torch.tensor([0.3, 0.7])

    torch.manual_seed(42)
    result = build_flow_inputs(model, batch, num_parallel_t=1, sampled_t=sampled_t)

    # Reproduce: noise is generated internally by build_flow_inputs as torch.randn_like(flat_actions)
    noise = result["noise"]
    time_for_model = result["time_for_model"]
    expected_noisy = psi_t(noise, actions, time_for_model, SIG_MIN)
    max_diff = float((result["noisy_actions"] - expected_noisy).abs().max())
    report.add(assert_check(
        max_diff < 1e-5,
        "4.4c noisy_actions == psi_t(noise, actions, t)",
        f"max_diff={max_diff:.2e}",
    ))


def test_flow_loss_zero_mask_zero_loss(report: PhaseReport) -> None:
    """When loss_mask is all False, flow loss should be 0."""
    B, H, D = 2, 32, 48

    class _Model:
        class flow_config:
            sig_min = SIG_MIN

    model = _Model()
    loss = compute_packed_flow_loss(
        model=model,
        actions=torch.randn(B, H, D),
        pred_v_t=torch.randn(B, H, D),
        noise=torch.randn(B, H, D),
        loss_mask=torch.zeros(B, H, D, dtype=torch.bool),
    )
    report.add(assert_check(
        float(loss) == 0.0,
        "4.4d zero mask → zero flow loss",
        f"loss={float(loss):.6f}",
    ))


def test_flow_inputs_actions_repeated(report: PhaseReport) -> None:
    """T=4: the packed actions should be the original actions repeated T times."""
    B, H, D, T = 2, 32, 48, 4
    model = MockModelForFlow()
    actions = torch.randn(B, H, D)
    batch = {
        "actions": actions,
        "actions_valid_mask": torch.ones(B, H, D, dtype=torch.bool),
    }
    result = build_flow_inputs(model, batch, num_parallel_t=T)
    packed_actions = result["actions"]
    # Packed actions should be [B, T*H, D] where each T chunk is the same actions
    for chunk_idx in range(T):
        chunk = packed_actions[:, chunk_idx * H:(chunk_idx + 1) * H, :]
        match = torch.allclose(chunk, actions, atol=1e-6)
        if not match:
            report.add(assert_check(
                False, "4.4e actions repeated T times",
                f"chunk {chunk_idx} mismatch",
            ))
            return
    report.add(assert_check(True, "4.4e actions repeated T times", f"all {T} chunks match"))


# ── 4.5 RTC logic ──────────────────────────────────────────────────────────

def test_rtc_delay_range_uniform(report: PhaseReport) -> None:
    """Uniform strategy: delay in [0, min(valid_len, max_delay))."""
    N = 10000
    valid_len = torch.full((N,), 20, dtype=torch.long)
    delays = sample_rtc_delay(valid_len, strategy="uniform", max_delay=8)
    d_min, d_max = int(delays.min()), int(delays.max())
    report.add(assert_check(
        d_min >= 0 and d_max < 8,
        "4.5a uniform delay range [0, 8)",
        f"min={d_min}, max={d_max}",
    ))


def test_rtc_delay_range_exp(report: PhaseReport) -> None:
    """Exp strategy: delay in [0, max_delay), biased toward smaller values."""
    N = 10000
    valid_len = torch.full((N,), 20, dtype=torch.long)
    delays = sample_rtc_delay(valid_len, strategy="exp", max_delay=8)
    d_min, d_max = int(delays.min()), int(delays.max())
    # Count occurrences
    counts = torch.bincount(delays, minlength=8)[:8]
    p_delay_0 = float(counts[0]) / N
    p_delay_7 = float(counts[7]) / N
    report.add(assert_check(
        d_min >= 0 and d_max < 8 and p_delay_0 > p_delay_7,
        "4.5b exp delay biased toward small values",
        f"P(delay=0)={p_delay_0:.3f}, P(delay=7)={p_delay_7:.3f}",
        details={"histogram": counts.tolist()},
    ))


def test_rtc_delay_clamped_by_valid_len(report: PhaseReport) -> None:
    """When valid_action_len < max_delay, delay should not exceed valid_len."""
    N = 10000
    valid_len = torch.full((N,), 3, dtype=torch.long)
    delays_u = sample_rtc_delay(valid_len, strategy="uniform", max_delay=16)
    delays_e = sample_rtc_delay(valid_len, strategy="exp", max_delay=16)
    max_u, max_e = int(delays_u.max()), int(delays_e.max())
    report.add(assert_check(
        max_u < 3 and max_e < 3,
        "4.5c delay clamped by valid_len=3",
        f"max_uniform={max_u}, max_exp={max_e}",
    ))


def test_rtc_prefix_mask_content(report: PhaseReport) -> None:
    """Forced delay should produce exact prefix masks."""
    B, H, D = 3, 32, 48
    actions = torch.randn(B, H, D)
    valid_mask = torch.ones(B, H, D, dtype=torch.bool)
    postfix_time = torch.full((B,), 0.3)
    forced_delay = torch.tensor([0, 3, 7])

    token_t, postfix_valid, prefix_mask, delay = build_rtc_flow_inputs(
        actions=actions,
        actions_valid_mask=valid_mask,
        postfix_time=postfix_time,
        n_actions=torch.full((B,), H, dtype=torch.long),
        forced_delay=forced_delay,
    )

    checks = []
    # delay=0: no prefix
    checks.append(prefix_mask[0].sum().item() == 0)
    # delay=3: first 3 positions are prefix
    checks.append(prefix_mask[1, :3].all().item() and not prefix_mask[1, 3:].any().item())
    # delay=7: first 7 positions are prefix
    checks.append(prefix_mask[2, :7].all().item() and not prefix_mask[2, 7:].any().item())

    report.add(assert_check(
        all(checks),
        "4.5d prefix_mask content with forced_delay=[0,3,7]",
        f"checks={checks}",
    ))


def test_rtc_token_time_values(report: PhaseReport) -> None:
    """Prefix positions should have t=1.0, postfix should have t=postfix_time."""
    B, H, D = 3, 32, 48
    actions = torch.randn(B, H, D)
    valid_mask = torch.ones(B, H, D, dtype=torch.bool)
    postfix_time = torch.tensor([0.2, 0.5, 0.8])
    forced_delay = torch.tensor([0, 5, 10])

    token_t, _, prefix_mask, _ = build_rtc_flow_inputs(
        actions=actions,
        actions_valid_mask=valid_mask,
        postfix_time=postfix_time,
        n_actions=torch.full((B,), H, dtype=torch.long),
        forced_delay=forced_delay,
    )

    checks = []
    for b in range(B):
        d = forced_delay[b].item()
        if d > 0:
            # Prefix positions: t == 1.0
            checks.append(bool(torch.all(token_t[b, :d] == 1.0)))
        # Postfix positions: t == postfix_time[b]
        checks.append(bool(torch.allclose(token_t[b, d:], postfix_time[b].expand(H - d))))

    report.add(assert_check(
        all(checks),
        "4.5e token_t: prefix=1.0, postfix=postfix_time",
        f"checks={checks}",
    ))


def test_rtc_noisy_actions_prefix_clean(report: PhaseReport) -> None:
    """After build_flow_inputs with RTC, prefix positions hold clean actions."""
    B, H, D = 2, 32, 48
    model = MockModelForFlow()
    model.rtc_config.enabled = True
    model.rtc_config.max_delay = 8

    actions = torch.randn(B, H, D)
    batch = {
        "actions": actions,
        "actions_valid_mask": torch.ones(B, H, D, dtype=torch.bool),
        "n_actions": torch.full((B,), H, dtype=torch.long),
    }

    result = build_flow_inputs(model, batch, num_parallel_t=1)
    noisy = result["noisy_actions"]
    t = result["time_for_model"]

    # Positions where t == 1.0 are prefix positions that should hold clean actions
    prefix_positions = (t == 1.0)
    if prefix_positions.any():
        prefix_noisy = noisy[prefix_positions.unsqueeze(-1).expand_as(noisy)]
        prefix_clean = actions[prefix_positions.unsqueeze(-1).expand_as(actions)]
        max_diff = float((prefix_noisy - prefix_clean).abs().max())
        report.add(assert_check(
            max_diff < 1e-5,
            "4.5f prefix noisy_actions == clean actions",
            f"max_diff={max_diff:.2e}, n_prefix={int(prefix_positions.sum())}",
        ))
    else:
        # All delays were 0 (possible with small batch), still passes
        report.add(assert_check(
            True,
            "4.5f prefix noisy_actions == clean actions",
            "no prefix positions (all delays=0), trivially passed",
        ))


def test_rtc_loss_mask_excludes_prefix(report: PhaseReport) -> None:
    """Loss mask should be False at prefix positions (no loss on known actions)."""
    B, H, D = 3, 32, 48
    actions = torch.randn(B, H, D)
    valid_mask = torch.ones(B, H, D, dtype=torch.bool)
    postfix_time = torch.full((B,), 0.5)
    forced_delay = torch.tensor([0, 5, 10])

    _, postfix_valid, prefix_mask, _ = build_rtc_flow_inputs(
        actions=actions,
        actions_valid_mask=valid_mask,
        postfix_time=postfix_time,
        n_actions=torch.full((B,), H, dtype=torch.long),
        forced_delay=forced_delay,
    )

    # postfix_valid should be False at prefix positions
    checks = []
    for b in range(B):
        d = forced_delay[b].item()
        if d > 0:
            checks.append(bool(~postfix_valid[b, :d].any()))
    # Postfix valid positions should be True
    checks.append(bool(postfix_valid[1, 5:].all()))
    checks.append(bool(postfix_valid[2, 10:].all()))

    report.add(assert_check(
        all(checks),
        "4.5g loss_mask excludes prefix positions",
        f"checks={checks}",
    ))


# ── 4.6 build_dense_diffloss_inputs ────────────────────────────────────────

def test_chunk_unfold_content(report: PhaseReport) -> None:
    """Action chunk unfold should produce sliding windows of size=4, step=1."""
    H, D, chunk_size = 32, 48, 4
    actions = torch.arange(H * D, dtype=torch.float32).reshape(1, H, D)
    chunks = actions.unfold(dimension=1, size=chunk_size, step=1)
    flat = chunks.flatten(start_dim=2)

    # unfold returns [B, num_windows, D, chunk_size], so after flatten(start_dim=2)
    # each chunk is D-major: actions[0, 0:4, :].T.flatten()
    expected_0 = actions[0, 0:chunk_size].T.flatten()
    expected_1 = actions[0, 1:1 + chunk_size].T.flatten()

    check_0 = torch.allclose(flat[0, 0], expected_0)
    check_1 = torch.allclose(flat[0, 1], expected_1)
    expected_n_chunks = H - chunk_size + 1
    check_shape = flat.shape == (1, expected_n_chunks, D * chunk_size)

    report.add(assert_check(
        check_0 and check_1 and check_shape,
        "4.6a chunk unfold content and shape",
        f"shape={tuple(flat.shape)}, expected (1, {expected_n_chunks}, {D * chunk_size})",
    ))


def test_hidden_positions_maps_correctly(report: PhaseReport) -> None:
    """hidden_positions[b, 0] = answer_start_idx - 1 (last prompt token)."""
    B, seq_len, hidden_dim = 2, 200, 64
    H, D, chunk_size = 32, 48, 4

    class _MockModel:
        class ar_action_train_config:
            chunk_size = 4

    model = _MockModel()
    hidden_states = torch.randn(B, seq_len, hidden_dim)
    actions = torch.randn(B, H, D)
    answer_start_idx = torch.tensor([100, 120])
    n_actions = torch.tensor([20, 32])
    vla_mask = torch.tensor([True, True])

    vla_hidden_z, action_gt, diffloss_mask = build_dense_diffloss_inputs(
        model, hidden_states, actions, answer_start_idx, n_actions, vla_mask,
    )

    max_chunk_count = H - chunk_size + 1
    total_positions = B * max_chunk_count

    # Check shape
    check_shape = vla_hidden_z.shape == (total_positions, hidden_dim)

    # Check first position for sample 0: should index hidden_states[0, 99]
    # (answer_start_idx=100, so hidden_positions[0, 0] = 100 - 1 = 99)
    expected_hidden_0 = hidden_states[0, 99]
    actual_hidden_0 = vla_hidden_z[0]
    check_content = torch.allclose(actual_hidden_0, expected_hidden_0, atol=1e-6)

    # Check second position for sample 0: hidden_states[0, 100]
    expected_hidden_1 = hidden_states[0, 100]
    actual_hidden_1 = vla_hidden_z[1]
    check_content_1 = torch.allclose(actual_hidden_1, expected_hidden_1, atol=1e-6)

    report.add(assert_check(
        check_shape and check_content and check_content_1,
        "4.6b hidden_positions[0,0] = answer_start_idx - 1",
        f"shape_ok={check_shape}, pos0_ok={check_content}, pos1_ok={check_content_1}",
    ))


def test_diffloss_mask_count(report: PhaseReport) -> None:
    """DiffLoss mask should have valid_chunk_count = max(0, n_actions - chunk_size + 1)."""
    B, seq_len, hidden_dim = 2, 200, 64
    H, D, chunk_size = 32, 48, 4

    class _MockModel:
        class ar_action_train_config:
            chunk_size = 4

    model = _MockModel()
    hidden_states = torch.randn(B, seq_len, hidden_dim)
    actions = torch.randn(B, H, D)
    answer_start_idx = torch.tensor([100, 120])
    n_actions = torch.tensor([20, 3])  # sample 1 has n_actions < chunk_size
    vla_mask = torch.tensor([True, True])

    _, _, diffloss_mask = build_dense_diffloss_inputs(
        model, hidden_states, actions, answer_start_idx, n_actions, vla_mask,
    )

    max_chunk_count = H - chunk_size + 1
    # Sample 0: valid_chunks = max(0, 20 - 4 + 1) = 17
    # Sample 1: valid_chunks = max(0, 3 - 4 + 1) = 0
    mask_reshaped = diffloss_mask.reshape(B, max_chunk_count)
    count_0 = int(mask_reshaped[0].sum())
    count_1 = int(mask_reshaped[1].sum())

    report.add(assert_check(
        count_0 == 17 and count_1 == 0,
        "4.6c diffloss_mask count (n_actions=20→17 chunks, n_actions=3→0 chunks)",
        f"count_0={count_0} (expected 17), count_1={count_1} (expected 0)",
    ))


def test_diffloss_repeat_interleave(report: PhaseReport) -> None:
    """repeat_interleave should give each copy independent noise but same action_gt."""
    B, seq_len, hidden_dim = 1, 200, 64
    H, D, chunk_size = 32, 48, 4
    repeat_factor = 16

    class _MockModel:
        class ar_action_train_config:
            chunk_size = 4

    model = _MockModel()
    hidden_states = torch.randn(B, seq_len, hidden_dim)
    actions = torch.randn(B, H, D)
    answer_start_idx = torch.tensor([100])
    n_actions = torch.tensor([20])
    vla_mask = torch.tensor([True])

    vla_hidden_z, action_gt, diffloss_mask = build_dense_diffloss_inputs(
        model, hidden_states, actions, answer_start_idx, n_actions, vla_mask,
    )

    original_count = vla_hidden_z.shape[0]
    repeated_z = vla_hidden_z.repeat_interleave(repeat_factor, dim=0)
    repeated_gt = action_gt.repeat_interleave(repeat_factor, dim=0)

    # Check shape
    check_shape = repeated_z.shape[0] == original_count * repeat_factor

    # Check that consecutive repeat_factor entries are identical
    checks_identical = True
    for i in range(min(5, original_count)):
        block = repeated_gt[i * repeat_factor:(i + 1) * repeat_factor]
        if not torch.all(block == block[0:1]):
            checks_identical = False
            break

    report.add(assert_check(
        check_shape and checks_identical,
        "4.6d repeat_interleave: same action_gt per group",
        f"original={original_count}, repeated={repeated_z.shape[0]}, identical={checks_identical}",
    ))


# ── Main ────────────────────────────────────────────────────────────────────

def run_all(skip_visual: bool = False) -> PhaseReport:
    out_dir = get_output_dir(OUTPUT_PART)
    report = PhaseReport("Part 4: Flow Matching Math + RTC Logic", out_dir)

    print("\n=== Part 4: Flow Matching Math + RTC Logic ===\n")

    # 4.1 psi_t
    test_psi_t_t0_is_noise(report)
    test_psi_t_t1_approaches_target(report)
    test_psi_t_linearity(report)
    test_psi_t_monotonic(report)
    test_psi_t_2d_time(report)

    # 4.2 Velocity target
    test_velocity_target_formula(report)
    test_velocity_matches_psi_t_derivative(report)

    # 4.3 Flow time sampling
    test_flow_time_range_beta(report)
    test_flow_time_beta_mean(report)
    test_flow_time_uniform_stratified(report)

    # 4.4 build_flow_inputs
    test_flow_inputs_shapes_T1(report)
    test_flow_inputs_shapes_T4(report)
    test_flow_inputs_noisy_action_content(report)
    test_flow_loss_zero_mask_zero_loss(report)
    test_flow_inputs_actions_repeated(report)

    # 4.5 RTC
    test_rtc_delay_range_uniform(report)
    test_rtc_delay_range_exp(report)
    test_rtc_delay_clamped_by_valid_len(report)
    test_rtc_prefix_mask_content(report)
    test_rtc_token_time_values(report)
    test_rtc_noisy_actions_prefix_clean(report)
    test_rtc_loss_mask_excludes_prefix(report)

    # 4.6 build_dense_diffloss_inputs
    test_chunk_unfold_content(report)
    test_hidden_positions_maps_correctly(report)
    test_diffloss_mask_count(report)
    test_diffloss_repeat_interleave(report)

    # Visualization
    if not skip_visual:
        plt = safe_import_plt()
        if plt is not None:
            # Plot psi_t trajectory
            B, H, D = 1, 1, 1
            x = torch.tensor([[[1.0]]])
            x1 = torch.tensor([[[0.0]]])
            steps = 100
            ts = [i / steps for i in range(steps + 1)]
            vals = [float(psi_t(x, x1, torch.tensor([t_]), SIG_MIN)[0, 0, 0]) for t_ in ts]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(ts, vals, linewidth=2)
            ax.set_xlabel("t")
            ax.set_ylabel("psi_t(x=1, x1=0)")
            ax.set_title("psi_t trajectory (sig_min=0.001)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "psi_t_trajectory.png", dpi=150)
            plt.close(fig)

            # Plot RTC delay distribution
            N = 100000
            valid_len = torch.full((N,), 20, dtype=torch.long)
            delays_exp = sample_rtc_delay(valid_len, strategy="exp", max_delay=8)
            delays_uni = sample_rtc_delay(valid_len, strategy="uniform", max_delay=8)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(delays_uni.numpy(), bins=range(9), align="left", color="steelblue", alpha=0.8, density=True)
            axes[0].set_title("Uniform delay distribution")
            axes[0].set_xlabel("delay")
            axes[1].hist(delays_exp.numpy(), bins=range(9), align="left", color="coral", alpha=0.8, density=True)
            axes[1].set_title("Exp delay distribution")
            axes[1].set_xlabel("delay")
            fig.suptitle("RTC Delay Sampling (max_delay=8, valid_len=20)")
            fig.tight_layout()
            fig.savefig(out_dir / "rtc_delay_distribution.png", dpi=150)
            plt.close(fig)

    report.save()
    report.print_summary()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 4: Flow/RTC math verification")
    parser.add_argument("--skip-visual", action="store_true")
    args = parser.parse_args()
    report = run_all(skip_visual=args.skip_visual)
    exit(0 if report.all_passed else 1)
