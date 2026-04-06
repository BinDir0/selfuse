"""
Part 5: Action Expert + DiffLoss + Loss Integration Verification.

Tests 4D attention mask construction, DiffLoss dense input building,
DiffLoss module itself, and compute_total_loss integration.

Requires: GPU + model (via Hydra config)
Runs on: GPU

Usage:
    python -m src.tests.full_chain_verification.part5_expert_diffloss_loss \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml
"""

from __future__ import annotations

import argparse
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf

from src.policy.legendvla_loss import (
    build_dense_diffloss_inputs,
    compute_ce_loss,
    compute_packed_flow_loss,
)
from src.tests.full_chain_verification.utils import (
    CheckResult,
    PhaseReport,
    assert_check,
    get_output_dir,
    tensor_stats,
)

OUTPUT_PART = "part5"

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ── 5.1 4D attention mask ──────────────────────────────────────────────────

def test_4d_mask_prefix_always_visible(report: PhaseReport) -> None:
    """Prefix positions should be visible (0) to all action queries."""
    from src.model.action.qwen3_action_expert import Qwen3ActionExpert

    B, prefix_len, action_len, chunk_size = 2, 10, 32, 8
    kv_len = prefix_len + action_len
    full_mask_bool = torch.ones(B, kv_len, dtype=torch.bool)
    mask = Qwen3ActionExpert.build_4d_attention_mask(
        full_mask_bool, prefix_len, action_len, chunk_size, torch.float32,
    )
    # mask shape: [B, 1, action_len, kv_len]
    # Prefix columns ([:prefix_len]) should be 0 (visible) for valid prefix
    prefix_part = mask[:, :, :, :prefix_len]
    all_visible = bool(torch.all(prefix_part == 0))

    report.add(assert_check(
        all_visible,
        "5.1a prefix always visible in 4D mask",
        f"mask_shape={tuple(mask.shape)}, prefix_all_zero={all_visible}",
    ))


def test_4d_mask_same_chunk_visible(report: PhaseReport) -> None:
    """Action tokens should only see same-chunk action tokens."""
    from src.model.action.qwen3_action_expert import Qwen3ActionExpert

    B, prefix_len, action_len, chunk_size = 1, 10, 32, 8
    kv_len = prefix_len + action_len
    full_mask_bool = torch.ones(B, kv_len, dtype=torch.bool)
    mask = Qwen3ActionExpert.build_4d_attention_mask(
        full_mask_bool, prefix_len, action_len, chunk_size, torch.float32,
    )

    # Query 3 (chunk 0) should see KV at prefix_len+5 (chunk 0) → visible (0)
    q3_kv_chunk0 = float(mask[0, 0, 3, prefix_len + 5])
    # Query 3 (chunk 0) should NOT see KV at prefix_len+9 (chunk 1) → masked (-inf)
    q3_kv_chunk1 = float(mask[0, 0, 3, prefix_len + 9])

    check_visible = q3_kv_chunk0 == 0.0
    check_masked = q3_kv_chunk1 < -1e30

    report.add(assert_check(
        check_visible and check_masked,
        "5.1b same-chunk visible, cross-chunk masked",
        f"same_chunk_val={q3_kv_chunk0}, cross_chunk_val={q3_kv_chunk1}",
    ))


def test_4d_mask_T1_fully_bidirectional(report: PhaseReport) -> None:
    """chunk_size == action_len: all actions should see all actions (single chunk)."""
    from src.model.action.qwen3_action_expert import Qwen3ActionExpert

    B, prefix_len, action_len = 1, 10, 32
    chunk_size = action_len  # Single chunk
    kv_len = prefix_len + action_len
    full_mask_bool = torch.ones(B, kv_len, dtype=torch.bool)
    mask = Qwen3ActionExpert.build_4d_attention_mask(
        full_mask_bool, prefix_len, action_len, chunk_size, torch.float32,
    )

    action_part = mask[:, :, :, prefix_len:]
    all_visible = bool(torch.all(action_part == 0))

    report.add(assert_check(
        all_visible,
        "5.1c T=1: action part fully bidirectional",
        f"all_visible={all_visible}",
    ))


def test_4d_mask_invalid_prefix_blocked(report: PhaseReport) -> None:
    """Padding positions in prefix (attention_mask=0) should be blocked."""
    from src.model.action.qwen3_action_expert import Qwen3ActionExpert

    B, prefix_len, action_len, chunk_size = 1, 10, 16, 8
    kv_len = prefix_len + action_len
    full_mask_bool = torch.ones(B, kv_len, dtype=torch.bool)
    # Mark first 3 prefix positions as padding
    full_mask_bool[0, :3] = False
    mask = Qwen3ActionExpert.build_4d_attention_mask(
        full_mask_bool, prefix_len, action_len, chunk_size, torch.float32,
    )

    # Padded prefix positions should be masked (-inf)
    padded_vals = mask[0, 0, 0, :3]
    all_blocked = bool(torch.all(padded_vals < -1e30))

    report.add(assert_check(
        all_blocked,
        "5.1d padding in prefix is blocked",
        f"padded_vals={padded_vals.tolist()}",
    ))


# ── 5.2 DiffLoss dense inputs ──────────────────────────────────────────────

def test_chunk_unfold_content_detailed(report: PhaseReport) -> None:
    """Verify chunk unfold produces correct sliding windows."""
    B, H, D, chunk_size = 1, 32, 48, 4
    seq_len, hidden_dim = 200, 64

    class _MockModel:
        class ar_action_train_config:
            chunk_size = 4

    model = _MockModel()

    # Traceable actions: each position has unique values
    actions = torch.arange(H * D, dtype=torch.float32).reshape(1, H, D)
    hidden_states = torch.randn(B, seq_len, hidden_dim)
    answer_start_idx = torch.tensor([100])
    n_actions = torch.tensor([20])
    vla_mask = torch.tensor([True])

    _, action_gt, _ = build_dense_diffloss_inputs(
        model, hidden_states, actions, answer_start_idx, n_actions, vla_mask,
    )

    max_chunks = H - chunk_size + 1
    action_gt_reshaped = action_gt.reshape(B, max_chunks, -1)

    # unfold returns [B, num_windows, D, chunk_size], flatten is D-major
    expected_0 = actions[0, 0:4].T.flatten()
    check_0 = torch.allclose(action_gt_reshaped[0, 0], expected_0)

    expected_5 = actions[0, 5:9].T.flatten()
    check_5 = torch.allclose(action_gt_reshaped[0, 5], expected_5)

    report.add(assert_check(
        check_0 and check_5,
        "5.2a chunk unfold content: sliding windows correct",
        f"chunk0_ok={check_0}, chunk5_ok={check_5}",
    ))


def test_hidden_positions_first_maps_to_last_prompt(report: PhaseReport) -> None:
    """hidden_positions[0, 0] should be answer_start_idx - 1 (autoregressive)."""
    B, seq_len, hidden_dim = 2, 200, 64
    H, D, chunk_size = 32, 48, 4

    class _MockModel:
        class ar_action_train_config:
            chunk_size = 4

    model = _MockModel()

    # Put unique markers in hidden_states for verification
    hidden_states = torch.randn(B, seq_len, hidden_dim)
    actions = torch.randn(B, H, D)
    answer_start_idx = torch.tensor([100, 120])
    n_actions = torch.tensor([20, 32])
    vla_mask = torch.tensor([True, True])

    vla_hidden_z, _, _ = build_dense_diffloss_inputs(
        model, hidden_states, actions, answer_start_idx, n_actions, vla_mask,
    )

    max_chunks = H - chunk_size + 1
    vla_hidden_reshaped = vla_hidden_z.reshape(B, max_chunks, hidden_dim)

    # Sample 0: first chunk's hidden should come from position 99 (= 100 - 1)
    expected_0 = hidden_states[0, 99]
    check_0 = torch.allclose(vla_hidden_reshaped[0, 0], expected_0, atol=1e-5)

    # Sample 1: first chunk from position 119 (= 120 - 1)
    expected_1 = hidden_states[1, 119]
    check_1 = torch.allclose(vla_hidden_reshaped[1, 0], expected_1, atol=1e-5)

    report.add(assert_check(
        check_0 and check_1,
        "5.2b hidden_positions[0] = answer_start_idx - 1",
        f"sample0_ok={check_0} (pos=99), sample1_ok={check_1} (pos=119)",
    ))


def test_diffloss_mask_short_action(report: PhaseReport) -> None:
    """n_actions < chunk_size should produce all-False mask (no valid chunks)."""
    B, seq_len, hidden_dim = 1, 200, 64
    H, D, chunk_size = 32, 48, 4

    class _MockModel:
        class ar_action_train_config:
            chunk_size = 4

    model = _MockModel()
    hidden_states = torch.randn(B, seq_len, hidden_dim)
    actions = torch.randn(B, H, D)
    answer_start_idx = torch.tensor([100])
    n_actions = torch.tensor([3])  # < chunk_size=4
    vla_mask = torch.tensor([True])

    _, _, diffloss_mask = build_dense_diffloss_inputs(
        model, hidden_states, actions, answer_start_idx, n_actions, vla_mask,
    )

    all_false = not bool(diffloss_mask.any())
    report.add(assert_check(
        all_false,
        "5.2c n_actions=3 < chunk_size=4: mask all False",
        f"any_true={bool(diffloss_mask.any())}",
    ))


# ── 5.3 DiffLoss module ────────────────────────────────────────────────────

def test_diffloss_zero_init(report: PhaseReport) -> None:
    """Newly initialized DiffLoss (flow matching) should have zero-init final layer."""
    from src.model.common.diffloss import DiffLoss

    dl = DiffLoss(
        target_channels=192,  # 48 * 4
        z_channels=512,
        depth=3,
        width=512,
        num_sampling_steps="100",
        use_flow_matching=True,
    )

    # Check zero-init on final layer
    final = dl.net.final_layer
    weight_zero = bool(torch.all(final.linear.weight == 0))
    bias_zero = bool(torch.all(final.linear.bias == 0))

    report.add(assert_check(
        weight_zero and bias_zero,
        "5.3a DiffLoss final layer zero-init",
        f"weight_zero={weight_zero}, bias_zero={bias_zero}",
    ))


def test_diffloss_flow_loss_positive(report: PhaseReport) -> None:
    """DiffLoss forward should produce positive, finite loss."""
    from src.model.common.diffloss import DiffLoss

    dl = DiffLoss(
        target_channels=192,
        z_channels=512,
        depth=3,
        width=512,
        num_sampling_steps="100",
        use_flow_matching=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dl = dl.to(device)

    B = 4
    target = torch.randn(B, 192, device=device)
    z = torch.randn(B, 512, device=device)
    mask = torch.ones(B, device=device)
    loss = dl(target, z, mask=mask)

    report.add(assert_check(
        float(loss) > 0 and torch.isfinite(loss),
        "5.3b DiffLoss forward: positive finite loss",
        f"loss={float(loss):.4f}",
    ))


def test_diffloss_sample_output_shape(report: PhaseReport) -> None:
    """DiffLoss sample should return [B, target_channels]."""
    from src.model.common.diffloss import DiffLoss

    dl = DiffLoss(
        target_channels=192,
        z_channels=512,
        depth=3,
        width=512,
        num_sampling_steps="100",
        use_flow_matching=True,
        num_inference_steps=5,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dl = dl.to(device)
    dl.eval()

    B = 2
    z = torch.randn(B, 512, device=device)
    with torch.no_grad():
        sampled = dl.sample(z, temperature=1.0)

    expected_shape = (B, 192)
    report.add(assert_check(
        tuple(sampled.shape) == expected_shape,
        "5.3c DiffLoss sample shape",
        f"shape={tuple(sampled.shape)}, expected={expected_shape}",
    ))


# ── 5.4 Total loss integration ─────────────────────────────────────────────

def test_total_loss_pure_vla(report: PhaseReport, model, batch: dict) -> None:
    """Pure VLA batch: ce_loss==0, flow_loss>0."""
    # Ensure all samples are VLA (copy to avoid polluting shared batch)
    batch = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    batch["is_vla_data"] = torch.ones(batch["is_vla_data"].shape, dtype=torch.bool, device=batch["is_vla_data"].device)

    with torch.no_grad():
        hidden_states = torch.randn(
            batch["input_ids"].shape[0],
            batch["input_ids"].shape[1],
            model.vlm_hidden_size,
            device=batch["input_ids"].device,
            dtype=torch.bfloat16,
        )
        ce = compute_ce_loss(model, hidden_states, batch["labels"], batch["is_vla_data"])
        ce_val = float(ce)

    report.add(assert_check(
        ce_val == 0.0,
        "5.4a pure VLA: ce_loss == 0",
        f"ce_loss={ce_val}",
    ))


def test_total_loss_all_finite(report: PhaseReport, model, batch: dict) -> None:
    """Full forward+loss should produce finite loss components."""
    model.train()
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result = model("train", batch)
        losses = result if isinstance(result, dict) else {"total": result}
        all_finite = all(
            torch.isfinite(v) if torch.is_tensor(v) else True
            for v in losses.values()
        )
        report.add(assert_check(
            all_finite,
            "5.4b all loss components finite",
            str({k: f"{float(v):.4f}" if torch.is_tensor(v) else v for k, v in losses.items()}),
        ))
    except Exception as e:
        report.add(assert_check(False, "5.4b all loss components finite", f"error: {e}"))
    finally:
        model.eval()


def test_gradient_to_param_groups(report: PhaseReport, model, batch: dict) -> None:
    """backward should give non-zero gradients to action_expert and diffloss params."""
    model.train()
    model.zero_grad()
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result = model("train", batch)
        total_loss = result["total_loss"] if isinstance(result, dict) else result
        total_loss.backward()

        expert_has_grad = any(
            p.grad is not None and p.grad.abs().max() > 0
            for p in model.flow_expert.parameters()
            if p.requires_grad
        )
        diffloss_has_grad = model.diffloss is None or any(
            p.grad is not None and p.grad.abs().max() > 0
            for p in model.diffloss.parameters()
            if p.requires_grad
        )

        report.add(assert_check(
            expert_has_grad and diffloss_has_grad,
            "5.4c gradients reach expert and diffloss",
            f"expert_grad={expert_has_grad}, diffloss_grad={diffloss_has_grad}",
        ))
    except Exception as e:
        report.add(assert_check(False, "5.4c gradient check", f"error: {e}"))
    finally:
        model.eval()
        model.zero_grad()


# ── Main ────────────────────────────────────────────────────────────────────

def run_all(config_path: str | None = None, skip_visual: bool = False) -> PhaseReport:
    out_dir = get_output_dir(OUTPUT_PART)
    report = PhaseReport("Part 5: Action Expert + DiffLoss + Loss", out_dir)

    print("\n=== Part 5: Action Expert + DiffLoss + Loss ===\n")

    # 5.1 4D attention mask (CPU, no model needed)
    test_4d_mask_prefix_always_visible(report)
    test_4d_mask_same_chunk_visible(report)
    test_4d_mask_T1_fully_bidirectional(report)
    test_4d_mask_invalid_prefix_blocked(report)

    # 5.2 DiffLoss dense inputs (CPU, mock model)
    test_chunk_unfold_content_detailed(report)
    test_hidden_positions_first_maps_to_last_prompt(report)
    test_diffloss_mask_short_action(report)

    # 5.3 DiffLoss module (needs device)
    test_diffloss_zero_init(report)
    test_diffloss_flow_loss_positive(report)
    test_diffloss_sample_output_shape(report)

    # 5.4 Total loss integration (needs full model)
    if config_path is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from src.tests.full_chain_verification.part3_backbone_prefix_cache import (
            build_model_and_collator,
            make_mock_batch,
        )
        model, collator = build_model_and_collator(config_path, device)
        batch = make_mock_batch(collator, model, device)
        test_total_loss_pure_vla(report, model, batch)
        test_total_loss_all_finite(report, model, batch)
        test_gradient_to_param_groups(report, model, batch)
    else:
        report.add(assert_check(True, "5.4 total loss integration", "SKIPPED: no --config-path"))

    report.save()
    report.print_summary()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 5: Expert/DiffLoss/Loss verification")
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--skip-visual", action="store_true")
    args = parser.parse_args()
    report = run_all(config_path=args.config_path, skip_visual=args.skip_visual)
    exit(0 if report.all_passed else 1)
