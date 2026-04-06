"""
Part 6: Inference Pipeline + Train-Infer Parity Verification.

Tests flow/AR/VLM inference modes and train-infer consistency.

Requires: GPU + model (via Hydra config)
Runs on: GPU

Usage:
    python -m src.tests.full_chain_verification.part6_inference_parity \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml
"""

from __future__ import annotations

import argparse

import hydra
import torch
from omegaconf import OmegaConf

from src.policy.legendvla_inference import (
    infer_ar_action,
    infer_flow_action,
    infer_vlm_generation,
    prepare_prefix_memory,
)
from src.tests.full_chain_verification.utils import (
    CheckResult,
    PhaseReport,
    assert_check,
    get_output_dir,
    tensor_stats,
)

OUTPUT_PART = "part6"

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ── 6.1 Flow inference ─────────────────────────────────────────────────────

def test_flow_inference_shape_and_finite(report: PhaseReport, model, batch: dict) -> None:
    model.eval()
    with torch.no_grad():
        result = infer_flow_action(model, dict(batch))

    if isinstance(result, dict):
        actions = result["generated_actions"]
    else:
        actions = result

    B = batch["input_ids"].shape[0]
    expected = (B, model.num_action_tokens, model.action_dim)
    check_shape = tuple(actions.shape) == expected
    check_finite = bool(torch.isfinite(actions).all())

    report.add(assert_check(
        check_shape and check_finite,
        "6.1a flow inference shape and finite",
        f"shape={tuple(actions.shape)}, expected={expected}, finite={check_finite}",
    ))


def test_flow_inference_invalid_positions_zeroed(report: PhaseReport, model, batch: dict) -> None:
    """Positions beyond n_actions should be zero."""
    test_batch = dict(batch)
    horizon = test_batch["actions"].shape[1] if "actions" in test_batch else None
    if horizon is None or horizon < 2:
        report.add(assert_check(True, "6.1b invalid positions", "SKIPPED: need horizon >= 2"))
        return

    # Force n_actions to be less than full horizon so some positions are invalid
    truncated_n = max(1, horizon // 2)
    test_batch["n_actions"] = torch.full_like(
        test_batch.get("n_actions", torch.tensor([truncated_n])),
        truncated_n,
    )
    # Remove pre-built mask so _build_action_valid_mask reconstructs from n_actions
    test_batch.pop("actions_valid_mask", None)

    model.eval()
    with torch.no_grad():
        result = infer_flow_action(model, test_batch)

    actions = result["generated_actions"] if isinstance(result, dict) else result

    # Check per-sample: positions beyond truncated_n should be zero
    B = actions.shape[0]
    checks = []
    for b in range(B):
        tail = actions[b, truncated_n:]
        checks.append(bool(torch.all(tail == 0)))

    report.add(assert_check(
        all(checks) and len(checks) > 0,
        "6.1b invalid positions zeroed",
        f"truncated_n={truncated_n}, horizon={horizon}, all_zero_tail={all(checks)}",
    ))


def test_flow_inference_deterministic_eval(report: PhaseReport, model, batch: dict) -> None:
    """Same input + eval mode + fixed seed → same output."""
    model.eval()
    with torch.no_grad():
        torch.manual_seed(42)
        r1 = infer_flow_action(model, dict(batch))
        torch.manual_seed(42)
        r2 = infer_flow_action(model, dict(batch))

    a1 = r1["generated_actions"] if isinstance(r1, dict) else r1
    a2 = r2["generated_actions"] if isinstance(r2, dict) else r2
    max_diff = float((a1 - a2).abs().max())

    report.add(assert_check(
        max_diff < 1e-4,
        "6.1c flow inference deterministic",
        f"max_diff={max_diff:.2e}",
    ))


def test_flow_inference_step_count_effect(report: PhaseReport, model, batch: dict) -> None:
    """Different num_inference_steps should produce different outputs."""
    model.eval()
    original_steps = model.num_inference_steps
    results = {}
    try:
        for steps in [1, 10]:
            model.num_inference_steps = steps
            with torch.no_grad():
                torch.manual_seed(0)
                r = infer_flow_action(model, dict(batch))
            results[steps] = (r["generated_actions"] if isinstance(r, dict) else r).clone()
    finally:
        model.num_inference_steps = original_steps

    if 1 in results and 10 in results:
        diff = float((results[1] - results[10]).abs().max())
        report.add(assert_check(
            diff > 1e-3,
            "6.1d different step counts → different outputs",
            f"diff(steps=1 vs 10)={diff:.4f}",
        ))
    else:
        report.add(assert_check(False, "6.1d step count effect", "failed to run"))


# ── 6.2 AR inference ───────────────────────────────────────────────────────

def test_ar_inference_shape_and_finite(report: PhaseReport, model, batch: dict) -> None:
    if model.diffloss is None:
        report.add(assert_check(True, "6.2a AR inference", "SKIPPED: no diffloss module"))
        return

    model.eval()
    with torch.no_grad():
        result = infer_ar_action(model, dict(batch), max_new_tokens=4, temperature=1.0, cfg=1.0)

    actions = result["generated_actions"]
    B = batch["input_ids"].shape[0]
    check_finite = bool(torch.isfinite(actions).all())
    check_batch = actions.shape[0] == B

    report.add(assert_check(
        check_finite and check_batch,
        "6.2a AR inference shape and finite",
        f"shape={tuple(actions.shape)}, finite={check_finite}",
    ))


# ── 6.3 VLM inference ──────────────────────────────────────────────────────

def test_vlm_inference_generates_tokens(report: PhaseReport, model, batch: dict) -> None:
    model.eval()
    with torch.no_grad():
        result = infer_vlm_generation(model, dict(batch), max_new_tokens=5)

    gen_ids = result["generated_ids"]
    report.add(assert_check(
        gen_ids.shape[1] > 0,
        "6.3a VLM inference generates tokens",
        f"generated_length={gen_ids.shape[1]}",
    ))


def test_vlm_inference_respects_max_new_tokens(report: PhaseReport, model, batch: dict) -> None:
    model.eval()
    max_tokens = 5
    with torch.no_grad():
        result = infer_vlm_generation(model, dict(batch), max_new_tokens=max_tokens)

    gen_ids = result["generated_ids"]
    report.add(assert_check(
        gen_ids.shape[1] <= max_tokens,
        "6.3b max_new_tokens respected",
        f"generated={gen_ids.shape[1]}, max={max_tokens}",
    ))


# ── 6.4 Train-Infer parity ─────────────────────────────────────────────────

def test_backbone_hidden_parity(report: PhaseReport, model, batch: dict) -> None:
    """Training forward and inference forward should produce same prompt hidden states."""
    model.eval()
    with torch.no_grad():
        # Training path
        slot_embeds_train = model.build_slot_embeddings(batch, add_action_noise=False)
        train_output = model.forward_backbone_stream(batch, slot_embeds_train)
        train_hidden = train_output.last_hidden_states

        # Inference path
        infer_output = prepare_prefix_memory(model, dict(batch))
        infer_hidden = infer_output.last_hidden_states

    answer_start = int(batch["answer_start_idx"][0])
    # Compare prompt region (before answer_start)
    train_prompt = train_hidden[0, :answer_start].float()
    infer_prompt = infer_hidden[0, :answer_start].float()
    max_diff = float((train_prompt - infer_prompt).abs().max())

    report.add(assert_check(
        max_diff < 1e-3,
        "6.4a backbone hidden parity (prompt region)",
        f"max_diff={max_diff:.2e}",
    ))


def test_prefix_cache_parity(report: PhaseReport, model, batch: dict) -> None:
    """Training and inference prefix caches should match at prompt positions."""
    model.eval()
    with torch.no_grad():
        slot_embeds_train = model.build_slot_embeddings(batch, add_action_noise=False)
        train_output = model.forward_backbone_stream(batch, slot_embeds_train)
        train_cache = train_output.prefix_cache

        infer_output = prepare_prefix_memory(model, dict(batch))
        infer_cache = infer_output.prefix_cache

    if train_cache is None or infer_cache is None:
        report.add(assert_check(
            False, "6.4b prefix cache parity",
            f"train_cache={train_cache is not None}, infer_cache={infer_cache is not None}",
        ))
        return

    # Compare first layer keys at prefix positions
    answer_start = int(batch["answer_start_idx"][0])
    train_k = train_cache.keys[0, 0, :, :answer_start].float()
    infer_k = infer_cache.keys[0, 0, :, :answer_start].float()
    max_diff = float((train_k - infer_k).abs().max())

    report.add(assert_check(
        max_diff < 1e-3,
        "6.4b prefix cache parity (first layer keys)",
        f"max_diff={max_diff:.2e}",
    ))


# ── 6.5 RTC inference ──────────────────────────────────────────────────────

def test_flow_inference_rtc_prefix_preservation(report: PhaseReport, model, batch: dict) -> None:
    """With prev_action_chunk, first d steps should be pinned."""
    model.eval()
    delay = 5
    B = batch["input_ids"].shape[0]
    prev_actions = torch.randn(B, model.num_action_tokens, model.action_dim,
                                device=batch["input_ids"].device, dtype=torch.bfloat16)

    test_batch = dict(batch)
    test_batch["prev_action_chunk"] = prev_actions
    test_batch["inference_delay"] = delay

    with torch.no_grad():
        result = infer_flow_action(model, test_batch)

    actions = result["generated_actions"] if isinstance(result, dict) else result
    prefix_diff = float((actions[:, :delay] - prev_actions[:, :delay]).abs().max())

    report.add(assert_check(
        prefix_diff < 1e-4,
        "6.5a RTC prefix preserved in inference",
        f"prefix_diff={prefix_diff:.2e}, delay={delay}",
    ))


# ── Main ────────────────────────────────────────────────────────────────────

def run_all(config_path: str, skip_visual: bool = False) -> PhaseReport:
    out_dir = get_output_dir(OUTPUT_PART)
    report = PhaseReport("Part 6: Inference + Train-Infer Parity", out_dir)

    print("\n=== Part 6: Inference + Train-Infer Parity ===\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.tests.full_chain_verification.part3_backbone_prefix_cache import (
        build_model_and_collator,
        make_mock_batch,
    )
    model, collator = build_model_and_collator(config_path, device)
    batch = make_mock_batch(collator, model, device)

    # 6.1 Flow inference
    test_flow_inference_shape_and_finite(report, model, batch)
    test_flow_inference_invalid_positions_zeroed(report, model, batch)
    test_flow_inference_deterministic_eval(report, model, batch)
    test_flow_inference_step_count_effect(report, model, batch)

    # 6.2 AR inference
    test_ar_inference_shape_and_finite(report, model, batch)

    # 6.3 VLM inference
    test_vlm_inference_generates_tokens(report, model, batch)
    test_vlm_inference_respects_max_new_tokens(report, model, batch)

    # 6.4 Train-Infer parity
    test_backbone_hidden_parity(report, model, batch)
    test_prefix_cache_parity(report, model, batch)

    # 6.5 RTC inference
    test_flow_inference_rtc_prefix_preservation(report, model, batch)

    report.save()
    report.print_summary()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 6: Inference parity verification")
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--skip-visual", action="store_true")
    args = parser.parse_args()
    report = run_all(config_path=args.config_path, skip_visual=args.skip_visual)
    exit(0 if report.all_passed else 1)
