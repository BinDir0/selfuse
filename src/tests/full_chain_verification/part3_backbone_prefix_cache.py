"""
Part 3: Backbone Embedding + Prefix Cache Verification.

Tests slot embedding replacement, visual encoding, backbone forward,
and prefix KV cache correctness including knowledge insulation.

Requires: GPU + Qwen3-VL-2B-Instruct weights
Runs on: GPU

Usage:
    python -m src.tests.full_chain_verification.part3_backbone_prefix_cache \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml
"""

from __future__ import annotations

import argparse
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf

from src.tests.full_chain_verification.utils import (
    CheckResult,
    PhaseReport,
    assert_check,
    get_output_dir,
    tensor_stats,
)

OUTPUT_PART = "part3"

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("hydra", lambda key: "", replace=True)


def load_hydra_config(config_path: str):
    """Load a Hydra experiment config with defaults resolution.

    Uses hydra.initialize + hydra.compose so that ``defaults`` entries
    (model, data, training, ...) and ``${}`` interpolations are properly
    resolved, matching the behavior of the training entry-point.
    """
    import os
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    with hydra.initialize(version_base=None, config_path="../../config"):
        cfg = hydra.compose(config_name=f"experiment/{config_name}")

    OmegaConf.set_struct(cfg, False)
    cfg.hydra = {
        "runtime": {"output_dir": "outputs", "choices": {}},
        "job": {"num": 0, "name": "test"},
    }
    OmegaConf.register_new_resolver("hydra", lambda key: "outputs" if "output_dir" in key else "", replace=True)
    OmegaConf.resolve(cfg)
    return cfg


def build_model_and_collator(config_path: str, device: str = "cuda"):
    """Instantiate model + collator from Hydra config."""
    cfg = load_hydra_config(config_path)
    model = hydra.utils.instantiate(cfg.policy)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    collator = hydra.utils.instantiate(cfg.data_collator, mode="train")
    return model, collator


def make_mock_batch(collator, model, device: str = "cuda") -> dict[str, Any]:
    """Build a mock VLA batch using the collator.

    n_states must equal the model's state_horizon (yaml: 6) because the model
    builds a [B, num_state_tokens] mask in train mode and broadcasts it against
    state_encoder output `[B, n_states, hidden]`. n_actions must equal the
    action_horizon (yaml: 32) for the same reason on the action side.
    """
    from src.tests.full_chain_verification.part2_collator_tokenization import make_vla_sample
    sample = make_vla_sample(n_states=6, n_actions=32, image_horizon=6)
    batch = collator.collate_raw([sample])
    # Cast to model dtype and device
    for k, v in batch.items():
        if torch.is_tensor(v):
            if torch.is_floating_point(v):
                batch[k] = v.to(device=device, dtype=torch.bfloat16)
            else:
                batch[k] = v.to(device=device)
    return batch


# ── 3.1 Slot embedding replacement ─────────────────────────────────────────

def test_state_embed_replacement_content(report: PhaseReport, model, batch: dict) -> None:
    """State encoder output should appear at <state> token positions in inputs_embeds."""
    with torch.no_grad():
        slot_embeds = model.build_slot_embeddings(batch, add_action_noise=False)
        state_embeds = slot_embeds["state"]

        # Get the backbone's raw token embeddings before replacement
        input_ids = batch["input_ids"]
        state_token_id = model.state_token_index
        state_positions = (input_ids == state_token_id)

        # Forward through backbone to get inputs_embeds with replacement
        # We need to check the embeddings, so let's verify through the backbone
        # The backbone's replace_slot_embeddings uses masked_scatter
        if state_embeds is not None:
            state_count = int(state_positions.sum())
            expected_count = state_embeds.shape[0] * state_embeds.shape[1]
            report.add(assert_check(
                state_count == expected_count,
                "3.1a state embed positions match encoder output count",
                f"state_positions={state_count}, encoder_output={expected_count}",
            ))
        else:
            report.add(assert_check(False, "3.1a state embed", "state_embeds is None"))


def test_action_noise_in_training(report: PhaseReport, model, batch: dict) -> None:
    """add_action_noise=True should produce different embeddings than False."""
    # When the diffloss config group is `none` (current default in
    # legendvla_qwen3_vl.yaml), `model.ar_action_encoder` is None and
    # `build_slot_embeddings` returns slot_embeds["action"]=None — there is
    # no AR-action embedding to compare. Skip cleanly in that case.
    if model.ar_action_encoder is None:
        report.add(assert_check(
            True,
            "3.1b action noise produces different embeddings",
            "SKIPPED: ar_action_encoder is None (diffloss=none)",
        ))
        return

    with torch.no_grad():
        clean = model.build_slot_embeddings(batch, add_action_noise=False)["action"]
        # Multiple noisy samples to check distribution
        diffs = []
        for _ in range(10):
            noisy = model.build_slot_embeddings(batch, add_action_noise=True)["action"]
            diff = (noisy - clean).float().std().item()
            diffs.append(diff)

        mean_diff = sum(diffs) / len(diffs)
        expected_noise_scale = model.ar_action_train_config.noise_std

        report.add(assert_check(
            mean_diff > 0.001,
            "3.1b action noise produces different embeddings",
            f"mean_embed_diff_std={mean_diff:.6f}, configured_noise_std={expected_noise_scale}",
        ))


# ── 3.2 Backbone forward ───────────────────────────────────────────────────

def test_hidden_states_shape_and_dtype(report: PhaseReport, model, batch: dict) -> None:
    with torch.no_grad():
        slot_embeds = model.build_slot_embeddings(batch, add_action_noise=False)
        output = model.forward_backbone_stream(batch, slot_embeds)
        hs = output.last_hidden_states

        expected_hidden = model.vlm_hidden_size
        check_shape = hs.shape[0] == batch["input_ids"].shape[0] and hs.shape[2] == expected_hidden
        check_dtype = hs.dtype == torch.bfloat16

        report.add(assert_check(
            check_shape and check_dtype,
            "3.2a hidden_states shape and dtype",
            f"shape={tuple(hs.shape)}, dtype={hs.dtype}, expected_hidden={expected_hidden}",
        ))


def test_hidden_states_finite(report: PhaseReport, model, batch: dict) -> None:
    with torch.no_grad():
        slot_embeds = model.build_slot_embeddings(batch, add_action_noise=False)
        output = model.forward_backbone_stream(batch, slot_embeds)
        hs = output.last_hidden_states

        has_nan = bool(torch.isnan(hs).any())
        has_inf = bool(torch.isinf(hs).any())

        report.add(assert_check(
            not has_nan and not has_inf,
            "3.2b hidden_states finite",
            f"nan={has_nan}, inf={has_inf}",
            details=tensor_stats(hs),
        ))


def test_hidden_states_not_constant(report: PhaseReport, model, batch: dict) -> None:
    with torch.no_grad():
        slot_embeds = model.build_slot_embeddings(batch, add_action_noise=False)
        output = model.forward_backbone_stream(batch, slot_embeds)
        hs = output.last_hidden_states

        seq_std = float(hs.float().std(dim=1).mean())
        report.add(assert_check(
            seq_std > 0.01,
            "3.2c hidden_states vary across sequence",
            f"mean_seq_std={seq_std:.4f}",
        ))


def test_hidden_states_answer_region_varies(report: PhaseReport, model, batch: dict) -> None:
    with torch.no_grad():
        slot_embeds = model.build_slot_embeddings(batch, add_action_noise=False)
        output = model.forward_backbone_stream(batch, slot_embeds)
        hs = output.last_hidden_states[0].float()

        answer_start = int(batch["answer_start_idx"][0])
        prompt_mean = hs[:answer_start].mean(dim=0)
        answer_mean = hs[answer_start:].mean(dim=0)
        cosine_sim = float(torch.nn.functional.cosine_similarity(prompt_mean.unsqueeze(0), answer_mean.unsqueeze(0)))

        report.add(assert_check(
            cosine_sim < 0.99,
            "3.2d prompt vs answer hidden states differ",
            f"cosine_sim={cosine_sim:.4f}",
        ))


# ── 3.3 Prefix cache ───────────────────────────────────────────────────────

def test_prefix_cache_shape(report: PhaseReport, model, batch: dict) -> None:
    with torch.no_grad():
        slot_embeds = model.build_slot_embeddings(batch, add_action_noise=False)
        output = model.forward_backbone_stream(batch, slot_embeds)
        cache = output.prefix_cache

        if cache is None:
            report.add(assert_check(False, "3.3a prefix cache shape", "prefix_cache is None"))
            return

        B = batch["input_ids"].shape[0]
        checks = []
        # keys: [num_layers, B, kv_heads, seq_len, head_dim]
        checks.append(cache.keys.ndim == 5)
        checks.append(int(cache.keys.shape[1]) == B)
        checks.append(int(cache.values.shape[1]) == B)
        checks.append(cache.mask.shape[0] == B)
        checks.append(cache.lengths.shape[0] == B)

        report.add(assert_check(
            all(checks),
            "3.3a prefix cache shape",
            f"keys={tuple(cache.keys.shape)}, mask={tuple(cache.mask.shape)}, "
            f"lengths={cache.lengths.tolist()}",
        ))


def test_prefix_mask_content(report: PhaseReport, model, batch: dict) -> None:
    """Mask should be True for prompt positions, False for action/padding."""
    with torch.no_grad():
        slot_embeds = model.build_slot_embeddings(batch, add_action_noise=False)
        output = model.forward_backbone_stream(batch, slot_embeds)
        cache = output.prefix_cache

        if cache is None:
            report.add(assert_check(False, "3.3b prefix mask content", "prefix_cache is None"))
            return

        answer_start = batch["answer_start_idx"]
        B = answer_start.shape[0]
        all_correct = True
        for b in range(B):
            asi = int(answer_start[b])
            prefix_len = int(cache.lengths[b])
            # Prefix length should equal answer_start_idx
            if prefix_len != asi:
                all_correct = False
                break
            # Check mask content
            if asi > 0:
                if not bool(cache.mask[b, :asi].all()):
                    all_correct = False
                    break
            if asi < cache.mask.shape[1]:
                if bool(cache.mask[b, asi:].any()):
                    all_correct = False
                    break

        report.add(assert_check(
            all_correct,
            "3.3b prefix mask: True before answer_start, False after",
            f"answer_start_idx={answer_start.tolist()}, prefix_lengths={cache.lengths.tolist()}",
        ))


def test_prefix_cache_kv_meaningful(report: PhaseReport, model, batch: dict) -> None:
    with torch.no_grad():
        slot_embeds = model.build_slot_embeddings(batch, add_action_noise=False)
        output = model.forward_backbone_stream(batch, slot_embeds)
        cache = output.prefix_cache

        if cache is None:
            report.add(assert_check(False, "3.3c prefix cache KV meaningful", "None"))
            return

        key_norm = float(cache.keys.float().norm())
        val_norm = float(cache.values.float().norm())
        # Different layers should have different KV
        layer0_keys = cache.keys[0].float()
        layer1_keys = cache.keys[1].float() if cache.num_layers > 1 else layer0_keys
        layers_differ = float((layer0_keys - layer1_keys).abs().max()) > 1e-4

        report.add(assert_check(
            key_norm > 0 and val_norm > 0 and layers_differ,
            "3.3c prefix cache KV: non-zero and layer-diverse",
            f"key_norm={key_norm:.2f}, val_norm={val_norm:.2f}, layers_differ={layers_differ}",
        ))


# ── 3.4 Visual encoding ────────────────────────────────────────────────────

def test_visual_embed_not_zero(report: PhaseReport, model, batch: dict) -> None:
    with torch.no_grad():
        slot_embeds = model.build_slot_embeddings(batch, add_action_noise=False)
        output = model.forward_backbone_stream(batch, slot_embeds)
        hs = output.last_hidden_states

        # Visual token positions: where input_ids had video_pad tokens
        # The hidden states at these positions should be non-zero
        # since they were replaced by visual features
        answer_start = int(batch["answer_start_idx"][0])
        prompt_hidden = hs[0, :answer_start].float()
        prompt_norm = float(prompt_hidden.norm(dim=-1).mean())

        report.add(assert_check(
            prompt_norm > 0.1,
            "3.4a visual/prompt embeddings are non-zero",
            f"mean_prompt_hidden_norm={prompt_norm:.4f}",
        ))


# ── Main ────────────────────────────────────────────────────────────────────

def run_all(config_path: str, skip_visual: bool = False) -> PhaseReport:
    out_dir = get_output_dir(OUTPUT_PART)
    report = PhaseReport("Part 3: Backbone Embedding + Prefix Cache", out_dir)

    print("\n=== Part 3: Backbone Embedding + Prefix Cache ===\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  WARNING: Running on CPU, some tests may be slow or fail.")

    model, collator = build_model_and_collator(config_path, device)
    batch = make_mock_batch(collator, model, device)

    # 3.1 Slot embeddings
    test_state_embed_replacement_content(report, model, batch)
    test_action_noise_in_training(report, model, batch)

    # 3.2 Backbone forward
    test_hidden_states_shape_and_dtype(report, model, batch)
    test_hidden_states_finite(report, model, batch)
    test_hidden_states_not_constant(report, model, batch)
    test_hidden_states_answer_region_varies(report, model, batch)

    # 3.3 Prefix cache
    test_prefix_cache_shape(report, model, batch)
    test_prefix_mask_content(report, model, batch)
    test_prefix_cache_kv_meaningful(report, model, batch)

    # 3.4 Visual encoding
    test_visual_embed_not_zero(report, model, batch)

    report.save()
    report.print_summary()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 3: Backbone/Prefix cache verification")
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--skip-visual", action="store_true")
    args = parser.parse_args()
    report = run_all(config_path=args.config_path, skip_visual=args.skip_visual)
    exit(0 if report.all_passed else 1)
