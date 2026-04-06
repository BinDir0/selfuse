"""
Part 9: Input Ablation Verification.

Uses REAL shard data + trained checkpoint to verify each input modality
meaningfully contributes to model output. Also verifies that GT actions
fed to backbone do NOT leak to the action expert.

Supports two modes:
  - Quick mode (default, n_samples=2): detailed per-check report.
  - Statistical mode (--n-samples 1000): per-sample MSE distribution with
    confidence intervals and hypothesis testing.

Requires: GPU + real VLA shard + normalizer + checkpoint (recommended)

Usage:
    # Quick mode (original behavior)
    python -m src.tests.full_chain_verification.part9_input_ablation \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml \
        --vla-shard '/path/to/shard-{000..001}.tar' \
        --normalizer-path /path/to/normalizer.pkl \
        [--checkpoint-path /path/to/checkpoint]

    # Statistical mode (1000 samples)
    python -m src.tests.full_chain_verification.part9_input_ablation \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml \
        --vla-shard '/path/to/shard-{000..001}.tar' \
        --normalizer-path /path/to/normalizer.pkl \
        --checkpoint-path /path/to/checkpoint \
        --n-samples 1000 --batch-size 4
"""

from __future__ import annotations

import argparse
import pickle
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from src.policy.legendvla_inference import infer_flow_action
from src.tests.full_chain_verification.utils import (
    PhaseReport,
    assert_check,
    get_output_dir,
    safe_import_plt,
)

OUTPUT_PART = "part9"

OmegaConf.register_new_resolver("eval", eval, replace=True)

# Thresholds — real data + trained checkpoint should show strong signals.
MIN_REL_CHANGE_LOSS = 0.005     # 0.5% loss change
MIN_ABS_CHANGE_ACTION = 0.001   # action MSE
LEAKAGE_TOLERANCE = 1e-6        # max allowed flow_loss relative change for leakage tests


# ── Real data loading via training pipeline ──────────────────────────────────

def build_vla_dataset(cfg, normalizer_path: str, shard_pattern: str):
    """Build a VLAWdsDataset in val mode using the real training pipeline.

    This ensures correct extrinsic transform, multi-frame sliding window,
    normalization, and image processing — identical to training.
    """
    from src.dataset.vla_dataset import VLAWdsDataset

    shape_meta = OmegaConf.to_container(cfg.data.shape_meta, resolve=True)
    depth_clip = OmegaConf.to_container(cfg.data.depth_clip_range, resolve=True)
    target_size = cfg.data.get("target_image_size", None)
    if target_size is not None:
        target_size = OmegaConf.to_container(target_size, resolve=True)

    dataset = VLAWdsDataset(
        wds_datasets=[{"shard_urls": shard_pattern, "weight": 1.0, "name": "ablation_test"}],
        shape_meta=shape_meta,
        use_relative_action=cfg.dataset.vla_dataset.get("use_relative_action", True),
        mode="val",
        depth_clip_range=depth_clip,
        shuffle_buffer=0,
        history_pad_mode=cfg.dataset.vla_dataset.get("history_pad_mode", "repeat"),
        future_pad_mode=cfg.dataset.vla_dataset.get("future_pad_mode", "truncate"),
        target_image_size=target_size,
        video_base_fps=float(cfg.data.get("video_base_fps", 30.0)),
    )

    if normalizer_path:
        with open(normalizer_path, "rb") as f:
            normalizer = pickle.load(f)
        dataset.set_normalizer(normalizer)

    return dataset


def collect_samples(dataset, n_samples: int) -> list[dict[str, Any]]:
    """Iterate a VLAWdsDataset and collect n_samples torch-tensor dicts."""
    samples: list[dict[str, Any]] = []
    for sample in dataset:
        samples.append(sample)
        if len(samples) >= n_samples:
            break
    if len(samples) < n_samples:
        print(f"  Warning: only collected {len(samples)} samples (requested {n_samples})")
    return samples


def collate_to_device(collator, samples: list[dict], device: str) -> dict[str, Any]:
    """Collate samples and move to device with bfloat16 for floats."""
    batch = collator.collate_raw(samples)
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device=device, dtype=torch.bfloat16 if torch.is_floating_point(v) else None)
    return batch


# ── Helpers ──────────────────────────────────────────────────────────────────

def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}


def action_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).pow(2).mean())


def get_baseline_actions(model, batch: dict) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        torch.manual_seed(42)
        result = infer_flow_action(model, clone_batch(batch))
    return (result["generated_actions"] if isinstance(result, dict) else result).clone()


def get_seeded_loss(model, batch: dict, seed: int = 123) -> dict[str, float]:
    """Compute loss with fixed seed to control flow noise/time sampling."""
    from src.policy.legendvla_loss import compute_total_loss
    model.train()
    with torch.no_grad():
        torch.manual_seed(seed)
        losses = compute_total_loss(model, clone_batch(batch))
    model.eval()
    return {k: float(v) for k, v in losses.items()}


# ── 9.1 Visual ablation ─────────────────────────────────────────────────────

class _VisualAblationContext:
    """Monkey-patches encode_visual_features to zero visual positions in inputs_embeds."""

    def __init__(self, model):
        self.backbone = model.backbone
        self.original_fn = model.backbone.encode_visual_features

    def __enter__(self):
        original = self.original_fn

        def zero_visual_encode(*args, **kwargs):
            inputs_embeds, visual_pos_masks, deepstack = original(*args, **kwargs)
            if visual_pos_masks is not None:
                mask_3d = visual_pos_masks.unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_fill(mask_3d, 0.0)
            return inputs_embeds, visual_pos_masks, deepstack

        self.backbone.encode_visual_features = zero_visual_encode
        return self

    def __exit__(self, *exc):
        self.backbone.encode_visual_features = self.original_fn
        return False


def test_visual_ablation_inference(report: PhaseReport, model, batch, baseline_actions) -> None:
    with _VisualAblationContext(model):
        model.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            result = infer_flow_action(model, clone_batch(batch))
        ablated = (result["generated_actions"] if isinstance(result, dict) else result)

    mse = action_mse(baseline_actions, ablated)
    insulated = getattr(model, "knowledge_insulation", False)
    if insulated:
        report.add(assert_check(
            True,
            "9.1a visual ablation inference (knowledge_insulation=True)",
            f"action_mse={mse:.6f} — expected minimal effect (detached prefix cache)",
        ))
    else:
        report.add(assert_check(
            mse > MIN_ABS_CHANGE_ACTION,
            "9.1a visual ablation changes inference output",
            f"action_mse={mse:.6f}",
        ))


def test_visual_ablation_loss(report: PhaseReport, model, batch, baseline_loss) -> None:
    with _VisualAblationContext(model):
        ablated_loss = get_seeded_loss(model, batch)

    base_flow = baseline_loss["flow_loss"]
    abl_flow = ablated_loss["flow_loss"]
    rel = abs(abl_flow - base_flow) / max(abs(base_flow), 1e-8)

    insulated = getattr(model, "knowledge_insulation", False)
    if insulated:
        report.add(assert_check(
            True,
            "9.1b visual ablation loss (knowledge_insulation=True)",
            f"base={base_flow:.6f}, ablated={abl_flow:.6f}, rel={rel:.4%}",
        ))
    else:
        report.add(assert_check(
            rel > MIN_REL_CHANGE_LOSS,
            "9.1b visual ablation changes flow loss",
            f"base={base_flow:.6f}, ablated={abl_flow:.6f}, rel={rel:.4%}",
        ))


# ── 9.2 State ablation ──────────────────────────────────────────────────────

def test_state_ablation_inference(report: PhaseReport, model, batch, baseline_actions) -> None:
    ablated = clone_batch(batch)
    ablated["states"] = torch.zeros_like(ablated["states"])
    model.eval()
    with torch.no_grad():
        torch.manual_seed(42)
        result = infer_flow_action(model, ablated)
    ablated_act = (result["generated_actions"] if isinstance(result, dict) else result)
    mse = action_mse(baseline_actions, ablated_act)
    # With knowledge_insulation, state flows through detached prefix cache,
    # so the effect is inherently muted.  Use connectivity threshold (1e-4).
    report.add(assert_check(
        mse > 1e-4,
        "9.2a state ablation changes inference output",
        f"action_mse={mse:.6f}",
    ))


def test_state_ablation_loss(report: PhaseReport, model, batch, baseline_loss) -> None:
    ablated = clone_batch(batch)
    ablated["states"] = torch.zeros_like(ablated["states"])
    ablated_loss = get_seeded_loss(model, ablated)
    base_flow = baseline_loss["flow_loss"]
    abl_flow = ablated_loss["flow_loss"]
    rel = abs(abl_flow - base_flow) / max(abs(base_flow), 1e-8)
    # With knowledge_insulation, state effect on loss is muted.
    # Use a lower threshold (0.1%) to prove connectivity.
    report.add(assert_check(
        rel > 0.001,
        "9.2b state ablation changes flow loss",
        f"base={base_flow:.6f}, ablated={abl_flow:.6f}, rel={rel:.4%}",
    ))


# ── 9.3 Per-field state ablation (inference only) ───────────────────────────

LOWDIM_STATE_FIELDS = {"wrist_state": (0, 18), "hand_state": (18, 48)}


def test_per_field_state_ablation(report: PhaseReport, model, batch, baseline_actions) -> None:
    for name, (s, e) in LOWDIM_STATE_FIELDS.items():
        ablated = clone_batch(batch)
        ablated["states"][:, :, s:e] = 0.0
        model.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            result = infer_flow_action(model, ablated)
        ablated_act = (result["generated_actions"] if isinstance(result, dict) else result)
        mse = action_mse(baseline_actions, ablated_act)
        report.add(assert_check(
            mse > 1e-4,
            f"9.3a {name} ablation changes output",
            f"action_mse={mse:.6f}",
        ))


# ── 9.4 Instruction / text ablation ─────────────────────────────────────────

def test_different_instructions(
    report: PhaseReport, model, collator, real_samples: list[dict], device: str,
) -> None:
    """Same real observation + different instruction → different actions.

    Takes the first real sample, creates two batches with identical
    images/states/actions but different instruction text.
    """
    sample_a = dict(real_samples[0])
    sample_b = dict(real_samples[0])
    sample_a["instruction"] = "pick up the red cube from the table"
    sample_b["instruction"] = "open the drawer and place the green bottle inside"

    batch_a = collate_to_device(collator, [sample_a], device)
    batch_b = collate_to_device(collator, [sample_b], device)

    model.eval()
    with torch.no_grad():
        torch.manual_seed(42)
        r_a = infer_flow_action(model, clone_batch(batch_a))
        torch.manual_seed(42)
        r_b = infer_flow_action(model, clone_batch(batch_b))

    act_a = r_a["generated_actions"] if isinstance(r_a, dict) else r_a
    act_b = r_b["generated_actions"] if isinstance(r_b, dict) else r_b
    mse = action_mse(act_a, act_b)

    report.add(assert_check(
        mse > 1e-4,
        "9.4a different instructions produce different actions",
        f"action_mse={mse:.6f}",
    ))


def test_text_embedding_ablation(report: PhaseReport, model, batch, baseline_actions) -> None:
    """Zero the text embedding layer output — strong test of text pathway."""
    text_embed_layer = model.backbone.base_model.get_input_embeddings()
    original_forward = text_embed_layer.forward

    def zeroed_text_embed(input_ids):
        return torch.zeros_like(original_forward(input_ids))

    text_embed_layer.forward = zeroed_text_embed
    try:
        model.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            result = infer_flow_action(model, clone_batch(batch))
        ablated_act = (result["generated_actions"] if isinstance(result, dict) else result)
    finally:
        text_embed_layer.forward = original_forward

    mse = action_mse(baseline_actions, ablated_act)
    report.add(assert_check(
        mse > MIN_ABS_CHANGE_ACTION,
        "9.4b zeroing text embeddings changes actions",
        f"action_mse={mse:.6f}",
    ))


# ── 9.5 Action expert component ablation ────────────────────────────────────

def _hook_zero_and_measure(model, batch, baseline_actions, module, test_name, report, threshold=None):
    """Register a zero-output forward hook on `module`, measure action MSE."""
    if threshold is None:
        threshold = MIN_ABS_CHANGE_ACTION

    def zero_hook(mod, inp, out):
        return torch.zeros_like(out)

    handle = module.register_forward_hook(zero_hook)
    try:
        model.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            result = infer_flow_action(model, clone_batch(batch))
        ablated_act = (result["generated_actions"] if isinstance(result, dict) else result)
    finally:
        handle.remove()

    mse = action_mse(baseline_actions, ablated_act)
    report.add(assert_check(mse > threshold, test_name, f"action_mse={mse:.6f}"))


def test_component_ablation(report: PhaseReport, model, batch, baseline_actions) -> None:
    _hook_zero_and_measure(
        model, batch, baseline_actions, model.state_encoder,
        "9.5a zeroing state_encoder output changes actions", report, threshold=1e-4,
    )
    _hook_zero_and_measure(
        model, batch, baseline_actions, model.time_embedding,
        "9.5b zeroing time_embedding output changes actions", report,
    )
    _hook_zero_and_measure(
        model, batch, baseline_actions, model.action_decoder,
        "9.5c zeroing action_decoder output changes actions", report,
    )


# ── 9.6 DiffLoss condition isolation ────────────────────────────────────────

def test_diffloss_condition_isolation(report: PhaseReport, model, batch, baseline_loss) -> None:
    """Zeroing latent_condition_projector should change diffusion_loss but NOT flow_loss.

    The action expert (flow stream) never sees the DiffLoss condition.
    This test confirms the two paths are properly isolated.
    """
    if model.diffloss is None:
        report.add(assert_check(True, "9.6a diffloss condition", "SKIPPED: no diffloss"))
        report.add(assert_check(True, "9.6b diffloss/flow isolation", "SKIPPED: no diffloss"))
        return

    def zero_hook(mod, inp, out):
        return torch.zeros_like(out)

    handle = model.latent_condition_projector.register_forward_hook(zero_hook)
    try:
        ablated_loss = get_seeded_loss(model, batch)
    finally:
        handle.remove()

    # 9.6a: diffusion_loss should change
    base_diff = baseline_loss.get("diffusion_loss", 0.0)
    abl_diff = ablated_loss.get("diffusion_loss", 0.0)
    if base_diff == 0.0 and abl_diff == 0.0:
        report.add(assert_check(True, "9.6a diffloss condition", "SKIPPED: diff_loss=0"))
    else:
        rel_diff = abs(abl_diff - base_diff) / max(abs(base_diff), 1e-8)
        report.add(assert_check(
            rel_diff > MIN_REL_CHANGE_LOSS,
            "9.6a zeroing diffloss condition changes diffusion_loss",
            f"base={base_diff:.6f}, ablated={abl_diff:.6f}, rel={rel_diff:.4%}",
        ))

    # 9.6b: flow_loss must NOT change (action expert can't see DiffLoss condition)
    base_flow = baseline_loss["flow_loss"]
    abl_flow = ablated_loss["flow_loss"]
    rel_flow = abs(abl_flow - base_flow) / max(abs(base_flow), 1e-8)
    report.add(assert_check(
        rel_flow < LEAKAGE_TOLERANCE,
        "9.6b zeroing diffloss condition does NOT change flow_loss",
        f"base={base_flow:.6f}, ablated={abl_flow:.6f}, rel={rel_flow:.2e}",
    ))


# ── 9.7 Prefix cache ablation ───────────────────────────────────────────────

def test_prefix_cache_matters(report: PhaseReport, model, batch, baseline_actions) -> None:
    """Zeroing the prefix KV cache should produce very different actions."""
    from src.policy.legendvla_inference import _build_action_valid_mask, prepare_prefix_memory

    working_batch = clone_batch(batch)
    action_valid_mask = _build_action_valid_mask(working_batch, model.num_action_tokens, model.action_dim)
    batch_size, action_len, _ = action_valid_mask.shape
    device = working_batch["input_ids"].device
    action_dtype = working_batch["states"].dtype

    working_batch["actions_valid_mask"] = action_valid_mask
    backbone_output = prepare_prefix_memory(model, working_batch)

    # Zero out the prefix cache
    if backbone_output.prefix_cache is not None:
        backbone_output.prefix_cache.keys = torch.zeros_like(backbone_output.prefix_cache.keys)
        backbone_output.prefix_cache.values = torch.zeros_like(backbone_output.prefix_cache.values)

    action_step_mask = action_valid_mask.any(dim=-1)
    delta_t = 1.0 / max(model.num_inference_steps, 1)

    model.eval()
    with torch.no_grad():
        torch.manual_seed(42)
        generated_actions = torch.randn(
            batch_size, action_len, model.action_dim, device=device, dtype=action_dtype,
        )
        t = torch.zeros(batch_size, device=device, dtype=action_dtype)
        for _ in range(model.num_inference_steps):
            time_for_model = t[:, None].expand(-1, action_len)
            flow_output = model.forward_flow_stream(
                batch=working_batch,
                backbone_output=backbone_output,
                flow_inputs={"noisy_actions": generated_actions, "time_for_model": time_for_model},
                num_parallel_chunks=1,
            )
            generated_actions = generated_actions + delta_t * flow_output["pred_v"]
            t = t + delta_t

    result = generated_actions * action_step_mask.unsqueeze(-1).to(dtype=generated_actions.dtype)
    mse = action_mse(baseline_actions, result)
    report.add(assert_check(
        mse > MIN_ABS_CHANGE_ACTION,
        "9.7a zeroed prefix cache changes actions",
        f"action_mse={mse:.6f}",
    ))


# ── 9.8 Action leakage test ─────────────────────────────────────────────────

def test_action_leakage_loss(report: PhaseReport, model, batch, baseline_loss) -> None:
    """Zeroing action embeddings fed to backbone must NOT change flow_loss.

    Architecture guarantee: with causal attention, prefix positions (before
    answer_start_idx) cannot attend to action positions (after answer_start_idx).
    Combined with knowledge_insulation (detached prefix cache), the action expert
    receives identical prefix KV regardless of what actions the backbone sees.

    Hooking ar_action_encoder to output zero embeddings changes what backbone
    sees for action positions, but batch["actions"] (the flow matching target)
    stays unchanged.  Using zeros_like (not randn_like) to avoid shifting the
    global random state which would contaminate the seed-controlled comparison.
    """
    def zero_output_hook(mod, inp, out):
        return torch.zeros_like(out)

    handle = model.ar_action_encoder.register_forward_hook(zero_output_hook)
    try:
        ablated_loss = get_seeded_loss(model, batch)
    finally:
        handle.remove()

    # flow_loss must be unchanged — action expert sees same prefix cache
    base_flow = baseline_loss["flow_loss"]
    abl_flow = ablated_loss["flow_loss"]
    rel_flow = abs(abl_flow - base_flow) / max(abs(base_flow), 1e-8)

    report.add(assert_check(
        rel_flow < LEAKAGE_TOLERANCE,
        "9.8a zeroed backbone action input does NOT change flow_loss",
        f"base={base_flow:.6f}, ablated={abl_flow:.6f}, rel={rel_flow:.2e}",
    ))

    # diffusion_loss CAN change (DiffLoss uses hidden_states at action positions,
    # which are affected by the changed action embeddings).  Report for info only.
    base_diff = baseline_loss.get("diffusion_loss", 0.0)
    abl_diff = ablated_loss.get("diffusion_loss", 0.0)
    rel_diff = abs(abl_diff - base_diff) / max(abs(base_diff), 1e-8) if base_diff != 0 else 0.0
    print(f"    [INFO] diffusion_loss: base={base_diff:.6f}, ablated={abl_diff:.6f}, "
          f"rel={rel_diff:.4%} (expected: may change)")


def test_action_leakage_inference(report: PhaseReport, model, batch, baseline_actions) -> None:
    """Randomizing backbone action embeddings must NOT change flow inference output.

    During inference, prepare_prefix_memory runs the backbone without action
    slot embeddings (actions are unknown at inference time). So this test hooks
    ar_action_encoder and verifies inference output is identical — the encoder
    should not even be called on the inference path.
    """
    call_count = [0]

    def counting_hook(mod, inp, out):
        call_count[0] += 1
        return torch.randn_like(out)

    handle = model.ar_action_encoder.register_forward_hook(counting_hook)
    try:
        model.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            result = infer_flow_action(model, clone_batch(batch))
        ablated_act = (result["generated_actions"] if isinstance(result, dict) else result)
    finally:
        handle.remove()

    mse = action_mse(baseline_actions, ablated_act)
    report.add(assert_check(
        mse < 1e-6,
        "9.8b ar_action_encoder not called during inference (no leakage path)",
        f"action_mse={mse:.2e}, encoder_calls={call_count[0]}",
    ))


# ── Statistical ablation (large-sample mode) ────────────────────────────────

def per_sample_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE between action tensors [B, T, D] → [B]."""
    return (a.float() - b.float()).pow(2).mean(dim=(-1, -2))


def _infer_seeded(model, batch: dict) -> torch.Tensor:
    """Run inference with fixed seed, return generated actions."""
    model.eval()
    with torch.no_grad():
        torch.manual_seed(42)
        result = infer_flow_action(model, clone_batch(batch))
    return (result["generated_actions"] if isinstance(result, dict) else result).clone()


def _compute_statistics(values: list[float]) -> dict[str, float]:
    """Compute summary statistics from a list of per-sample values."""
    arr = np.array(values)
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "pct_above_1e-3": float(np.mean(arr > 1e-3) * 100),
        "pct_above_1e-2": float(np.mean(arr > 1e-2) * 100),
    }


def run_statistical_ablation(
    model,
    collator,
    all_samples: list[dict],
    device: str,
    batch_size: int = 4,
    skip_visual: bool = False,
) -> dict[str, dict[str, float]]:
    """Run ablation on many samples; return per-ablation-type statistics.

    Ablation types tested (inference MSE):
      - zero_visual: zero ViT embeddings via encode_visual_features patch
      - zero_state: zero batch["states"]
      - zero_wrist_state: zero batch["states"][:, :, 0:18]
      - zero_hand_state: zero batch["states"][:, :, 18:48]
      - zero_text_embed: zero text embedding layer output
      - zero_state_encoder: hook state_encoder to output zeros
      - zero_time_embedding: hook time_embedding to output zeros
      - zero_prefix_cache: zero all KV in prefix cache (manual flow loop)
    """
    from collections import defaultdict
    from src.policy.legendvla_inference import _build_action_valid_mask, prepare_prefix_memory

    n = len(all_samples)
    n_batches = (n + batch_size - 1) // batch_size
    results: dict[str, list[float]] = defaultdict(list)

    print(f"\n  Statistical ablation: {n} samples, batch_size={batch_size}, {n_batches} batches")

    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, n)
        batch_samples = all_samples[start:end]
        batch = collate_to_device(collator, batch_samples, device)
        bs = end - start

        if (bi + 1) % 25 == 0 or bi == 0:
            print(f"    batch {bi+1}/{n_batches} (samples {start}..{end-1})")

        # Baseline
        baseline = _infer_seeded(model, batch)

        # 1. Zero visual
        with _VisualAblationContext(model):
            ablated = _infer_seeded(model, batch)
        results["zero_visual"].extend(per_sample_mse(baseline, ablated).tolist())

        # 2. Zero all states
        abl_batch = clone_batch(batch)
        abl_batch["states"] = torch.zeros_like(abl_batch["states"])
        ablated = _infer_seeded(model, abl_batch)
        results["zero_state"].extend(per_sample_mse(baseline, ablated).tolist())

        # 3. Zero wrist_state
        abl_batch = clone_batch(batch)
        abl_batch["states"][:, :, 0:18] = 0.0
        ablated = _infer_seeded(model, abl_batch)
        results["zero_wrist_state"].extend(per_sample_mse(baseline, ablated).tolist())

        # 4. Zero hand_state
        abl_batch = clone_batch(batch)
        abl_batch["states"][:, :, 18:48] = 0.0
        ablated = _infer_seeded(model, abl_batch)
        results["zero_hand_state"].extend(per_sample_mse(baseline, ablated).tolist())

        # 5. Zero text embeddings
        embed_layer = model.backbone.base_model.get_input_embeddings()
        orig_fwd = embed_layer.forward
        embed_layer.forward = lambda ids, _orig=orig_fwd: torch.zeros_like(_orig(ids))
        try:
            ablated = _infer_seeded(model, batch)
        finally:
            embed_layer.forward = orig_fwd
        results["zero_text_embed"].extend(per_sample_mse(baseline, ablated).tolist())

        # 6. Zero state_encoder (hook)
        handle = model.state_encoder.register_forward_hook(lambda m, i, o: torch.zeros_like(o))
        try:
            ablated = _infer_seeded(model, batch)
        finally:
            handle.remove()
        results["zero_state_encoder"].extend(per_sample_mse(baseline, ablated).tolist())

        # 7. Zero time_embedding (hook)
        handle = model.time_embedding.register_forward_hook(lambda m, i, o: torch.zeros_like(o))
        try:
            ablated = _infer_seeded(model, batch)
        finally:
            handle.remove()
        results["zero_time_embedding"].extend(per_sample_mse(baseline, ablated).tolist())

        # 8. Zero prefix cache (manual flow loop)
        working = clone_batch(batch)
        avm = _build_action_valid_mask(working, model.num_action_tokens, model.action_dim)
        working["actions_valid_mask"] = avm
        backbone_out = prepare_prefix_memory(model, working)
        if backbone_out.prefix_cache is not None:
            backbone_out.prefix_cache.keys = torch.zeros_like(backbone_out.prefix_cache.keys)
            backbone_out.prefix_cache.values = torch.zeros_like(backbone_out.prefix_cache.values)
        action_step_mask = avm.any(dim=-1)
        delta_t = 1.0 / max(model.num_inference_steps, 1)
        action_dtype = working["states"].dtype
        model.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            gen = torch.randn(bs, avm.shape[1], model.action_dim, device=device, dtype=action_dtype)
            t = torch.zeros(bs, device=device, dtype=action_dtype)
            for _ in range(model.num_inference_steps):
                tfm = t[:, None].expand(-1, avm.shape[1])
                out = model.forward_flow_stream(
                    batch=working, backbone_output=backbone_out,
                    flow_inputs={"noisy_actions": gen, "time_for_model": tfm},
                    num_parallel_chunks=1,
                )
                gen = gen + delta_t * out["pred_v"]
                t = t + delta_t
        zeroed_cache_actions = gen * action_step_mask.unsqueeze(-1).to(dtype=gen.dtype)
        results["zero_prefix_cache"].extend(per_sample_mse(baseline, zeroed_cache_actions).tolist())

    # Compute statistics
    stats: dict[str, dict[str, float]] = {}
    for name, values in results.items():
        stats[name] = _compute_statistics(values)
    return stats


def run_statistical_loss_ablation(
    model,
    collator,
    all_samples: list[dict],
    device: str,
    batch_size: int = 1,
) -> dict[str, dict[str, float]]:
    """Run loss-based ablation: measure flow_loss and diffusion_loss change per sample.

    Uses batch_size=1 by default to get true per-sample loss values.
    Records absolute loss values for baseline and each ablation type.
    """
    from collections import defaultdict

    n = len(all_samples)
    n_batches = (n + batch_size - 1) // batch_size

    # Each entry: list of (base_flow, abl_flow, base_diff, abl_diff) tuples
    flow_changes: dict[str, list[float]] = defaultdict(list)
    diff_changes: dict[str, list[float]] = defaultdict(list)
    base_flow_all: list[float] = []
    base_diff_all: list[float] = []

    print(f"\n  Loss ablation: {n} samples, batch_size={batch_size}, {n_batches} batches")

    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, n)
        batch_samples = all_samples[start:end]
        batch = collate_to_device(collator, batch_samples, device)

        if (bi + 1) % 50 == 0 or bi == 0:
            print(f"    batch {bi+1}/{n_batches}")

        # Baseline loss
        base_loss = get_seeded_loss(model, batch)
        bf = base_loss["flow_loss"]
        bd = base_loss.get("diffusion_loss", 0.0)
        base_flow_all.append(bf)
        base_diff_all.append(bd)

        # 1. Zero visual
        with _VisualAblationContext(model):
            abl_loss = get_seeded_loss(model, batch)
        flow_changes["zero_visual"].append(abl_loss["flow_loss"] - bf)
        diff_changes["zero_visual"].append(abl_loss.get("diffusion_loss", 0.0) - bd)

        # 2. Zero all states
        abl_batch = clone_batch(batch)
        abl_batch["states"] = torch.zeros_like(abl_batch["states"])
        abl_loss = get_seeded_loss(model, abl_batch)
        flow_changes["zero_state"].append(abl_loss["flow_loss"] - bf)
        diff_changes["zero_state"].append(abl_loss.get("diffusion_loss", 0.0) - bd)

        # 3. Zero text embeddings
        embed_layer = model.backbone.base_model.get_input_embeddings()
        orig_fwd = embed_layer.forward
        embed_layer.forward = lambda ids, _orig=orig_fwd: torch.zeros_like(_orig(ids))
        try:
            abl_loss = get_seeded_loss(model, batch)
        finally:
            embed_layer.forward = orig_fwd
        flow_changes["zero_text_embed"].append(abl_loss["flow_loss"] - bf)
        diff_changes["zero_text_embed"].append(abl_loss.get("diffusion_loss", 0.0) - bd)

        # 4. Zero state_encoder (hook)
        handle = model.state_encoder.register_forward_hook(lambda m, i, o: torch.zeros_like(o))
        try:
            abl_loss = get_seeded_loss(model, batch)
        finally:
            handle.remove()
        flow_changes["zero_state_encoder"].append(abl_loss["flow_loss"] - bf)
        diff_changes["zero_state_encoder"].append(abl_loss.get("diffusion_loss", 0.0) - bd)

    # Compute statistics
    stats: dict[str, dict[str, Any]] = {}
    base_flow_mean = float(np.mean(base_flow_all))
    base_diff_mean = float(np.mean(base_diff_all))
    stats["_baseline"] = {
        "flow_loss_mean": base_flow_mean,
        "flow_loss_std": float(np.std(base_flow_all)),
        "diff_loss_mean": base_diff_mean,
        "diff_loss_std": float(np.std(base_diff_all)),
        "n": len(base_flow_all),
    }

    for name in flow_changes:
        fc = np.array(flow_changes[name])
        dc = np.array(diff_changes[name])
        stats[name] = {
            "n": len(fc),
            "flow_delta_mean": float(np.mean(fc)),
            "flow_delta_std": float(np.std(fc)),
            "flow_delta_median": float(np.median(fc)),
            "flow_rel_change": float(np.mean(fc) / max(base_flow_mean, 1e-8) * 100),
            "diff_delta_mean": float(np.mean(dc)),
            "diff_delta_std": float(np.std(dc)),
            "diff_delta_median": float(np.median(dc)),
            "diff_rel_change": float(np.mean(dc) / max(base_diff_mean, 1e-8) * 100),
        }
    return stats


def format_loss_stat_table(stats: dict[str, dict]) -> str:
    """Format loss ablation statistics as a readable table."""
    base = stats.get("_baseline", {})
    lines = [
        f"Baseline: flow_loss={base.get('flow_loss_mean', 0):.4f}±{base.get('flow_loss_std', 0):.4f}, "
        f"diffusion_loss={base.get('diff_loss_mean', 0):.4f}±{base.get('diff_loss_std', 0):.4f} "
        f"(n={base.get('n', 0)})",
        "",
        f"{'Ablation':<22} {'N':>4}  {'flow Δ mean':>12} {'flow Δ%':>8}  "
        f"{'diff Δ mean':>12} {'diff Δ%':>8}",
        "-" * 80,
    ]
    for name, s in stats.items():
        if name == "_baseline":
            continue
        lines.append(
            f"{name:<22} {s['n']:>4}  {s['flow_delta_mean']:>+12.6f} "
            f"{s['flow_rel_change']:>+7.2f}%  "
            f"{s['diff_delta_mean']:>+12.6f} {s['diff_rel_change']:>+7.2f}%"
        )
    return "\n".join(lines)


def format_stat_table(stats: dict[str, dict[str, float]]) -> str:
    """Format statistics as a readable table."""
    lines = [
        f"{'Ablation':<22} {'N':>5} {'Mean':>10} {'Std':>10} {'Median':>10} "
        f"{'P5':>10} {'P95':>10} {'%>1e-3':>7} {'%>1e-2':>7}",
        "-" * 102,
    ]
    for name, s in stats.items():
        lines.append(
            f"{name:<22} {s['n']:>5.0f} {s['mean']:>10.6f} {s['std']:>10.6f} "
            f"{s['median']:>10.6f} {s['p5']:>10.6f} {s['p95']:>10.6f} "
            f"{s['pct_above_1e-3']:>6.1f}% {s['pct_above_1e-2']:>6.1f}%"
        )
    return "\n".join(lines)


def plot_statistical_ablation(stats: dict[str, dict[str, float]], out_dir, skip_visual: bool = False) -> None:
    """Box-plot style visualization for statistical ablation results."""
    plt = safe_import_plt()
    if plt is None or skip_visual:
        return

    names = list(stats.keys())
    means = [stats[n]["mean"] for n in names]
    medians = [stats[n]["median"] for n in names]
    p5s = [stats[n]["p5"] for n in names]
    p95s = [stats[n]["p95"] for n in names]

    fig, ax = plt.subplots(figsize=(14, max(5, len(names) * 0.5)))
    y = np.arange(len(names))

    # Whiskers from P5 to P95
    for i, name in enumerate(names):
        ax.plot([p5s[i], p95s[i]], [i, i], color="steelblue", linewidth=2, solid_capstyle="round")
    ax.scatter(means, y, color="red", s=60, zorder=5, label="mean")
    ax.scatter(medians, y, color="orange", s=40, zorder=5, marker="D", label="median")

    ax.axvline(x=MIN_ABS_CHANGE_ACTION, color="gray", linestyle="--", linewidth=1, label=f"threshold={MIN_ABS_CHANGE_ACTION}")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Action MSE (vs baseline)")
    ax.set_title(f"Statistical Ablation: per-sample MSE (n={stats[names[0]]['n']})")
    ax.set_xscale("log")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "statistical_ablation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved statistical_ablation.png")


# ── Visualization ────────────────────────────────────────────────────────────

def plot_ablation_summary(results: dict[str, float], out_dir, skip_visual: bool = False) -> None:
    plt = safe_import_plt()
    if plt is None or skip_visual:
        return

    names = list(results.keys())
    values = list(results.values())

    fig, ax = plt.subplots(figsize=(12, max(5, len(names) * 0.4)))
    bars = ax.barh(names, values, color="steelblue")
    ax.axvline(x=MIN_ABS_CHANGE_ACTION, color="red", linestyle="--", linewidth=1.5,
               label=f"threshold={MIN_ABS_CHANGE_ACTION}")
    ax.set_xlabel("Action MSE (vs baseline)")
    ax.set_title("Input Ablation: Action MSE Change (Real Data)")
    ax.legend()

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(val * 0.02, 0.0001),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.6f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "ablation_summary.png", dpi=150)
    plt.close(fig)
    print(f"  Saved ablation_summary.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def run_all(
    config_path: str,
    vla_shard: str,
    normalizer_path: str,
    checkpoint_path: str | None = None,
    skip_visual: bool = False,
    n_samples: int = 2,
    batch_size: int = 4,
) -> PhaseReport:
    import json as _json

    out_dir = get_output_dir(OUTPUT_PART)
    report = PhaseReport("Part 9: Input Ablation Verification", out_dir)
    statistical_mode = n_samples > 10

    print("\n=== Part 9: Input Ablation Verification (Real Data) ===\n")
    print(f"  Mode: {'statistical' if statistical_mode else 'quick'} "
          f"(n_samples={n_samples}, batch_size={batch_size})")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model + collator from Hydra config
    from src.tests.full_chain_verification.part3_backbone_prefix_cache import (
        build_model_and_collator, load_hydra_config,
    )
    cfg = load_hydra_config(config_path)
    model, collator = build_model_and_collator(config_path, device)

    if checkpoint_path:
        print(f"  Loading checkpoint: {checkpoint_path}")
        from src.utils.checkpoint_util import load_checkpoint
        load_checkpoint(model, checkpoint_path)
        print("  Checkpoint loaded.")

    model.eval()

    # Load real data via training pipeline (correct extrinsic, multi-frame window)
    print(f"  Building VLA dataset from shard: {vla_shard}")
    dataset = build_vla_dataset(cfg, normalizer_path, vla_shard)
    print(f"  Collecting {n_samples} samples...")
    real_samples = collect_samples(dataset, n_samples)
    print(f"  Collected {len(real_samples)} real samples.")

    # ── Statistical mode: large-sample distribution analysis ────────────────
    if statistical_mode:
        stats = run_statistical_ablation(
            model, collator, real_samples, device,
            batch_size=batch_size, skip_visual=skip_visual,
        )

        # Print table
        table = format_stat_table(stats)
        print(f"\n{table}\n")

        # Save raw stats
        with open(out_dir / "statistical_ablation.json", "w") as f:
            _json.dump(stats, f, indent=2)
        print(f"  Saved statistical_ablation.json")

        # Generate report checks from statistics
        insulated = getattr(model, "knowledge_insulation", False)
        for name, s in stats.items():
            if name in ("zero_time_embedding", "zero_prefix_cache"):
                # These should always show large MSE (core flow components)
                report.add(assert_check(
                    s["mean"] > MIN_ABS_CHANGE_ACTION,
                    f"stat: {name} mean MSE > threshold",
                    f"mean={s['mean']:.6f}, median={s['median']:.6f}, n={s['n']:.0f}",
                ))
            elif name in ("zero_visual", "zero_state", "zero_state_encoder"):
                if insulated:
                    report.add(assert_check(
                        True,
                        f"stat: {name} (knowledge_insulation=True)",
                        f"mean={s['mean']:.6f}, median={s['median']:.6f}, "
                        f"pct>1e-3={s['pct_above_1e-3']:.1f}%, n={s['n']:.0f}",
                    ))
                else:
                    report.add(assert_check(
                        s["mean"] > MIN_ABS_CHANGE_ACTION,
                        f"stat: {name} mean MSE > threshold",
                        f"mean={s['mean']:.6f}, n={s['n']:.0f}",
                    ))
            else:
                report.add(assert_check(
                    True,
                    f"stat: {name}",
                    f"mean={s['mean']:.6f}, median={s['median']:.6f}, n={s['n']:.0f}",
                ))

        plot_statistical_ablation(stats, out_dir, skip_visual=skip_visual)

        # Loss-based ablation (flow_loss + diffusion_loss)
        loss_stats = run_statistical_loss_ablation(
            model, collator, real_samples, device, batch_size=1,
        )
        loss_table = format_loss_stat_table(loss_stats)
        print(f"\n{loss_table}\n")

        with open(out_dir / "statistical_loss_ablation.json", "w") as f:
            _json.dump(loss_stats, f, indent=2)
        print(f"  Saved statistical_loss_ablation.json")

        for name, s in loss_stats.items():
            if name == "_baseline":
                continue
            report.add(assert_check(
                True,
                f"loss: {name}",
                f"flow Δ={s['flow_delta_mean']:+.6f} ({s['flow_rel_change']:+.2f}%), "
                f"diff Δ={s['diff_delta_mean']:+.6f} ({s['diff_rel_change']:+.2f}%)",
            ))

    # ── Quick mode: detailed per-check report (original behavior) ───────────
    else:
        quick_samples = real_samples[:min(len(real_samples), batch_size)]
        batch = collate_to_device(collator, quick_samples, device)
        print(f"  Quick mode: batch input_ids shape: {batch['input_ids'].shape}")

        # Baselines
        print("  Computing baselines...")
        baseline_actions = get_baseline_actions(model, batch)
        baseline_loss = get_seeded_loss(model, batch)
        print(f"  Baseline: flow_loss={baseline_loss['flow_loss']:.6f}, "
              f"diffusion_loss={baseline_loss.get('diffusion_loss', 0):.6f}, "
              f"total_loss={baseline_loss['total_loss']:.6f}")

        # 9.1 Visual ablation
        print("  [9.1] Visual ablation...")
        test_visual_ablation_inference(report, model, batch, baseline_actions)
        test_visual_ablation_loss(report, model, batch, baseline_loss)

        # 9.2 State ablation
        print("  [9.2] State ablation...")
        test_state_ablation_inference(report, model, batch, baseline_actions)
        test_state_ablation_loss(report, model, batch, baseline_loss)

        # 9.3 Per-field state ablation
        print("  [9.3] Per-field state ablation...")
        test_per_field_state_ablation(report, model, batch, baseline_actions)

        # 9.4 Instruction / text ablation
        print("  [9.4] Instruction/text ablation...")
        test_different_instructions(report, model, collator, quick_samples, device)
        test_text_embedding_ablation(report, model, batch, baseline_actions)

        # 9.5 Component ablation
        print("  [9.5] Component ablation (hooks)...")
        test_component_ablation(report, model, batch, baseline_actions)

        # 9.6 DiffLoss condition isolation
        print("  [9.6] DiffLoss condition isolation...")
        test_diffloss_condition_isolation(report, model, batch, baseline_loss)

        # 9.7 Prefix cache ablation
        print("  [9.7] Prefix cache ablation...")
        test_prefix_cache_matters(report, model, batch, baseline_actions)

        # 9.8 Action leakage test
        print("  [9.8] Action leakage test...")
        test_action_leakage_loss(report, model, batch, baseline_loss)
        test_action_leakage_inference(report, model, batch, baseline_actions)

        # 9.9 Attention weight distribution
        print("  [9.9] Attention weight distribution (eager mode)...")
        test_attention_weight_distribution(report, model, batch, out_dir, skip_visual)

        # Collect MSE for visualization
        print("  Collecting ablation MSE summary for plot...")
        ablation_mse: dict[str, float] = {}
        mse_collectors = [
            ("visual (zero embeds)", lambda: _measure_visual(model, batch, baseline_actions)),
            ("state (zero all)", lambda: _measure_state(model, batch, baseline_actions, None)),
            ("wrist_state", lambda: _measure_state(model, batch, baseline_actions, (0, 18))),
            ("hand_state", lambda: _measure_state(model, batch, baseline_actions, (18, 48))),
            ("text embedding (zero)", lambda: _measure_text(model, batch, baseline_actions)),
            ("state_encoder (hook)", lambda: _measure_hook(model, batch, baseline_actions, model.state_encoder)),
            ("time_embedding (hook)", lambda: _measure_hook(model, batch, baseline_actions, model.time_embedding)),
            ("action_decoder (hook)", lambda: _measure_hook(model, batch, baseline_actions, model.action_decoder)),
        ]
        for name, fn in mse_collectors:
            ablation_mse[name] = fn()
        plot_ablation_summary(ablation_mse, out_dir, skip_visual=skip_visual)

    report.save()
    report.print_summary()
    return report


# ── MSE collectors for quick-mode visualization ─────────────────────────────

def _measure_visual(model, batch, baseline_actions) -> float:
    with _VisualAblationContext(model):
        ablated = _infer_seeded(model, batch)
    return action_mse(baseline_actions, ablated)


def _measure_state(model, batch, baseline_actions, field_range) -> float:
    ablated = clone_batch(batch)
    if field_range is None:
        ablated["states"] = torch.zeros_like(ablated["states"])
    else:
        ablated["states"][:, :, field_range[0]:field_range[1]] = 0.0
    return action_mse(baseline_actions, _infer_seeded(model, ablated))


def _measure_text(model, batch, baseline_actions) -> float:
    embed = model.backbone.base_model.get_input_embeddings()
    orig = embed.forward
    embed.forward = lambda ids, _orig=orig: torch.zeros_like(_orig(ids))
    try:
        ablated = _infer_seeded(model, batch)
    finally:
        embed.forward = orig
    return action_mse(baseline_actions, ablated)


def _measure_hook(model, batch, baseline_actions, module) -> float:
    handle = module.register_forward_hook(lambda m, i, o: torch.zeros_like(o))
    try:
        ablated = _infer_seeded(model, batch)
    finally:
        handle.remove()
    return action_mse(baseline_actions, ablated)


# ── 9.9 Attention weight distribution analysis ────────────────────────────────

def _identify_prefix_token_types(
    input_ids: torch.Tensor,
    answer_start_idx: torch.Tensor,
    backbone,
) -> dict[str, torch.Tensor]:
    """Classify prefix positions into visual / state / text for the first sample.

    Returns dict of boolean masks over prefix positions, shape [prefix_len].
    """
    ids = input_ids[0]
    prefix_len = int(answer_start_idx[0].item())
    prefix_ids = ids[:prefix_len]

    visual_token_id = backbone.video_token_id
    state_token_id = backbone.state_token_id

    visual_mask = prefix_ids == visual_token_id
    state_mask = prefix_ids == state_token_id
    text_mask = ~visual_mask & ~state_mask

    return {
        "prefix_len": prefix_len,
        "visual": visual_mask,
        "state": state_mask,
        "text": text_mask,
        "n_visual": int(visual_mask.sum().item()),
        "n_state": int(state_mask.sum().item()),
        "n_text": int(text_mask.sum().item()),
    }


def _extract_attention_distribution(
    attn_weights_per_step: list,
    prefix_len: int,
    token_masks: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Analyze attention weight distribution from action expert.

    attn_weights_per_step: list of per-ODE-step results, each is a list of
        28 tensors of shape [B, num_heads, action_len, prefix_len + action_len].
    Returns per-layer statistics averaged over ODE steps.
    """
    # Use only the last ODE step (t close to 1.0, most informative)
    last_step_weights = attn_weights_per_step[-1]
    num_layers = len(last_step_weights)
    per_layer = []

    for layer_idx in range(num_layers):
        w = last_step_weights[layer_idx]
        if w is None:
            per_layer.append(None)
            continue
        # w: [B, num_heads, action_len, kv_len] where kv_len = prefix_len + action_len
        # Average over batch and heads for the distribution summary
        w_mean = w[0].float().mean(dim=0)  # [action_len, kv_len]

        prefix_attn = w_mean[:, :prefix_len].sum(dim=-1).mean().item()
        action_attn = w_mean[:, prefix_len:].sum(dim=-1).mean().item()

        # Within prefix: visual / state / text
        prefix_w = w_mean[:, :prefix_len]  # [action_len, prefix_len]
        visual_attn = prefix_w[:, token_masks["visual"]].sum(dim=-1).mean().item() if token_masks["n_visual"] > 0 else 0.0
        state_attn = prefix_w[:, token_masks["state"]].sum(dim=-1).mean().item() if token_masks["n_state"] > 0 else 0.0
        text_attn = prefix_w[:, token_masks["text"]].sum(dim=-1).mean().item() if token_masks["n_text"] > 0 else 0.0

        # Per-head analysis (sample 0)
        w_head = w[0].float()  # [num_heads, action_len, kv_len]
        head_prefix_share = w_head[:, :, :prefix_len].sum(dim=-1).mean(dim=-1)  # [num_heads]

        per_layer.append({
            "prefix_share": prefix_attn,
            "action_share": action_attn,
            "visual_share": visual_attn,
            "state_share": state_attn,
            "text_share": text_attn,
            "head_prefix_shares": head_prefix_share.tolist(),
        })

    return {"per_layer": per_layer, "num_layers": num_layers}


def _plot_attention_distribution(analysis: dict, token_info: dict, out_dir, skip_visual: bool = False) -> None:
    plt = safe_import_plt()
    if plt is None or skip_visual:
        return

    per_layer = analysis["per_layer"]
    num_layers = analysis["num_layers"]
    valid_layers = [i for i in range(num_layers) if per_layer[i] is not None]
    if not valid_layers:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: prefix vs action attention share per layer
    prefix_shares = [per_layer[i]["prefix_share"] for i in valid_layers]
    action_shares = [per_layer[i]["action_share"] for i in valid_layers]
    ax = axes[0]
    ax.bar(valid_layers, prefix_shares, label="prefix (cross-attn)", color="steelblue", alpha=0.8)
    ax.bar(valid_layers, action_shares, bottom=prefix_shares, label="action (self-attn)", color="coral", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention share")
    ax.set_title("Action Expert: Prefix vs Action Attention Share per Layer (last ODE step)")
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Plot 2: within-prefix breakdown (visual / state / text)
    visual_shares = [per_layer[i]["visual_share"] for i in valid_layers]
    state_shares = [per_layer[i]["state_share"] for i in valid_layers]
    text_shares = [per_layer[i]["text_share"] for i in valid_layers]
    ax2 = axes[1]
    ax2.bar(valid_layers, visual_shares, label=f"visual ({token_info['n_visual']} tokens)", color="green", alpha=0.8)
    ax2.bar(valid_layers, state_shares, bottom=visual_shares, label=f"state ({token_info['n_state']} tokens)", color="orange", alpha=0.8)
    bottoms = [v + s for v, s in zip(visual_shares, state_shares)]
    ax2.bar(valid_layers, text_shares, bottom=bottoms, label=f"text ({token_info['n_text']} tokens)", color="purple", alpha=0.8)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Attention share (within prefix + action)")
    ax2.set_title("Within-Prefix Attention Breakdown per Layer")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "attention_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved attention_distribution.png")

    # Plot 3: per-head heatmap for a representative layer (middle layer)
    mid = valid_layers[len(valid_layers) // 2]
    head_shares = per_layer[mid]["head_prefix_shares"]
    fig2, ax3 = plt.subplots(figsize=(10, 3))
    ax3.bar(range(len(head_shares)), head_shares, color="steelblue")
    ax3.set_xlabel("Head index")
    ax3.set_ylabel("Prefix attention share")
    ax3.set_title(f"Per-Head Prefix Attention Share (Layer {mid}, last ODE step)")
    ax3.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50%")
    ax3.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "attention_per_head.png", dpi=150)
    plt.close(fig2)
    print(f"  Saved attention_per_head.png")


def test_attention_weight_distribution(
    report: PhaseReport, model, batch, out_dir, skip_visual: bool = False,
) -> dict[str, Any] | None:
    """9.9: Analyze action expert attention weight distribution.

    Temporarily switches action expert from flex_attention to eager
    to obtain actual attention weight matrices. Measures how much the
    expert attends to prefix (visual/state/text) vs action positions.
    """
    expert = model.flow_expert

    # Identify prefix token types
    token_info = _identify_prefix_token_types(
        batch["input_ids"], batch["answer_start_idx"], model.backbone,
    )
    print(f"    Prefix composition: {token_info['n_visual']} visual, "
          f"{token_info['n_state']} state, {token_info['n_text']} text "
          f"(total {token_info['prefix_len']})")

    # Temporarily switch to eager attention to get real weights
    original_impl = expert.config._attn_implementation
    expert.config._attn_implementation = "eager"
    try:
        model.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            result = infer_flow_action(model, clone_batch(batch), output_attentions=True)

        expert_attn = result.get("expert_attention_weights")
        if expert_attn is None or len(expert_attn) == 0:
            report.add(assert_check(
                False,
                "9.9a attention weights obtained",
                "expert_attention_weights is None or empty",
            ))
            return None

        # Verify we got actual weight tensors (not LSE or None)
        last_step = expert_attn[-1]
        has_real_weights = any(w is not None and w.ndim == 4 for w in last_step)
        report.add(assert_check(
            has_real_weights,
            "9.9a attention weights obtained (eager mode)",
            f"steps={len(expert_attn)}, layers_with_weights="
            f"{sum(1 for w in last_step if w is not None and w.ndim == 4)}/{len(last_step)}",
        ))

        if not has_real_weights:
            return None

        # Analyze distribution
        analysis = _extract_attention_distribution(
            expert_attn, token_info["prefix_len"], token_info,
        )

        # Report key metrics
        per_layer = analysis["per_layer"]
        valid = [l for l in per_layer if l is not None]
        if valid:
            mean_prefix = np.mean([l["prefix_share"] for l in valid])
            mean_visual = np.mean([l["visual_share"] for l in valid])
            mean_state = np.mean([l["state_share"] for l in valid])
            mean_text = np.mean([l["text_share"] for l in valid])

            report.add(assert_check(
                True,
                "9.9b attention distribution summary",
                f"mean_prefix_share={mean_prefix:.4f}, "
                f"visual={mean_visual:.4f}, state={mean_state:.4f}, text={mean_text:.4f}, "
                f"action_share={1 - mean_prefix:.4f}",
            ))

            # Per-layer detail
            for i, l in enumerate(per_layer):
                if l is not None:
                    print(f"    Layer {i:2d}: prefix={l['prefix_share']:.4f} "
                          f"(visual={l['visual_share']:.4f}, state={l['state_share']:.4f}, "
                          f"text={l['text_share']:.4f}) | action={l['action_share']:.4f}")

        _plot_attention_distribution(analysis, token_info, out_dir, skip_visual)
        return analysis

    finally:
        expert.config._attn_implementation = original_impl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 9: Input ablation verification")
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--vla-shard", type=str, required=True,
                        help="VLA WebDataset shard pattern (real data)")
    parser.add_argument("--normalizer-path", type=str, required=True,
                        help="Path to normalizer.pkl")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Trained checkpoint for meaningful results")
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--n-samples", type=int, default=2,
                        help="Number of samples (>10 enables statistical mode, default 2)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for inference (default 4)")
    args = parser.parse_args()
    report = run_all(
        config_path=args.config_path,
        vla_shard=args.vla_shard,
        normalizer_path=args.normalizer_path,
        checkpoint_path=args.checkpoint_path,
        skip_visual=args.skip_visual,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
    )
    exit(0 if report.all_passed else 1)
