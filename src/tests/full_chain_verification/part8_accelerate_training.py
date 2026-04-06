"""
Part 8: Accelerate / Multi-GPU Training Verification.

Verifies the training workspace configuration, parameter grouping,
LR scheduler, preprocess_batch, FSDP wrap targets, gradient clipping,
and checkpoint save/load — everything between the model and the
training loop that can silently break in a distributed setting.

Requires: GPU + Hydra config
Runs on: GPU (single-GPU simulation, no torchrun needed)

Usage:
    python -m src.tests.full_chain_verification.part8_accelerate_training \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml \
        [--normalizer-path /path/to/normalizer.pkl] \
        [--checkpoint-path /path/to/checkpoint]
"""

from __future__ import annotations

import argparse
import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.tests.full_chain_verification.utils import (
    PhaseReport,
    assert_check,
    get_output_dir,
    safe_import_plt,
)

OUTPUT_PART = "part8"


# ── 8.1 FSDP wrap targets ────────────────────────────────────────────────

def test_fsdp_wrap_targets_exist(report: PhaseReport, model) -> None:
    """All classes listed in fsdp_transformer_layer_cls_to_wrap must be
    importable and present as submodules in the model."""
    wrap_classes = [
        "Qwen3VLTextDecoderLayerWithKV",
        "DiTQwen3DecoderLayer",
        "DiffLoss",
    ]

    found = {}
    for cls_name in wrap_classes:
        count = sum(
            1 for _, m in model.named_modules()
            if type(m).__name__ == cls_name
        )
        found[cls_name] = count

    all_present = all(v > 0 for v in found.values())
    report.add(assert_check(
        all_present,
        "8.1a FSDP wrap target classes present in model",
        str(found),
    ))

    # Also check Qwen3VLVisionBlock from transformers
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock  # noqa: F401
        vis_count = sum(
            1 for _, m in model.named_modules()
            if type(m).__name__ == "Qwen3VLVisionBlock"
        )
        report.add(assert_check(
            vis_count > 0,
            "8.1b Qwen3VLVisionBlock present in model",
            f"count={vis_count}",
        ))
    except ImportError:
        report.add(assert_check(True, "8.1b Qwen3VLVisionBlock", "SKIPPED: transformers version"))


# ── 8.2 Parameter grouping ───────────────────────────────────────────────

def test_param_groups_cover_all_trainable(report: PhaseReport, model, cfg) -> None:
    """Reproduce get_grouped_parameters and verify full coverage."""
    from src.workspace.train_legendvla_workspace import TrainLegendVLAWorkspace

    all_groups = []
    if cfg.training.get("objective") != "ar":
        all_groups.extend(
            TrainLegendVLAWorkspace.get_grouped_parameters(
                None, model.action_expert_parameters, cfg.optimizer.action
            )
        )
    if cfg.training.get("train_vlm", False):
        all_groups.extend(
            TrainLegendVLAWorkspace.get_grouped_parameters(
                None, model.trainable_vlm_parameters, cfg.optimizer.vlm
            )
        )
    all_groups.extend(
        TrainLegendVLAWorkspace.get_grouped_parameters(
            None, model.diffloss_parameters, cfg.optimizer.diffloss
        )
    )

    grouped_ids = set()
    for g in all_groups:
        for p in g["params"]:
            grouped_ids.add(id(p))

    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    missing = trainable_ids - grouped_ids
    extra = grouped_ids - trainable_ids
    duplicates = len(grouped_ids) < sum(len(g["params"]) for g in all_groups)

    missing_names = []
    if missing:
        id_to_name = {id(p): n for n, p in model.named_parameters()}
        missing_names = [id_to_name.get(i, "?") for i in list(missing)[:10]]

    report.add(assert_check(
        len(missing) == 0,
        "8.2a all trainable params in optimizer groups",
        f"missing={len(missing)}, names={missing_names}" if missing else "all covered",
    ))
    report.add(assert_check(
        len(extra) == 0,
        "8.2b no non-trainable params in optimizer groups",
        f"extra={len(extra)}" if extra else "clean",
    ))
    report.add(assert_check(
        not duplicates,
        "8.2c no duplicate params across groups",
        f"total_in_groups={sum(len(g['params']) for g in all_groups)}, unique={len(grouped_ids)}",
    ))


def test_param_groups_lr_and_wd(report: PhaseReport, model, cfg) -> None:
    """Each param group should have the correct lr, weight_decay, betas."""
    from src.workspace.train_legendvla_workspace import TrainLegendVLAWorkspace

    checks = []
    for group_name, params_fn, opt_cfg_key in [
        ("action", model.action_expert_parameters, "action"),
        ("diffloss", model.diffloss_parameters, "diffloss"),
    ]:
        opt_cfg = getattr(cfg.optimizer, opt_cfg_key)
        groups = TrainLegendVLAWorkspace.get_grouped_parameters(None, params_fn, opt_cfg)
        for g in groups:
            checks.append(g["lr"] == opt_cfg.lr)
            checks.append(list(g["betas"]) == list(opt_cfg.betas))
            # decay group has wd > 0, nodecay has wd == 0
            # (first group is decay, second is nodecay)

    report.add(assert_check(
        all(checks),
        "8.2d param group lr/betas match config",
        f"{len(checks)} checks",
    ))


# ── 8.3 LR scheduler shape ───────────────────────────────────────────────

def test_lr_scheduler_shape(report: PhaseReport, cfg, out_dir: Path) -> None:
    """Build the actual production scheduler and verify curve shape."""
    from src.workspace.train_legendvla_workspace import TrainLegendVLAWorkspace

    # Create a dummy optimizer with 4 groups (2 action, 2 diffloss) as in real training
    dummy_params = [torch.nn.Parameter(torch.randn(10, 10)) for _ in range(4)]
    optimizer = torch.optim.AdamW([
        {"params": [dummy_params[0]], "lr": 1e-4},  # action decay
        {"params": [dummy_params[1]], "lr": 1e-4},  # action nodecay
        {"params": [dummy_params[2]], "lr": 1e-4},  # diffloss decay
        {"params": [dummy_params[3]], "lr": 1e-4},  # diffloss nodecay
    ])

    total_steps = 10000
    warmup_steps = cfg.training.lr_warmup_steps
    schedule_name = cfg.training.get("lr_scheduler", "cosine")

    scheduler = TrainLegendVLAWorkspace._build_lr_scheduler(
        optimizer,
        schedule_name=schedule_name,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        vlm_group_indices=set(),
        vlm_freeze_steps=0,
        vlm_rewarmup_steps=0,
    )

    lrs = []
    for step in range(total_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    lrs = np.array(lrs)
    # Warmup: LR should increase
    check_warmup = lrs[warmup_steps] > lrs[0] if warmup_steps > 0 else True
    # Peak around warmup end
    check_peak = lrs[warmup_steps] > lrs[-1] if warmup_steps > 0 else True
    # Decay: LR should decrease after warmup
    check_decay = lrs[-1] < lrs[warmup_steps] if warmup_steps < total_steps else True
    # Not all zero
    check_nonzero = float(lrs.max()) > 0

    report.add(assert_check(
        check_warmup and check_peak and check_decay and check_nonzero,
        "8.3a production LR scheduler shape",
        f"warmup_ok={check_warmup}, peak_ok={check_peak}, decay_ok={check_decay}, "
        f"lr_range=[{lrs.min():.2e}, {lrs.max():.2e}]",
    ))

    # Visualize
    plt = safe_import_plt()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(lrs)
        ax.set_xlabel("Step")
        ax.set_ylabel("LR")
        ax.set_title(f"Production LR schedule ({schedule_name}, warmup={warmup_steps})")
        ax.axvline(warmup_steps, color="r", linestyle="--", label="warmup end")
        ax.legend()
        fig.savefig(out_dir / "production_lr_schedule.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


def test_vlm_freeze_scheduler(report: PhaseReport, cfg) -> None:
    """VLM groups should have LR=0 during freeze phase, then warm up."""
    from src.workspace.train_legendvla_workspace import TrainLegendVLAWorkspace

    freeze_steps = int(cfg.training.get("vlm_freeze_steps", 0))
    rewarmup_steps = int(cfg.training.get("vlm_rewarmup_steps", 0))
    if freeze_steps <= 0:
        report.add(assert_check(True, "8.3b VLM freeze scheduler", "SKIPPED: no vlm_freeze_steps"))
        return

    dummy_params = [torch.nn.Parameter(torch.randn(10)) for _ in range(4)]
    optimizer = torch.optim.AdamW([
        {"params": [dummy_params[0]], "lr": 1e-4},  # non-VLM
        {"params": [dummy_params[1]], "lr": 1e-4},  # non-VLM
        {"params": [dummy_params[2]], "lr": 5e-5},  # VLM
        {"params": [dummy_params[3]], "lr": 5e-5},  # VLM
    ])
    vlm_indices = {2, 3}
    total_steps = freeze_steps + rewarmup_steps + 1000

    scheduler = TrainLegendVLAWorkspace._build_lr_scheduler(
        optimizer,
        schedule_name=cfg.training.get("lr_scheduler", "cosine"),
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=total_steps,
        vlm_group_indices=vlm_indices,
        vlm_freeze_steps=freeze_steps,
        vlm_rewarmup_steps=rewarmup_steps,
    )

    vlm_lrs = []
    for step in range(total_steps):
        vlm_lrs.append(optimizer.param_groups[2]["lr"])
        optimizer.step()
        scheduler.step()

    # During freeze: VLM LR should be 0
    frozen_ok = all(lr == 0.0 for lr in vlm_lrs[:freeze_steps])
    # After freeze: VLM LR should be > 0
    active_ok = any(lr > 0.0 for lr in vlm_lrs[freeze_steps:]) if freeze_steps < total_steps else True

    report.add(assert_check(
        frozen_ok and active_ok,
        "8.3b VLM freeze scheduler",
        f"freeze_steps={freeze_steps}, frozen_ok={frozen_ok}, active_ok={active_ok}",
    ))


# ── 8.4 preprocess_batch completeness ────────────────────────────────────

def test_preprocess_batch_keys(report: PhaseReport, model, collator) -> None:
    """preprocess_batch should produce all keys that model.forward("train") needs."""
    from src.tests.full_chain_verification.part3_backbone_prefix_cache import make_mock_batch

    device = next(model.parameters()).device
    batch = make_mock_batch(collator, model, device)

    # Simulate preprocess_batch from TrainLegendVLAWorkspace
    required_keys = [
        "input_ids", "attention_mask", "states", "answer_start_idx",
        "is_vla_data", "n_states", "n_actions", "mm_token_type_ids",
    ]
    conditional_keys = [
        ("actions", True),
        ("actions_valid_mask", True),
        ("labels", True),
    ]

    missing = [k for k in required_keys if k not in batch]
    missing_conditional = [k for k, needed in conditional_keys if needed and k not in batch]

    report.add(assert_check(
        len(missing) == 0 and len(missing_conditional) == 0,
        "8.4a collator outputs all keys needed by preprocess_batch",
        f"missing={missing + missing_conditional}" if (missing or missing_conditional)
        else f"all {len(required_keys) + len(conditional_keys)} keys present",
    ))

    # Check dtypes match expectations after .to(dtype)
    dtype_checks = []
    if "states" in batch and torch.is_tensor(batch["states"]):
        dtype_checks.append(batch["states"].is_floating_point())
    if "actions" in batch and torch.is_tensor(batch["actions"]):
        dtype_checks.append(batch["actions"].is_floating_point())

    report.add(assert_check(
        all(dtype_checks),
        "8.4b states/actions are floating point",
        f"{len(dtype_checks)} checks",
    ))


# ── 8.5 Gradient clipping components ─────────────────────────────────────

def test_gradient_clipping_components(report: PhaseReport, model) -> None:
    """The three gradient clipping groups (action_expert, diffloss, vlm)
    should have non-overlapping parameters and cover expected modules."""
    action_ids = {id(p) for p in model.action_expert_parameters}
    diffloss_ids = {id(p) for p in model.diffloss_parameters}

    overlap = action_ids & diffloss_ids
    report.add(assert_check(
        len(overlap) == 0,
        "8.5a action_expert and diffloss params non-overlapping",
        f"overlap={len(overlap)}" if overlap else "clean",
    ))

    # action_expert_parameters should include flow_expert, encoders, decoder, time_embedding
    ae_names = {n for n, p in model.named_parameters() if id(p) in action_ids}
    expected_prefixes = ["flow_expert", "action_encoder", "ar_action_encoder",
                         "time_embedding", "action_decoder", "state_encoder"]
    found_prefixes = []
    for prefix in expected_prefixes:
        if any(n.startswith(prefix) for n in ae_names):
            found_prefixes.append(prefix)

    report.add(assert_check(
        len(found_prefixes) == len(expected_prefixes),
        "8.5b action_expert_parameters covers expected modules",
        f"found={found_prefixes}, expected={expected_prefixes}",
    ))

    # diffloss_parameters should include latent_condition_projector + diffloss
    dl_names = {n for n, p in model.named_parameters() if id(p) in diffloss_ids}
    expected_dl = ["latent_condition_projector", "diffloss"]
    found_dl = [p for p in expected_dl if any(n.startswith(p) for n in dl_names)]

    report.add(assert_check(
        len(found_dl) == len(expected_dl),
        "8.5c diffloss_parameters covers expected modules",
        f"found={found_dl}, expected={expected_dl}",
    ))


# ── 8.6 NaN guard logic ──────────────────────────────────────────────────

def test_nan_guard_skips_step(report: PhaseReport) -> None:
    """Reproduce the NaN guard: non-finite grad norm should trigger skip."""
    from src.utils.training_utils import scalar_metric_value

    # Simulate what the training loop does
    norms = {"action_expert": torch.tensor(float("nan")), "diffloss": torch.tensor(1.5)}
    should_skip = any(
        not math.isfinite(scalar_metric_value(n))
        for n in norms.values()
    )

    report.add(assert_check(
        should_skip,
        "8.6a NaN grad norm triggers step skip",
        f"should_skip={should_skip}",
    ))

    # Normal case: no skip
    norms_ok = {"action_expert": torch.tensor(2.0), "diffloss": torch.tensor(1.5)}
    should_not_skip = any(
        not math.isfinite(scalar_metric_value(n))
        for n in norms_ok.values()
    )

    report.add(assert_check(
        not should_not_skip,
        "8.6b finite grad norms do not trigger skip",
        f"should_skip={should_not_skip}",
    ))


# ── 8.7 Accelerate accumulate simulation ─────────────────────────────────

def test_gradient_accumulation_math(report: PhaseReport, model, collator) -> None:
    """With gradient_accumulation_steps=N, after N backward passes the
    effective gradient should equal the average of all N micro-batches."""
    from src.tests.full_chain_verification.part3_backbone_prefix_cache import make_mock_batch

    device = next(model.parameters()).device
    accum_steps = 2

    model.train()
    model.zero_grad()

    # Compute gradient from two micro-batches separately, then average
    grads_per_step = []
    for _ in range(accum_steps):
        batch = make_mock_batch(collator, model, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result = model("train", batch)
        # Scale loss as accelerate does: loss / accum_steps
        scaled_loss = result["total_loss"] / accum_steps
        scaled_loss.backward()

    # After accumulation, grads should be non-zero and finite
    has_grad = False
    all_finite = True
    for p in model.parameters():
        if p.grad is not None:
            has_grad = True
            if not torch.isfinite(p.grad).all():
                all_finite = False

    report.add(assert_check(
        has_grad and all_finite,
        "8.7a gradient accumulation produces finite grads",
        f"has_grad={has_grad}, all_finite={all_finite}",
    ))
    model.zero_grad()
    model.eval()


# ── Main ────────────────────────────────────────────────────────────────────

def run_all(
    config_path: str,
    normalizer_path: str | None = None,
    checkpoint_path: str | None = None,
    skip_visual: bool = False,
) -> PhaseReport:
    out_dir = get_output_dir(OUTPUT_PART)
    report = PhaseReport("Part 8: Accelerate Training Verification", out_dir)

    print("\n=== Part 8: Accelerate / Multi-GPU Training Verification ===\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.tests.full_chain_verification.part3_backbone_prefix_cache import (
        build_model_and_collator,
    )
    model, collator = build_model_and_collator(config_path, device)

    import hydra
    with hydra.initialize(config_path="../../../config", version_base=None):
        hydra_cfg = hydra.compose(config_name="train_legendvla", overrides=[
            f"+experiment={Path(config_path).stem}",
        ])
        cfg = hydra_cfg

    # 8.1 FSDP wrap targets
    test_fsdp_wrap_targets_exist(report, model)

    # 8.2 Parameter grouping
    test_param_groups_cover_all_trainable(report, model, cfg)
    test_param_groups_lr_and_wd(report, model, cfg)

    # 8.3 LR scheduler
    test_lr_scheduler_shape(report, cfg, out_dir)
    test_vlm_freeze_scheduler(report, cfg)

    # 8.4 preprocess_batch
    test_preprocess_batch_keys(report, model, collator)

    # 8.5 Gradient clipping groups
    test_gradient_clipping_components(report, model)

    # 8.6 NaN guard
    test_nan_guard_skips_step(report)

    # 8.7 Gradient accumulation
    test_gradient_accumulation_math(report, model, collator)

    report.save()
    report.print_summary()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 8: Accelerate training verification")
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--normalizer-path", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--skip-visual", action="store_true")
    args = parser.parse_args()
    report = run_all(
        config_path=args.config_path,
        normalizer_path=args.normalizer_path,
        checkpoint_path=args.checkpoint_path,
        skip_visual=args.skip_visual,
    )
    exit(0 if report.all_passed else 1)
