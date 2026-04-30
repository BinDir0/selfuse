"""
Part 7: End-to-End Training + Checkpoint Verification.

Tests parameter groups, LR schedule, overfit convergence,
gradient health, dtype correctness, and checkpoint validation.

Requires: GPU + Hydra config + (optional) real data + (optional) checkpoint
Runs on: GPU (heaviest part)

Usage:
    python -m src.tests.full_chain_verification.part7_e2e_training \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml \
        [--normalizer-path /path/to/normalizer.pkl] \
        [--checkpoint-path /path/to/checkpoint.pt]
"""

from __future__ import annotations

import argparse

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from src.tests.full_chain_verification.utils import (
    CheckResult,
    PhaseReport,
    assert_check,
    get_output_dir,
    plot_loss_curves,
    safe_import_plt,
    tensor_stats,
)

OUTPUT_PART = "part7"

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ── 7.1 Parameter groups ───────────────────────────────────────────────────

def test_all_trainable_in_optimizer(report: PhaseReport, model, optimizer) -> None:
    """Every requires_grad=True parameter should be in the optimizer."""
    opt_param_ids = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            opt_param_ids.add(id(p))

    missing = []
    for name, param in model.named_parameters():
        if param.requires_grad and id(param) not in opt_param_ids:
            missing.append(name)

    report.add(assert_check(
        len(missing) == 0,
        "7.1a all trainable params in optimizer",
        f"missing={missing[:10]}" if missing else "all present",
    ))


def test_no_duplicate_params(report: PhaseReport, optimizer) -> None:
    """No parameter should appear twice in the optimizer."""
    seen = set()
    duplicates = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            pid = id(p)
            if pid in seen:
                duplicates += 1
            seen.add(pid)

    report.add(assert_check(
        duplicates == 0,
        "7.1b no duplicate params",
        f"duplicates={duplicates}, total={len(seen)}",
    ))


def test_frozen_params_not_in_optimizer(report: PhaseReport, model, optimizer) -> None:
    """requires_grad=False parameters should not be in optimizer."""
    opt_param_ids = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            opt_param_ids.add(id(p))

    frozen_in_opt = []
    for name, param in model.named_parameters():
        if not param.requires_grad and id(param) in opt_param_ids:
            frozen_in_opt.append(name)

    report.add(assert_check(
        len(frozen_in_opt) == 0,
        "7.1c frozen params not in optimizer",
        f"frozen_in_opt={frozen_in_opt[:10]}" if frozen_in_opt else "clean",
    ))


# ── 7.2 LR schedule ────────────────────────────────────────────────────────

def test_warmup_curve(report: PhaseReport, out_dir) -> None:
    """Simulated cosine schedule with warmup should have correct shape."""
    from torch.optim.lr_scheduler import LambdaLR

    warmup = 2000
    total = 10000
    base_lr = 1e-4

    # Simulate cosine with warmup
    dummy_param = torch.randn(1, requires_grad=True)
    optimizer = torch.optim.AdamW([dummy_param], lr=base_lr)

    import math
    def lr_lambda(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        progress = float(step - warmup) / float(max(1, total - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    lrs = []
    for step in range(total):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    check_warmup_start = lrs[0] < lrs[warmup - 1]
    check_peak = abs(lrs[warmup] - base_lr) < base_lr * 0.05
    check_decay = lrs[-1] < lrs[warmup]

    report.add(assert_check(
        check_warmup_start and check_peak and check_decay,
        "7.2a warmup + cosine LR curve",
        f"lr[0]={lrs[0]:.2e}, lr[warmup]={lrs[warmup]:.2e}, lr[-1]={lrs[-1]:.2e}",
    ))

    plt = safe_import_plt()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(lrs, linewidth=1.2)
        ax.set_xlabel("step")
        ax.set_ylabel("LR")
        ax.set_title("Cosine with Warmup Schedule")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "lr_schedule.png", dpi=150)
        plt.close(fig)


# ── 7.3 Overfit test ───────────────────────────────────────────────────────

def test_overfit_single_batch(report: PhaseReport, model, batch: dict, out_dir) -> None:
    """Train on a single batch for 100 steps. Loss should decrease."""
    device = batch["input_ids"].device
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
    )

    loss_history = {"total": []}
    steps = 100

    for step in range(steps):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result = model("train", dict(batch))

        total_loss = result["total_loss"] if isinstance(result, dict) else result
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = float(total_loss)
        loss_history["total"].append(loss_val)

        if step % 20 == 0:
            print(f"    step {step}: loss={loss_val:.4f}")

    # Check convergence
    first_10 = sum(loss_history["total"][:10]) / 10
    last_10 = sum(loss_history["total"][-10:]) / 10
    decreased = last_10 < first_10 * 0.95  # At least 5% decrease

    has_nan = any(torch.isnan(torch.tensor(v)) for v in loss_history["total"])

    report.add(assert_check(
        decreased and not has_nan,
        "7.3a overfit 100 steps: loss decreases",
        f"first_10_avg={first_10:.4f}, last_10_avg={last_10:.4f}, "
        f"decrease={1 - last_10 / first_10:.1%}, nan={has_nan}",
    ))

    plot_loss_curves(loss_history, "Overfit Single Batch", out_dir / "overfit_loss.png")
    model.eval()


# ── 7.4 Gradient health ────────────────────────────────────────────────────

def test_gradient_norms_per_component(report: PhaseReport, model, batch: dict) -> None:
    """One forward+backward: check grad norms per component."""
    model.train()
    model.zero_grad()

    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result = model("train", dict(batch))
        total_loss = result["total_loss"] if isinstance(result, dict) else result
        total_loss.backward()

        components = {
            "flow_expert": model.flow_expert,
            "state_encoder": model.state_encoder,
            "action_encoder": model.action_encoder,
            "ar_action_encoder": model.ar_action_encoder,
            "time_embedding": model.time_embedding,
            "action_decoder": model.action_decoder,
            "latent_condition_projector": model.latent_condition_projector,
        }
        if model.diffloss is not None:
            components["diffloss"] = model.diffloss

        grad_norms = {}
        all_finite = True
        all_nonzero = True
        for name, module in components.items():
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            if grads:
                # Compute total L2 norm directly; clip_grad_norm_ expects params, not grads.
                norm = float(torch.linalg.vector_norm(
                    torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in grads])
                ))
                grad_norms[name] = norm
                if not torch.isfinite(torch.tensor(norm)):
                    all_finite = False
                if norm == 0:
                    all_nonzero = False
            else:
                grad_norms[name] = "no_grad"
                all_nonzero = False

        report.add(assert_check(
            all_finite and all_nonzero,
            "7.4a gradient norms per component",
            str({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in grad_norms.items()}),
            details=grad_norms,
        ))
    except Exception as e:
        report.add(assert_check(False, "7.4a gradient norms", f"error: {e}"))
    finally:
        model.eval()
        model.zero_grad()


def test_no_unused_parameters(report: PhaseReport, model, batch: dict) -> None:
    """All requires_grad=True params should have grad after backward."""
    model.train()
    model.zero_grad()

    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result = model("train", dict(batch))
        total_loss = result["total_loss"] if isinstance(result, dict) else result
        total_loss.backward()

        no_grad_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                no_grad_params.append(name)

        report.add(assert_check(
            len(no_grad_params) == 0,
            "7.4b no unused parameters (all have grad)",
            f"missing_grad={no_grad_params[:10]}" if no_grad_params else "all params have grad",
        ))
    except Exception as e:
        report.add(assert_check(False, "7.4b unused params", f"error: {e}"))
    finally:
        model.eval()
        model.zero_grad()


# ── 7.5 dtype verification ─────────────────────────────────────────────────

def test_preprocess_batch_dtypes(report: PhaseReport, batch: dict) -> None:
    """Check batch tensor dtypes match model expectations."""
    checks = []
    if "input_ids" in batch:
        checks.append(("input_ids", batch["input_ids"].dtype in (torch.long, torch.int64)))
    if "attention_mask" in batch:
        checks.append(("attention_mask", batch["attention_mask"].dtype in (torch.long, torch.int64)))
    if "labels" in batch:
        checks.append(("labels", batch["labels"].dtype in (torch.long, torch.int64)))
    if "is_vla_data" in batch:
        checks.append(("is_vla_data", batch["is_vla_data"].dtype == torch.bool))
    if "states" in batch:
        checks.append(("states", batch["states"].dtype == torch.bfloat16))
    if "actions" in batch:
        checks.append(("actions", batch["actions"].dtype == torch.bfloat16))

    all_ok = all(ok for _, ok in checks)
    details = {name: str(ok) for name, ok in checks}
    report.add(assert_check(
        all_ok,
        "7.5a batch dtypes",
        str(details),
        details=details,
    ))


# ── 7.6 Checkpoint validation ──────────────────────────────────────────────

def test_checkpoint_load(report: PhaseReport, model, checkpoint_path: str) -> None:
    """Load checkpoint and check for missing/unexpected keys."""
    from src.utils.checkpoint_util import load_checkpoint

    try:
        # load_checkpoint returns None; raises on missing/unexpected keys (strict=True)
        load_checkpoint(model, checkpoint_path)
        report.add(assert_check(
            True,
            "7.6a checkpoint load",
            "loaded successfully (strict=True, 0 missing, 0 unexpected)",
        ))
    except Exception as e:
        report.add(assert_check(
            False,
            "7.6a checkpoint load",
            f"error: {e}",
        ))


def test_checkpoint_inference_reasonable(report: PhaseReport, model, batch: dict) -> None:
    """After loading checkpoint, flow inference should produce reasonable values."""
    from src.policy.legendvla_inference import infer_flow_action

    model.eval()
    with torch.no_grad():
        result = infer_flow_action(model, dict(batch))
    actions = result["generated_actions"] if isinstance(result, dict) else result

    # Check not all zero, not all same, reasonable range
    not_zero = float(actions.abs().max()) > 1e-6
    not_constant = float(actions.std()) > 1e-6
    reasonable_range = float(actions.abs().max()) < 100

    report.add(assert_check(
        not_zero and not_constant and reasonable_range,
        "7.6b checkpoint inference reasonable",
        f"max_abs={float(actions.abs().max()):.4f}, std={float(actions.std()):.4f}",
    ))


# ── 7.7 Real shard full pipeline ──────────────────────────────────────────

def test_real_shard_forward_loss(report: PhaseReport, model, collator, normalizer_path: str | None, vla_shard: str | None) -> None:
    """Load real shard data, build batch through collator, forward, compute loss.

    This test covers the complete real-data training path:
    real shard → wds_dataset sample dict → collator → model forward → loss
    """
    if vla_shard is None:
        report.add(assert_check(True, "7.7a real shard forward loss", "SKIPPED: no --vla-shard"))
        return

    import pickle
    import webdataset as wds
    from src.dataset.wds_dataset import LOWDIM_SLICES
    from src.dataset.data_transforms import process_state_action

    normalizer = None
    if normalizer_path:
        with open(normalizer_path, "rb") as f:
            normalizer = pickle.load(f)
    use_relative = normalizer is not None and "actions" in normalizer.params_dict

    dataset = wds.WebDataset(vla_shard).decode("l")
    real_samples = []
    for raw in dataset:
        ld = raw.get("lowdim.npy")
        if ld is None or ld.shape != (116,):
            continue
        img_key = next((k for k in raw if k.endswith((".png", ".jpg", ".jpeg"))), None)
        if img_key is None:
            continue
        img = raw[img_key]
        if hasattr(img, "convert"):
            img = np.array(img.convert("RGB"))
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        # Resize to target_image_size to keep token count within max_length
        from PIL import Image as PILImage
        target_h, target_w = 384, 384
        pil_img = PILImage.fromarray((img * 255).astype(np.uint8) if img.dtype == np.float32 else img)
        pil_img = pil_img.resize((target_w, target_h), PILImage.BILINEAR)
        img = np.array(pil_img)

        ws = ld[LOWDIM_SLICES["wrist_state"][0]:LOWDIM_SLICES["wrist_state"][1]][np.newaxis]
        hs = ld[LOWDIM_SLICES["hand_state"][0]:LOWDIM_SLICES["hand_state"][1]][np.newaxis]
        wa = ld[LOWDIM_SLICES["wrist_action"][0]:LOWDIM_SLICES["wrist_action"][1]][np.newaxis]
        ha = ld[LOWDIM_SLICES["hand_action"][0]:LOWDIM_SLICES["hand_action"][1]][np.newaxis]
        ext = np.eye(4, dtype=np.float32)
        state, action = process_state_action(
            ws, hs, wa, ha, ext,
            hand_ndim=15,
            normalizer=normalizer,
            use_relative_action=use_relative,
        )
        if isinstance(state, torch.Tensor):
            state = state.float()
        else:
            state = torch.tensor(state, dtype=torch.float32)
        if isinstance(action, torch.Tensor):
            action = action.float()
        else:
            action = torch.tensor(action, dtype=torch.float32)

        # Repeat action along horizon to fill n_actions
        n_actions = model.num_action_tokens
        action_horizon = action.repeat(n_actions, 1) if action.ndim == 2 else action.unsqueeze(0).repeat(n_actions, 1)

        # n_states = number of state timesteps in this sample
        if isinstance(state, torch.Tensor):
            n_states = state.shape[0]
        else:
            n_states = 1

        sample = {
            "images": torch.tensor(img, dtype=torch.uint8).unsqueeze(0),
            "instruction": "pick up the object",
            "intrinsic": torch.tensor(ld[LOWDIM_SLICES["intrinsic"][0]:LOWDIM_SLICES["intrinsic"][1]], dtype=torch.float32),
            "active_views": ["head"],
            "view_mask": torch.tensor([True, False], dtype=torch.bool),
            "vision_type": "video",
            "video_fps": torch.tensor(5.0),
            "states": state,
            "actions": action_horizon,
            "actions_valid_mask": torch.ones(n_actions, model.action_dim, dtype=torch.bool),
            "n_states": torch.tensor(n_states, dtype=torch.long),
            "n_actions": torch.tensor(n_actions, dtype=torch.long),
            "is_vla_data": torch.tensor(True),
        }
        real_samples.append(sample)
        if len(real_samples) >= 2:
            break

    if not real_samples:
        report.add(assert_check(False, "7.7a real shard forward loss", "no valid samples from shard"))
        return

    try:
        batch = collator.collate_raw(real_samples)
        device = next(model.parameters()).device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        model.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result = model("train", batch)

        total_loss = result["total_loss"]
        finite = bool(torch.isfinite(total_loss))
        loss_val = float(total_loss)

        # Check all loss components
        loss_components = {}
        for key in ["total_loss", "flow_loss", "ce_loss"]:
            if key in result and result[key] is not None:
                v = float(result[key])
                loss_components[key] = v

        report.add(assert_check(
            finite and loss_val > 0,
            "7.7a real shard forward loss",
            f"loss={loss_val:.4f}, finite={finite}, components={loss_components}",
        ))

        # Check gradients flow
        total_loss.backward()
        grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in model.parameters() if p.requires_grad)
        report.add(assert_check(
            grad_ok,
            "7.7b real shard gradient flow",
            "gradients flowing to trainable parameters",
        ))
        model.zero_grad()
        model.eval()

    except Exception as e:
        report.add(assert_check(False, "7.7a real shard forward loss", f"error: {e}"))
        import traceback
        traceback.print_exc()


def test_inference_denormalize_reasonable(report: PhaseReport, model, batch: dict, normalizer_path: str | None) -> None:
    """Inference output denormalized back to absolute action space should be reasonable."""
    if normalizer_path is None:
        report.add(assert_check(True, "7.8a inference denormalize", "SKIPPED: no --normalizer-path"))
        return

    import pickle
    from src.policy.legendvla_inference import infer_flow_action

    with open(normalizer_path, "rb") as f:
        normalizer = pickle.load(f)

    key = "actions" if "actions" in normalizer.params_dict else "motions"
    if key not in normalizer.params_dict:
        key = list(normalizer.params_dict.keys())[0]

    model.eval()
    with torch.no_grad():
        result = infer_flow_action(model, dict(batch))
    actions = result["generated_actions"] if isinstance(result, dict) else result

    # Denormalize: move to CPU for normalizer
    actions_cpu = actions.float().cpu()
    try:
        denormalized = normalizer[key].unnormalize(actions_cpu)
        if isinstance(denormalized, torch.Tensor):
            denormalized = denormalized.numpy()
        has_nan = bool(np.any(np.isnan(denormalized)))
        has_inf = bool(np.any(np.isinf(denormalized)))
        max_abs = float(np.max(np.abs(denormalized)))

        report.add(assert_check(
            not has_nan and not has_inf and max_abs < 1000,
            "7.8a inference denormalize reasonable",
            f"max_abs={max_abs:.4f}, nan={has_nan}, inf={has_inf}",
        ))
    except Exception as e:
        report.add(assert_check(False, "7.8a inference denormalize", f"error: {e}"))


# ── Main ────────────────────────────────────────────────────────────────────

def run_all(
    config_path: str,
    normalizer_path: str | None = None,
    checkpoint_path: str | None = None,
    vla_shard: str | None = None,
    skip_visual: bool = False,
) -> PhaseReport:
    out_dir = get_output_dir(OUTPUT_PART)
    report = PhaseReport("Part 7: E2E Training + Checkpoint", out_dir)

    print("\n=== Part 7: E2E Training + Checkpoint ===\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.tests.full_chain_verification.part3_backbone_prefix_cache import (
        build_model_and_collator,
        make_mock_batch,
    )
    model, collator = build_model_and_collator(config_path, device)
    batch = make_mock_batch(collator, model, device)

    # Build optimizer for param group tests
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
    )

    # 7.1 Parameter groups
    test_all_trainable_in_optimizer(report, model, optimizer)
    test_no_duplicate_params(report, optimizer)
    test_frozen_params_not_in_optimizer(report, model, optimizer)

    # 7.2 LR schedule
    test_warmup_curve(report, out_dir)

    # 7.3 Overfit
    test_overfit_single_batch(report, model, batch, out_dir)

    # 7.4 Gradient health
    test_gradient_norms_per_component(report, model, batch)
    test_no_unused_parameters(report, model, batch)

    # 7.5 Dtypes
    test_preprocess_batch_dtypes(report, batch)

    # 7.6 Checkpoint (optional)
    if checkpoint_path:
        test_checkpoint_load(report, model, checkpoint_path)
        test_checkpoint_inference_reasonable(report, model, batch)
    else:
        report.add(assert_check(True, "7.6 checkpoint", "SKIPPED: no --checkpoint-path"))

    # 7.7 Real shard full pipeline (optional)
    test_real_shard_forward_loss(report, model, collator, normalizer_path, vla_shard)

    # 7.8 Inference denormalize (optional)
    test_inference_denormalize_reasonable(report, model, batch, normalizer_path)

    report.save()
    report.print_summary()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 7: E2E training verification")
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--normalizer-path", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--vla-shard", type=str, default=None,
                        help="Real VLA shard for full pipeline test (7.7)")
    parser.add_argument("--skip-visual", action="store_true")
    args = parser.parse_args()
    report = run_all(
        config_path=args.config_path,
        normalizer_path=args.normalizer_path,
        checkpoint_path=args.checkpoint_path,
        vla_shard=args.vla_shard,
        skip_visual=args.skip_visual,
    )
    exit(0 if report.all_passed else 1)
