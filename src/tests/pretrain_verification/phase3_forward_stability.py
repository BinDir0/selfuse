"""
Phase 3: Forward pass numerical stability verification.

Checks:
  3.1  Activation/parameter global audit -- NaN/Inf detection, AdaLNZero gate monitoring
  3.2  Prefix KV cache shape & detach   -- layer count, batch dim, knowledge insulation
  3.3  BFloat16 precision               -- bf16 vs fp32 loss relative error (CUDA only)

Requires: nothing by default (uses DummyBackbone)
Outputs:  outputs/pretrain_verification/phase3/  (report.json + PNG plots)

Usage:
    # With DummyBackbone (no real weights needed):
    python -m src.tests.pretrain_verification.phase3_forward_stability
    python -m src.tests.pretrain_verification.phase3_forward_stability --skip-visual

    # With real model (on training cluster):
    python -m src.tests.pretrain_verification.phase3_forward_stability \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from src.tests.pretrain_verification.utils import (
    CheckResult,
    PhaseReport,
    get_output_dir,
    safe_import_plt,
    tensor_stats,
)


# ---------------------------------------------------------------------------
# Helpers: build dummy model and batch (reuse from test_e2e_forward_backward)
# ---------------------------------------------------------------------------

def _build_dummy_model_and_batch():
    from src.tests.test_e2e_forward_backward import build_model, build_batch
    model = build_model(with_diffloss=True, knowledge_insulation=True)
    batch = build_batch(batch_size=2)
    return model, batch


# ---------------------------------------------------------------------------
# Check 3.1: Activation / parameter global audit
# ---------------------------------------------------------------------------

def check_3_1_activation_audit(model, batch, skip_visual: bool, output_dir: Path) -> CheckResult:
    """Run forward with hooks and check for NaN/Inf/outliers in all activations."""
    from src.tests.test_legendvla_hooks import (
        ActivationStatsCollector,
        collect_parameter_stats,
        visualize_stats,
    )

    collector = ActivationStatsCollector(leaf_only=True, max_dim=16384, sample_size=200000)
    collector.register(model)

    try:
        model.eval()
        with torch.no_grad():
            model("train", batch)
    finally:
        collector.remove()

    act_stats = collector.summarize()
    param_stats = collect_parameter_stats(model, include_buffers=False)

    # Check for NaN/Inf in activations
    errors = []
    nan_modules = []
    inf_modules = []
    high_outlier_modules = []

    for name, entry in act_stats.items():
        if entry.get("nan_count", 0) > 0:
            nan_modules.append(name)
            errors.append(f"NaN in activation: {name} ({entry['nan_count']} NaNs)")
        if entry.get("inf_count", 0) > 0:
            inf_modules.append(name)
            errors.append(f"Inf in activation: {name} ({entry['inf_count']} Infs)")
        ratio = entry.get("outlier_ratio", 0)
        if ratio > 0.05:
            high_outlier_modules.append((name, ratio))

    # Check AdaLNZero gate values specifically
    adaln_warnings = []
    for name, entry in act_stats.items():
        if "adaln" in name.lower() or "adaLN" in name:
            overall_max = entry.get("overall_max")
            if overall_max is not None and abs(overall_max) > 10:
                adaln_warnings.append(f"{name}: gate max = {overall_max:.2f} (>10, risk of grad spike)")

    # Visualization
    if not skip_visual:
        full_stats = {"activations": act_stats, "parameters": param_stats}
        visualize_stats(full_stats, str(output_dir / "stats_viz"), max_items=30)

    passed = len(nan_modules) == 0 and len(inf_modules) == 0
    msg = f"No NaN/Inf" if passed else f"NaN modules: {len(nan_modules)}, Inf modules: {len(inf_modules)}"
    if adaln_warnings:
        msg += f" | AdaLN warnings: {len(adaln_warnings)}"

    return CheckResult(
        name="3.1 activation_audit", passed=passed, message=msg,
        details={
            "nan_modules": nan_modules[:5],
            "inf_modules": inf_modules[:5],
            "high_outlier_modules": [(n, f"{r:.3f}") for n, r in high_outlier_modules[:5]],
            "adaln_warnings": adaln_warnings,
            "total_activation_modules": len(act_stats),
        },
    )


# ---------------------------------------------------------------------------
# Check 3.2: Prefix KV cache shape and detach
# ---------------------------------------------------------------------------

def check_3_2_prefix_cache(model, batch) -> CheckResult:
    """Verify prefix KV cache shape and gradient detachment under knowledge insulation."""
    errors = []

    model.eval()
    with torch.no_grad():
        slot_embeds = model.build_slot_embeddings(batch, add_action_noise=False)
        backbone_output = model.forward_backbone_stream(batch, slot_embeds)

    pc = backbone_output.prefix_cache
    if pc is None:
        return CheckResult(name="3.2 prefix_cache", passed=False,
                           message="prefix_cache is None", details={})

    # Check keys shape: [num_layers, B, num_kv_heads, seq_len, head_dim]
    keys = pc.keys
    num_layers = keys.shape[0]
    B = batch["input_ids"].shape[0]

    if keys.shape[0] != model.backbone.num_layers:
        errors.append(f"num_layers mismatch: cache={keys.shape[0]}, model={model.backbone.num_layers}")
    if keys.shape[1] != B:
        errors.append(f"batch dim mismatch: cache={keys.shape[1]}, batch={B}")

    # Check NaN/Inf
    if torch.isnan(keys).any():
        errors.append("NaN in prefix cache keys")
    if torch.isnan(pc.values).any():
        errors.append("NaN in prefix cache values")

    # Check mask
    mask = pc.mask
    if mask is not None:
        if mask.shape[0] != B:
            errors.append(f"mask batch dim mismatch: {mask.shape[0]} != {B}")

    # Knowledge insulation: check detach
    if hasattr(model, "knowledge_insulation") and model.knowledge_insulation:
        if keys.requires_grad:
            errors.append("knowledge_insulation=True but prefix_cache.keys has requires_grad=True")

    passed = len(errors) == 0
    msg = f"Cache shape [{list(keys.shape)}]" if passed else f"{len(errors)} errors"
    return CheckResult(name="3.2 prefix_cache", passed=passed, message=msg,
                       details={"cache_shape": list(keys.shape), "errors": errors})


# ---------------------------------------------------------------------------
# Check 3.3: BFloat16 precision
# ---------------------------------------------------------------------------

def check_3_3_bf16_precision(model_fn, batch_fn) -> CheckResult:
    """Compare forward pass results in fp32 vs bf16."""
    if not torch.cuda.is_available():
        return CheckResult(name="3.3 bf16_precision", passed=True,
                           message="Skipped (no CUDA)", details={"skipped": True})

    errors = []
    device = torch.device("cuda")

    # FP32 forward
    torch.manual_seed(42)
    model_fp32 = model_fn()
    model_fp32.to(device=device, dtype=torch.float32)
    model_fp32.eval()
    batch_fp32 = batch_fn()
    batch_fp32 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_fp32.items()}
    batch_fp32["t"] = batch_fp32["t"].float()
    with torch.no_grad():
        loss_fp32 = model_fp32("train", batch_fp32)

    # BF16 forward
    torch.manual_seed(42)
    model_bf16 = model_fn()
    model_bf16.to(device=device, dtype=torch.bfloat16)
    model_bf16.eval()
    batch_bf16 = batch_fn()
    batch_bf16 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_bf16.items()}
    batch_bf16["t"] = batch_bf16["t"].bfloat16()
    batch_bf16["states"] = batch_bf16["states"].bfloat16()
    batch_bf16["actions"] = batch_bf16["actions"].bfloat16()
    batch_bf16["pixel_values"] = batch_bf16["pixel_values"].bfloat16()
    with torch.no_grad():
        loss_bf16 = model_bf16("train", batch_bf16)

    # Compare total loss
    total_fp32 = loss_fp32["total_loss"].float().item()
    total_bf16 = loss_bf16["total_loss"].float().item()
    rel_error = abs(total_bf16 - total_fp32) / (abs(total_fp32) + 1e-8)

    if rel_error > 0.1:
        errors.append(f"bf16/fp32 loss relative error {rel_error:.4f} > 0.1")

    passed = len(errors) == 0
    msg = f"rel_error={rel_error:.4f}" if passed else errors[0]
    return CheckResult(name="3.3 bf16_precision", passed=passed, message=msg,
                       details={"fp32_loss": total_fp32, "bf16_loss": total_bf16, "rel_error": rel_error})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Forward stability verification")
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--config-path", type=str, default=None,
                        help="Optional: use real model config instead of dummy")
    args = parser.parse_args()

    output_dir = get_output_dir("phase3")
    report = PhaseReport("Phase 3: Forward Stability", output_dir)

    print("Building model and batch...\n")
    from src.tests.test_e2e_forward_backward import build_model, build_batch
    model = build_model(with_diffloss=True, knowledge_insulation=True)
    batch = build_batch(batch_size=2)

    # Check 3.1: Activation audit
    report.add(check_3_1_activation_audit(model, batch, args.skip_visual, output_dir))

    # Check 3.2: Prefix cache
    report.add(check_3_2_prefix_cache(model, batch))

    # Check 3.3: BF16 precision
    report.add(check_3_3_bf16_precision(
        lambda: build_model(with_diffloss=True, knowledge_insulation=True),
        lambda: build_batch(batch_size=2),
    ))

    report.print_summary()
    report.save()
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
