"""
Phase 2: Embedding layer verification.

Checks:
  2.1  State/Action encoder output distribution -- mean/std/min/max, per-dim curves
  2.2  Time embedding correctness               -- t=0 vs t=1 distance, PCA visualization
  2.3  Fourier feature determinism              -- buffer frozen, output reproducible

Requires: nothing (standalone, no model weights needed)
Outputs:  outputs/pretrain_verification/phase2/  (report.json + PNG plots)

Usage:
    python -m src.tests.pretrain_verification.phase2_embeddings
    python -m src.tests.pretrain_verification.phase2_embeddings --skip-visual
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from src.model.action.action_head import FourierActionEncoder, MLPProjector
from src.model.common.modules import TimeEmbedding, GaussianFourierFeatureTransform

from src.tests.pretrain_verification.utils import (
    CheckResult,
    PhaseReport,
    assert_check,
    get_output_dir,
    plot_bar_chart,
    plot_scatter_2d,
    safe_import_plt,
    tensor_stats,
)


# ---------------------------------------------------------------------------
# Check 2.1: State/Action encoder output distribution
# ---------------------------------------------------------------------------

def check_2_1_encoder_distributions(skip_visual: bool, output_dir) -> CheckResult:
    """Verify that encoder outputs have reasonable magnitude and variance."""
    errors = []
    details = {}

    # Build encoders matching config dimensions
    configs = {
        "state_encoder": {"action_dim": 48, "width": 2560, "mlp_depth": 2},
        "ar_action_encoder": {"action_dim": 48, "width": 2560, "mlp_depth": 2},
        "action_encoder": {"action_dim": 48, "width": 1024, "mlp_depth": 2},
    }

    torch.manual_seed(42)
    for name, cfg in configs.items():
        encoder = FourierActionEncoder(
            action_dim=cfg["action_dim"], width=cfg["width"],
            time_cond=False, enable_fourier_embed=True,
            fourier_embed_dim=256, mlp_depth=cfg["mlp_depth"],
            final_layer_norm=True, use_mlp_layer_norm=True,
        )
        encoder.eval()

        # Input: normalized data in [-1, 1]
        x = torch.randn(8, 64, cfg["action_dim"]) * 0.5
        with torch.no_grad():
            out = encoder(x)

        stats = tensor_stats(out)
        details[name] = stats

        if stats["nan_count"] > 0:
            errors.append(f"{name}: contains NaN")
        if stats["inf_count"] > 0:
            errors.append(f"{name}: contains Inf")
        if abs(stats["max"]) > 100:
            errors.append(f"{name}: max magnitude {stats['max']:.2f} > 100")
        if stats["std"] < 1e-4:
            errors.append(f"{name}: near-constant output (std={stats['std']:.6f})")

        # Per-dim visualization
        if not skip_visual:
            out_np = out.detach().numpy().reshape(-1, out.shape[-1])
            dim_mean = out_np.mean(axis=0)
            dim_std = out_np.std(axis=0)
            plt = safe_import_plt()
            if plt is not None:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
                ax1.plot(dim_mean, linewidth=0.5)
                ax1.set_title(f"{name}: per-dim mean (width={cfg['width']})")
                ax1.set_xlabel("dimension")
                ax2.plot(dim_std, linewidth=0.5)
                ax2.set_title(f"{name}: per-dim std")
                ax2.set_xlabel("dimension")
                fig.tight_layout()
                fig.savefig(output_dir / f"{name}_per_dim.png", dpi=150)
                plt.close(fig)

    passed = len(errors) == 0
    msg = "All encoder outputs have reasonable distribution" if passed else f"{len(errors)} errors"
    return CheckResult(name="2.1 encoder_dist", passed=passed, message=msg, details=details)


# ---------------------------------------------------------------------------
# Check 2.2: Time embedding correctness
# ---------------------------------------------------------------------------

def check_2_2_time_embedding(skip_visual: bool, output_dir) -> CheckResult:
    """Verify time embedding has structure: different t -> different embeddings."""
    errors = []
    te = TimeEmbedding(time_hidden_size=1024)
    te.eval()

    # Test t=0 and t=1 are different
    with torch.no_grad():
        emb_0 = te(torch.tensor([[0.0]]))  # [1, 1, 1024]
        emb_1 = te(torch.tensor([[1.0]]))
        emb_05 = te(torch.tensor([[0.5]]))

    dist_01 = (emb_0 - emb_1).norm().item()
    dist_005 = (emb_0 - emb_05).norm().item()
    dist_051 = (emb_05 - emb_1).norm().item()

    if dist_01 < 1.0:
        errors.append(f"t=0 and t=1 too similar: L2 dist = {dist_01:.4f}")
    if dist_005 < 0.5:
        errors.append(f"t=0 and t=0.5 too similar: L2 dist = {dist_005:.4f}")

    # Check NaN/Inf
    for name, emb in [("t=0", emb_0), ("t=1", emb_1), ("t=0.5", emb_05)]:
        if torch.isnan(emb).any():
            errors.append(f"{name} embedding contains NaN")
        if torch.isinf(emb).any():
            errors.append(f"{name} embedding contains Inf")

    # Visualization: PCA of 100 time steps
    if not skip_visual:
        t_vals = torch.linspace(0, 1, 100).unsqueeze(1) # [100, 1]
        # TimeEmbedding expects [B, 1]
        with torch.no_grad():
            all_emb = te(t_vals)  # [100, 1024]

        all_emb_np = all_emb.numpy()
        # PCA to 2D
        mean = all_emb_np.mean(axis=0, keepdims=True)
        centered = all_emb_np - mean
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        pca_2d = centered @ vh[:2].T

        plot_scatter_2d(
            pca_2d, colors=np.linspace(0, 1, 100),
            title="Time Embedding PCA (color = t value)",
            save_path=output_dir / "time_embedding_pca.png",
            xlabel="PC1", ylabel="PC2",
            colorbar_label="t", cmap="coolwarm",
        )

        # L2 norm vs t
        norms = np.linalg.norm(all_emb_np, axis=1)
        plt = safe_import_plt()
        if plt is not None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(np.linspace(0, 1, 100), norms, linewidth=1.2)
            ax.set_xlabel("t")
            ax.set_ylabel("L2 norm")
            ax.set_title("Time Embedding L2 Norm vs t")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / "time_embedding_l2_norm.png", dpi=150)
            plt.close(fig)

    passed = len(errors) == 0
    msg = f"dist(t=0,t=1)={dist_01:.2f}" if passed else f"{len(errors)} errors"
    return CheckResult(name="2.2 time_embedding", passed=passed, message=msg,
                       details={"dist_01": dist_01, "dist_005": dist_005, "dist_051": dist_051})


# ---------------------------------------------------------------------------
# Check 2.3: Fourier feature determinism
# ---------------------------------------------------------------------------

def check_2_3_fourier_determinism() -> CheckResult:
    """Verify Fourier features are deterministic and buffer is frozen."""
    errors = []
    encoder = FourierActionEncoder(
        action_dim=48, width=1024, time_cond=False,
        enable_fourier_embed=True, fourier_embed_dim=256,
        mlp_depth=2, final_layer_norm=True, use_mlp_layer_norm=True,
    )
    encoder.eval()

    # Check buffer is not trainable
    if hasattr(encoder, "fourier"):
        b = encoder.fourier.b
        if b.requires_grad:
            errors.append("Fourier buffer b has requires_grad=True")
    elif hasattr(encoder, "input_proj") and hasattr(encoder.input_proj, "b"):
        b = encoder.input_proj.b
        if b.requires_grad:
            errors.append("Fourier buffer b has requires_grad=True")

    # Check determinism
    x = torch.randn(4, 16, 48)
    with torch.no_grad():
        out1 = encoder(x)
        out2 = encoder(x)

    if not torch.equal(out1, out2):
        max_diff = (out1 - out2).abs().max().item()
        errors.append(f"Non-deterministic: max diff = {max_diff}")

    passed = len(errors) == 0
    msg = "Fourier features deterministic" if passed else f"{len(errors)} errors"
    return CheckResult(name="2.3 fourier_determinism", passed=passed, message=msg,
                       details={"errors": errors})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Embedding verification")
    parser.add_argument("--skip-visual", action="store_true")
    args = parser.parse_args()

    output_dir = get_output_dir("phase2")
    report = PhaseReport("Phase 2: Embeddings", output_dir)

    print("Running embedding checks...\n")
    report.add(check_2_1_encoder_distributions(args.skip_visual, output_dir))
    report.add(check_2_2_time_embedding(args.skip_visual, output_dir))
    report.add(check_2_3_fourier_determinism())

    report.print_summary()
    report.save()
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
