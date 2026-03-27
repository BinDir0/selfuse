"""
Phase 0: Normalizer statistics verification.

Checks:
  0.1  Stats distribution review  -- scale/offset/q01/q99 per dim, ignored dims
  0.2  Round-trip consistency      -- normalize -> unnormalize reconstruction error
  0.3  Normalized data distribution -- per-dim histogram, bounded in [-1, 1]

Requires: normalizer.pkl (pre-computed normalizer checkpoint)
Outputs:  outputs/pretrain_verification/phase0/  (report.json + PNG plots)

Usage:
    # Basic (required):
    python -m src.tests.pretrain_verification.phase0_normalizer \
        --normalizer-path /path/to/normalizer.pkl

    # With real data shards for distribution check:
    python -m src.tests.pretrain_verification.phase0_normalizer \
        --normalizer-path /path/to/normalizer.pkl \
        --data-shards '/data/shards/{000..099}.tar'

    # Skip visualization:
    python -m src.tests.pretrain_verification.phase0_normalizer \
        --normalizer-path /path/to/normalizer.pkl --skip-visual
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.tests.pretrain_verification.utils import (
    CheckResult,
    PhaseReport,
    assert_check,
    get_output_dir,
    plot_bar_chart,
    plot_histogram_panel,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dims 6-18 are typically ignored (e.g. wrist rotation dims set by ignore_dim)
IGNORED_DIM_SLICE = slice(6, 18)
IGNORED_DIM_INDICES = list(range(6, 18))

OUTPUT_PHASE = "phase0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_normalizer(path: str | Path) -> torch.nn.Module:
    """Load a pickled LinearNormalizer checkpoint."""
    path = Path(path)
    with open(path, "rb") as f:
        normalizer = pickle.load(f)
    normalizer.eval()
    return normalizer


def get_normalizer_keys(normalizer) -> List[str]:
    """Return the top-level keys stored in the normalizer params_dict."""
    return [k for k in normalizer.params_dict.keys()]


def extract_params(params_dict_entry) -> Dict[str, np.ndarray]:
    """
    Extract scale, offset, and input_stats from a single key entry
    in the normalizer params_dict. Returns numpy arrays.
    """
    result = {}
    result["scale"] = params_dict_entry["scale"].detach().cpu().numpy()
    result["offset"] = params_dict_entry["offset"].detach().cpu().numpy()

    input_stats = params_dict_entry["input_stats"]
    for stat_name in ("q01", "q99", "mean", "std", "min", "max"):
        if stat_name in input_stats:
            result[stat_name] = input_stats[stat_name].detach().cpu().numpy()

    # ignored_dim_mask may or may not exist
    if "ignored_dim_mask" in params_dict_entry:
        result["ignored_dim_mask"] = (
            params_dict_entry["ignored_dim_mask"].detach().cpu().numpy()
        )
    return result


def build_ignored_mask(params: Dict[str, np.ndarray], n_dims: int) -> np.ndarray:
    """
    Build a boolean mask for ignored dims. Prefer the stored ignored_dim_mask
    if available; otherwise fall back to the hard-coded IGNORED_DIM_INDICES
    (only when n_dims >= 18).
    """
    if "ignored_dim_mask" in params:
        return params["ignored_dim_mask"] > 0.5

    mask = np.zeros(n_dims, dtype=bool)
    if n_dims >= 18:
        mask[IGNORED_DIM_INDICES] = True
    return mask


# ---------------------------------------------------------------------------
# Check 0.1: Stats Distribution Review
# ---------------------------------------------------------------------------

def check_stats_distribution(
    normalizer,
    output_dir: Path,
    skip_visual: bool,
) -> CheckResult:
    """
    Validate per-dim scale, offset, q01, q99 for every key in the
    normalizer. Assert ignored dims have scale=1, offset=0, and
    non-ignored dims have reasonable scale range and q99 > q01.
    """
    issues: list[str] = []
    details: dict = {}

    keys = get_normalizer_keys(normalizer)
    for key in keys:
        params = extract_params(normalizer.params_dict[key])
        n_dims = len(params["scale"])
        ignored = build_ignored_mask(params, n_dims)
        non_ignored = ~ignored

        scale = params["scale"]
        offset = params["offset"]

        # --- Assert ignored dims ---
        if ignored.any():
            bad_scale = ~np.isclose(scale[ignored], 1.0, atol=1e-6)
            bad_offset = ~np.isclose(offset[ignored], 0.0, atol=1e-6)
            if bad_scale.any():
                idx = np.where(ignored)[0][bad_scale]
                issues.append(
                    f"[{key}] ignored dims with scale != 1.0: {idx.tolist()}"
                )
            if bad_offset.any():
                idx = np.where(ignored)[0][bad_offset]
                issues.append(
                    f"[{key}] ignored dims with offset != 0.0: {idx.tolist()}"
                )

        # --- Assert non-ignored dims have scale in [0.01, 1000] ---
        if non_ignored.any():
            abs_scale = np.abs(scale[non_ignored])
            out_of_range = (abs_scale < 0.01) | (abs_scale > 1000)
            if out_of_range.any():
                idx = np.where(non_ignored)[0][out_of_range]
                issues.append(
                    f"[{key}] non-ignored dims with |scale| outside [0.01, 1000]: "
                    f"{idx.tolist()}, values={abs_scale[out_of_range].tolist()}"
                )

        # --- Assert q99 > q01 for non-ignored dims ---
        if "q01" in params and "q99" in params and non_ignored.any():
            q01 = params["q01"]
            q99 = params["q99"]
            bad_range = q99[non_ignored] <= q01[non_ignored]
            if bad_range.any():
                idx = np.where(non_ignored)[0][bad_range]
                issues.append(
                    f"[{key}] non-ignored dims with q99 <= q01: {idx.tolist()}"
                )

        # --- Visualization ---
        if not skip_visual:
            # Anomalous dim mask: ignored dims OR out-of-range scale
            anomaly_mask = np.zeros(n_dims, dtype=bool)
            anomaly_mask[ignored] = True
            if non_ignored.any():
                abs_s = np.abs(scale)
                anomaly_mask |= (abs_s < 0.01) | (abs_s > 1000)

            plot_bar_chart(
                values=scale,
                title=f"[{key}] Scale per dim",
                xlabel="dim",
                ylabel="scale",
                save_path=output_dir / f"stats_{key}_scale.png",
                highlight_mask=anomaly_mask,
            )
            plot_bar_chart(
                values=offset,
                title=f"[{key}] Offset per dim",
                xlabel="dim",
                ylabel="offset",
                save_path=output_dir / f"stats_{key}_offset.png",
                highlight_mask=anomaly_mask,
            )
            if "q01" in params and "q99" in params:
                q_range = params["q99"] - params["q01"]
                plot_bar_chart(
                    values=q_range,
                    title=f"[{key}] q99 - q01 range per dim",
                    xlabel="dim",
                    ylabel="q99 - q01",
                    save_path=output_dir / f"stats_{key}_q_range.png",
                    highlight_mask=anomaly_mask,
                )

        details[key] = {
            "n_dims": n_dims,
            "n_ignored": int(ignored.sum()),
            "scale_min": float(np.abs(scale[non_ignored]).min()) if non_ignored.any() else None,
            "scale_max": float(np.abs(scale[non_ignored]).max()) if non_ignored.any() else None,
        }

    passed = len(issues) == 0
    message = "all dims OK" if passed else "; ".join(issues)
    return CheckResult(
        name="0.1 Stats Distribution Review",
        passed=passed,
        details=details,
        message=message,
    )


# ---------------------------------------------------------------------------
# Check 0.2: Round-trip Consistency
# ---------------------------------------------------------------------------

def check_roundtrip(normalizer) -> CheckResult:
    """
    Generate random data in [q01, q99], normalize -> unnormalize,
    and verify reconstruction error. Also test outlier clamping behavior.
    """
    issues: list[str] = []
    details: dict = {}
    n_samples = 1000

    keys = get_normalizer_keys(normalizer)
    for key in keys:
        params = extract_params(normalizer.params_dict[key])
        n_dims = len(params["scale"])
        ignored = build_ignored_mask(params, n_dims)

        q01 = params.get("q01")
        q99 = params.get("q99")

        if q01 is None or q99 is None:
            # Fall back to min/max if quantiles unavailable
            q01 = params["min"]
            q99 = params["max"]

        q01_t = torch.from_numpy(q01).float()
        q99_t = torch.from_numpy(q99).float()

        # Generate random data uniformly in [q01, q99]
        rand_01 = torch.rand(n_samples, n_dims)
        data = q01_t.unsqueeze(0) + rand_01 * (q99_t - q01_t).unsqueeze(0)

        # Round-trip: normalize -> unnormalize
        normed = normalizer._normalize_impl({key: data}, forward=True)[key]
        recon = normalizer._normalize_impl({key: normed}, forward=False)[key]

        err = (data - recon).abs()
        max_err = err.max().item()
        details[f"{key}_roundtrip_max_err"] = max_err

        if max_err > 1e-5:
            issues.append(
                f"[{key}] round-trip max error {max_err:.2e} exceeds 1e-5"
            )

        # --- Outlier test: 3x outside [q01, q99] ---
        range_width = q99_t - q01_t
        # Clamp range_width to avoid zero for ignored dims
        range_width = range_width.clamp(min=1e-6)
        outlier_low = q01_t - 3 * range_width
        outlier_high = q99_t + 3 * range_width
        outlier_data = torch.cat([
            outlier_low.unsqueeze(0).expand(n_samples // 2, -1),
            outlier_high.unsqueeze(0).expand(n_samples // 2, -1),
        ], dim=0)

        normed_outlier = normalizer._normalize_impl(
            {key: outlier_data}, forward=True
        )[key]

        # For non-ignored dims the normalized values should be within a
        # reasonable range because the forward pass clips to [q01, q99]
        # before scaling, so normalized output is bounded by
        # [output_min, output_max] (typically [-1, 1]).
        non_ignored_mask = ~torch.from_numpy(ignored)
        if non_ignored_mask.any():
            normed_abs = normed_outlier[:, non_ignored_mask].abs()
            p99 = torch.quantile(normed_abs, 0.99).item()
            details[f"{key}_outlier_normed_p99_abs"] = p99
            # After clipping + scaling, values should stay near [-1, 1]
            if p99 > 5.0:
                issues.append(
                    f"[{key}] outlier normalized p99 abs = {p99:.2f}, "
                    f"expected < 5.0"
                )

    passed = len(issues) == 0
    message = "round-trip OK" if passed else "; ".join(issues)
    return CheckResult(
        name="0.2 Round-trip Consistency",
        passed=passed,
        details=details,
        message=message,
    )


# ---------------------------------------------------------------------------
# Check 0.3: Normalized Data Distribution
# ---------------------------------------------------------------------------

def check_normalized_distribution(
    normalizer,
    output_dir: Path,
    data_shards: Optional[str],
    skip_visual: bool,
) -> CheckResult:
    """
    Normalize real shard data (if provided) or synthetic data drawn from
    normalizer stats, then verify per-dim distributions are bounded.
    """
    issues: list[str] = []
    details: dict = {}

    keys = get_normalizer_keys(normalizer)

    for key in keys:
        params = extract_params(normalizer.params_dict[key])
        n_dims = len(params["scale"])
        ignored = build_ignored_mask(params, n_dims)
        non_ignored = ~ignored

        normed_data: Optional[torch.Tensor] = None

        # --- Try to load real data from WebDataset shards ---
        if data_shards is not None:
            try:
                import webdataset as wds

                dataset = (
                    wds.WebDataset(data_shards)
                    .decode()
                    .to_tuple(key + ".pth", handler=wds.warn_and_continue)
                )
                collected: list[torch.Tensor] = []
                total = 0
                target = 50000
                for (sample,) in dataset:
                    t = torch.as_tensor(sample).float()
                    if t.dim() == 1:
                        t = t.unsqueeze(0)
                    collected.append(t)
                    total += t.shape[0]
                    if total >= target:
                        break
                if collected:
                    raw = torch.cat(collected, dim=0)[:target]
                    normed_data = normalizer._normalize_impl(
                        {key: raw}, forward=True
                    )[key]
            except Exception as e:
                print(f"  WARNING: could not load shards for key={key}: {e}")
                normed_data = None

        # --- Fall back to synthetic data from normalizer stats ---
        if normed_data is None:
            n_synth = 50000
            q01 = torch.from_numpy(params.get("q01", params["min"])).float()
            q99 = torch.from_numpy(params.get("q99", params["max"])).float()
            mean = torch.from_numpy(params["mean"]).float()
            std = torch.from_numpy(params["std"]).float()

            # Draw from truncated normal within [q01, q99]
            synth = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(n_synth, n_dims)
            synth = synth.clamp(min=q01.unsqueeze(0), max=q99.unsqueeze(0))
            normed_data = normalizer._normalize_impl(
                {key: synth}, forward=True
            )[key]
            details[f"{key}_data_source"] = "synthetic"
        else:
            details[f"{key}_data_source"] = "shards"

        # --- Assertions ---
        normed_np = normed_data.detach().cpu().numpy()
        if non_ignored.any():
            abs_vals = np.abs(normed_np[:, non_ignored])
            per_dim_p99 = np.percentile(abs_vals, 99, axis=0)
            worst_dim = int(np.argmax(per_dim_p99))
            worst_val = float(per_dim_p99[worst_dim])
            details[f"{key}_worst_p99_abs"] = worst_val
            details[f"{key}_worst_p99_dim"] = int(np.where(non_ignored)[0][worst_dim])

            bad_dims = per_dim_p99 > 2.0
            if bad_dims.any():
                bad_indices = np.where(non_ignored)[0][bad_dims]
                bad_values = per_dim_p99[bad_dims]
                issues.append(
                    f"[{key}] dims with 99th pct abs > 2.0: "
                    f"{bad_indices.tolist()}, values={bad_values.tolist()}"
                )

        # --- Visualization ---
        if not skip_visual:
            plot_histogram_panel(
                data=normed_np,
                title=f"[{key}] Normalized data distribution",
                save_path=output_dir / f"normed_hist_{key}.png",
            )

    passed = len(issues) == 0
    message = "distributions OK" if passed else "; ".join(issues)
    return CheckResult(
        name="0.3 Normalized Data Distribution",
        passed=passed,
        details=details,
        message=message,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 0 normalizer verification",
    )
    parser.add_argument(
        "--normalizer-path",
        type=str,
        required=True,
        help="Path to normalizer.pkl",
    )
    parser.add_argument(
        "--data-shards",
        type=str,
        default=None,
        help="Optional WebDataset shard URL pattern for real data",
    )
    parser.add_argument(
        "--skip-visual",
        action="store_true",
        help="Skip generating visualization plots",
    )
    args = parser.parse_args()

    output_dir = get_output_dir(OUTPUT_PHASE)
    report = PhaseReport("Phase 0: Normalizer", output_dir)

    print(f"Loading normalizer from: {args.normalizer_path}")
    normalizer = load_normalizer(args.normalizer_path)
    keys = get_normalizer_keys(normalizer)
    print(f"  Keys: {keys}")

    # Check 0.1
    result_01 = check_stats_distribution(normalizer, output_dir, args.skip_visual)
    report.add(result_01)

    # Check 0.2
    result_02 = check_roundtrip(normalizer)
    report.add(result_02)

    # Check 0.3
    result_03 = check_normalized_distribution(
        normalizer, output_dir, args.data_shards, args.skip_visual
    )
    report.add(result_03)

    report.print_summary()
    report_path = report.save()
    print(f"Report saved to: {report_path}")

    if not report.all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

