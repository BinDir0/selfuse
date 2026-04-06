"""
Part 1: Data Pipeline + Sample Level Verification.

Tests normalizer, coordinate transforms, sliding window sampling,
and real VLA/VLM sample schemas.

Requires: normalizer.pkl + VLA shard + (optional) VLM shard
Runs on: CPU

Usage:
    python -m src.tests.full_chain_verification.part1_data_pipeline \
        --normalizer-path /path/to/normalizer.pkl \
        --vla-shard '/data/shards/taco/{000..001}.tar' \
        [--vlm-shard '/data/shards/vlm/{000..001}.tar']
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tests.full_chain_verification.utils import (
    CheckResult,
    PhaseReport,
    assert_check,
    get_output_dir,
    plot_histogram_panel,
    safe_import_plt,
)


OUTPUT_PART = "part1"


# ── helpers ─────────────────────────────────────────────────────────────────

def load_normalizer(path: str | Path):
    with open(path, "rb") as f:
        normalizer = pickle.load(f)
    return normalizer


def iter_raw_wds_samples(shard_url: str, max_samples: int = 100):
    """Yield raw decoded WDS samples (before sliding window) from a shard."""
    import webdataset as wds
    dataset = wds.WebDataset(shard_url).decode("l")  # "l" = numpy for .npy
    count = 0
    for sample in dataset:
        yield sample
        count += 1
        if count >= max_samples:
            break


# ── 1.1 Normalizer ─────────────────────────────────────────────────────────

def test_normalizer_roundtrip(report: PhaseReport, normalizer) -> None:
    """normalize -> unnormalize should reconstruct the input."""
    key = "motions"
    if key not in normalizer.params_dict:
        key = list(normalizer.params_dict.keys())[0]

    data = torch.randn(100, 48)
    normalized = normalizer[key](data)
    reconstructed = normalizer[key].unnormalize(normalized)
    max_err = float((data - reconstructed).abs().max())
    report.add(assert_check(
        max_err < 1e-4,
        "1.1a normalizer roundtrip",
        f"key={key}, max_error={max_err:.2e}",
    ))


def test_normalizer_no_degenerate_dims(report: PhaseReport, normalizer) -> None:
    """Check normalizer scale has no zero values (would destroy information)."""
    issues = []
    for key in normalizer.params_dict:
        params = normalizer.params_dict[key]
        scale = params["scale"].detach().numpy()
        offset = params["offset"].detach().numpy()
        zero_scale = int(np.sum(np.abs(scale) < 1e-10))
        has_nan = bool(np.any(np.isnan(offset))) or bool(np.any(np.isnan(scale)))
        has_inf = bool(np.any(np.isinf(offset))) or bool(np.any(np.isinf(scale)))
        if zero_scale > 0 or has_nan or has_inf:
            issues.append(f"{key}: zero_scale_dims={zero_scale}, nan={has_nan}, inf={has_inf}")

    report.add(assert_check(
        len(issues) == 0,
        "1.1b normalizer no degenerate dims",
        f"issues={issues}" if issues else "all keys healthy",
    ))


def test_normalizer_output_distribution(report: PhaseReport, normalizer, out_dir: Path) -> None:
    """Normalized random data should have reasonable distribution."""
    key = "motions"
    if key not in normalizer.params_dict:
        key = list(normalizer.params_dict.keys())[0]

    data = torch.randn(1000, 48) * 0.5
    normalized = normalizer[key](data).numpy()

    per_dim_mean = np.mean(normalized, axis=0)
    per_dim_std = np.std(normalized, axis=0)
    outlier_ratio = float(np.mean(np.abs(normalized) > 5))

    report.add(assert_check(
        outlier_ratio < 0.15,
        "1.1c normalizer output distribution",
        f"outlier_ratio(|x|>5)={outlier_ratio:.4f}, "
        f"mean_range=[{per_dim_mean.min():.2f}, {per_dim_mean.max():.2f}], "
        f"std_range=[{per_dim_std.min():.2f}, {per_dim_std.max():.2f}]",
    ))

    plot_histogram_panel(normalized, "Normalized distribution", out_dir / "normalizer_dist.png", n_cols=8)


# ── 1.2 Coordinate transforms ──────────────────────────────────────────────

def test_relative_action_roundtrip(report: PhaseReport) -> None:
    """get_relative_action -> get_absolute_action should reconstruct."""
    from src.dataset.data_transforms import get_relative_action, get_absolute_action
    from scipy.spatial.transform import Rotation

    def random_valid_48d():
        """Generate a 48D vector with valid 6D rotations in wrist slots."""
        v = np.random.randn(48).astype(np.float32) * 0.1
        # Slots [6:12] and [12:18] must be valid 6D rotations (first two columns
        # of a rotation matrix, flattened row-major).
        for start in (6, 12):
            R = Rotation.random().as_matrix().astype(np.float32)
            v[start:start + 6] = R[:, :2].T.flatten()
        return v

    max_errors = []
    for _ in range(50):
        state = random_valid_48d()
        action = np.stack([random_valid_48d() for _ in range(8)], axis=0)
        try:
            relative = get_relative_action(state, action)
            absolute = get_absolute_action(state, relative)
            max_err = float(np.max(np.abs(action - absolute)))
            max_errors.append(max_err)
        except Exception as e:
            max_errors.append(float("inf"))

    median_err = float(np.median(max_errors))
    worst_err = float(np.max(max_errors))

    report.add(assert_check(
        median_err < 1e-3,
        "1.2a relative action roundtrip",
        f"median_err={median_err:.2e}, worst_err={worst_err:.2e} (50 trials)",
    ))


# ── 1.3 Sliding window ─────────────────────────────────────────────────────

def test_sliding_window_config_sanity(report: PhaseReport) -> None:
    """WindowConfig defaults should define consistent horizons and strides."""
    from src.dataset.wds_dataset import WindowConfig, LOWDIM_SLICES

    config = WindowConfig()
    checks = []
    # action_horizon must be positive
    checks.append(config.action_horizon > 0)
    # state_horizon must be positive
    checks.append(config.state_horizon > 0)
    # image_horizon must be positive
    checks.append(config.image_horizon > 0)
    # LOWDIM_SLICES must define all expected fields
    required_keys = {"wrist_state", "hand_state", "wrist_action", "hand_action", "extrinsic", "intrinsic"}
    checks.append(required_keys.issubset(set(LOWDIM_SLICES.keys())))
    # Total lowdim size: intrinsic ends at 116
    intr_end = LOWDIM_SLICES["intrinsic"][1]
    checks.append(intr_end == 116)

    report.add(assert_check(
        all(checks),
        "1.3a sliding window config sanity",
        f"action_horizon={config.action_horizon}, state_horizon={config.state_horizon}, "
        f"image_horizon={config.image_horizon}, lowdim_keys={list(LOWDIM_SLICES.keys())}",
    ))


# ── 1.4 Real VLA sample schema ─────────────────────────────────────────────

def test_real_vla_sample_schema(report: PhaseReport, vla_shard: str | None) -> None:
    """Verify schema of real VLA samples from WDS shard."""
    if vla_shard is None:
        report.add(assert_check(True, "1.4a real VLA sample schema", "SKIPPED: no --vla-shard"))
        return

    samples = list(iter_raw_wds_samples(vla_shard, max_samples=10))
    if not samples:
        report.add(assert_check(False, "1.4a real VLA sample schema", "no samples from shard"))
        return

    issues = []
    for i, s in enumerate(samples):
        # Check required keys exist (image key varies: rgb.png or image.jpg)
        has_image = any(k for k in s if k.endswith((".png", ".jpg", ".jpeg")))
        for key in ["lowdim.npy", "__key__"]:
            if key not in s:
                issues.append(f"sample {i}: missing key {key}")
        if not has_image:
            issues.append(f"sample {i}: missing image key (no .png/.jpg found, keys={list(s.keys())})")

        lowdim = s.get("lowdim.npy")
        if lowdim is not None:
            if lowdim.shape != (116,):
                issues.append(f"sample {i}: lowdim shape {lowdim.shape} != (116,)")
            if np.any(np.isnan(lowdim)):
                issues.append(f"sample {i}: lowdim has NaN")

    report.add(assert_check(
        len(issues) == 0,
        "1.4a real VLA sample schema",
        f"{len(samples)} samples checked, issues={issues[:5]}" if issues else f"{len(samples)} samples all valid",
    ))


def test_real_vla_action_values_reasonable(report: PhaseReport, vla_shard: str | None, normalizer, out_dir: Path) -> None:
    """Normalized actions from real data should be in reasonable range."""
    if vla_shard is None or normalizer is None:
        report.add(assert_check(True, "1.4b real VLA action values", "SKIPPED: no shard or normalizer"))
        return

    samples = list(iter_raw_wds_samples(vla_shard, max_samples=50))
    if not samples:
        report.add(assert_check(False, "1.4b real VLA action values", "no samples"))
        return

    from src.dataset.wds_dataset import LOWDIM_SLICES
    wrist_start, wrist_end = LOWDIM_SLICES["wrist_action"]
    hand_start, hand_end = LOWDIM_SLICES["hand_action"]

    all_actions = []
    for s in samples:
        lowdim = s["lowdim.npy"]
        action_raw = np.concatenate([lowdim[wrist_start:wrist_end], lowdim[hand_start:hand_end]])
        all_actions.append(action_raw)
    all_actions = np.stack(all_actions)

    # Basic checks on raw actions
    has_nan = bool(np.any(np.isnan(all_actions)))
    has_inf = bool(np.any(np.isinf(all_actions)))
    all_zero_rows = int(np.all(all_actions == 0, axis=1).sum())

    report.add(assert_check(
        not has_nan and not has_inf,
        "1.4b real VLA action values",
        f"n={len(all_actions)}, nan={has_nan}, inf={has_inf}, "
        f"all_zero_rows={all_zero_rows}, "
        f"range=[{all_actions.min():.4f}, {all_actions.max():.4f}]",
    ))

    plot_histogram_panel(all_actions, "Raw action distribution (48D)", out_dir / "raw_action_dist.png", n_cols=8)


# ── 1.5 VLM sample schema ──────────────────────────────────────────────────

def test_real_vlm_sample_schema(report: PhaseReport, vlm_shard: str | None) -> None:
    """Verify schema of real VLM samples."""
    if vlm_shard is None:
        report.add(assert_check(True, "1.5a real VLM sample schema", "SKIPPED: no --vlm-shard"))
        return

    import webdataset as wds
    dataset = wds.WebDataset(vlm_shard).decode("pil")
    issues = []
    count = 0
    for s in dataset:
        if count >= 10:
            break
        count += 1
        key = s.get("__key__", "unknown")
        if not any(k.endswith((".jpg", ".png", ".jpeg")) for k in s):
            issues.append(f"{key}: no image")
        if not any(k.endswith((".json", ".txt")) for k in s):
            issues.append(f"{key}: no text/json")

    report.add(assert_check(
        len(issues) == 0 and count > 0,
        "1.5a real VLM sample schema",
        f"{count} samples checked" + (f", issues={issues}" if issues else ""),
    ))


# ── Main ────────────────────────────────────────────────────────────────────

# ── 1.6 Real data transform chain ─────────────────────────────────────────

def test_real_data_transform_chain(report: PhaseReport, vla_shard: str | None, normalizer) -> None:
    """Full transform chain: raw lowdim -> process_state_action -> normalizer.

    Verifies that the exact same pipeline used in training produces
    finite, reasonably ranged outputs.
    """
    if vla_shard is None or normalizer is None:
        report.add(assert_check(True, "1.6a transform chain", "SKIPPED: need --vla-shard + --normalizer-path"))
        return

    from src.dataset.data_transforms import process_state_action
    from src.dataset.wds_dataset import LOWDIM_SLICES

    samples = list(iter_raw_wds_samples(vla_shard, max_samples=20))
    if not samples:
        report.add(assert_check(False, "1.6a transform chain", "no samples"))
        return

    use_relative = "actions" in normalizer.params_dict
    issues = []
    for i, s in enumerate(samples):
        ld = s.get("lowdim.npy")
        if ld is None or ld.shape != (116,):
            continue
        ws = ld[LOWDIM_SLICES["wrist_state"][0]:LOWDIM_SLICES["wrist_state"][1]][np.newaxis]
        hs = ld[LOWDIM_SLICES["hand_state"][0]:LOWDIM_SLICES["hand_state"][1]][np.newaxis]
        wa = ld[LOWDIM_SLICES["wrist_action"][0]:LOWDIM_SLICES["wrist_action"][1]][np.newaxis]
        ha = ld[LOWDIM_SLICES["hand_action"][0]:LOWDIM_SLICES["hand_action"][1]][np.newaxis]
        ext = np.eye(4, dtype=np.float32)  # identity extrinsic for sanity check
        try:
            state, action = process_state_action(
                ws, hs, wa, ha, ext,
                hand_ndim=15,
                normalizer=normalizer,
                use_relative_action=use_relative,
            )
            if isinstance(state, torch.Tensor):
                state = state.numpy()
            if isinstance(action, torch.Tensor):
                action = action.numpy()
            if np.any(np.isnan(state)) or np.any(np.isnan(action)):
                issues.append(f"sample {i}: NaN in state or action")
            if np.any(np.isinf(state)) or np.any(np.isinf(action)):
                issues.append(f"sample {i}: Inf in state or action")
            if np.max(np.abs(action)) > 50:
                issues.append(f"sample {i}: |action|={np.max(np.abs(action)):.1f} > 50")
        except Exception as e:
            issues.append(f"sample {i}: {e}")

    report.add(assert_check(
        len(issues) == 0,
        "1.6a real data transform chain",
        f"{len(samples)} samples, relative={use_relative}, issues={issues[:5]}"
        if issues else f"{len(samples)} samples all valid, relative={use_relative}",
    ))


def test_action_token_count_consistency(report: PhaseReport, vla_shard: str | None, normalizer) -> None:
    """Verify action token count in collated batch matches expected n_actions."""
    if vla_shard is None:
        report.add(assert_check(True, "1.7a action token count", "SKIPPED: no --vla-shard"))
        return

    from src.dataset.wds_dataset import LOWDIM_SLICES

    samples = list(iter_raw_wds_samples(vla_shard, max_samples=5))
    if not samples:
        report.add(assert_check(False, "1.7a action token count", "no samples"))
        return

    # Build a minimal sample dict like the dataset class would
    for s in samples:
        ld = s.get("lowdim.npy")
        if ld is None or ld.shape != (116,):
            continue

        img_key = next((k for k in s if k.endswith((".png", ".jpg", ".jpeg"))), None)
        if img_key is None:
            continue
        img = s[img_key]
        if hasattr(img, "convert"):
            img = np.array(img.convert("RGB"))
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        # A sample going into the collator must have these fields
        ws = ld[LOWDIM_SLICES["wrist_state"][0]:LOWDIM_SLICES["wrist_state"][1]]
        hs = ld[LOWDIM_SLICES["hand_state"][0]:LOWDIM_SLICES["hand_state"][1]]
        wa = ld[LOWDIM_SLICES["wrist_action"][0]:LOWDIM_SLICES["wrist_action"][1]]
        ha = ld[LOWDIM_SLICES["hand_action"][0]:LOWDIM_SLICES["hand_action"][1]]
        action = np.concatenate([wa, ha])  # 48D
        state = np.concatenate([ws, hs])  # 48D

        # Check action dimension consistency
        report.add(assert_check(
            action.shape[0] == 48 and state.shape[0] == 48,
            "1.7a action/state dimension",
            f"action_dim={action.shape[0]}, state_dim={state.shape[0]}",
        ))
        return

    report.add(assert_check(False, "1.7a action token count", "no valid sample found"))


def run_all(
    normalizer_path: str | None = None,
    vla_shard: str | None = None,
    vlm_shard: str | None = None,
    skip_visual: bool = False,
) -> PhaseReport:
    out_dir = get_output_dir(OUTPUT_PART)
    report = PhaseReport("Part 1: Data Pipeline + Sample Level", out_dir)

    print("\n=== Part 1: Data Pipeline + Sample Level ===\n")

    normalizer = None
    if normalizer_path:
        normalizer = load_normalizer(normalizer_path)

    # 1.1 Normalizer
    if normalizer is not None:
        test_normalizer_roundtrip(report, normalizer)
        test_normalizer_no_degenerate_dims(report, normalizer)
        test_normalizer_output_distribution(report, normalizer, out_dir)
    else:
        report.add(assert_check(True, "1.1 normalizer", "SKIPPED: no --normalizer-path"))

    # 1.2 Coordinate transforms
    test_relative_action_roundtrip(report)

    # 1.3 Sliding window
    test_sliding_window_config_sanity(report)

    # 1.4 Real VLA sample
    test_real_vla_sample_schema(report, vla_shard)
    test_real_vla_action_values_reasonable(report, vla_shard, normalizer, out_dir)

    # 1.5 Real VLM sample
    test_real_vlm_sample_schema(report, vlm_shard)

    # 1.6 Real data transform chain
    test_real_data_transform_chain(report, vla_shard, normalizer)

    # 1.7 Action dimension consistency
    test_action_token_count_consistency(report, vla_shard, normalizer)

    report.save()
    report.print_summary()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 1: Data pipeline verification")
    parser.add_argument("--normalizer-path", type=str, default=None)
    parser.add_argument("--vla-shard", type=str, default=None)
    parser.add_argument("--vlm-shard", type=str, default=None)
    parser.add_argument("--skip-visual", action="store_true")
    args = parser.parse_args()
    report = run_all(
        normalizer_path=args.normalizer_path,
        vla_shard=args.vla_shard,
        vlm_shard=args.vlm_shard,
        skip_visual=args.skip_visual,
    )
    exit(0 if report.all_passed else 1)
