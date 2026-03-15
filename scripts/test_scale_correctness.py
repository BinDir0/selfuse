#!/usr/bin/env python3
"""Verify that est_scale_hybrid_batch produces identical results to per-keyframe est_scale_hybrid.

Usage:
    python scripts/test_scale_correctness.py

Generates synthetic depth data mimicking real SLAM keyframes, then compares
batch vs per-keyframe scale estimation. Prints per-keyframe differences and
reports PASS/FAIL based on max absolute difference.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import time
from lib.pipeline.est_scale import est_scale_hybrid, est_scale_hybrid_batch

TOLERANCE = 1e-4
NUM_KEYFRAMES = 30  # typical number of SLAM keyframes
H, W = 480, 640


def make_synthetic_data(n_keyframes, seed=42):
    """Generate synthetic slam_depth, pred_depth, and masks for testing."""
    rng = np.random.RandomState(seed)

    slam_depths = []
    pred_depths = []
    masks = []

    for i in range(n_keyframes):
        # Random true scale for this keyframe
        true_scale = rng.uniform(0.5, 2.0)

        # Synthetic slam depth (random scene geometry)
        slam_depth = rng.uniform(0.1, 5.0, size=(H, W)).astype(np.float32)

        # pred_depth = slam_depth * true_scale + noise
        noise = rng.normal(0, 0.05, size=(H, W)).astype(np.float32)
        pred_depth = slam_depth * true_scale + noise
        pred_depth = np.clip(pred_depth, 0.01, 10.0)

        # Random mask (small masked region)
        mask = np.zeros((H, W), dtype=np.uint8)
        r, c = rng.randint(0, H-50), rng.randint(0, W-50)
        mask[r:r+50, c:c+50] = 255

        slam_depths.append(slam_depth)
        pred_depths.append(pred_depth)
        masks.append(mask)

    return slam_depths, pred_depths, masks


def main():
    print(f"Testing scale estimation correctness: {NUM_KEYFRAMES} keyframes, {H}x{W}")
    print("=" * 70)

    slam_depths, pred_depths, masks = make_synthetic_data(NUM_KEYFRAMES)

    # --- Per-keyframe (original) ---
    t0 = time.time()
    scales_original = []
    for i in range(NUM_KEYFRAMES):
        scale = est_scale_hybrid(slam_depths[i], pred_depths[i], sigma=0.5,
                                 msk=masks[i], near_thresh=0.4, far_thresh=0.7)
        scales_original.append(scale)
    time_original = time.time() - t0

    # --- Batch ---
    t0 = time.time()
    scales_batch = est_scale_hybrid_batch(slam_depths, pred_depths, sigma=0.5,
                                          masks=masks, near_thresh=0.4, far_thresh=0.7)
    time_batch = time.time() - t0

    # --- Compare ---
    print(f"\n{'KF':>4s}  {'Original':>12s}  {'Batch':>12s}  {'Diff':>12s}  {'Status':>8s}")
    print("-" * 60)

    max_diff = 0
    nan_mismatch = 0
    for i in range(NUM_KEYFRAMES):
        o = scales_original[i]
        b = scales_batch[i]

        if np.isnan(o) and np.isnan(b):
            diff_str = "NaN==NaN"
            status = "OK"
            diff = 0
        elif np.isnan(o) != np.isnan(b):
            diff_str = "NaN MISMATCH"
            status = "FAIL"
            nan_mismatch += 1
            diff = float('inf')
        else:
            diff = abs(o - b)
            diff_str = f"{diff:.2e}"
            status = "OK" if diff < TOLERANCE else "FAIL"

        max_diff = max(max_diff, diff) if not np.isinf(diff) else max_diff
        print(f"{i:4d}  {o:12.6f}  {b:12.6f}  {diff_str:>12s}  {status:>8s}")

    print("=" * 70)
    print(f"Max difference:   {max_diff:.2e}")
    print(f"NaN mismatches:   {nan_mismatch}")
    print(f"Time (original):  {time_original:.3f}s")
    print(f"Time (batch):     {time_batch:.3f}s")
    print(f"Speedup:          {time_original/time_batch:.2f}x")
    print()

    if max_diff < TOLERANCE and nan_mismatch == 0:
        print("RESULT: PASS - batch estimation matches per-keyframe estimation")
        return 0
    else:
        print("RESULT: FAIL - batch estimation differs from per-keyframe estimation!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
