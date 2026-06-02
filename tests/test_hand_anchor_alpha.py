"""Unit tests for the hand-anchor depth->hand alignment (lib/pipeline/hand_metric_anchor.py).

Pure-numpy (no torch / GPU). We synthesize a clip where, at the hand, the HaWoR camera-frame z is a
KNOWN smooth per-frame multiple of the Any4D depth, then check:
  * compute_hand_anchor_k recovers the global ratio k,
  * compute_hand_anchor_alpha recovers the per-frame curve alpha(t) (alpha(t)*depth_at_hand ~= hand_z),
  * alpha/k is mean-preserving (median ~= 1) so the depth-map refinement keeps the global level,
  * both fail open (no cam_space -> k=1, alpha=ones).

Runnable two ways:
    pytest tests/test_hand_anchor_alpha.py
    python3 tests/test_hand_anchor_alpha.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.pipeline.hand_metric_anchor import (  # noqa: E402
    compute_hand_anchor_alpha,
    compute_hand_anchor_k,
)

N, H, W = 40, 16, 16


def _build_clip(tmp: str):
    """Whole-frame depth = depth_at_hand(t); hand cam-z = ratio(t)*depth_at_hand(t) (smooth ramp)."""
    t = np.arange(N)
    depth_at_hand = (0.4 + 0.1 * np.sin(t * 0.2)).astype(np.float64)   # 0.3..0.5 m, in (min,max)
    ratio = (3.0 + 1.0 * (t / (N - 1))).astype(np.float64)            # smooth ramp 3.0 -> 4.0
    hand_z = ratio * depth_at_hand

    depths = np.empty((N, H, W), np.float32)
    for i in range(N):
        depths[i] = depth_at_hand[i]

    # cam_space/0/{s}_{e}.json with init_trans shape (1, N, 3); z column = hand_z
    hand_dir = os.path.join(tmp, "cam_space", "0")
    os.makedirs(hand_dir, exist_ok=True)
    init_trans = np.zeros((1, N, 3), np.float64)
    init_trans[0, :, 2] = hand_z
    with open(os.path.join(hand_dir, f"0_{N - 1}.json"), "w", encoding="utf-8") as fh:
        json.dump({"init_trans": init_trans.tolist()}, fh)

    mask = np.zeros((H, W), bool)
    mask[4:12, 4:12] = True  # 64 px >= min_mask_pixels (50)

    def get_mask(_fid, _m=mask):
        return _m

    return depths, np.arange(N, dtype=np.int64), get_mask, depth_at_hand, hand_z, ratio


def test_global_k_recovered():
    with tempfile.TemporaryDirectory() as tmp:
        depths, fids, get_mask, depth_at_hand, hand_z, ratio = _build_clip(tmp)
        k, info = compute_hand_anchor_k(depths, fids, tmp, get_mask)
        assert info["applied"] and info["n_frames_used"] == N
        assert abs(k - float(np.median(ratio))) / np.median(ratio) < 0.02, (k, np.median(ratio))


def test_alpha_tracks_per_frame_ratio():
    with tempfile.TemporaryDirectory() as tmp:
        depths, fids, get_mask, depth_at_hand, hand_z, ratio = _build_clip(tmp)
        alpha, k, info = compute_hand_anchor_alpha(depths, fids, tmp, get_mask)
        assert info["applied"] and info["smooth_applied"]
        assert alpha.shape == (N,)
        # alpha(t)*depth_at_hand(t) ~= hand_z(t) per frame (smooth ramp is ~exactly representable)
        recon = alpha.astype(np.float64) * depth_at_hand
        rel = np.abs(recon - hand_z) / hand_z
        assert float(np.median(rel)) < 0.02, float(np.median(rel))
        assert float(np.max(rel)) < 0.05, float(np.max(rel))
        # k is the global level; alpha/k is a mean-preserving per-frame refinement
        assert abs(k - float(np.median(ratio))) / np.median(ratio) < 0.05, (k, np.median(ratio))
        assert abs(float(np.median(alpha.astype(np.float64) / k)) - 1.0) < 0.02


def test_fail_open_without_cam_space():
    with tempfile.TemporaryDirectory() as tmp:
        depths = np.full((N, H, W), 0.4, np.float32)
        fids = np.arange(N, dtype=np.int64)

        def get_mask(_fid):
            m = np.zeros((H, W), bool)
            m[4:12, 4:12] = True
            return m

        k, info_k = compute_hand_anchor_k(depths, fids, tmp, get_mask)
        alpha, k2, info_a = compute_hand_anchor_alpha(depths, fids, tmp, get_mask)
        assert k == 1.0 and not info_k["applied"]
        assert k2 == 1.0 and not info_a["applied"] and np.allclose(alpha, 1.0)


def _main() -> int:
    failures = 0
    for name in ("test_global_k_recovered", "test_alpha_tracks_per_frame_ratio",
                 "test_fail_open_without_cam_space"):
        try:
            globals()[name]()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
