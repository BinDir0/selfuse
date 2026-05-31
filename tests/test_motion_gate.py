"""Unit tests for the camera-stability motion gate.

These exercise ``compute_motion_signals`` on synthetic numpy frames only -- no
video files, no GPU, no inference. The gate decides on CAMERA STABILITY only
(``passed == stable_camera``); hand presence is handled separately by the
detector gate (gate_a). The signal comes from a RANSAC global-motion fit, with a
smart fallback that previously had zero coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from lib.clip.heuristic_video_clipper import (  # noqa: E402
    _roi_bounds,
    compute_motion_signals,
    load_clip_config,
)

H, W = 256, 448
ROI = _roi_bounds(W, H, [0.15, 0.20, 0.85, 1.00])

# Generous defaults so the geometry, not the thresholds, drives each assertion.
GATE_B = {
    "flow_max_corners": 400,
    "flow_quality_level": 0.01,
    "flow_min_distance": 5,
    "flow_block_size": 7,
    "flow_min_tracked": 24,
    "camera_motion_thresh": 0.2,
    "ransac_reproj_thresh": 3.0,
    "min_inlier_ratio": 0.30,
}
# gate_c no longer feeds the motion gate, but the signature still accepts it.
GATE_C: dict = {}


def _textured(seed: int, h: int = H, w: int = W) -> np.ndarray:
    """A deterministic, corner-rich grayscale frame that LK can track."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(h // 8, w // 8), dtype=np.uint8)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    # A scatter of bright squares gives strong, unambiguous Shi-Tomasi corners.
    for _ in range(60):
        y = int(rng.integers(0, h - 6))
        x = int(rng.integers(0, w - 6))
        img[y:y + 6, x:x + 6] = 255 if rng.random() > 0.5 else 0
    return img


def _translate(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def test_small_camera_translation_is_stable_and_passes():
    base = _textured(0)
    prev, cur = base, _translate(base, 8, 0)
    sig = compute_motion_signals(prev, cur, gate_b=GATE_B, gate_c=GATE_C, roi_px=ROI)

    assert sig.have_flow
    assert sig.inlier_ratio > 0.6  # a single global motion explains the whole frame
    assert 5.0 < sig.camera_motion_px < 11.0  # recovers the ~8px shift
    assert sig.stable_camera is True
    assert sig.passed is True


def test_camera_threshold_flips_when_tightened():
    base = _textured(1)
    prev, cur = base, _translate(base, 8, 0)
    tight = {**GATE_B, "camera_motion_thresh": 0.01}  # ~4.5px ceiling < 8px motion
    sig = compute_motion_signals(prev, cur, gate_b=tight, gate_c=GATE_C, roi_px=ROI)

    assert sig.have_flow
    assert sig.stable_camera is False
    assert sig.passed is False


def test_static_camera_with_moving_roi_block_still_passes():
    # A moving foreground block does NOT fail the gate: the background pins the
    # global model near identity, so the camera reads as stable. (Hand activity
    # is intentionally not part of this gate.)
    bg = _textured(2)
    block = _textured(99, h=200, w=260)
    bx, by = 90, 56

    prev = bg.copy()
    prev[by:by + 200, bx:bx + 260] = block
    cur = bg.copy()
    cur[by:by + 200, bx + 10:bx + 10 + 260] = block  # block shifts 10px, bg static

    sig = compute_motion_signals(prev, cur, gate_b=GATE_B, gate_c=GATE_C, roi_px=ROI)

    assert sig.have_flow
    assert sig.camera_motion_px < 2.0  # background pins the global model near identity
    assert sig.stable_camera is True
    assert sig.passed is True


def test_flat_texture_is_assumed_stable():
    # No trackable corners -> fallback branch assumes the camera is stable and
    # passes (hand presence is gate_a's job, not the flow gate's).
    prev = np.full((H, W), 100, np.uint8)
    cur = np.full((H, W), 160, np.uint8)
    sig = compute_motion_signals(prev, cur, gate_b=GATE_B, gate_c=GATE_C, roi_px=ROI)

    assert sig.have_flow is False
    assert sig.stable_camera is True
    assert sig.passed is True
    assert sig.diff_score > 0.0  # reporting-only activity score still computed


def test_incoherent_scene_is_rejected_as_unstable():
    # Decorrelated frames: no global motion consensus -> camera judged unstable.
    # This is the fix for the old fail-open bug (fast motion / blur slipping through).
    prev, cur = _textured(10), _textured(20)
    sig = compute_motion_signals(prev, cur, gate_b=GATE_B, gate_c=GATE_C, roi_px=ROI)

    assert sig.stable_camera is False
    assert sig.passed is False


def test_first_frame_has_no_signal():
    cur = _textured(5)
    sig = compute_motion_signals(None, cur, gate_b=GATE_B, gate_c=GATE_C, roi_px=ROI)
    assert sig.have_flow is False
    assert sig.passed is False


def test_shipped_config_keys_parse():
    cfg = load_clip_config()
    gate_b = cfg["heuristic"]["gate_b"]
    for key in ("camera_motion_thresh", "ransac_reproj_thresh", "min_inlier_ratio"):
        assert key in gate_b
    # The hand-motion keys are gone now that the gate is camera-only.
    gate_c = cfg["heuristic"]["gate_c"]
    assert "hand_residual_thresh" not in gate_c
    assert "hand_min_outlier_ratio" not in gate_c
    assert "hand_motion_thresh" not in gate_c
