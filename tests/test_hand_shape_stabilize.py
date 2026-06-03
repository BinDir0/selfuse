"""Unit tests for per-clip hand-shape stabilization (lib/pipeline/hand_shape_stabilize.py).

numpy-only (real MANO from _DATA, no torch). Verifies the core guarantee the design rests on:
constant per-clip betas + per-frame trans×f preserves the 2D projection for size-driven beta wobble
(the depth-ambiguous dimension), and the output is continuous (one constant shape, no per-frame
switching). Simulates the monocular size-depth ambiguity: per frame HaWoR could pick a different
size (beta[0]) with a compensating depth, all projecting to the same observed 2D.

    pytest tests/test_hand_shape_stabilize.py
    python3 tests/test_hand_shape_stabilize.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.pipeline.hand_shape_stabilize import (  # noqa: E402
    load_default_mano,
    shape_size,
    stabilize_betas_trans,
)
from lib.utils.mano_numpy import mano_forward  # noqa: E402

_MANO = load_default_mano()
_K = np.array([[500.0, 0.0, 128.0], [0.0, 500.0, 128.0], [0.0, 0.0, 1.0]], np.float64)
_ORIENT = np.zeros(3)
_POSE = np.zeros((15, 3))
_TRUE_TRANS = np.array([0.05, 0.02, 0.6])  # hand ~0.6 m in front, z>0


def _project(betas, trans):
    _, joints = mano_forward(_MANO, betas, _ORIENT, _POSE, trans, return_joints=True)
    uv = joints @ _K.T
    return uv[:, :2] / uv[:, 2:3]  # (16,2) px


def _make_size_ambiguous_clip(T=12, beta0_amp=0.6):
    """Per-frame betas differ only in the global-size PC (beta[0]); trans set (size-depth
    ambiguity) so every frame projects to ~the same observed 2D as the true hand."""
    b_true = np.zeros(10)
    s_true = shape_size(_MANO, b_true)
    wob = beta0_amp * np.sin(np.linspace(0, 2 * np.pi, T))
    betas = np.tile(b_true, (T, 1))
    betas[:, 0] = wob
    trans = np.empty((T, 3))
    for t in range(T):
        trans[t] = _TRUE_TRANS * (shape_size(_MANO, betas[t]) / s_true)  # depth compensates size
    observed_2d = _project(b_true, _TRUE_TRANS)
    return betas, trans, observed_2d


def test_output_contract():
    betas, trans, _ = _make_size_ambiguous_clip()
    b_out, t_out, info = stabilize_betas_trans(betas, trans, _MANO)
    assert info["applied"]
    # one constant shape (== median), broadcast to every frame -> no per-frame switching
    med = np.median(betas, axis=0)
    assert np.allclose(b_out, med[None, :]), "betas_out must be the constant median"
    assert np.allclose(b_out[0], b_out[-1]), "shape must be identical across frames (continuity)"
    # trans scaled by a per-frame scalar f = size(b*)/size(beta_t)
    f = t_out / trans
    assert np.allclose(f, f[:, :1], atol=1e-9), "trans scaled by a single scalar per frame"
    for t in range(betas.shape[0]):
        f_expect = shape_size(_MANO, med) / shape_size(_MANO, betas[t])
        assert abs(f[t, 0] - f_expect) < 1e-6


def test_size_wobble_2d_preserved():
    betas, trans, observed_2d = _make_size_ambiguous_clip()
    b_out, t_out, _ = stabilize_betas_trans(betas, trans, _MANO)
    stab_err, orig_err = [], []
    for t in range(betas.shape[0]):
        stab_err.append(np.abs(_project(b_out[t], t_out[t]) - observed_2d).max())
        orig_err.append(np.abs(_project(betas[t], trans[t]) - observed_2d).max())
    stab_err, orig_err = np.array(stab_err), np.array(orig_err)
    # stabilized fits the observed 2D tightly (size wobble is depth-compensated)
    assert stab_err.max() < 1.0, f"stabilized reproj too large: {stab_err.max():.3f}px"
    # and is no worse than the original per-frame fit (the hard requirement)
    assert stab_err.max() <= orig_err.max() + 0.5, (stab_err.max(), orig_err.max())


def test_proportion_wobble_bounded_and_smooth():
    # wobble a PROPORTION PC (beta[3]) instead of size: residual is larger but bounded + smooth
    T = 12
    betas = np.zeros((T, 10))
    betas[:, 3] = 0.5 * np.sin(np.linspace(0, 2 * np.pi, T))
    trans = np.tile(_TRUE_TRANS, (T, 1))
    b_out, t_out, info = stabilize_betas_trans(betas, trans, _MANO)
    assert info["applied"] and np.allclose(b_out, np.median(betas, axis=0)[None, :])
    dev = np.array([np.abs(_project(b_out[t], t_out[t]) - _project(betas[t], trans[t])).max()
                    for t in range(T)])
    assert dev.max() < 12.0, f"proportion residual unexpectedly large: {dev.max():.2f}px"
    # smoothness: frame-to-frame change in the deviation has no jumps (continuous output)
    assert np.abs(np.diff(dev)).max() < 6.0


def test_fail_open_empty():
    b, t, info = stabilize_betas_trans(np.zeros((0, 10)), np.zeros((0, 3)), _MANO)
    assert not info["applied"] and b.shape[0] == 0


def _main() -> int:
    failures = 0
    for name in ("test_output_contract", "test_size_wobble_2d_preserved",
                 "test_proportion_wobble_bounded_and_smooth", "test_fail_open_empty"):
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
