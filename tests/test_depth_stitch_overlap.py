"""Unit tests for the Any4D overlap-stitch (lib/pipeline/depth_stitch.py).

These are pure-numpy (no torch / GPU): we synthesize a globally-consistent geometry ``G(f)``,
split it into overlapping chunks, inject KNOWN per-chunk scalars, and assert the stitch recovers a
single global scale (steps removed). A second test injects a NON-scalar (spatially varying)
corruption on one boundary and asserts the flatness gate flags that link instead of chaining it.

Runnable two ways:
    pytest tests/test_depth_stitch_overlap.py
    python3 tests/test_depth_stitch_overlap.py      # standalone driver (no pytest needed)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.pipeline.depth_stitch import assemble_overlapping_chunks  # noqa: E402


def _make_geometry(n_frames: int, h: int, w: int) -> np.ndarray:
    """Smooth, strictly-positive depth field; per-frame median varies so re-anchoring is real."""
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    spatial = 0.8 + 0.6 * xs + 0.4 * ys  # in-frame structure (must NOT cancel as a scalar)
    g = np.empty((n_frames, h, w), np.float32)
    for f in range(n_frames):
        base = 1.0 + 0.5 * np.sin(f * 0.13)  # per-frame level
        g[f] = (base * spatial).astype(np.float32)
    return g


def _overlap_chunks(geometry: np.ndarray, batch: int, overlap: int):
    """Replicate build_any4d_chunk_specs layout (without importing torch) as (start, depths)."""
    n = geometry.shape[0]
    stride = batch - overlap
    chunks, starts = [], []
    for start in range(0, n, stride):
        end = min(start + batch, n)
        chunks.append((start, geometry[start:end].copy()))
        starts.append(start)
        if start + batch >= n:
            break
    return chunks, starts


def test_recovers_known_per_chunk_scalars():
    n, h, w = 100, 32, 32
    batch, overlap = 32, 4
    g = _make_geometry(n, h, w)
    chunks, _ = _overlap_chunks(g, batch, overlap)
    assert len(chunks) == 4  # [0,32) [28,60) [56,88) [84,100); 3 boundaries

    scalars = [1.0, 1.3, 0.8, 1.15]
    injected = [(s, (sc * d).astype(np.float32)) for (s, d), sc in zip(chunks, scalars)]

    out, cf, info = assemble_overlapping_chunks(injected, n, (h, w))

    assert info["applied"] and info["n_solved"] == 3 and info["n_flagged"] == 0
    assert info["boundary_trusted"].all()

    # measured links are s_{k-1}/s_k
    expected = np.array([scalars[0] / scalars[1], scalars[1] / scalars[2], scalars[2] / scalars[3]])
    assert np.allclose(info["boundary_ratio"], expected, rtol=1e-3), info["boundary_ratio"]

    # step-free: per-frame (median out / median G) must be one constant across the whole clip
    ratio_t = np.array([np.nanmedian(out[t]) / np.nanmedian(g[t]) for t in range(n)])
    assert np.isfinite(ratio_t).all()
    assert (ratio_t.max() / ratio_t.min() - 1.0) < 1e-3, (ratio_t.min(), ratio_t.max())


def test_flags_non_scalar_boundary():
    n, h, w = 100, 32, 32
    batch, overlap = 32, 4
    g = _make_geometry(n, h, w)
    chunks, starts = _overlap_chunks(g, batch, overlap)
    scalars = [1.0, 1.3, 0.8, 1.15]
    injected = [(s, (sc * d).astype(np.float32)) for (s, d), sc in zip(chunks, scalars)]

    # Corrupt chunk1's first `overlap` frames (shared with chunk0) with a strong width gradient,
    # so the shared-frame ratio is NOT a flat field (structure mismatch, not a pure scalar).
    grad = np.linspace(0.4, 2.5, w, dtype=np.float32)[None, :]  # varies across width
    c1_start, c1_depth = injected[1]
    c1_depth[:overlap] *= grad
    injected[1] = (c1_start, c1_depth)

    _out, _cf, info = assemble_overlapping_chunks(injected, n, (h, w))

    # boundary index 0 == chunk0<->chunk1 link
    assert info["boundary_trusted"][0] == False  # noqa: E712
    assert info["n_flagged"] >= 1
    assert info["boundary_flatness"][0] > info["max_mad"]
    # the other two boundaries are still clean
    assert bool(info["boundary_trusted"][1]) and bool(info["boundary_trusted"][2])


def test_no_chunks_fail_open():
    out, cf, info = assemble_overlapping_chunks([], 10, (8, 8))
    assert not info["applied"] and cf.shape == (10,) and np.allclose(cf, 1.0)


def _main() -> int:
    failures = 0
    for name in ("test_recovers_known_per_chunk_scalars", "test_flags_non_scalar_boundary",
                 "test_no_chunks_fail_open"):
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
