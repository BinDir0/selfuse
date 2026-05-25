"""Stitch Any4D per-batch metric-scale steps out of a dense depth video.

Any4D predicts ONE metric scale ``s̃`` per inference call (see any4d.pdf: Scale MLP,
exponentiated; per-view depth is scale-normalized and consistent WITHIN a call). The
dense depth is produced in batches of ``HAWOR_ANY4D_BATCH_SIZE`` frames, each with its
own ``s̃`` that is NOT renormalized across batches, so the absolute scale jumps at batch
boundaries (frame positions B, 2B, ...). Offline analysis (scripts/analyze_alignment_methods.py)
showed this defect is PURELY a per-batch scalar: after a single median scale alignment the
cross-boundary structure already agrees (δ<1.25 ≈ 1.0), so removing one scalar per batch is
sufficient and well-posed.

This module removes the steps cheaply and WITHOUT re-running Any4D: at each batch boundary
the two straddling frames are 1 frame apart (same structure, only ``s̃`` differs), so the
optical-flow-matched, static-pixel median of ``depth[b-1] / warp(depth[b])`` is exactly the
scale ratio ``s̃_{b-1}/s̃_b``. We chain these ratios into a per-batch correction and then
re-anchor the global median so the overall metric level is unchanged (absolute scale is set
elsewhere, by the camera/hand layers).

Cost is ~one optical-flow per batch boundary (negligible next to Any4D inference).
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def stitch_enabled() -> bool:
    """Post-hoc flow-based stitch (no Any4D re-run). Gated OFF; enable HAWOR_ANY4D_STITCH=1."""
    return str(os.environ.get("HAWOR_ANY4D_STITCH", "")).strip().lower() in ("1", "true", "yes", "on")


def overlap_frames() -> int:
    """Frames of overlap between consecutive Any4D chunks (HAWOR_ANY4D_OVERLAP, default 0).

    > 0 enables the rigorous overlap-stitch: shared frames are the SAME frame predicted by
    two chunks, giving a motion-free per-chunk scale ratio. A small value (e.g. 4) suffices.
    """
    try:
        return max(0, int(os.environ.get("HAWOR_ANY4D_OVERLAP", "0")))
    except ValueError:
        return 0


def _resize(arr: np.ndarray, hw, nearest: bool = False) -> np.ndarray:
    h, w = hw
    if arr.shape[:2] == (h, w):
        return arr
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(arr, (w, h), interpolation=interp)


def _boundary_ratio(depthA, depthB, grayA, grayB, mask_bool, *, max_flow_frac=0.15, min_pixels=200):
    """Median depthA / warp(depthB) over flow-matched static valid pixels = s̃_A / s̃_B."""
    if grayA is None or grayB is None:
        return np.nan
    flow = cv2.calcOpticalFlowFarneback(grayA, grayB, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    h, w = grayA.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    mapx, mapy = xs + flow[..., 0], ys + flow[..., 1]
    dBw = cv2.remap(depthB, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderValue=np.nan)
    valid = (
        (mapx >= 0) & (mapx < w) & (mapy >= 0) & (mapy < h)
        & (np.hypot(flow[..., 0], flow[..., 1]) < max_flow_frac * w)
        & (~mask_bool)
        & np.isfinite(depthA) & (depthA > 0.05)
        & np.isfinite(dBw) & (dBw > 0.05)
    )
    if int(valid.sum()) < min_pixels:
        return np.nan
    return float(np.median(depthA[valid] / dBw[valid]))


def compute_batch_correction(
    depths: np.ndarray,
    frame_indices,
    batch_size: int,
    get_gray: Callable[[int], Optional[np.ndarray]],
    get_mask: Optional[Callable[[int], Optional[np.ndarray]]] = None,
    *,
    min_pixels: int = 200,
):
    """Per-frame multiplicative correction ``cf`` that removes per-batch scale steps.

    ``depths``        : (n, H, W) dense metric depth in prediction order.
    ``frame_indices`` : (n,) actual frame ids parallel to ``depths`` (for get_gray/get_mask).
    ``batch_size``    : Any4D inference batch size (boundaries at positions B, 2B, ...).
    ``get_gray(fid)`` : returns a grayscale image (any resolution; resized to depth HxW here).
    ``get_mask(fid)`` : optional bool/uint hand mask to exclude (resized, nearest).

    Returns (cf[n] float32, info dict). cf is piecewise-constant per batch, globally
    re-anchored so median(depths*cf) == median(depths). Fail-open: cf=1 on any problem.
    """
    info = {"applied": False, "reason": "", "n_boundaries": 0, "n_solved": 0}
    if cv2 is None:
        info["reason"] = "cv2 unavailable"
        return np.ones(depths.shape[0], np.float32), info
    n, H, W = depths.shape
    if n < batch_size + 1 or batch_size < 1:
        info["reason"] = "too few frames for any boundary"
        return np.ones(n, np.float32), info

    fids = np.asarray(frame_indices).reshape(-1)
    boundaries = list(range(batch_size, n, batch_size))  # prediction-order positions
    edges = [0] + boundaries + [n]
    nb = len(boundaries) + 1
    cf_b = np.ones(nb, np.float64)

    gray_cache: dict[int, Optional[np.ndarray]] = {}

    def gray(pos):
        fid = int(fids[pos])
        if fid not in gray_cache:
            g = get_gray(fid)
            gray_cache[fid] = _resize(g, (H, W)) if g is not None else None
        return gray_cache[fid]

    n_solved = 0
    for k, b in enumerate(boundaries):
        mask_bool = np.zeros((H, W), bool)
        if get_mask is not None:
            m = get_mask(int(fids[b - 1]))
            if m is not None:
                mask_bool = _resize(np.asarray(m).astype(np.uint8), (H, W), nearest=True) > 0
        ratio = _boundary_ratio(depths[b - 1], depths[b], gray(b - 1), gray(b), mask_bool, min_pixels=min_pixels)
        if np.isfinite(ratio) and ratio > 0:
            cf_b[k + 1] = cf_b[k] * ratio
            n_solved += 1
        else:
            cf_b[k + 1] = cf_b[k]  # carry forward on a failed boundary

    cf = np.ones(n, np.float64)
    for k in range(nb):
        cf[edges[k]:edges[k + 1]] = cf_b[k]

    # Re-anchor: hold the global median fixed (this layer removes STEPS, not absolute level).
    # Use per-frame medians (memory-safe: the full depth array can be GBs; never materialize
    # depths*cf just to take a median).
    per_frame_med = np.array([np.nanmedian(depths[t]) for t in range(n)], np.float64)
    finite = np.isfinite(per_frame_med)
    if finite.any():
        g_raw = float(np.median(per_frame_med[finite]))
        g_cor = float(np.median((per_frame_med * cf)[finite]))
        if np.isfinite(g_raw) and np.isfinite(g_cor) and g_cor > 0:
            cf *= g_raw / g_cor

    info.update(applied=True, n_boundaries=len(boundaries), n_solved=n_solved,
                cf_min=float(cf.min()), cf_max=float(cf.max()))
    return cf.astype(np.float32), info


def assemble_overlapping_chunks(chunks, n_frames, hw, *, min_pixels=500):
    """Stitch + assemble Any4D chunks that were run with frame OVERLAP.

    ``chunks`` : list of (start_pos, depths[m,H,W]) — m frames per chunk, possibly
                 overlapping the next chunk; depths already at output resolution.
    On the overlap frames the two chunks predict the SAME frame, so the per-pixel
    median ratio depth_a/depth_b is exactly s̃_a/s̃_b (zero motion, no dynamic confound).
    Chain those ratios into a per-chunk correction, re-anchor the global median, then
    assemble one depth per frame from the chunk whose center is closest (least edge bias).

    Returns (dense[n,H,W] float32, cf_per_frame[n] float32, info).
    """
    H, W = hw
    info = {"applied": False, "reason": "", "n_chunks": len(chunks), "n_solved": 0}
    chunks = sorted(chunks, key=lambda c: int(c[0]))
    K = len(chunks)
    if K == 0:
        info["reason"] = "no chunks"
        return np.full((n_frames, H, W), np.nan, np.float32), np.ones(n_frames, np.float32), info
    cf_chunk = np.ones(K, np.float64)
    n_solved = 0
    for k in range(1, K):
        a_start, a_d = int(chunks[k - 1][0]), chunks[k - 1][1]
        b_start, b_d = int(chunks[k][0]), chunks[k][1]
        ov_lo, ov_hi = b_start, min(a_start + a_d.shape[0], b_start + b_d.shape[0])
        ratios = []
        for p in range(ov_lo, ov_hi):
            da, db = a_d[p - a_start], b_d[p - b_start]
            v = np.isfinite(da) & (da > 0.05) & np.isfinite(db) & (db > 0.05)
            if int(v.sum()) >= min_pixels:
                ratios.append(float(np.median(da[v] / db[v])))
        if ratios:
            cf_chunk[k] = cf_chunk[k - 1] * float(np.median(ratios))
            n_solved += 1
        else:
            cf_chunk[k] = cf_chunk[k - 1]  # no usable overlap -> carry forward

    centers = [int(c[0]) + c[1].shape[0] / 2.0 for c in chunks]
    out = np.full((n_frames, H, W), np.nan, np.float32)
    owner = np.full(n_frames, -1, np.int32)
    for k, (s, d) in enumerate(chunks):
        s = int(s)
        for off in range(d.shape[0]):
            p = s + off
            if p >= n_frames:
                break
            if owner[p] < 0 or abs(p - centers[k]) < abs(p - centers[owner[p]]):
                out[p] = d[off]
                owner[p] = k
    cf_per_frame = np.where(owner >= 0, cf_chunk[np.clip(owner, 0, K - 1)], 1.0).astype(np.float64)

    # re-anchor global median (memory-safe per-frame medians); this removes STEPS, not level
    pfm = np.array([np.nanmedian(out[t]) for t in range(n_frames)], np.float64)
    fin = np.isfinite(pfm)
    if fin.any():
        g_raw = float(np.median(pfm[fin]))
        g_cor = float(np.median((pfm * cf_per_frame)[fin]))
        if np.isfinite(g_raw) and np.isfinite(g_cor) and g_cor > 0:
            cf_per_frame *= g_raw / g_cor

    out *= cf_per_frame[:, None, None].astype(out.dtype)
    info.update(applied=True, n_solved=n_solved, cf_min=float(cf_per_frame.min()), cf_max=float(cf_per_frame.max()))
    return out, cf_per_frame.astype(np.float32), info


def stitch_dense_depth(depths, frame_indices, batch_size, get_gray, get_mask=None, *, min_pixels=200, in_place=True):
    """Convenience wrapper: returns (stitched_depths, cf, info).

    Applies the correction IN PLACE by default (the dense depth array can be GBs; avoid a
    full duplicate). Pass in_place=False to keep the input untouched.
    """
    cf, info = compute_batch_correction(depths, frame_indices, batch_size, get_gray, get_mask, min_pixels=min_pixels)
    if not info.get("applied"):
        return depths, cf, info
    out = depths if in_place else depths.copy()
    out *= cf[:, None, None].astype(out.dtype)
    return out, cf, info
