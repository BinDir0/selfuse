"""Anchor the metric world to the HaWoR hand (trusted) rather than Any4D depth.

Decision (validated on production with the CORRECT focal): Any4D's absolute scale is
NOT trustworthy in egocentric close-range scenes, whereas HaWoR's camera-frame hand
depth (hand-size prior + correct intrinsics) is. After fixing the focal, the hand vs
Any4D-at-hand disagreement is a near-constant GLOBAL factor (~3.8x), not per-frame.

So we re-anchor the whole metric world to the hand with ONE factor

    k = median_t( hand_cam_z(t) / Any4D_depth_at_hand(t) )

and multiply the dense depth by k. Applied BEFORE the SLAM scale estimation, so the
camera scale (fit by est_scale to match this depth) auto-inherits the hand metric —
no separate camera-scale correction needed. The hand itself is unchanged (it is the
anchor). Assumes Any4D's scale bias is ~uniform (its relative/temporal structure is
already validated by the overlap stitch); if non-uniform, far-scene depth is only
approximate, which is acceptable for a hand-centric dataset.

Gated OFF by default; enable with HAWOR_HAND_ANCHOR=1.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Callable, Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def hand_anchor_enabled() -> bool:
    return str(os.environ.get("HAWOR_HAND_ANCHOR", "")).strip().lower() in ("1", "true", "yes", "on")


def _load_hand_cam_z(seq_folder: str, n_frames: int) -> np.ndarray:
    """Per-frame camera-frame wrist z (meters) from cam_space chunk JSONs (init_trans[...,2]),
    combined over hands. NaN where no hand. Indexed by absolute frame id."""
    z = np.full((n_frames,), np.nan, np.float32)
    per_hand = []
    for hand_dir in sorted(glob.glob(os.path.join(seq_folder, "cam_space", "*"))):
        if not os.path.isdir(hand_dir):
            continue
        zi = np.full((n_frames,), np.nan, np.float32)
        for jf in sorted(glob.glob(os.path.join(hand_dir, "*.json"))):
            base = os.path.splitext(os.path.basename(jf))[0]
            try:
                s, _e = (int(x) for x in base.split("_"))
            except ValueError:
                continue
            try:
                trans = np.asarray(json.load(open(jf))["init_trans"], np.float32)  # (1, T, 3)
            except Exception:
                continue
            tz = trans[0, :, 2]
            idx = np.arange(s, s + tz.shape[0])
            ok = (idx >= 0) & (idx < n_frames)
            zi[idx[ok]] = tz[ok]
        per_hand.append(zi)
    if per_hand:
        with np.errstate(invalid="ignore"):
            z = np.nanmedian(np.vstack(per_hand), axis=0).astype(np.float32)
    return z


def _resize_mask(mask: np.ndarray, hw) -> np.ndarray:
    h, w = hw
    m = np.asarray(mask)
    if m.dtype == bool:
        m = m.astype(np.uint8)
    if m.shape[:2] == (h, w):
        return m > 0
    if cv2 is not None:
        return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    ys = np.linspace(0, m.shape[0] - 1, h).astype(int)
    xs = np.linspace(0, m.shape[1] - 1, w).astype(int)
    return m[np.ix_(ys, xs)] > 0


def compute_hand_anchor_k(
    depths: np.ndarray,
    frame_indices,
    seq_folder: str,
    get_mask: Optional[Callable[[int], Optional[np.ndarray]]] = None,
    *,
    depth_min: float = 0.05,
    depth_max: float = 10.0,
    min_mask_pixels: int = 50,
    min_frames: int = 8,
    k_lo: float = 1e-2,
    k_hi: float = 1e2,
):
    """Global hand-anchor factor k so that depth*k matches the HaWoR hand metric.

    k = median over hand-present frames of ( hand_cam_z / median(Any4D depth at hand mask) ).
    Returns (k float, info). Fail-open: k=1.0 with reason on any problem.
    """
    info = {"applied": False, "reason": "", "n_frames_used": 0}
    n, H, W = depths.shape
    fids = np.asarray(frame_indices).reshape(-1)
    hand_z = _load_hand_cam_z(seq_folder, int(fids.max()) + 1 if fids.size else n)
    if not np.isfinite(hand_z).any():
        info["reason"] = "no hand cam_space"
        return 1.0, info
    if get_mask is None:
        info["reason"] = "no mask accessor"
        return 1.0, info

    ratios = []
    for i in range(n):
        fid = int(fids[i])
        hz = hand_z[fid] if 0 <= fid < hand_z.shape[0] else np.nan
        if not np.isfinite(hz) or hz <= 0:
            continue
        m = get_mask(fid)
        if m is None:
            continue
        mk = _resize_mask(m, (H, W))
        d = depths[i]
        sel = mk & np.isfinite(d) & (d > depth_min) & (d < depth_max)
        if int(sel.sum()) < min_mask_pixels:
            continue
        d_hand = float(np.median(d[sel]))
        if d_hand > 0:
            ratios.append(hz / d_hand)

    if len(ratios) < min_frames:
        info["reason"] = f"insufficient hand frames ({len(ratios)}<{min_frames})"
        return 1.0, info
    k = float(np.clip(np.median(ratios), k_lo, k_hi))
    info.update(applied=True, n_frames_used=len(ratios), k=k,
                offset=float(1.0 / k) if k > 0 else float("nan"))
    return k, info
