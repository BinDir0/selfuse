#!/usr/bin/env python3
"""Attribute metric-scale drift in one clip to its three independent sources.

The HaWoR pipeline carries three separate, individually-drifting metric signals
that were never unified into one consistent scale field:

  (1) Any4D batch steps   — depth is predicted in batches of HAWOR_METRIC3D_BATCH_SIZE
      (default 48); each batch has its own metric scale_token, so the absolute scale
      can JUMP at batch boundaries (frame positions 48, 96, ...).
  (2) DPVO time-varying scale — the monocular camera scale is fit per-keyframe
      (median Any4D-depth / SLAM-depth on background) but collapsed to ONE global
      scalar (slam.py _estimate_scale). A slow trend in that per-keyframe curve is
      drift that a single scalar cannot remove.
  (3) Hand residual       — the hand's camera-frame depth comes from HaWoR's
      hand-size prior, independent of the scene; r(t)=d_map/z_hawor (from the
      hand_depth_align sidecar) reveals a systematic offset / trend.

This produces a single per-clip report quantifying each source so we can decide
where to invest (Phase 1 stitching vs Phase 2 time-varying scale vs Phase 3 hand).

Run on the production machine (after an export), clips spanning >2s (>=2 batches):
  python tools/diagnostics/diagnose_scale_attribution.py --seq_folder /path/to/outputs/<clip_id>
"""
import argparse
import glob
import os

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

# Reuse the batch-step machinery so attribution stays consistent across scripts.
from diagnose_depth_batch_steps import (  # type: ignore
    _batch_size_default,
    analyze,
    find_depth_npz,
    load_depth_npz,
    load_masks,
    per_frame_stat,
)


def _find_slam_npz(seq_folder):
    cand = sorted(glob.glob(os.path.join(seq_folder, "SLAM", "hawor_slam_w_scale_*.npz")))
    if not cand:
        raise FileNotFoundError(f"no hawor_slam_w_scale_*.npz under {seq_folder}/SLAM")
    return cand[-1]


def _resize_to(arr, hw):
    h, w = hw
    if arr.shape == (h, w):
        return arr
    if cv2 is not None:
        return cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    ys = np.linspace(0, arr.shape[0] - 1, h).astype(int)
    xs = np.linspace(0, arr.shape[1] - 1, w).astype(int)
    return arr[np.ix_(ys, xs)]


def per_keyframe_scale(disps, tstamp, depths, frame_indices, masks,
                       near=0.4, far=0.7):
    """Per-keyframe scale = median(pred_depth / slam_depth) on background.

    Mirrors stage-1 of est_scale_hybrid (iterative median); enough to expose the
    per-keyframe trend WITHOUT the median-collapse that slam._estimate_scale does.
    """
    idx_by_frame = {int(f): i for i, f in enumerate(frame_indices)}
    s = np.full(len(tstamp), np.nan, np.float32)
    for i, ts in enumerate(tstamp):
        ts = int(ts)
        if ts not in idx_by_frame:
            continue
        pred = np.asarray(depths[idx_by_frame[ts]], np.float32)
        slam = 1.0 / np.maximum(_resize_to(np.asarray(disps[i], np.float32), pred.shape), 1e-6)
        bg = np.ones(pred.shape, bool)
        if masks is not None and ts < masks.shape[0]:
            mk = _resize_to(masks[ts].astype(np.uint8), pred.shape)
            bg = mk <= 0
        ratio = pred / np.maximum(slam, 1e-6)
        scale = np.nan
        for _ in range(10):
            sd = slam * (scale if np.isfinite(scale) else 1.0)
            robust = bg & (pred > near) & (pred < far)
            if np.isfinite(scale):
                robust = robust & (sd > 0) & (sd < far)
            vals = ratio[robust]
            if vals.size == 0:
                break
            scale = float(np.median(vals))
        s[i] = scale
    return s


def _trend(curve):
    """Relative drift (linear-fit endpoints span / mean) and coefficient of variation."""
    v = np.asarray(curve, np.float64)
    m = np.isfinite(v)
    if m.sum() < 3:
        return dict(rel_drift=float("nan"), cv=float("nan"), slope=float("nan"))
    t = np.arange(v.size)[m]
    y = v[m]
    a, b = np.polyfit(t, y, 1)  # y ~ a t + b
    mean = float(np.mean(y))
    span = a * (t.max() - t.min())
    return dict(
        rel_drift=float(abs(span) / abs(mean)) if mean else float("nan"),
        cv=float(np.std(y) / abs(mean)) if mean else float("nan"),
        slope=float(a),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq_folder", required=True)
    p.add_argument("--depth_npz", default=None)
    p.add_argument("--batch_size", type=int, default=_batch_size_default())
    p.add_argument("--out", default=None)
    args = p.parse_args()

    # --- load artifacts ---
    slam_path = _find_slam_npz(args.seq_folder)
    sl = np.load(slam_path, allow_pickle=True)
    tstamp = np.asarray(sl["tstamp"], np.int64).reshape(-1)
    disps = np.asarray(sl["disps"], np.float32)
    global_scale = float(sl["scale"]) if "scale" in sl.files else float("nan")

    depth_path = find_depth_npz(args.seq_folder, args.depth_npz)
    depths, frame_indices = load_depth_npz(depth_path)
    n = depths.shape[0]
    masks = load_masks(args.seq_folder, n)

    print(f"seq_folder: {args.seq_folder}")
    print(f"SLAM npz:   {slam_path}   global scale={global_scale:.4f}  keyframes={tstamp.size}")
    print(f"depth npz:  {depth_path}  frames={n}  batch_size={args.batch_size}")

    # --- (1) Any4D batch steps ---
    s_bg = per_frame_stat(depths, masks, use_mask=masks is not None)
    batch = analyze(s_bg, args.batch_size)
    print("\n[1] Any4D batch steps")
    print(f"    boundary jump factor = {batch['factor']:.2f}  (≈1 good; >=3 strong steps)")

    # --- (2) DPVO time-varying scale ---
    skf = per_keyframe_scale(disps, tstamp, depths, frame_indices, masks)
    tr = _trend(skf)
    print("\n[2] DPVO per-keyframe scale s(t)")
    print(f"    rel_drift={tr['rel_drift']:.3f}  cv={tr['cv']:.3f}  slope={tr['slope']:.3e}/kf")
    print(f"    (rel_drift >> cv-of-noise => single global scalar discards real drift)")

    # --- (3) hand residual ---
    hand = None
    side = sorted(glob.glob(os.path.join(args.seq_folder, "SLAM", "hand_depth_align_*.npz")))
    if side:
        sd = np.load(side[-1], allow_pickle=True)
        r = np.asarray(sd["r"], np.float32).reshape(-1) if "r" in sd.files else None
        if r is not None and np.isfinite(r).any():
            ht = _trend(r)
            off = float(np.nanmedian(r))
            hand = dict(offset=off, **ht)
            print("\n[3] Hand residual r(t)=d_map/z_hawor")
            print(f"    median offset={off:.3f} (1.0=consistent)  rel_drift={ht['rel_drift']:.3f}  cv={ht['cv']:.3f}")
    if hand is None:
        print("\n[3] Hand residual: no hand_depth_align sidecar (run export with HAWOR_HAND_DEPTH_ALIGN=1)")

    # --- attribution verdict ---
    print("\n=== ATTRIBUTION ===")
    contrib = []
    if np.isfinite(batch["factor"]) and batch["factor"] >= 1.8:
        contrib.append(("Any4D batch steps", batch["factor"], "Phase 1 source-side stitching"))
    if np.isfinite(tr["rel_drift"]) and tr["rel_drift"] >= 0.10:
        contrib.append(("DPVO time-varying scale", tr["rel_drift"], "Phase 2 smooth s(t)"))
    if hand and np.isfinite(hand["offset"]) and (abs(hand["offset"] - 1.0) >= 0.10 or hand["rel_drift"] >= 0.10):
        contrib.append(("Hand prior mismatch", abs(hand["offset"] - 1.0), "Phase 3 residual align"))
    if not contrib:
        print("No source exceeds its threshold — drift is mild; verify on more clips.")
    else:
        contrib.sort(key=lambda x: -x[1])
        for name, mag, fix in contrib:
            print(f"  * {name:24s} magnitude={mag:.2f}  -> {fix}")
        print(f"  dominant source: {contrib[0][0]}")

    # --- plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable ({e}); printed summary only.")
        return
    out = args.out or os.path.join(args.seq_folder, "scale_attribution.png")
    fig, ax = plt.subplots(3, 1, figsize=(11, 10))
    t = np.arange(n)
    ax[0].plot(t, s_bg, color="#2563eb", lw=1.3, label="background depth s(t)")
    for b in batch["boundaries"]:
        ax[0].axvline(b - 0.5, color="#ef4444", ls="--", lw=1, alpha=0.6)
    ax[0].set_title(f"[1] Any4D batch steps  factor={batch['factor']:.2f}")
    ax[0].set_ylabel("depth (m)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
    ax[1].plot(tstamp, skf, color="#10b981", marker=".", lw=1.0, label="per-keyframe scale")
    ax[1].axhline(global_scale, color="#9333ea", ls=":", label=f"global scalar={global_scale:.3f}")
    ax[1].set_title(f"[2] DPVO scale s(t)  rel_drift={tr['rel_drift']:.3f}")
    ax[1].set_ylabel("scale"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)
    if hand is not None and side:
        sd = np.load(side[-1], allow_pickle=True)
        r = np.asarray(sd["r"], np.float32).reshape(-1)
        ax[2].plot(np.arange(r.size), r, color="#f59e0b", lw=1.2, label="r(t)=d_map/z_hawor")
        ax[2].axhline(1.0, color="#6b7280", ls=":", label="consistent=1.0")
        ax[2].set_title(f"[3] Hand residual  offset={hand['offset']:.3f}")
        ax[2].legend(fontsize=8)
    else:
        ax[2].text(0.5, 0.5, "no hand_depth_align sidecar", ha="center", va="center")
    ax[2].set_xlabel("frame / keyframe"); ax[2].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=130); print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
