#!/usr/bin/env python3
"""Measure WHOLE-SEQUENCE temporal consistency of the two metric chains: the HAND and
the HEAD (camera). This tests the assumption behind the global hand-anchor factor k —
that the two depths share a ~uniform (low-drift) metric over the whole clip.

Outputs three curves + a drift decomposition (batch-step vs end-to-end drift vs noise):
  [HAND]   r(t)   = Any4D depth at hand mask / HaWoR wrist cam-z   (should be flat if uniform)
  [HEAD]   s(t)   = per-keyframe  Any4D depth / DPVO inverse-depth  on background (camera scale over time)
  [SCENE]  s_bg(t)= per-frame background trimmed-median depth       (shows batch steps + global drift)

For each: median, cv (std/mean), rel_drift (linear-fit endpoint span / mean), corr-with-time,
and a per-batch decomposition that separates the Any4D batch STEPS from any END-TO-END drift
(trend of per-batch medians across the clip) and the residual NOISE (after de-trend).

Read-only. numpy + cv2 + matplotlib. Run on a finished seq_folder (pre- or post-fix):
  python scripts/diagnose_temporal_consistency.py --seq_folder /path/to/stage_outputs/<clip>
"""
import argparse
import glob
import json
import os

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def _find(seq, *pats):
    for p in pats:
        c = sorted(glob.glob(os.path.join(seq, p)))
        if c:
            return c[-1]
    return None


def _resize(a, hw, nearest=False):
    h, w = hw
    if a.shape[:2] == (h, w):
        return a
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(a.astype(np.float32) if not nearest else a.astype(np.uint8), (w, h), interpolation=interp)


def load_depths(seq):
    p = _find(seq, "SLAM/any4d_depth_*_allframes.npz", "SLAM/any4d_depth_*.npz", "SLAM/dense_depth_any4d_*.npz")
    d = np.load(p, allow_pickle=False)
    return (np.asarray(d["depths"], np.float32) if "depths" in d.files
            else d["depths_uint16"].astype(np.float32) * 1e-3), p


def load_masks(seq, n):
    p = _find(seq, "tracks_*_*/model_masks.npy")
    if p is None:
        return None
    m = np.asarray(np.load(p, allow_pickle=True))
    if m.ndim == 4:
        m = m.any(1)
    return m if m.ndim == 3 and m.shape[0] >= n else None


def load_wrist_z(seq, n):
    per = []
    for hd in sorted(glob.glob(os.path.join(seq, "cam_space", "*"))):
        if not os.path.isdir(hd):
            continue
        zi = np.full(n, np.nan, np.float32)
        for jf in sorted(glob.glob(os.path.join(hd, "*.json"))):
            try:
                s, _e = (int(x) for x in os.path.splitext(os.path.basename(jf))[0].split("_"))
                tz = np.asarray(json.load(open(jf))["init_trans"], np.float32)[0, :, 2]
            except Exception:
                continue
            idx = np.arange(s, s + tz.shape[0]); ok = (idx >= 0) & (idx < n)
            zi[idx[ok]] = tz[ok]
        per.append(zi)
    if not per:
        return np.full(n, np.nan, np.float32)
    with np.errstate(invalid="ignore"):
        return np.nanmedian(np.vstack(per), axis=0).astype(np.float32)


def tmed(v, lo=0.05, hi=0.95):
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 50:
        return np.nan
    a, b = np.quantile(v, [lo, hi]); vv = v[(v >= a) & (v <= b)]
    return float(np.median(vv)) if vv.size else float(np.median(v))


def stats(curve, name):
    v = np.asarray(curve, float); m = np.isfinite(v)
    if m.sum() < 5:
        return f"  {name:8s}: too few valid points ({int(m.sum())})"
    t = np.arange(v.size)[m]; y = v[m]
    a, b = np.polyfit(t, y, 1); mean = float(np.mean(y))
    detrend = y - (a * t + b)
    rel_drift = abs(a * (t.max() - t.min())) / abs(mean) if mean else np.nan
    cv = float(np.std(y) / abs(mean)) if mean else np.nan
    cv_detrend = float(np.std(detrend) / abs(mean)) if mean else np.nan
    corr = float(np.corrcoef(t, y)[0, 1]) if np.std(y) > 0 else np.nan
    return (f"  {name:8s}: median={mean:.3f}  cv={cv:.3f}  rel_drift={rel_drift:.3f}  "
            f"corr_t={corr:+.2f}  noise_cv(detrended)={cv_detrend:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq_folder", required=True)
    ap.add_argument("--batch_size", type=int, default=int(os.environ.get("HAWOR_METRIC3D_BATCH_SIZE", "32")))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seq, B = a.seq_folder, a.batch_size

    depths, dp = load_depths(seq)
    n, H, W = depths.shape
    masks = load_masks(seq, n)
    wrist = load_wrist_z(seq, n)
    dpvo = np.load(_find(seq, "SLAM/dpvo_raw_*.npz"), allow_pickle=False)
    disps, kts = np.asarray(dpvo["disps"], np.float32), np.asarray(dpvo["tstamp_disps"], np.int64).reshape(-1)
    print(f"seq={seq}\nany4d {depths.shape} median={np.nanmedian(depths):.3f}m  keyframes={kts.size}  batch={B}\n")

    # [HAND] r(t) and [SCENE] s_bg(t)
    r = np.full(n, np.nan, np.float32); s_bg = np.full(n, np.nan, np.float32)
    for t in range(n):
        f = depths[t]; bg = np.isfinite(f) & (f > 0.05) & (f < 10)
        if masks is not None:
            mk = _resize(masks[t], (H, W), nearest=True) > 0
            s_bg[t] = tmed(f[bg & ~mk])
            sel = mk & bg
            if sel.sum() >= 50 and np.isfinite(wrist[t]) and wrist[t] > 0:
                r[t] = float(np.median(f[sel])) / wrist[t]
        else:
            s_bg[t] = tmed(f[bg])

    # [HEAD] s(t): per-keyframe camera scale = Any4D / (1/disps) on background
    s_kf = np.full(kts.size, np.nan, np.float32)
    for i, ts in enumerate(kts):
        if not (0 <= ts < n):
            continue
        sl = 1.0 / np.maximum(_resize(disps[i], (H, W)), 1e-9)
        pred = depths[ts]; bg = np.isfinite(pred) & (pred > 0.05) & (pred < 10) & np.isfinite(sl) & (sl > 0)
        if masks is not None:
            bg &= ~(_resize(masks[ts], (H, W), nearest=True) > 0)
        ratio = (pred / sl)[bg]
        if ratio.size:
            lo, hi = np.quantile(ratio, [0.1, 0.9]); rr = ratio[(ratio >= lo) & (ratio <= hi)]
            s_kf[i] = float(np.median(rr)) if rr.size else float(np.median(ratio))

    print("=== whole-sequence temporal consistency (lower cv/rel_drift = more uniform) ===")
    print(stats(r, "HAND r"))
    print(stats(s_kf, "HEAD s"))
    print(stats(s_bg, "SCENE bg"))

    # decomposition: separate Any4D batch STEPS from END-TO-END drift using per-batch medians
    nb = int(np.ceil(n / B))
    with np.errstate(invalid="ignore"):
        batch_med = np.array([np.nanmedian(s_bg[bi * B:min((bi + 1) * B, n)]) for bi in range(nb)], float)
    bm = batch_med[np.isfinite(batch_med)]
    if bm.size >= 3:
        tb = np.arange(batch_med.size)[np.isfinite(batch_med)]
        ab, bb = np.polyfit(tb, bm, 1)
        step_cv = float(np.std(bm) / np.mean(bm))                       # spread of per-batch levels = batch STEPS
        e2e_drift = float(abs(ab * (tb.max() - tb.min())) / np.mean(bm))  # trend across batches = END-TO-END drift
        print("\n=== SCENE decomposition (per-batch level over the clip) ===")
        print(f"  batch-to-batch spread (STEP) cv = {step_cv:.3f}   "
              f"end-to-end drift (trend across {bm.size} batches) = {e2e_drift:.3f}")
        print("  (big STEP cv = Any4D per-batch jumps -> fixed by overlap stitch; "
              "big end-to-end drift = global metric drift -> a single k/scale can't hold)")

    # plot
    out = a.out or os.path.join(seq, "temporal_consistency.png")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        t = np.arange(n)
        fig, ax = plt.subplots(3, 1, figsize=(12, 10))
        ax[0].plot(t, r, lw=0.8, color="#e11d48"); ax[0].set_title("[HAND] r(t)=Any4D@hand / HaWoR wrist-z"); ax[0].grid(alpha=.3)
        ax[1].plot(kts, s_kf, ".-", lw=0.8, ms=3, color="#2563eb"); ax[1].set_title("[HEAD] per-keyframe camera scale Any4D/(1/disps)"); ax[1].grid(alpha=.3)
        ax[2].plot(t, s_bg, lw=0.8, color="#16a34a"); ax[2].set_title("[SCENE] background depth s_bg(t)")
        for b in range(B, n, B):
            ax[2].axvline(b - .5, color="#ef4444", ls="--", lw=.4, alpha=.3)
        ax[2].set_xlabel("frame"); ax[2].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out, dpi=120); print(f"\nsaved {out}")
    except Exception as e:
        print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()
