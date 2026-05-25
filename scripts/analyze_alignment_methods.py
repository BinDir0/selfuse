#!/usr/bin/env python3
"""Offline decision tool: which metric-alignment method does this clip actually need?

Reads a finished pipeline result (read-only) and judges candidate alignment methods
on REAL data using OBJECTIVE, source-decoupled metrics (so we don't self-confirm):

  Objective (no GT, literature-standard video-depth metric):
    A. Flow-warp TEMPORAL CONSISTENCY (TC) — warp depth_t -> t+1 by optical flow on
       static pixels; scale-invariant relative disagreement. Any4D per-batch scale
       steps spike TC at batch boundaries. This needs only Any4D depth + RGB frames
       (NOT DPVO), so it is independent of our pipeline's scale fitting.

  Diagnostics / internal alignment:
    [1] Any4D batch-step factor (boundary |Δ bg-depth| / interior).
    [3] hand vs depth global offset  r = d_hand / z_wrist  (z_wrist from cam_space).

Key correctness points (learned the hard way):
  * DPVO disps are SPARSE patches (~0.3% of pixels are real; the rest is a constant
    fill), so DPVO gives NO reliable dense depth reference. We use it only at real
    patch pixels, and only as a weak cross-check — NOT as the metric anchor.
  * Stitching removes per-batch STEPS but leaves ONE global scale free. We therefore
    re-anchor the global median after stitching, so the hand-offset comparison is not
    confounded by an arbitrary global-level shift (that shift is layer-2/3's job).

Self-contained: numpy + cv2 (flow + resize) + matplotlib. No torch/MANO/joblib.

Usage:
  python scripts/analyze_alignment_methods.py --seq_folder /root/hawor_result/.../stage_outputs/<clip>
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


# ----------------------------- loading -----------------------------

def _find(seq, *patterns):
    for pat in patterns:
        c = sorted(glob.glob(os.path.join(seq, pat)))
        if c:
            return c[-1]
    return None


def load_any4d_depths(seq):
    p = _find(seq, "SLAM/any4d_depth_*_allframes.npz", "SLAM/any4d_depth_*.npz", "SLAM/dense_depth_any4d_*.npz")
    d = np.load(p, allow_pickle=False)
    if "depths" in d.files:
        depths = np.asarray(d["depths"], np.float32)
    else:
        depths = d["depths_uint16"].astype(np.float32) * 1e-3
    return depths, p


def load_slam(seq):
    p = _find(seq, "SLAM/hawor_slam_w_scale_*.npz")
    d = np.load(p, allow_pickle=True)
    return dict(scale=float(d["scale"]), disps=np.asarray(d["disps"], np.float32),
                tstamp=np.asarray(d["tstamp"], np.int64).reshape(-1)), p


def load_masks(seq, n):
    p = _find(seq, "tracks_*_*/model_masks.npy")
    if p is None:
        return None
    m = np.asarray(np.load(p, allow_pickle=True))
    if m.ndim == 4:
        m = m.any(axis=1)
    return m if m.ndim == 3 and m.shape[0] >= n else None


def load_wrist_cam_z(seq, n):
    """Per-frame camera-frame wrist z from cam_space chunk JSONs, combined over hands."""
    per_hand = []
    for hand_dir in sorted(glob.glob(os.path.join(seq, "cam_space", "*"))):
        if not os.path.isdir(hand_dir):
            continue
        zi = np.full((n,), np.nan, np.float32)
        for jf in sorted(glob.glob(os.path.join(hand_dir, "*.json"))):
            base = os.path.splitext(os.path.basename(jf))[0]
            try:
                s, _e = (int(x) for x in base.split("_"))
            except ValueError:
                continue
            trans = np.asarray(json.load(open(jf))["init_trans"], np.float32)  # (1,T,3)
            tz = trans[0, :, 2]
            idx = np.arange(s, s + tz.shape[0])
            ok = (idx >= 0) & (idx < n)
            zi[idx[ok]] = tz[ok]
        per_hand.append(zi)
    if not per_hand:
        return np.full((n,), np.nan, np.float32)
    with np.errstate(invalid="ignore"):
        return np.nanmedian(np.vstack(per_hand), axis=0).astype(np.float32)


def _frames_dir(seq):
    cand = [os.path.join(seq, "frames"),
            os.path.join(os.path.dirname(os.path.dirname(seq)), "frames", os.path.basename(seq))]
    for c in cand:
        if os.path.isdir(c) and glob.glob(os.path.join(c, "*.jpg")):
            return c
    hit = glob.glob(os.path.join(os.path.dirname(os.path.dirname(seq)), "frames", "*", "000000.jpg"))
    return os.path.dirname(hit[0]) if hit else None


# ----------------------------- helpers -----------------------------

def resize_to(arr, hw, nearest=False):
    h, w = hw
    if arr.shape[:2] == (h, w):
        return arr
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(arr, (w, h), interpolation=interp)


def trimmed_median(vals, qlo=0.05, qhi=0.95):
    v = vals[np.isfinite(vals) & (vals > 0)]
    if v.size < 50:
        return np.nan
    lo, hi = np.quantile(v, [qlo, qhi])
    vv = v[(v >= lo) & (v <= hi)]
    return float(np.median(vv)) if vv.size else float(np.median(v))


def bg_depth_series(depths, masks, dmin=0.05, dmax=10.0):
    n = depths.shape[0]
    s = np.full(n, np.nan, np.float32)
    for t in range(n):
        f = depths[t]
        sel = np.isfinite(f) & (f > dmin) & (f < dmax)
        if masks is not None:
            sel &= ~(resize_to(masks[t].astype(np.uint8), f.shape, nearest=True) > 0)
        s[t] = trimmed_median(f[sel])
    return s


def boundary_factor(s, B):
    n = s.shape[0]
    ds = np.abs(np.diff(s))
    bnd = [b for b in range(B, n, B)]
    bset = {b - 1 for b in bnd}
    interior = np.array([ds[i] for i in range(len(ds)) if i not in bset and np.isfinite(ds[i])])
    bound = np.array([ds[b - 1] for b in bnd if b - 1 < len(ds) and np.isfinite(ds[b - 1])])
    med_int = float(np.median(interior)) if interior.size else np.nan
    factor = (float(np.median(bound)) / med_int) if (bound.size and med_int and med_int > 0) else np.nan
    return dict(factor=factor, boundaries=bnd, med_int=med_int)


def trend(curve):
    v = np.asarray(curve, float); m = np.isfinite(v)
    if m.sum() < 3:
        return dict(rel_drift=np.nan, cv=np.nan, corr_t=np.nan)
    t = np.arange(v.size)[m]; y = v[m]
    a, _ = np.polyfit(t, y, 1); mean = float(np.mean(y))
    return dict(rel_drift=float(abs(a * (t.max() - t.min())) / abs(mean)) if mean else np.nan,
                cv=float(np.std(y) / abs(mean)) if mean else np.nan,
                corr_t=float(np.corrcoef(t, y)[0, 1]) if np.std(y) > 0 else np.nan)


# ----------------------------- objective metric A: flow-warp TC -----------------------------

def _gray_at(frames_dir, idx, hw):
    img = cv2.imread(os.path.join(frames_dir, f"{idx:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return resize_to(img, hw)


def flow_warp_metrics_pair(depthA, depthB, gA, gB, mask_hw_bool):
    """Warp depthB->A by optical flow; report depth-agreement in the Any4D video-depth
    metric form (AbsRel, delta<1.25) over static valid pixels, used here as a no-GT
    TEMPORAL consistency. Returns dict with:
      absrel/delta            : NO per-pair scale align -> catches absolute scale STEPS.
      absrel_si/delta_si      : median-scale aligned first (paper protocol) -> catches
                                STRUCTURE error independent of the scale jump.
    """
    if gA is None or gB is None:
        return None
    flow = cv2.calcOpticalFlowFarneback(gA, gB, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    h, w = gA.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    mapx, mapy = xs + flow[..., 0], ys + flow[..., 1]
    dB_w = cv2.remap(depthB, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderValue=np.nan)
    inb = (mapx >= 0) & (mapx < w) & (mapy >= 0) & (mapy < h)
    fmag = np.hypot(flow[..., 0], flow[..., 1])
    valid = inb & (fmag < 0.15 * w) & (~mask_hw_bool) \
        & np.isfinite(depthA) & (depthA > 0.05) & np.isfinite(dB_w) & (dB_w > 0.05)
    if valid.sum() < 200:
        return None
    a, b = depthA[valid], dB_w[valid]

    def _absrel_delta(pred, ref):
        absrel = float(np.median(np.abs(pred - ref) / ref))
        ratio = np.maximum(pred / ref, ref / pred)
        return absrel, float(np.mean(ratio < 1.25))

    absrel, delta = _absrel_delta(a, b)
    k = float(np.median(b / a))                      # median scale align (Any4D protocol)
    absrel_si, delta_si = _absrel_delta(a * k, b)
    return dict(absrel=absrel, delta=delta, absrel_si=absrel_si, delta_si=delta_si)


def measure_flow_metrics(depths, masks, frames_dir, pairs):
    h, w = depths.shape[1:]
    out, gcache = {}, {}
    def g(idx):
        if idx not in gcache:
            gcache[idx] = _gray_at(frames_dir, idx, (h, w))
        return gcache[idx]
    for (a, b) in pairs:
        mk = (resize_to(masks[a].astype(np.uint8), (h, w), nearest=True) > 0) if masks is not None else np.zeros((h, w), bool)
        out[(a, b)] = flow_warp_metrics_pair(depths[a], depths[b], g(a), g(b), mk)
    return out


def _agg(metrics, pairs, key):
    vals = [metrics[p][key] for p in pairs if metrics.get(p) is not None]
    return float(np.median(vals)) if vals else float("nan")


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq_folder", required=True)
    ap.add_argument("--batch_size", type=int, default=int(os.environ.get("HAWOR_METRIC3D_BATCH_SIZE", "32")))
    ap.add_argument("--tc_stride", type=int, default=6, help="sample 1 interior pair every N frames for the TC baseline")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seq, B = args.seq_folder, args.batch_size

    depths, dpath = load_any4d_depths(seq)
    n, H, W = depths.shape
    slam, _ = load_slam(seq)
    masks = load_masks(seq, n)
    wrist_z = load_wrist_cam_z(seq, n)
    frames_dir = _frames_dir(seq)

    print(f"seq: {seq}")
    print(f"any4d depths {depths.shape} from {os.path.basename(dpath)}  median={np.nanmedian(depths):.3f} m")
    print(f"global scale={slam['scale']:.4g}  frames_dir={'OK' if frames_dir else 'MISSING'}  batch={B}")
    report = {"n": n, "B": B}

    # [1] Any4D batch-step factor on background depth
    s_bg = bg_depth_series(depths, masks)
    bf = boundary_factor(s_bg, B)
    print(f"\n[1] Any4D batch-step factor = {bf['factor']:.2f}  (~1 clean, >=3 strong steps)")
    report["any4d_factor_raw"] = bf["factor"]

    # ---- M1: stitch by adjacent-boundary bg-depth ratio (removes per-batch step),
    #          then RE-ANCHOR global median (so absolute level is held fixed). ----
    nbatch = int(np.ceil(n / B))
    cf_b = np.ones(nbatch)
    for bi in range(1, nbatch):
        lo = bi * B
        # adjacent frames straddling the boundary ~ same content; their bg-depth ratio ~ the s̃ step
        pre = np.nanmedian(s_bg[max(0, lo - 2):lo])
        post = np.nanmedian(s_bg[lo:lo + 2])
        cf_b[bi] = cf_b[bi - 1] * (pre / post) if (np.isfinite(pre) and np.isfinite(post) and post > 0) else cf_b[bi - 1]
    cf = np.repeat(cf_b, B)[:n]
    s_bg_st = s_bg * cf
    # re-anchor: hold global median equal to the raw global median (M1 must not move absolute level)
    g_raw, g_st = np.nanmedian(s_bg), np.nanmedian(s_bg_st)
    if np.isfinite(g_raw) and np.isfinite(g_st) and g_st > 0:
        cf = cf * (g_raw / g_st)
        s_bg_st = s_bg * cf
    bf_st = boundary_factor(s_bg_st, B)
    print(f"    after stitch+re-anchor: factor {bf['factor']:.2f} -> {bf_st['factor']:.2f}  "
          f"(global median held: {np.nanmedian(s_bg):.3f} -> {np.nanmedian(s_bg_st):.3f} m)")
    report["any4d_factor_stitched"] = bf_st["factor"]

    # ---- OBJECTIVE METRIC A: flow-warp temporal consistency in Any4D video-depth form ----
    tc_b = tc_b_st = tc_int = float("nan")
    if frames_dir is not None and cv2 is not None:
        bnd_pairs = [(b - 1, b) for b in bf["boundaries"] if 0 <= b - 1 and b < n]
        int_pairs = [(t, t + 1) for t in range(0, n - 1, args.tc_stride) if (t % B) not in (B - 1, 0)]
        m_raw = measure_flow_metrics(depths, masks, frames_dir, bnd_pairs + int_pairs)
        m_sti = measure_flow_metrics(depths * cf[:, None, None], masks, frames_dir, bnd_pairs)
        print("\n[A] OBJECTIVE temporal consistency = Any4D video-depth metrics (AbsRel, δ<1.25) on flow-warped pairs")
        print("    (NO per-pair scale align -> catches batch scale STEPS; δ higher=better, AbsRel lower=better)")
        print(f"    {'group':<26}{'AbsRel↓':>9}{'δ<1.25↑':>10}")
        print(f"    {'interior (within batch)':<26}{_agg(m_raw, int_pairs,'absrel'):>9.3f}{_agg(m_raw, int_pairs,'delta'):>10.3f}")
        print(f"    {'boundary  RAW':<26}{_agg(m_raw, bnd_pairs,'absrel'):>9.3f}{_agg(m_raw, bnd_pairs,'delta'):>10.3f}")
        print(f"    {'boundary  STITCHED':<26}{_agg(m_sti, bnd_pairs,'absrel'):>9.3f}{_agg(m_sti, bnd_pairs,'delta'):>10.3f}")
        print("    -- scale-invariant (median-scaled first, Any4D protocol) -> isolates STRUCTURE error --")
        print(f"    {'interior  (SI)':<26}{_agg(m_raw, int_pairs,'absrel_si'):>9.3f}{_agg(m_raw, int_pairs,'delta_si'):>10.3f}")
        print(f"    {'boundary RAW (SI)':<26}{_agg(m_raw, bnd_pairs,'absrel_si'):>9.3f}{_agg(m_raw, bnd_pairs,'delta_si'):>10.3f}")
        print("    => if boundary-SI ≈ interior but boundary-RAW is bad: the batch defect is PURELY a scale step.")
        tc_int = _agg(m_raw, int_pairs, 'absrel'); tc_b = _agg(m_raw, bnd_pairs, 'absrel'); tc_b_st = _agg(m_sti, bnd_pairs, 'absrel')
        report.update(absrel_interior=tc_int, absrel_boundary_raw=tc_b, absrel_boundary_stitched=tc_b_st,
                      delta_interior=_agg(m_raw, int_pairs,'delta'), delta_boundary_raw=_agg(m_raw, bnd_pairs,'delta'),
                      delta_boundary_stitched=_agg(m_sti, bnd_pairs,'delta'),
                      absrel_boundary_raw_si=_agg(m_raw, bnd_pairs,'absrel_si'))
    else:
        print("\n[A] flow-warp metrics SKIPPED (no frames dir or no cv2)")

    # ---- [3] hand vs depth global offset; check M1 leaves it ~unchanged (re-anchored) ----
    d_hand = np.full(n, np.nan, np.float32)
    for t in range(n):
        if masks is None:
            break
        mk = resize_to(masks[t].astype(np.uint8), (H, W), nearest=True) > 0
        sel = mk & np.isfinite(depths[t]) & (depths[t] > 0.05) & (depths[t] < 10.0)
        if sel.sum() >= 50:
            d_hand[t] = float(np.median(depths[t][sel]))
    r = np.where(np.isfinite(d_hand) & np.isfinite(wrist_z) & (wrist_z > 0), d_hand / np.maximum(wrist_z, 1e-6), np.nan)
    r_st = np.where(np.isfinite(d_hand) & np.isfinite(wrist_z) & (wrist_z > 0), (d_hand * cf) / np.maximum(wrist_z, 1e-6), np.nan)
    off = float(np.nanmedian(r)) if np.isfinite(r).sum() >= 8 else np.nan
    off_st = float(np.nanmedian(r_st)) if np.isfinite(r_st).sum() >= 8 else np.nan
    tr = trend(r)
    print(f"\n[3] hand vs depth  r=d_hand/z_wrist : median offset={off:.3f} (1=consistent), rel_drift={tr['rel_drift']:.3f}, cv={tr['cv']:.3f}")
    print(f"    after M1 (re-anchored): offset {off:.3f} -> {off_st:.3f}  (expect ~unchanged: M1 removes steps, not absolute level)")
    report.update(hand_offset=off, hand_offset_after_M1=off_st, hand_rel_drift=tr["rel_drift"])

    # ---- verdict ----
    print("\n=== VERDICT ===")
    if np.isfinite(bf["factor"]) and bf["factor"] >= 3:
        gain = f"; objective TC boundary {tc_b:.3f} -> {tc_b_st:.3f} vs interior {tc_int:.3f}" if np.isfinite(tc_b_st) else ""
        print(f"  M1 (Any4D batch stitch): NEEDED — factor {bf['factor']:.1f}; stitch -> {bf_st['factor']:.1f}{gain}")
    print(f"  Camera drift: pure-DPVO drifts (pose+scale) over long clips; correct fix = DPV-SLAM backend "
          f"(loop closure). NOT testable offline; flagged for production.")
    print(f"  Hand reconciliation: GLOBAL offset {off:.2f}x (drift {tr['rel_drift']:.3f}) — a single factor, NOT per-frame; "
          f"investigate focal (=600 default) as root cause before any rescale.")

    # ---- plot ----
    out = args.out or os.path.join(seq, "alignment_analysis.png")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        t = np.arange(n)
        fig, ax = plt.subplots(3, 1, figsize=(12, 10))
        ax[0].plot(t, s_bg, color="#2563eb", lw=0.9, label="bg depth raw")
        ax[0].plot(t, s_bg_st, color="#16a34a", lw=0.9, alpha=0.85, label="after M1 stitch (re-anchored)")
        for b in bf["boundaries"]:
            ax[0].axvline(b - 0.5, color="#ef4444", ls="--", lw=0.4, alpha=0.3)
        ax[0].set_title(f"[1] Any4D batch steps  factor {bf['factor']:.1f} -> {bf_st['factor']:.1f}")
        ax[0].set_ylabel("depth (m)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
        if np.isfinite(tc_b):
            ax[1].bar(["interior", "boundary RAW", "boundary STITCHED"], [tc_int, tc_b, tc_b_st],
                      color=["#9ca3af", "#ef4444", "#16a34a"])
            ax[1].set_title("[A] OBJECTIVE temporal AbsRel (Any4D metric form; lower=better)")
            ax[1].set_ylabel("AbsRel (flow-warp, no scale align)")
        ax[2].plot(t, r, color="#e11d48", lw=0.9, label=f"r=d_hand/z_wrist (offset {off:.2f})")
        ax[2].axhline(1.0, color="#6b7280", ls=":")
        ax[2].set_title("[3] hand vs depth global offset"); ax[2].set_xlabel("frame")
        ax[2].legend(fontsize=8); ax[2].grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(out, dpi=120); print(f"\nsaved {out}")
    except Exception as e:
        print(f"[plot] skipped: {e}")

    with open(os.path.join(seq, "alignment_analysis.json"), "w") as f:
        json.dump({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in report.items()}, f, indent=2)
    print(f"saved {os.path.join(seq, 'alignment_analysis.json')}")


if __name__ == "__main__":
    main()
