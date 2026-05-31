#!/usr/bin/env python3
"""Diagnose per-batch metric-scale STEPS in the exported Any4D depth.

Any4D predicts ONE metric scaling factor per inference (scale_token / ScaleAdaptor;
see thirdparty/Any4D/any4d/models/any4d/model.py). The dense per-frame depth is run
in batches of HAWOR_METRIC3D_BATCH_SIZE (default 48), each batch anchored to its own
reference frame, and the exported depth is NOT renormalized across batches. So the
absolute scale can jump at batch boundaries (frame positions 48, 96, ...). This script
quantifies whether such steps are real and how large, BEFORE any fix is chosen.

What it does for one clip:
  * s(t) = robust trimmed-median depth per frame (whole-frame, or background = ¬hand-mask)
  * compares |Δs| at batch boundaries vs interior frames -> "boundary jump factor"
  * reports per-boundary scale ratio s(b-1)/s(b) and significance vs interior noise
  * overlays d_map(t) from SLAM/hand_depth_align_*.npz (if present) on the boundaries
  * optional cross-check vs a Metric3D depth npz (per-frame, no multi-view batch scale):
    a control that should show NO boundary-aligned steps
  * reports the SLAM global `scale` for context

Run on the production machine (after an export) on clips spanning >2s (>=2 batches):
  python tools/diagnostics/diagnose_depth_batch_steps.py --seq_folder /path/to/outputs/<clip_id>
"""
import argparse
import glob
import os

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def _batch_size_default():
    try:
        return int(os.environ.get("HAWOR_METRIC3D_BATCH_SIZE", "48"))
    except ValueError:
        return 48


def load_depth_npz(path):
    """Return (depths[N,H,W] meters float32, frame_indices[N])."""
    d = np.load(path, allow_pickle=False)
    files = getattr(d, "files", [])
    if "depths_uint16" in files:                       # dense uint16 mm
        depths = d["depths_uint16"].astype(np.float32) * 1e-3
    elif "depths" in files:                            # merged any4d float meters
        depths = np.asarray(d["depths"], dtype=np.float32)
    else:
        raise ValueError(f"{path}: no 'depths' or 'depths_uint16' key (keys={list(files)})")
    fi = (np.asarray(d["frame_indices"], dtype=np.int64).reshape(-1)
          if "frame_indices" in files else np.arange(depths.shape[0]))
    return depths, fi


def find_depth_npz(seq_folder, explicit):
    if explicit:
        return explicit
    slam = os.path.join(seq_folder, "SLAM")
    for pat in ("any4d_depth_*_*.npz", "dense_depth_any4d_*.npz"):
        c = sorted(glob.glob(os.path.join(slam, pat)))
        if c:
            return c[-1]
    raise FileNotFoundError(f"no any4d_depth/dense_depth npz under {slam}")


def load_masks(seq_folder, n):
    cand = sorted(glob.glob(os.path.join(seq_folder, "tracks_*_*", "model_masks.npy")))
    if not cand:
        return None
    try:
        m = np.load(cand[-1], allow_pickle=True)
    except Exception:
        return None
    m = np.asarray(m)
    if m.ndim == 4:
        m = m.any(axis=1)
    return m if m.ndim == 3 and m.shape[0] >= n else None


def trimmed_median(frame, mask_bg=None, qlo=0.05, qhi=0.95):
    f = frame
    sel = np.isfinite(f) & (f > 0)
    if mask_bg is not None:
        sel = sel & mask_bg
    v = f[sel]
    if v.size < 50:
        return np.nan
    lo, hi = np.quantile(v, [qlo, qhi])
    vv = v[(v >= lo) & (v <= hi)]
    return float(np.median(vv)) if vv.size else float(np.median(v))


def per_frame_stat(depths, masks=None, use_mask=False):
    n = depths.shape[0]
    s = np.full(n, np.nan, np.float32)
    for t in range(n):
        bg = None
        if use_mask and masks is not None and t < masks.shape[0]:
            mk = masks[t].astype(np.uint8)
            if cv2 is not None and mk.shape != depths[t].shape:
                mk = cv2.resize(mk, (depths[t].shape[1], depths[t].shape[0]), interpolation=cv2.INTER_NEAREST)
            if mk.shape == depths[t].shape:
                bg = mk <= 0  # background = NOT hand
        s[t] = trimmed_median(depths[t], bg)
    return s


def analyze(s, batch_size):
    n = s.shape[0]
    ds = np.abs(np.diff(s))                    # |Δs| over adjacent frames, length n-1; index i = transition i->i+1
    boundaries = [b for b in range(batch_size, n, batch_size)]  # transition (b-1)->b lives at ds[b-1]
    bset = set(b - 1 for b in boundaries)
    interior = np.array([ds[i] for i in range(len(ds)) if i not in bset and np.isfinite(ds[i])])
    bound = np.array([ds[b - 1] for b in boundaries if np.isfinite(ds[b - 1])])
    med_int = float(np.median(interior)) if interior.size else float("nan")
    p95_int = float(np.percentile(interior, 95)) if interior.size else float("nan")
    mad_int = float(np.median(np.abs(interior - med_int))) if interior.size else float("nan")
    factor = (float(np.median(bound)) / med_int) if (bound.size and med_int > 0) else float("nan")
    per_b = []
    for b in boundaries:
        if b - 1 >= len(ds):
            continue
        ratio = float(s[b - 1] / s[b]) if np.isfinite(s[b - 1]) and np.isfinite(s[b]) and s[b] > 0 else float("nan")
        z = ((ds[b - 1] - med_int) / (1.4826 * mad_int)) if mad_int and np.isfinite(ds[b - 1]) else float("nan")
        per_b.append((b, ratio, float(ds[b - 1]) if np.isfinite(ds[b - 1]) else float("nan"), z))
    return dict(boundaries=boundaries, med_int=med_int, p95_int=p95_int, factor=factor, per_b=per_b)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq_folder", required=True)
    p.add_argument("--depth_npz", default=None, help="explicit Any4D depth npz (else auto-find under SLAM/)")
    p.add_argument("--metric3d_npz", default=None, help="optional Metric3D depth npz for cross-check (control)")
    p.add_argument("--batch_size", type=int, default=_batch_size_default())
    p.add_argument("--use_mask", action="store_true", help="restrict s(t) to background (¬hand-mask)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    path = find_depth_npz(args.seq_folder, args.depth_npz)
    depths, fi = load_depth_npz(path)
    n = depths.shape[0]
    masks = load_masks(args.seq_folder, n) if args.use_mask else None
    s = per_frame_stat(depths, masks, args.use_mask)
    res = analyze(s, args.batch_size)

    print(f"depth npz: {path}   frames={n}  batch_size={args.batch_size}")
    print(f"boundaries (frame pos): {res['boundaries']}")
    print(f"interior |Δs|: median={res['med_int']:.4f}  p95={res['p95_int']:.4f}")
    print(f"BOUNDARY JUMP FACTOR (median boundary |Δs| / median interior |Δs|): {res['factor']:.2f}")
    print("per-boundary  pos  scale_ratio(s[b-1]/s[b])  |Δs|   z_vs_interior")
    for b, ratio, jump, z in res["per_b"]:
        flag = "  <== step" if (np.isfinite(z) and z > 5) else ""
        print(f"             {b:5d}      {ratio:8.3f}        {jump:7.4f}  {z:8.2f}{flag}")

    # d_map(t) from align sidecar
    dmap = None
    side = sorted(glob.glob(os.path.join(args.seq_folder, "SLAM", "hand_depth_align_*.npz")))
    if side:
        try:
            sd = np.load(side[-1], allow_pickle=True)
            dmap = np.asarray(sd["d_map"], np.float32).reshape(-1)
            print(f"hand d_map(t) loaded from {side[-1]} (len={dmap.shape[0]})")
        except Exception:
            pass

    # Metric3D control
    m3 = None
    if args.metric3d_npz:
        try:
            md, _ = load_depth_npz(args.metric3d_npz)
            m3s = per_frame_stat(md, masks, args.use_mask)
            m3 = analyze(m3s, args.batch_size)
            print(f"[control Metric3D] BOUNDARY JUMP FACTOR: {m3['factor']:.2f}  "
                  f"(≈1 expected if no batch-scale steps)")
        except Exception as e:
            print(f"[control Metric3D] skipped: {e}")

    # SLAM global scale for context
    sl = sorted(glob.glob(os.path.join(args.seq_folder, "SLAM", "hawor_slam_w_scale_*.npz")))
    if sl:
        try:
            print(f"[context] SLAM global scale = {float(np.load(sl[-1], allow_pickle=True)['scale']):.4f}")
        except Exception:
            pass

    # verdict
    f = res["factor"]
    if np.isfinite(f) and f >= 3.0:
        verdict = ("STEPS LIKELY REAL — boundary jumps >> interior noise; consistent with Any4D "
                   "per-batch metric-scale re-prediction. → favor source-side scale stitching.")
    elif np.isfinite(f) and f >= 1.8:
        verdict = "BORDERLINE — some boundary excess; inspect plot and try the Metric3D control."
    else:
        verdict = ("NO SIGNIFICANT STEPS at batch boundaries — drift is likely NOT from Any4D batching; "
                   "re-evaluate hand-vs-depth attribution.")
    print("VERDICT:", verdict)

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable ({e}); printed summary only.")
        return
    out = args.out or os.path.join(args.seq_folder, "depth_batch_steps.png")
    rows = 2 if dmap is not None else 1
    fig, ax = plt.subplots(rows, 1, figsize=(11, 4 * rows), squeeze=False)
    t = np.arange(n)
    ax[0][0].plot(t, s, color="#2563eb", lw=1.4, label="s(t) background trimmed-median depth")
    for b in res["boundaries"]:
        ax[0][0].axvline(b - 0.5, color="#ef4444", ls="--", lw=1, alpha=0.7)
    ax[0][0].set_ylabel("depth (m)"); ax[0][0].set_title(os.path.basename(path)); ax[0][0].legend(fontsize=8)
    ax[0][0].grid(True, alpha=0.25)
    if dmap is not None:
        td = np.arange(dmap.shape[0])
        ax[1][0].plot(td, dmap, color="#10b981", lw=1.4, label="d_map(t) hand-region depth")
        for b in res["boundaries"]:
            ax[1][0].axvline(b - 0.5, color="#ef4444", ls="--", lw=1, alpha=0.7)
        ax[1][0].set_ylabel("depth (m)"); ax[1][0].set_xlabel("frame"); ax[1][0].legend(fontsize=8)
        ax[1][0].grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=130); print(f"saved {out}")


if __name__ == "__main__":
    main()
