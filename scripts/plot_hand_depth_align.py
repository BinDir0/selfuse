#!/usr/bin/env python3
"""Inspect / plot a hand-depth-align sidecar (verification for the α(t) alignment).

Reads ``SLAM/hand_depth_align_{s}_{e}.npz`` (written by lib/pipeline/hand_depth_align.py
when HAWOR_HAND_DEPTH_ALIGN=1) and produces, per clip:
  * top:    z_hawor(t), α(t)·z_hawor(t), d_map(t)   (valid frames marked)
  * middle: α(t)
  * bottom: per-frame relative residual |α z − d| / d  (before vs after)
plus a printed summary (median/p90 residual before & after, corr(z,d), α range).

Run on the production machine after an aligned export:
  python scripts/plot_hand_depth_align.py --seq_folder /path/to/outputs/<clip_id>
  python scripts/plot_hand_depth_align.py --seq_folder ... --out /tmp/align_check.png
"""
import argparse
import glob
import os

import numpy as np


def find_sidecar(seq_folder, explicit):
    if explicit:
        return explicit
    cands = sorted(glob.glob(os.path.join(seq_folder, "SLAM", "hand_depth_align_*.npz")))
    if not cands:
        raise FileNotFoundError(f"no SLAM/hand_depth_align_*.npz under {seq_folder}")
    return cands[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq_folder", help="clip dir containing SLAM/hand_depth_align_*.npz")
    p.add_argument("--npz", help="explicit sidecar path (overrides --seq_folder lookup)")
    p.add_argument("--out", default=None, help="output png (default <seq_folder>/hand_depth_align.png)")
    args = p.parse_args()
    if not args.seq_folder and not args.npz:
        p.error("provide --seq_folder or --npz")

    path = find_sidecar(args.seq_folder, args.npz)
    d = np.load(path, allow_pickle=True)
    alpha = np.asarray(d["alpha"], np.float32).reshape(-1)
    z = np.asarray(d["z_hawor"], np.float32).reshape(-1)
    dm = np.asarray(d["d_map"], np.float32).reshape(-1)
    valid = np.asarray(d["valid"]).reshape(-1).astype(bool)
    T = alpha.shape[0]
    t = np.arange(T)

    v = valid & np.isfinite(z) & np.isfinite(dm) & (dm > 0)
    rel_before = np.full(T, np.nan, np.float32)
    rel_after = np.full(T, np.nan, np.float32)
    rel_before[v] = np.abs(z[v] - dm[v]) / dm[v]
    rel_after[v] = np.abs(alpha[v] * z[v] - dm[v]) / dm[v]

    def med(x):
        x = x[np.isfinite(x)]
        return float(np.median(x)) if x.size else float("nan")

    def p90(x):
        x = x[np.isfinite(x)]
        return float(np.percentile(x, 90)) if x.size else float("nan")

    corr = (float(np.corrcoef(z[v], dm[v])[0, 1])
            if v.sum() > 1 and np.std(z[v]) > 0 and np.std(dm[v]) > 0 else float("nan"))

    print(f"sidecar: {path}")
    print(f"frames: {T}  valid: {int(v.sum())}")
    print(f"rel residual  before: median={med(rel_before):.3f} p90={p90(rel_before):.3f}")
    print(f"rel residual  after : median={med(rel_after):.3f} p90={p90(rel_after):.3f}")
    print(f"corr(z_hawor, d_map): {corr:.3f}")
    print(f"alpha: range=[{float(alpha.min()):.3f},{float(alpha.max()):.3f}] "
          f"max|Δ²α|={float(np.max(np.abs(np.diff(alpha, 2)))) if T > 2 else 0.0:.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"[plot] matplotlib unavailable ({error}); printed summary only.")
        return

    out = args.out or (os.path.join(args.seq_folder, "hand_depth_align.png") if args.seq_folder
                       else os.path.splitext(path)[0] + ".png")
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(t, z, color="#94a3b8", lw=1, label="z_hawor (raw)")
    ax[0].plot(t, alpha * z, color="#6366f1", lw=1.6, label="α·z_hawor (aligned)")
    ax[0].plot(t, dm, color="#10b981", lw=1.6, label="d_map (depth map @ hand)")
    ax[0].scatter(t[v], dm[v], s=8, color="#10b981", zorder=3)
    ax[0].set_ylabel("depth (m)"); ax[0].legend(loc="best", fontsize=8); ax[0].set_title(os.path.basename(path))
    ax[1].plot(t, alpha, color="#f59e0b", lw=1.6); ax[1].set_ylabel("α(t)")
    ax[2].plot(t, rel_before, color="#ef4444", lw=1, label="before")
    ax[2].plot(t, rel_after, color="#2563eb", lw=1.6, label="after")
    ax[2].set_ylabel("|αz−d|/d"); ax[2].set_xlabel("frame"); ax[2].legend(loc="best", fontsize=8)
    for a in ax:
        a.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=130); print(f"saved {out}")


if __name__ == "__main__":
    main()
