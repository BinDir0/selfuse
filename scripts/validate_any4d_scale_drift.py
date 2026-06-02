#!/usr/bin/env python3
"""Independent check that Any4D's per-batch scale steps are gone from the saved dense depth.

This does NOT use the overlap-stitch's own measurements. It reuses an INDEPENDENT signal — the
DPVO SLAM keyframe geometry, which is one globally-consistent solve and therefore *step-free* — to
detect whether the dense depth still carries per-batch scale steps:

  scale_kf = median( pred_depth(kf) / slam_depth(kf) )   over hand-masked, near/far-gated pixels

Because the SLAM depth has no batch structure, any variation of ``scale_kf`` that lines up with the
32-frame Any4D batch grid is an Any4D scale step. We group keyframes by their original batch index
and report the batch-correlated step (spread of per-batch medians). A clip whose dense depth is
step-free should be ~flat; the overlap=0 arm of the A/B should show a visible step, the overlap=4
arm should be flat.

Artifacts read (per episode ``seq_folder``, all under ``SLAM/``):
  - hawor_slam_w_scale_{s}_{e}.npz : tstamp (keyframe frame ids), disps (K,h,w)
  - dense_depth_any4d_{s}_{e}.npz  : depths_uint16 (T,H,W) mm  + frame_indices
  - (optional) ../tracks_{s}_{e}/model_masks.npy : per-frame hand masks
  - (optional) any4d_stitch_cf_{s}_{e}.npz : overlap-stitch diagnostics (printed if present)

numpy + cv2 only (no torch / GPU), so it runs on any machine.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

ANY4D_BATCH_SIZE = int(os.environ.get("HAWOR_ANY4D_BATCH_SIZE", "32"))


def _find_slam_pair(seq_folder: str):
    slam_dir = os.path.join(seq_folder, "SLAM")
    matches = sorted(glob.glob(os.path.join(slam_dir, "hawor_slam_w_scale_*.npz")))
    if not matches:
        raise FileNotFoundError(f"no hawor_slam_w_scale_*.npz under {slam_dir}")
    slam_path = matches[0]
    m = re.search(r"hawor_slam_w_scale_(\d+)_(\d+)\.npz$", os.path.basename(slam_path))
    if not m:
        raise ValueError(f"cannot parse track range from {slam_path}")
    s, e = int(m.group(1)), int(m.group(2))
    dense_path = os.path.join(slam_dir, f"dense_depth_any4d_{s}_{e}.npz")
    if not os.path.exists(dense_path):
        raise FileNotFoundError(f"missing dense depth: {dense_path}")
    return slam_path, dense_path, s, e


def _load_dense(dense_path: str):
    with np.load(dense_path, allow_pickle=False) as d:
        idx = np.asarray(d["frame_indices"], np.int64)
        if "depths_uint16" in d.files:
            depths = d["depths_uint16"].astype(np.float32) * 1e-3
        else:
            depths = np.asarray(d["depths"], np.float32)
    return idx, depths


def _load_masks(seq_folder: str, s: int, e: int) -> Optional[np.ndarray]:
    path = os.path.join(seq_folder, f"tracks_{s}_{e}", "model_masks.npy")
    if not os.path.exists(path):
        return None
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None


def _robust_scale(slam_depth, pred_depth, msk, near=0.4, far=0.7) -> float:
    """median(pred/slam) over hand-masked, near/far-gated pixels; iterative like est_scale stage-1.

    Widens the near/far window if too few pixels survive (mirrors _estimate_scale's fallback)."""
    if slam_depth.shape != pred_depth.shape:
        slam_depth = cv2.resize(slam_depth, (pred_depth.shape[1], pred_depth.shape[0]))
    if msk is None:
        msk = np.zeros_like(pred_depth, np.float32)
    elif msk.shape != pred_depth.shape:
        msk = cv2.resize(msk.astype(np.float32), (pred_depth.shape[1], pred_depth.shape[0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        s = pred_depth / slam_depth
    base = (msk < 0.5) & np.isfinite(s) & (slam_depth > 0)
    n0, f0 = near, far
    for _ in range(11):
        valid = base & (pred_depth > n0) & (pred_depth < f0)
        if int(valid.sum()) >= 50:
            scale = float(np.median(s[valid]))
            for _ in range(10):
                sd0 = slam_depth * scale
                v = valid & (sd0 > 0) & (sd0 < f0)
                if int(v.sum()) < 50:
                    break
                scale = float(np.median(s[v]))
            return scale
        n0 = max(0.01, n0 - 0.1)
        f0 = f0 + 0.2
    return float("nan")


def analyze_seq(seq_folder: str, batch_size: int = ANY4D_BATCH_SIZE) -> dict:
    slam_path, dense_path, s, e = _find_slam_pair(seq_folder)
    with np.load(slam_path, allow_pickle=False) as sl:
        tstamp = np.asarray(sl["tstamp"], np.int64).reshape(-1)
        disps = np.asarray(sl["disps"], np.float32)
    idx, depths = _load_dense(dense_path)
    pos_of = {int(f): i for i, f in enumerate(idx.tolist())}
    masks = _load_masks(seq_folder, s, e)

    rows = []  # (frame_id, batch_idx, scale_kf)
    for k, fid in enumerate(tstamp.tolist()):
        if fid not in pos_of or k >= disps.shape[0]:
            continue
        pred = depths[pos_of[fid]]
        with np.errstate(divide="ignore", invalid="ignore"):
            slam_depth = 1.0 / disps[k]
        msk = None
        if masks is not None and 0 <= fid < len(masks):
            mk = masks[fid]
            msk = np.asarray(mk.cpu().numpy() if hasattr(mk, "cpu") else mk, np.float32)
        sc = _robust_scale(slam_depth, pred, msk)
        if np.isfinite(sc) and sc > 0:
            rows.append((int(fid), int(fid) // batch_size, float(sc)))

    out = {"seq_folder": seq_folder, "track": [s, e], "n_keyframes": int(len(tstamp)),
           "n_scaled": len(rows), "batch_size": batch_size}
    if len(rows) < 3:
        out["reason"] = "too few usable keyframes"
        return out

    batch_ids = sorted({b for _, b, _ in rows})
    batch_med = {b: float(np.median([sc for _, bb, sc in rows if bb == b])) for b in batch_ids}
    g = float(np.median([sc for _, _, sc in rows]))
    rel = np.array([batch_med[b] / g for b in batch_ids], np.float64)  # per-batch level vs global
    logrel = np.log(rel)
    # batch-correlated step metrics (lower = flatter = step-free)
    out.update(
        global_scale=g,
        n_batches=len(batch_ids),
        batch_rel_std=float(np.std(logrel)),                       # spread of per-batch log-levels
        batch_rel_range=float(rel.max() / rel.min() - 1.0),        # max/min spread (fractional)
        max_consecutive_jump=float(np.max(np.abs(np.diff(logrel)))) if len(logrel) > 1 else 0.0,
        per_batch_rel={int(b): float(batch_med[b] / g) for b in batch_ids},
    )

    stitch_path = os.path.join(seq_folder, "SLAM", f"any4d_stitch_cf_{s}_{e}.npz")
    if os.path.exists(stitch_path):
        with np.load(stitch_path, allow_pickle=False) as st:
            out["overlap_stitch"] = {
                "overlap": int(st["overlap"]) if "overlap" in st.files else None,
                "n_solved": int(st["n_solved"]) if "n_solved" in st.files else None,
                "n_flagged": int(st["n_flagged"]) if "n_flagged" in st.files else None,
                "flatness_median": (float(np.nanmedian(st["boundary_flatness"]))
                                    if "boundary_flatness" in st.files and len(st["boundary_flatness"]) else None),
            }
    return out


def _print_report(res: dict) -> None:
    print(f"\n=== {res['seq_folder']}  track {res.get('track')} ===")
    if "reason" in res:
        print(f"  SKIP: {res['reason']} (keyframes={res.get('n_keyframes')}, scaled={res.get('n_scaled')})")
        return
    print(f"  keyframes scaled : {res['n_scaled']}/{res['n_keyframes']}  over {res['n_batches']} batches")
    print(f"  global scale     : {res['global_scale']:.4f}")
    print(f"  batch step (log-std)        : {res['batch_rel_std']:.4f}   <-- lower = flatter/step-free")
    print(f"  batch step (max/min range)  : {res['batch_rel_range'] * 100:.2f}%")
    print(f"  max consecutive batch jump  : {res['max_consecutive_jump']:.4f} (log)")
    if "overlap_stitch" in res:
        os_ = res["overlap_stitch"]
        print(f"  overlap-stitch diag : overlap={os_['overlap']} solved={os_['n_solved']} "
              f"flagged={os_['n_flagged']} flatness_med={os_['flatness_median']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seq_folder", help="episode output dir containing SLAM/ (overlap arm)")
    ap.add_argument("--compare", default=None, help="second seq_folder (e.g. overlap=0 baseline) for A/B")
    ap.add_argument("--batch_size", type=int, default=ANY4D_BATCH_SIZE)
    ap.add_argument("--json", default=None, help="write the full result(s) as JSON to this path")
    args = ap.parse_args()
    if cv2 is None:
        raise SystemExit("cv2 is required")

    results = {"overlap_arm": analyze_seq(args.seq_folder, args.batch_size)}
    _print_report(results["overlap_arm"])
    if args.compare:
        results["baseline_arm"] = analyze_seq(args.compare, args.batch_size)
        _print_report(results["baseline_arm"])
        a, b = results["overlap_arm"], results["baseline_arm"]
        if "batch_rel_std" in a and "batch_rel_std" in b:
            print("\n--- A/B (baseline overlap=0  ->  overlap arm) ---")
            print(f"  batch step (log-std): {b['batch_rel_std']:.4f}  ->  {a['batch_rel_std']:.4f}")
            print(f"  expectation: the overlap arm should be NOTABLY LOWER (steps removed).")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
