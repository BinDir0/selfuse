#!/usr/bin/env python3
"""Eyeball the static-background correspondences that the drift diagnostic relies on.

The whole drift verdict rests on the SIFT matches between two frames being CORRECT. In
repetitive scenes (factory shelves, tiles, grids) RANSAC can lock onto a self-consistent but
WRONG match set, which would fake drift. This tool draws the exact same matches the diagnostic
uses (same SIFT + Lowe ratio + RANSAC + hand mask) for a chosen frame pair (a, b), so you can
visually confirm the inlier lines connect the SAME physical structure — especially at large Δ.

It writes a side-by-side image with the RANSAC-inlier matches drawn, plus, for reference, the
per-match homography residual under DPVO's rotation (so you can see which matches the pose
explains vs not). Read-only.

Usage:
  python scripts/viz_match_pair.py --seq_folder <SEQ> --a 100 --b 1600
  # or sweep a few b's against one a:
  python scripts/viz_match_pair.py --seq_folder <SEQ> --a 100 --bs 200,600,1100,1600
"""
import argparse
import glob
import os

import cv2
import numpy as np


def _find(seq, *pats):
    for p in pats:
        c = sorted(glob.glob(os.path.join(seq, p)))
        if c:
            return c[-1]
    return None


def quat_to_R(q):
    x, y, z, w = q
    n = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], np.float64)


def frames_dir(seq):
    for c in [os.path.join(seq, "frames"),
              os.path.join(os.path.dirname(os.path.dirname(seq)), "frames", os.path.basename(seq))]:
        if glob.glob(os.path.join(c, "*.jpg")):
            return c
    hit = glob.glob(os.path.join(os.path.dirname(os.path.dirname(seq)), "frames", "*", "000000.jpg"))
    return os.path.dirname(hit[0]) if hit else None


def load_masks(seq, n):
    p = _find(seq, "tracks_*_*/model_masks.npy")
    if p is None:
        return None
    m = np.asarray(np.load(p, allow_pickle=True))
    if m.ndim == 4:
        m = m.any(1)
    return m if m.ndim == 3 and m.shape[0] >= n else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq_folder", required=True)
    ap.add_argument("--a", type=int, required=True, help="anchor frame index")
    ap.add_argument("--b", type=int, help="second frame index")
    ap.add_argument("--bs", type=str, help="comma list of second frames (overrides --b)")
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--ransac", choices=["homography", "fundamental"], default="homography")
    ap.add_argument("--max_draw", type=int, default=80, help="cap drawn matches for clarity")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seq = args.seq_folder

    sl = np.load(_find(seq, "SLAM/hawor_slam_w_scale_*.npz"), allow_pickle=True)
    focal = float(sl["img_focal"]); cx0, cy0 = [float(v) for v in np.asarray(sl["img_center"]).reshape(-1)[:2]]
    dp = np.load(_find(seq, "SLAM/dpvo_raw_*.npz"), allow_pickle=False)
    traj = np.asarray(dp["traj"], np.float64)
    n = traj.shape[0]
    fdir = frames_dir(seq); assert fdir, "no frames dir"
    masks = load_masks(seq, n)
    sift = cv2.SIFT_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    def load_gray_and_feats(fi):
        img = cv2.imread(os.path.join(fdir, f"{fi:06d}.jpg"))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mk = None
        if masks is not None and fi < masks.shape[0]:
            mm = cv2.resize(masks[fi].astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            mk = (255 * (mm == 0)).astype(np.uint8)
        kp, des = sift.detectAndCompute(gray, mk)
        return img, kp, des

    H, W = cv2.imread(glob.glob(os.path.join(fdir, "*.jpg"))[0]).shape[:2]
    sx, sy = W / (2 * cx0), H / (2 * cy0)
    K = np.array([[focal * sx, 0, cx0 * sx], [0, focal * sy, cy0 * sy], [0, 0, 1]], np.float64)
    Kinv = np.linalg.inv(K)

    def dpvo_H(a, b):
        Ra = quat_to_R(traj[a, 3:7]).T; Rb = quat_to_R(traj[b, 3:7]).T
        return K @ (Rb @ Ra.T) @ Kinv

    bs = [int(x) for x in args.bs.split(",")] if args.bs else [args.b]
    imgA, kpA, desA = load_gray_and_feats(args.a)

    panels = []
    for b in bs:
        imgB, kpB, desB = load_gray_and_feats(b)
        m = bf.knnMatch(desA, desB, k=2)
        good = [p for p, q in (mm for mm in m if len(mm) == 2) if p.distance < args.ratio * q.distance]
        info = f"a={args.a} b={b} Δ={b-args.a}  good={len(good)}"
        if len(good) < 8:
            print(info + "  -> too few matches");
            continue
        ptsA = np.float32([kpA[g.queryIdx].pt for g in good])
        ptsB = np.float32([kpB[g.trainIdx].pt for g in good])
        if args.ransac == "homography":
            _, mask_in = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 3.0)
        else:
            _, mask_in = cv2.findFundamentalMat(ptsA, ptsB, cv2.FM_RANSAC, 1.5, 0.999)
        inl = mask_in.ravel().astype(bool) if mask_in is not None else np.zeros(len(good), bool)
        pa, pb = ptsA[inl], ptsB[inl]
        # per-match homography residual under DPVO rotation (green=explained, red=not)
        Hd = dpvo_H(args.a, b)
        ah = np.hstack([pa, np.ones((len(pa), 1))]) @ Hd.T
        ah = ah[:, :2] / ah[:, 2:3]
        res = np.linalg.norm(ah - pb, axis=1)
        info += f"  inliers={int(inl.sum())}  homog_res med={np.median(res):.1f}px"
        print(info)

        canvas = np.hstack([imgA, imgB])
        order = np.argsort(res)
        for idx in order[: args.max_draw]:
            xa, ya = pa[idx]; xb, yb = pb[idx]
            col = (0, 200, 0) if res[idx] < 5 else ((0, 165, 255) if res[idx] < 20 else (0, 0, 255))
            cv2.circle(canvas, (int(xa), int(ya)), 2, col, -1)
            cv2.circle(canvas, (int(xb) + W, int(yb)), 2, col, -1)
            cv2.line(canvas, (int(xa), int(ya)), (int(xb) + W, int(yb)), col, 1)
        cv2.putText(canvas, info, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "green<5px  orange<20px  red>=20px (homog residual under DPVO rot)",
                    (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        panels.append(canvas)

    if not panels:
        print("nothing drawn"); return
    out = args.out or os.path.join(seq, f"match_viz_a{args.a}.png")
    cv2.imwrite(out, np.vstack(panels))
    print(f"saved {out}  ({len(panels)} pair(s))")
    print("EYEBALL: do the lines connect the SAME physical structure? Parallel, ordered lines = good. "
          "Crossing/scattered lines, or matches onto a repeated-but-different shelf/tile = BAD (fake drift).")


if __name__ == "__main__":
    main()
