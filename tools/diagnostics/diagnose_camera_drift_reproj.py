#!/usr/bin/env python3
"""Independent, Any4D-free, scale-free test of whether the DPVO camera trajectory drifts.

Principle: a STATIC scene point observed across frames must triangulate to ONE 3D point
given consistent camera poses. If the trajectory drifts (pose-shape OR scale), the
observations of a long-lived static track become mutually inconsistent, so the multi-view
triangulation residual (reprojection error) GROWS with the track's temporal span.

We therefore compare the reprojection error of SHORT-span tracks (adjacent frames — pure
local consistency, which DPVO's windowed BA guarantees) vs LONG-span tracks (spanning much
of the clip — only consistent if there is NO global drift):

  long-span err ≈ short-span err   -> DPVO globally consistent, NO drift -> s(t) NOT justified
  long-span err >> short-span err  -> DPVO drifts over the clip          -> s(t) justified

Uses ONLY DPVO poses + RGB frames + intrinsics + hand mask (to exclude the dynamic hand).
Scale-free (X and translations scale together; reprojection is invariant). Independent of
Any4D and of the noisy HEAD-ratio metric.

Honest limits: egocentric small baseline => far points have weak parallax => triangulation
noisy. Mitigated by (a) keeping only tracks with enough parallax, (b) robust medians,
(c) the RELATIVE long-vs-short comparison (common noise cancels). If the data is too noisy
to separate, the script says so.

Read-only. Run on a finished seq_folder:
  python tools/diagnostics/diagnose_camera_drift_reproj.py --seq_folder /path/to/stage_outputs/<clip>
"""
import argparse
import glob
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


def quat_to_R(q):  # q = [qx,qy,qz,qw]
    x, y, z, w = q
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_poses_K(seq):
    """Per-frame world->cam projection matrices P_i = K[R_w2c|t_w2c], scaled to frame res."""
    dp = np.load(_find(seq, "SLAM/dpvo_raw_*.npz"), allow_pickle=False)
    traj = np.asarray(dp["traj"], np.float64)  # (N,7) c2w [tx,ty,tz,qx,qy,qz,qw], DPVO units
    sl = np.load(_find(seq, "SLAM/hawor_slam_w_scale_*.npz"), allow_pickle=True)
    focal = float(sl["img_focal"]); cx0, cy0 = [float(v) for v in np.asarray(sl["img_center"]).reshape(-1)[:2]]
    return traj, focal, cx0, cy0


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


def build_tracks(fdir, n, masks, stride=2, max_corners=400, reseed_every=20, fb_thresh=1.0):
    """LK forward tracks on static background. Returns list of dict{frame_idx:(u,v)}."""
    tracks = {}            # track_id -> {f: (u,v)}
    alive = {}             # track_id -> (u,v) at current frame
    next_id = 0
    g_prev = None
    frame_list = list(range(0, n, stride))
    H = W = None
    for fi in frame_list:
        img = cv2.imread(os.path.join(fdir, f"{fi:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if H is None:
            H, W = img.shape
        g = img
        mk = None
        if masks is not None and fi < masks.shape[0]:
            mk = cv2.resize(masks[fi].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        if g_prev is not None and alive:
            ids = list(alive.keys())
            p0 = np.float32([alive[i] for i in ids]).reshape(-1, 1, 2)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(g_prev, g, p0, None, winSize=(21, 21), maxLevel=3)
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(g, g_prev, p1, None, winSize=(21, 21), maxLevel=3)
            fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
            new_alive = {}
            for k, i in enumerate(ids):
                if st[k] == 1 and st2[k] == 1 and fb[k] < fb_thresh:
                    u, v = p1[k, 0]
                    if 0 <= u < W and 0 <= v < H and not (mk is not None and mk[int(v), int(u)]):
                        tracks[i][fi] = (float(u), float(v))
                        new_alive[i] = (float(u), float(v))
            alive = new_alive
        # reseed
        if (fi // stride) % reseed_every == 0 or len(alive) < max_corners // 4:
            occ = np.zeros((H, W), np.uint8)
            if mk is not None:
                occ[mk] = 255
            corners = cv2.goodFeaturesToTrack(g, maxCorners=max_corners, qualityLevel=0.01,
                                              minDistance=12, mask=(255 - occ).astype(np.uint8))
            if corners is not None:
                for c in corners.reshape(-1, 2):
                    u, v = float(c[0]), float(c[1])
                    tracks[next_id] = {fi: (u, v)}
                    alive[next_id] = (u, v)
                    next_id += 1
        g_prev = g
    return tracks, (H, W)


def triangulate(obs, P):  # obs: list of (frame, u, v); P: dict frame->3x4
    A = []
    for f, u, v in obs:
        Pi = P[f]
        A.append(u * Pi[2] - Pi[0])
        A.append(v * Pi[2] - Pi[1])
    A = np.asarray(A)
    _, s, Vt = np.linalg.svd(A)
    X = Vt[-1]
    cond = s[-2] / max(s[-1], 1e-12)   # conditioning (higher = better-constrained)
    if abs(X[3]) < 1e-12:
        return None, 0.0, cond
    return X[:3] / X[3], float(X[3]), cond


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq_folder", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--min_obs", type=int, default=5)
    ap.add_argument("--short_span", type=int, default=20, help="frames: track span <= this = SHORT")
    ap.add_argument("--long_span", type=int, default=200, help="frames: track span >= this = LONG")
    args = ap.parse_args()
    seq = args.seq_folder
    assert cv2 is not None, "needs cv2"

    traj, focal, cx0, cy0 = load_poses_K(seq)
    n = traj.shape[0]
    fdir = frames_dir(seq)
    assert fdir, "no frames dir"
    masks = load_masks(seq, n)
    print(f"seq={seq}\nframes={n}  focal0={focal:.1f} center0=({cx0:.1f},{cy0:.1f})  frames_dir=OK")

    tracks, (H, W) = build_tracks(fdir, n, masks, stride=args.stride)
    # K scaled to tracked frame resolution (img_center implies the calib resolution)
    w0, h0 = 2 * cx0, 2 * cy0
    sx, sy = W / w0, H / h0
    K = np.array([[focal * sx, 0, cx0 * sx], [0, focal * sy, cy0 * sy], [0, 0, 1]], np.float64)

    # per-frame projection matrices P = K [R_w2c | t_w2c]
    P = {}
    for fi in range(n):
        R_c2w = quat_to_R(traj[fi, 3:7]); t_c2w = traj[fi, :3]
        R_w2c = R_c2w.T; t_w2c = -R_w2c @ t_c2w
        P[fi] = K @ np.hstack([R_w2c, t_w2c.reshape(3, 1)])

    # evaluate each track: triangulate, per-obs reprojection error, span
    rows = []  # (span, median_reproj_err, n_obs, cond)
    for tid, d in tracks.items():
        if len(d) < args.min_obs:
            continue
        obs = [(f, u, v) for f, (u, v) in sorted(d.items())]
        span = obs[-1][0] - obs[0][0]
        X, w, cond = triangulate(obs, P)
        if X is None:
            continue
        errs = []
        for f, u, v in obs:
            x = P[f] @ np.array([X[0], X[1], X[2], 1.0])
            if abs(x[2]) < 1e-9:
                continue
            errs.append(np.hypot(x[0] / x[2] - u, x[1] / x[2] - v))
        if not errs:
            continue
        rows.append((span, float(np.median(errs)), len(obs), cond))

    if not rows:
        print("no usable tracks — inconclusive"); return
    rows = np.array([(s, e, no, c) for s, e, no, c in rows], float)
    span, err = rows[:, 0], rows[:, 1]
    # parallax/conditioning gate: keep better-constrained half (drop degenerate low-parallax)
    cond = rows[:, 3]
    keep = cond >= np.median(cond)
    span, err = span[keep], err[keep]

    short = err[span <= args.short_span]
    longg = err[span >= args.long_span]
    mid = err[(span > args.short_span) & (span < args.long_span)]
    print(f"\nusable tracks={len(err)} (after parallax gate)  "
          f"spans: short(<= {args.short_span})={short.size}, mid={mid.size}, long(>= {args.long_span})={longg.size}")
    print(f"reprojection error (px), median:")
    print(f"  SHORT-span : {np.median(short):.3f}" if short.size else "  SHORT-span : (none)")
    print(f"  MID-span   : {np.median(mid):.3f}" if mid.size else "  MID-span   : (none)")
    print(f"  LONG-span  : {np.median(longg):.3f}" if longg.size else "  LONG-span  : (none)")
    if short.size and longg.size:
        ratio = float(np.median(longg) / max(np.median(short), 1e-6))
        print(f"\n  LONG/SHORT reproj-error ratio = {ratio:.2f}")
        if ratio >= 3.0:
            print("  => long-span tracks reproject MUCH worse -> DPVO trajectory DRIFTS over the clip -> s(t) justified.")
        elif ratio <= 1.5:
            print("  => long-span ~ short-span -> DPVO globally CONSISTENT (no significant drift) -> the HEAD trend")
            print("     is likely Any4D/measurement, NOT camera; s(t) on the camera would be unjustified.")
        else:
            print("  => borderline; inspect the per-span plot, try denser tracks / different clip.")
    # also report a continuous trend: reproj err vs span
    if len(err) > 10 and np.std(span) > 0:
        cc = float(np.corrcoef(span, err)[0, 1])
        print(f"  reproj-err vs span correlation = {cc:+.2f} (positive = error grows with time-span = drift)")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.scatter(span, err, s=6, alpha=0.4)
        plt.xlabel("track temporal span (frames)"); plt.ylabel("median reprojection error (px)")
        plt.title("static-point reprojection error vs track span (rising = camera drift)")
        plt.grid(alpha=0.3)
        out = os.path.join(seq, "camera_drift_reproj.png"); plt.savefig(out, dpi=120)
        print(f"saved {out}")
    except Exception as e:
        print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()
