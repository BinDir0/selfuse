#!/usr/bin/env python3
"""Whole-episode consistency: does the CAMERA (head) drift over the FULL clip, and is the
measurement even feasible? Plus a cheap hand-metric self-consistency check.

The earlier reprojection test only used LK tracks (lifetime ~<=200 frames = ~7.5% of a
2670-frame clip), so it could only prove LOCAL consistency, never whole-episode. This sweeps
the time-baseline Δ from small to the whole clip using SIFT matches on static background
(hand masked), and reports, per Δ:

  (M1-coverage) how many geometrically-consistent static matches survive at baseline Δ
                -> if this collapses to ~0 at large Δ, the camera never revisits and
                   WHOLE-EPISODE geometric drift is simply NOT measurable from images
                   (be honest, pivot to hand self-consistency).
  (M1/M2 reproj) for surviving matches, triangulate with DPVO poses and report reprojection
                error vs Δ. FLAT vs Δ -> globally consistent (no drift). RISING with Δ ->
                the trajectory drifts over the episode (and the magnitude is quantified).
                Scale-free; independent of Any4D; independent of the noisy HEAD ratio.

  (M3-hand) HaWoR shape (betas) variation over the episode = proxy for whether the recovered
            hand keeps a fixed metric size (needs no co-visibility; computable from cam_space
            alone). betas drift => HaWoR metric is unstable. (Full bone-length check needs
            MANO/torch -> run on a torch env; here we use betas as the offline proxy.)

Read-only. cv2 + numpy. Run on a finished seq_folder.
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


def quat_to_R(q):  # [qx,qy,qz,qw]
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


def hand_betas_variation(seq):
    """M3 proxy: per-frame MANO betas (shape) stability over the episode (no torch needed)."""
    per = {}
    for hd in sorted(glob.glob(os.path.join(seq, "cam_space", "*"))):
        if not os.path.isdir(hd):
            continue
        bl = []
        for jf in sorted(glob.glob(os.path.join(hd, "*.json"))):
            try:
                b = np.asarray(json.load(open(jf))["init_betas"], np.float64)[0]  # (T,10)
            except Exception:
                continue
            bl.append(b)
        if bl:
            per[os.path.basename(hd)] = np.concatenate(bl, 0)  # (sumT,10)
    out = {}
    for k, b in per.items():
        # relative std of each beta dim across the episode, summarized
        mu = np.mean(b, 0); sd = np.std(b, 0)
        out[k] = dict(frames=int(b.shape[0]),
                      beta_std_mean=float(np.mean(sd)),
                      beta_rel_std=float(np.mean(sd / (np.abs(mu) + 1e-6))),
                      beta0_range=(float(b[:, 0].min()), float(b[:, 0].max())))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq_folder", required=True)
    ap.add_argument("--anchors", type=int, default=40, help="number of anchor frames sampled across the clip")
    ap.add_argument("--ratio", type=float, default=0.8, help="Lowe ratio test")
    args = ap.parse_args()
    seq = args.seq_folder
    assert cv2 is not None

    dp = np.load(_find(seq, "SLAM/dpvo_raw_*.npz"), allow_pickle=False)
    traj = np.asarray(dp["traj"], np.float64)
    n = traj.shape[0]
    sl = np.load(_find(seq, "SLAM/hawor_slam_w_scale_*.npz"), allow_pickle=True)
    focal = float(sl["img_focal"]); cx0, cy0 = [float(v) for v in np.asarray(sl["img_center"]).reshape(-1)[:2]]
    fdir = frames_dir(seq); assert fdir, "no frames dir"
    masks = load_masks(seq, n)
    print(f"seq={seq}\nframes={n} focal0={focal:.1f} center0=({cx0:.1f},{cy0:.1f})")

    # --- M3 hand betas self-consistency (cheap, no co-visibility) ---
    print("\n[M3] HaWoR hand shape (betas) self-consistency over the episode (proxy for fixed hand size):")
    for k, v in hand_betas_variation(seq).items():
        print(f"  hand {k}: frames={v['frames']}  beta_std(mean over dims)={v['beta_std_mean']:.4f}  "
              f"rel_std={v['beta_rel_std']:.3f}  beta0_range=({v['beta0_range'][0]:.2f},{v['beta0_range'][1]:.2f})")
    print("  (beta_std ~0 => HaWoR keeps a fixed hand shape/size => hand metric internally stable)")

    # --- SIFT features on sampled frames ---
    sift = cv2.SIFT_create(nfeatures=1200)
    anchors = sorted(set(np.linspace(0, n - 1, args.anchors).astype(int).tolist()))
    feats = {}
    hw = {}

    def get_feats(fi):
        if fi in feats:
            return feats[fi]
        img = cv2.imread(os.path.join(fdir, f"{fi:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            feats[fi] = (None, None); return feats[fi]
        if "h" not in hw:
            hw["h"], hw["w"] = img.shape[:2]
        mk = None
        if masks is not None and fi < masks.shape[0]:
            mm = cv2.resize(masks[fi].astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            mk = (255 * (mm == 0)).astype(np.uint8)  # SIFT only on background (exclude hand)
        kp, des = sift.detectAndCompute(img, mk)
        feats[fi] = (kp, des)
        return feats[fi]

    bf = cv2.BFMatcher(cv2.NORM_L2)

    def build_K():
        w0, h0 = 2 * cx0, 2 * cy0
        sx, sy = hw["w"] / w0, hw["h"] / h0
        return np.array([[focal * sx, 0, cx0 * sx], [0, focal * sy, cy0 * sy], [0, 0, 1]], np.float64)

    def pose_w2c(fi):
        R_c2w = quat_to_R(traj[fi, 3:7]); t_c2w = traj[fi, :3]
        R = R_c2w.T
        return R, (-R @ t_c2w)  # R_w2c, t_w2c

    def skew(t):
        return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], np.float64)

    def dpvo_F(a, b, Kinv):
        """Fundamental matrix implied by DPVO relative pose A->B. Scale-free (E uses t direction)."""
        Ra, ta = pose_w2c(a); Rb, tb = pose_w2c(b)
        R_rel = Rb @ Ra.T
        t_rel = tb - R_rel @ ta
        nt = np.linalg.norm(t_rel)
        if nt < 1e-9:
            return None
        E = skew(t_rel / nt) @ R_rel
        return Kinv.T @ E @ Kinv, (Ra, ta, Rb, tb)

    def dpvo_H(a, b, K, Kinv):
        """Infinite homography K R_rel K^-1 (a->b). The CORRECT consistency model when the camera
        is ROTATION-dominated (near-zero translation => no parallax => epipolar/triangulation degenerate).
        Tests whether DPVO's ROTATION is globally consistent; residual rises with Δ => rotation drift."""
        Ra, _ = pose_w2c(a); Rb, _ = pose_w2c(b)
        return K @ (Rb @ Ra.T) @ Kinv

    def homog_err(H, ptsA, ptsB):
        a = np.hstack([ptsA, np.ones((len(ptsA), 1))])
        p = a @ H.T
        p = p[:, :2] / p[:, 2:3]
        return np.linalg.norm(p - ptsB, axis=1)

    def sampson(F, ptsA, ptsB):
        """Symmetric Sampson distance (px) of correspondences to F (per-point)."""
        a = np.hstack([ptsA, np.ones((len(ptsA), 1))])
        b = np.hstack([ptsB, np.ones((len(ptsB), 1))])
        Fa = a @ F.T          # F x_A
        Ftb = b @ F            # F^T x_B
        num = np.einsum("ij,ij->i", b, Fa) ** 2
        den = Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Ftb[:, 0] ** 2 + Ftb[:, 1] ** 2 + 1e-12
        return np.sqrt(num / den)

    def triangulate_reproj(uvA, uvB, K, Kinv, pose):
        """Cheirality + parallax gated 2-view reprojection error (px). None if degenerate."""
        Ra, ta, Rb, tb = pose
        PA = K @ np.hstack([Ra, ta.reshape(3, 1)])
        PB = K @ np.hstack([Rb, tb.reshape(3, 1)])
        A = np.vstack([uvA[0] * PA[2] - PA[0], uvA[1] * PA[2] - PA[1],
                       uvB[0] * PB[2] - PB[0], uvB[1] * PB[2] - PB[1]])
        _, _, Vt = np.linalg.svd(A); X = Vt[-1]
        if abs(X[3]) < 1e-12:
            return None
        Xw = X[:3] / X[3]
        if (Ra @ Xw + ta)[2] <= 0 or (Rb @ Xw + tb)[2] <= 0:   # cheirality
            return None
        dA = Ra.T @ (Kinv @ np.array([uvA[0], uvA[1], 1.0])); dA /= np.linalg.norm(dA)
        dB = Rb.T @ (Kinv @ np.array([uvB[0], uvB[1], 1.0])); dB /= np.linalg.norm(dB)
        if np.degrees(np.arccos(np.clip(abs(dA @ dB), -1, 1))) < 1.0:   # parallax gate
            return None
        errs = []
        for P, uv in ((PA, uvA), (PB, uvB)):
            x = P @ np.array([Xw[0], Xw[1], Xw[2], 1.0])
            if abs(x[2]) > 1e-9:
                errs.append(np.hypot(x[0] / x[2] - uv[0], x[1] / x[2] - uv[1]))
        return float(np.mean(errs)) if errs else None

    # --- sweep baseline Δ ---
    deltas = [d for d in [1, 5, 15, 30, 60, 120, 240, 480, 960, 1440, 1920] if d < n]
    MIN_INL = 8
    get_feats(anchors[0])           # ensure hw populated before build_K
    K = build_K(); Kinv = np.linalg.inv(K)
    # --- motion regime: is the camera rotation-dominated (near-zero translation)? ---
    # Parallax ~ |Δt| / scene_depth. If tiny, epipolar/triangulation are DEGENERATE (no parallax)
    # and only the homography (rotation) test is valid. All in raw DPVO units (scale-invariant ratio).
    tpath = float(np.linalg.norm(np.diff(traj[:, :3], axis=0), axis=1).sum())
    tspan = float(np.linalg.norm(traj[:, :3].max(0) - traj[:, :3].min(0)))
    scene_d = float(np.median(1.0 / np.clip(np.load(_find(seq, "SLAM/hawor_slam_w_scale_*.npz"),
                                                    allow_pickle=True)["disps"], 1e-9, None)))
    parallax = tspan / max(scene_d, 1e-9)
    rot_dom = parallax < 0.05
    print(f"\n[regime] camera translation path={tpath:.1f} span={tspan:.1f} (raw) vs scene depth~{scene_d:.1f} "
          f"=> parallax≈{100*parallax:.1f}%  {'=> ROTATION-DOMINATED: trust HOMOG, epipolar/tri are DEGENERATE' if rot_dom else '=> translation present: epipolar/tri valid'}")
    print(f"\n[M1] consistency vs time-baseline Δ  (anchors={len(anchors)}, MIN_INL={MIN_INL})")
    print("  homog_px=rotation-consistency (valid under low parallax); sampson/reproj need translation (degenerate if rotation-dominated)")
    print(f"  {'Δ(frames)':>10}{'pairs':>7}{'med_inl':>9}{'homog_px':>10}{'sampson_px':>12}{'reproj_px':>11}")
    rows = []
    for d in deltas:
        inl_counts, samp_meds, reproj, tri_rates, homogs = [], [], [], [], []
        for a in anchors:
            b = a + d
            if b >= n:
                continue
            kpA, desA = get_feats(a); kpB, desB = get_feats(b)
            if desA is None or desB is None or len(kpA) < MIN_INL or len(kpB) < MIN_INL:
                continue
            m = bf.knnMatch(desA, desB, k=2)
            good = [p for p, q in (mm for mm in m if len(mm) == 2) if p.distance < args.ratio * q.distance]
            if len(good) < MIN_INL:
                inl_counts.append(0); continue
            ptsA = np.float64([kpA[g.queryIdx].pt for g in good])
            ptsB = np.float64([kpB[g.trainIdx].pt for g in good])
            # filter correspondences with the model VALID for the regime: homography when
            # rotation-dominated (F is degenerate at zero parallax), else fundamental matrix.
            if rot_dom:
                _, mask_in = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 3.0)
            else:
                _, mask_in = cv2.findFundamentalMat(ptsA, ptsB, cv2.FM_RANSAC, 1.5, 0.999)
            if mask_in is None:
                inl_counts.append(0); continue
            inl = mask_in.ravel().astype(bool)
            inl_counts.append(int(inl.sum()))
            if inl.sum() < MIN_INL:
                continue
            pa, pb = ptsA[inl], ptsB[inl]      # data-consistent good correspondences
            # rotation-consistency (valid even at zero parallax): does DPVO's rotation transfer a->b?
            homogs.append(float(np.median(homog_err(dpvo_H(a, b, K, Kinv), pa, pb))))
            fd = dpvo_F(a, b, Kinv)
            if fd is None:
                continue
            Fdp, pose = fd
            # scale-free: do DPVO poses explain these good correspondences? (needs translation)
            samp_meds.append(float(np.median(sampson(Fdp, pa, pb))))
            # full (incl. scale): cheirality + parallax gated triangulation reprojection (needs translation)
            es = [e for e in (triangulate_reproj(ua, ub, K, Kinv, pose) for ua, ub in zip(pa, pb)) if e is not None]
            if es:
                reproj.append(float(np.median(es)))
                tri_rates.append(len(es) / len(pa))
        med_inl = float(np.median(inl_counts)) if inl_counts else 0.0
        med_homog = float(np.median(homogs)) if homogs else float("nan")
        med_samp = float(np.median(samp_meds)) if samp_meds else float("nan")
        med_rep = float(np.median(reproj)) if reproj else float("nan")
        rows.append((d, len(inl_counts), med_inl, med_homog, med_samp, med_rep))
        print(f"  {d:>10}{len(inl_counts):>7}{med_inl:>9.0f}{med_homog:>10.3f}{med_samp:>12.3f}{med_rep:>11.2f}")

    # --- verdict ---
    rr = np.array(rows, float)   # cols: d, pairs, med_inl, homog, sampson, reproj

    def trend(col, name, rising_msg, flat_msg):
        v = rr[np.isfinite(rr[:, col])]
        if v.shape[0] < 3:
            return
        sh = v[v[:, 0] <= 30, col]; lo = v[v[:, 0] >= max(60, 0.3 * (n - 1)), col]
        if sh.size and lo.size:
            ratio = np.median(lo) / max(np.median(sh), 1e-6)
            print(f"  [{name}] short-Δ {np.median(sh):.2f}px vs long-Δ {np.median(lo):.2f}px ratio={ratio:.2f}"
                  f"  => {rising_msg if ratio >= 2 else flat_msg}")

    print("\n=== VERDICT ===")
    usable = rr[rr[:, 2] >= 8]
    max_cov_delta = int(usable[-1, 0]) if usable.size else 0
    print(f"  co-visibility usable (median inliers>=8) up to Δ≈{max_cov_delta} frames "
          f"({100*max_cov_delta/max(n-1,1):.0f}% of clip).")
    # ROTATION consistency (homography) — the VALID metric when rotation-dominated
    trend(3, "ROTATION (homog)", "ROTATION DRIFTS over episode", "rotation globally consistent")
    if rot_dom:
        print("  ^ camera is ROTATION-DOMINATED => the homog row above is the trustworthy verdict.")
        print("    The sampson/reproj rows below are DEGENERATE here (no parallax) — IGNORE them.")
    trend(4, "scale-free Sampson (needs translation)", "pose-direction inconsistent", "consistent")
    trend(5, "full+scale reproj (needs translation)", "rising", "flat")
    if rot_dom:
        print("  => If rotation is consistent but the rerun cloud still blobs, the smear is DEPTH "
              "inconsistency (Any4D across frames), NOT camera drift. Different fix.")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(20, 4))
        ax[0].plot(rr[:, 0], rr[:, 2], "o-"); ax[0].set_ylabel("median inlier matches")
        ax[0].set_title("co-visibility vs Δ")
        ax[1].plot(rr[:, 0], rr[:, 3], "o-", color="#16a34a"); ax[1].set_ylabel("median homography err (px)")
        ax[1].set_title("ROTATION consistency (valid @ low parallax)")
        ax[2].plot(rr[:, 0], rr[:, 4], "o-", color="#2563eb"); ax[2].set_ylabel("median Sampson (px)")
        ax[2].set_title("scale-free pose-dir drift (needs translation)")
        ax[3].plot(rr[:, 0], rr[:, 5], "o-", color="#e11d48"); ax[3].set_ylabel("median reproj err (px)")
        ax[3].set_title("full (+scale) drift (needs translation)")
        for a in ax:
            a.set_xlabel("Δ (frames)"); a.set_xscale("log"); a.grid(alpha=.3)
        out = os.path.join(seq, "episode_consistency.png"); fig.tight_layout(); fig.savefig(out, dpi=120)
        print(f"saved {out}")
    except Exception as e:
        print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()
