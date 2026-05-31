#!/usr/bin/env python3
"""C.0 prototype — global rotation averaging to undo DPVO's accumulated rotation drift.

Why: this episode is ROTATION-DOMINATED (parallax ≈0.1%, ~1cm total translation). DPVO's
per-step rotation is excellent (~2px homog @ Δ=1) but the steps accumulate (~730px @ Δ=960).
Loop-closure was discarded (no positional revisits) — but the camera DOES revisit viewing
directions while panning, so we DO have long-baseline static-background co-visibility
(36–72% on this clip). Those long-Δ correspondences give long-range relative-rotation
CONSTRAINTS that DPVO's local windowed BA never uses. Global rotation averaging over them
should pull the drifted DPVO rotations back into global consistency.

Pipeline (offline, read-only):
  1. Load DPVO traj (c2w quats) + K + frames + hand masks.
  2. For each Δ in a sweep, sample anchor pairs; on static background, SIFT-match +
     findHomography RANSAC -> H_data. Decompose H_data to a clean relative rotation:
        M = K^-1 H_data K   ;   SVD M = U Σ V^T   ;   R_ab^data = U · diag(1,1,det(UV^T)) · V^T
     (Σ ≈ (1,1,1) confirms pure-rotation assumption is good; we filter on it.)
     R_ab^data is the cam_a -> cam_b rotation that maps vectors expressed in cam_a frame to
     their cam_b-frame coordinates; equivalently in c2w terms: R_a = R_b · R_ab^data, i.e.,
     R_b = R_a · R_ab^data^T.
  3. Also add DPVO's own consecutive (Δ=1) rotations as edges — locally accurate, the priors.
  4. Initialize per-frame rotation R*_i from DPVO; iterate robust rotation averaging
     (weighted quaternion mean of per-edge predictions, with Geman-McClure weight on angular
     residual). Converges in ~20 iters typically.
  5. Re-measure homog-vs-Δ with R* in place of DPVO rotations. Success if long-Δ homog
     drops to < 20px (was ~700+) and Δ=1 stays < 5px (was ~2px → don't break local).

Success ⇒ build the production C.1 module that injects R* back into traj_dense before save.
Run: python scripts/prototype_rotation_averaging.py --seq_folder <stage_outputs>/<clip>
"""
import argparse
import glob
import os

import cv2
import numpy as np


# ---------- I/O helpers (mirrored from diagnose_episode_consistency.py) ----------

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


def R_to_quat(R):  # returns [qx,qy,qz,qw], shortest-arc convention (w>=0)
    R = np.asarray(R, np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = (tr + 1.0) ** 0.5 * 2
        qw = 0.25 * s; qx = (R[2, 1] - R[1, 2]) / s; qy = (R[0, 2] - R[2, 0]) / s; qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5 * 2
        qw = (R[2, 1] - R[1, 2]) / s; qx = 0.25 * s; qy = (R[0, 1] + R[1, 0]) / s; qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5 * 2
        qw = (R[0, 2] - R[2, 0]) / s; qx = (R[0, 1] + R[1, 0]) / s; qy = 0.25 * s; qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5 * 2
        qw = (R[1, 0] - R[0, 1]) / s; qx = (R[0, 2] + R[2, 0]) / s; qy = (R[1, 2] + R[2, 1]) / s; qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], np.float64)
    if q[3] < 0:
        q = -q
    return q / (np.linalg.norm(q) + 1e-12)


def geodesic_deg(R1, R2):
    c = (np.trace(R1.T @ R2) - 1.0) * 0.5
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))


def quat_weighted_average(quats, weights, ref_quat):
    """Weighted quaternion mean (sign-aligned to ref). Numpy-only; uses Markley eigen method."""
    if not quats:
        return ref_quat
    Q = np.stack(quats, 0)
    w = np.asarray(weights, np.float64)
    # align signs so each q has dot(q, ref) >= 0 (shortest arc)
    s = np.sign(Q @ ref_quat)
    s[s == 0] = 1
    Q = Q * s[:, None]
    M = np.einsum("n,ni,nj->ij", w, Q, Q)
    vals, vecs = np.linalg.eigh(M)
    q = vecs[:, -1]
    if q @ ref_quat < 0:
        q = -q
    return q / (np.linalg.norm(q) + 1e-12)


def orthonormalize_R(M):
    """Find closest rotation to M via SVD with det correction. Also returns σ ratio of M."""
    U, S, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.linalg.det(U @ Vt)
    return U @ D @ Vt, float(S.min() / (S.max() + 1e-12))


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


# ---------- Edge extraction (relative rotation observations) ----------

def extract_data_edges(traj, focal, cx0, cy0, fdir, masks, deltas, anchors_per_delta,
                       sift_n=1200, ratio=0.8, min_inliers=12, sigma_ratio_max=0.3):
    """SIFT static-bg match + RANSAC homography + H -> R decomposition. One edge per pair.

    Returns list of (a, b, R_ab^data, weight, info dict). Filters out pairs where the
    homography's singular-value ratio is too off (rejecting bad pure-rotation fits).
    """
    n = traj.shape[0]
    sift = cv2.SIFT_create(nfeatures=sift_n)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    feats = {}; hw = {}

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
            mk = (255 * (mm == 0)).astype(np.uint8)
        kp, des = sift.detectAndCompute(img, mk)
        feats[fi] = (kp, des)
        return feats[fi]

    # warm up to fix resolution
    for a in np.linspace(0, n - 1, min(10, n)).astype(int):
        if get_feats(int(a))[0] is not None:
            break

    if "h" not in hw:
        return [], None
    w0, h0 = 2 * cx0, 2 * cy0
    sx, sy = hw["w"] / w0, hw["h"] / h0
    K = np.array([[focal * sx, 0, cx0 * sx], [0, focal * sy, cy0 * sy], [0, 0, 1]], np.float64)
    Kinv = np.linalg.inv(K)

    edges = []
    n_tried = n_kept = n_rej_sigma = 0
    for d in deltas:
        anchors = sorted(set(np.linspace(0, n - 1 - d, anchors_per_delta).astype(int).tolist()))
        for a in anchors:
            b = a + d
            if b >= n:
                continue
            n_tried += 1
            kpA, desA = get_feats(int(a)); kpB, desB = get_feats(int(b))
            if desA is None or desB is None or len(kpA) < min_inliers or len(kpB) < min_inliers:
                continue
            mm = bf.knnMatch(desA, desB, k=2)
            good = [p for p, q in (mp for mp in mm if len(mp) == 2) if p.distance < ratio * q.distance]
            if len(good) < min_inliers:
                continue
            ptsA = np.float64([kpA[g.queryIdx].pt for g in good])
            ptsB = np.float64([kpB[g.trainIdx].pt for g in good])
            H, mask_in = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 3.0)
            if H is None or mask_in is None:
                continue
            inl = mask_in.ravel().astype(bool)
            n_inl = int(inl.sum())
            if n_inl < min_inliers:
                continue
            # decompose: M = K^-1 H K should be ~rotation if motion is rotation-dominated
            M = Kinv @ H @ K
            R, sigma_ratio = orthonormalize_R(M)
            # σ_min/σ_max far from 1 ⇒ H not well explained by pure rotation
            if abs(1.0 - sigma_ratio) > sigma_ratio_max:
                n_rej_sigma += 1
                continue
            # residual under fitted R (sanity): how well does R reproduce the inlier flow?
            Hr = K @ R @ Kinv
            ones = np.ones((n_inl, 1))
            aH = np.hstack([ptsA[inl], ones]) @ Hr.T
            aH = aH[:, :2] / aH[:, 2:3]
            res = float(np.median(np.linalg.norm(aH - ptsB[inl], axis=1)))
            # store inlier points too — needed for the non-circular validation below
            edges.append((int(a), int(b), R, float(n_inl),
                          {"res_data": res, "sigma_ratio": sigma_ratio,
                           "ptsA": ptsA[inl].copy(), "ptsB": ptsB[inl].copy()}))
            n_kept += 1
    info = {"tried": n_tried, "kept": n_kept, "rejected_sigma": n_rej_sigma}
    return edges, (K, info)


def add_dpvo_consecutive_edges(traj, weight=4.0):
    """DPVO's per-step rotation is locally accurate (~2px homog) — use as priors."""
    n = traj.shape[0]
    edges = []
    Rs = np.stack([quat_to_R(traj[i, 3:7]) for i in range(n)], 0)  # c2w
    for i in range(n - 1):
        # R_ab (cam_a -> cam_b) = R_b^T R_a   (c2w convention; see header)
        R_ab = Rs[i + 1].T @ Rs[i]
        edges.append((int(i), int(i + 1), R_ab, float(weight), {"src": "dpvo_step"}))
    return edges, Rs


# ---------- Robust global rotation averaging ----------

def rotation_average(R_init, edges, n_iters=30, sigma_deg=2.0, tol_deg=0.01):
    """Iterative weighted-quaternion rotation averaging with Geman-McClure robust weights
    on per-edge geodesic residual.

    Edge convention (a, b, R_ab) means R_ab maps a vector in cam_a frame to its cam_b-frame
    coordinates; in c2w terms R_b = R_a · R_ab^T  (so R_a = R_b · R_ab).
    """
    N = R_init.shape[0]
    Q = np.stack([R_to_quat(R_init[i]) for i in range(N)], 0)
    # adjacency: for each frame, the edges that touch it, with predicted-R callable
    adj = [[] for _ in range(N)]
    for (a, b, R_ab, w, _info) in edges:
        adj[a].append((b, R_ab, w, "a"))   # to predict R_a we need R_b: R_a = R_b · R_ab
        adj[b].append((a, R_ab, w, "b"))   # to predict R_b we need R_a: R_b = R_a · R_ab^T

    history = []
    for it in range(n_iters):
        Q_new = Q.copy()
        max_change = 0.0
        for i in range(N):
            if not adj[i]:
                continue
            preds_q, weights = [], []
            R_self = quat_to_R(Q[i])
            for (j, R_ab, w_edge, role) in adj[i]:
                R_j = quat_to_R(Q[j])
                R_pred = R_j @ R_ab if role == "a" else R_j @ R_ab.T
                ang = geodesic_deg(R_self, R_pred)
                # Geman-McClure robust weight on geodesic residual
                w_rob = sigma_deg ** 2 / (sigma_deg ** 2 + ang ** 2)
                preds_q.append(R_to_quat(R_pred))
                weights.append(w_edge * w_rob)
            if sum(weights) <= 0:
                continue
            q_new = quat_weighted_average(preds_q, weights, Q[i])
            ch = geodesic_deg(R_self, quat_to_R(q_new))
            max_change = max(max_change, ch)
            Q_new[i] = q_new
        Q = Q_new
        history.append(max_change)
        if max_change < tol_deg:
            break
    R_out = np.stack([quat_to_R(Q[i]) for i in range(N)], 0)
    return R_out, Q, history


# ---------- Re-measure homog vs Δ with corrected rotations ----------

def measure_homog_px(R_set, edges_by_pair, K, Kinv):
    """The NON-CIRCULAR validation: for each kept pair, compute the homography
    H_pred(R_set) = K · (R_b^T R_a) · K^-1, transfer the original inlier ptsA -> predicted
    ptsB, measure pixel error against actual ptsB. Bin by Δ. This is the SAME metric
    `scripts/diagnose_episode_consistency.py` measures as "homog_px", just done with R_set
    substituted for DPVO's rotations. Robust median per pair, then median over pairs per Δ.
    """
    by_delta = {}
    for (a, b, _R_ab_data, _w, info) in edges_by_pair:
        ptsA = info.get("ptsA"); ptsB = info.get("ptsB")
        if ptsA is None or ptsB is None or len(ptsA) < 4:
            continue
        H = K @ (R_set[b].T @ R_set[a]) @ Kinv
        a_h = np.hstack([ptsA, np.ones((len(ptsA), 1))]) @ H.T
        a_h = a_h[:, :2] / a_h[:, 2:3]
        per_match = np.linalg.norm(a_h - ptsB, axis=1)
        delta = b - a
        by_delta.setdefault(delta, []).append(float(np.median(per_match)))
    rows = []
    for d in sorted(by_delta):
        v = np.array(by_delta[d])
        rows.append((d, len(v), float(np.median(v)), float(np.percentile(v, 25)),
                     float(np.percentile(v, 75))))
    return rows


def report_homog_table(name, rows):
    print(f"\n[{name}]  {'Δ':>6}{'pairs':>7}{'med_homog_px':>14}{'p25':>8}{'p75':>8}")
    for d, n, med, p25, p75 in rows:
        print(f"          {d:>6}{n:>7}{med:>14.2f}{p25:>8.2f}{p75:>8.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq_folder", required=True)
    ap.add_argument("--anchors", type=int, default=30, help="anchor pairs per Δ")
    ap.add_argument("--deltas", type=str, default="5,15,30,60,120,240,480,960,1440")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--sigma_deg", type=float, default=2.0, help="Geman-McClure scale on geodesic deg")
    ap.add_argument("--dpvo_step_weight", type=float, default=4.0)
    ap.add_argument("--sigma_ratio_max", type=float, default=0.3,
                    help="reject pair if |1 - σmin/σmax| > this in H decomposition (0.99 = effectively off)")
    args = ap.parse_args()

    seq = args.seq_folder
    dp = np.load(_find(seq, "SLAM/dpvo_raw_*.npz"), allow_pickle=False)
    traj = np.asarray(dp["traj"], np.float64)
    sl = np.load(_find(seq, "SLAM/hawor_slam_w_scale_*.npz"), allow_pickle=True)
    focal = float(sl["img_focal"]); cx0, cy0 = [float(v) for v in np.asarray(sl["img_center"]).reshape(-1)[:2]]
    fdir = frames_dir(seq); assert fdir, "no frames dir"
    n = traj.shape[0]
    masks = load_masks(seq, n)
    print(f"seq={seq}\nframes={n} focal0={focal:.1f} center0=({cx0:.1f},{cy0:.1f})")

    deltas = [int(x) for x in args.deltas.split(",") if int(x) < n]
    print(f"\n[C.0 prototype] extracting long-Δ relative-rotation edges via SIFT+H ...")
    edges_data, build = extract_data_edges(traj, focal, cx0, cy0, fdir, masks, deltas, args.anchors,
                                           sigma_ratio_max=args.sigma_ratio_max)
    if build is None or not edges_data:
        print("  no edges extracted; abort"); return
    K, info = build
    print(f"  tried={info['tried']} kept={info['kept']} rejected_for_sigma_ratio={info['rejected_sigma']}")

    edges_step, R_dpvo = add_dpvo_consecutive_edges(traj, weight=args.dpvo_step_weight)
    edges_all = edges_step + edges_data

    Kinv = np.linalg.inv(K)
    # BEFORE: real homography-pixel error (the same metric as diagnose_episode_consistency.py)
    print("\n[BEFORE] homography pixel error with DPVO rotations  (this IS the drift metric):")
    before_rows = measure_homog_px(R_dpvo, edges_data, K, Kinv)
    report_homog_table("DPVO", before_rows)

    # Solve
    print(f"\n[averaging] {n} frames, {len(edges_all)} edges (step={len(edges_step)}, data={len(edges_data)}), "
          f"sigma_deg={args.sigma_deg}, dpvo_step_weight={args.dpvo_step_weight}")
    R_star, Q_star, history = rotation_average(R_dpvo, edges_all, n_iters=args.iters,
                                               sigma_deg=args.sigma_deg)
    print(f"  iterations: {len(history)}, max-change history: "
          f"[{', '.join(f'{h:.3f}' for h in history[:5])}, ..., {history[-1]:.4f}]")
    diffs = np.array([geodesic_deg(R_dpvo[i], R_star[i]) for i in range(n)])
    print(f"  R*-vs-DPVO geodesic diff (deg): median={np.median(diffs):.3f} p95={np.percentile(diffs,95):.3f} "
          f"max={diffs.max():.3f}")

    # AFTER: same metric with R*. NOT circular: the averaging optimized rotation-CONSISTENCY
    # of relative R_ab^data (an orthonormalized projection of H_data). homog_px here measures
    # whether H_pred(R*) reproduces the ACTUAL inlier ptsA->ptsB transfer in pixels — a stricter,
    # observation-side metric, the very one we use to detect drift in the diagnose script.
    print("\n[AFTER] homography pixel error with R* (rotation averaged):")
    after_rows = measure_homog_px(R_star, edges_data, K, Kinv)
    report_homog_table("R*", after_rows)

    print("\n=== VERDICT (C.0) ===")
    def m_of(rows, lo, hi):
        v = [r[2] for r in rows if lo <= r[0] <= hi]
        return float(np.median(v)) if v else float("nan")
    sb, sa = m_of(before_rows, 0, 30), m_of(after_rows, 0, 30)
    lb, la = m_of(before_rows, 60, 9999), m_of(after_rows, 60, 9999)
    print(f"  homog_px median   short-Δ (≤30):  DPVO={sb:.2f} -> R*={sa:.2f}   "
          f"({sa/max(sb,1e-6):.2f}×)")
    print(f"  homog_px median   long-Δ  (≥60):  DPVO={lb:.2f} -> R*={la:.2f}   "
          f"({la/max(lb,1e-6):.2f}×)")
    drop = la / max(lb, 1e-6)
    if not np.isnan(la) and la < 20 and drop < 0.3 and (np.isnan(sa) or sa < 8):
        print("  ✓ Long-Δ homog px DROPPED significantly AND short-Δ stayed clean")
        print("    => rotation averaging WORKS on this clip. Premise validated.")
    else:
        print("  ✗ Insufficient drop or local accuracy broken — diagnose: edge density, "
              "weights, sigma. Inspect rows above; possibly retune dpvo_step_weight + sigma_deg.")
    print("  (Next: integrate R* into slam.py:935 as C.1; the plumbing is trivial — write Q_star into")
    print("   traj_dense[:, 3:7] before _save_slam_outputs.)")

    np.savez(os.path.join(seq, "rot_avg_prototype.npz"),
             R_dpvo=R_dpvo.astype(np.float32),
             R_star=R_star.astype(np.float32),
             Q_star_xyzw=Q_star.astype(np.float32),
             history=np.asarray(history, np.float32))
    print(f"  saved: {os.path.join(seq, 'rot_avg_prototype.npz')}")


if __name__ == "__main__":
    main()
