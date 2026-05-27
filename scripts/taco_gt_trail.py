#!/usr/bin/env python3
"""Dump a HaWoR-Figure-1 trail .npz from TACO ground truth.

TACO ships clean GT (MANO params + GT camera extrinsics), so this makes a much
nicer teaser than running SLAM. It mirrors scripts/eval_compare/gt_adapters/taco.py
for loading, but runs MANO FK to get VERTICES (not just joints) so we can draw
hand meshes, plus camera frustums along the GT trajectory.

Output is the SAME .npz schema as `render_world_traj.py --dump_npz`, so convert
it to a viewable .rrd with the same standalone converter (in a rerun-only env):

  # hawor env (needs torch + MANO models in _DATA):
  python scripts/taco_gt_trail.py \
      --data_root /path/to/TACO --seq_id "(dust, roller, pan)/20230927_032" \
      --dump_npz /tmp/taco_trail.npz --num_samples 8

  # rerun-only env:
  python scripts/rrd_from_npz.py /tmp/taco_trail.npz /tmp/taco_hand_cam.rrd
  rerun /tmp/taco_hand_cam.rrd

seq_id is "<triplet>/<seq>", e.g. "(dust, roller, pan)/20230927_032". List them
with: ls "<root>/Hand_Poses".
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# same palm-fill triangles render_world_traj uses to close the right-hand mesh
_PALM = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
                  [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
                  [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
                  [120, 108, 78], [78, 108, 79]])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True, help="TACO root (has Hand_Poses/, Egocentric_Camera_Parameters/)")
    p.add_argument("--seq_id", required=True, help='"<triplet>/<seq>", e.g. "(dust, roller, pan)/20230927_032"')
    p.add_argument("--dump_npz", required=True, help="output trail .npz (feed to scripts/rrd_from_npz.py)")
    p.add_argument("--num_samples", type=int, default=8,
                   help="target # hand poses to show, chosen from the WHOLE episode to minimise "
                        "pairwise visual (2D) overlap -- 'just non-overlapping', not maxed apart")
    p.add_argument("--fps", type=float, default=30.0, help="for logging only")
    p.add_argument("--frame_start", type=int, default=0, help="optional crop start frame")
    p.add_argument("--frame_end", type=int, default=-1, help="optional crop end frame (-1 = last)")
    p.add_argument("--layout", choices=["sequence", "scatter", "taco"], default="sequence",
                   help="sequence = time-ordered hand sequence along a gentle arc that reads as an "
                        "action unfolding (default; both hands per step, temporal fade). "
                        "scatter = random-ish jittered grid of real poses. "
                        "taco = keep real TACO positions, low-overlap subset.")
    p.add_argument("--arc", type=float, default=0.25,
                   help="[sequence] arc rise as a fraction of the row width (0 = straight row)")
    # --- taco layout knobs ---
    p.add_argument("--spread", type=float, default=0.6,
                   help="[taco] 2D scatter: 0 = compact/central, 1 = maximally spread.")
    p.add_argument("--overlap", type=float, default=0.25,
                   help="[taco] max allowed pairwise 2D bbox IoU between kept poses")
    # --- scatter layout knobs ---
    p.add_argument("--gap", type=float, default=0.3,
                   help="[scatter] spacing between hands as a fraction of hand size (bigger = more spread)")
    p.add_argument("--jitter_rot", type=float, default=20.0,
                   help="[scatter] random in-plane rotation per hand, degrees (variety)")
    p.add_argument("--depth_jitter", type=float, default=0.15,
                   help="[scatter] random depth offset as a fraction of hand size (non-flat look)")
    p.add_argument("--seed", type=int, default=0, help="[scatter] RNG seed for positions/pose picks")
    p.add_argument("--hands", choices=["both", "left", "right"], default="both")
    p.add_argument("--no_fade", action="store_true", help="solid colour instead of fade-to-white")
    p.add_argument("--alpha_min", type=float, default=0.45,
                   help="oldest-pose colour strength, blended toward white (lower = lighter/more faded)")
    p.add_argument("--camera", action="store_true",
                   help="also draw camera frustums + trajectory (off by default)")
    p.add_argument("--ground", action="store_true",
                   help="add a checkerboard ground (off by default; TACO world up-axis is not assumed)")
    p.add_argument("--frustum_radius", type=float, default=0.05)
    p.add_argument("--frustum_height", type=float, default=0.10)
    p.add_argument("--cpu", action="store_true", help="run MANO on CPU")
    return p.parse_args()


def _to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _load_camera(data_root, triplet, seq):
    cam_dir = os.path.join(data_root, "Egocentric_Camera_Parameters", triplet, seq)
    extr = np.load(os.path.join(cam_dir, "egocentric_frame_extrinsic.npy"))  # (N,4,4) world->cam
    R_w2c = extr[:, :3, :3].astype(np.float64)
    t_w2c = extr[:, :3, 3].astype(np.float64)
    R_c2w = np.transpose(R_w2c, (0, 2, 1))
    cam_pos = -np.einsum("tij,tj->ti", R_c2w, t_w2c)  # camera centre in world
    return R_c2w, cam_pos, extr.shape[0]


def _load_hand_verts(data_root, triplet, seq, T, use_cuda):
    """World-frame MANO vertices (and validity) per hand, mirroring taco.py's
    param loading + wrist-centring correction, but keeping vertices."""
    import torch

    from hawor.utils.process import run_mano, run_mano_left

    hp_dir = os.path.join(data_root, "Hand_Poses", triplet, seq)
    out = {}
    for name, is_right in (("left", False), ("right", True)):
        verts = np.zeros((T, 778, 3), np.float32)
        valid = np.zeros(T, bool)
        pose_pkl = os.path.join(hp_dir, f"{name}_hand.pkl")
        if not os.path.exists(pose_pkl):
            out[name] = (verts, valid)
            continue
        with open(pose_pkl, "rb") as f:
            data = pickle.load(f)
        shape = np.zeros(10, np.float32)
        shape_pkl = os.path.join(hp_dir, f"{name}_hand_shape.pkl")
        if os.path.exists(shape_pkl):
            with open(shape_pkl, "rb") as f:
                sd = pickle.load(f)
            shape = _to_np(sd["hand_shape"] if isinstance(sd, dict) and "hand_shape" in sd else sd).reshape(10)

        keys = sorted(data.keys())  # sorted position aligns to ego frame order
        g_aa = np.zeros((T, 3), np.float32); pose_aa = np.zeros((T, 45), np.float32)
        tsl = np.zeros((T, 3), np.float32); betas = np.tile(shape, (T, 1)).astype(np.float32)
        present = np.zeros(T, bool)
        for i in range(min(T, len(keys))):
            entry = data[keys[i]]
            full = _to_np(entry["hand_pose"]).reshape(-1)  # 48 aa (1 global + 15)
            g_aa[i] = full[:3]; pose_aa[i] = full[3:48]
            tsl[i] = _to_np(entry["hand_trans"]).reshape(3)
            present[i] = True

        fk = run_mano if is_right else run_mano_left
        res = fk(torch.as_tensor(tsl)[None], torch.as_tensor(g_aa)[None],
                 torch.as_tensor(pose_aa)[None], betas=torch.as_tensor(betas)[None], use_cuda=use_cuda)
        v = res["vertices"][0].detach().cpu().numpy().astype(np.float32)  # (T,778,3)
        j = res["joints"][0].detach().cpu().numpy().astype(np.float32)    # (T,21,3)
        # TACO MANO is wrist-centred (center_idx=0) then +trans => wrist sits at trans.
        v = v - j[:, 0:1, :] + tsl[:, None, :]
        valid = present & np.isfinite(v).all(axis=(1, 2))
        out[name] = (v, valid)
    return out["right"], out["left"]


def _principal_plane(centroids):
    """Top-2 principal axes of the hand-centroid cloud -- the view in which the
    poses are MOST spread out, so 2D overlap there is the worst-case visual one."""
    c = centroids - centroids.mean(0, keepdims=True)
    if len(c) < 3:
        return np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt[0], vt[1]


def _iou(a, b):
    """IoU of two axis-aligned 2D bboxes (xmin,ymin,xmax,ymax)."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _select_scatter(cent2d, bboxes, num, spread=0.6, overlap=0.25):
    """Pick `num` poses SCATTERED across the 2D projection (not strung along one
    axis), via damped farthest-point sampling:

      * seed at the medoid (central pose, not an outlier),
      * each step add the pose maximising  min-dist-to-kept - (1-spread)*dist-to-centre,
        so it fills the 2D area but `spread`<1 reins in edge/outlier picks,
      * never add a pose whose 2D bbox IoU with a kept one exceeds `overlap`.

    spread: 0 = compact/central, 1 = maximally spread. Returns candidate indices.
    """
    M = len(cent2d)
    if M <= num:
        return list(range(M))
    centre = cent2d.mean(0)
    rad = float(np.linalg.norm(cent2d - centre, axis=1).max()) + 1e-9
    dcent = np.linalg.norm(cent2d - centre, axis=1) / rad

    kept = [int(np.argmin(dcent))]  # medoid seed
    while len(kept) < num:
        best, bi = -1e18, -1
        for i in range(M):
            if i in kept:
                continue
            if max(_iou(bboxes[i], bboxes[j]) for j in kept) > overlap:
                continue
            md = min(np.linalg.norm(cent2d[i] - cent2d[j]) for j in kept) / rad
            score = md - (1.0 - spread) * dcent[i]
            if score > best:
                best, bi = score, i
        if bi < 0:  # overlap cap blocks everything -> stop early
            break
        kept.append(bi)
    return kept


def _arclen_sample(times, pts, num):
    """Pick `num` frames spread by hand TRAVEL (so dwelling frames don't repeat)."""
    if len(times) <= num:
        return list(times)
    pts = np.asarray(pts)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if cum[-1] < 1e-6:
        sel = np.linspace(0, len(times) - 1, num).round().astype(int)
        return [times[i] for i in sel]
    targ = np.linspace(0.0, cum[-1], num)
    sel = sorted(set(int(np.argmin(np.abs(cum - tv))) for tv in targ))
    return [times[i] for i in sel]


def _compose_sequence(rv, lv, vr_all, vl_all, lo, hi, want_r, want_l,
                      num, gap=0.35, arc=0.25, depth_jitter=0.1, seed=0):
    """Lay a time-ordered hand sequence along a gentle left->right arc so it
    reads as an action unfolding. Both hands are shown per step (keeping their
    real relative configuration), so colours stay balanced; the temporal fade
    in rrd_from_npz then shows direction (older = lighter, newest = solid).
    Only the per-step POSITION is fabricated; poses are real TACO frames."""
    rng = np.random.RandomState(seed)
    valid = [t for t in range(lo, hi + 1) if (want_r and bool(vr_all[t])) or (want_l and bool(vl_all[t]))]
    if not valid:
        raise SystemExit(f"no valid hand poses in frames {lo}-{hi}")

    def pair_pts(t):
        pp = []
        if want_r and bool(vr_all[t]):
            pp.append(rv[t])
        if want_l and bool(vl_all[t]):
            pp.append(lv[t])
        return np.concatenate(pp, 0)

    rep = [pair_pts(t).mean(0) for t in valid]
    idxs = _arclen_sample(valid, rep, num)
    N = len(idxs)
    pair = float(np.median([np.linalg.norm(pair_pts(t).max(0) - pair_pts(t).min(0)) for t in idxs]))
    spacing = pair * (1.0 + gap)
    width = spacing * max(1, N - 1)
    zero = np.zeros((778, 3), np.float32)

    right, left, vr, vl = [], [], [], []
    for k, t in enumerate(idxs):
        pc = pair_pts(t).mean(0)
        x = (k - (N - 1) / 2.0) * spacing
        u = (2.0 * k / (N - 1) - 1.0) if N > 1 else 0.0  # -1..1
        z = arc * width * 0.5 * (1.0 - u * u) + rng.uniform(-1, 1) * spacing * 0.05
        y = rng.uniform(-depth_jitter, depth_jitter) * pair
        P = np.array([x, y, z], np.float64)
        has_r = want_r and bool(vr_all[t])
        has_l = want_l and bool(vl_all[t])
        right.append((rv[t] - pc + P).astype(np.float32) if has_r else zero)
        left.append((lv[t] - pc + P).astype(np.float32) if has_l else zero)
        vr.append(has_r); vl.append(has_l)
    return np.stack(right), np.stack(left), np.array(vr), np.array(vl), idxs


def _rot_about(axis, ang):
    a = np.asarray(axis, np.float64); a = a / (np.linalg.norm(a) + 1e-9)
    c, s = np.cos(ang), np.sin(ang)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def _compose_scatter(pool, num, seed=0, gap=0.3, jitter_rot=20.0, depth_jitter=0.15):
    """Compose a teaser figure: lay `num` hands on a jittered grid (a poster
    wall in the X-Z plane, Y = depth), each using a real TACO hand pose sampled
    from `pool` (so finger articulation/orientation is genuine; only the wrist
    POSITION is fabricated for looks). Returns right/left/vr/vl arrays + src ids.

    Grid is centred at the origin; spacing = hand_size*(1+gap); small per-hand
    position jitter, optional in-plane rotation, and depth jitter add variety.
    """
    rng = np.random.RandomState(seed)
    diag = float(np.median([np.linalg.norm(v.max(0) - v.min(0)) for v, _ in pool]))
    spacing = diag * (1.0 + gap)
    cols = int(np.ceil(np.sqrt(num)))
    rows = int(np.ceil(num / cols))
    jit = spacing * 0.33

    positions = []
    for r in range(rows):
        for c in range(cols):
            if len(positions) >= num:
                break
            x = (c - (cols - 1) / 2.0) * spacing + rng.uniform(-jit, jit)
            z = ((rows - 1) / 2.0 - r) * spacing + rng.uniform(-jit, jit)
            y = rng.uniform(-depth_jitter, depth_jitter) * diag
            positions.append(np.array([x, y, z], np.float64))

    order = rng.permutation(len(pool))
    zero = None
    right, left, vr, vl, src = [], [], [], [], []
    for i, P in enumerate(positions):
        v, side = pool[order[i % len(pool)]]
        Vc = v - v.mean(0)
        if jitter_rot > 0:
            Vc = Vc @ _rot_about([0, 1, 0], np.deg2rad(rng.uniform(-jitter_rot, jitter_rot))).T
        placed = (Vc + P).astype(np.float32)
        if zero is None:
            zero = np.zeros_like(placed)
        if side == "r":
            right.append(placed); left.append(zero); vr.append(True); vl.append(False)
        else:
            left.append(placed); right.append(zero); vl.append(True); vr.append(False)
        src.append(int(order[i % len(pool)]))
    return np.stack(right), np.stack(left), np.array(vr), np.array(vl), src


def main():
    args = parse_args()
    triplet, seq = args.seq_id.rsplit("/", 1)
    triplet, seq = triplet.strip(), seq.strip()

    from hawor.utils.process import get_mano_faces

    R_c2w, cam_pos, T_cam = _load_camera(args.data_root, triplet, seq)
    (rv, vr_all), (lv, vl_all) = _load_hand_verts(args.data_root, triplet, seq, T_cam, use_cuda=not args.cpu)
    T = min(T_cam, rv.shape[0], lv.shape[0])

    faces = get_mano_faces()
    faces_right = np.concatenate([faces, _PALM], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]

    want_r = args.hands in ("both", "right")
    want_l = args.hands in ("both", "left")
    lo = max(0, args.frame_start)
    hi = (T - 1) if args.frame_end < 0 else min(args.frame_end, T - 1)

    if args.layout == "sequence":
        # time-ordered arc of hand-pairs -> reads as an action unfolding
        right, left, vr, vl, src = _compose_sequence(
            rv, lv, vr_all, vl_all, lo, hi, want_r, want_l, args.num_samples,
            gap=args.gap, arc=args.arc, depth_jitter=args.depth_jitter, seed=args.seed)
        idxs = list(range(len(src)))
        sample_idx = np.asarray(src, np.int64)
        no_fade_flag = bool(args.no_fade)  # fade ON by default -> shows time direction
        print(f"[sequence] {len(idxs)} time-ordered hand poses along an arc "
              f"(frames {src[0]}..{src[-1]}, gap={args.gap}, arc={args.arc})")
    elif args.layout == "scatter":
        # compose a figure: real TACO hand poses on fabricated, scattered positions
        pool = []
        for t in range(lo, hi + 1):
            if want_r and bool(vr_all[t]):
                pool.append((rv[t], "r"))
            if want_l and bool(vl_all[t]):
                pool.append((lv[t], "l"))
        if not pool:
            raise SystemExit(f"no valid hand poses in frames {lo}-{hi}")
        right, left, vr, vl, src = _compose_scatter(
            pool, args.num_samples, seed=args.seed, gap=args.gap,
            jitter_rot=args.jitter_rot, depth_jitter=args.depth_jitter)
        idxs = list(range(len(src)))
        sample_idx = np.asarray(src, np.int64)
        no_fade_flag = True  # temporal fade is meaningless for a composed layout
        print(f"[scatter] composed {len(idxs)} hands from {len(pool)} TACO poses "
              f"(gap={args.gap}, jitter_rot={args.jitter_rot})")
    else:
        # keep real TACO positions; pick a low-overlap, 2D-spread subset
        cand = [t for t in range(lo, hi + 1)
                if (want_r and bool(vr_all[t])) or (want_l and bool(vl_all[t]))]
        if not cand:
            raise SystemExit(f"no valid hand poses in frames {lo}-{hi}")
        SUB = 120  # subsample verts for cheap bbox/PCA
        pts_list, centroids = [], []
        for t in cand:
            pp = []
            if want_r and bool(vr_all[t]):
                pp.append(rv[t][::max(1, 778 // SUB)])
            if want_l and bool(vl_all[t]):
                pp.append(lv[t][::max(1, 778 // SUB)])
            p = np.concatenate(pp, 0)
            pts_list.append(p)
            centroids.append(p.mean(0))
        centroids = np.asarray(centroids)
        e1, e2 = _principal_plane(centroids)
        bboxes = [(float((p @ e1).min()), float((p @ e2).min()),
                   float((p @ e1).max()), float((p @ e2).max())) for p in pts_list]
        cent2d = np.stack([centroids @ e1, centroids @ e2], axis=1)
        sel = _select_scatter(cent2d, bboxes, args.num_samples, spread=args.spread, overlap=args.overlap)
        idxs = sorted(cand[i] for i in sel)  # time order for the temporal fade
        right = np.stack([rv[t] for t in idxs], 0)
        left = np.stack([lv[t] for t in idxs], 0)
        vr = np.array([want_r and bool(vr_all[t]) for t in idxs])
        vl = np.array([want_l and bool(vl_all[t]) for t in idxs])
        sample_idx = np.asarray(idxs, np.int64)
        no_fade_flag = bool(args.no_fade)
        print(f"[select] {len(cand)} valid frames in {lo}-{hi} -> {len(idxs)} poses "
              f"(spread={args.spread}, overlap<= {args.overlap})")

    data = dict(
        sample_idx=sample_idx,
        right_verts=right.astype(np.float32), left_verts=left.astype(np.float32),
        faces_right=faces_right.astype(np.int32), faces_left=faces_left.astype(np.int32),
        valid_r=vr, valid_l=vl,
        no_fade=np.asarray(bool(no_fade_flag)), alpha_min=np.asarray(float(args.alpha_min)),
    )
    if args.camera and args.layout == "taco":
        from lib.vis.run_vis2 import camera_marker_geometry
        mverts, mfaces, _ = camera_marker_geometry(args.frustum_radius, args.frustum_height)
        cam = np.stack([np.einsum("ij,nj->ni", R_c2w[t], mverts) + cam_pos[t][None] for t in idxs], 0)
        data.update(cam_verts=cam.astype(np.float32), cam_faces=np.asarray(mfaces, np.int32),
                    cam_centers=np.stack([cam_pos[t] for t in idxs], 0).astype(np.float32))
    if args.ground:
        from lib.vis.wham_tools.tools import checkerboard_geometry
        gv, gf, gvc, _ = checkerboard_geometry(length=100, c1=0, c2=0, up="z")
        z_floor = float(np.minimum(right[vr].reshape(-1, 3)[:, 2].min() if vr.any() else 0.0,
                                   left[vl].reshape(-1, 3)[:, 2].min() if vl.any() else 0.0))
        gv[:, 2] += z_floor - 0.05
        data.update(ground_v=gv.astype(np.float32), ground_f=np.asarray(gf, np.int32),
                    ground_c=(np.asarray(gvc)[:, :3] * 255).astype(np.uint8))

    os.makedirs(os.path.dirname(os.path.abspath(args.dump_npz)) or ".", exist_ok=True)
    np.savez_compressed(args.dump_npz, **data)
    print(f"saved {args.dump_npz}  (seq={args.seq_id}, {len(idxs)} samples, T={T}). convert with:\n"
          f"  python scripts/rrd_from_npz.py {args.dump_npz} <out.rrd>")


if __name__ == "__main__":
    main()
