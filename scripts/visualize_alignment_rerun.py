#!/usr/bin/env python3
"""Rerun.io 3D viz to EYEBALL HaWoR pipeline metric-world consistency.

Overlays, in ONE metric world, the three signals the pipeline must keep aligned:
  * Any4D metric scene depth      -> back-projected, RGB-colored, accumulated point cloud
  * DPVO camera trajectory        -> pinhole + frustum (per frame) + full polyline
  * HaWoR hand mesh(es)           -> MANO mesh in world frame (per frame)

so a human can scrub the timeline and judge:
  - does the hand mesh sit on the surfaces the depth says are there (contact frames)?
  - is the hand the right SIZE relative to the scene?
  - does the static scene structure stay put across frames, or smear (camera drift)?

WORLD-FRAME CONVENTIONS (faithful to the pipeline, file:line evidence in the report):
  - Camera c2w from lib/eval_utils/custom_utils.load_slam_cam:
        t_c2w = traj[:, :3] * scale   (scale: DPVO units -> meters)
        R_c2w = quaternion_to_matrix(q[[3,0,1,2]])    (c2w rotation)
  - Any4D depth is already METRIC meters (depths_uint16 * 1e-3, see slam.py
    _save_dense_depth_uint16_npz / _load_dense_depth_cache). No scale applied.
  - A depth pixel (u,v) back-projects to world as:
        X_cam   = depth * K^-1 @ [u, v, 1]      (OpenCV cam frame, +z forward)
        X_world = R_c2w @ X_cam + t_c2w
    using intrinsics scaled from img_focal/img_center (defined at 456x256) to the
    584x328 depth grid by the resolution ratio.
  - Hand: world_space_res.pth holds world-frame MANO params produced by
    cam2world_convert() using the SAME R_c2w/t_c2w (scale-applied). So MANO verts
    from those params are in the SAME metric world. We do NOT apply demo.py's R_x
    flip (that flip is render-only; here we keep the raw pipeline world for all three).

This script is READ-ONLY w.r.t. the pipeline; it never re-runs any stage. It does a
lightweight numpy MANO LBS forward on CPU (torch not required) to get hand vertices.

Run:
  python scripts/visualize_alignment_rerun.py \
      --seq_folder /root/hawor_result/.../f001_w009_v00179_i000 \
      --stride 20 --max_points 1500000 --out <seq>/alignment_viz.rrd
Then open with:
  rerun alignment_viz.rrd
"""
import argparse
import glob
import os
import pickle
import re
import sys
import types

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# MANO (numpy, torch-free)                                                     #
# --------------------------------------------------------------------------- #
# Palm-fill triangles HaWoR adds to close the right-hand mesh (from process.py).
_PALM = np.array(
    [[92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
     [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
     [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
     [120, 108, 78], [78, 108, 79]], dtype=np.int64)


def _install_chumpy_stub():
    """Let pickle load the chumpy-backed MANO_*.pkl without the chumpy package.

    The pkl references chumpy.ch.Ch and chumpy.reordering.Select. We restore them as
    plain holders; their numpy payload lives in __dict__ ('x' for Ch; 'a'/'idxs' for
    Select). We extract the arrays manually afterwards.
    """
    if "chumpy" in sys.modules:
        return
    chumpy = types.ModuleType("chumpy")
    chumpy.__path__ = []
    ch = types.ModuleType("chumpy.ch")
    reordering = types.ModuleType("chumpy.reordering")

    class Ch:
        def __setstate__(self, s):
            self.__dict__.update(s if isinstance(s, dict) else {})

    class Select:
        def __setstate__(self, s):
            self.__dict__.update(s if isinstance(s, dict) else {})

    ch.Ch = Ch
    reordering.Select = Select
    chumpy.Ch = Ch
    chumpy.ch = ch
    chumpy.reordering = reordering
    sys.modules["chumpy"] = chumpy
    sys.modules["chumpy.ch"] = ch
    sys.modules["chumpy.reordering"] = reordering


def _ch_data(obj):
    """Return the numpy payload of a chumpy Ch / Select holder, else np.asarray(obj)."""
    if hasattr(obj, "__dict__"):
        d = obj.__dict__
        if "x" in d:  # Ch
            return np.asarray(d["x"])
        if "a" in d and "idxs" in d:  # Select (reordering of a.x)
            base = _ch_data(d["a"]).flatten()
            return base[np.asarray(d["idxs"])]
    return np.asarray(obj)


def load_mano(mano_pkl, n_betas=10):
    """Load MANO components needed for an LBS forward (numpy)."""
    _install_chumpy_stub()
    with open(mano_pkl, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    v_template = _ch_data(d["v_template"]).reshape(778, 3).astype(np.float64)
    # shapedirs: select picks (778,3,10) out of the full (778,3,20)
    shapedirs = _ch_data(d["shapedirs"]).reshape(778, 3, -1)[:, :, :n_betas].astype(np.float64)
    posedirs = _ch_data(d["posedirs"]).reshape(778, 3, -1).astype(np.float64)  # (778,3,135)
    J_reg = d["J_regressor"]
    J_regressor = (J_reg.toarray() if hasattr(J_reg, "toarray") else np.asarray(J_reg)).astype(np.float64)
    weights = _ch_data(d["weights"]).reshape(778, 16).astype(np.float64)
    kintree = np.asarray(d["kintree_table"]).astype(np.int64)  # (2,16)
    faces = np.asarray(d["f"]).astype(np.int64)  # (1538,3)
    parents = kintree[0].copy()
    parents[0] = -1
    return dict(
        v_template=v_template, shapedirs=shapedirs, posedirs=posedirs,
        J_regressor=J_regressor, weights=weights, parents=parents, faces=faces,
    )


def _rodrigues(rotvecs):
    """(N,3) axis-angle -> (N,3,3) rotation matrices."""
    rotvecs = np.asarray(rotvecs, np.float64).reshape(-1, 3)
    theta = np.linalg.norm(rotvecs, axis=1, keepdims=True)
    theta_safe = np.where(theta < 1e-8, 1.0, theta)
    k = rotvecs / theta_safe
    K = np.zeros((rotvecs.shape[0], 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    I = np.eye(3)[None]
    s = np.sin(theta)[:, :, None]
    c = np.cos(theta)[:, :, None]
    R = I + s * K + (1 - c) * (K @ K)
    R[theta[:, 0] < 1e-8] = np.eye(3)
    return R


def mano_forward(mano, betas, global_orient_aa, hand_pose_aa, transl, return_joints=False):
    """Numpy MANO LBS forward. Mirrors smplx MANOLayer with pose2rot=True semantics,
    flat_hand_mean (hands_mean NOT added — pipeline uses pose2rot=False on rotmats,
    where hands_mean is never applied).

    betas: (10,)  global_orient_aa: (3,)  hand_pose_aa: (15,3)  transl: (3,)
    returns vertices (778,3) in the same frame as transl; if return_joints, also the
    16 MANO joints (J_regressor @ posed verts) — joint 0 is the wrist root_loc used by
    cam2world_convert.
    """
    v_template = mano["v_template"]
    shapedirs = mano["shapedirs"]
    posedirs = mano["posedirs"]
    J_regressor = mano["J_regressor"]
    weights = mano["weights"]
    parents = mano["parents"]

    betas = np.asarray(betas, np.float64).reshape(-1)[: shapedirs.shape[2]]
    # shape blend
    v_shaped = v_template + np.einsum("vck,k->vc", shapedirs, betas)  # (778,3)
    J = J_regressor @ v_shaped  # (16,3)

    full_pose_aa = np.concatenate(
        [np.asarray(global_orient_aa, np.float64).reshape(1, 3),
         np.asarray(hand_pose_aa, np.float64).reshape(15, 3)], axis=0)  # (16,3)
    R = _rodrigues(full_pose_aa)  # (16,3,3)

    # pose blend shapes: (R[1:] - I) flattened, dot posedirs
    pose_feat = (R[1:] - np.eye(3)[None]).reshape(-1)  # 15*9 = 135
    v_posed = v_shaped + np.einsum("vck,k->vc", posedirs, pose_feat)

    # build per-joint global transforms (rigid kinematic chain)
    G = np.zeros((16, 4, 4))
    G[0, :3, :3] = R[0]
    G[0, :3, 3] = J[0]
    G[0, 3, 3] = 1.0
    for i in range(1, 16):
        T_local = np.eye(4)
        T_local[:3, :3] = R[i]
        T_local[:3, 3] = J[i] - J[parents[i]]
        G[i] = G[parents[i]] @ T_local
    # remove rest-pose joint offset
    for i in range(16):
        Jh = np.array([J[i, 0], J[i, 1], J[i, 2], 0.0])
        G[i, :, 3] = G[i, :, 3] - G[i] @ Jh

    T = np.einsum("vj,jab->vab", weights, G)  # (778,4,4)
    v_h = np.concatenate([v_posed, np.ones((778, 1))], axis=1)  # (778,4)
    v_out = np.einsum("vab,vb->va", T, v_h)[:, :3]
    transl = np.asarray(transl, np.float64).reshape(1, 3)
    verts = v_out + transl
    if return_joints:
        joints = (J_regressor @ v_out) + transl  # (16,3); joint 0 = wrist root_loc
        return verts, joints
    return verts


# --------------------------------------------------------------------------- #
# Pipeline I/O                                                                 #
# --------------------------------------------------------------------------- #
def quaternion_to_matrix_np(q):
    """q: (...,4) real-first (w,x,y,z) -> (...,3,3). Mirrors custom_utils."""
    q = np.asarray(q, np.float64)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two_s = 2.0 / (q * q).sum(-1)
    o = np.stack([
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ], -1)
    return o.reshape(q.shape[:-1] + (3, 3))


def load_slam_cam_np(npz_path):
    """numpy port of lib/eval_utils/custom_utils.load_slam_cam.
    Returns R_c2w (T,3,3), t_c2w (T,3) in metric meters, plus focal/center/scale."""
    d = dict(np.load(npz_path, allow_pickle=True))
    traj = d["traj"]
    scale = float(d["scale"])
    t_c2w = traj[:, :3].astype(np.float64) * scale
    q_xyzw = traj[:, 3:].astype(np.float64)
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
    R_c2w = quaternion_to_matrix_np(q_wxyz)
    return R_c2w, t_c2w, float(d["img_focal"]), np.asarray(d["img_center"], np.float64), scale


def _rotmat_to_aa(R):
    """(N,3,3) -> (N,3) axis-angle via scipy."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def cam2world_convert_np(R_c2w, t_c2w, frame_ck, root_orient_rotmat, hand_pose_rotmat,
                         trans_cam, betas, mano):
    """numpy port of lib/eval_utils/custom_utils.cam2world_convert for ONE chunk.

    Inputs are CAMERA-frame MANO params for absolute frames `frame_ck`:
      root_orient_rotmat (n,3,3), hand_pose_rotmat (n,15,3,3), trans_cam (n,3), betas (n,10)
      R_c2w/t_c2w: per-frame camera->world (metric, scale applied), indexed at frame_ck.
    Returns per-frame (orient_aa (n,3), pose_aa (n,15,3), trans_world (n,3), betas).
    """
    n = trans_cam.shape[0]
    orient_aa = _rotmat_to_aa(root_orient_rotmat.reshape(-1, 3, 3)).reshape(n, 3)
    pose_aa = _rotmat_to_aa(hand_pose_rotmat.reshape(-1, 3, 3)).reshape(n, 15, 3)
    # world root orientation = R_c2w @ root_orient (rotmat) -> aa
    Rw = np.einsum("tij,tjk->tik", R_c2w, root_orient_rotmat)  # (n,3,3)
    orient_world_aa = _rotmat_to_aa(Rw.reshape(-1, 3, 3)).reshape(n, 3)
    # root_loc: wrist joint in cam frame (run_mano with cam trans), then world transform
    trans_world = np.empty((n, 3), np.float64)
    for i in range(n):
        _, joints = mano_forward(mano, betas[i], orient_aa[i], pose_aa[i], trans_cam[i],
                                 return_joints=True)
        root_loc = joints[0]               # (3,) wrist in cam frame
        offset = trans_cam[i] - root_loc   # constant
        trans_world[i] = R_c2w[i] @ root_loc + t_c2w[i] + offset
    return orient_world_aa, pose_aa, trans_world, betas


def load_world_hands(seq_folder, mano, R_c2w, t_c2w, n_frames, cam_valid=None):
    """Reconstruct world-frame hands from cam_space/<idx>/<s>_<e>.json (camera-frame MANO
    params) using the pipeline's own cam2world_convert. Torch-free; faithful to
    scripts_test_video/hawor_video.py (idx>0 -> right, idx==0 -> left).

    NOTE: this reconstructs the OBSERVED (tracked) frames stored in cam_space; infiller-
    only gaps in world_space_res.pth are not reproduced here and are simply left invalid.

    Returns list (idx0=left, idx1=right) of dict(valid (T,), verts[t] cache).
    """
    hands = []
    for idx in (0, 1):
        cdir = os.path.join(seq_folder, "cam_space", str(idx))
        verts_by_frame = {}
        if os.path.isdir(cdir):
            for jf in sorted(glob.glob(os.path.join(cdir, "*.json"))):
                base = os.path.splitext(os.path.basename(jf))[0]
                m = re.match(r"(\d+)_(\d+)$", base)
                if not m:
                    continue
                s, e = int(m.group(1)), int(m.group(2))
                try:
                    import json
                    with open(jf) as f:
                        d = json.load(f)
                except Exception:
                    continue
                root_orient = np.asarray(d["init_root_orient"], np.float64)[0]  # (n,3,3)
                hand_pose = np.asarray(d["init_hand_pose"], np.float64)[0]       # (n,15,3,3)
                trans = np.asarray(d["init_trans"], np.float64)[0]              # (n,3)
                betas = np.asarray(d["init_betas"], np.float64)[0]             # (n,10)
                n = trans.shape[0]
                frame_ck = np.arange(s, s + n)
                ok = (frame_ck >= 0) & (frame_ck < min(n_frames, R_c2w.shape[0]))
                if cam_valid is not None:
                    ok &= cam_valid[np.clip(frame_ck, 0, R_c2w.shape[0] - 1)]
                if not ok.any():
                    continue
                fk = frame_ck[ok]
                ow, pw, tw, bw = cam2world_convert_np(
                    R_c2w[fk], t_c2w[fk], fk,
                    root_orient[ok], hand_pose[ok], trans[ok], betas[ok], mano)
                for j, t in enumerate(fk):
                    verts_by_frame[int(t)] = mano_forward(mano, bw[j], ow[j], pw[j], tw[j])
        valid = np.zeros(n_frames, bool)
        for t in verts_by_frame:
            if 0 <= t < n_frames:
                valid[t] = True
        hands.append(dict(valid=valid, verts=verts_by_frame))
    return hands


def build_intrinsics(img_focal, img_center, depth_hw, ref_hw=(256, 456)):
    """Scale intrinsics (defined at ref_hw, default 456x256) to the depth grid.
    img_center is [cx, cy] at ref resolution. Returns K (3,3) at depth resolution."""
    dh, dw = depth_hw
    rh, rw = ref_hw
    sx = dw / float(rw)
    sy = dh / float(rh)
    fx = img_focal * sx
    fy = img_focal * sy
    cx = float(img_center[0]) * sx
    cy = float(img_center[1]) * sy
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)


def backproject(depth, K, R_c2w, t_c2w, rgb=None, mask_keep=None, depth_min=0.05, depth_max=15.0):
    """Back-project a depth map to world points (metric).

    depth: (H,W) meters. mask_keep: (H,W) bool of pixels to KEEP (hand removed -> False).
    rgb: (H,W,3) uint8 aligned to depth grid (resized). Returns (P,3) pts, (P,3) cols."""
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    sel = np.isfinite(depth) & (depth > depth_min) & (depth < depth_max)
    if mask_keep is not None:
        sel &= mask_keep
    if not sel.any():
        return np.empty((0, 3)), np.empty((0, 3), np.uint8)
    u = uu[sel].astype(np.float64)
    v = vv[sel].astype(np.float64)
    z = depth[sel].astype(np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    X_cam = np.stack([x, y, z], axis=1)  # (P,3) OpenCV cam frame
    X_world = X_cam @ R_c2w.T + t_c2w[None]
    if rgb is not None:
        cols = rgb[sel]
    else:
        cols = np.full((X_world.shape[0], 3), 200, np.uint8)
    return X_world, cols


def voxel_downsample(pts, cols, voxel):
    """Voxel-grid downsample keeping one (mean) point per occupied voxel."""
    if pts.shape[0] == 0 or voxel <= 0:
        return pts, cols
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    # average within voxel for smooth look
    nvox = idx.shape[0]
    sums = np.zeros((nvox, 3))
    csum = np.zeros((nvox, 3))
    cnt = np.zeros((nvox, 1))
    np.add.at(sums, inv, pts)
    np.add.at(csum, inv, cols.astype(np.float64))
    np.add.at(cnt, inv, 1.0)
    return sums / cnt, (csum / cnt).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #
def _find(seq_folder, pat):
    cands = sorted(glob.glob(os.path.join(seq_folder, "SLAM", pat)))
    return cands[-1] if cands else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq_folder", required=True)
    ap.add_argument("--stride", type=int, default=20, help="subsample every N frames")
    ap.add_argument("--max_points", type=int, default=1_500_000, help="cap on total scene points")
    ap.add_argument("--voxel", type=float, default=0.02, help="voxel size (m) for cloud downsample")
    ap.add_argument("--depth_min", type=float, default=0.05)
    ap.add_argument("--depth_max", type=float, default=15.0)
    ap.add_argument("--no_hand_removal", action="store_true",
                    help="keep hand pixels in the scene cloud")
    ap.add_argument("--out", default=None, help="output .rrd (default <seq>/alignment_viz.rrd)")
    ap.add_argument("--mano_pkl", default=None, help="MANO_RIGHT.pkl (default repo _DATA)")
    ap.add_argument("--no_hand", action="store_true", help="skip hand mesh entirely")
    args = ap.parse_args()

    import rerun as rr

    seq = args.seq_folder
    out_rrd = args.out or os.path.join(seq, "alignment_viz.rrd")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mano_pkl = args.mano_pkl or os.path.join(repo_root, "_DATA", "data", "mano", "MANO_RIGHT.pkl")

    # ---- camera ----
    slam_npz = _find(seq, "hawor_slam_w_scale_*.npz")
    if slam_npz is None:
        raise FileNotFoundError("no SLAM/hawor_slam_w_scale_*.npz")
    R_c2w, t_c2w, img_focal, img_center, scale = load_slam_cam_np(slam_npz)
    n_cam = R_c2w.shape[0]
    cam_valid = np.isfinite(R_c2w).all(axis=(1, 2)) & np.isfinite(t_c2w).all(axis=1)
    print(f"[cam] {n_cam} poses, {int(cam_valid.sum())} finite "
          f"({100*cam_valid.mean():.1f}%), focal={img_focal}, center={img_center}, "
          f"scale={scale:.6g}")
    if cam_valid.sum() == 0:
        raise RuntimeError("no finite camera poses")

    # ---- depth ----
    depth_npz = _find(seq, "dense_depth_any4d_*.npz")
    if depth_npz is None:
        raise FileNotFoundError("no SLAM/dense_depth_any4d_*.npz")
    dd = np.load(depth_npz, allow_pickle=True)
    depth_frame_idx = dd["frame_indices"].astype(np.int64).reshape(-1)
    depths_u16 = dd["depths_uint16"]  # (T,H,W) mm, lazy
    Hd, Wd = int(dd["height"]), int(dd["width"])
    K = build_intrinsics(img_focal, img_center, (Hd, Wd))
    print(f"[depth] {depths_u16.shape} mm, K=\n{K}")

    # ---- frames (rgb) ----
    frame_dirs = glob.glob(os.path.join(seq, "..", "..", "..", "frames", "*")) + \
        glob.glob(os.path.join(seq, "frames", "*"))
    frames_root = None
    # robust: search upward for a frames/<clip>/ with jpgs
    clip_id = os.path.basename(os.path.normpath(seq))
    for cand in [
        os.path.join(seq, "frames", clip_id),
        os.path.normpath(os.path.join(seq, "..", "..", "..", "frames", clip_id)),
        os.path.normpath(os.path.join(seq, "..", "..", "frames", clip_id)),
    ]:
        if os.path.isdir(cand) and glob.glob(os.path.join(cand, "*.jpg")):
            frames_root = cand
            break
    print(f"[frames] {frames_root}")

    # ---- hand masks (optional removal) ----
    masks = None
    if not args.no_hand_removal:
        mpaths = sorted(glob.glob(os.path.join(seq, "tracks_*", "model_masks.npy")))
        if mpaths:
            masks = np.load(mpaths[0])  # (T,256,456) bool
            print(f"[masks] {masks.shape}")

    # ---- hands ----
    hands = None
    if not args.no_hand:
        try:
            mano = load_mano(mano_pkl)
            hands = load_world_hands(seq, mano, R_c2w, t_c2w, n_cam, cam_valid=cam_valid)
            faces_right = np.concatenate([mano["faces"], _PALM], axis=0)
            faces_left = faces_right[:, [0, 2, 1]]
            nvalid = [int(h["valid"].sum()) for h in hands]
            print(f"[hand] reconstructed from cam_space: left {nvalid[0]} frames, "
                  f"right {nvalid[1]} frames; faces={faces_right.shape}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[hand] SKIPPED: {e}")
            hands = None

    # ---- rerun setup ----
    rr.init("hawor_alignment", spawn=False)
    # We log everything in-memory then rr.save() to the .rrd at the end.

    # frame sampling
    sample = list(range(0, n_cam, max(1, args.stride)))
    depth_lookup = {int(f): i for i, f in enumerate(depth_frame_idx)}

    rr.log("world", rr.ViewCoordinates.RDF, static=True)  # x-right, y-down, z-fwd (OpenCV)

    # ---- accumulate scene cloud across sampled frames (static) ----
    point_budget = args.max_points
    all_pts = []
    all_cols = []
    per_frame_quota = max(1, point_budget // max(1, len(sample)))

    for t in sample:
        if not cam_valid[t]:
            continue
        di = depth_lookup.get(t)
        if di is None:
            continue
        depth = depths_u16[di].astype(np.float32) * 1e-3  # meters
        # rgb
        rgb = None
        if frames_root is not None:
            fp = os.path.join(frames_root, f"{t:06d}.jpg")
            if os.path.exists(fp):
                img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(img, (Wd, Hd), interpolation=cv2.INTER_LINEAR)
        # mask: keep = NOT hand
        mask_keep = None
        if masks is not None and t < masks.shape[0]:
            hand = masks[t].astype(np.uint8)
            hand_d = cv2.resize(hand, (Wd, Hd), interpolation=cv2.INTER_NEAREST) > 0
            mask_keep = ~hand_d
        pts, cols = backproject(depth, K, R_c2w[t], t_c2w[t], rgb=rgb,
                                mask_keep=mask_keep, depth_min=args.depth_min,
                                depth_max=args.depth_max)
        if pts.shape[0] == 0:
            continue
        pts, cols = voxel_downsample(pts, cols, args.voxel)
        if pts.shape[0] > per_frame_quota:
            sel = np.random.choice(pts.shape[0], per_frame_quota, replace=False)
            pts, cols = pts[sel], cols[sel]
        all_pts.append(pts)
        all_cols.append(cols)

    if all_pts:
        scene_pts = np.concatenate(all_pts, axis=0)
        scene_cols = np.concatenate(all_cols, axis=0)
        # global voxel dedup + final cap
        scene_pts, scene_cols = voxel_downsample(scene_pts, scene_cols, args.voxel)
        if scene_pts.shape[0] > point_budget:
            sel = np.random.choice(scene_pts.shape[0], point_budget, replace=False)
            scene_pts, scene_cols = scene_pts[sel], scene_cols[sel]
        rr.log("world/scene_cloud",
               rr.Points3D(scene_pts, colors=scene_cols, radii=0.006), static=True)
        print(f"[scene] {scene_pts.shape[0]} points logged")
    else:
        scene_pts = np.empty((0, 3))
        print("[scene] WARNING: no scene points")

    # ---- full camera trajectory polyline (static) ----
    # Split into contiguous finite runs so NaN gaps don't draw spurious long edges.
    traj_segments = []
    run = []
    for t in range(n_cam):
        if cam_valid[t]:
            run.append(t_c2w[t])
        elif run:
            if len(run) >= 2:
                traj_segments.append(np.asarray(run))
            run = []
    if len(run) >= 2:
        traj_segments.append(np.asarray(run))
    if traj_segments:
        rr.log("world/camera_trajectory",
               rr.LineStrips3D(traj_segments, colors=[255, 180, 0], radii=0.004),
               static=True)

    # ---- per-frame: camera pinhole+frustum, hand mesh ----
    PURPLE = [205, 153, 209]
    BLUE = [53, 152, 202]
    for t in sample:
        if not cam_valid[t]:
            continue
        rr.set_time("frame", sequence=t)
        # camera transform (c2w)
        rr.log("world/camera",
               rr.Transform3D(translation=t_c2w[t], mat3x3=R_c2w[t]))
        rr.log("world/camera/image",
               rr.Pinhole(image_from_camera=K, width=Wd, height=Hd,
                          camera_xyz=rr.ViewCoordinates.RDF))
        # rgb on the image plane for context
        if frames_root is not None:
            fp = os.path.join(frames_root, f"{t:06d}.jpg")
            if os.path.exists(fp):
                img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (Wd, Hd), interpolation=cv2.INTER_LINEAR)
                rr.log("world/camera/image/rgb", rr.Image(img))
        # hands (idx0=left, idx1=right)
        if hands is not None:
            for hi, h in enumerate(hands):
                verts = h["verts"].get(int(t))
                name = "right" if hi == 1 else "left"
                if verts is None:
                    rr.log(f"world/hand_{name}", rr.Clear(recursive=True))
                    continue
                color = PURPLE if hi == 1 else BLUE
                faces = faces_right if hi == 1 else faces_left
                rr.log(f"world/hand_{name}",
                       rr.Mesh3D(vertex_positions=verts,
                                 triangle_indices=faces,
                                 albedo_factor=np.array(color, np.float32) / 255.0))

    # ---- save ----
    rr.save(out_rrd)
    sz = os.path.getsize(out_rrd)
    print(f"\n[done] wrote {out_rrd}  ({sz/1e6:.1f} MB)")
    print(f"  frames sampled: {len(sample)} (stride {args.stride})")
    print(f"  scene points  : {scene_pts.shape[0]}")
    print(f"  open with     : rerun {out_rrd}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
