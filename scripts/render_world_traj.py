#!/usr/bin/env python3
"""Render a HaWoR-Figure-1-style STATIC trajectory composite ("trail") for a paper.

For one processed clip it draws, in a single still image:
  * world-space MANO hand meshes at several sampled timesteps (fading "afterimage"),
  * a string of camera frustum markers along the camera trajectory,
  * a checkerboard ground plane.

It does NOT re-run detection / motion / SLAM. It reuses the two artifacts a clip
already has after the motion stage:
  <seq_folder>/world_space_res.pth                 -> [pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid]
  <seq_folder>/SLAM/hawor_slam_w_scale_<s>_<e>.npz -> camera trajectory (c2w)

This is the static counterpart of demo.py --vis_mode world (which animates the same
scene). All geometry/coordinate handling mirrors demo.py so the look matches.

Run on a GPU machine with aitviewer installed (headless EGL), e.g.:
  python scripts/render_world_traj.py --seq_folder /path/to/outputs/<clip_id> \
      --num_samples 8 --out /tmp/world_traj.png

Camera angle almost always needs a couple of iterations for a paper figure --
tune with --view / --cam_azim / --cam_elev / --cam_dist_scale, or --rebase.
"""
import argparse
import glob
import os
import re
import shutil
import sys

# headless GL before any aitviewer import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import torch

from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import (
    interpolate_slam_cameras_at_video_frames,
    load_slam_cam,
)
from lib.vis.run_vis2 import camera_marker_geometry, lookat_matrix
from lib.vis.wham_tools.tools import checkerboard_geometry
import lib.vis.viewer as viewer_utils
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.material import Material

HAND2IDX = {"left": 0, "right": 1}
PURPLE = (0.804, 0.600, 0.820)   # director-purple  (right)
BLUE = (0.207, 0.596, 0.792)     # director-blue    (left)

# palm-fill triangles HaWoR adds to close the right-hand mesh (from demo.py)
_PALM = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
                  [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
                  [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
                  [120, 108, 78], [78, 108, 79]])


def vertex_normals(verts, faces):
    """Smooth (area-weighted) per-vertex normals so rerun shades the mesh with
    a soft light->dark gradient instead of rendering it flat single-colour."""
    verts = np.asarray(verts, np.float64)
    faces = np.asarray(faces, np.int64)
    nrm = np.zeros_like(verts)
    tris = verts[faces]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])  # area-weighted face normals
    for i in range(3):
        np.add.at(nrm, faces[:, i], fn)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return (nrm / ln).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq_folder", required=True, help="clip dir with world_space_res.pth and SLAM/")
    p.add_argument("--out", default=None, help="output png (default <seq_folder>/world_traj.png)")
    p.add_argument("--rrd", default=None,
                   help="if set, skip aitviewer/GL and log the same trail to this rerun .rrd file "
                        "(open in the rerun web viewer; no display/EGL needed). "
                        "Needs rerun-sdk in THIS env -- if that conflicts with the hawor numpy, "
                        "use --dump_npz instead and convert with scripts/rrd_from_npz.py.")
    p.add_argument("--dump_npz", default=None,
                   help="if set, skip all rendering and dump the trail geometry (hands + cameras + "
                        "ground) to this .npz. Runs in the hawor env (no rerun/GL). Convert to a "
                        ".rrd in a rerun-only env with: python scripts/rrd_from_npz.py <npz> <rrd>")
    p.add_argument("--slam_npz", default=None, help="explicit SLAM npz (else auto-glob SLAM/hawor_slam_w_scale_*.npz)")
    p.add_argument("--num_samples", type=int, default=8, help="number of timesteps to lay along the trail")
    p.add_argument("--stride", type=int, default=0, help="if >0, sample every N frames instead of num_samples")
    p.add_argument("--select", choices=["even", "scatter"], default="even",
                   help="how to pick the trail timesteps in [frame_start, frame_end]: "
                        "'even' = linspace (original behaviour). "
                        "'scatter' = 2D-overlap-aware pick (same as taco_gt_trail's --layout taco) "
                        "so hands don't visually overlap in the trail; great for real pipeline output.")
    p.add_argument("--spread", type=float, default=0.6,
                   help="[--select scatter] 0 = compact/central, 1 = maximally spread")
    p.add_argument("--overlap", type=float, default=-1.0,
                   help="[--select scatter] max pairwise 2D bbox IoU between kept poses. "
                        "Default -1 = auto-tune so we actually get --num_samples poses "
                        "(real clustered trajectories need this; set a fixed value like 0.25 "
                        "to enforce a hard clearance and accept fewer poses).")
    p.add_argument("--no_camera", action="store_true",
                   help="hands only: omit camera frustums + trajectory")
    p.add_argument("--frame_start", type=int, default=0,
                   help="restrict the trail to video frames >= this (default 0)")
    p.add_argument("--frame_end", type=int, default=-1,
                   help="restrict the trail to video frames <= this (default -1 = last frame)")
    p.add_argument("--hands", choices=["both", "left", "right"], default="both")
    p.add_argument("--no_fade", action="store_true", help="solid meshes instead of fading alpha")
    p.add_argument("--alpha_min", type=float, default=0.30, help="alpha of the oldest mesh (newest is 1.0)")
    p.add_argument("--no_ground", action="store_true")
    p.add_argument("--frustum_radius", type=float, default=0.05)
    p.add_argument("--frustum_height", type=float, default=0.10)
    p.add_argument("--rebase", action="store_true",
                   help="rebase world so frame-0 camera is identity (matches demo.py --world_rebase_to_first_camera)")
    p.add_argument("--align", action="store_true",
                   help="apply the hand-depth-align α(t) (from SLAM/hand_depth_align_*.npz) to the mesh "
                        "verts so the figure matches the depth-aligned dataset (ray-scale about camera).")
    p.add_argument("--align_npz", default=None, help="explicit hand_depth_align npz (else auto-glob under SLAM/)")
    # viewer camera
    p.add_argument("--view", choices=["auto", "hawor"], default="auto",
                   help="auto = fit to scene with azim/elev; hawor = demo.py's fixed side view")
    p.add_argument("--cam_azim", type=float, default=60.0, help="[auto] azimuth degrees")
    p.add_argument("--cam_elev", type=float, default=18.0, help="[auto] elevation degrees")
    p.add_argument("--cam_dist_scale", type=float, default=2.6, help="[auto] distance = scene_extent * this")
    p.add_argument("--width", type=int, default=2000)
    p.add_argument("--height", type=int, default=1500)
    return p.parse_args()


def _find_slam_npz(seq_folder, explicit):
    if explicit:
        return explicit
    cands = sorted(glob.glob(os.path.join(seq_folder, "SLAM", "hawor_slam_w_scale_*.npz")))
    if not cands:
        raise FileNotFoundError(f"no SLAM/hawor_slam_w_scale_*.npz under {seq_folder}")
    return cands[-1]


def _slam_index_range(npz_path):
    m = re.search(r"hawor_slam_w_scale_(\d+)_(\d+)\.npz$", os.path.basename(npz_path))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _load_align_alpha(seq_folder, explicit):
    """Load α(t) from a hand_depth_align sidecar (written by lib/pipeline/hand_depth_align.py)."""
    path = explicit
    if path is None:
        cands = sorted(glob.glob(os.path.join(seq_folder, "SLAM", "hand_depth_align_*.npz")))
        if not cands:
            return None
        path = cands[-1]
    try:
        data = np.load(path, allow_pickle=True)
        alpha = np.asarray(data["alpha"], dtype=np.float32).reshape(-1)
        return alpha if alpha.size else None
    except Exception:
        return None


def _sampled_trail_arrays(args, idxs, right_verts, left_verts, valid_r, valid_l,
                          want_r, want_l, R_c2w, t_c2w):
    """Pull the per-sample hand/camera geometry the rerun trail needs."""
    mverts, mfaces, _ = camera_marker_geometry(args.frustum_radius, args.frustum_height)
    right = np.stack([right_verts[t] for t in idxs], 0)
    left = np.stack([left_verts[t] for t in idxs], 0)
    cam = np.stack([np.einsum("ij,nj->ni", R_c2w[t], mverts) + t_c2w[t][None] for t in idxs], 0)
    centers = np.stack([t_c2w[t] for t in idxs], 0)
    vr = np.array([bool(want_r) and bool(valid_r[min(t, len(valid_r) - 1)]) for t in idxs])
    vl = np.array([bool(want_l) and bool(valid_l[min(t, len(valid_l) - 1)]) for t in idxs])
    return right, left, cam, mfaces, centers, vr, vl


def _dump_trail_npz(args, idxs, right_verts, left_verts, faces_right, faces_left,
                    valid_r, valid_l, want_r, want_l, R_c2w, t_c2w):
    """Dump trail geometry to .npz so a rerun-only env can build the .rrd
    without importing torch/aitviewer (avoids the hawor<->rerun numpy clash)."""
    right, left, cam, cam_faces, centers, vr, vl = _sampled_trail_arrays(
        args, idxs, right_verts, left_verts, valid_r, valid_l, want_r, want_l, R_c2w, t_c2w)
    data = dict(
        sample_idx=np.asarray(idxs, np.int64),
        right_verts=right.astype(np.float32), left_verts=left.astype(np.float32),
        faces_right=np.asarray(faces_right, np.int32), faces_left=np.asarray(faces_left, np.int32),
        valid_r=vr, valid_l=vl,
        no_fade=np.asarray(bool(args.no_fade)), alpha_min=np.asarray(float(args.alpha_min)),
    )
    if not getattr(args, "no_camera", False):
        data.update(cam_verts=cam.astype(np.float32), cam_faces=np.asarray(cam_faces, np.int32),
                    cam_centers=centers.astype(np.float32))
    if not args.no_ground:
        gv, gf, gvc, _ = checkerboard_geometry(length=100, c1=0, c2=0, up="z")
        gv[:, 2] -= 2
        data.update(ground_v=gv.astype(np.float32), ground_f=np.asarray(gf, np.int32),
                    ground_c=(np.asarray(gvc)[:, :3] * 255).astype(np.uint8))
    os.makedirs(os.path.dirname(os.path.abspath(args.dump_npz)) or ".", exist_ok=True)
    np.savez_compressed(args.dump_npz, **data)
    print(f"saved {args.dump_npz}  ({len(idxs)} samples). convert with:\n"
          f"  python scripts/rrd_from_npz.py {args.dump_npz} <out.rrd>")


def _log_rerun_trail(args, idxs, right_verts, left_verts, faces_right, faces_left,
                     valid_r, valid_l, want_r, want_l, R_c2w, t_c2w):
    """Log the Figure-1 static trail to a rerun .rrd (no GL context needed).

    Same geometry the aitviewer path renders -- just logged as static rerun
    entities so it can be opened/orbited/screenshotted in the rerun web viewer.
    """
    try:
        import rerun as rr
    except ImportError:
        raise SystemExit("rerun not installed; `pip install rerun-sdk` "
                         "(see requirements-rerun-viewer.txt)")

    rr.init("hawor_world_trail", spawn=False)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    if not args.no_ground:
        gv, gf, gvc, _ = checkerboard_geometry(length=100, c1=0, c2=0, up="z")
        gv[:, 2] -= 2
        rr.log("world/ground",
               rr.Mesh3D(vertex_positions=gv, triangle_indices=gf,
                         vertex_colors=(np.asarray(gvc)[:, :3] * 255).astype(np.uint8)),
               static=True)

    mverts, mfaces, _ = camera_marker_geometry(args.frustum_radius, args.frustum_height)

    # rerun renders meshes opaque, so convey temporal order by fading older
    # poses toward WHITE (afterimage), never toward black.
    def shade(color, k):
        if args.no_fade or len(idxs) == 1:
            f = 1.0
        else:
            f = args.alpha_min + (1.0 - args.alpha_min) * (k / (len(idxs) - 1))
        return [float(c) * f + (1.0 - f) for c in color]

    for k, t in enumerate(idxs):
        if want_r and bool(valid_r[min(t, len(valid_r) - 1)]):
            rr.log(f"world/hand_right/t{t:05d}",
                   rr.Mesh3D(vertex_positions=right_verts[t], triangle_indices=faces_right,
                             vertex_normals=vertex_normals(right_verts[t], faces_right),
                             albedo_factor=shade(PURPLE, k)), static=True)
        if want_l and bool(valid_l[min(t, len(valid_l) - 1)]):
            rr.log(f"world/hand_left/t{t:05d}",
                   rr.Mesh3D(vertex_positions=left_verts[t], triangle_indices=faces_left,
                             vertex_normals=vertex_normals(left_verts[t], faces_left),
                             albedo_factor=shade(BLUE, k)), static=True)
        if not getattr(args, "no_camera", False):
            cam_v = np.einsum("ij,nj->ni", R_c2w[t], mverts) + t_c2w[t][None]
            rr.log(f"world/camera/t{t:05d}",
                   rr.Mesh3D(vertex_positions=cam_v, triangle_indices=mfaces,
                             albedo_factor=shade((0.6, 0.6, 0.6), k)), static=True)

    if not getattr(args, "no_camera", False):
        cam_centers = np.stack([t_c2w[t] for t in idxs], 0)
        rr.log("world/camera_trajectory",
               rr.LineStrips3D([cam_centers], colors=[255, 180, 0], radii=0.004), static=True)

    rr.save(args.rrd)
    print(f"saved {args.rrd}  ({len(idxs)} samples). open with:  rerun {args.rrd}")


def main():
    args = parse_args()
    seq = args.seq_folder
    out_png = args.out or os.path.join(seq, "world_traj.png")

    # ---- load hands (world space) ----
    wpath = os.path.join(seq, "world_space_res.pth")
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(wpath)
    to_t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
    pred_trans, pred_rot, pred_hand_pose, pred_betas = map(to_t, (pred_trans, pred_rot, pred_hand_pose, pred_betas))
    pred_valid = np.asarray(pred_valid)
    T = pred_trans.shape[1]
    vs, ve = 0, T - 1  # mirror demo.py slicing (mesh time dim = ve-vs)

    faces = get_mano_faces()
    faces_right = np.concatenate([faces, _PALM], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]

    ri, li = HAND2IDX["right"], HAND2IDX["left"]
    right_verts = run_mano(pred_trans[ri:ri+1, vs:ve], pred_rot[ri:ri+1, vs:ve],
                           pred_hand_pose[ri:ri+1, vs:ve], betas=pred_betas[ri:ri+1, vs:ve])["vertices"][0]
    left_verts = run_mano_left(pred_trans[li:li+1, vs:ve], pred_rot[li:li+1, vs:ve],
                               pred_hand_pose[li:li+1, vs:ve], betas=pred_betas[li:li+1, vs:ve])["vertices"][0]

    # ---- load camera trajectory (c2w) ----
    npz = _find_slam_npz(seq, args.slam_npz)
    R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam(npz)
    n_slam = int(R_c2w.shape[0])
    backend_txt = os.path.join(seq, "SLAM", "slam_backend.txt")
    use_dpvo = os.path.isfile(backend_txt) and open(backend_txt).read().strip().lower() == "dpvo"
    if use_dpvo or n_slam < ve:
        vis_idx = np.arange(vs, ve, dtype=np.int64)
        R_c2w, t_c2w = interpolate_slam_cameras_at_video_frames(npz, vis_idx)

    # ---- coordinate flip (matches demo.py) ----
    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32)
    R_c2w = torch.einsum("ij,njk->nik", R_x, R_c2w)
    t_c2w = torch.einsum("ij,nj->ni", R_x, t_c2w)
    left_verts = torch.einsum("ij,tnj->tni", R_x, left_verts.cpu())
    right_verts = torch.einsum("ij,tnj->tni", R_x, right_verts.cpu())

    if args.rebase and R_c2w.shape[0] > 0:
        R0, t0 = R_c2w[0], t_c2w[0]
        Ra = R0.transpose(-1, -2)
        ta = -torch.einsum("ij,j->i", Ra, t0)
        R_c2w = torch.einsum("ij,njk->nik", Ra, R_c2w)
        t_c2w = torch.einsum("ij,nj->ni", Ra, t_c2w) + ta[None]
        left_verts = torch.einsum("ij,tnj->tni", Ra, left_verts) + ta[None, None]
        right_verts = torch.einsum("ij,tnj->tni", Ra, right_verts) + ta[None, None]

    right_verts, left_verts = right_verts.numpy(), left_verts.numpy()
    R_c2w, t_c2w = R_c2w.numpy(), t_c2w.numpy()
    n = min(right_verts.shape[0], left_verts.shape[0], R_c2w.shape[0])

    # ---- optional: apply hand-depth-align α(t) to verts (ray-scale about camera) ----
    if args.align:
        alpha = _load_align_alpha(seq, args.align_npz)
        if alpha is None:
            print("[align] no hand_depth_align npz found; rendering unaligned verts")
        else:
            m = min(n, alpha.shape[0])
            a = alpha[:m].reshape(m, 1, 1)
            cp = t_c2w[:m].reshape(m, 1, 3)  # camera centers (world); ray-scale commutes with the R_x flip/rebase
            right_verts[:m] = cp + a * (right_verts[:m] - cp)
            left_verts[:m] = cp + a * (left_verts[:m] - cp)
            print(f"[align] applied α(t) to {m} frames (range [{float(alpha[:m].min()):.3f},{float(alpha[:m].max()):.3f}])")

    # ---- choose sample timesteps (optionally restricted to a frame window) ----
    # array index == video frame here (vs == 0), so frame_start/end clamp directly.
    lo = max(0, args.frame_start)
    hi = (n - 1) if args.frame_end < 0 else min(args.frame_end, n - 1)
    if lo > hi:
        raise ValueError(f"empty frame window: frame_start={args.frame_start} > frame_end={args.frame_end} (n={n})")
    want_r = args.hands in ("both", "right")
    want_l = args.hands in ("both", "left")
    valid_r = pred_valid[ri] if pred_valid.ndim == 2 else np.ones(T, bool)
    valid_l = pred_valid[li] if pred_valid.ndim == 2 else np.ones(T, bool)

    if args.stride > 0:
        idxs = sorted(set(range(lo, hi + 1, args.stride)))
    elif args.select == "scatter":
        # 2D-overlap-aware pick within [lo, hi] -- same recipe as taco_gt_trail's
        # --layout taco; keeps real pipeline positions but spreads them visually.
        from scripts.taco_gt_trail import _principal_plane, _select_scatter
        SUB = 120
        step = max(1, 778 // SUB)
        cand, pts_list, centroids = [], [], []
        for t in range(lo, hi + 1):
            pp = []
            if want_r and bool(valid_r[t]):
                pp.append(right_verts[t][::step])
            if want_l and bool(valid_l[t]):
                pp.append(left_verts[t][::step])
            if not pp:
                continue
            cand.append(t)
            p = np.concatenate(pp, 0)
            pts_list.append(p)
            centroids.append(p.mean(0))
        if not cand:
            raise ValueError(f"no valid hand poses in frames {lo}-{hi}")
        centroids = np.asarray(centroids)
        e1, e2 = _principal_plane(centroids)
        bboxes = [(float((p @ e1).min()), float((p @ e2).min()),
                   float((p @ e1).max()), float((p @ e2).max())) for p in pts_list]
        cent2d = np.stack([centroids @ e1, centroids @ e2], axis=1)
        sel = _select_scatter(cent2d, bboxes, args.num_samples,
                              spread=args.spread, overlap=args.overlap)
        idxs = sorted(cand[i] for i in sel)
        print(f"[select scatter] {len(cand)} valid frames in {lo}-{hi} -> {len(idxs)} poses "
              f"(spread={args.spread}, overlap<= {args.overlap})")
    else:
        idxs = sorted(set(np.linspace(lo, hi, max(2, args.num_samples)).round().astype(int).tolist()))

    # ---- npz backend: dump geometry only (hawor env, no rerun/GL), then stop ----
    if args.dump_npz:
        _dump_trail_npz(args, idxs, right_verts, left_verts, faces_right, faces_left,
                        valid_r, valid_l, want_r, want_l, R_c2w, t_c2w)
        return

    # ---- rerun backend: log the same trail, no GL needed, then stop ----
    if args.rrd:
        _log_rerun_trail(args, idxs, right_verts, left_verts, faces_right, faces_left,
                         valid_r, valid_l, want_r, want_l, R_c2w, t_c2w)
        return

    # ---- build static scene: one mesh per sampled timestep ----
    meshes = {}
    if not args.no_ground:
        gv, gf, gvc, _ = checkerboard_geometry(length=100, c1=0, c2=0, up="z")
        gv[:, 2] -= 2
        meshes["ground"] = Meshes(gv, gf, vertex_colors=gvc, name="ground", flat_shading=False)

    mverts, mfaces, mfcolors = camera_marker_geometry(args.frustum_radius, args.frustum_height)

    def alpha_for(k):
        if args.no_fade or len(idxs) == 1:
            return 1.0
        return args.alpha_min + (1.0 - args.alpha_min) * (k / (len(idxs) - 1))

    centers = []
    for k, t in enumerate(idxs):
        a = alpha_for(k)
        if want_r and bool(valid_r[min(t, len(valid_r) - 1)]):
            meshes[f"hand_r_{t}"] = Meshes(right_verts[t][None], faces_right, name=f"hand_r_{t}",
                                           material=Material(color=(*PURPLE, a), ambient=0.2), flat_shading=False)
            centers.append(right_verts[t].mean(0))
        if want_l and bool(valid_l[min(t, len(valid_l) - 1)]):
            meshes[f"hand_l_{t}"] = Meshes(left_verts[t][None], faces_left, name=f"hand_l_{t}",
                                           material=Material(color=(*BLUE, a), ambient=0.2), flat_shading=False)
            centers.append(left_verts[t].mean(0))
        if not args.no_camera:
            cam_v = np.einsum("ij,nj->ni", R_c2w[t], mverts) + t_c2w[t][None]
            meshes[f"cam_{t}"] = Meshes(cam_v[None], mfaces, face_colors=mfcolors, name=f"cam_{t}",
                                        material=Material(color=(0.5, 0.5, 0.5, a), ambient=0.2))
            centers.append(t_c2w[t])

    # ---- viewer camera (single static still) ----
    if args.view == "hawor":
        src = torch.tensor([0.463, -0.478, 2.456])
        tgt = torch.tensor([0.026, -0.481, -3.184])
        up = torch.tensor([1.0, 0.0, 0.0])
    else:
        pts = np.stack(centers, 0)
        center = pts.mean(0)
        extent = float(np.linalg.norm(pts - center, axis=1).max()) + 1e-3
        dist = extent * args.cam_dist_scale
        az, el = np.radians(args.cam_azim), np.radians(args.cam_elev)
        # up convention = +X (same as demo.py side view); orbit in the Y-Z plane
        dir_yz = np.array([np.sin(el), np.cos(el) * np.cos(az), np.cos(el) * np.sin(az)])
        src = torch.tensor(center + dist * dir_yz, dtype=torch.float32)
        tgt = torch.tensor(center, dtype=torch.float32)
        up = torch.tensor([1.0, 0.0, 0.0])

    view_cam = lookat_matrix(src, tgt, up)
    viewer_Rt = np.tile(view_cam[:3, :4].numpy(), (1, 1, 1))  # num_frames = 1
    K = np.array([[1000.0, 0, args.width / 2], [0, 1000.0, args.height / 2], [0, 0, 1]])
    data = viewer_utils.ViewerData(viewer_Rt, K, args.width, args.height)

    viewer = viewer_utils.ARCTICViewer(interactive=False, size=(args.width, args.height), render_types=["rgb"])
    tmp_out = os.path.join(os.path.dirname(out_png) or ".", "_world_traj_tmp")
    viewer.render_seq((meshes, data), out_folder=tmp_out)

    rgb = os.path.join(tmp_out, "images", "rgb", "0000.png")
    if not os.path.exists(rgb):
        raise RuntimeError(f"render produced no image at {rgb}")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    shutil.copy(rgb, out_png)
    shutil.rmtree(tmp_out, ignore_errors=True)
    print(f"saved {out_png}  ({len(idxs)} samples, hands={args.hands}, view={args.view})")


if __name__ == "__main__":
    main()
