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
    p.add_argument("--num_samples", type=int, default=8, help="hand/cam poses along the trail")
    p.add_argument("--stride", type=int, default=0, help="if >0, sample every N frames instead of num_samples")
    p.add_argument("--frame_start", type=int, default=0)
    p.add_argument("--frame_end", type=int, default=-1, help="-1 = last frame")
    p.add_argument("--hands", choices=["both", "left", "right"], default="both")
    p.add_argument("--no_fade", action="store_true", help="solid meshes instead of brightness ramp")
    p.add_argument("--alpha_min", type=float, default=0.30)
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


def _sample_idxs(args, n):
    lo = max(0, args.frame_start)
    hi = (n - 1) if args.frame_end < 0 else min(args.frame_end, n - 1)
    if lo > hi:
        raise ValueError(f"empty window: frame_start={args.frame_start} > frame_end={args.frame_end} (n={n})")
    if args.stride > 0:
        idxs = list(range(lo, hi + 1, args.stride))
    else:
        idxs = np.linspace(lo, hi, max(2, args.num_samples)).round().astype(int).tolist()
    return sorted(set(idxs))


def main():
    args = parse_args()
    triplet, seq = args.seq_id.rsplit("/", 1)
    triplet, seq = triplet.strip(), seq.strip()

    from hawor.utils.process import get_mano_faces
    from lib.vis.run_vis2 import camera_marker_geometry

    R_c2w, cam_pos, T_cam = _load_camera(args.data_root, triplet, seq)
    (rv, vr_all), (lv, vl_all) = _load_hand_verts(args.data_root, triplet, seq, T_cam, use_cuda=not args.cpu)
    T = min(T_cam, rv.shape[0], lv.shape[0])

    faces = get_mano_faces()
    faces_right = np.concatenate([faces, _PALM], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]

    want_r = args.hands in ("both", "right")
    want_l = args.hands in ("both", "left")

    idxs = _sample_idxs(args, T)
    mverts, mfaces, _ = camera_marker_geometry(args.frustum_radius, args.frustum_height)

    right = np.stack([rv[t] for t in idxs], 0)
    left = np.stack([lv[t] for t in idxs], 0)
    cam = np.stack([np.einsum("ij,nj->ni", R_c2w[t], mverts) + cam_pos[t][None] for t in idxs], 0)
    centers = np.stack([cam_pos[t] for t in idxs], 0)
    vr = np.array([want_r and bool(vr_all[t]) for t in idxs])
    vl = np.array([want_l and bool(vl_all[t]) for t in idxs])

    data = dict(
        sample_idx=np.asarray(idxs, np.int64),
        right_verts=right.astype(np.float32), left_verts=left.astype(np.float32),
        faces_right=faces_right.astype(np.int32), faces_left=faces_left.astype(np.int32),
        valid_r=vr, valid_l=vl,
        cam_verts=cam.astype(np.float32), cam_faces=np.asarray(mfaces, np.int32),
        cam_centers=centers.astype(np.float32),
        no_fade=np.asarray(bool(args.no_fade)), alpha_min=np.asarray(float(args.alpha_min)),
    )
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
