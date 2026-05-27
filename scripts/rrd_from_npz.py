#!/usr/bin/env python3
"""Build a HaWoR-Figure-1 trail .rrd from a trail .npz.

Companion to `render_world_traj.py --dump_npz`. The npz is produced in the
`hawor` conda env (torch/MANO/SLAM); this converter only needs numpy + rerun,
so run it in a SEPARATE env where rerun-sdk is installed -- that sidesteps the
numpy version clash between rerun-sdk and the hawor env.

  # env with rerun only (e.g. a throwaway venv):
  python -m venv /tmp/rrenv && /tmp/rrenv/bin/pip install rerun-sdk==0.27.3
  /tmp/rrenv/bin/python scripts/rrd_from_npz.py trail.npz hand_cam.rrd

  # then view / screenshot:
  rerun hand_cam.rrd
"""
import argparse

import numpy as np

PURPLE = (0.804, 0.600, 0.820)   # right hand
BLUE = (0.207, 0.596, 0.792)     # left hand


def vertex_normals(verts, faces):
    """Smooth per-vertex normals so rerun shades the mesh with a soft
    gradient instead of a flat single colour (mirrors render_world_traj)."""
    verts = np.asarray(verts, np.float64)
    faces = np.asarray(faces, np.int64)
    nrm = np.zeros_like(verts)
    tris = verts[faces]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    for i in range(3):
        np.add.at(nrm, faces[:, i], fn)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return (nrm / ln).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz", help="trail npz from render_world_traj.py --dump_npz")
    p.add_argument("rrd", help="output .rrd path")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import rerun as rr
    except ImportError:
        raise SystemExit("rerun not installed in this env; `pip install rerun-sdk==0.27.3`")

    d = np.load(args.npz, allow_pickle=False)
    idxs = d["sample_idx"]
    n = len(idxs)
    no_fade = bool(d["no_fade"])
    alpha_min = float(d["alpha_min"])

    def shade(color, k):
        f = 1.0 if (no_fade or n == 1) else alpha_min + (1.0 - alpha_min) * (k / (n - 1))
        return [float(c) * f for c in color]

    rr.init("hawor_world_trail", spawn=False)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    if "ground_v" in d.files:
        rr.log("world/ground",
               rr.Mesh3D(vertex_positions=d["ground_v"], triangle_indices=d["ground_f"],
                         vertex_colors=d["ground_c"]), static=True)

    for k in range(n):
        t = int(idxs[k])
        if bool(d["valid_r"][k]):
            rr.log(f"world/hand_right/t{t:05d}",
                   rr.Mesh3D(vertex_positions=d["right_verts"][k], triangle_indices=d["faces_right"],
                             vertex_normals=vertex_normals(d["right_verts"][k], d["faces_right"]),
                             albedo_factor=shade(PURPLE, k)), static=True)
        if bool(d["valid_l"][k]):
            rr.log(f"world/hand_left/t{t:05d}",
                   rr.Mesh3D(vertex_positions=d["left_verts"][k], triangle_indices=d["faces_left"],
                             vertex_normals=vertex_normals(d["left_verts"][k], d["faces_left"]),
                             albedo_factor=shade(BLUE, k)), static=True)
        rr.log(f"world/camera/t{t:05d}",
               rr.Mesh3D(vertex_positions=d["cam_verts"][k], triangle_indices=d["cam_faces"],
                         albedo_factor=shade((0.6, 0.6, 0.6), k)), static=True)

    rr.log("world/camera_trajectory",
           rr.LineStrips3D([d["cam_centers"]], colors=[255, 180, 0], radii=0.004), static=True)

    rr.save(args.rrd)
    print(f"saved {args.rrd}  ({n} samples). open with:  rerun {args.rrd}")


if __name__ == "__main__":
    main()
