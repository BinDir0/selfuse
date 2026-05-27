"""H2O adapter.

Egocentric view = ``cam4``. Per-frame files under ``<data_root>/<seq_id>/cam4/``:
  rgb/<f>.png             1280x720 ego frame
  cam_pose/<f>.txt        16 floats = 4x4 camera-to-world matrix (row-major)
  hand_pose/<f>.txt       128 floats = [L_flag, L_kp(21*3), R_flag, R_kp(21*3)],
                          3D joints in CAMERA frame, metres
  cam_intrinsics.txt      6 floats = fx fy cx cy width height

Sequences are dirs like ``subject1/h1/0``; ids come from pose_lists/pose_*.txt or a
filesystem walk. Joint order is assumed already OpenPose-compatible (verify on prod).
"""

from __future__ import annotations

import glob
import os

import numpy as np

from .base import GTSequence, OPENPOSE_IDENTITY, permute_joints

EGO_CAM = "cam4"
JOINT_PERM = OPENPOSE_IDENTITY  # override after verify_joints.py if H2O order differs


def list_sequences(data_root: str, split_file: str | None = None, limit: int | None = None):
    seqs: list[str] = []
    if split_file and os.path.exists(split_file):
        seen = set()
        for line in open(split_file):
            line = line.strip()
            if not line:
                continue
            # e.g. subject1/h1/0/cam4/rgb/000000.png -> subject1/h1/0
            parts = line.split("/")
            if EGO_CAM in parts:
                sid = "/".join(parts[: parts.index(EGO_CAM)])
                if sid not in seen:
                    seen.add(sid)
                    seqs.append(sid)
    else:
        for cam_dir in sorted(glob.glob(os.path.join(data_root, "*", "*", "*", EGO_CAM))):
            seqs.append(os.path.relpath(os.path.dirname(cam_dir), data_root))
    return seqs[:limit] if limit else seqs


def _read_intrinsics(path: str) -> np.ndarray:
    v = np.loadtxt(path).reshape(-1)
    fx, fy, cx, cy = v[0], v[1], v[2], v[3]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def load_sequence(data_root: str, seq_id: str, use_cuda: bool = True, fps: float = 30.0) -> GTSequence:
    cam_dir = os.path.join(data_root, seq_id, EGO_CAM)
    rgb = sorted(glob.glob(os.path.join(cam_dir, "rgb", "*.png")))
    if not rgb:
        rgb = sorted(glob.glob(os.path.join(cam_dir, "rgb", "*.jpg")))
    if not rgb:
        raise FileNotFoundError(f"no ego frames under {cam_dir}/rgb")

    K = _read_intrinsics(os.path.join(cam_dir, "cam_intrinsics.txt"))
    T = len(rgb)
    R_w2c = np.zeros((T, 3, 3))
    t_w2c = np.zeros((T, 3))
    joints = np.full((2, T, 21, 3), np.nan, dtype=np.float32)
    valid = np.zeros((2, T), dtype=bool)

    for i, img in enumerate(rgb):
        fid = os.path.splitext(os.path.basename(img))[0]
        c2w = np.loadtxt(os.path.join(cam_dir, "cam_pose", f"{fid}.txt")).reshape(4, 4)
        R_c2w, t_c2w = c2w[:3, :3], c2w[:3, 3]
        R_w2c[i] = R_c2w.T
        t_w2c[i] = -R_c2w.T @ t_c2w

        hp = np.loadtxt(os.path.join(cam_dir, "hand_pose", f"{fid}.txt")).reshape(-1)
        # [L_flag, L_kp(63), R_flag, R_kp(63)]
        for hand_idx, base in ((0, 0), (1, 64)):
            flag = hp[base]
            kp = hp[base + 1: base + 64].reshape(21, 3)
            if flag > 0.5 and np.isfinite(kp).all():
                # camera-frame -> world via this frame's c2w
                joints[hand_idx, i] = (R_c2w @ kp.T).T + t_c2w
                valid[hand_idx, i] = True

    joints = permute_joints(joints, JOINT_PERM)
    return GTSequence(
        seq_id=seq_id, dataset="h2o", fps=fps, K=K,
        cam_R_w2c=R_w2c, cam_t_w2c=t_w2c, joints_world=joints, valid=valid,
        frame_paths=rgb,
    )
