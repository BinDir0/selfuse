"""EgoVerse adapter (zarr episodes).

Per-episode ``<episode>.zarr`` arrays:
  images.front_1        (T,H,W,3) jpeg-encoded ego frames (required)
  obs_head_pose         (T,7) = [tx,ty,tz, qw,qx,qy,qz] device->world pose (SLAM world)
  left.obs_keypoints    (T,63) = 21 landmarks xyz, WORLD frame, metres (optional)
  right.obs_keypoints   (T,63) (optional)

Keypoints are already world-frame 3D joints (no MANO FK). Order assumed
MANO/MediaPipe == OpenPose-compatible; verify on production. Intrinsics are
per-embodiment, not per-frame; K is filled from attrs if present else a focal
guess (K is unused by the metrics, only carried for completeness).
"""

from __future__ import annotations

import glob
import os

import numpy as np

from .base import GTSequence, OPENPOSE_IDENTITY, permute_joints

JOINT_PERM = OPENPOSE_IDENTITY


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    """(N,4) [w,x,y,z] -> (N,3,3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def list_sequences(data_root: str, split_file: str | None = None, limit: int | None = None):
    seqs = [os.path.relpath(p, data_root) for p in sorted(glob.glob(os.path.join(data_root, "**", "*.zarr"), recursive=True))]
    return seqs[:limit] if limit else seqs


def _open_zarr(path: str):
    import zarr
    return zarr.open(path, mode="r")


def load_sequence(data_root: str, seq_id: str, use_cuda: bool = True, fps: float = 30.0) -> GTSequence:
    z = _open_zarr(os.path.join(data_root, seq_id))
    attrs = dict(z.attrs)
    fps = float(attrs.get("fps", fps))

    head = np.asarray(z["obs_head_pose"][:], dtype=np.float64)  # (T,7)
    T = head.shape[0]
    t_c2w = head[:, :3]
    R_c2w = _quat_wxyz_to_R(head[:, 3:7])
    R_w2c = np.transpose(R_c2w, (0, 2, 1))
    t_w2c = -np.einsum("tij,tj->ti", R_w2c, t_c2w)

    joints = np.full((2, T, 21, 3), np.nan, dtype=np.float32)
    valid = np.zeros((2, T), dtype=bool)
    for hand_idx, key in ((0, "left.obs_keypoints"), (1, "right.obs_keypoints")):
        if key in z:
            kp = np.asarray(z[key][:], dtype=np.float32).reshape(T, 21, 3)
            joints[hand_idx] = kp
            valid[hand_idx] = np.isfinite(kp).all(axis=(1, 2))
    joints = permute_joints(joints, JOINT_PERM)

    # K: only carried for completeness (metrics don't use it).
    feats = attrs.get("features", {})
    img_shape = feats.get("images.front_1", {}).get("shape", [1080, 1440, 3])
    h, w = img_shape[0], img_shape[1]
    f = float(attrs.get("focal", 0.8 * w))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)

    return GTSequence(
        seq_id=seq_id, dataset="egoverse", fps=fps, K=K,
        cam_R_w2c=R_w2c, cam_t_w2c=t_w2c, joints_world=joints, valid=valid,
        frame_paths=None, video_path=os.path.join(data_root, seq_id),  # zarr; prepare extracts frames
    )
