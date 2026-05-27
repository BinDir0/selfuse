"""OakInk2 adapter.

GT lives in ``anno_preview/<seq_token>.pkl`` (seq key '/' -> '++' in the filename):
  cam_intr[cam][frame]   (3,3)
  cam_extr[cam][frame]   (4,4) world->cam
  raw_mano[frame][...]   rh__pose_coeffs/lh__pose_coeffs (1,16,4) quaternion [w,x,y,z],
                         rh__tsl/lh__tsl (1,3) world translation,
                         rh__betas/lh__betas (1,10)
  frame_id_list          ordered frame ids

Ego RGB frames: ``<data_root>/<seq_dir>/<ego_cam>/<frame>.jpg``. MANO params are FK'd
with the repo's own ``run_mano``/``run_mano_left`` so joint order matches predictions.
"""

from __future__ import annotations

import glob
import os
import pickle

import numpy as np

from .base import GTSequence, mano_fk_world

EGO_CAM_DEFAULT = "egocentric"  # camera key/serial of the head-mounted view


def _quat_wxyz_to_aa(q: np.ndarray) -> np.ndarray:
    """(...,4) [w,x,y,z] -> (...,3) angle-axis."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w = np.clip(q[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.clip(1.0 - w * w, 0.0, 1.0))
    small = sin_half < 1e-8
    axis = np.where(small[..., None], np.array([1.0, 0.0, 0.0]), q[..., 1:] / np.where(small, 1.0, sin_half)[..., None])
    return axis * angle[..., None]


def _anno_path(data_root: str, seq_id: str) -> str:
    token = seq_id.replace("/", "++")
    return os.path.join(data_root, "anno_preview", f"{token}.pkl")


def list_sequences(data_root: str, split_file: str | None = None, limit: int | None = None):
    anno = sorted(glob.glob(os.path.join(data_root, "anno_preview", "*.pkl")))
    seqs = [os.path.splitext(os.path.basename(p))[0].replace("++", "/") for p in anno]
    return seqs[:limit] if limit else seqs


def load_sequence(
    data_root: str, seq_id: str, use_cuda: bool = True, fps: float = 30.0, ego_cam: str = EGO_CAM_DEFAULT
) -> GTSequence:
    with open(_anno_path(data_root, seq_id), "rb") as f:
        anno = pickle.load(f)
    frame_ids = list(anno["frame_id_list"])
    T = len(frame_ids)

    K = None
    R_w2c = np.zeros((T, 3, 3))
    t_w2c = np.zeros((T, 3))
    extr, intr = anno["cam_extr"][ego_cam], anno["cam_intr"][ego_cam]
    for i, fid in enumerate(frame_ids):
        E = np.asarray(extr[fid]).reshape(4, 4)
        R_w2c[i], t_w2c[i] = E[:3, :3], E[:3, 3]
        if K is None:
            K = np.asarray(intr[fid]).reshape(3, 3).astype(np.float64)

    # collect per-hand MANO params over frames
    raw = anno["raw_mano"]
    joints = np.full((2, T, 21, 3), np.nan, dtype=np.float32)
    valid = np.zeros((2, T), dtype=bool)
    for hand_idx, pref, is_right in ((0, "lh", False), (1, "rh", True)):
        g_aa = np.zeros((T, 3), np.float32)
        pose_aa = np.zeros((T, 45), np.float32)
        tsl = np.zeros((T, 3), np.float32)
        betas = np.zeros((T, 10), np.float32)
        present = np.zeros(T, bool)
        for i, fid in enumerate(frame_ids):
            entry = raw.get(fid, {})
            coeff_key = f"{pref}__pose_coeffs"
            if coeff_key not in entry:
                continue
            coeffs = np.asarray(entry[coeff_key]).reshape(16, 4)
            aa = _quat_wxyz_to_aa(coeffs)  # (16,3)
            g_aa[i] = aa[0]
            pose_aa[i] = aa[1:16].reshape(-1)
            tsl[i] = np.asarray(entry[f"{pref}__tsl"]).reshape(3)
            betas[i] = np.asarray(entry[f"{pref}__betas"]).reshape(10)
            present[i] = True
        if present.any():
            j = mano_fk_world(g_aa, pose_aa, tsl, betas, is_right=is_right, use_cuda=use_cuda)
            joints[hand_idx] = j
            valid[hand_idx] = present & np.isfinite(j).all(axis=(1, 2))

    frames = sorted(glob.glob(os.path.join(data_root, seq_id, ego_cam, "*.jpg")))
    return GTSequence(
        seq_id=seq_id, dataset="oakink2", fps=fps, K=K if K is not None else np.eye(3),
        cam_R_w2c=R_w2c, cam_t_w2c=t_w2c, joints_world=joints, valid=valid,
        frame_paths=frames or None,
    )
