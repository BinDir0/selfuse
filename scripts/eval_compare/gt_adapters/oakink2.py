"""OakInk2 adapter (layout/keys verified against convert_oakink2_to_wds.py).

``data_root`` is the OakInk2 ROOT containing ``data/``, ``anno_preview/``, ``program/``.

GT in ``anno_preview/<token>.pkl`` (token = seq key with '/' written as '++'):
  cam_def                 {serial: layout_name}; ego layout name = "egocentric"
  cam_intr["egocentric"][frame]   (3,3)          # params keyed by LAYOUT NAME
  cam_extr["egocentric"][frame]   (4,4) world->cam
  raw_mano[frame]         {lh/rh__pose_coeffs (1,16,4) quat[w,x,y,z], __tsl (1,3), __betas (1,10)}
Ego frames are keyed by the SERIAL (reverse of cam_def):
  <root>/data/<token>/<serial>/<frame:06>.png|jpg

World frame, metres. MANO params FK'd with the repo's run_mano (joint order ==
predictions). center/placement: world_joints = mano(global,pose,betas) + tsl.
"""

from __future__ import annotations

import glob
import os
import pickle

import numpy as np

from .base import GTSequence, mano_fk_world

EGO_CAM_DEFAULT = "egocentric"  # OakInk2 camera-layout name of the head-mounted view


def _to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _quat_wxyz_to_aa(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w = np.clip(q[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.clip(1.0 - w * w, 0.0, 1.0))
    small = sin_half < 1e-8
    axis = np.where(small[..., None], np.array([1.0, 0.0, 0.0]), q[..., 1:] / np.where(small, 1.0, sin_half)[..., None])
    return axis * angle[..., None]


def list_sequences(data_root: str, split_file: str | None = None, limit: int | None = None):
    anno = sorted(glob.glob(os.path.join(data_root, "anno_preview", "*.pkl")))
    seqs = [os.path.splitext(os.path.basename(p))[0].replace("++", "/") for p in anno]
    return seqs[:limit] if limit else seqs


def _image_path(data_root, token, serial, fid):
    for ext in ("png", "jpg", "jpeg"):
        fp = os.path.join(data_root, "data", token, serial, f"{fid:0>6}.{ext}")
        if os.path.exists(fp):
            return fp
    return None


def _tar_frame_index(data_root, token, serial):
    """Map fid -> member name for frames of `serial` inside data/<token>.tar.

    Returns (tar_path, {fid: member}) or (None, {}). Matches members whose path has
    `serial` as a component and a 6-digit image basename.
    """
    import re
    import tarfile

    tar_path = os.path.join(data_root, "data", f"{token}.tar")
    if not os.path.exists(tar_path):
        return None, {}
    idx = {}
    pat = re.compile(rf"(?:^|/){re.escape(serial)}/0*([0-9]+)\.(?:png|jpe?g)$")
    with tarfile.open(tar_path, "r") as tf:
        for m in tf.getnames():
            mm = pat.search(m)
            if mm:
                idx[int(mm.group(1))] = m
    return tar_path, idx


def load_sequence(
    data_root: str, seq_id: str, use_cuda: bool = True, fps: float = 30.0, ego_cam: str = EGO_CAM_DEFAULT
) -> GTSequence:
    token = seq_id.replace("/", "++")
    with open(os.path.join(data_root, "anno_preview", f"{token}.pkl"), "rb") as f:
        anno = pickle.load(f)

    serial = {v: k for k, v in anno["cam_def"].items()}[ego_cam]
    cam_intr, cam_extr, raw_mano = anno["cam_intr"][ego_cam], anno["cam_extr"][ego_cam], anno["raw_mano"]

    # frames present in cam params, MANO, and on disk (loose files OR inside <token>.tar)
    cand = sorted(set(cam_extr) & set(cam_intr) & set(raw_mano))
    frame_paths, tar_path, tar_members = None, None, None
    fids = [f for f in cand if _image_path(data_root, token, serial, f) is not None]
    if fids:
        frame_paths = [_image_path(data_root, token, serial, f) for f in fids]
    else:
        tar_path, tar_idx = _tar_frame_index(data_root, token, serial)
        if tar_idx:
            fids = [f for f in cand if f in tar_idx]
            tar_members = [tar_idx[f] for f in fids]
    if not fids:
        raise FileNotFoundError(
            f"no ego frames for {seq_id} (serial {serial}); checked loose files and data/{token}.tar"
        )
    T = len(fids)

    K = np.asarray(cam_intr[fids[0]], dtype=np.float64).reshape(3, 3)
    R_w2c = np.zeros((T, 3, 3)); t_w2c = np.zeros((T, 3))
    for i, f in enumerate(fids):
        E = np.asarray(cam_extr[f], dtype=np.float64).reshape(4, 4)
        R_w2c[i], t_w2c[i] = E[:3, :3], E[:3, 3]

    joints = np.full((2, T, 21, 3), np.nan, dtype=np.float32)
    valid = np.zeros((2, T), dtype=bool)
    for hand_idx, pref, is_right in ((0, "lh", False), (1, "rh", True)):
        g_aa = np.zeros((T, 3), np.float32); pose_aa = np.zeros((T, 45), np.float32)
        tsl = np.zeros((T, 3), np.float32); betas = np.zeros((T, 10), np.float32)
        present = np.zeros(T, bool)
        for i, f in enumerate(fids):
            entry = raw_mano.get(f, {})
            if f"{pref}__pose_coeffs" not in entry:
                continue
            coeffs = _to_np(entry[f"{pref}__pose_coeffs"]).reshape(16, 4)
            aa = _quat_wxyz_to_aa(coeffs)
            g_aa[i] = aa[0]; pose_aa[i] = aa[1:16].reshape(-1)
            tsl[i] = _to_np(entry[f"{pref}__tsl"]).reshape(3)
            betas[i] = _to_np(entry[f"{pref}__betas"]).reshape(10)
            present[i] = True
        if present.any():
            j = mano_fk_world(g_aa, pose_aa, tsl, betas, is_right=is_right, use_cuda=use_cuda)
            joints[hand_idx] = j
            valid[hand_idx] = present & np.isfinite(j).all(axis=(1, 2))

    return GTSequence(
        seq_id=seq_id, dataset="oakink2", fps=fps, K=K,
        cam_R_w2c=R_w2c, cam_t_w2c=t_w2c, joints_world=joints, valid=valid,
        frame_paths=frame_paths, frame_archive=tar_path, frame_members=tar_members,
    )
