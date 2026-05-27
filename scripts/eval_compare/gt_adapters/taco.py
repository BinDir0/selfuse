"""TACO adapter.

Per sequence ``<data_root>/<triplet>/<seq_id>/``:
  Hand_Poses/left_hand.pkl    dict[frame] -> {hand_pose (48,) axis-axis, hand_trans (3,)}
  Hand_Poses/left_hand_shape.pkl   {hand_shape (10,)}  (shared across frames)
  Hand_Poses/right_hand.pkl / right_hand_shape.pkl
  Egocentric_RGB_Videos/color.mp4
  Egocentric_Camera_Parameters/egocentric_intrinsic.txt        (3,3)
  Egocentric_Camera_Parameters/egocentric_frame_extrinsic.npy  (N,4,4) world->cam

hand_pose is 48 axis-angle = 1 global (3) + 15 joints (45). World frame. FK'd with the
repo's MANO so joint order matches predictions. ``trans_unit`` scales translation to
metres (default 1.0 == already metres; verify on production).
"""

from __future__ import annotations

import glob
import os
import pickle

import numpy as np

from .base import GTSequence, mano_fk_world


def list_sequences(data_root: str, split_file: str | None = None, limit: int | None = None):
    seqs: list[str] = []
    if split_file and os.path.exists(split_file):
        for line in open(split_file):
            line = line.strip()
            if not line:
                continue
            # e.g. "(stir, spoon, bowl) 20231105_019"  OR  "20231105_019,train"
            tok = line.replace(",", " ").split()
            seqs.append(tok[-2] if len(tok) >= 2 and tok[-1] in ("train", "test", "val") else tok[-1])
    else:
        for hp in sorted(glob.glob(os.path.join(data_root, "*", "*", "Hand_Poses"))):
            seqs.append(os.path.relpath(os.path.dirname(hp), data_root))
    # de-dup, keep order
    out, seen = [], set()
    for s in seqs:
        if s not in seen:
            seen.add(s); out.append(s)
    return out[:limit] if limit else out


def _resolve_seq_dir(data_root: str, seq_id: str) -> str:
    if os.path.isdir(os.path.join(data_root, seq_id, "Hand_Poses")):
        return os.path.join(data_root, seq_id)
    # seq_id may be just the date id -> search one level of triplet folders
    hits = glob.glob(os.path.join(data_root, "*", seq_id, "Hand_Poses"))
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError(f"cannot locate TACO sequence {seq_id} under {data_root}")


def load_sequence(
    data_root: str, seq_id: str, use_cuda: bool = True, fps: float = 30.0, trans_unit: float = 1.0
) -> GTSequence:
    seq_dir = _resolve_seq_dir(data_root, seq_id)
    cam_dir = os.path.join(seq_dir, "Egocentric_Camera_Parameters")
    K = np.loadtxt(os.path.join(cam_dir, "egocentric_intrinsic.txt")).reshape(3, 3).astype(np.float64)
    extr = np.load(os.path.join(cam_dir, "egocentric_frame_extrinsic.npy"))  # (N,4,4) world->cam
    T = extr.shape[0]
    R_w2c = extr[:, :3, :3].astype(np.float64)
    t_w2c = extr[:, :3, 3].astype(np.float64)

    joints = np.full((2, T, 21, 3), np.nan, dtype=np.float32)
    valid = np.zeros((2, T), dtype=bool)
    hp_dir = os.path.join(seq_dir, "Hand_Poses")
    for hand_idx, name, is_right in ((0, "left", False), (1, "right", True)):
        pose_pkl = os.path.join(hp_dir, f"{name}_hand.pkl")
        shape_pkl = os.path.join(hp_dir, f"{name}_hand_shape.pkl")
        if not os.path.exists(pose_pkl):
            continue
        with open(pose_pkl, "rb") as f:
            data = pickle.load(f)
        shape = np.zeros(10, np.float32)
        if os.path.exists(shape_pkl):
            with open(shape_pkl, "rb") as f:
                sd = pickle.load(f)
            shape = np.asarray(sd["hand_shape"] if isinstance(sd, dict) and "hand_shape" in sd else sd).reshape(10)

        g_aa = np.zeros((T, 3), np.float32)
        pose_aa = np.zeros((T, 45), np.float32)
        tsl = np.zeros((T, 3), np.float32)
        betas = np.tile(shape, (T, 1)).astype(np.float32)
        present = np.zeros(T, bool)
        for i in range(T):
            entry = data.get(i, data.get(str(i)))
            if entry is None:
                continue
            full = np.asarray(entry["hand_pose"]).reshape(-1)  # 48 axis-angle
            g_aa[i] = full[:3]
            pose_aa[i] = full[3:48]
            tsl[i] = np.asarray(entry["hand_trans"]).reshape(3) * trans_unit
            present[i] = True
        if present.any():
            j = mano_fk_world(g_aa, pose_aa, tsl, betas, is_right=is_right, use_cuda=use_cuda)
            joints[hand_idx] = j
            valid[hand_idx] = present & np.isfinite(j).all(axis=(1, 2))

    video = os.path.join(seq_dir, "Egocentric_RGB_Videos", "color.mp4")
    return GTSequence(
        seq_id=seq_id, dataset="taco", fps=fps, K=K,
        cam_R_w2c=R_w2c, cam_t_w2c=t_w2c, joints_world=joints, valid=valid,
        video_path=video if os.path.exists(video) else None,
    )
