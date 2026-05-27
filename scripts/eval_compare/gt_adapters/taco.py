"""TACO adapter (layout/keys verified against convert_taco_to_wds.py).

``data_root`` is the TACO root. Layout is ``<root>/<TYPE>/<triplet>/<seq>/``:
  Egocentric_RGB_Videos/<triplet>/<seq>/color.mp4
  Egocentric_Camera_Parameters/<triplet>/<seq>/egocentric_intrinsic.txt       (3,3)
  Egocentric_Camera_Parameters/<triplet>/<seq>/egocentric_frame_extrinsic.npy (N,4,4) world->cam
  Hand_Poses/<triplet>/<seq>/{left,right}_hand.pkl         dict[frame]->{hand_pose(48) aa, hand_trans(3)}
  Hand_Poses/<triplet>/<seq>/{left,right}_hand_shape.pkl   {hand_shape(10)}

triplet is e.g. "(dust, roller, pan)"; sequence ids come from
data_lists/v1_egocentric_data_available_sequences.txt lines "(triplet) seq".
seq_id is encoded as "<triplet>/<seq>". World frame, metres (trans_unit=1.0). MANO
params FK'd with the repo's run_mano so joint order matches predictions.
"""

from __future__ import annotations

import glob
import os
import pickle

import numpy as np

from .base import GTSequence, mano_fk_world


def _to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def list_sequences(data_root: str, split_file: str | None = None, limit: int | None = None):
    seqs: list[str] = []
    if split_file and os.path.exists(split_file):
        for line in open(split_file):
            line = line.strip()
            if not line:
                continue
            triplet, seq = line.rsplit(" ", 1)  # "(dust, roller, pan) 20230927_032"
            seqs.append(f"{triplet.strip()}/{seq.strip()}")
    else:
        for hp in sorted(glob.glob(os.path.join(data_root, "Hand_Poses", "*", "*"))):
            if os.path.isdir(hp):
                seqs.append(os.path.relpath(hp, os.path.join(data_root, "Hand_Poses")))
    return seqs[:limit] if limit else seqs


def _split_seq(seq_id: str) -> tuple[str, str]:
    triplet, seq = seq_id.rsplit("/", 1)  # triplet has no '/'
    return triplet, seq


def load_sequence(
    data_root: str, seq_id: str, use_cuda: bool = True, fps: float = 30.0, trans_unit: float = 1.0
) -> GTSequence:
    triplet, seq = _split_seq(seq_id)
    cam_dir = os.path.join(data_root, "Egocentric_Camera_Parameters", triplet, seq)
    hp_dir = os.path.join(data_root, "Hand_Poses", triplet, seq)

    K = np.loadtxt(os.path.join(cam_dir, "egocentric_intrinsic.txt")).reshape(3, 3).astype(np.float64)
    extr = np.load(os.path.join(cam_dir, "egocentric_frame_extrinsic.npy"))  # (N,4,4) world->cam
    T = extr.shape[0]
    R_w2c = extr[:, :3, :3].astype(np.float64)
    t_w2c = extr[:, :3, 3].astype(np.float64)

    joints = np.full((2, T, 21, 3), np.nan, dtype=np.float32)
    valid = np.zeros((2, T), dtype=bool)
    for hand_idx, name, is_right in ((0, "left", False), (1, "right", True)):
        pose_pkl = os.path.join(hp_dir, f"{name}_hand.pkl")
        if not os.path.exists(pose_pkl):
            continue
        with open(pose_pkl, "rb") as f:
            data = pickle.load(f)
        shape = np.zeros(10, np.float32)
        shape_pkl = os.path.join(hp_dir, f"{name}_hand_shape.pkl")
        if os.path.exists(shape_pkl):
            with open(shape_pkl, "rb") as f:
                sd = pickle.load(f)
            shape = _to_np(sd["hand_shape"] if isinstance(sd, dict) and "hand_shape" in sd else sd).reshape(10)

        g_aa = np.zeros((T, 3), np.float32); pose_aa = np.zeros((T, 45), np.float32)
        tsl = np.zeros((T, 3), np.float32); betas = np.tile(shape, (T, 1)).astype(np.float32)
        present = np.zeros(T, bool)
        for i in range(T):
            entry = data.get(i, data.get(str(i)))
            if entry is None:
                continue
            full = _to_np(entry["hand_pose"]).reshape(-1)  # 48 axis-angle (1 global + 15)
            g_aa[i] = full[:3]; pose_aa[i] = full[3:48]
            tsl[i] = _to_np(entry["hand_trans"]).reshape(3) * trans_unit
            present[i] = True
        if present.any():
            j = mano_fk_world(g_aa, pose_aa, tsl, betas, is_right=is_right, use_cuda=use_cuda)
            joints[hand_idx] = j
            valid[hand_idx] = present & np.isfinite(j).all(axis=(1, 2))

    video = os.path.join(data_root, "Egocentric_RGB_Videos", triplet, seq, "color.mp4")
    return GTSequence(
        seq_id=seq_id, dataset="taco", fps=fps, K=K,
        cam_R_w2c=R_w2c, cam_t_w2c=t_w2c, joints_world=joints, valid=valid,
        video_path=video if os.path.exists(video) else None,
    )
