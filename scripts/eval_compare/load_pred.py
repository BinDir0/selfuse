"""Load a HaWoR prediction folder (fork OR upstream — identical format) into the
common structure the evaluator consumes.

Both ``scripts/batch_infer.py`` (fork) and ``/root/HaWoR/demo.py`` (upstream) write:
  <seq_folder>/world_space_res.pth                          (joblib list of 5 tensors)
  <seq_folder>/SLAM/hawor_slam_w_scale_<start>_<end>.npz    (camera trajectory + scale)

This module reuses the repo's own MANO forward kinematics and SLAM-camera loaders so
that predicted joints are produced exactly as in inference/visualisation — guaranteeing
the 21-joint OpenPose ordering matches what the GT adapters target.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import joblib
import numpy as np
import torch

from hawor.utils.process import run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam

LEFT, RIGHT = 0, 1  # dim-0 convention of world_space_res.pth


@dataclass
class Prediction:
    joints_world: np.ndarray   # (2, T, 21, 3) metres, OpenPose order, [left, right]
    valid: np.ndarray          # (2, T) bool
    cam_pos_world: np.ndarray  # (T, 3) camera centre in HaWoR world, metric (scaled)
    cam_R_c2w: np.ndarray      # (T, 3, 3)
    scale: float               # HaWoR SLAM metric scale
    start_idx: int
    end_idx: int


def _to_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def find_slam_npz(seq_folder: str) -> str:
    matches = sorted(glob.glob(os.path.join(seq_folder, "SLAM", "hawor_slam_w_scale_*.npz")))
    if not matches:
        raise FileNotFoundError(f"no SLAM npz under {seq_folder}/SLAM/")
    return matches[-1]


def _parse_start_end(npz_path: str) -> tuple[int, int]:
    m = re.search(r"hawor_slam_w_scale_(\d+)_(\d+)\.npz$", os.path.basename(npz_path))
    if not m:
        return 0, 0
    return int(m.group(1)), int(m.group(2))


def _hand_joints_world(pred_trans, pred_rot, pred_hand_pose, pred_betas, hand_idx, use_cuda):
    """Run MANO FK for one hand. Inputs are angle-axis (run_mano converts internally)."""
    trans = _to_tensor(pred_trans[hand_idx])[None]        # (1, T, 3)
    rot = _to_tensor(pred_rot[hand_idx])[None]            # (1, T, 3) angle-axis
    pose = _to_tensor(pred_hand_pose[hand_idx])[None]     # (1, T, 45) angle-axis
    betas = _to_tensor(pred_betas[hand_idx])[None]        # (1, T, 10)
    fk = run_mano if hand_idx == RIGHT else run_mano_left
    out = fk(trans, rot, pose, betas=betas, use_cuda=use_cuda)
    return out["joints"][0].detach().cpu().numpy()        # (T, 21, 3)


def load_prediction(seq_folder: str, use_cuda: bool = True) -> Prediction:
    world_res = os.path.join(seq_folder, "world_space_res.pth")
    if not os.path.exists(world_res):
        raise FileNotFoundError(world_res)
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(world_res)

    n_hands = _to_tensor(pred_trans).shape[0]
    n_frames = _to_tensor(pred_trans).shape[1]
    joints = np.full((2, n_frames, 21, 3), np.nan, dtype=np.float32)
    for h in range(min(n_hands, 2)):
        joints[h] = _hand_joints_world(pred_trans, pred_rot, pred_hand_pose, pred_betas, h, use_cuda)

    valid = _to_tensor(pred_valid).numpy().astype(bool)
    if valid.shape[0] < 2:  # pad single-hand outputs
        valid = np.vstack([valid, np.zeros((2 - valid.shape[0], n_frames), bool)])

    npz_path = find_slam_npz(seq_folder)
    start_idx, end_idx = _parse_start_end(npz_path)
    _, _, r_c2w, t_c2w = load_slam_cam(npz_path)  # tensors; t already * scale (metric)
    scale = float(np.load(npz_path, allow_pickle=True)["scale"])

    return Prediction(
        joints_world=joints,
        valid=valid,
        cam_pos_world=t_c2w.cpu().numpy().astype(np.float64),
        cam_R_c2w=r_c2w.cpu().numpy().astype(np.float64),
        scale=scale,
        start_idx=start_idx,
        end_idx=end_idx,
    )
