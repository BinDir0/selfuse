"""Common ground-truth representation + shared helpers for all dataset adapters.

Every adapter turns one dataset sequence into a ``GTSequence`` with:
  * the egocentric RGB source (video path or ordered frame paths),
  * per-frame camera extrinsics (world->cam) and intrinsics,
  * per-frame 3D hand joints in WORLD frame, 21 joints in HaWoR/OpenPose order,
  * a per-hand validity mask.

Two joint provenances:
  * raw-joint datasets (H2O, EgoVerse): we permute their joint order to OpenPose
    via ``JOINT_PERM`` (default identity; verify on production with verify_joints.py).
  * MANO-param datasets (OakInk2, TACO): we run the SAME MANO FK used on predictions
    (``mano_fk_world``) so the 21-joint order is guaranteed identical.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

# HaWoR/OpenPose hand order: 0=wrist, then thumb,index,middle,ring,pinky (base->tip, 4 each).
# Raw-joint GT that already follows this order uses identity. Override per dataset if needed.
OPENPOSE_IDENTITY = list(range(21))

LEFT, RIGHT = 0, 1


@dataclass
class GTSequence:
    seq_id: str
    dataset: str
    fps: float
    K: np.ndarray              # (3, 3)
    cam_R_w2c: np.ndarray      # (T, 3, 3)
    cam_t_w2c: np.ndarray      # (T, 3)
    joints_world: np.ndarray   # (2, T, 21, 3) metres, OpenPose order, [left, right]
    valid: np.ndarray          # (2, T) bool
    frame_paths: list[str] | None = None   # ordered ego RGB frames (alt to video)
    video_path: str | None = None          # ego RGB video (alt to frames)
    frame_archive: str | None = None        # tar holding ego frames (OakInk2)
    frame_members: list[str] | None = None  # ordered member names inside frame_archive

    # ----- derived camera geometry --------------------------------------- #
    @property
    def cam_R_c2w(self) -> np.ndarray:
        return np.transpose(self.cam_R_w2c, (0, 2, 1))

    @property
    def cam_pos_world(self) -> np.ndarray:
        # camera centre C = -R_w2c^T t_w2c
        return -np.einsum("tij,tj->ti", self.cam_R_c2w, self.cam_t_w2c)

    # ----- (de)serialisation --------------------------------------------- #
    def save_npz(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            seq_id=self.seq_id,
            dataset=self.dataset,
            fps=np.float32(self.fps),
            K=self.K.astype(np.float32),
            cam_R_w2c=self.cam_R_w2c.astype(np.float32),
            cam_t_w2c=self.cam_t_w2c.astype(np.float32),
            joints_world=self.joints_world.astype(np.float32),
            valid=self.valid.astype(bool),
        )
        meta = {
            "frame_paths": self.frame_paths,
            "video_path": self.video_path,
            "frame_archive": self.frame_archive,
            "frame_members": self.frame_members,
        }
        with open(path + ".meta.json", "w") as f:
            json.dump(meta, f)

    @staticmethod
    def load_npz(path: str) -> "GTSequence":
        d = np.load(path, allow_pickle=True)
        meta_path = path + ".meta.json"
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        return GTSequence(
            seq_id=str(d["seq_id"]),
            dataset=str(d["dataset"]),
            fps=float(d["fps"]),
            K=d["K"],
            cam_R_w2c=d["cam_R_w2c"],
            cam_t_w2c=d["cam_t_w2c"],
            joints_world=d["joints_world"],
            valid=d["valid"],
            frame_paths=meta.get("frame_paths"),
            video_path=meta.get("video_path"),
            frame_archive=meta.get("frame_archive"),
            frame_members=meta.get("frame_members"),
        )


# --------------------------------------------------------------------------- #
# Shared geometry helpers
# --------------------------------------------------------------------------- #


def cam_frame_to_world(joints_cam: np.ndarray, R_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    """Transform per-frame camera-frame joints (T, J, 3) to world using world->cam extrinsics.

    x_cam = R_w2c x_world + t_w2c  =>  x_world = R_w2c^T (x_cam - t_w2c).
    """
    R_c2w = np.transpose(R_w2c, (0, 2, 1))
    return np.einsum("tij,tnj->tni", R_c2w, joints_cam - t_w2c[:, None, :])


def permute_joints(joints: np.ndarray, perm: list[int]) -> np.ndarray:
    """Reorder the joint axis (..., J, 3) to OpenPose order."""
    return joints[..., perm, :]


def mano_fk_world(
    global_orient_aa: np.ndarray,  # (T, 3) angle-axis, world frame
    hand_pose_aa: np.ndarray,      # (T, 45) angle-axis (15 joints)
    trans: np.ndarray,             # (T, 3) world translation (metres)
    betas: np.ndarray,             # (T, 10)
    is_right: bool,
    use_cuda: bool = True,
) -> np.ndarray:
    """Run the repo's MANO FK to get (T, 21, 3) world joints in OpenPose order.

    Used by MANO-param datasets so GT joints share the exact convention of predictions.
    """
    import torch

    from hawor.utils.process import run_mano, run_mano_left

    g = torch.as_tensor(global_orient_aa, dtype=torch.float32)[None]
    p = torch.as_tensor(hand_pose_aa, dtype=torch.float32)[None]
    tr = torch.as_tensor(trans, dtype=torch.float32)[None]
    b = torch.as_tensor(betas, dtype=torch.float32)[None]
    fk = run_mano if is_right else run_mano_left
    out = fk(tr, g, p, betas=b, use_cuda=use_cuda)
    return out["joints"][0].detach().cpu().numpy().astype(np.float32)
