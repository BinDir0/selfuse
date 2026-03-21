"""Feature extraction helpers for WebDataset export."""

import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

from .webdataset_discovery import get_episode_feature_cache_path, load_or_build_frame_index
from .webdataset_geometry import axis_angle_to_rot6d, interpolate_extrinsics, normalize_slam_keyframes

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
DEFAULT_INTRINSIC = np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float32)
LOWDIM_SIZE = 116


def run_mano_forward(mano_model, trans, root_orient, hand_pose, betas, device):
    """Run MANO forward pass, return joints (1, T, J, 3) on CPU."""
    from hawor.utils.geometry import aa_to_rotmat

    batch_size, num_frames, _ = root_orient.shape
    num_joints = 15

    params = {
        "global_orient": aa_to_rotmat(root_orient.reshape(batch_size * num_frames, 3)).view(batch_size * num_frames, 1, 3, 3),
        "hand_pose": aa_to_rotmat(hand_pose.reshape(batch_size * num_frames * num_joints, 3)).view(batch_size * num_frames, num_joints, 3, 3),
        "transl": trans.reshape(batch_size * num_frames, 3),
        "betas": betas.reshape(batch_size * num_frames, -1),
    }

    with torch.no_grad():
        output = mano_model(**{k: v.float().to(device) for k, v in params.items()}, pose2rot=False)

    return output.joints.reshape(batch_size, num_frames, -1, 3).cpu()


def build_mano_models(device, mano_dir=None):
    """Create right and left MANO models."""
    from lib.models.mano_wrapper import MANO

    if mano_dir is None:
        mano_dir = "/share_data/guantianrui/manopth/mano/models/"

    mano_right = MANO(
        data_dir=mano_dir,
        model_path=mano_dir,
        gender="neutral",
        num_hand_joints=15,
        create_body_pose=False,
    ).to(device)

    mano_left = MANO(
        data_dir=mano_dir,
        model_path=mano_dir,
        gender="neutral",
        num_hand_joints=15,
        create_body_pose=False,
        is_rhand=False,
    ).to(device)
    mano_left.shapedirs[:, 0, :] *= -1
    return mano_right, mano_left


def load_episode_features(ep, mano_right, mano_left, device, rescan_frame_index=False, feature_cache_dir=None):
    """Load one episode and compute per-frame lowdim features."""
    crop_dir = ep["crop_dir"]
    world_res_path = os.path.join(crop_dir, "world_space_res.pth")
    extracted_dir = os.path.join(crop_dir, "extracted_images")

    if feature_cache_dir and not rescan_frame_index:
        cache_path = get_episode_feature_cache_path(ep, feature_cache_dir)
        if os.path.exists(cache_path):
            try:
                cached = joblib.load(cache_path)
                if cached.get("cache_version") == 1 and cached.get("crop_dir") == crop_dir:
                    frame_index = load_or_build_frame_index(extracted_dir, rescan=False)
                    if frame_index:
                        return {
                            "frame_index": frame_index,
                            "frame_ids": cached["frame_ids"],
                            "lowdim_all": cached["lowdim_all"],
                            "presence_per_frame": cached["presence_per_frame"],
                        }
            except Exception:
                pass

    try:
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(world_res_path)
    except Exception as e:
        print(f"  Skip {ep['episode_id']}: failed to load world_space_res.pth: {e}")
        return None

    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.tensor(np.array(x), dtype=torch.float32)

    pred_trans = _to_tensor(pred_trans)
    pred_rot = _to_tensor(pred_rot)
    pred_hand_pose = _to_tensor(pred_hand_pose)
    pred_betas = _to_tensor(pred_betas)
    num_frames = int(pred_trans.shape[1])

    rot6d = axis_angle_to_rot6d(pred_rot.float())
    wrist_state = torch.cat(
        [pred_trans[0].float(), pred_trans[1].float(), rot6d[0], rot6d[1]],
        dim=-1,
    )

    right_pose = pred_hand_pose[1].float().reshape(1, num_frames, 15, 3)
    right_output = run_mano_forward(
        mano_right,
        pred_trans[1].float().unsqueeze(0),
        pred_rot[1].float().unsqueeze(0),
        right_pose,
        pred_betas[1].float().unsqueeze(0),
        device,
    )
    right_tips = right_output[:, :, FINGERTIP_INDICES, :]

    left_pose = pred_hand_pose[0].float().reshape(1, num_frames, 15, 3)
    left_output = run_mano_forward(
        mano_left,
        pred_trans[0].float().unsqueeze(0),
        pred_rot[0].float().unsqueeze(0),
        left_pose,
        pred_betas[0].float().unsqueeze(0),
        device,
    )
    left_tips = left_output[:, :, FINGERTIP_INDICES, :]

    hand_state = torch.cat([left_tips[0].reshape(num_frames, 15), right_tips[0].reshape(num_frames, 15)], dim=-1)

    wrist_action = torch.zeros_like(wrist_state)
    wrist_action[:-1] = wrist_state[1:]

    hand_action = torch.zeros_like(hand_state)
    hand_action[:-1] = hand_state[1:]

    slam_dir = os.path.join(crop_dir, "SLAM")
    extrinsics = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
    intrinsic = DEFAULT_INTRINSIC.copy()

    slam_files = sorted(Path(slam_dir).glob("hawor_slam_w_scale_*.npz")) if os.path.isdir(slam_dir) else []
    if slam_files:
        try:
            slam_data = np.load(str(slam_files[0]), allow_pickle=True)
            tstamps = slam_data["tstamp"].astype(np.int64)
            traj = slam_data["traj"]
            scale = float(slam_data["scale"])
            img_focal = float(slam_data["img_focal"])
            img_center = slam_data["img_center"]
            tstamps, traj = normalize_slam_keyframes(tstamps, traj)
            if len(tstamps) == 0:
                raise ValueError("no valid SLAM keyframes after alignment")
            extrinsics = interpolate_extrinsics(tstamps, traj, scale, num_frames)
            intrinsic = np.array(
                [img_focal, img_focal, float(img_center[0]), float(img_center[1])],
                dtype=np.float32,
            )
        except Exception as e:
            print(f"  Warning: SLAM load failed for {ep['episode_id']}: {e}")

    if isinstance(pred_valid, np.ndarray):
        valid = pred_valid.astype(np.float32)
    else:
        valid = pred_valid.float().cpu().numpy()
    if valid.ndim == 1:
        valid = np.tile(valid[:, None], (1, num_frames))
    presence_per_frame = ((valid[0] > 0.5).astype(int)) | (((valid[1] > 0.5).astype(int)) << 1)

    wrist_state_np = wrist_state.cpu().numpy().astype(np.float32)
    hand_state_np = hand_state.cpu().numpy().astype(np.float32)
    wrist_action_np = wrist_action.cpu().numpy().astype(np.float32)
    hand_action_np = hand_action.cpu().numpy().astype(np.float32)
    extrinsics_flat = extrinsics.reshape(num_frames, 16)
    intrinsic_tiled = np.tile(intrinsic, (num_frames, 1))
    lowdim_all = np.concatenate(
        [wrist_state_np, hand_state_np, wrist_action_np, hand_action_np, extrinsics_flat, intrinsic_tiled],
        axis=-1,
    )
    assert lowdim_all.shape == (num_frames, LOWDIM_SIZE), f"lowdim shape mismatch: {lowdim_all.shape}"

    frame_index = load_or_build_frame_index(extracted_dir, rescan=rescan_frame_index)
    frame_ids = sorted(frame_idx for frame_idx in frame_index if frame_idx < num_frames)
    if not frame_ids:
        return None

    episode_data = {
        "frame_index": frame_index,
        "frame_ids": frame_ids,
        "lowdim_all": lowdim_all,
        "presence_per_frame": presence_per_frame,
    }

    if feature_cache_dir:
        cache_path = get_episode_feature_cache_path(ep, feature_cache_dir)
        cache_tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        cache_payload = {
            "cache_version": 1,
            "crop_dir": crop_dir,
            "frame_ids": frame_ids,
            "lowdim_all": lowdim_all,
            "presence_per_frame": presence_per_frame.astype(np.uint8),
        }
        try:
            os.makedirs(feature_cache_dir, exist_ok=True)
            joblib.dump(cache_payload, cache_tmp_path)
            os.replace(cache_tmp_path, cache_path)
        except OSError:
            if os.path.exists(cache_tmp_path):
                os.remove(cache_tmp_path)

    return episode_data


def run_infill_for_episode(crop_dir, checkpoint, infiller_weight, device):
    """Run infiller as a small-scale fallback for a single episode."""
    seq_folder = Path(crop_dir)
    world_res = seq_folder / "world_space_res.pth"
    if world_res.exists():
        return True

    gpu = "" if str(device).startswith("cpu") else (device.split(":")[-1] if ":" in device else device)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "batch_worker.py"),
        "--stage",
        "infiller",
        "--video_path",
        str(seq_folder),
        "--gpu",
        str(gpu),
        "--checkpoint",
        checkpoint,
        "--infiller_weight",
        infiller_weight,
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(f"  Infill failed for {seq_folder.name}:\n{result.stdout}")
    return world_res.exists()
