"""Feature extraction helpers for WebDataset export."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import torch

from .mano_codec import build_mano_pca_frame_features
from .webdataset_discovery import get_episode_feature_cache_path, load_or_build_frame_index
from .webdataset_geometry import axis_angle_to_rot6d, interpolate_extrinsics, normalize_slam_keyframes, quat_to_4x4

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
DEFAULT_INTRINSIC = np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float32)
LOWDIM_SIZE = 116
EPISODE_FEATURE_CACHE_VERSION = 4


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


def _to_float_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.tensor(np.array(value), dtype=torch.float32)


def _load_cached_episode_features(ep, extracted_dir, feature_cache_dir):
    if not feature_cache_dir:
        return None

    cache_path = get_episode_feature_cache_path(ep, feature_cache_dir)
    if not os.path.exists(cache_path):
        return None

    try:
        cached = joblib.load(cache_path)
    except Exception:
        return None

    if cached.get("cache_version") != EPISODE_FEATURE_CACHE_VERSION or cached.get("crop_dir") != ep["crop_dir"]:
        return None

    frame_index = load_or_build_frame_index(extracted_dir, rescan=False)
    if not frame_index:
        return None

    return {
        "frame_index": frame_index,
        "frame_ids": cached["frame_ids"],
        "lowdim_all": cached["lowdim_all"],
        "mano_all": cached["mano_all"],
        "presence_per_frame": cached["presence_per_frame"],
    }


def _load_world_space_prediction(ep, world_res_path):
    try:
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(world_res_path)
    except Exception as error:
        print(f"  Skip {ep['episode_id']}: failed to load world_space_res.pth: {error}")
        return None

    return {
        "pred_trans": _to_float_tensor(pred_trans),
        "pred_rot": _to_float_tensor(pred_rot),
        "pred_hand_pose": _to_float_tensor(pred_hand_pose),
        "pred_betas": _to_float_tensor(pred_betas),
        "pred_valid": pred_valid,
    }


def _compute_wrist_state(left_joints, right_joints, pred_rot):
    rot6d = axis_angle_to_rot6d(pred_rot.float())
    return torch.cat(
        [left_joints[:, 0, :].float(), right_joints[:, 0, :].float(), rot6d[0], rot6d[1]],
        dim=-1,
    )


def _compute_hand_joints(mano_model, pred_trans, pred_rot, pred_hand_pose, pred_betas, hand_index, device):
    num_frames = int(pred_trans.shape[1])
    hand_pose = pred_hand_pose[hand_index].float().reshape(1, num_frames, 15, 3)
    output = run_mano_forward(
        mano_model,
        pred_trans[hand_index].float().unsqueeze(0),
        pred_rot[hand_index].float().unsqueeze(0),
        hand_pose,
        pred_betas[hand_index].float().unsqueeze(0),
        device,
    )
    return output[0]


def _compute_hand_state(left_joints, right_joints):
    num_frames = int(left_joints.shape[0])
    left_tips = left_joints[:, FINGERTIP_INDICES, :]
    right_tips = right_joints[:, FINGERTIP_INDICES, :]
    return torch.cat(
        [left_tips.reshape(num_frames, 15), right_tips.reshape(num_frames, 15)],
        dim=-1,
    )


def _compute_joint_states(pred_trans, pred_rot, pred_hand_pose, pred_betas, mano_right, mano_left, device):
    num_frames = int(pred_trans.shape[1])
    right_joints = _compute_hand_joints(
        mano_right,
        pred_trans,
        pred_rot,
        pred_hand_pose,
        pred_betas,
        hand_index=1,
        device=device,
    )
    left_joints = _compute_hand_joints(
        mano_left,
        pred_trans,
        pred_rot,
        pred_hand_pose,
        pred_betas,
        hand_index=0,
        device=device,
    )
    wrist_state = _compute_wrist_state(left_joints, right_joints, pred_rot)
    hand_state = _compute_hand_state(left_joints, right_joints)
    return wrist_state, hand_state


def _shift_next_frame_action(state):
    action = torch.zeros_like(state)
    action[:-1] = state[1:]
    return action


def _load_episode_camera_features(ep, num_frames):
    slam_dir = os.path.join(ep["crop_dir"], "SLAM")
    extrinsics = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
    intrinsic = DEFAULT_INTRINSIC.copy()

    slam_files = sorted(Path(slam_dir).glob("hawor_slam_w_scale_*.npz")) if os.path.isdir(slam_dir) else []
    if not slam_files:
        return extrinsics, intrinsic

    try:
        slam_data = np.load(str(slam_files[0]), allow_pickle=True)
        tstamps = slam_data["tstamp"].astype(np.int64)
        traj = slam_data["traj"]
        scale = float(slam_data["scale"])
        img_focal = float(slam_data["img_focal"])
        img_center = slam_data["img_center"]
        frame_index = load_or_build_frame_index(os.path.join(ep["crop_dir"], "extracted_images"), rescan=False)
        image_center = None
        if frame_index:
            first_frame_path = frame_index.get(min(frame_index))
            if first_frame_path and os.path.exists(first_frame_path):
                import cv2

                first_image = cv2.imread(first_frame_path, cv2.IMREAD_COLOR)
                if first_image is not None:
                    h0, w0 = first_image.shape[:2]
                    image_center = np.array([float(w0) / 2.0, float(h0) / 2.0], dtype=np.float32)

        intrinsic = np.array(
            [
                img_focal,
                img_focal,
                float(image_center[0]) if image_center is not None else float(img_center[0]),
                float(image_center[1]) if image_center is not None else float(img_center[1]),
            ],
            dtype=np.float32,
        )

        traj = np.asarray(traj, dtype=np.float32)
        if traj.shape[0] == num_frames:
            c2w = np.stack([quat_to_4x4(traj_row, scale) for traj_row in traj], axis=0)
            extrinsics = np.linalg.inv(c2w).astype(np.float32)
        else:
            tstamps, traj = normalize_slam_keyframes(tstamps, traj)
            if len(tstamps) == 0:
                raise ValueError("no valid SLAM keyframes after alignment")
            extrinsics = interpolate_extrinsics(tstamps, traj, scale, num_frames)
    except Exception as error:
        print(f"  Warning: SLAM load failed for {ep['episode_id']}: {error}")

    return extrinsics, intrinsic


def _compute_presence_per_frame(pred_valid, num_frames):
    if isinstance(pred_valid, np.ndarray):
        valid = pred_valid.astype(np.float32)
    else:
        valid = pred_valid.float().cpu().numpy()
    if valid.ndim == 1:
        valid = np.tile(valid[:, None], (1, num_frames))
    return ((valid[0] > 0.5).astype(int)) | (((valid[1] > 0.5).astype(int)) << 1)


def _build_lowdim_features(wrist_state, hand_state, extrinsics, intrinsic):
    wrist_action = _shift_next_frame_action(wrist_state)
    hand_action = _shift_next_frame_action(hand_state)
    num_frames = int(wrist_state.shape[0])

    lowdim_all = np.concatenate(
        [
            wrist_state.cpu().numpy().astype(np.float32),
            hand_state.cpu().numpy().astype(np.float32),
            wrist_action.cpu().numpy().astype(np.float32),
            hand_action.cpu().numpy().astype(np.float32),
            extrinsics.reshape(num_frames, 16),
            np.tile(intrinsic, (num_frames, 1)),
        ],
        axis=-1,
    )
    assert lowdim_all.shape == (num_frames, LOWDIM_SIZE), f"lowdim shape mismatch: {lowdim_all.shape}"
    return lowdim_all


def _build_episode_data(extracted_dir, num_frames, lowdim_all, presence_per_frame, rescan_frame_index):
    frame_index = load_or_build_frame_index(extracted_dir, rescan=rescan_frame_index)
    frame_ids = sorted(frame_idx for frame_idx in frame_index if frame_idx < num_frames)
    if not frame_ids:
        return None

    return {
        "frame_index": frame_index,
        "frame_ids": frame_ids,
        "lowdim_all": lowdim_all,
        "presence_per_frame": presence_per_frame,
    }


def _build_episode_data_from_known_frame_ids(extracted_dir, frame_ids, lowdim_all, presence_per_frame):
    valid_frame_ids = [int(frame_idx) for frame_idx in frame_ids if int(frame_idx) < int(lowdim_all.shape[0])]
    if not valid_frame_ids:
        return None

    frame_index = {
        frame_idx: os.path.join(extracted_dir, f"{frame_idx}.jpg")
        for frame_idx in valid_frame_ids
    }
    return {
        "frame_index": frame_index,
        "frame_ids": valid_frame_ids,
        "lowdim_all": lowdim_all,
        "presence_per_frame": presence_per_frame,
    }


def _write_episode_feature_cache(ep, feature_cache_dir, episode_data):
    if not feature_cache_dir:
        return

    cache_path = get_episode_feature_cache_path(ep, feature_cache_dir)
    cache_tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    cache_payload = {
        "cache_version": EPISODE_FEATURE_CACHE_VERSION,
        "crop_dir": ep["crop_dir"],
        "frame_ids": episode_data["frame_ids"],
        "lowdim_all": episode_data["lowdim_all"],
        "mano_all": episode_data["mano_all"],
        "presence_per_frame": episode_data["presence_per_frame"].astype(np.uint8),
    }
    try:
        os.makedirs(feature_cache_dir, exist_ok=True)
        joblib.dump(cache_payload, cache_tmp_path)
        os.replace(cache_tmp_path, cache_path)
    except OSError:
        if os.path.exists(cache_tmp_path):
            os.remove(cache_tmp_path)


def load_episode_features(
    ep,
    mano_right,
    mano_left,
    device,
    rescan_frame_index=False,
    feature_cache_dir=None,
    require_cache=False,
    mano_dir=None,
):
    """Load one episode and compute per-frame lowdim features."""
    crop_dir = ep["crop_dir"]
    world_res_path = os.path.join(crop_dir, "world_space_res.pth")
    extracted_dir = os.path.join(crop_dir, "extracted_images")

    # After precompute, shard writers call with require_cache=True; allow disk cache load
    # even if --rescan was used (precompute already rewrote .joblib for this run).
    if feature_cache_dir and (not rescan_frame_index or require_cache):
        cached = _load_cached_episode_features(ep, extracted_dir, feature_cache_dir)
        if cached is not None:
            return cached
        if require_cache:
            raise RuntimeError(f"Missing episode feature cache for {crop_dir}")

    if require_cache:
        raise RuntimeError(f"Feature cache mode requires --feature_cache for {crop_dir}")

    prediction = _load_world_space_prediction(ep, world_res_path)
    if prediction is None:
        return None

    pred_trans = prediction["pred_trans"]
    pred_rot = prediction["pred_rot"]
    pred_hand_pose = prediction["pred_hand_pose"]
    pred_betas = prediction["pred_betas"]
    pred_valid = prediction["pred_valid"]
    num_frames = int(pred_trans.shape[1])
    mano_all = build_mano_pca_frame_features(
        pred_hand_pose.cpu().numpy(),
        pred_betas.cpu().numpy(),
        mano_dir=mano_dir,
    )

    wrist_state, hand_state = _compute_joint_states(
        pred_trans,
        pred_rot,
        pred_hand_pose,
        pred_betas,
        mano_right,
        mano_left,
        device,
    )
    extrinsics, intrinsic = _load_episode_camera_features(ep, num_frames)
    presence_per_frame = _compute_presence_per_frame(pred_valid, num_frames)
    lowdim_all = _build_lowdim_features(wrist_state, hand_state, extrinsics, intrinsic)

    if not rescan_frame_index and ep.get("frame_ids"):
        episode_data = _build_episode_data_from_known_frame_ids(
            extracted_dir,
            ep["frame_ids"],
            lowdim_all,
            presence_per_frame,
        )
    else:
        episode_data = _build_episode_data(
            extracted_dir,
            num_frames,
            lowdim_all,
            presence_per_frame,
            rescan_frame_index=rescan_frame_index,
        )
    if episode_data is None:
        return None
    episode_data["mano_all"] = mano_all

    _write_episode_feature_cache(ep, feature_cache_dir, episode_data)
    return episode_data


def run_infill_for_episode(crop_dir, checkpoint, infiller_weight, device):
    """Run infiller as a small-scale fallback for a single episode."""
    seq_folder = Path(crop_dir)
    world_res = seq_folder / "world_space_res.pth"
    if world_res.exists():
        return True

    gpu = "" if str(device).startswith("cpu") else (device.split(":")[-1] if ":" in device else device)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write(f"{seq_folder}\n")
        video_list_path = handle.name

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "batch_worker.py"),
        "--stage",
        "infiller",
        "--video_list",
        video_list_path,
        "--gpu",
        str(gpu),
        "--checkpoint",
        checkpoint,
        "--infiller_weight",
        infiller_weight,
    ]
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    finally:
        if os.path.exists(video_list_path):
            os.remove(video_list_path)
    if result.returncode != 0:
        print(f"  Infill failed for {seq_folder.name}:\n{result.stdout}")
    return world_res.exists()
