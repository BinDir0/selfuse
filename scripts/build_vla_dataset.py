#!/usr/bin/env python3
"""
Build VLA WebDataset from BuildAI 10K + HaWoR outputs.

Usage:
  python scripts/build_vla_dataset.py \
    --input_dir /share_data/lvjianan/datasets/BuildAI-processed/ \
    --output_dir /share_data/guantianrui/datasets/BuildAI-VLA/ \
    --frames_per_shard 10000

  # Quick test with few episodes:
  python scripts/build_vla_dataset.py \
    --input_dir /share_data/lvjianan/datasets/BuildAI-processed/ \
    --output_dir /tmp/test-vla/ \
    --max_episodes 5
"""
import argparse
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# MANO fingertip joint indices in OpenPose convention (after HaWoR remapping)
FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky


def axis_angle_to_rot6d(axis_angle):
    """Convert axis-angle (*, 3) to 6D rotation representation (*, 6).

    6D = first two columns of the rotation matrix, flattened.
    """
    from hawor.utils.geometry import aa_to_rotmat

    orig_shape = axis_angle.shape[:-1]
    flat = axis_angle.reshape(-1, 3)
    rotmat = aa_to_rotmat(flat)  # (N, 3, 3)
    rot6d = rotmat[:, :, :2].reshape(-1, 6)  # take first 2 cols, flatten
    return rot6d.reshape(*orig_shape, 6)


def quat_to_4x4(traj_row, scale):
    """Convert SLAM traj row [tx, ty, tz, qx, qy, qz, qw] to 4x4 c2w matrix."""
    t = traj_row[:3] * scale
    quat_xyzw = traj_row[3:7]
    # scipy uses xyzw convention
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = R
    mat[:3, 3] = t
    return mat


def interpolate_extrinsics(tstamps, traj, scale, total_frames):
    """Interpolate SLAM keyframe extrinsics to all frames.

    Returns (total_frames, 4, 4) float32 array.
    """
    tstamps = tstamps.astype(np.float64)
    N = len(tstamps)

    if N == 0:
        return np.tile(np.eye(4, dtype=np.float32), (total_frames, 1, 1))

    if N == 1:
        mat = quat_to_4x4(traj[0], scale)
        return np.tile(mat, (total_frames, 1, 1))

    # Build keyframe matrices
    translations = traj[:, :3] * scale
    quats_xyzw = traj[:, 3:7]
    rotations = Rotation.from_quat(quats_xyzw)

    # Interpolate rotation with Slerp, translation with linear interp
    slerp = Slerp(tstamps, rotations)
    all_frames = np.arange(total_frames, dtype=np.float64)
    # Clamp to SLAM range
    all_frames_clamped = np.clip(all_frames, tstamps[0], tstamps[-1])

    interp_rots = slerp(all_frames_clamped).as_matrix()  # (T, 3, 3)
    interp_trans = np.stack([
        np.interp(all_frames_clamped, tstamps, translations[:, i])
        for i in range(3)
    ], axis=-1)  # (T, 3)

    mats = np.zeros((total_frames, 4, 4), dtype=np.float32)
    mats[:, :3, :3] = interp_rots
    mats[:, :3, 3] = interp_trans
    mats[:, 3, 3] = 1.0
    return mats


def discover_episodes(input_dir, episode_list=None, max_episodes=None, cache_file=None):
    """Discover episodes with world_space_res.pth.

    Results are cached to a JSON file for fast reuse. Pass --rescan to force refresh.
    Returns list of dicts with keys: crop_dir, episode_id, episode_index.
    """
    # Try loading from cache
    if cache_file is None:
        cache_file = os.path.join(input_dir, "_vla_episodes_cache.json")

    if os.path.exists(cache_file):
        print(f"Loading cached episode list from {cache_file}")
        with open(cache_file) as f:
            episodes = json.load(f)
        for i, ep in enumerate(episodes):
            ep["episode_index"] = i
        if max_episodes:
            episodes = episodes[:max_episodes]
        print(f"  {len(episodes)} episodes from cache")
        return episodes

    # Scan
    print("Scanning for episodes (first run, will be cached)...")
    episodes = []

    if episode_list and os.path.exists(episode_list):
        with open(episode_list) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                crop_dir = line[:-4] if line.endswith(".mp4") else line
                if not os.path.exists(os.path.join(crop_dir, "world_space_res.pth")):
                    continue
                episodes.append({
                    "crop_dir": crop_dir,
                    "episode_id": Path(crop_dir).name,
                })
    else:
        input_path = Path(input_dir)
        for extracted_dir in sorted(input_path.glob("*/*/processed/*/extracted_images")):
            crop_dir = str(extracted_dir.parent)
            if not os.path.exists(os.path.join(crop_dir, "world_space_res.pth")):
                continue
            episodes.append({
                "crop_dir": crop_dir,
                "episode_id": Path(crop_dir).name,
            })

    # Assign episode indices
    for i, ep in enumerate(episodes):
        ep["episode_index"] = i

    # Save cache
    try:
        with open(cache_file, "w") as f:
            json.dump(episodes, f, ensure_ascii=False)
        print(f"  Cached {len(episodes)} episodes to {cache_file}")
    except OSError as e:
        print(f"  Warning: failed to write cache: {e}")

    if max_episodes:
        episodes = episodes[:max_episodes]

    return episodes


def find_frame_path(extracted_dir, frame_idx):
    """Try common naming formats for a frame image."""
    for fmt in (f"{frame_idx:06d}.jpg", f"{frame_idx:04d}.jpg", f"{frame_idx}.jpg"):
        p = os.path.join(extracted_dir, fmt)
        if os.path.exists(p):
            return p
    return None


def process_episode(ep, mano_right, mano_left, device):
    """Process one episode, return list of (sample_key, image_bytes, lowdim, meta).

    Returns None on failure.
    """
    crop_dir = ep["crop_dir"]
    episode_idx = ep["episode_index"]

    # --- Load world_space_res.pth ---
    world_res_path = os.path.join(crop_dir, "world_space_res.pth")
    try:
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(world_res_path)
    except Exception as e:
        print(f"  Skip {ep['episode_id']}: failed to load world_space_res.pth: {e}")
        return None

    # Convert to tensors if needed (pred_valid kept as numpy for presence calc)
    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.tensor(np.array(x), dtype=torch.float32)

    pred_trans = _to_tensor(pred_trans)
    pred_rot = _to_tensor(pred_rot)
    pred_hand_pose = _to_tensor(pred_hand_pose)
    pred_betas = _to_tensor(pred_betas)
    # pred_valid stays as-is (handled later with numpy)

    T = pred_trans.shape[1]  # num frames

    # --- Compute wrist_state (18) ---
    # pred_trans: (2, T, 3), pred_rot: (2, T, 3) axis-angle
    rot6d = axis_angle_to_rot6d(pred_rot.float())  # (2, T, 6)
    # wrist_state layout: [left_trans(3), right_trans(3), left_rot6d(6), right_rot6d(6)]
    # Match EgoDex format: [trans(6), rot6d(12)]
    wrist_state = torch.cat([
        pred_trans[0].float(),  # left trans (T, 3)
        pred_trans[1].float(),  # right trans (T, 3)
        rot6d[0],               # left rot6d (T, 6)
        rot6d[1],               # right rot6d (T, 6)
    ], dim=-1)  # (T, 18)

    # --- Compute hand_state (30) via MANO forward pass ---
    # Right hand
    right_pose = pred_hand_pose[1].float().reshape(1, T, 15, 3)
    right_output = run_mano_forward(
        mano_right,
        pred_trans[1].float().unsqueeze(0),   # (1, T, 3)
        pred_rot[1].float().unsqueeze(0),     # (1, T, 3)
        right_pose,                            # (1, T, 15, 3)
        pred_betas[1].float().unsqueeze(0),   # (1, T, 10)
        device,
    )
    right_tips = right_output[:, :, FINGERTIP_INDICES, :]  # (1, T, 5, 3)

    # Left hand
    left_pose = pred_hand_pose[0].float().reshape(1, T, 15, 3)
    left_output = run_mano_forward(
        mano_left,
        pred_trans[0].float().unsqueeze(0),
        pred_rot[0].float().unsqueeze(0),
        left_pose,
        pred_betas[0].float().unsqueeze(0),
        device,
    )
    left_tips = left_output[:, :, FINGERTIP_INDICES, :]  # (1, T, 5, 3)

    # hand_state: [left_5tips×3(15), right_5tips×3(15)]
    hand_state = torch.cat([
        left_tips[0].reshape(T, 15),
        right_tips[0].reshape(T, 15),
    ], dim=-1)  # (T, 30)

    # --- Compute actions (next frame state, not diff) ---
    # Match EgoDex format: action = next_state (not delta)
    wrist_action = torch.zeros_like(wrist_state)
    wrist_action[:-1] = wrist_state[1:]

    hand_action = torch.zeros_like(hand_state)
    hand_action[:-1] = hand_state[1:]

    # --- Load SLAM extrinsic + intrinsic ---
    slam_dir = os.path.join(crop_dir, "SLAM")
    extrinsics = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    intrinsic = np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float32)  # fallback

    slam_files = sorted(Path(slam_dir).glob("hawor_slam_w_scale_*.npz")) if os.path.isdir(slam_dir) else []
    if slam_files:
        try:
            slam_data = np.load(str(slam_files[0]), allow_pickle=True)
            tstamps = slam_data["tstamp"].astype(np.int64)
            traj = slam_data["traj"]
            scale = float(slam_data["scale"])
            img_focal = float(slam_data["img_focal"])
            img_center = slam_data["img_center"]

            # traj may have all frames; tstamp indexes the keyframes
            if len(traj) != len(tstamps):
                traj = traj[tstamps]

            extrinsics = interpolate_extrinsics(tstamps, traj, scale, T)
            intrinsic = np.array([img_focal, img_focal, float(img_center[0]), float(img_center[1])], dtype=np.float32)
        except Exception as e:
            print(f"  Warning: SLAM load failed for {ep['episode_id']}: {e}")

    extrinsics_flat = extrinsics.reshape(T, 16)

    # --- Compute presence ---
    if isinstance(pred_valid, np.ndarray):
        valid = pred_valid.astype(np.float32)
    else:
        valid = pred_valid.float().cpu().numpy()
    if valid.ndim == 1:
        valid = np.tile(valid[:, None], (1, T))
    presence_per_frame = ((valid[0] > 0.5).astype(int)) | (((valid[1] > 0.5).astype(int)) << 1)

    # --- Convert to numpy ---
    wrist_state_np = wrist_state.cpu().numpy().astype(np.float32)
    hand_state_np = hand_state.cpu().numpy().astype(np.float32)
    wrist_action_np = wrist_action.cpu().numpy().astype(np.float32)
    hand_action_np = hand_action.cpu().numpy().astype(np.float32)
    intrinsic_tiled = np.tile(intrinsic, (T, 1))  # (T, 4)

    # Assemble lowdim: (T, 116)
    lowdim_all = np.concatenate([
        wrist_state_np,     # 18
        hand_state_np,      # 30
        wrist_action_np,    # 18
        hand_action_np,     # 30
        extrinsics_flat,    # 16
        intrinsic_tiled,    # 4
    ], axis=-1)

    assert lowdim_all.shape == (T, 116), f"lowdim shape mismatch: {lowdim_all.shape}"

    # --- Collect per-frame samples ---
    extracted_dir = os.path.join(crop_dir, "extracted_images")
    samples = []

    for t in range(T):
        frame_path = find_frame_path(extracted_dir, t)
        if frame_path is None:
            continue

        with open(frame_path, "rb") as f:
            image_bytes = f.read()

        lowdim = lowdim_all[t]

        meta = {
            "dataset_name": "buildai",
            "episode_index": episode_idx,
            "instruction": [],
            "instruction_num": 0,
            "presence": int(presence_per_frame[t]),
        }

        sample_key = f"buildai_ep{episode_idx:06d}_f{t:05d}"
        samples.append((sample_key, image_bytes, lowdim, meta))

    return samples


def run_mano_forward(mano_model, trans, root_orient, hand_pose, betas, device):
    """Run MANO forward pass, return joints (1, T, J, 3) on CPU."""
    from hawor.utils.geometry import aa_to_rotmat

    B, T, _ = root_orient.shape
    NUM_JOINTS = 15

    params = {
        "global_orient": aa_to_rotmat(root_orient.reshape(B * T, 3)).view(B * T, 1, 3, 3),
        "hand_pose": aa_to_rotmat(hand_pose.reshape(B * T * NUM_JOINTS, 3)).view(B * T, NUM_JOINTS, 3, 3),
        "transl": trans.reshape(B * T, 3),
        "betas": betas.reshape(B * T, -1),
    }

    with torch.no_grad():
        output = mano_model(**{k: v.float().to(device) for k, v in params.items()}, pose2rot=False)

    joints = output.joints.reshape(B, T, -1, 3).cpu()
    return joints


def build_mano_models(device, mano_dir=None):
    """Create right and left MANO models."""
    from lib.models.mano_wrapper import MANO

    if mano_dir is None:
        mano_dir = os.path.join(str(PROJECT_ROOT), "_DATA", "data", "mano")

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
    mano_left.shapedirs[:, 0, :] *= -1  # fix left hand bug

    return mano_right, mano_left


def main():
    parser = argparse.ArgumentParser(description="Build VLA WebDataset from BuildAI + HaWoR")
    parser.add_argument("--input_dir", default="/share_data/lvjianan/datasets/BuildAI-processed/")
    parser.add_argument("--output_dir", default="/share_data/guantianrui/datasets/BuildAI-VLA/")
    parser.add_argument("--episode_list", default=None,
                        help="Text file with one episode path per line (optional)")
    parser.add_argument("--frames_per_shard", type=int, default=10000)
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes for testing")
    parser.add_argument("--device", default="cuda:0", help="Device for MANO forward pass")
    parser.add_argument("--mano_dir", default=None,
                        help="Directory containing MANO_RIGHT.pkl and MANO_LEFT.pkl "
                             "(default: PROJECT_ROOT/_DATA/data/mano)")
    parser.add_argument("--rescan", action="store_true", help="Force rescan episodes (ignore cache)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover episodes
    cache_file = os.path.join(args.input_dir, "_vla_episodes_cache.json")
    if args.rescan and os.path.exists(cache_file):
        os.remove(cache_file)
    episodes = discover_episodes(args.input_dir, args.episode_list, args.max_episodes)
    print(f"Found {len(episodes)} episodes with world_space_res.pth")

    if not episodes:
        print("No episodes found!")
        return

    # Build MANO models
    print("Loading MANO models...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mano_right, mano_left = build_mano_models(device, mano_dir=args.mano_dir)
    mano_right.eval()
    mano_left.eval()

    # Process episodes and write shards
    shard_idx = 0
    frame_count_in_shard = 0
    tar_writer = None
    total_frames = 0
    total_episodes_written = 0
    skipped = 0

    def open_new_shard():
        nonlocal shard_idx, frame_count_in_shard, tar_writer
        if tar_writer is not None:
            tar_writer.close()
        shard_path = os.path.join(args.output_dir, f"shard-{shard_idx:06d}.tar")
        tar_writer = tarfile.open(shard_path, "w")
        frame_count_in_shard = 0
        shard_idx += 1

    def add_to_tar(key, image_bytes, lowdim, meta):
        nonlocal frame_count_in_shard
        # image.jpg
        img_info = tarfile.TarInfo(name=f"{key}.image.jpg")
        img_info.size = len(image_bytes)
        tar_writer.addfile(img_info, io.BytesIO(image_bytes))

        # lowdim.npy
        lowdim_buf = io.BytesIO()
        np.save(lowdim_buf, lowdim)
        lowdim_bytes = lowdim_buf.getvalue()
        ld_info = tarfile.TarInfo(name=f"{key}.lowdim.npy")
        ld_info.size = len(lowdim_bytes)
        tar_writer.addfile(ld_info, io.BytesIO(lowdim_bytes))

        # meta.json
        meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        meta_info = tarfile.TarInfo(name=f"{key}.meta.json")
        meta_info.size = len(meta_bytes)
        tar_writer.addfile(meta_info, io.BytesIO(meta_bytes))

        frame_count_in_shard += 1

    open_new_shard()

    for ep in tqdm(episodes, desc="Episodes"):
        try:
            samples = process_episode(ep, mano_right, mano_left, device)
        except Exception as e:
            print(f"  Error processing {ep['episode_id']}: {e}")
            skipped += 1
            continue

        if samples is None:
            skipped += 1
            continue

        for key, image_bytes, lowdim, meta in samples:
            if frame_count_in_shard >= args.frames_per_shard:
                open_new_shard()
            add_to_tar(key, image_bytes, lowdim, meta)
            total_frames += 1

        total_episodes_written += 1

    if tar_writer is not None:
        tar_writer.close()

    print(f"\nDone!")
    print(f"  Episodes: {total_episodes_written} written, {skipped} skipped")
    print(f"  Frames: {total_frames}")
    print(f"  Shards: {shard_idx}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
