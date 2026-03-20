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
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from multiprocessing import Pool, current_process
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
    """Convert axis-angle (*, 3) to 6D rotation representation (*, 6)."""
    from hawor.utils.geometry import aa_to_rotmat

    orig_shape = axis_angle.shape[:-1]
    flat = axis_angle.reshape(-1, 3)
    rotmat = aa_to_rotmat(flat)
    rot6d = rotmat[:, :, :2].reshape(-1, 6)
    return rot6d.reshape(*orig_shape, 6)


def quat_to_4x4(traj_row, scale):
    """Convert SLAM traj row [tx, ty, tz, qx, qy, qz, qw] to 4x4 c2w matrix."""
    t = traj_row[:3] * scale
    quat_xyzw = traj_row[3:7]
    rot = Rotation.from_quat(quat_xyzw).as_matrix()
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot
    mat[:3, 3] = t
    return mat


def interpolate_extrinsics(tstamps, traj, scale, total_frames):
    """Interpolate SLAM keyframe extrinsics to all frames."""
    tstamps = tstamps.astype(np.float64)
    num_keyframes = len(tstamps)

    if num_keyframes == 0:
        return np.tile(np.eye(4, dtype=np.float32), (total_frames, 1, 1))

    if num_keyframes == 1:
        mat = quat_to_4x4(traj[0], scale)
        return np.tile(mat, (total_frames, 1, 1))

    translations = traj[:, :3] * scale
    quats_xyzw = traj[:, 3:7]
    rotations = Rotation.from_quat(quats_xyzw)

    slerp = Slerp(tstamps, rotations)
    all_frames = np.arange(total_frames, dtype=np.float64)
    all_frames_clamped = np.clip(all_frames, tstamps[0], tstamps[-1])

    interp_rots = slerp(all_frames_clamped).as_matrix()
    interp_trans = np.stack(
        [np.interp(all_frames_clamped, tstamps, translations[:, i]) for i in range(3)],
        axis=-1,
    )

    mats = np.zeros((total_frames, 4, 4), dtype=np.float32)
    mats[:, :3, :3] = interp_rots
    mats[:, :3, 3] = interp_trans
    mats[:, 3, 3] = 1.0
    return mats


def discover_episodes(input_dir, episode_list=None, max_episodes=None, cache_file=None, require_world_res=True):
    """Discover episode directories under BuildAI processed output."""
    if cache_file is None:
        cache_file = os.path.join(input_dir, "_vla_episodes_cache.json")

    if require_world_res and os.path.exists(cache_file):
        print(f"Loading cached episode list from {cache_file}")
        with open(cache_file) as f:
            episodes = json.load(f)
        for i, ep in enumerate(episodes):
            ep["episode_index"] = i
        if max_episodes:
            episodes = episodes[:max_episodes]
        print(f"  {len(episodes)} episodes from cache")
        return episodes

    print("Scanning for episodes...")
    episodes = []

    if episode_list and os.path.exists(episode_list):
        with open(episode_list) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                crop_dir = line[:-4] if line.endswith(".mp4") else line
                if require_world_res and not os.path.exists(os.path.join(crop_dir, "world_space_res.pth")):
                    continue
                episodes.append({"crop_dir": crop_dir, "episode_id": Path(crop_dir).name})
    else:
        input_path = Path(input_dir)
        for extracted_dir in sorted(input_path.glob("*/*/processed/*/extracted_images")):
            crop_dir = str(extracted_dir.parent)
            if require_world_res and not os.path.exists(os.path.join(crop_dir, "world_space_res.pth")):
                continue
            episodes.append({"crop_dir": crop_dir, "episode_id": Path(crop_dir).name})

    for i, ep in enumerate(episodes):
        ep["episode_index"] = i

    if require_world_res:
        try:
            with open(cache_file, "w") as f:
                json.dump(episodes, f, ensure_ascii=False)
            print(f"  Cached {len(episodes)} episodes to {cache_file}")
        except OSError as e:
            print(f"  Warning: failed to write episode cache: {e}")

    if max_episodes:
        episodes = episodes[:max_episodes]
    return episodes


def load_or_build_frame_index(extracted_dir, rescan=False):
    """Load cached frame index or build it from extracted_images."""
    cache_path = os.path.join(extracted_dir, "_frame_index.json")

    if not rescan and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            return {int(k): os.path.join(extracted_dir, v) for k, v in cached.items()}
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    index = {}
    try:
        files = os.listdir(extracted_dir)
    except OSError:
        return index

    for name in files:
        if not name.endswith(".jpg"):
            continue
        stem = name[:-4]
        try:
            frame_idx = int(stem)
        except ValueError:
            continue
        index[frame_idx] = os.path.join(extracted_dir, name)

    try:
        with open(cache_path, "w") as f:
            json.dump({str(k): os.path.basename(v) for k, v in sorted(index.items())}, f)
    except OSError:
        pass

    return index


def load_episode_stats(ep, rescan_frame_index=False):
    """Augment one episode with sequence length and valid frame ids."""
    extracted_dir = os.path.join(ep["crop_dir"], "extracted_images")
    frame_index = load_or_build_frame_index(extracted_dir, rescan=rescan_frame_index)
    if not frame_index:
        return None

    world_res_path = os.path.join(ep["crop_dir"], "world_space_res.pth")
    try:
        pred_trans, *_ = joblib.load(world_res_path)
    except Exception as e:
        print(f"  Skip {ep['episode_id']}: failed to load world_space_res.pth for stats: {e}")
        return None

    seq_len = int(np.array(pred_trans).shape[1])
    frame_ids = sorted(frame_idx for frame_idx in frame_index if frame_idx < seq_len)
    if not frame_ids:
        return None

    ep_with_stats = dict(ep)
    ep_with_stats["sequence_length"] = seq_len
    ep_with_stats["frame_ids"] = frame_ids
    ep_with_stats["num_valid_frames"] = len(frame_ids)
    return ep_with_stats


def discover_episode_stats(episodes, rescan_frame_index=False):
    stats = []
    for ep in tqdm(episodes, desc="Episode stats"):
        ep_stats = load_episode_stats(ep, rescan_frame_index=rescan_frame_index)
        if ep_stats is None:
            continue
        stats.append(ep_stats)
    return stats


def repeat_episode_stats(episodes, repeat_count):
    """Repeat episodes as [1..N, 1..N, ...] with fresh episode indices."""
    if repeat_count <= 1:
        repeated = []
        for new_index, ep in enumerate(episodes):
            ep_copy = dict(ep)
            ep_copy["source_episode_index"] = ep["episode_index"]
            ep_copy["episode_index"] = new_index
            repeated.append(ep_copy)
        return repeated

    repeated = []
    for repeat_idx in range(repeat_count):
        for ep in episodes:
            ep_copy = dict(ep)
            ep_copy["source_episode_index"] = ep["episode_index"]
            ep_copy["repeat_index"] = repeat_idx
            ep_copy["episode_index"] = len(repeated)
            repeated.append(ep_copy)
    return repeated


def get_episode_feature_cache_path(ep, feature_cache_dir):
    crop_hash = hashlib.md5(ep["crop_dir"].encode("utf-8")).hexdigest()
    return os.path.join(feature_cache_dir, f"{crop_hash}.joblib")


def plan_shards(episodes, frames_per_shard, output_dir):
    """Split valid episode frames into fixed-size shard tasks."""
    tasks = []
    shard_slices = []
    shard_frame_count = 0
    shard_idx = 0

    def flush_current():
        nonlocal shard_slices, shard_frame_count, shard_idx
        if not shard_slices:
            return
        output_path = os.path.join(output_dir, f"shard-{shard_idx:06d}.tar")
        tmp_path = output_path + ".tmp"
        tasks.append(
            {
                "shard_idx": shard_idx,
                "output_path": output_path,
                "tmp_path": tmp_path,
                "frame_count": shard_frame_count,
                "episode_slices": shard_slices,
            }
        )
        shard_idx += 1
        shard_slices = []
        shard_frame_count = 0

    for ep in episodes:
        num_frames = ep["num_valid_frames"]
        start = 0
        while start < num_frames:
            remain = frames_per_shard - shard_frame_count
            take = min(remain, num_frames - start)
            shard_slices.append(
                {
                    "crop_dir": ep["crop_dir"],
                    "episode_id": ep["episode_id"],
                    "episode_index": ep["episode_index"],
                    "frame_start": start,
                    "frame_end": start + take,
                }
            )
            shard_frame_count += take
            start += take
            if shard_frame_count >= frames_per_shard:
                flush_current()

    flush_current()
    return tasks


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
    intrinsic = np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float32)

    slam_files = sorted(Path(slam_dir).glob("hawor_slam_w_scale_*.npz")) if os.path.isdir(slam_dir) else []
    if slam_files:
        try:
            slam_data = np.load(str(slam_files[0]), allow_pickle=True)
            tstamps = slam_data["tstamp"].astype(np.int64)
            traj = slam_data["traj"]
            scale = float(slam_data["scale"])
            img_focal = float(slam_data["img_focal"])
            img_center = slam_data["img_center"]
            if len(traj) != len(tstamps):
                traj = traj[tstamps]
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
    assert lowdim_all.shape == (num_frames, 116), f"lowdim shape mismatch: {lowdim_all.shape}"

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


def iter_episode_samples(ep, episode_data, frame_start, frame_end):
    """Yield frame samples for one planned episode slice."""
    frame_ids = episode_data["frame_ids"][frame_start:frame_end]
    for frame_idx in frame_ids:
        frame_path = episode_data["frame_index"].get(frame_idx)
        if frame_path is None:
            continue
        meta = {
            "dataset_name": "buildai",
            "episode_index": ep["episode_index"],
            "instruction": [],
            "instruction_num": 0,
            "presence": int(episode_data["presence_per_frame"][frame_idx]),
        }
        sample_key = f"buildai_ep{ep['episode_index']:06d}_f{frame_idx:05d}"
        yield sample_key, frame_path, episode_data["lowdim_all"][frame_idx], meta


def add_sample_to_tar(tar_writer, key, frame_path, lowdim, meta):
    """Write one WebDataset sample to a tar file."""
    with open(frame_path, "rb") as f:
        image_bytes = f.read()

    img_info = tarfile.TarInfo(name=f"{key}.image.jpg")
    img_info.size = len(image_bytes)
    tar_writer.addfile(img_info, io.BytesIO(image_bytes))

    lowdim_buf = io.BytesIO()
    np.save(lowdim_buf, lowdim)
    lowdim_bytes = lowdim_buf.getvalue()
    lowdim_info = tarfile.TarInfo(name=f"{key}.lowdim.npy")
    lowdim_info.size = len(lowdim_bytes)
    tar_writer.addfile(lowdim_info, io.BytesIO(lowdim_bytes))

    meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    meta_info = tarfile.TarInfo(name=f"{key}.meta.json")
    meta_info.size = len(meta_bytes)
    tar_writer.addfile(meta_info, io.BytesIO(meta_bytes))


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


def normalize_mano_devices(mano_device, mano_gpus):
    if mano_gpus:
        devices = []
        for gpu in mano_gpus.split(","):
            gpu = gpu.strip()
            if not gpu:
                continue
            if gpu.startswith("cuda:"):
                devices.append(gpu)
            else:
                devices.append(f"cuda:{gpu}")
        if devices:
            return devices
    return [mano_device]


def _worker_init(device_specs, mano_dir, rescan_frame_index, feature_cache_dir):
    global _worker_mano_right, _worker_mano_left, _worker_device, _worker_rescan_frame_index, _worker_feature_cache_dir, _worker_episode_cache
    identity = current_process()._identity
    worker_idx = identity[0] - 1 if identity else 0
    device_str = device_specs[worker_idx % len(device_specs)]
    _worker_device = torch.device(device_str)
    _worker_mano_right, _worker_mano_left = build_mano_models(_worker_device, mano_dir=mano_dir)
    _worker_mano_right.eval()
    _worker_mano_left.eval()
    _worker_rescan_frame_index = rescan_frame_index
    _worker_feature_cache_dir = feature_cache_dir
    _worker_episode_cache = {}


def _worker_process_shard(task):
    """Build one shard in a worker process and write directly to disk."""
    frames_written = 0
    skipped_episodes = 0
    touched_episodes = set()
    tar_writer = None
    output_path = task["output_path"]
    tmp_path = task["tmp_path"]

    try:
        for episode_slice in task["episode_slices"]:
            cache_key = episode_slice["crop_dir"]
            if cache_key not in _worker_episode_cache:
                _worker_episode_cache[cache_key] = load_episode_features(
                    episode_slice,
                    _worker_mano_right,
                    _worker_mano_left,
                    _worker_device,
                    rescan_frame_index=_worker_rescan_frame_index,
                    feature_cache_dir=_worker_feature_cache_dir,
                )

            episode_data = _worker_episode_cache[cache_key]
            if episode_data is None:
                skipped_episodes += 1
                continue

            sample_iter = iter_episode_samples(
                episode_slice,
                episode_data,
                episode_slice["frame_start"],
                episode_slice["frame_end"],
            )
            for key, frame_path, lowdim, meta in sample_iter:
                if tar_writer is None:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    tar_writer = tarfile.open(tmp_path, "w")
                add_sample_to_tar(tar_writer, key, frame_path, lowdim, meta)
                frames_written += 1

            touched_episodes.add(cache_key)
    except Exception:
        if tar_writer is not None:
            tar_writer.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if tar_writer is not None:
        tar_writer.close()

    if frames_written == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    else:
        os.replace(tmp_path, output_path)

    return {
        "shard_idx": task["shard_idx"],
        "frames_written": frames_written,
        "episodes_written": len(touched_episodes),
        "skipped_episodes": skipped_episodes,
        "output_path": output_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Build VLA WebDataset from BuildAI + HaWoR")
    parser.add_argument("--input_dir", default="/share_data/lvjianan/datasets/BuildAI-processed/")
    parser.add_argument("--output_dir", default="/share_data/guantianrui/datasets/BuildAI-VLA/")
    parser.add_argument("--episode_list", default=None, help="Text file with one episode path per line")
    parser.add_argument("--frames_per_shard", type=int, default=10000)
    parser.add_argument("--repeat_episodes", type=int, default=1, help="Repeat the full episode list this many times in order")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes for testing")
    parser.add_argument("--device", default="cuda:0", help="Deprecated alias for --mano_device")
    parser.add_argument("--mano_device", default=None, help="Device for MANO forward pass")
    parser.add_argument("--mano_gpus", default=None, help="Comma-separated GPU ids for parallel MANO workers, e.g. 0,1,2,3")
    parser.add_argument("--mano_dir", default=None, help="Directory containing MANO_RIGHT.pkl and MANO_LEFT.pkl")
    parser.add_argument("--rescan", action="store_true", help="Force rescan episodes and frame indexes")
    parser.add_argument("--num_workers", type=int, default=8, help="Deprecated alias for --writer_workers")
    parser.add_argument("--writer_workers", type=int, default=None, help="Number of parallel shard writers")
    parser.add_argument("--shard_manifest_out", default=None, help="Optional JSON manifest of planned shards")
    parser.add_argument("--auto_infill", action="store_true", help="Run infill for missing world_space_res.pth")
    parser.add_argument("--checkpoint", default=None, help="HaWoR checkpoint path (required if --auto_infill)")
    parser.add_argument("--infiller_weight", default=None, help="Infiller weight path (required if --auto_infill)")
    args = parser.parse_args()

    args.mano_device = args.mano_device or args.device
    writer_workers = args.writer_workers if args.writer_workers is not None else args.num_workers
    if args.repeat_episodes < 1:
        raise ValueError("--repeat_episodes must be >= 1")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.auto_infill:
        if not args.checkpoint or not args.infiller_weight:
            print("Error: --auto_infill requires --checkpoint and --infiller_weight")
            return
        if not os.path.exists(args.checkpoint):
            print(f"Error: checkpoint not found: {args.checkpoint}")
            return
        if not os.path.exists(args.infiller_weight):
            print(f"Error: infiller_weight not found: {args.infiller_weight}")
            return

    cache_file = os.path.join(args.input_dir, "_vla_episodes_cache.json")
    if args.rescan and os.path.exists(cache_file):
        os.remove(cache_file)

    if args.auto_infill:
        all_episodes = discover_episodes(
            args.input_dir,
            episode_list=args.episode_list,
            max_episodes=args.max_episodes,
            require_world_res=False,
        )
        missing_infill = [
            ep
            for ep in all_episodes
            if not os.path.exists(os.path.join(ep["crop_dir"], "world_space_res.pth"))
        ]
        if missing_infill:
            if len(missing_infill) > max(8, writer_workers):
                print(
                    "Warning: many episodes need infill. For large runs, prefer "
                    "`scripts/batch_infer.py --stages infiller` before building WebDataset."
                )
            print(f"Running infill for {len(missing_infill)} episodes...")
            for ep in tqdm(missing_infill, desc="Infill"):
                run_infill_for_episode(ep["crop_dir"], args.checkpoint, args.infiller_weight, args.mano_device)
        if os.path.exists(cache_file):
            os.remove(cache_file)

    episodes = discover_episodes(args.input_dir, args.episode_list, args.max_episodes, cache_file=cache_file)
    print(f"Found {len(episodes)} episodes with world_space_res.pth")
    if not episodes:
        print("No episodes found!")
        return

    print("Collecting episode stats...")
    episode_stats = discover_episode_stats(episodes, rescan_frame_index=args.rescan)
    if not episode_stats:
        print("No valid episodes with extracted frames found!")
        return

    episode_stats = repeat_episode_stats(episode_stats, args.repeat_episodes)
    feature_cache_dir = None
    if args.repeat_episodes > 1:
        print(
            f"Expanded dataset by repeating {len(episodes)} episodes x{args.repeat_episodes} "
            f"-> {len(episode_stats)} episode entries"
        )
        feature_cache_dir = os.path.join(args.output_dir, "_episode_feature_cache")
        os.makedirs(feature_cache_dir, exist_ok=True)
        print(f"Episode feature cache enabled: {feature_cache_dir}")

    shard_tasks = plan_shards(episode_stats, args.frames_per_shard, args.output_dir)
    print(f"Planned {len(shard_tasks)} shards from {sum(ep['num_valid_frames'] for ep in episode_stats)} frames")

    if args.shard_manifest_out:
        with open(args.shard_manifest_out, "w") as f:
            json.dump(shard_tasks, f, ensure_ascii=False, indent=2)
        print(f"Wrote shard manifest to {args.shard_manifest_out}")

    mano_device = torch.device(args.mano_device if torch.cuda.is_available() else "cpu")
    mano_device_specs = normalize_mano_devices(str(mano_device), args.mano_gpus if mano_device.type == "cuda" else None)
    if mano_device.type == "cuda":
        if len(mano_device_specs) > 1:
            if writer_workers > len(mano_device_specs):
                print(
                    f"Capping shard workers from {writer_workers} to {len(mano_device_specs)} "
                    f"to match MANO GPU workers: {', '.join(mano_device_specs)}"
                )
                writer_workers = len(mano_device_specs)
        elif writer_workers > 1:
            print(
                f"MANO device {mano_device} is CUDA with a single GPU worker; capping shard workers "
                f"from {writer_workers} to 1 to avoid GPU contention. Use --mano_gpus for multi-GPU writing."
            )
            writer_workers = 1

    total_frames = 0
    total_shards = 0
    total_episodes_written = 0
    total_skipped = 0

    if len(mano_device_specs) > 1:
        print(
            f"Writing shards with {writer_workers} worker(s) across MANO GPUs: "
            f"{', '.join(mano_device_specs)}"
        )
    else:
        print(f"Writing shards with {writer_workers} worker(s) on MANO device {mano_device_specs[0]}...")
    if writer_workers <= 1:
        _worker_init(mano_device_specs, args.mano_dir, args.rescan, feature_cache_dir)
        results_iter = (_worker_process_shard(task) for task in shard_tasks)
    else:
        pool = Pool(
            writer_workers,
            initializer=_worker_init,
            initargs=(mano_device_specs, args.mano_dir, args.rescan, feature_cache_dir),
        )
        results_iter = pool.imap_unordered(_worker_process_shard, shard_tasks)

    try:
        for result in tqdm(results_iter, total=len(shard_tasks), desc="Shards"):
            total_frames += result["frames_written"]
            total_shards += 1 if result["frames_written"] > 0 else 0
            total_episodes_written += result["episodes_written"]
            total_skipped += result["skipped_episodes"]
    finally:
        if writer_workers > 1:
            pool.close()
            pool.join()

    print("\nDone!")
    print(f"  Episodes touched: {total_episodes_written}")
    print(f"  Skipped episode slices: {total_skipped}")
    print(f"  Frames: {total_frames}")
    print(f"  Shards: {total_shards}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
