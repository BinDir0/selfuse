#!/usr/bin/env python3
import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import zarr
import cv2
import h5py

FPS = 30
IMG_SIZE = 384
INSTRUCTION_MAX_CHARS = 512

def axis_angle_to_rotmat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle representation to rotation matrix"""
    aa = np.asarray(axis_angle, dtype=np.float32)
    angle = float(np.linalg.norm(aa))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = aa / angle
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    R = np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C    ],
    ], dtype=np.float32)
    return R


def rotmat_to_rot6(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to 6D rotation representation"""
    R = np.asarray(R, dtype=np.float32)
    return np.concatenate([R[:, 0], R[:, 1]], axis=0).astype(np.float32)


def forward_fill_rows(a: np.ndarray) -> np.ndarray:
    """Forward fill zero-value rows"""
    if a is None or a.size == 0:
        return a
    out = a.copy()
    last = out[0]
    for i in range(out.shape[0]):
        if i > 0 and np.allclose(out[i], 0):
            out[i] = last
        else:
            last = out[i]
    return out


def list_task_dirs(data_root: str) -> List[Tuple[str, str]]:
    """List all task directories"""
    task_dirs = []
    
    # Traverse part1-part5 and extra directories
    for part_name in ['part1', 'part2', 'part3', 'part4', 'part5', 'extra']:
        part_dir = os.path.join(data_root, part_name)
        if not os.path.isdir(part_dir):
            continue
            
        for task_name in sorted(os.listdir(part_dir)):
            task_path = os.path.join(part_dir, task_name)
            if os.path.isdir(task_path):
                task_dirs.append((part_name, task_name))
                
    return task_dirs


def load_video_files(task_dir: str) -> List[str]:
    """Load all video files in the task directory"""
    video_files = []
    for fname in sorted(os.listdir(task_dir)):
        if fname.endswith('.mp4') and os.path.isfile(os.path.join(task_dir, fname)):
            video_files.append(fname[:-4])  # Remove .mp4 suffix
    return video_files


def _scale_intrinsics(intrinsics: np.ndarray, original_size: Tuple[int, int], target_size: Tuple[int, int]) -> np.ndarray:
    """Scale camera intrinsics to match image size changes
    
    Args:
        intrinsics: [T, 9] intrinsic matrix (fx, 0, cx, 0, fy, cy, 0, 0, 1)
        original_size: (width, height) original image size
        target_size: (width, height) target image size
    
    Returns:
        scaled_intrinsics: [T, 9] scaled intrinsic matrix
    """
    if intrinsics.size == 0:
        return intrinsics
    
    original_width, original_height = original_size
    target_width, target_height = target_size
    
    # Compute scaling ratios
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    
    scaled_intrinsics = intrinsics.copy()
    
    # Scale intrinsics: fx, fy, cx, cy
    scaled_intrinsics[:, 0] *= scale_x  # fx
    scaled_intrinsics[:, 2] *= scale_x  # cx  
    scaled_intrinsics[:, 4] *= scale_y  # fy
    scaled_intrinsics[:, 5] *= scale_y  # cy
    
    return scaled_intrinsics


def _read_intrinsics_from_hdf5(f: h5py.File, T: int, original_size: Tuple[int, int] = None) -> np.ndarray:
    def to_flat9(K: np.ndarray) -> np.ndarray:
        K = np.asarray(K, dtype=np.float32)
        if K.shape == (3, 3):
            return K.reshape(9)
        raise ValueError(f"K must be 3x3, got {K.shape}")

    def build_K_from_fx_fy_cx_cy(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        return K.reshape(9)

    # Try matrix forms
    candidate_paths = [
        ('camera', 'intrinsic'),
    ]
    for grp, key in candidate_paths:
        if grp in f and key in f[grp]:
            arr = f[grp][key][:]
            # print(f[grp][key])
            if arr.ndim == 2 and arr.shape == (3, 3):
                flat = np.tile(arr.reshape(1, 9), (T, 1))
                if original_size is not None:
                    flat = _scale_intrinsics(flat, original_size, (IMG_SIZE, IMG_SIZE))
                return flat.astype(np.float32)
            if arr.ndim == 3 and arr.shape[1:] == (3, 3):
                # If length mismatches T, align by tiling/clipping
                t_src = arr.shape[0]
                if t_src == T:
                    flat = arr.reshape(T, 9).astype(np.float32)
                    if original_size is not None:
                        flat = _scale_intrinsics(flat, original_size, (IMG_SIZE, IMG_SIZE))
                    return flat
                if t_src > 0:
                    out = np.zeros((T, 9), dtype=np.float32)
                    if t_src >= T:
                        out[:] = arr[:T].reshape(T, 9)
                    else:
                        reps = int(np.ceil(T / t_src))
                        tmp = np.tile(arr, (reps, 1, 1))[:T]
                        out[:] = tmp.reshape(T, 9)
                    if original_size is not None:
                        out = _scale_intrinsics(out, original_size, (IMG_SIZE, IMG_SIZE))
                    return out
    # Try scalar params
    def get_scalar(path: str) -> Optional[float]:
        parts = path.split('/')
        try:
            if len(parts) == 2:
                g, k = parts
                if g in f and k in f[g]:
                    v = f[g][k][()]
                    return float(np.asarray(v).reshape(()))
            elif len(parts) == 1:
                if parts[0] in f:
                    v = f[parts[0]][()]
                    return float(np.asarray(v).reshape(()))
        except Exception:
            return None
        return None
    for prefixes in [("intrinsics/", ''), ('', '')]:
        fx = get_scalar(prefixes[0] + 'fx')
        fy = get_scalar(prefixes[0] + 'fy')
        cx = get_scalar(prefixes[0] + 'cx')
        cy = get_scalar(prefixes[0] + 'cy')
        if fx is not None and fy is not None and cx is not None and cy is not None:
            flat = np.tile(build_K_from_fx_fy_cx_cy(fx, fy, cx, cy).reshape(1, 9), (T, 1))
            if original_size is not None:
                flat = _scale_intrinsics(flat, original_size, (IMG_SIZE, IMG_SIZE))
            return flat.astype(np.float32)
    # Fallback zeros
    fallback_intrinsics = np.zeros((T, 9), dtype=np.float32)
    
    # If original size information is available, apply scaling adjustment
    if original_size is not None and fallback_intrinsics is not None:
        return _scale_intrinsics(fallback_intrinsics, original_size, (IMG_SIZE, IMG_SIZE))
    return fallback_intrinsics


def load_hdf5_data(hdf5_path: str, original_size: Tuple[int, int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load hand transforms, camera extrinsics and intrinsics data from HDF5 file
    Returns: hand_transforms [T,32], camera_extr [T,16], camera_intr [T,9]
    
    Note: Camera extrinsics are converted from world2cam to cam2world format
    """
    if not os.path.isfile(hdf5_path):
        return (
            np.zeros((0, 32), dtype=np.float32),
            np.zeros((0, 16), dtype=np.float32),
            np.zeros((0, 9), dtype=np.float32),
        )
    try:
        with h5py.File(hdf5_path, 'r') as f:
            left_hand_transforms = f['transforms/leftHand'][:] if 'transforms' in f and 'leftHand' in f['transforms'] else None
            right_hand_transforms = f['transforms/rightHand'][:] if 'transforms' in f and 'rightHand' in f['transforms'] else None
            camera_transforms = f['transforms/camera'][:] if 'transforms' in f and 'camera' in f['transforms'] else None

            if left_hand_transforms is None or right_hand_transforms is None or camera_transforms is None:
                return (
                    np.zeros((0, 32), dtype=np.float32),
                    np.zeros((0, 16), dtype=np.float32),
                    np.zeros((0, 9), dtype=np.float32),
                )
            T = left_hand_transforms.shape[0]
            left_hand_flat = left_hand_transforms.reshape(T, 16)
            right_hand_flat = right_hand_transforms.reshape(T, 16)
            
            camera_flat = np.zeros_like(camera_transforms.reshape(T, 16))
            for i in range(T):
                world2cam = camera_transforms[i].reshape(4, 4)  # [4, 4] world2cam matrix
                try:
                    cam2world = np.linalg.inv(world2cam)  # Convert to cam2world
                    camera_flat[i] = cam2world.flatten()
                except np.linalg.LinAlgError:
                    raise ValueError(f"Camera extrinsics conversion failed: {world2cam}")
            
            hand_transforms = np.concatenate([left_hand_flat, right_hand_flat], axis=1).astype(np.float32)

            # Intrinsics
            camera_intr = _read_intrinsics_from_hdf5(f, T, original_size)  # [T,9]
            return hand_transforms, camera_flat.astype(np.float32), camera_intr.astype(np.float32)
    except Exception as e:
        print(f"Error loading HDF5 file {hdf5_path}: {e}")
        return (
            np.zeros((0, 32), dtype=np.float32),
            np.zeros((0, 16), dtype=np.float32),
            np.zeros((0, 9), dtype=np.float32),
        )


def load_mano_data(mano_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load hand pose data from MANO .npy file, convert global rotation to 6D representation"""
    if not os.path.isfile(mano_path):
        return np.zeros((0, 15), dtype=np.float32), np.zeros((0, 6), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    
    try:
        data = np.load(mano_path, allow_pickle=True).item()
        pose15 = np.asarray(data.get('pose_coeff', np.zeros((0, 15), dtype=np.float32)), dtype=np.float32)
        aa3 = np.asarray(data.get('global_rot', np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        t3 = np.asarray(data.get('trans', np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        
        # Convert axis-angle to 6D rotation representation
        if aa3.size > 0:
            T = aa3.shape[0]
            rot6 = np.zeros((T, 6), dtype=np.float32)
            for i in range(T):
                rotmat = axis_angle_to_rotmat(aa3[i])
                rot6[i] = rotmat_to_rot6(rotmat)
        else:
            rot6 = np.zeros((0, 6), dtype=np.float32)
            
        return pose15, rot6, t3
    except Exception as e:
        print(f"Error loading MANO file {mano_path}: {e}")
        return np.zeros((0, 15), dtype=np.float32), np.zeros((0, 6), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)


def decode_video_frames(video_path: str, start_idx: int, end_idx: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Decode video frames, return frame data and original video size"""
    if not os.path.isfile(video_path):
        target_len = max(0, end_idx - start_idx)
        return np.zeros((target_len, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8), (IMG_SIZE, IMG_SIZE)

    target_len = max(0, end_idx - start_idx)
    if target_len == 0:
        return np.zeros((0, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8), (IMG_SIZE, IMG_SIZE)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros((target_len, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8), (IMG_SIZE, IMG_SIZE)
    
    # Get original video size
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_size = (original_width, original_height)
    
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_idx))
    except Exception:
        pass
    
    frames: List[np.ndarray] = []
    for _ in range(target_len):
        ret, frame = cap.read()
        if not ret or frame is None:
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
            continue
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    
    cap.release()
    return np.stack(frames, axis=0), original_size



def extract_wrist_from_mano(left_rot6: np.ndarray, left_t3: np.ndarray, right_rot6: np.ndarray, right_t3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build wrist rotation and translation data directly from MANO data"""
    # Get maximum length
    T = max(left_rot6.shape[0] if left_rot6.size > 0 else 0,
            right_rot6.shape[0] if right_rot6.size > 0 else 0,
            left_t3.shape[0] if left_t3.size > 0 else 0,
            right_t3.shape[0] if right_t3.size > 0 else 0)
    
    if T == 0:
        return np.zeros((0, 12), dtype=np.float32), np.zeros((0, 6), dtype=np.float32)
    
    # Align data length
    def pad_to_length(arr: np.ndarray, target_len: int, dim: int) -> np.ndarray:
        if arr.size == 0:
            return np.zeros((target_len, dim), dtype=np.float32)
        arr = arr.astype(np.float32)
        if arr.shape[0] < target_len:
            pad = np.zeros((target_len - arr.shape[0], arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > target_len:
            arr = arr[:target_len]
        return arr
    
    left_rot6_aligned = pad_to_length(left_rot6, T, 6)
    left_t3_aligned = pad_to_length(left_t3, T, 3)
    right_rot6_aligned = pad_to_length(right_rot6, T, 6)
    right_t3_aligned = pad_to_length(right_t3, T, 3)
    
    # Merge left and right hand data: [T, 12] rotation, [T, 6] translation
    rot6_combined = np.concatenate([left_rot6_aligned, right_rot6_aligned], axis=1)
    t3_combined = np.concatenate([left_t3_aligned, right_t3_aligned], axis=1)
    
    return rot6_combined, t3_combined


def generate_instruction_for_task(task_name: str, T: int) -> np.ndarray:
    """Generate instruction sequence for task"""
    # Simply use task name as instruction (may need more complex mapping in practice)
    instruction = task_name.replace('_', ' ')[:INSTRUCTION_MAX_CHARS]
    instr = np.empty((T,), dtype=f'<U{INSTRUCTION_MAX_CHARS}')
    instr[:] = instruction
    return instr


def build_egodex_zarr(data_root: str, output_zarr: str):
    """Build EgoDex Zarr dataset"""
    print(f"Starting to build EgoDex Zarr dataset")
    print(f"Data root directory: {data_root}")
    print(f"Output path: {output_zarr}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_zarr), exist_ok=True)
    store = zarr.DirectoryStore(output_zarr)
    root = zarr.group(store=store, overwrite=True)

    data_grp = root.create_group('data')
    meta_grp = root.create_group('meta')

    state_grp = data_grp.create_group('state')
    action_grp = data_grp.create_group('action')

    # Create datasets - same format as HoloAssist
    state_hand_ds = state_grp.create_dataset('hand', shape=(0, 30), chunks=(65536, 30), dtype=np.float32, overwrite=True, maxshape=(None, 30))
    state_wrist_ds = state_grp.create_dataset('wrist', shape=(0, 18), chunks=(65536, 18), dtype=np.float32, overwrite=True, maxshape=(None, 18))
    action_hand_ds = action_grp.create_dataset('hand', shape=(0, 30), chunks=(65536, 30), dtype=np.float32, overwrite=True, maxshape=(None, 30))
    action_wrist_ds = action_grp.create_dataset('wrist', shape=(0, 18), chunks=(65536, 18), dtype=np.float32, overwrite=True, maxshape=(None, 18))
    instruction_ds = data_grp.create_dataset('instruction', shape=(0,), chunks=(65536,), dtype=f"U{INSTRUCTION_MAX_CHARS}", overwrite=True, maxshape=(None,))
    image_ds = data_grp.create_dataset('image', shape=(0, IMG_SIZE, IMG_SIZE, 3), chunks=(64, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8, overwrite=True, maxshape=(None, IMG_SIZE, IMG_SIZE, 3))
    extrinsic_ds = data_grp.create_dataset('extrinsic', shape=(0, 16), chunks=(65536, 16), dtype=np.float32, overwrite=True, maxshape=(None,))
    intrinsic_ds = data_grp.create_dataset('intrinsic', shape=(0, 9), chunks=(65536, 9), dtype=np.float32, overwrite=True, maxshape=(None, 9))

    episode_ends: List[int] = []
    presence_codes: List[int] = []
    total_frames = 0
    total_episodes = 0

    # Get all task directories
    task_dirs = list_task_dirs(data_root)
    print(f"Found {len(task_dirs)} task directories")

    for part_name, task_name in task_dirs:
        print(f"Processing task: {part_name}/{task_name}")
        
        task_dir = os.path.join(data_root, part_name, task_name)
        mano_left_dir = os.path.join(data_root, 'mano', 'left_hand', part_name, task_name)
        mano_right_dir = os.path.join(data_root, 'mano', 'right_hand', part_name, task_name)
        
        # Get all video files for this task
        video_files = load_video_files(task_dir)
        print(f"  Found {len(video_files)} video files")
        
        for video_id in video_files:
            # Load data
            hdf5_path = os.path.join(task_dir, f"{video_id}.hdf5")
            video_path = os.path.join(task_dir, f"{video_id}.mp4")
            mano_left_path = os.path.join(mano_left_dir, f"{video_id}.npy")
            mano_right_path = os.path.join(mano_right_dir, f"{video_id}.npy")
            
            # Load HDF5 data first to get time length
            hand_transforms, camera_extr, camera_intr_temp = load_hdf5_data(hdf5_path)
            if hand_transforms.shape[0] == 0:
                print(f"    Skipping {video_id}: No HDF5 data")
                continue
                
            T = hand_transforms.shape[0]
            
            # Decode video frames to get original size
            temp_frames, original_size = decode_video_frames(video_path, 0, min(1, T))
            
            # Reload intrinsics data with image scaling applied
            _, _, camera_intr = load_hdf5_data(hdf5_path, original_size)
            
            # Load MANO data
            L_pose15, L_rot6, L_t3 = load_mano_data(mano_left_path)
            R_pose15, R_rot6, R_t3 = load_mano_data(mano_right_path)
            
            # Align data length
            def align(a: np.ndarray, dim: int) -> np.ndarray:
                if a is None or a.size == 0:
                    return np.zeros((T, dim), dtype=np.float32)
                a = a.astype(np.float32)
                if a.ndim == 1:
                    a = a.reshape(-1, dim)
                if a.shape[0] < T:
                    pad = np.zeros((T - a.shape[0], a.shape[1]), dtype=np.float32)
                    a = np.concatenate([a, pad], axis=0)
                elif a.shape[0] > T:
                    a = a[:T]
                return a

            Lp15_full = align(L_pose15, 15)
            Rp15_full = align(R_pose15, 15)
            
            # Forward fill
            Lp15_full = forward_fill_rows(Lp15_full)
            Rp15_full = forward_fill_rows(Rp15_full)
            
            # Build wrist data directly from MANO data
            wrist_rot6, wrist_t3 = extract_wrist_from_mano(L_rot6, L_t3, R_rot6, R_t3)
            
            # Align wrist data length and forward fill
            wrist_rot6_aligned = align(wrist_rot6, 12)
            wrist_t3_aligned = align(wrist_t3, 6)
            wrist_rot6_aligned = forward_fill_rows(wrist_rot6_aligned)
            wrist_t3_aligned = forward_fill_rows(wrist_t3_aligned)
            
            # Build state and action data
            state_hand_full = np.concatenate([Lp15_full, Rp15_full], axis=1)  # [T,30]
            state_wrist_full = np.concatenate([wrist_t3_aligned, wrist_rot6_aligned], axis=1)  # [T,18]
            
            # Generate instructions
            instr_series_full = generate_instruction_for_task(task_name, T)
            
            # For EgoDex, we use the entire sequence as one episode
            if T <= 1:
                continue
                
            # Decode video frames
            images_full, _ = decode_video_frames(video_path, 0, T)  # Ignore size, already obtained
            
            # Build state-action pairs (excluding the last frame)
            valid_T = T - 1
            if valid_T <= 0:
                continue
                
            state_hand_eff = state_hand_full[:-1]  # State: t=0 to t=T-2
            state_wrist_eff = state_wrist_full[:-1]
            action_hand_eff = state_hand_full[1:]   # Action: t=1 to t=T-1
            action_wrist_eff = state_wrist_full[1:]
            images_eff = images_full[:-1]
            instr_eff = instr_series_full[:-1]
            extr_eff = camera_extr[:-1] if camera_extr.shape[0] >= T else np.tile(np.eye(4, dtype=np.float32).flatten(), (valid_T, 1))
            
            # Add to dataset
            n = state_hand_ds.shape[0]
            state_hand_ds.resize((n + valid_T, 30))
            state_wrist_ds.resize((n + valid_T, 18))
            action_hand_ds.resize((n + valid_T, 30))
            action_wrist_ds.resize((n + valid_T, 18))
            instruction_ds.resize((n + valid_T,))
            image_ds.resize((n + valid_T, IMG_SIZE, IMG_SIZE, 3))
            extrinsic_ds.resize((n + valid_T, 16))
            intrinsic_ds.resize((n + valid_T, 9))

            state_hand_ds[n:n+valid_T] = state_hand_eff
            state_wrist_ds[n:n+valid_T] = state_wrist_eff
            action_hand_ds[n:n+valid_T] = action_hand_eff
            action_wrist_ds[n:n+valid_T] = action_wrist_eff
            instruction_ds[n:n+valid_T] = instr_eff
            image_ds[n:n+valid_T] = images_eff
            extrinsic_ds[n:n+valid_T] = extr_eff.astype(np.float32)
            
            # Add intrinsics data
            intr_eff = camera_intr[:-1] if camera_intr.shape[0] >= T else np.tile(np.eye(3, dtype=np.float32).flatten(), (valid_T, 1))
            # print(intr_eff[0])
            intrinsic_ds[n:n+valid_T] = intr_eff.astype(np.float32)

            end_index = n + valid_T
            episode_ends.append(int(end_index))
            
            # Compute presence code (1=left hand present, 2=right hand present)
            presence_code = (1 if Lp15_full.size > 0 else 0) + (2 if Rp15_full.size > 0 else 0)
            presence_codes.append(int(presence_code))
            
            total_frames += int(valid_T)
            total_episodes += 1
            
            print(f"    Processed video {video_id}: {valid_T} frames, original size {original_size} → {IMG_SIZE}x{IMG_SIZE}")

    # Save metadata
    meta_grp.create_dataset('episode_ends', data=np.array(episode_ends, dtype=np.int64), dtype=np.int64, overwrite=True)
    meta_grp.create_dataset('presence', data=np.array(presence_codes, dtype=np.int8), dtype=np.int8, overwrite=True)
    root.attrs['n_episodes'] = int(total_episodes)
    root.attrs['n_frames'] = int(total_frames)

    print(f"Completed! Episodes: {total_episodes}, Total frames: {total_frames}")


def main():
    parser = argparse.ArgumentParser(description='Build EgoDex Zarr dataset')
    parser.add_argument('--data_root', type=str, default='/share_data/datasets/EgoDex', help='EgoDex root directory')
    parser.add_argument('--output', type=str, default='/share_data/datasets/EgoDex/egodex.zarr', help='Output Zarr path')
    args = parser.parse_args()

    build_egodex_zarr(args.data_root, args.output)


if __name__ == '__main__':
    main()
