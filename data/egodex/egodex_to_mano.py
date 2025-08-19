#!/usr/bin/env python3
"""
python egodex/to_mano.py \
  --data_root D:/EgoDex/some_dir \
  --output_root D:/EgoDex_mano \
  --mano_root D:/manopth/mano/models \
  --hand_side right \
  --pose_file_name pose_right.npy \
  --trans_file_name trans_right.npy \
  --shape_file_name shape_right.npy \
  --recursive
"""

import os
import json
import pickle
import argparse
import re
from typing import Any, Dict, List, Tuple, Union
from enum import Enum

import numpy as np
import torch
from tqdm import tqdm
from manopth.manolayer import ManoLayer
import h5py
from scipy.spatial.transform import Rotation as R_scipy


# -----------------------------
# MANO Coordinate System Constants (following dex-retargeting style)
# -----------------------------

class HandType(Enum):
    """Hand type enumeration"""
    right = "right"
    left = "left"


# AVP to MANO coordinate system transformation matrices (from egodex-retargeting)
# MANO representation follows dex-retargeting conventions:
# middle to wrist: x ; middle to index: z ; normal: y
MANO_RIGHT2AVP_RIGHT = np.eye(4)
AVP_RIGHT2MANO_RIGHT = np.eye(4)

MANO_LEFT2AVP_LEFT = np.eye(4)
# Left hand has 180-degree rotation around Y-axis
MANO_LEFT2AVP_LEFT[:3, :3] = R_scipy.from_euler('xyz', [np.pi, 0, 0], degrees=False).as_matrix()
AVP_LEFT2MANO_LEFT = np.linalg.inv(MANO_LEFT2AVP_LEFT)

def get_coordinate_transform_matrix(hand_side: str) -> np.ndarray:
    """Get the coordinate transformation matrix for converting AVP to MANO coordinates.
    
    Args:
        hand_side: 'left' or 'right'
        
    Returns:
        Transformation matrix [3,3] for converting from AVP to MANO coordinate system
    """
    if hand_side.lower() == 'right':
        mano2avp = AVP_RIGHT2MANO_RIGHT
    else:  # left
        mano2avp = AVP_LEFT2MANO_LEFT
    # return np.eye(3, dtype=np.float32)
    return mano2avp[:3, :3]


# -----------------------------
# MANO layers (PCA mode)
# -----------------------------
manolayer_pca_right: ManoLayer = None  # type: ignore[assignment]
manolayer_pca_left: ManoLayer = None   # type: ignore[assignment]


def init_mano_layers(mano_root: Union[str, None]) -> None:
    """Initialize MANO layers in PCA mode for both hands."""
    global manolayer_pca_right, manolayer_pca_left

    if mano_root is None:
        mano_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manopth/mano/models')

    manolayer_pca_right = ManoLayer(
        mano_root=mano_root,
        use_pca=True,
        ncomps=15,
        flat_hand_mean=True,
        side='right',
        center_idx=0
    )

    manolayer_pca_left = ManoLayer(
        mano_root=mano_root,
        use_pca=True,
        ncomps=15,
        flat_hand_mean=True,
        side='left',
        center_idx=0
    )


# -----------------------------
# Generic loaders for EgoDex
# -----------------------------

def _to_numpy_array(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)


def _maybe_expand_time(arr: np.ndarray, target_last_dim: int) -> np.ndarray:
    """Ensure arr is [T, target_last_dim]. If arr is [target_last_dim], expand T=1."""
    if arr.ndim == 1 and arr.shape[0] == target_last_dim:
        return arr.reshape(1, target_last_dim)
    if arr.ndim == 2 and arr.shape[1] == target_last_dim:
        return arr
    raise ValueError(f"Array shape {arr.shape} not compatible with expected last dim {target_last_dim}")

# -----------------------------
# EgoDex HDF5 loaders
# -----------------------------

def get_camera_to_world_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """Get camera to world transform from camera extrinsics.
    
    Args:
        extrinsics: [4,4] camera extrinsics matrix (world to camera transform)
    
    Returns:
        camera_to_world: [4,4] camera to world transform matrix
    """
    if extrinsics.shape != (4, 4):
        raise ValueError(f"Extrinsics must be [4,4], got {extrinsics.shape}")
    return np.eye(4, dtype=np.float32)
    # return np.linalg.inv(extrinsics)


def _camera_to_world_transform(camera_pose: np.ndarray, 
                              camera_to_world: np.ndarray) -> np.ndarray:
    """Transform a camera pose (4x4) to world pose (4x4) using a camera-to-world transform.
    
    Args:
        camera_pose: [4,4] camera pose matrix
        camera_to_world: [4,4] camera to world transform matrix
    
    Returns:
        world_pose: [4,4] world pose matrix
    """
    return np.matmul(camera_to_world, camera_pose)


def _safe_axis_angle_from_rotmats(rotmats: np.ndarray) -> np.ndarray:
    """Convert rotation matrices [T,3,3] to axis-angle [T,3] with numerical stability."""
    assert rotmats.ndim == 3 and rotmats.shape[1:] == (3, 3)
    T = rotmats.shape[0]
    aa = np.zeros((T, 3), dtype=np.float32)

    # Clamp trace to valid range [-1, 3]
    trace = np.clip(rotmats[:, 0, 0] + rotmats[:, 1, 1] + rotmats[:, 2, 2], -1.0, 3.0)
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # Handle small angles via first-order approx: axis ~ vee(R - R^T)/2
    small = theta < 1e-6
    not_small = ~small

    # For small angles
    if np.any(small):
        R = rotmats[small]
        v = np.stack([
            R[:, 2, 1] - R[:, 1, 2],
            R[:, 0, 2] - R[:, 2, 0],
            R[:, 1, 0] - R[:, 0, 1],
        ], axis=1) * 0.5
        aa[small] = v.astype(np.float32)

    # For general case
    if np.any(not_small):
        R = rotmats[not_small]
        t = theta[not_small]
        sin_theta = np.sin(t)
        # Avoid division by zero
        sin_theta[sin_theta == 0.0] = 1e-8
        v = np.stack([
            R[:, 2, 1] - R[:, 1, 2],
            R[:, 0, 2] - R[:, 2, 0],
            R[:, 1, 0] - R[:, 0, 1],
        ], axis=1)
        axis = (v / (2.0 * sin_theta[:, None]))
        aa[not_small] = (axis * t[:, None]).astype(np.float32)

    return aa


def _invert_homogeneous(Tmats: np.ndarray) -> np.ndarray:
    """Invert homogeneous transforms [T,4,4] efficiently."""
    R = Tmats[:, :3, :3]
    t = Tmats[:, :3, 3:4]
    R_inv = np.transpose(R, (0, 2, 1))
    t_inv = -np.matmul(R_inv, t)
    out = np.zeros_like(Tmats)
    out[:, :3, :3] = R_inv
    out[:, :3, 3] = t_inv[:, :, 0]
    out[:, 3, 3] = 1.0
    return out


def _matmul_homogeneous(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Batch multiply homogeneous transforms [T,4,4]."""
    return np.matmul(A, B)


def _get_group_keys_case_insensitive(h5_group) -> List[str]:
    return [str(k) for k in h5_group.keys()]


def _find_first_key(available: List[str], candidates: List[str]) -> Union[str, None]:
    lower_set = {k.lower(): k for k in available}
    for cand in candidates:
        key = lower_set.get(cand.lower(), None)
        if key is not None:
            return key
    return None


def _collect_joint_world_tf(h5f: h5py.File, side: str) -> Tuple[Dict[str, np.ndarray], int, List[str]]:
    """Collect world transforms for all joints for a given side under transforms/.

    Returns (name->Tf[N,4,4], N, available_names)
    """
    if 'transforms' not in h5f:
        raise KeyError('HDF5 missing group: transforms')
    g = h5f['transforms']
    available = _get_group_keys_case_insensitive(g)

    # Load all arrays for this side (prefix 'left' or 'right')
    tfs: Dict[str, np.ndarray] = {}
    N = None
    prefix = side.lower()
    for name in available:
        if not name.lower().startswith(prefix):
            continue
        arr = np.array(g[name], dtype=np.float32)  # [N,4,4]
        if arr.ndim != 3 or arr.shape[1:] != (4, 4):
            continue
        tfs[name] = arr
        if N is None:
            N = arr.shape[0]
    if N is None:
        raise ValueError(f'No transforms found for side={side} under transforms/')
    return tfs, N, available


def _maybe_load_confidences(h5f: h5py.File, joint_name: str) -> Union[np.ndarray, None]:
    if 'confidences' not in h5f:
        return None
    g = h5f['confidences']
    if joint_name in g:
        return np.array(g[joint_name], dtype=np.float32).reshape(-1)
    # case-insensitive match
    low = {k.lower(): k for k in g.keys()}
    key = low.get(joint_name.lower(), None)
    if key and key in g:
        return np.array(g[key], dtype=np.float32).reshape(-1)
    return None


def load_pose_from_egodex_hdf5(h5_path: str,
                               hand_side: str,
                               conf_threshold: float = 0.0,
                               use_confidence: bool = False,
                               camera_extrinsics: Union[np.ndarray, None] = None,
                               use_mano_convention: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Load MANO axis-angle [T,48] and wrist translation [T,3] from EgoDex HDF5.

    - Extract wrist global rotation/translation from transforms/<side>Hand
    - For each finger, compute local rotations for 3 joints (thumb: metacarpal/proximal/distal; others: proximal/intermediate/distal)
    - Convert local rotation matrices to axis-angle per frame
    - Optionally zero-out joints where confidence < threshold
    - Apply MANO coordinate system transformation following egodex-retargeting conventions
    - Coordinate transform chain: AVP -> MANO (for consistency with dex-retargeting)
    
    Args:
        h5_path: Path to HDF5 file
        hand_side: 'left' or 'right'
        conf_threshold: Confidence threshold for joint filtering
        use_confidence: Whether to use confidence filtering
        camera_extrinsics: [4,4] camera extrinsics matrix (world to camera transform). If None, will try to load from HDF5 transforms/camera
        use_mano_convention: Whether to apply MANO coordinate system transformation (default: True, following egodex-retargeting)
    
    Returns:
        Tuple of (pose_aa, trans) where:
        - pose_aa: [T, 48] MANO axis-angle parameters (3 global + 45 joint)
        - trans: [T, 3] wrist translation in world coordinates (with coordinate transform applied if enabled)
    """
    side = hand_side.lower()
    if side not in ('left', 'right'):
        raise ValueError('hand_side must be left or right')
    
    hand_type = HandType.right if side == 'right' else HandType.left

    with h5py.File(h5_path, 'r') as f:
        # Load per-frame camera extrinsics from HDF5 if not provided
        per_frame_camera_extrinsics = None
        if camera_extrinsics is None:
            if 'transforms' in f and 'camera' in f['transforms']:
                per_frame_camera_extrinsics = np.array(f['transforms/camera'], dtype=np.float32)  # [N, 4, 4]
                # print(f"Loaded per-frame camera extrinsics from HDF5 transforms/camera: shape {per_frame_camera_extrinsics.shape}")
            else:
                print(f"Warning: No camera extrinsics found in HDF5 transforms/camera, assuming data is already in world frame")
        else:
            raise ValueError("camera_extrinsics is not None")

        # print(f"use_mano_convention: {use_mano_convention}")
        
        tfs, N, available = _collect_joint_world_tf(f, side)

        # Prepare coordinate transformation matrices
        coordinate_transform = None
        if use_mano_convention:
            coordinate_transform = get_coordinate_transform_matrix(side)
            # print(f"Prepared MANO coordinate transformation for {side} hand")
        coordinate_transform_4x4 = np.eye(4, dtype=np.float32)
        if coordinate_transform is not None:
            coordinate_transform_4x4[:3, :3] = coordinate_transform
        
        # Convert camera frame to world frame using per-frame extrinsics if available
        if per_frame_camera_extrinsics is not None:
            if per_frame_camera_extrinsics.shape[0] != N:
                raise ValueError(f"Camera extrinsics frames ({per_frame_camera_extrinsics.shape[0]}) != joint transform frames ({N})")
            
            for key in tfs:
                # Apply per-frame camera to world transformation
                for i in range(N):
                    camera_to_world_i = get_camera_to_world_from_extrinsics(per_frame_camera_extrinsics[i])
                    world_to_camera_i = np.linalg.inv(camera_to_world_i)
                    # tfs[key][i] = world_to_camera_i @ tfs[key][i]
                    tfs[key][i][:3, :3] = tfs[key][i][:3, :3] @ coordinate_transform
                    # tfs[key][i] = camera_to_world_i @ tfs[key][i]

        # Wrist/root key
        wrist_key = _find_first_key(available, [f'{side}Hand', f'{side}Wrist'])
        if wrist_key is None or wrist_key not in tfs:
            raise KeyError(f'Cannot find wrist transform for side={side}')
        T_wrist = tfs[wrist_key]  # [N,4,4]
        R_wrist = T_wrist[:, :3, :3]
        t_wrist = T_wrist[:, :3, 3]

        # Build mapping per finger
        # Using typical ARKit-like naming
        finger_defs = [
            # (finger_name_in_keys, is_thumb)
            ('Index', False),
            ('Middle', False),
            ('Ring', False),
            ('Little', False),
            ('Thumb', True),
        ]

        # Candidate key templates for each phalanx
        # Order for MANO per finger: joint1, joint2, joint3
        thumb_candidates = [
            [f'{side}ThumbKnuckle'],  # joint1 (~MCP/CMC)
            [f'{side}ThumbIntermediateBase'],               # joint2 (~PIP/MCP)
            [f'{side}ThumbIntermediateTip'],# joint3 (~DIP/IP)
        ]
        other_candidates = [
            # MCP
            [f'{side}{{F}}FingerKnuckle'],
            # PIP
            [f'{side}{{F}}FingerIntermediateBase'],
            # DIP
            [f'{side}{{F}}FingerIntermediateTip'],
        ]

        # Resolve concrete joint keys
        def resolve_key_list(cands: List[str]) -> str:
            # print(available)
            key = _find_first_key(available, cands)
            if key is None:
                raise KeyError(f'None of keys found: {cands}')
            return key

        # Prepare arrays for 45 joint axis-angle
        joint_aa_list: List[np.ndarray] = []

        # Precompute inverse wrist for local computation
        T_wrist_inv = _invert_homogeneous(T_wrist)

        for finger_name, is_thumb in finger_defs:
            if is_thumb:
                # thumb three joints
                keys_level = []
                for cand_group in thumb_candidates:
                    key = resolve_key_list(cand_group)
                    keys_level.append(key)
            else:
                # replace {F}
                def expand(group: List[str]) -> List[str]:
                    return [s.replace('{F}', finger_name) for s in group]
                keys_level = []
                for group in other_candidates:
                    cand_group = expand(group)
                    key = resolve_key_list(cand_group)
                    keys_level.append(key)

            # Load transforms for the three joints (world frame coordinates)
            T0 = tfs[keys_level[0]]  # [N,4,4]
            T1 = tfs[keys_level[1]]
            T2 = tfs[keys_level[2]]

            # Local rotations: relative to parent in the chain: wrist->j0->j1->j2
            # j0 local = inv(T_wrist) @ T0
            L0 = _matmul_homogeneous(T_wrist_inv, T0)
            R0 = L0[:, :3, :3]

            # j1 local = inv(T0) @ T1
            L1 = _matmul_homogeneous(_invert_homogeneous(T0), T1)
            R1 = L1[:, :3, :3]

            # j2 local = inv(T1) @ T2
            L2 = _matmul_homogeneous(_invert_homogeneous(T1), T2)
            R2 = L2[:, :3, :3]

            # Apply coordinate transformation to joint rotations if needed
            # if coordinate_transform is not None:
            #     R0 = np.array([R0[i] @ coordinate_transform for i in range(N)])
            #     R1 = np.array([R1[i] @ coordinate_transform for i in range(N)])
            #     R2 = np.array([R2[i] @ coordinate_transform for i in range(N)])

            # coordinate_transform = None
            aa0 = _safe_axis_angle_from_rotmats(R0)
            aa1 = _safe_axis_angle_from_rotmats(R1)
            aa2 = _safe_axis_angle_from_rotmats(R2)

            joint_aa_list.extend([aa0, aa1, aa2])

        # Stack to [N, 45]
        joint_aa = np.concatenate(joint_aa_list, axis=1)
        
        if side == 'right':
            global_aa = _safe_axis_angle_from_rotmats(R_wrist)
        else:
            global_aa = _safe_axis_angle_from_rotmats(R_wrist)
        trans = t_wrist.astype(np.float32)
        
        pose_aa = np.concatenate([global_aa, joint_aa], axis=1).astype(np.float32)  # [N,48]

        return pose_aa, trans


# -----------------------------
# Conversion core
# -----------------------------

def convert_mano_aa_to_pca(pose_aa: np.ndarray,
                           trans: np.ndarray,
                           beta: np.ndarray,
                           hand_side: str) -> Dict[str, Any]:
    """Compute MANO PCA coeffs from MANO axis-angle inputs.

    pose_aa: [T, 48] axis-angle (first 3: global_orient, next 45: hand joints)
    trans:   [T, 3]
    beta:    [10]
    """
    assert pose_aa.ndim == 2 and pose_aa.shape[1] == 48, f"pose_aa shape invalid: {pose_aa.shape}"
    num_frames = pose_aa.shape[0]

    global_rots = pose_aa[:, 0:3].copy()
    joint_params = pose_aa[:, 3:]  # [T, 45]

    manolayer_pca = manolayer_pca_right if hand_side == 'right' else manolayer_pca_left
    pca_components = manolayer_pca.th_selected_comps  # [15, 45]
    pca_components_tensor = torch.FloatTensor(pca_components)  # [15, 45]
    # print(pca_components_tensor @ pca_components_tensor.t())

    joint_params_tensor = torch.FloatTensor(joint_params)  # [T, 45]
    pca_coeffs = torch.matmul(joint_params_tensor, pca_components_tensor.t())  # [T, 15]

    result = {
        'pose_coeff': pca_coeffs.detach().numpy(),
        'global_rot': global_rots,
        'trans': _maybe_expand_time(trans, 3),
        'beta': beta.reshape(10),
        'num_frames': num_frames,
        'hand_side': hand_side
    }
    return result


# -----------------------------
# Dataset traversal
# -----------------------------

def find_pose_files(root_dir: str, pose_pattern: str, recursive: bool) -> List[str]:
    """Find pose files by pattern; supports wildcards like *.hdf5; optionally recursive search."""
    import fnmatch
    found: List[str] = []
    has_wildcard = any(ch in pose_pattern for ch in ['*', '?', '[', ']'])

    if not recursive:
        for name in os.listdir(root_dir):
            full = os.path.join(root_dir, name)
            if os.path.isfile(full):
                if (has_wildcard and fnmatch.fnmatch(name, pose_pattern)) or (not has_wildcard and name == pose_pattern):
                    found.append(full)
            elif os.path.isdir(full):
                # one-level deep
                for sub_name in os.listdir(full):
                    sub_full = os.path.join(full, sub_name)
                    if os.path.isfile(sub_full):
                        if (has_wildcard and fnmatch.fnmatch(sub_name, pose_pattern)) or (not has_wildcard and sub_name == pose_pattern):
                            found.append(sub_full)
        return found

    for curr_root, _, files in os.walk(root_dir):
        for f in files:
            if (has_wildcard and fnmatch.fnmatch(f, pose_pattern)) or (not has_wildcard and f == pose_pattern):
                found.append(os.path.join(curr_root, f))
    return found


def process_egodex_dataset(data_root: str,
                           output_root: str,
                           mano_root: Union[str, None],
                           hand_side: str,
                           pose_file_name: str,
                           trans_file_name: Union[str, None],
                           shape_file_name: Union[str, None],
                           recursive: bool,
                           pose_key_candidates: List[str],
                           trans_key_candidates: List[str],
                           shape_key_candidates: List[str],
                           use_confidence: bool,
                           conf_threshold: float,
                           concat_per_dir: bool,
                           concat_filename: str,
                           camera_extrinsics: Union[np.ndarray, None] = None,
                           use_mano_convention: bool = True) -> None:
    os.makedirs(output_root, exist_ok=True)

    init_mano_layers(mano_root)

    pose_files = find_pose_files(data_root, pose_file_name, recursive)
    print(f"Found {len(pose_files)} pose files to process: {pose_file_name}")
    if len(pose_files) == 0:
        print("Warning: No matching pose files found, please check --pose_file_name or --recursive parameters")
        return

    processed_count = 0
    error_count = 0

    # For directory aggregation
    dir_to_results: Dict[str, List[Dict[str, Any]]] = {}
    dir_to_order: Dict[str, List[str]] = {}

    for pose_path in tqdm(pose_files, desc=f"Processing {hand_side} hand data"):
        try:
            # trans/shape files in the same directory (optional)
            seq_dir = os.path.dirname(pose_path)
            trans_path = os.path.join(seq_dir, trans_file_name) if trans_file_name else None

            ext = os.path.splitext(pose_path)[1].lower()
            if ext in ('.hdf5', '.h5'):
                pose_aa, trans = load_pose_from_egodex_hdf5(
                    pose_path, hand_side,
                    conf_threshold=conf_threshold,
                    use_confidence=use_confidence,
                    camera_extrinsics=camera_extrinsics,
                    use_mano_convention=use_mano_convention,
                )
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
                pose_aa = load_pose_array(pose_path, pose_key_candidates)  # [T, 48]
                trans = load_trans_array(trans_path, trans_key_candidates, pose_aa.shape[0])  # [T, 3]
            beta = np.zeros((10,), dtype=np.float32)

            result = convert_mano_aa_to_pca(pose_aa, trans, beta, hand_side)

            # Generate output path (mirror directory structure)
            rel_dir = os.path.relpath(seq_dir, data_root)
            out_dir = os.path.join(output_root, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(pose_path))[0]
            out_name = f"{base_name}.npy"
            out_path = os.path.join(out_dir, out_name)

            np.save(out_path, result)
            processed_count += 1

            if concat_per_dir:
                if rel_dir not in dir_to_results:
                    dir_to_results[rel_dir] = []
                    dir_to_order[rel_dir] = []
                dir_to_results[rel_dir].append(result)
                dir_to_order[rel_dir].append(base_name)

        except Exception as e:
            error_count += 1
            print(f"Error processing {pose_path}: {e}")

    # Directory-level aggregation
    if concat_per_dir and dir_to_results:
        print("Starting directory-level aggregation and saving ...")

        def natural_key(s: str) -> List[Union[int, str]]:
            return [int(text) if text.isdigit() else text.lower() for text in re.split('(\\d+)', s)]

        for rel_dir, results in dir_to_results.items():
            names = dir_to_order[rel_dir]
            order = sorted(range(len(names)), key=lambda i: natural_key(names[i]))
            # Concatenate time dimensions in order
            pose_coeff_list = [results[i]['pose_coeff'] for i in order]
            global_rot_list = [results[i]['global_rot'] for i in order]
            trans_list = [results[i]['trans'] for i in order]
            betas = [results[i]['beta'] for i in order]
            hand_sides = [results[i]['hand_side'] for i in order]

            pose_coeff_cat = np.concatenate(pose_coeff_list, axis=0)
            global_rot_cat = np.concatenate(global_rot_list, axis=0)
            trans_cat = np.concatenate(trans_list, axis=0)
            beta_final = betas[0] if len(betas) > 0 else np.zeros((10,), dtype=np.float32)
            num_frames_total = int(sum(res['num_frames'] for res in results))
            hand_side_val = hand_sides[0] if len(set(hand_sides)) == 1 else hand_sides[0]

            merged = {
                'pose_coeff': pose_coeff_cat,
                'global_rot': global_rot_cat,
                'trans': trans_cat,
                'beta': beta_final,
                'num_frames': num_frames_total,
                'hand_side': hand_side_val,
            }

            out_dir = os.path.join(output_root, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, concat_filename)
            np.save(out_path, merged)

    print("Processing completed!")
    print(f"  - Successfully processed: {processed_count}")
    print(f"  - Failed: {error_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert EgoDex hand pose to 15-dimensional PCA components (generic loader)')
    parser.add_argument('--data_root', type=str, default='/share_data/datasets/EgoDex', help='EgoDex data root directory')
    parser.add_argument('--output_root', type=str, default='/share_data/datasets/EgoDex/mano', help='Output directory')
    parser.add_argument('--mano_root', type=str, default='/home/guantianrui/manopth/mano/models', help='MANO model path')
    parser.add_argument('--hand_side', type=str, default='both', choices=['left', 'right', 'both'], help='Hand side to process')
    parser.add_argument('--pose_file_name', type=str, default='*.hdf5', help='Pose file name or pattern (e.g. *.hdf5, supports .hdf5/.h5/.npy/.pkl/.json)')
    parser.add_argument('--trans_file_name', type=str, default=None, help='Trans file name, optional')
    parser.add_argument('--shape_file_name', type=str, default=None, help='Shape file name, optional')
    parser.add_argument('--recursive', action='store_true', help='Recursively search all subdirectories under data_root')
    parser.add_argument('--use_confidence', action='store_true', help='In HDF5 mode, zero out joint rotations with confidence below threshold')
    parser.add_argument('--conf_threshold', type=float, default=0.0, help='Joint confidence threshold, default 0.0 (no filtering)')
    parser.add_argument('--concat_per_dir', action='store_true', help='Concatenate all input files in the same directory in chronological order into one output')
    parser.add_argument('--concat_filename', type=str, default='all.npy', help='Directory aggregation output filename, default all.npy')

    # Adjustable key field names (when files are pkl/json/npy object arrays)
    parser.add_argument('--pose_keys', type=str, default='pose_aa,hand_pose,pose', help='Comma-separated candidate pose field names')
    parser.add_argument('--trans_keys', type=str, default='hand_trans,trans,translation', help='Comma-separated candidate trans field names')
    parser.add_argument('--shape_keys', type=str, default='hand_shape,beta,shape', help='Comma-separated candidate shape field names')
    
    # MANO coordinate system options (following dex-retargeting style)
    parser.add_argument('--use_mano_convention', action='store_true', default=True, help='Whether to apply MANO coordinate system transformation (default: True)')
    parser.add_argument('--no_mano_convention', action='store_true', help='Disable MANO coordinate system transformation')

    args = parser.parse_args()
    args.recursive = True
    
    # Handle MANO convention flags
    if args.no_mano_convention:
        args.use_mano_convention = False

    pose_key_candidates = [k.strip() for k in args.pose_keys.split(',') if k.strip()]
    trans_key_candidates = [k.strip() for k in args.trans_keys.split(',') if k.strip()]
    shape_key_candidates = [k.strip() for k in args.shape_keys.split(',') if k.strip()]

    print(f"Starting to process EgoDex dataset: {args.data_root}")
    print(f"Output directory: {args.output_root}")
    print(f"Hand side to process: {args.hand_side}")
    print(f"Recursive mode: {args.recursive}")
    print(f"Pose file name: {args.pose_file_name}")
    print(f"Trans file name: {args.trans_file_name}")
    print(f"Shape file name: {args.shape_file_name}")
    print(f"Pose key candidates: {pose_key_candidates}")
    print(f"Trans key candidates: {trans_key_candidates}")
    print(f"Shape key candidates: {shape_key_candidates}")
    print(f"Using MANO coordinate convention: {args.use_mano_convention}")

    if args.hand_side in ['left', 'right']:
        process_egodex_dataset(
            data_root=args.data_root,
            output_root=args.output_root,
            mano_root=args.mano_root,
            hand_side=args.hand_side,
            pose_file_name=args.pose_file_name,
            trans_file_name=args.trans_file_name,
            shape_file_name=args.shape_file_name,
            recursive=args.recursive,
            pose_key_candidates=pose_key_candidates,
            trans_key_candidates=trans_key_candidates,
            shape_key_candidates=shape_key_candidates,
            use_confidence=args.use_confidence,
            conf_threshold=args.conf_threshold,
            concat_per_dir=args.concat_per_dir,
            concat_filename=args.concat_filename,
            camera_extrinsics=None,
            use_mano_convention=args.use_mano_convention,
        )
    else:
        left_out = os.path.join(args.output_root, 'left_hand')
        right_out = os.path.join(args.output_root, 'right_hand')
        os.makedirs(left_out, exist_ok=True)
        os.makedirs(right_out, exist_ok=True)

        print("Processing left hand first ...")
        process_egodex_dataset(
            data_root=args.data_root,
            output_root=left_out,
            mano_root=args.mano_root,
            hand_side='left',
            pose_file_name=args.pose_file_name.replace('right', 'left'),
            trans_file_name=(args.trans_file_name.replace('right', 'left') if args.trans_file_name else None),
            shape_file_name=(args.shape_file_name.replace('right', 'left') if args.shape_file_name else None),
            recursive=args.recursive,
            pose_key_candidates=pose_key_candidates,
            trans_key_candidates=trans_key_candidates,
            shape_key_candidates=shape_key_candidates,
            use_confidence=args.use_confidence,
            conf_threshold=args.conf_threshold,
            concat_per_dir=args.concat_per_dir,
            concat_filename=args.concat_filename,
            camera_extrinsics=None,
            use_mano_convention=args.use_mano_convention,
        )

        print("Processing right hand next ...")
        process_egodex_dataset(
            data_root=args.data_root,
            output_root=right_out,
            mano_root=args.mano_root,
            hand_side='right',
            pose_file_name=args.pose_file_name.replace('left', 'right'),
            trans_file_name=(args.trans_file_name.replace('left', 'right') if args.trans_file_name else None),
            shape_file_name=(args.shape_file_name.replace('left', 'right') if args.shape_file_name else None),
            recursive=args.recursive,
            pose_key_candidates=pose_key_candidates,
            trans_key_candidates=trans_key_candidates,
            shape_key_candidates=shape_key_candidates,
            use_confidence=args.use_confidence,
            conf_threshold=args.conf_threshold,
            concat_per_dir=args.concat_per_dir,
            concat_filename=args.concat_filename,
            camera_extrinsics=None,
            use_mano_convention=args.use_mano_convention,
        )

    print("Dataset processing completed!")


if __name__ == '__main__':
    main()

