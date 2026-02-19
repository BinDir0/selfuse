import os
import sys
import argparse
import cv2
import rerun as rr
import numpy as np
import collections
import zarr
from omegaconf import OmegaConf

from src.utils.geometry import (
    homo_matrix_from_trans_6drot,
    homo_matrix_to_trans_6drot,
    rot_matrix_to_6drot,
    rot_matrix_from_6drot,
    transform_hand_points_to_wrist_frame,
    transform_wrist_to_target_frame,
    transform_hand_points_to_target_frame,
)
from src.dataset.legendvla_dataset import get_absolute_action, get_relative_action, transform_hand_from_wrist_to_camera
from src.utils.mano_vis import mano_forward
import torch

def get_colored_point_cloud(color_rgb, depth, width, height, K, depth_scale=0.001, min_depth=0.1, max_depth=5.0):
    X_grid, Y_grid = np.meshgrid(np.arange(width), np.arange(height))
    depth_raw = depth.reshape(-1).astype(np.float32) * depth_scale
    Z_valid = (depth_raw > min_depth) & (depth_raw < max_depth)
    Z = depth_raw[Z_valid]
    X = (X_grid.reshape(-1)[Z_valid] - K[0, 2]) * Z / K[0, 0]
    Y = (Y_grid.reshape(-1)[Z_valid] - K[1, 2]) * Z / K[1, 1]
    points = np.stack([X, Y, Z], axis=-1)
    colors = color_rgb.reshape(-1, 3)[Z_valid]
    return points, colors

class HandVisualizer:
    def __init__(self, history_len=20, color_scheme='gt'):
        """
        Args:
            history_len: Length of trajectory history
            color_scheme: 'gt' for ground truth (bright colors) or 'pred' for prediction (darker/muted colors)
        """
        # 轨迹历史
        self.history_len = history_len
        self.history = {
            'left': collections.defaultdict(lambda: collections.deque(maxlen=history_len)),
            'right': collections.defaultdict(lambda: collections.deque(maxlen=history_len))
        }
        
        # 颜色定义 - GT使用亮色，Pred使用暗色
        if color_scheme == 'gt':
            self.finger_colors = {
                'Thumb':  [255, 0, 0],    # 红
                'Index':  [0, 255, 0],    # 绿
                'Middle': [0, 0, 255],    # 蓝
                'Ring':   [255, 255, 0],  # 黄
                'Little': [255, 0, 255]   # 紫
            }
            self.wrist_color = [200, 200, 200]  # 白色
            self.structure_color = [255, 255, 255]  # 白色
            self.spoke_color = [150, 150, 150]  # 灰色
        else:  # pred
            self.finger_colors = {
                'Thumb':  [180, 0, 0],    # 暗红
                'Index':  [0, 180, 0],    # 暗绿
                'Middle': [0, 0, 180],    # 暗蓝
                'Ring':   [180, 180, 0],  # 暗黄
                'Little': [180, 0, 180]   # 暗紫
            }
            self.wrist_color = [150, 150, 150]  # 灰色
            self.structure_color = [100, 150, 255]  # 浅蓝
            self.spoke_color = [0, 100, 200]  # 深蓝
        
        self.finger_order = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']

    def _project_to_2d(self, point_cam, K):
        if point_cam[2] <= 0.01: # 过滤相机背后的点 (Z <= 0)
            return None
        
        # P_pixel = K @ P_cam
        # u = fx*x + cx*z / z
        # v = fy*y + cy*z / z
        uv_homo = K @ point_cam
        u = uv_homo[0] / uv_homo[2]
        v = uv_homo[1] / uv_homo[2]
        return [u, v]

    def log(self, root_3d_path, image_root_path, hand_name, hand_data, frame_idx, world_to_camera, K):

        if frame_idx >= len(hand_data['wrist']): return
        
        base_path_3d = f"{root_3d_path}/{hand_name}"
        # Extract action type (gt/pred) from root_3d_path for 2D overlay path
        # root_3d_path format: "/world/camera_pose/hands/gt" or "/world/camera_pose/hands/pred"
        action_type = root_3d_path.split('/')[-1] if '/' in root_3d_path else 'gt'
        base_path_2d = f"{image_root_path}/overlay/{action_type}/{hand_name}" 

        wrist_pose_cam = hand_data['wrist'][frame_idx]  # T_camera_wrist
        wrist_t_cam = wrist_pose_cam[:3, 3]
        wrist_R_cam = wrist_pose_cam[:3, :3] 

        current_tips_cam = {} 
        
        for finger in self.finger_order:
            if finger in hand_data['fingers']:
                tip_arr = hand_data['fingers'][finger]
                if frame_idx < len(tip_arr):
                    tip_matrix = tip_arr[frame_idx]  # 4x4 matrix, tip position already in camera frame
                    tip_cam = tip_matrix[:3, 3]
                    current_tips_cam[finger] = tip_cam
                    
                    self.history[hand_name][finger].append(tip_cam)


        rr.log(f"{base_path_3d}/wrist/point", rr.Points3D([wrist_t_cam], radii=0.015, colors=self.wrist_color))
        axis_len = 0.04
        rr.log(f"{base_path_3d}/wrist/axis_x", rr.Arrows3D(origins=[wrist_t_cam], vectors=[wrist_R_cam[:,0]*axis_len], colors=[255,0,0]))
        rr.log(f"{base_path_3d}/wrist/axis_y", rr.Arrows3D(origins=[wrist_t_cam], vectors=[wrist_R_cam[:,1]*axis_len], colors=[0,255,0]))
        rr.log(f"{base_path_3d}/wrist/axis_z", rr.Arrows3D(origins=[wrist_t_cam], vectors=[wrist_R_cam[:,2]*axis_len], colors=[0,0,255]))


        tips_3d_pos = [current_tips_cam[f] for f in self.finger_order if f in current_tips_cam]
        tips_color = [self.finger_colors[f] for f in self.finger_order if f in current_tips_cam]
        if tips_3d_pos:
            rr.log(f"{base_path_3d}/fingertips", rr.Points3D(tips_3d_pos, radii=0.008, colors=tips_color))


        web_lines = []
        valid_fingers = [f for f in self.finger_order if f in current_tips_cam]
        for i in range(len(valid_fingers) - 1):
            web_lines.append([current_tips_cam[valid_fingers[i]], current_tips_cam[valid_fingers[i+1]]])
        if 'Thumb' in current_tips_cam: web_lines.append([wrist_t_cam, current_tips_cam['Thumb']])
        if 'Little' in current_tips_cam: web_lines.append([wrist_t_cam, current_tips_cam['Little']])
        if web_lines:
            rr.log(f"{base_path_3d}/structure/web", rr.LineStrips3D(web_lines, radii=0.001, colors=self.structure_color))

        spokes = [[wrist_t_cam, pos] for pos in tips_3d_pos]
        if spokes:
            rr.log(f"{base_path_3d}/structure/spokes", rr.LineStrips3D(spokes, radii=0.0015, colors=self.spoke_color))

        for finger, color in self.finger_colors.items():
            if finger in self.history[hand_name] and len(self.history[hand_name][finger]) > 1:
                rr.log(f"{base_path_3d}/trails/{finger}", rr.LineStrips3D([list(self.history[hand_name][finger])], radii=0.001, colors=color))

        if K is None: return

        wrist_2d = self._project_to_2d(wrist_t_cam, K)
        if wrist_2d:
            rr.log(f"{base_path_2d}/wrist", rr.Points2D([wrist_2d], radii=20, colors=self.structure_color))

        points_2d = []
        colors_2d = []
        
        for finger in self.finger_order:
            if finger in current_tips_cam:
                tip_2d = self._project_to_2d(current_tips_cam[finger], K)
                
                if tip_2d:
                    points_2d.append(tip_2d)
                    colors_2d.append(self.finger_colors[finger])
                    
                    if wrist_2d:
                        rr.log(
                            f"{base_path_2d}/bones/{finger}",
                            rr.LineStrips2D([[wrist_2d, tip_2d]], radii=5, colors=self.spoke_color)
                        )
        
        if points_2d:
            rr.log(f"{base_path_2d}/tips", rr.Points2D(points_2d, radii=10, colors=colors_2d))

    def log_mesh(self, root_3d_path, hand_name, hand_data, frame_idx):
        if frame_idx >= len(hand_data['verts']): return
        
        base_path_3d = f"{root_3d_path}/{hand_name}"
        
        verts = hand_data['verts'][frame_idx]
        faces = hand_data['faces']
        
        if isinstance(verts, torch.Tensor):
            verts = verts.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
            
        # Log mesh to Rerun
        # Use structure_color for mesh
        color = self.structure_color
        rr.log(
            f"{base_path_3d}/mesh",
            rr.Mesh3D(
                vertex_positions=verts,
                indices=faces,
                vertex_colors=np.tile(color, (len(verts), 1))
            )
        )

def get_data_from_zarr(inference_zarr_path, origin_zarr_path=None, sample_idx=None, use_relative_action=None, motion_type='fingertips'):
    """
    Load data from inference results zarr file and corresponding original dataset.
    
    Args:
        inference_zarr_path: Path to inference results zarr file (contains pred_actions, gt_actions, etc.)
        origin_zarr_path: Path to original dataset zarr file (for loading image, depth, etc.). 
                         If None, try to infer from inference zarr metadata.
        sample_idx: Index of sample to load from inference results. If None, randomly select one.
        use_relative_action: Whether actions are relative. If None, default to True.
    
    Returns:
        dict with keys: 'image', 'depth', 'intrinsic', 'extrinsic', 'gt_actions', 'pred_actions', 'frame_idx'
        All arrays have shape (T,) where T=30 is the number of timesteps for the selected sample
        'frame_idx' is the actual frame index in original dataset (from origin_frame_indices)
    """
    if not os.path.exists(inference_zarr_path):
        raise FileNotFoundError(f"Inference results zarr file not found: {inference_zarr_path}")
    
    # Load inference results
    inference_z = zarr.open(inference_zarr_path, mode='r')
    
    # Check required keys
    required_keys = ['pred_actions', 'gt_actions', 'actions_valid_mask', 'origin_frame_indices']
    for key in required_keys:
        if key not in inference_z:
            raise ValueError(f"{key} not found in inference results. Please ensure inference has been completed.")
    
    pred_actions_all = inference_z['pred_actions']  # (N, H, D)
    gt_actions_all = inference_z['gt_actions']  # (N, H, D)
    actions_valid_mask_all = inference_z['actions_valid_mask']  # (N, H, D)
    origin_frame_indices_all = inference_z['origin_frame_indices']  # (N,)
    
    num_samples = pred_actions_all.shape[0]
    horizon = pred_actions_all.shape[1]  # Typically 30
    
    print(f"Inference results: {num_samples} samples, Horizon: {horizon}")
    
    # Randomly select a sample if not specified
    if sample_idx is None:
        sample_idx = np.random.randint(0, num_samples)
    else:
        if sample_idx >= num_samples:
            raise ValueError(f"sample_idx {sample_idx} is out of range (num_samples: {num_samples})")
    
    # Get data for selected sample
    pred_actions = pred_actions_all[sample_idx]  # (H, D)
    gt_actions = gt_actions_all[sample_idx]  # (H, D)
    actions_valid_mask = actions_valid_mask_all[sample_idx]  # (H, D)
    origin_frame_idx = int(origin_frame_indices_all[sample_idx])  # int - observation frame index in original dataset
    
    print(f"Selected sample {sample_idx}, origin frame index: {origin_frame_idx}")
    # Get origin_zarr_path from metadata if not provided
    if origin_zarr_path is None:
        if 'origin_zarr_path' in inference_z.attrs:
            origin_zarr_path = inference_z.attrs['origin_zarr_path']
        else:
            raise ValueError("origin_zarr_path not provided and not found in inference zarr metadata")
    
    if not os.path.exists(origin_zarr_path):
        raise FileNotFoundError(f"Original dataset zarr file not found: {origin_zarr_path}")
    
    # Load original dataset
    origin_z = zarr.open(origin_zarr_path, mode='r')
    data_group = origin_z['data']
    episode_ends = np.array(origin_z['meta']['episode_ends'][:])
    
    # Frame indices for this sample: origin_frame_idx is the observation frame
    # Actions are for frames [origin_frame_idx, origin_frame_idx + horizon)
    frame_indices = np.arange(origin_frame_idx, origin_frame_idx + horizon) + 1
    
    # Get dataset size
    dataset_size = data_group['image'].shape[0]
    
    # Final check: ensure frame_indices don't exceed dataset size or episode end
    if frame_indices[-1] >= dataset_size:
        # Truncate to available frames
        valid_mask = frame_indices < dataset_size
        frame_indices = frame_indices[valid_mask]
        pred_actions = pred_actions[:len(frame_indices)]
        gt_actions = gt_actions[:len(frame_indices)]
        actions_valid_mask = actions_valid_mask[:len(frame_indices)]
        horizon = len(frame_indices)
    
    # Check episode boundary
    episode_idx = np.searchsorted(episode_ends, origin_frame_idx, side='right')
    if episode_idx < len(episode_ends):
        episode_end = episode_ends[episode_idx]
        if frame_indices[-1] >= episode_end:
            # Truncate to episode end
            valid_mask = frame_indices < episode_end
            frame_indices = frame_indices[valid_mask]
            pred_actions = pred_actions[:len(frame_indices)]
            gt_actions = gt_actions[:len(frame_indices)]
            actions_valid_mask = actions_valid_mask[:len(frame_indices)]
            horizon = len(frame_indices)
    
    print(f"Selected sample {sample_idx} from inference results")
    print(f"Origin frame index: {origin_frame_idx}")
    print(f"Frame range: {frame_indices[0]} to {frame_indices[-1]} ({horizon} frames)")
    print(f"Inference zarr path: {inference_zarr_path}")
    print(f"Original dataset path: {origin_zarr_path}")
    
    # Calculate L1 loss between pred_actions and gt_actions (before coordinate transformation)
    # pred_actions and gt_actions shape: (H, D) where H=horizon, D=action_dim
    # Use actions_valid_mask from dataset if available, otherwise assume all actions are valid
    # actions_valid_mask shape: (H, D) where H=horizon, D=action_dim
    
    # Calculate L1 loss similar to inference script
    # In inference script: pred_actions shape is (batch_size, H, D)
    #   actions_valid_num = np.sum(actions_valid_mask, axis=(1,2))  # (batch_size,)
    #   batch_l1_loss = np.sum(np.abs(valid_pred - valid_gt), axis=(1,2)) / actions_valid_num.clip(min=1)
    # Here: pred_actions shape is (H, D), so we sum over all dimensions
    valid_pred = pred_actions * actions_valid_mask
    valid_gt = gt_actions * actions_valid_mask
    # Sum over all dimensions (horizon and action_dim) to get total valid actions
    actions_valid_num = np.sum(actions_valid_mask)
    
    if actions_valid_num > 0:
        # Calculate L1 loss: sum over all dimensions, then divide by total valid actions
        # This matches the inference script calculation
        total_l1_error = np.sum(np.abs(valid_pred - valid_gt))
        mean_l1_loss = total_l1_error / actions_valid_num
        
        # Also calculate per-timestep L1 loss for detailed information
        actions_valid_num_per_timestep = np.sum(actions_valid_mask, axis=1)  # (H,)
        l1_loss_per_timestep = np.sum(np.abs(valid_pred - valid_gt), axis=1) / actions_valid_num_per_timestep.clip(min=1)
        
        print(f"L1 Loss: {mean_l1_loss:.6f} (overall, matching inference script calculation)")
        print(f"  Per timestep: min={np.min(l1_loss_per_timestep):.6f}, max={np.max(l1_loss_per_timestep):.6f}, mean={np.mean(l1_loss_per_timestep):.6f}")

    else:
        print("Warning: No valid actions found for L1 loss calculation")
    
    # Default to True if not specified
    if use_relative_action is None:
        use_relative_action = True
    
    # Load image, depth, intrinsic, extrinsic from original dataset
    image = data_group['image'][frame_indices]  # (H, H_img, W_img, 3) uint8
    depth = data_group['depth'][frame_indices]  # (H, H_img, W_img) uint16
    intrinsic = data_group['intrinsic'][frame_indices]  # (H, 4) float32 [fx, fy, cx, cy]
    extrinsic_flat = data_group['extrinsic'][frame_indices]  # (H, 16) float32 (world2cam, 4x4 flattened)
    presence = data_group['presence'][frame_indices]  # (H, 2) int32 (presence of left and right hand)
    print(f"presence: {presence[0]}")
    # Reshape extrinsic from (H, 16) to (H, 4, 4)
    extrinsic = extrinsic_flat.reshape(-1, 4, 4)  # (H, 4, 4)
    
    if use_relative_action:
        # history is typically 30, the initial state is at origin_frame_idx
        state_frame_idx = origin_frame_idx
        
        # Load wrist and hand state from original dataset
        # Note: state is in WORLD coordinate system
        # Format: [left_trans3, right_trans3, left_6drot, right_6drot, left_hand, right_hand]
        wrist_state_world = data_group['state']['wrist'][state_frame_idx]  # (18,) = 2*9 (trans3 + 6drot for each hand)
        hand_state_world = data_group['state']['fingertips'][state_frame_idx]  # (30,) = 2*15 (15 keypoints for each hand)
        
        # Get camera extrinsic (world2cam) for the state frame
        # Note: extrinsic is already loaded above, but we need the one at state_frame_idx
        state_extrinsic_flat = data_group['extrinsic'][state_frame_idx]  # (16,) float32 (world2cam, 4x4 flattened)
        world2cam = state_extrinsic_flat.reshape(4, 4)  # (4, 4) world2cam
        
        # Convert fingertip keypoints from world to wrist frame using transform_hand_points_to_wrist_frame
        # hand_state_world format: [left_keypoints_15, right_keypoints_15] = (30,) = 2*15
        # wrist_state_world format: [left_trans3, right_trans3, left_6drot, right_6drot] = (18,)
        # Reshape to (1, D) for function compatibility
        hand_state_world_reshaped = hand_state_world.reshape(1, -1)  # (1, 30)
        wrist_state_world_reshaped = wrist_state_world.reshape(1, -1)  # (1, 18)
        hand_state_wrist_reshaped = transform_hand_points_to_wrist_frame(hand_state_world_reshaped, wrist_state_world_reshaped)
        hand_state_wrist = hand_state_wrist_reshaped.reshape(-1)  # (30,)
        
        # Convert wrist state from world to camera coordinate system using transform_wrist_to_target_frame
        wrist_state_world_reshaped = wrist_state_world.reshape(1, -1)  # (1, 18)
        wrist_state_cam_reshaped = transform_wrist_to_target_frame(wrist_state_world_reshaped, world2cam)
        wrist_state_cam = wrist_state_cam_reshaped.reshape(-1)  # (18,)
        
        initial_state = np.concatenate([wrist_state_cam, hand_state_wrist])  # (48,)

        wrist_action_world = data_group['action']['wrist'][frame_indices]  # (H, 18) = 2*9 (trans3 + 6drot for each hand)
        hand_action_world = data_group['action']['fingertips'][frame_indices]  # (H, 30) = 2*15 (15 keypoints for each hand)     
        
        # Transform wrist from world to camera
        wrist_action_cam = transform_wrist_to_target_frame(wrist_action_world, extrinsic)  # (H, 18) in camera coordinate

        fingertips_cam = transform_hand_points_to_target_frame(hand_action_world, extrinsic)  # (H, 30) in camera coordinate

        hand_action_wrist_reshaped = transform_hand_points_to_wrist_frame(hand_action_world, wrist_action_world)

        wrist_action_cam_reshaped = transform_wrist_to_target_frame(wrist_action_world, world2cam)

        gt_actions_cam = np.concatenate([wrist_action_cam, fingertips_cam], axis=-1) 

        initial_action = np.concatenate([wrist_action_cam_reshaped, hand_action_wrist_reshaped], axis=-1)  # (48,)

        relative_action = get_relative_action(initial_state, initial_action.copy())
        print(f"Using relative actions, initial state at frame {state_frame_idx}")
        print(f"Initial state shape: {initial_state.shape}")
        print(f"Converted state from world to camera coordinate system")
        
        gt_actions_rel2wrist = get_absolute_action(initial_state, gt_actions)
        gt_actions_cam = transform_hand_from_wrist_to_camera(gt_actions_rel2wrist, extrinsic)
        pred_actions_rel2wrist = get_absolute_action(initial_state, pred_actions)
        pred_actions_cam = transform_hand_from_wrist_to_camera(pred_actions_rel2wrist, extrinsic)
        print("Converted relative actions to absolute actions and transformed to corresponding frame's camera coordinate system")
    else:
        gt_actions_cam = transform_hand_from_wrist_to_camera(gt_actions, extrinsic)
        pred_actions_cam = transform_hand_from_wrist_to_camera(pred_actions, extrinsic)
        print("Converted absolute actions to corresponding frame's camera coordinate system")
    
    # Load MANO data if available
    mano_gt = None
    mano_pred = None
    mano_shape = None
    if motion_type == 'mano':
        # Check if inference results contain MANO data
        if 'pred_mano' in inference_z and 'gt_mano' in inference_z:
            mano_pred = inference_z['pred_mano'][sample_idx]  # (H, 90)
            mano_gt = inference_z['gt_mano'][sample_idx]  # (H, 90)
            
            if use_relative_action:
                # Load initial MANO state for absolute recovery
                initial_mano_state = data_group['state']['mano'][origin_frame_idx] # (90,)
                mano_pred = get_absolute_action(initial_mano_state, mano_pred)
                mano_gt = get_absolute_action(initial_mano_state, mano_gt)
                print("Converted relative MANO actions to absolute MANO actions")
            
            # Shape is usually in original dataset
            mano_shape = data_group['state']['shape'][origin_frame_idx:origin_frame_idx+horizon]
        else:
            # Try to load from original dataset if not in inference results
            mano_gt = data_group['action']['mano'][frame_indices]
            # For pred, we might not have it if it's not in inference_z
            print("Warning: pred_mano not found in inference results.")
            mano_shape = data_group['state']['shape'][frame_indices]

    data = {
        'image': image,  # (H, H_img, W_img, 3) uint8
        'depth': depth,  # (H, H_img, W_img) uint16
        'intrinsic': intrinsic,  # (H, 4) float32 [fx, fy, cx, cy]
        'extrinsic': extrinsic,  # (H, 4, 4) float32 (world2cam)
        'gt_actions': gt_actions_cam,  # (H, 48) float32
        'pred_actions': pred_actions_cam,  # (H, 48) float32
        'mano_gt': mano_gt,
        'mano_pred': mano_pred,
        'mano_shape': mano_shape,
        'instruction': data_group['instruction'][origin_frame_idx] if 'instruction' in data_group else None,
        'frame_idx': origin_frame_idx,  # int - observation frame index in original dataset
    }
    return data


def gt_actions_to_hands_data(gt_actions, motion_type='fingertips', mano_data=None):
    """
    Convert gt_actions to hands_data format for all timesteps.
    
    gt_actions format (T, 48) or (T, 90) depending on motion_type.
    
    Args:
        gt_actions: np.ndarray, shape (T, 48) or (T, 90)
        motion_type: 'fingertips' or 'mano'
        mano_data: dict with 'theta' and 'beta' if motion_type is 'mano'
    
    Returns:
        dict with 'left' and 'right' keys, each containing:
        - 'wrist': array of shape (T, 4, 4) transform matrices
        - 'fingers': dict with finger names and arrays of shape (T, 4, 4) transform matrices
        - 'verts': array of shape (T, V, 3) vertices (only for mano)
        - 'faces': array of shape (F, 3) faces (only for mano)
    """
    T = gt_actions.shape[0]  # Number of timesteps (30)
    
    if motion_type == 'mano' and mano_data is not None:
        wrist_actions = gt_actions # (T, 18)
        theta = mano_data['theta'] # (T, 90)
        beta = mano_data['beta'] # (T, 20)
        
        left_trans = torch.from_numpy(wrist_actions[:, :3]).float()
        right_trans = torch.from_numpy(wrist_actions[:, 3:6]).float()
        left_rot6d = torch.from_numpy(wrist_actions[:, 6:12]).float()
        right_rot6d = torch.from_numpy(wrist_actions[:, 12:18]).float()
        
        left_rot = rot_matrix_from_6drot(left_rot6d)
        right_rot = rot_matrix_from_6drot(right_rot6d)
        
        left_theta = torch.from_numpy(theta[:, :45]).float()
        right_theta = torch.from_numpy(theta[:, 45:]).float()
        
        left_beta = torch.from_numpy(beta[:, :10]).float()
        right_beta = torch.from_numpy(beta[:, 10:]).float()
        
        # MANO forward
        mano_results = mano_forward(
            rot={'left': left_rot, 'right': right_rot},
            trans={'left': left_trans, 'right': right_trans},
            theta={'left': left_theta, 'right': right_theta},
            beta={'left': left_beta, 'right': right_beta},
            sides=['left', 'right']
        )
        
        hands_data = {
            'left': {
                'wrist': mano_results['left']['joints'][:, 0].numpy(),
                'verts': mano_results['left']['verts'],
                'faces': mano_results['left']['faces'],
                'fingers': {}
            },
            'right': {
                'wrist': mano_results['right']['joints'][:, 0].numpy(),
                'verts': mano_results['right']['verts'],
                'faces': mano_results['right']['faces'],
                'fingers': {}
            }
        }
        
        # Build wrist matrices
        left_wrists = []
        right_wrists = []
        for t in range(T):
            lw = np.eye(4, dtype=np.float32)
            lw[:3, :3] = left_rot[t].numpy()
            lw[:3, 3] = left_trans[t].numpy()
            left_wrists.append(lw)
            
            rw = np.eye(4, dtype=np.float32)
            rw[:3, :3] = right_rot[t].numpy()
            rw[:3, 3] = right_trans[t].numpy()
            right_wrists.append(rw)
            
        hands_data['left']['wrist'] = np.array(left_wrists)
        hands_data['right']['wrist'] = np.array(right_wrists)
        
        return hands_data

    # Original fingertips logic
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
    
    # Initialize arrays for all timesteps
    left_wrists = []
    right_wrists = []
    left_fingers = {finger: [] for finger in finger_names}
    right_fingers = {finger: [] for finger in finger_names}
    
    # Process each timestep
    for t in range(T):
        action = gt_actions[t]  # (48,)
        # Extract data
        left_trans = action[0:3]
        right_trans = action[3:6]
        left_6d = action[6:12]
        right_6d = action[12:18]
        left_keypoints = action[18:33].reshape(5, 3)  # 5 fingers, 3 coords each
        right_keypoints = action[33:48].reshape(5, 3)
        
        # Convert 6D rotation to rotation matrix
        left_R = rot_matrix_from_6drot(left_6d)
        right_R = rot_matrix_from_6drot(right_6d)
        
        # Build 4x4 transform matrices for wrists
        left_wrist = np.eye(4, dtype=np.float32)
        left_wrist[:3, :3] = left_R
        left_wrist[:3, 3] = left_trans
        left_wrists.append(left_wrist)
        
        right_wrist = np.eye(4, dtype=np.float32)
        right_wrist[:3, :3] = right_R
        right_wrist[:3, 3] = right_trans
        right_wrists.append(right_wrist)
        
        # Build 4x4 transform matrices for fingertips
        for i, finger in enumerate(finger_names):
            # Left hand
            left_tip = np.eye(4, dtype=np.float32)
            left_tip[:3, 3] = left_keypoints[i]
            left_fingers[finger].append(left_tip)
            
            # Right hand
            right_tip = np.eye(4, dtype=np.float32)
            right_tip[:3, 3] = right_keypoints[i]
            right_fingers[finger].append(right_tip)
    
    # Convert to numpy arrays
    hands_data = {
        'left': {
            'wrist': np.array(left_wrists),  # (T, 4, 4)
            'fingers': {finger: np.array(left_fingers[finger]) for finger in finger_names}  # (T, 4, 4) each
        },
        'right': {
            'wrist': np.array(right_wrists),  # (T, 4, 4)
            'fingers': {finger: np.array(right_fingers[finger]) for finger in finger_names}  # (T, 4, 4) each
        }
    }
    
    return hands_data



def calculate_position_errors(gt_hands_data, pred_hands_data, frame_idx):
    """
    Calculate position errors for wrists and fingertips at a given frame.
    
    Returns:
        dict with structure:
        {
            'left': {
                'wrist': float (L2 distance),
                'Thumb': float,
                'Index': float,
                'Middle': float,
                'Ring': float,
                'Little': float
            },
            'right': { ... same as left ... }
        }
    """
    errors = {}
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
    
    for hand_name in ['left', 'right']:
        errors[hand_name] = {}
        
        # Wrist position error
        gt_wrist_pos = gt_hands_data[hand_name]['wrist'][frame_idx][:3, 3]
        pred_wrist_pos = pred_hands_data[hand_name]['wrist'][frame_idx][:3, 3]
        errors[hand_name]['wrist'] = float(np.linalg.norm(gt_wrist_pos - pred_wrist_pos))
        
        # Fingertip position errors
        for finger in finger_names:
            gt_finger_pos = gt_hands_data[hand_name]['fingers'][finger][frame_idx][:3, 3]
            pred_finger_pos = pred_hands_data[hand_name]['fingers'][finger][frame_idx][:3, 3]
            errors[hand_name][finger] = float(np.linalg.norm(gt_finger_pos - pred_finger_pos))
    
    return errors


INFERENCE_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "src", "config", "experiment", "inference_pretrain_legendvla.yaml",
)


def load_config_from_inference_config():
    """Load use_relative_action and motion_type from the fixed inference config.

    Reads ``inference_pretrain_legendvla.yaml``, follows
    ``inference.model_config_path`` to the training config, and extracts
    dataset / shape_meta parameters.
    """
    if not os.path.exists(INFERENCE_CONFIG_PATH):
        print(f"Warning: inference config not found: {INFERENCE_CONFIG_PATH}")
        return {}

    inference_cfg = OmegaConf.load(INFERENCE_CONFIG_PATH)
    result = {}

    model_config_path = OmegaConf.select(inference_cfg, "inference.model_config_path")
    if model_config_path and os.path.exists(model_config_path):
        model_cfg = OmegaConf.load(model_config_path)
        result["use_relative_action"] = OmegaConf.select(
            model_cfg, "dataset.vla_dataset.use_relative_action", default=None
        )
        result["motion_type"] = OmegaConf.select(
            model_cfg, "shape_meta.obs.state.type", default=None
        )
    else:
        if model_config_path:
            print(f"Warning: model_config_path not found: {model_config_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_zarr_path", type=str, required=True, help="Path to inference results zarr file (contains pred_actions, gt_actions, etc.).")
    parser.add_argument("--origin_zarr_path", type=str, default=None, help="Path to original dataset zarr file. If None, read from zarr attrs.")
    parser.add_argument("--sample_idx", type=int, default=None, help="Index of sample to load from inference results. If None, randomly select one.")
    parser.add_argument("--target_width", type=int, default=1920, help="Target image width. If None, use original width.")
    parser.add_argument("--target_height", type=int, default=1080, help="Target image height. If None, use original height.")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=1.5)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save rrd file. If None, use spawn/notebook_show.")
    args = parser.parse_args()

    if not os.path.exists(args.inference_zarr_path):
        raise FileNotFoundError(f"Inference zarr file not found: {args.inference_zarr_path}")

    config_defaults = load_config_from_inference_config()
    use_relative_action = config_defaults.get("use_relative_action")
    motion_type = config_defaults.get("motion_type")
    print(f"Loaded from config: use_relative_action={use_relative_action}, motion_type={motion_type}")

    print(f"Loading data from inference zarr: {args.inference_zarr_path}")
    zarr_data = get_data_from_zarr(
        args.inference_zarr_path, 
        origin_zarr_path=args.origin_zarr_path,
        sample_idx=args.sample_idx, 
        use_relative_action=args.use_relative_action,
        motion_type=args.motion_type
    )
    
    # Extract images and depth: (T, H, W, 3) and (T, H, W)
    rgb_images = np.array(zarr_data['image'])  # (T, H, W, 3) uint8
    depth_images = np.array(zarr_data['depth']).astype(np.float32) / 1000.0  # (T, H, W) uint16 -> float32 meters
    
    # Extract intrinsics: (T, 4) [fx, fy, cx, cy]
    intrinsics_flat = zarr_data['intrinsic']  # (T, 4)
    
    # Extract extrinsics: (T, 4, 4) world2cam
    camera_transforms_raw = zarr_data['extrinsic']  # (T, 4, 4) world2cam
    
    # Convert gt_actions and pred_actions to hands_data format: (T, 48) -> hands_data with T frames
    gt_actions = zarr_data['gt_actions']  # (T, 48)
    pred_actions = zarr_data['pred_actions']  # (T, 48)
    
    if motion_type == 'mano' and zarr_data['mano_gt'] is not None:
        gt_hands_data = gt_actions_to_hands_data(
            gt_actions[:, :18], 
            motion_type='mano', 
            mano_data={'theta': zarr_data['mano_gt'], 'beta': zarr_data['mano_shape']}
        )
        if zarr_data['mano_pred'] is not None:
            pred_hands_data = gt_actions_to_hands_data(
                pred_actions[:, :18], 
                motion_type='mano', 
                mano_data={'theta': zarr_data['mano_pred'], 'beta': zarr_data['mano_shape']}
            )
        else:
            pred_hands_data = gt_actions_to_hands_data(pred_actions)
    else:
        gt_hands_data = gt_actions_to_hands_data(gt_actions)
        pred_hands_data = gt_actions_to_hands_data(pred_actions)
    
    T = len(rgb_images)  # Number of timesteps (30)
    print(f"Loaded {T} frames (timesteps) from selected sample")
    print(f"GT hands data: left={len(gt_hands_data['left']['wrist'])}, right={len(gt_hands_data['right']['wrist'])}")
    print(f"Pred hands data: left={len(pred_hands_data['left']['wrist'])}, right={len(pred_hands_data['right']['wrist'])}")
    
    # Resize images, depth, and adjust intrinsics if target dimensions are specified
    orig_h, orig_w = rgb_images[0].shape[:2]
    target_w = args.target_width if args.target_width is not None else orig_w
    target_h = args.target_height if args.target_height is not None else orig_h
    
    if args.target_width is not None or args.target_height is not None:
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        print(f"Resizing from ({orig_w}, {orig_h}) to ({target_w}, {target_h})")
        print(f"Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")
        
        # Resize RGB images for all timesteps
        rgb_images_resized = []
        for img in rgb_images:
            rgb_images_resized.append(cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR))
        rgb_images = np.array(rgb_images_resized)
        
        # Resize depth images for all timesteps
        depth_images_resized = []
        for d in depth_images:
            depth_images_resized.append(cv2.resize(d, (target_w, target_h), interpolation=cv2.INTER_NEAREST))
        depth_images = np.array(depth_images_resized)
        
        # Adjust intrinsics: scale fx, fy, cx, cy for all timesteps
        intrinsics_flat = intrinsics_flat.copy()
        intrinsics_flat[:, 0] *= scale_x  # fx
        intrinsics_flat[:, 1] *= scale_y  # fy
        intrinsics_flat[:, 2] *= scale_x  # cx
        intrinsics_flat[:, 3] *= scale_y  # cy
    
    # Get actual frame_idx that was used
    actual_frame_idx = zarr_data['frame_idx']
    
    # Initialize Rerun with unique app name based on frame_idx
    app_name = f"rgbd_hand_action_vis_frame_{actual_frame_idx}"
    rr.init(app_name)
    
    # Save to rrd file if save_path is provided, otherwise use spawn
    if args.save_path:
        # If save_path is a directory, create unique filename with frame_idx
        if os.path.isdir(args.save_path):
            save_filename = f"visualization_frame_{actual_frame_idx}.rrd"
            save_path = os.path.join(args.save_path, save_filename)
        else:
            # If it's a file, add frame_idx to filename
            base_path = os.path.splitext(args.save_path)[0]
            ext = os.path.splitext(args.save_path)[1] or '.rrd'
            save_path = f"{base_path}_frame_{actual_frame_idx}{ext}"
        rr.save(save_path)
        print(f"Saving visualization to: {save_path}")
    else:
        rr.spawn(port=9878)
    
    # 设置世界坐标系 Y-UP
    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
    # Create two visualizers with different color schemes
    viz_gt = HandVisualizer(history_len=10, color_scheme='gt')
    viz_pred = HandVisualizer(history_len=10, color_scheme='pred')

    frame_idx = 0
    num_frames = T  # All timesteps for the selected sample
    
    # Store errors for statistics
    all_errors = {
        'left': {'wrist': [], 'Thumb': [], 'Index': [], 'Middle': [], 'Ring': [], 'Little': []},
        'right': {'wrist': [], 'Thumb': [], 'Index': [], 'Middle': [], 'Ring': [], 'Little': []}
    }

    for i in range(num_frames):
        rr.set_time("frame_idx", sequence=frame_idx)
        
        # Log instruction if available
        if zarr_data.get('instruction') is not None:
            instruction_candidates = zarr_data['instruction']  # shape (5,) array of strings
            valid_instructions = []
            if isinstance(instruction_candidates, (np.ndarray, list)):
                for idx, candidate in enumerate(instruction_candidates):
                    candidate_str = str(candidate).strip()
                    if candidate_str:
                        valid_instructions.append(f"[{idx}] {candidate_str}")
            
            if valid_instructions:
                instruction_text = "\n".join(valid_instructions)
                rr.log("/world/instruction", rr.TextLog(instruction_text))

        # 可视化世界坐标系原点和轴
        rr.log("/world/origin", rr.Points3D([0, 0, 0], radii=0.01, colors=[255, 255, 255]))
        rr.log("/world/x", rr.Arrows3D(origins=[0, 0, 0], vectors=[0.1, 0, 0], radii=0.005, colors=[255, 0, 0]))
        rr.log("/world/y", rr.Arrows3D(origins=[0, 0, 0], vectors=[0, 0.1, 0], radii=0.005, colors=[0, 255, 0]))
        rr.log("/world/z", rr.Arrows3D(origins=[0, 0, 0], vectors=[0, 0, 0.1], radii=0.005, colors=[0, 0, 255]))
        
        color_rgb = rgb_images[i]
        depth = depth_images[i]
        
        # zarr stores world2cam, need to invert for visualization
        world2cam = camera_transforms_raw[i]  # (4, 4)
        cam_pose_world = np.linalg.inv(world2cam)  # cam2world
        
        # Get intrinsic for this frame
        K_flat = intrinsics_flat[i]  # [fx, fy, cx, cy]
        K = np.array([
            [K_flat[0], 0, K_flat[2]],
            [0, K_flat[1], K_flat[3]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Log Camera Pose (World Frame)
        rr.log("/world/camera_pose", rr.Transform3D(
            translation=cam_pose_world[:3, 3], 
            mat3x3=cam_pose_world[:3, :3]
        ))
        
        if args.debug:
            rr.log("/world/camera_pose/debug_axes/x", rr.Arrows3D(origins=[0, 0, 0], vectors=[0.1, 0, 0], radii=0.005, colors=[255, 0, 0]))
            rr.log("/world/camera_pose/debug_axes/y", rr.Arrows3D(origins=[0, 0, 0], vectors=[0, 0.1, 0], radii=0.005, colors=[0, 255, 0]))
            rr.log("/world/camera_pose/debug_axes/z", rr.Arrows3D(origins=[0, 0, 0], vectors=[0, 0, 0.1], radii=0.005, colors=[0, 0, 255]))
        
        # Camera Intrinsics & Image
        rr.log("/world/camera_pose/camera", rr.Pinhole(
            image_from_camera=K, 
            width=color_rgb.shape[1], 
            height=color_rgb.shape[0], 
            camera_xyz=rr.ViewCoordinates.RDF 
        ))
        rr.log("/world/camera_pose/camera", rr.Image(color_rgb))
        
        # Resize depth to match image size if needed
        if depth.shape[:2] != color_rgb.shape[:2]:
            depth_resized = cv2.resize(depth, (color_rgb.shape[1], color_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            depth_resized = depth
        
        # Point Cloud
        points, colors = get_colored_point_cloud(
            color_rgb, depth_resized, color_rgb.shape[1], color_rgb.shape[0], K, 
            depth_scale=args.depth_scale, min_depth=args.min_depth, max_depth=args.max_depth
        )
        rr.log("/world/camera_pose/point_cloud", rr.Points3D(points, colors=colors))

        # Hands Visualization - GT and Pred
        identity_transform = np.eye(4, dtype=np.float32)
        
        # Visualize GT hands
        for hand_name in ['left', 'right']:
                if hand_name in gt_hands_data and i < len(gt_hands_data[hand_name]['wrist']):
                    viz_gt.log(
                        root_3d_path="/world/camera_pose/hands/gt", # 3D 
                        image_root_path="/world/camera_pose/camera", # 2D 
                        hand_name=hand_name,
                        hand_data=gt_hands_data[hand_name],
                        frame_idx=i,
                        world_to_camera=identity_transform,  
                        K=K
                    )
                    if i == 0 and args.debug:
                        wrist_pose_camera = gt_hands_data[hand_name]['wrist'][i]
                        print(f"\n=== Frame {i} - GT {hand_name} wrist (Camera Frame) ===")
                        print(f"Translation: {wrist_pose_camera[:3, 3]}")
                        print(f"Rotation matrix shape: {wrist_pose_camera[:3, :3].shape}")
                    
                    if motion_type == 'mano' and 'verts' in gt_hands_data[hand_name]:
                        viz_gt.log_mesh(
                            root_3d_path="/world/camera_pose/hands/gt",
                            hand_name=hand_name,
                            hand_data=gt_hands_data[hand_name],
                            frame_idx=i
                        )
        
        # Visualize Pred hands
        for hand_name in ['left', 'right']:
            if hand_name in pred_hands_data and i < len(pred_hands_data[hand_name]['wrist']):
                viz_pred.log(
                    root_3d_path="/world/camera_pose/hands/pred", # 3D 
                    image_root_path="/world/camera_pose/camera", # 2D 
                    hand_name=hand_name,
                    hand_data=pred_hands_data[hand_name],
                    frame_idx=i,
                    world_to_camera=identity_transform,  
                    K=K
                )
                if i == 0 and args.debug:
                    wrist_pose_camera = pred_hands_data[hand_name]['wrist'][i]
                    print(f"\n=== Frame {i} - Pred {hand_name} wrist (Camera Frame) ===")
                    print(f"Translation: {wrist_pose_camera[:3, 3]}")
                    print(f"Rotation matrix shape: {wrist_pose_camera[:3, :3].shape}")
                
                if motion_type == 'mano' and 'verts' in pred_hands_data[hand_name]:
                    viz_pred.log_mesh(
                        root_3d_path="/world/camera_pose/hands/pred",
                        hand_name=hand_name,
                        hand_data=pred_hands_data[hand_name],
                        frame_idx=i
                    )

        # Calculate and log position errors
        errors = calculate_position_errors(gt_hands_data, pred_hands_data, i)
        
        # Store errors for statistics
        for hand_name in ['left', 'right']:
            all_errors[hand_name]['wrist'].append(errors[hand_name]['wrist'])
            for finger in ['Thumb', 'Index', 'Middle', 'Ring', 'Little']:
                all_errors[hand_name][finger].append(errors[hand_name][finger])
        
        # Format errors as text for display
        error_text = f"Frame {i} Position Errors (meters)\n"
        error_text += "=" * 50 + "\n\n"
        
        # Left hand
        error_text += "LEFT Hand:\n"
        error_text += f"  Wrist:  {errors['left']['wrist']:.6f} m\n"
        error_text += f"  Thumb:  {errors['left']['Thumb']:.6f} m\n"
        error_text += f"  Index:  {errors['left']['Index']:.6f} m\n"
        error_text += f"  Middle: {errors['left']['Middle']:.6f} m\n"
        error_text += f"  Ring:   {errors['left']['Ring']:.6f} m\n"
        error_text += f"  Little: {errors['left']['Little']:.6f} m\n"
        
        
        # Right hand
        error_text += "RIGHT Hand:\n"
        error_text += f"  Wrist:  {errors['right']['wrist']:.6f} m\n"
        error_text += f"  Thumb:  {errors['right']['Thumb']:.6f} m\n"
        error_text += f"  Index:  {errors['right']['Index']:.6f} m\n"
        error_text += f"  Middle: {errors['right']['Middle']:.6f} m\n"
        error_text += f"  Ring:   {errors['right']['Ring']:.6f} m\n"
        error_text += f"  Little: {errors['right']['Little']:.6f} m\n"
        # Log as text document
        rr.log("/position_errors", rr.TextDocument(error_text, media_type=rr.MediaType.TEXT))

        frame_idx += 1

if __name__ == "__main__":
    main()