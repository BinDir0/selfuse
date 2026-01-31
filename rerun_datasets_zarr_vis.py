import os
import sys
import argparse
import cv2
import rerun as rr
import numpy as np
import collections
import zarr

from src.utils.geometry import (
    homo_matrix_from_trans_6drot,
    homo_matrix_to_trans_6drot,
    rot_matrix_to_6drot,
    rot_matrix_from_6drot,
    transform_hand_points_to_wrist_frame,
    transform_wrist_to_target_frame,
    transform_hand_points_to_target_frame,
)
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
        # Extract action type (state/action) from root_3d_path for 2D overlay path
        # root_3d_path format: "/world/camera_pose/hands/state" or "/world/camera_pose/hands/action"
        action_type = root_3d_path.split('/')[-1] if '/' in root_3d_path else 'state'
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
            # Use structure_color for 2D wrist (same as rerun_inference_vis.py)
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
                        # Use spoke_color for 2D bones (same as rerun_inference_vis.py)
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

def get_data_from_zarr(origin_zarr_path, frame_idx=None, horizon=30, motion_type='fingertips'):
    """
    Load data directly from original zarr dataset.
    
    Args:
        origin_zarr_path: Path to original dataset zarr file
        frame_idx: Index of frame to load. If None, randomly select one from valid frames.
        horizon: Number of timesteps to load (default: 30)
    
    Returns:
        dict with keys: 'image', 'depth', 'intrinsic', 'extrinsic', 'state', 'action', 'instruction', 'frame_idx'
        All arrays have shape (T,) where T=horizon is the number of timesteps for the selected sample
        'frame_idx' is the actual frame index used (may be randomly selected if None was passed)
    """
    if not os.path.exists(origin_zarr_path):
        raise FileNotFoundError(f"Original dataset zarr file not found: {origin_zarr_path}")
    
    # Load original dataset
    origin_z = zarr.open(origin_zarr_path, mode='r')
    data_group = origin_z['data']
    
    # Get dataset size from image data
    dataset_size = data_group['image'].shape[0]
    episode_ends = np.array(origin_z['meta']['episode_ends'][:])
    
    print(f"Dataset size: {dataset_size}, Horizon: {horizon}")
    
    # Find valid frames that are at least 'horizon' frames away from episode end

    valid_frames = []
    episode_start = 0
    
    for episode_end in episode_ends:
        # For this episode, valid frames are those where frame + horizon <= episode_end
        # i.e., frame <= episode_end - horizon
        episode_valid_end = episode_end - horizon
        if episode_valid_end >= episode_start:
            # Add all valid frames in this episode
            episode_valid = np.arange(episode_start, episode_valid_end + 1)
            valid_frames.append(episode_valid)
        episode_start = episode_end
    
    if len(valid_frames) > 0:
        valid_frames = np.concatenate(valid_frames)
    else:
        valid_frames = np.array([], dtype=np.int64)
    
    if len(valid_frames) == 0:
        raise ValueError(f"No valid frames found that are at least {horizon} frames away from episode end.")
    
    print(f"Found {len(valid_frames)} valid frames (out of {dataset_size} total)")
 
    
    # Randomly select a frame if not specified
    if frame_idx is None:
        if len(valid_frames) == 0:
            raise ValueError(f"No valid frames available. Dataset size: {dataset_size}, horizon: {horizon}")
        frame_idx = np.random.choice(valid_frames)
    else:
        if frame_idx >= dataset_size:
            raise ValueError(f"frame_idx {frame_idx} is out of range (dataset size: {dataset_size})")
        # Check if the selected frame is valid
        if episode_ends is not None:
            episode_idx = np.searchsorted(episode_ends, frame_idx, side='right')
            if episode_idx < len(episode_ends):
                episode_end = episode_ends[episode_idx]
                if frame_idx + horizon > episode_end:
                    raise ValueError(f"frame_idx {frame_idx} is too close to episode end (episode ends at {episode_end}, need {horizon} frames)")
    
    # Frame indices for this sample
    frame_indices = np.arange(frame_idx, frame_idx + horizon) 
    
    # Final check: ensure frame_indices don't exceed dataset size or episode end
    if len(frame_indices) > 0 and frame_indices[-1] >= dataset_size:
        # Truncate to available frames
        valid_mask = frame_indices < dataset_size
        frame_indices = frame_indices[valid_mask]
        horizon = len(frame_indices)
    # Check episode boundary
    episode_idx = np.searchsorted(episode_ends, frame_idx, side='right')
    if episode_idx < len(episode_ends):
        episode_end = episode_ends[episode_idx]
        if len(frame_indices) > 0 and frame_indices[-1] >= episode_end:
            # Truncate to episode end
            valid_mask = frame_indices < episode_end
            frame_indices = frame_indices[valid_mask]
            horizon = len(frame_indices)
    
    # Ensure we have at least one frame
    if len(frame_indices) == 0:
        raise ValueError(f"After truncation, no valid frames remain for frame_idx {frame_idx} with horizon {horizon}")
    
    print(f"Selected frame {frame_idx} from dataset (size: {dataset_size})")
    print(f"Frame range: {frame_indices[0]} to {frame_indices[-1]} ({horizon} frames)")
    print(f"Dataset path: {origin_zarr_path}")
    
    # Load image, depth, intrinsic, extrinsic, state, action from original dataset
    image = data_group['image'][frame_indices]  # (H, H_img, W_img, 3) uint8
    depth = data_group['depth'][frame_indices]  # (H, H_img, W_img) uint16
    intrinsic = data_group['intrinsic'][frame_indices]  # (H, 4) float32 [fx, fy, cx, cy]
    extrinsic_flat = data_group['extrinsic'][frame_indices]  # (H, 16) float32 (world2cam, 4x4 flattened)
    extrinsic = extrinsic_flat.reshape(-1, 4, 4)  # (H, 4, 4)
    wrist_state_world = data_group['state']['wrist'][frame_indices]  # (H, 18) = 2*9 (trans3 + 6drot for each hand)
    hand_state_world = data_group['state']['fingertips'][frame_indices]  # (H, 30) = 2*15 (15 keypoints for each hand)
    wrist_action_world = data_group['action']['wrist'][frame_indices]  # (H, 18) = 2*9 (trans3 + 6drot for each hand)
    hand_action_world = data_group['action']['fingertips'][frame_indices]  # (H, 30) = 2*15 (15 keypoints for each hand)
    
    # Transform to camera coordinate
    wrist_state_cam = transform_wrist_to_target_frame(wrist_state_world, extrinsic)  # (H, 18) in camera coordinate
    hand_state_cam = transform_hand_points_to_target_frame(hand_state_world, extrinsic)  # (H, 30) in camera coordinate
    state_cam = np.concatenate([wrist_state_cam, hand_state_cam], axis=-1)  # (H, 48) in camera coordinate
    wrist_action_cam = transform_wrist_to_target_frame(wrist_action_world, extrinsic)  # (H, 18) in camera coordinate
    fingertips_action_cam = transform_hand_points_to_target_frame(hand_action_world, extrinsic)  # (H, 30) in camera coordinate
    action_cam = np.concatenate([wrist_action_cam, fingertips_action_cam], axis=-1)  # (H, 48) in camera coordinate
    
    # Load MANO data if available
    mano_state = None
    mano_action = None
    mano_shape = None
    if motion_type == 'mano':
        if 'mano' in data_group['state'] and 'mano' in data_group['action']:
            mano_state = data_group['state']['mano'][frame_indices]  # (H, 90)
            mano_action = data_group['action']['mano'][frame_indices]  # (H, 90)
            mano_shape = data_group['state']['shape'][frame_indices]  # (H, 20)
        else:
            print(f"Warning: MANO data not found in {origin_zarr_path}. Falling back to fingertips.")

    # Load instruction
    # instruction shape: (H, 5) - each timestep has 5 candidate instructions
    if 'instruction' in data_group:
        instruction = data_group['instruction'][frame_indices]  # (H, 5) object array
    else:
        instruction = None

    data = {
        'image': image,  # (H, H_img, W_img, 3) uint8
        'depth': depth,  # (H, H_img, W_img) uint16
        'intrinsic': intrinsic,  # (H, 4) float32 [fx, fy, cx, cy]
        'extrinsic': extrinsic,  # (H, 4, 4) float32 (world2cam)
        'state': state_cam,  # (H, 48) float32
        'action': action_cam,  # (H, 48) float32
        'mano_state': mano_state,
        'mano_action': mano_action,
        'mano_shape': mano_shape,
        'instruction': instruction,  # (H,) string array or None
        'frame_idx': frame_idx,  # int - actual frame index used
    }
    return data


def actions_to_hands_data(actions, motion_type='fingertips', mano_data=None):
    """
    Convert actions to hands_data format for all timesteps.
    
    actions format (T, 48) or (T, 90) depending on motion_type.
    
    Args:
        actions: np.ndarray, shape (T, 48) or (T, 90)
        motion_type: 'fingertips' or 'mano'
        mano_data: dict with 'theta' and 'beta' if motion_type is 'mano'
    
    Returns:
        dict with 'left' and 'right' keys, each containing:
        - 'wrist': array of shape (T, 4, 4) transform matrices
        - 'fingers': dict with finger names and arrays of shape (T, 4, 4) transform matrices
        - 'verts': array of shape (T, V, 3) vertices (only for mano)
        - 'faces': array of shape (F, 3) faces (only for mano)
    """
    T = actions.shape[0]  # Number of timesteps
    
    if motion_type == 'mano' and mano_data is not None:
        # actions here are wrist actions (T, 18)
        # mano_data['theta'] is (T, 90)
        # mano_data['beta'] is (T, 20)
        
        wrist_actions = actions # (T, 18)
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
                'wrist': mano_results['left']['joints'][:, 0].numpy(), # Not actually used for mesh but kept for consistency
                'verts': mano_results['left']['verts'],
                'faces': mano_results['left']['faces'],
                'fingers': {} # Can be populated if needed
            },
            'right': {
                'wrist': mano_results['right']['joints'][:, 0].numpy(),
                'verts': mano_results['right']['verts'],
                'faces': mano_results['right']['faces'],
                'fingers': {}
            }
        }
        
        # Also build wrist matrices for axes visualization
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
        action = actions[t]  # (48,)
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_zarr_path", type=str, required=True, help="Path to original dataset zarr file.")
    parser.add_argument("--frame_idx", type=int, nargs='*', default=None, help="Index(es) of frame(s) to visualize. Can specify multiple frames: --frame_idx 100 200 300. If None, randomly select from valid frames.")
    parser.add_argument("--horizon", type=int, default=100, help="Number of timesteps to visualize (default: 30).")
    parser.add_argument("--target_width", type=int, default=None, help="Target image width. If None, use original width.")
    parser.add_argument("--target_height", type=int, default=None, help="Target image height. If None, use original height.")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=1.5)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save rrd file. If None, use spawn/notebook_show.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to save (default: 100).")
    parser.add_argument("--motion_type", type=str, default="fingertips", choices=['fingertips', 'mano'], help="Motion type to visualize.")
    args = parser.parse_args()

    if not os.path.exists(args.origin_zarr_path):
        raise FileNotFoundError(f"Zarr file not found: {args.origin_zarr_path}")

    # Handle frame_idx: can be empty list (None equivalent), or list of ints
    # With nargs='*', args.frame_idx is always a list (empty if not provided)
    if args.frame_idx is None or len(args.frame_idx) == 0:
        frame_indices_to_process = None
        num_samples_to_process = args.num_samples
    else:
        # frame_idx is a list of frame indices
        frame_indices_to_process = args.frame_idx
        num_samples_to_process = len(frame_indices_to_process)
        if args.num_samples > 1 and args.num_samples != num_samples_to_process:
            print(f"Warning: frame_idx is specified ({frame_indices_to_process}), ignoring num_samples ({args.num_samples}). Will process {num_samples_to_process} frame(s).")

    # Create save directory if it doesn't exist
    if args.save_path:
        if os.path.isdir(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
        else:
            # If it's a file path, create parent directory
            parent_dir = os.path.dirname(args.save_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

    print(f"Starting to process {num_samples_to_process} sample(s) from: {args.origin_zarr_path}")
    if frame_indices_to_process is not None:
        print(f"Using specified frame_idx: {frame_indices_to_process}")
    else:
        print(f"Will randomly select {num_samples_to_process} frame(s)")

    for sample_count in range(num_samples_to_process):
        print(f"\n{'='*60}")
        print(f"Processing Sample {sample_count + 1}/{num_samples_to_process}")
        print(f"{'='*60}")
        
        # Get frame_idx for this iteration
        if frame_indices_to_process is not None:
            current_frame_idx = frame_indices_to_process[sample_count]
            print(f"Processing frame_idx: {current_frame_idx}")
        else:
            current_frame_idx = None
        
        # Load data from zarr (randomly selects frame if current_frame_idx is None)
        zarr_data = get_data_from_zarr(args.origin_zarr_path, frame_idx=current_frame_idx, horizon=args.horizon, motion_type=args.motion_type)
    
        # Get actual frame_idx that was used (needed for filename)
        actual_frame_idx = zarr_data['frame_idx']
        
        # Save instruction to txt file if available
        if zarr_data['instruction'] is not None:
            all_instructions = zarr_data['instruction']  # Get all instructions (T,) string array
            if len(all_instructions) > 0 and args.save_path:
                # Determine the base path for txt file (same as rrd file)
                if os.path.isdir(args.save_path):
                    txt_filename = f"sample_{sample_count:03d}_frame_{actual_frame_idx}.txt"
                    txt_path = os.path.join(args.save_path, txt_filename)
                else:
                    base_path = os.path.splitext(args.save_path)[0]
                    txt_path = f"{base_path}_sample_{sample_count:03d}_frame_{actual_frame_idx}.txt"
                
                # Write all instructions to txt file
                with open(txt_path, 'w', encoding='utf-8') as f:
                    for i, instr in enumerate(all_instructions):
                        f.write(f"{instr}\n")
                print(f"Saved instruction to: {txt_path}")
        
        # Extract images and depth: (T, H, W, 3) and (T, H, W)
        rgb_images = np.array(zarr_data['image'])  # (T, H, W, 3) uint8
        depth_images = np.array(zarr_data['depth']).astype(np.float32) / 1000.0  # (T, H, W) uint16 -> float32 meters
        
        # Extract intrinsics: (T, 4) [fx, fy, cx, cy]
        intrinsics_flat = zarr_data['intrinsic']  # (T, 4)
        
        # Extract extrinsics: (T, 4, 4) world2cam
        camera_transforms_raw = zarr_data['extrinsic']  # (T, 4, 4) world2cam
        
        # Convert state and action to hands_data format: (T, 48) -> hands_data with T frames
        states = zarr_data['state']  # (T, 48)
        actions = zarr_data['action']  # (T, 48)
        
        if args.motion_type == 'mano' and zarr_data['mano_state'] is not None:
            state_hands_data = actions_to_hands_data(
                states[:, :18], 
                motion_type='mano', 
                mano_data={'theta': zarr_data['mano_state'], 'beta': zarr_data['mano_shape']}
            )
            action_hands_data = actions_to_hands_data(
                actions[:, :18], 
                motion_type='mano', 
                mano_data={'theta': zarr_data['mano_action'], 'beta': zarr_data['mano_shape']}
            )
        else:
            state_hands_data = actions_to_hands_data(states)
            action_hands_data = actions_to_hands_data(actions)
        
        T = len(rgb_images)  # Number of timesteps
        print(f"Loaded {T} frames (timesteps) from selected sample")
        print(f"State hands data: left={len(state_hands_data['left']['wrist'])}, right={len(state_hands_data['right']['wrist'])}")
        print(f"Action hands data: left={len(action_hands_data['left']['wrist'])}, right={len(action_hands_data['right']['wrist'])}")
        
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
        
        # Initialize Rerun with unique app name based on sample count and frame_idx
        app_name = f"sample_{sample_count:03d}_frame_{actual_frame_idx}"
        rr.init(app_name, spawn=False)
        
        # Save to rrd file if save_path is provided, otherwise use spawn (only for first sample)
        if args.save_path:
            # If save_path is a directory, create unique filename with sample count and frame_idx
            if os.path.isdir(args.save_path):
                save_filename = f"sample_{sample_count:03d}_frame_{actual_frame_idx}.rrd"
                current_save_path = os.path.join(args.save_path, save_filename)
            else:
                # If it's a file, add sample count and frame_idx to filename
                base_path = os.path.splitext(args.save_path)[0]
                ext = os.path.splitext(args.save_path)[1] or '.rrd'
                current_save_path = f"{base_path}_sample_{sample_count:03d}_frame_{actual_frame_idx}{ext}"
            rr.save(current_save_path)
            print(f"Saving visualization to: {current_save_path}")
        elif sample_count == 0:
            # Only spawn for the first sample if no save_path is provided
            rr.spawn(port=9878)
        
        # 设置世界坐标系 Y-UP
        rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
        # Create two visualizers with different color schemes: state (gt) and action (pred)
        viz_state = HandVisualizer(history_len=10, color_scheme='gt')
        viz_action = HandVisualizer(history_len=10, color_scheme='pred')

        frame_idx = 0
        num_frames = T  # All timesteps for the selected sample

        for i in range(num_frames):
            rr.set_time("frame_idx", sequence=frame_idx)
            
            # Log instruction if available
            # instruction shape: (H, 5) - each timestep has 5 candidate instructions
            if zarr_data['instruction'] is not None and i < len(zarr_data['instruction']):
                instruction_candidates = zarr_data['instruction'][i]  # shape (5,) array of strings
                # Collect all non-empty instructions
                valid_instructions = []
                if isinstance(instruction_candidates, np.ndarray):
                    for idx, candidate in enumerate(instruction_candidates):
                        candidate_str = str(candidate).strip()
                        if candidate_str:
                            valid_instructions.append(f"[{idx}] {candidate_str}")
                # Log all instructions, separated by newlines
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

            # Hands Visualization - State and Action
            identity_transform = np.eye(4, dtype=np.float32)
            
            # Visualize State hands (like GT)
            for hand_name in ['left', 'right']:
                if hand_name in state_hands_data and i < len(state_hands_data[hand_name]['wrist']):
                    viz_state.log(
                        root_3d_path="/world/camera_pose/hands/state", # 3D 
                        image_root_path="/world/camera_pose/camera", # 2D 
                        hand_name=hand_name,
                        hand_data=state_hands_data[hand_name],
                        frame_idx=i,
                        world_to_camera=identity_transform,  
                        K=K
                    )
                    if i == 0 and args.debug:
                        wrist_pose_camera = state_hands_data[hand_name]['wrist'][i]
                        print(f"\n=== Frame {i} - State {hand_name} wrist (Camera Frame) ===")
                        print(f"Translation: {wrist_pose_camera[:3, 3]}")
                        print(f"Rotation matrix shape: {wrist_pose_camera[:3, :3].shape}")
                    
                    if args.motion_type == 'mano' and 'verts' in state_hands_data[hand_name]:
                        viz_state.log_mesh(
                            root_3d_path="/world/camera_pose/hands/state",
                            hand_name=hand_name,
                            hand_data=state_hands_data[hand_name],
                            frame_idx=i
                        )
            
            # Visualize Action hands (like Pred)
            for hand_name in ['left', 'right']:
                if hand_name in action_hands_data and i < len(action_hands_data[hand_name]['wrist']):
                    viz_action.log(
                        root_3d_path="/world/camera_pose/hands/action", # 3D 
                        image_root_path="/world/camera_pose/camera", # 2D 
                        hand_name=hand_name,
                        hand_data=action_hands_data[hand_name],
                        frame_idx=i,
                        world_to_camera=identity_transform,  
                        K=K
                    )
                    if i == 0 and args.debug:
                        wrist_pose_camera = action_hands_data[hand_name]['wrist'][i]
                        print(f"\n=== Frame {i} - Action {hand_name} wrist (Camera Frame) ===")
                        print(f"Translation: {wrist_pose_camera[:3, 3]}")
                        print(f"Rotation matrix shape: {wrist_pose_camera[:3, :3].shape}")

                    if args.motion_type == 'mano' and 'verts' in action_hands_data[hand_name]:
                        viz_action.log_mesh(
                            root_3d_path="/world/camera_pose/hands/action",
                            hand_name=hand_name,
                            hand_data=action_hands_data[hand_name],
                            frame_idx=i
                        )

            frame_idx += 1
        
        print(f"Completed sample {sample_count + 1}/{num_samples_to_process}")
    
    print(f"\n{'='*60}")
    print(f"Finished processing all {num_samples_to_process} sample(s)!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()