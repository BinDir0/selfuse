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
    rot_matrix_from_6drot
)
from src.dataset.legendvla_dataset import get_absolute_action

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
            self.structure_color = [200, 200, 200]  # 浅灰
            self.spoke_color = [100, 100, 100]  # 深灰
        
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
                    tip_wrist = tip_arr[frame_idx]  # 4x4 matrix, tip position in wrist frame
                    # Wrist -> Camera: tip_cam = wrist_cam @ tip_wrist
                    tip_cam_homo = wrist_pose_cam @ tip_wrist
                    tip_cam = tip_cam_homo[:3, 3]
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
            rr.log(f"{base_path_2d}/wrist", rr.Points2D([wrist_2d], radii=20, colors=[255, 255, 255]))

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
                            rr.LineStrips2D([[wrist_2d, tip_2d]], radii=5, colors=[150, 150, 150])
                        )
        
        if points_2d:
            rr.log(f"{base_path_2d}/tips", rr.Points2D(points_2d, radii=10, colors=colors_2d))


def get_data_from_zarr(zarr_path, origin_zarr_path=None, sample_idx=None, use_relative_action=None):
    """
    Load data from zarr file.
    
    Args:
        zarr_path: Path to inference results zarr file
        origin_zarr_path: Path to original dataset zarr file. If None, try to read from inference zarr attributes.
        sample_idx: Index of sample to load. If None, randomly select one.
    
    Returns:
        dict with keys: 'image', 'depth', 'intrinsic', 'extrinsic', 'gt_actions', 'sample_idx'
        All arrays have shape (T,) where T=30 is the number of timesteps for the selected sample
        'sample_idx' is the actual sample index used (may be randomly selected if None was passed)
    """
    z = zarr.open(zarr_path, mode='r')
    
    # Get sample indices directly from zarr file
    sample_indices = z['sample_indices'][:]  # (N,) start frame indices in original dataset
    # Get original dataset path: prefer user input, then try zarr attributes
    if origin_zarr_path is None:
        if 'zarr_paths' in z.attrs:
            origin_zarr_path = z.attrs['zarr_paths']
        else:
            raise ValueError("Cannot find original dataset path. Please provide origin_zarr_path parameter or ensure zarr_paths is in zarr attributes.")
    
    if not os.path.exists(origin_zarr_path):
        raise FileNotFoundError(f"Original dataset zarr file not found: {origin_zarr_path}")
    
    # Randomly select a sample if not specified
    num_samples = len(sample_indices)
    if sample_idx is None:
        sample_idx = np.random.randint(0, num_samples)
    
    # sample_indices[i] is the START frame index in original dataset
    # Each sample contains 30 consecutive frames
    start_frame_idx = sample_indices[sample_idx]
    frame_indices = np.arange(start_frame_idx, start_frame_idx + 30)  # 30 consecutive frames
    
    print(f"Selected sample {sample_idx}/{num_samples}")
    print(f"Sample index in zarr: {sample_idx}")
    print(f"Original start frame index: {start_frame_idx}")
    print(f"Frame range: {frame_indices[0]} to {frame_indices[-1]} (30 frames)")
    print(f"Original dataset path: {origin_zarr_path}")
    
    # Get gt_actions and pred_actions for selected sample: (30, 48)
    gt_actions = z['gt_actions'][sample_idx]  # (30, 48)
    pred_actions = z['pred_actions'][sample_idx]  # (30, 48)
    
    # Check if actions are relative
    # Priority: user parameter > zarr attributes > default True (as per config)
    if use_relative_action is None:
        use_relative_action = z.attrs.get('use_relative_action', True)
    
    # Load original dataset
    origin_z = zarr.open(origin_zarr_path, mode='r')

    image = origin_z['data']['image'][frame_indices]  # (30, H, W, 3) uint8
    depth = origin_z['data']['depth'][frame_indices]  # (30, H, W) uint16
    intrinsic = origin_z['data']['intrinsic'][frame_indices]  # (30, 4) float32 [fx, fy, cx, cy]
    extrinsic_flat = origin_z['data']['extrinsic'][frame_indices]  # (30, 16) float32 (world2cam, 4x4 flattened)
    
    # Reshape extrinsic from (30, 16) to (30, 4, 4)
    extrinsic = extrinsic_flat.reshape(-1, 4, 4)  # (30, 4, 4)
    
    if use_relative_action:
        # history is typically 30, the initial state is at start_frame_idx
        state_frame_idx = start_frame_idx
        
        # Load wrist and hand state from original dataset
        # Note: state is in WORLD coordinate system
        # Format: [left_trans3, right_trans3, left_6drot, right_6drot, left_hand, right_hand]
        wrist_state_world = origin_z['data']['state']['wrist'][state_frame_idx]  # (18,) = 2*9 (trans3 + 6drot for each hand)
        hand_state_world = origin_z['data']['state']['fingertips'][state_frame_idx]  # (30,) = 2*15 (15 keypoints for each hand)
        
        # Get camera extrinsic (world2cam) for the state frame
        # Note: extrinsic is already loaded above, but we need the one at state_frame_idx
        state_extrinsic_flat = origin_z['data']['extrinsic'][state_frame_idx]  # (16,) float32 (world2cam, 4x4 flattened)
        world2cam = state_extrinsic_flat.reshape(4, 4)  # (4, 4) world2cam
        
        # Convert wrist state from world to camera coordinate system
        # wrist_state_world format: [left_trans3, right_trans3, left_6drot, right_6drot]
        wrist_state_cam = np.zeros_like(wrist_state_world)
        
        for idx in range(2):  # left and right hand
            # Get wrist transform in world coordinate
            trans_world = wrist_state_world[idx*3 : idx*3+3]
            rot_6d_world = wrist_state_world[6+idx*6 : 6+idx*6+6]
            wrist_world_homo = homo_matrix_from_trans_6drot(trans_world, rot_6d_world)  # [4, 4]
            
            # Transform to camera coordinate: T_cam_wrist = world2cam @ T_world_wrist
            wrist_cam_homo = world2cam @ wrist_world_homo  # [4, 4]
            
            # Convert back to trans and 6drot
            trans_cam, rot_6d_cam = homo_matrix_to_trans_6drot(wrist_cam_homo)
            wrist_state_cam[idx*3 : idx*3+3] = trans_cam
            wrist_state_cam[6+idx*6 : 6+idx*6+6] = rot_6d_cam
        
        # Convert fingertip keypoints: first to camera frame, then to wrist frame
        # Following process_state_action: transform_hand_points_to_wrist_frame
        # hand_state_world format: [left_keypoints_15, right_keypoints_15] = (30,) = 2*15
        hand_state_wrist = np.zeros_like(hand_state_world)
        
        for idx in range(2):  # left and right hand
            keypoints_world = hand_state_world[idx*15 : idx*15+15].reshape(5, 3)  # (5, 3) - 5 fingers, 3 coords each
            
            # Step 1: Transform from world to camera: P_cam = world2cam @ P_world
            keypoints_world_homo = np.concatenate([keypoints_world, np.ones((5, 1))], axis=-1)  # (5, 4)
            keypoints_cam_homo = (world2cam @ keypoints_world_homo.T).T  # (5, 4)
            keypoints_cam = keypoints_cam_homo[:, :3]  # (5, 3)
            
            # Step 2: Transform from camera to wrist frame: P_wrist = pinv(T_cam_wrist) @ P_cam
            # Get wrist transform in camera frame (already computed above)
            wrist_cam_homo = homo_matrix_from_trans_6drot(
                wrist_state_cam[idx*3 : idx*3+3],
                wrist_state_cam[6+idx*6 : 6+idx*6+6]
            )  # [4, 4] T_cam_wrist
            
            # Get T_wrist_cam = pinv(T_cam_wrist)
            wrist_wrist2cam_homo = np.linalg.pinv(wrist_cam_homo)  # [4, 4] T_wrist_cam
            
            # Transform keypoints from camera to wrist frame
            keypoints_cam_homo = np.concatenate([keypoints_cam, np.ones((5, 1))], axis=-1)  # (5, 4)
            keypoints_wrist_homo = (wrist_wrist2cam_homo @ keypoints_cam_homo.T).T  # (5, 4)
            keypoints_wrist = keypoints_wrist_homo[:, :3]  # (5, 3)
            
            hand_state_wrist[idx*15 : idx*15+15] = keypoints_wrist.reshape(-1)
        
        # Convert to format expected by get_absolute_action: [wrist_dim + hand_dim]
        # wrist_dim = 18 (left_trans3 + right_trans3 + left_6drot + right_6drot) - in camera coords
        # hand_dim = 30 (left_keypoints_15 + right_keypoints_15) - in wrist coords (matching process_state_action)
        initial_state = np.concatenate([wrist_state_cam, hand_state_wrist])  # (48,)
        
        print(f"Using relative actions, initial state at frame {state_frame_idx}")
        print(f"Initial state shape: {initial_state.shape}")
        print(f"Converted state from world to camera coordinate system")
        
        # Recover absolute actions from relative actions
        # Now both initial_state and gt_actions/pred_actions are in camera coordinate system
        gt_actions = get_absolute_action(initial_state, gt_actions)
        pred_actions = get_absolute_action(initial_state, pred_actions)
        print("Converted relative actions to absolute actions")
    
    data = {
        'image': image,  # (30, H, W, 3) uint8
        'depth': depth,  # (30, H, W) uint16
        'intrinsic': intrinsic,  # (30, 4) float32 [fx, fy, cx, cy]
        'extrinsic': extrinsic,  # (30, 4, 4) float32 (world2cam)
        'gt_actions': gt_actions,  # (30, 48) float32
        'pred_actions': pred_actions,  # (30, 48) float32
        'sample_idx': sample_idx,  # int - actual sample index used
    }
    return data


def gt_actions_to_hands_data(gt_actions):
    """
    Convert gt_actions to hands_data format for all timesteps.
    
    gt_actions format (T, 48) where T=30 is number of timesteps:
    For each timestep:
    - left_trans3: [0:3]
    - right_trans3: [3:6]
    - left_6d_rotation: [6:12]
    - right_6d_rotation: [12:18]
    - left_keypoints_15: [18:33] (5 fingers * 3 coords)
    - right_keypoints_15: [33:48] (5 fingers * 3 coords)
    
    Args:
        gt_actions: np.ndarray, shape (T, 48) where T=30
    
    Returns:
        dict with 'left' and 'right' keys, each containing:
        - 'wrist': array of shape (T, 4, 4) transform matrices
        - 'fingers': dict with finger names and arrays of shape (T, 4, 4) transform matrices
    """
    T = gt_actions.shape[0]  # Number of timesteps (30)
    
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", type=str, required=True, help="Path to zarr file with inference results.")
    parser.add_argument("--origin_zarr_path", type=str, default=None, help="Path to original dataset zarr file. If None, try to read from inference zarr attributes.")
    parser.add_argument("--sample_idx", type=int, default=None, help="Index of sample to visualize. If None, randomly select one.")
    parser.add_argument("--target_width", type=int, default=None, help="Target image width. If None, use original width.")
    parser.add_argument("--target_height", type=int, default=None, help="Target image height. If None, use original height.")
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=1.5)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save rrd file. If None, use spawn/notebook_show.")
    parser.add_argument("--use_relative_action", type=lambda x: (str(x).lower() == 'true'), default=None, help="Whether actions are relative. If None, try to read from zarr attributes, default to True.")
    args = parser.parse_args()

    if not os.path.exists(args.zarr_path):
        raise FileNotFoundError(f"Zarr file not found: {args.zarr_path}")

    print(f"Loading data from zarr: {args.zarr_path}")
    zarr_data = get_data_from_zarr(args.zarr_path, origin_zarr_path=args.origin_zarr_path, sample_idx=args.sample_idx, use_relative_action=args.use_relative_action)
    
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
    
    # Get actual sample_idx that was used
    actual_sample_idx = zarr_data['sample_idx']
    
    # Initialize Rerun with unique app name based on sample_idx
    app_name = f"rgbd_hand_action_visv2_sample_{actual_sample_idx}"
    rr.init(app_name)
    
    # Save to rrd file if save_path is provided, otherwise use spawn
    if args.save_path:
        # If save_path is a directory, create unique filename with sample_idx
        if os.path.isdir(args.save_path):
            save_filename = f"visualization_sample_{actual_sample_idx}.rrd"
            save_path = os.path.join(args.save_path, save_filename)
        else:
            # If it's a file, add sample_idx to filename
            base_path = os.path.splitext(args.save_path)[0]
            ext = os.path.splitext(args.save_path)[1] or '.rrd'
            save_path = f"{base_path}_sample_{actual_sample_idx}{ext}"
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

    for i in range(num_frames):
        rr.set_time("frame_idx", sequence=frame_idx)
        
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

        frame_idx += 1

if __name__ == "__main__":
    main()