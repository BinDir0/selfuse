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


# ── helpers for resolving dataset name → zarr path ──────────────────────────

def _load_default_mapping():
    """Load ``human_dataset_mapping`` from the inference config."""
    cfg = OmegaConf.load(INFERENCE_CONFIG_PATH)
    raw = OmegaConf.select(cfg, "human_dataset_mapping")
    if raw is None:
        raise RuntimeError(
            f"human_dataset_mapping not found in {INFERENCE_CONFIG_PATH}"
        )
    return OmegaConf.to_container(raw, resolve=True)


def _navigate_zarr(data_group, key_path, indices=None):
    """Navigate a zarr group using ``/``-separated *key_path*."""
    parts = key_path.split('/')
    cur = data_group
    for p in parts:
        cur = cur[p]
    return cur[indices] if indices is not None else cur


def build_dataset_registry(inference_zarr_path):
    """Return ``{name: {'path': str, 'mapping': dict}}`` from the inference config.

    Reads ``vla_dataset_paths`` from the saved ``inference_config.yaml``
    next to the zarr, falling back to ``INFERENCE_CONFIG_PATH``.
    """
    inference_dir = os.path.dirname(os.path.abspath(inference_zarr_path))
    saved_cfg_path = os.path.join(inference_dir, "inference_config.yaml")
    cfg_path = saved_cfg_path if os.path.exists(saved_cfg_path) else INFERENCE_CONFIG_PATH

    cfg = OmegaConf.load(cfg_path)
    items = OmegaConf.select(cfg, "vla_dataset_paths", default=[])
    if not items:
        raise RuntimeError(f"vla_dataset_paths not found in {cfg_path}")

    registry: dict = {}
    for item in items:
        mapping_raw = OmegaConf.to_container(item.get('mapping', {}), resolve=True)
        if not isinstance(mapping_raw, dict) or not mapping_raw:
            mapping_raw = _load_default_mapping()
        registry[item['name']] = {'path': item['path'], 'mapping': mapping_raw}
    return registry


def select_sample_indices(inference_zarr_path, dataset_names=None,
                          num_samples=1, sample_idx=None):
    """Pick sample indices, optionally filtering by *dataset_names*."""
    inf_z = zarr.open(inference_zarr_path, mode='r')

    if sample_idx is not None:
        return [sample_idx]

    num_total = inf_z['pred_actions'].shape[0]

    if dataset_names and 'dataset_name' in inf_z:
        all_names = [str(x) for x in inf_z['dataset_name'][:]]
        name_set = set(dataset_names)
        valid = np.array([i for i, n in enumerate(all_names) if n in name_set])
        if len(valid) == 0:
            avail = sorted(set(all_names))
            raise ValueError(
                f"No samples found for {dataset_names}. Available: {avail}"
            )
    else:
        valid = np.arange(num_total)

    n = min(num_samples, len(valid))
    sel = np.random.choice(valid, n, replace=False)
    return sorted(sel.tolist())


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

def get_data_from_zarr(inference_zarr_path, origin_zarr_path=None, sample_idx=None,
                       use_relative_action=None, motion_type='fingertips',
                       dataset_registry=None):
    """
    Load data from inference results zarr file and corresponding original dataset.
    
    Args:
        inference_zarr_path: Path to inference results zarr file.
        origin_zarr_path: Explicit path to original dataset zarr (takes precedence).
        sample_idx: Index of sample in inference results. If None, randomly select.
        use_relative_action: Whether actions are relative. If None, default to True.
        motion_type: 'fingertips' or 'mano'.
        dataset_registry: ``{name: {'path', 'mapping'}}`` built by
            :func:`build_dataset_registry`.  Used to auto-resolve *origin_zarr_path*
            and field-key mapping from the sample's ``dataset_name``.
    
    Returns:
        dict with image, depth, intrinsic, extrinsic, gt/pred_actions, frame_idx, dataset_name …
    """
    if not os.path.exists(inference_zarr_path):
        raise FileNotFoundError(f"Inference results zarr file not found: {inference_zarr_path}")
    
    inference_z = zarr.open(inference_zarr_path, mode='r')
    
    required_keys = ['pred_actions', 'gt_actions', 'actions_valid_mask', 'dataset_local_idx']
    for key in required_keys:
        if key not in inference_z:
            raise ValueError(f"{key} not found in inference results.")
    
    pred_actions_all = inference_z['pred_actions']
    gt_actions_all = inference_z['gt_actions']
    actions_valid_mask_all = inference_z['actions_valid_mask']
    origin_frame_indices_all = inference_z['dataset_local_idx']
    
    num_total = pred_actions_all.shape[0]
    horizon = pred_actions_all.shape[1]
    
    print(f"Inference results: {num_total} samples, Horizon: {horizon}")
    
    if sample_idx is None:
        sample_idx = np.random.randint(0, num_total)
    elif sample_idx >= num_total:
        raise ValueError(f"sample_idx {sample_idx} out of range ({num_total})")
    
    pred_actions = pred_actions_all[sample_idx]
    gt_actions = gt_actions_all[sample_idx]
    actions_valid_mask = actions_valid_mask_all[sample_idx]
    origin_frame_idx = int(origin_frame_indices_all[sample_idx])

    # Resolve dataset_name and field mapping for this sample
    sample_dataset_name = None
    field_mapping = _load_default_mapping()

    if 'dataset_name' in inference_z:
        ds_arr = inference_z['dataset_name']
        # Guard against corrupted char-level storage
        if ds_arr.shape[0] == num_total:
            sample_dataset_name = str(ds_arr[sample_idx])
        else:
            print(f"Warning: dataset_name corrupted, cannot resolve for sample {sample_idx}")

    if origin_zarr_path is None:
        if sample_dataset_name and dataset_registry and sample_dataset_name in dataset_registry:
            entry = dataset_registry[sample_dataset_name]
            origin_zarr_path = entry['path']
            field_mapping = entry.get('mapping', field_mapping)
        elif 'origin_zarr_path' in inference_z.attrs:
            origin_zarr_path = inference_z.attrs['origin_zarr_path']
        else:
            raise ValueError(
                f"Cannot resolve origin zarr for sample {sample_idx} "
                f"(dataset_name={sample_dataset_name!r}). "
                "Provide --origin_zarr_path or ensure config is accessible."
            )
    elif sample_dataset_name and dataset_registry and sample_dataset_name in dataset_registry:
        field_mapping = dataset_registry[sample_dataset_name].get('mapping', field_mapping)

    print(f"Selected sample {sample_idx}, dataset={sample_dataset_name}, "
          f"origin_frame={origin_frame_idx}, zarr={origin_zarr_path}")
    
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
    dataset_size = _navigate_zarr(data_group, field_mapping['image']).shape[0]
    
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
    
    # Load image, depth, intrinsic, extrinsic from original dataset using field_mapping
    fm = field_mapping
    image = _navigate_zarr(data_group, fm['image'], frame_indices)
    depth = _navigate_zarr(data_group, fm['depth'], frame_indices)
    intrinsic = _navigate_zarr(data_group, fm['intrinsic'], frame_indices)
    extrinsic_flat = _navigate_zarr(data_group, fm['extrinsic'], frame_indices)
    extrinsic = extrinsic_flat.reshape(-1, 4, 4)
    
    if use_relative_action:
        state_frame_idx = origin_frame_idx
        
        wrist_state_world = _navigate_zarr(data_group, fm['wrist_state'], state_frame_idx)
        hand_state_world = _navigate_zarr(data_group, fm['hand_state'], state_frame_idx)
        
        state_extrinsic_flat = _navigate_zarr(data_group, fm['extrinsic'], state_frame_idx)
        world2cam = state_extrinsic_flat.reshape(4, 4)
        
        hand_state_world_reshaped = hand_state_world.reshape(1, -1)
        wrist_state_world_reshaped = wrist_state_world.reshape(1, -1)
        hand_state_wrist_reshaped = transform_hand_points_to_wrist_frame(hand_state_world_reshaped, wrist_state_world_reshaped)
        hand_state_wrist = hand_state_wrist_reshaped.reshape(-1)
        
        wrist_state_world_reshaped = wrist_state_world.reshape(1, -1)
        wrist_state_cam_reshaped = transform_wrist_to_target_frame(wrist_state_world_reshaped, world2cam)
        wrist_state_cam = wrist_state_cam_reshaped.reshape(-1)
        
        initial_state = np.concatenate([wrist_state_cam, hand_state_wrist])

        wrist_action_world = _navigate_zarr(data_group, fm['wrist_action'], frame_indices)
        hand_action_world = _navigate_zarr(data_group, fm['hand_action'], frame_indices)
        
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
                mano_state_key = fm.get('mano_state', 'state/mano')
                initial_mano_state = _navigate_zarr(data_group, mano_state_key, origin_frame_idx)
                mano_pred = get_absolute_action(initial_mano_state, mano_pred)
                mano_gt = get_absolute_action(initial_mano_state, mano_gt)
                print("Converted relative MANO actions to absolute MANO actions")
            
            shape_key = fm.get('shape_state', 'state/shape')
            mano_shape = _navigate_zarr(data_group, shape_key)[origin_frame_idx:origin_frame_idx+horizon]
        else:
            mano_action_key = fm.get('mano_action', 'action/mano')
            mano_gt = _navigate_zarr(data_group, mano_action_key, frame_indices)
            print("Warning: pred_mano not found in inference results.")
            shape_key = fm.get('shape_state', 'state/shape')
            mano_shape = _navigate_zarr(data_group, shape_key, frame_indices)

    # Try to load instruction
    instr_key = fm.get('instruction', 'instruction')
    try:
        instruction = _navigate_zarr(data_group, instr_key, origin_frame_idx)
    except (KeyError, IndexError):
        instruction = None

    data = {
        'image': image,
        'depth': depth,
        'intrinsic': intrinsic,
        'extrinsic': extrinsic,
        'gt_actions': gt_actions_cam,
        'pred_actions': pred_actions_cam,
        'mano_gt': mano_gt,
        'mano_pred': mano_pred,
        'mano_shape': mano_shape,
        'instruction': instruction,
        'frame_idx': origin_frame_idx,
        'dataset_name': sample_dataset_name,
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


def visualize_one_sample(zarr_data, motion_type, save_path=None,
                         target_width=1920, target_height=1080,
                         depth_scale=1.0, min_depth=0.1, max_depth=1.5,
                         debug=False):
    """Run the Rerun visualization for a single sample loaded by *get_data_from_zarr*."""
    rgb_images = np.array(zarr_data['image'])
    depth_images = np.array(zarr_data['depth']).astype(np.float32) / 1000.0
    intrinsics_flat = zarr_data['intrinsic']
    camera_transforms_raw = zarr_data['extrinsic']
    gt_actions = zarr_data['gt_actions']
    pred_actions = zarr_data['pred_actions']
    ds_name = zarr_data.get('dataset_name') or 'unknown'

    if motion_type == 'mano' and zarr_data['mano_gt'] is not None:
        gt_hands_data = gt_actions_to_hands_data(
            gt_actions[:, :18], motion_type='mano',
            mano_data={'theta': zarr_data['mano_gt'], 'beta': zarr_data['mano_shape']},
        )
        if zarr_data['mano_pred'] is not None:
            pred_hands_data = gt_actions_to_hands_data(
                pred_actions[:, :18], motion_type='mano',
                mano_data={'theta': zarr_data['mano_pred'], 'beta': zarr_data['mano_shape']},
            )
        else:
            pred_hands_data = gt_actions_to_hands_data(pred_actions)
    else:
        gt_hands_data = gt_actions_to_hands_data(gt_actions)
        pred_hands_data = gt_actions_to_hands_data(pred_actions)

    T = len(rgb_images)
    print(f"Loaded {T} frames from sample (dataset={ds_name})")

    # Resize ------------------------------------------------------------------
    orig_h, orig_w = rgb_images[0].shape[:2]
    tw = target_width if target_width is not None else orig_w
    th = target_height if target_height is not None else orig_h

    if tw != orig_w or th != orig_h:
        sx, sy = tw / orig_w, th / orig_h
        rgb_images = np.array([cv2.resize(im, (tw, th), interpolation=cv2.INTER_LINEAR) for im in rgb_images])
        depth_images = np.array([cv2.resize(d, (tw, th), interpolation=cv2.INTER_NEAREST) for d in depth_images])
        intrinsics_flat = intrinsics_flat.copy()
        intrinsics_flat[:, 0] *= sx
        intrinsics_flat[:, 1] *= sy
        intrinsics_flat[:, 2] *= sx
        intrinsics_flat[:, 3] *= sy

    actual_frame_idx = zarr_data['frame_idx']

    # Rerun session ------------------------------------------------------------
    app_name = f"vis_{ds_name}_frame_{actual_frame_idx}"
    rr.init(app_name)

    if save_path:
        os.makedirs(save_path, exist_ok=True) if os.path.isdir(save_path) else None
        if os.path.isdir(save_path):
            rrd_file = os.path.join(save_path, f"{ds_name}_frame_{actual_frame_idx}.rrd")
        else:
            base, ext = os.path.splitext(save_path)
            rrd_file = f"{base}_{ds_name}_frame_{actual_frame_idx}{ext or '.rrd'}"
        rr.save(rrd_file)
        print(f"Saving visualization to: {rrd_file}")
    else:
        rr.spawn(port=9878)

    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
    viz_gt = HandVisualizer(history_len=10, color_scheme='gt')
    viz_pred = HandVisualizer(history_len=10, color_scheme='pred')

    all_errors = {
        side: {'wrist': [], 'Thumb': [], 'Index': [], 'Middle': [], 'Ring': [], 'Little': []}
        for side in ('left', 'right')
    }

    for i in range(T):
        rr.set_time("frame_idx", sequence=i)

        if zarr_data.get('instruction') is not None:
            instr = zarr_data['instruction']
            lines = []
            if isinstance(instr, (np.ndarray, list)):
                for idx, c in enumerate(instr):
                    s = str(c).strip()
                    if s:
                        lines.append(f"[{idx}] {s}")
            if lines:
                rr.log("/world/instruction", rr.TextLog("\n".join(lines)))

        rr.log("/world/origin", rr.Points3D([0, 0, 0], radii=0.01, colors=[255, 255, 255]))
        rr.log("/world/x", rr.Arrows3D(origins=[0, 0, 0], vectors=[0.1, 0, 0], radii=0.005, colors=[255, 0, 0]))
        rr.log("/world/y", rr.Arrows3D(origins=[0, 0, 0], vectors=[0, 0.1, 0], radii=0.005, colors=[0, 255, 0]))
        rr.log("/world/z", rr.Arrows3D(origins=[0, 0, 0], vectors=[0, 0, 0.1], radii=0.005, colors=[0, 0, 255]))

        color_rgb = rgb_images[i]
        depth = depth_images[i]
        world2cam = camera_transforms_raw[i]
        cam_pose_world = np.linalg.inv(world2cam)

        K_flat = intrinsics_flat[i]
        K = np.array([[K_flat[0], 0, K_flat[2]],
                       [0, K_flat[1], K_flat[3]],
                       [0, 0, 1]], dtype=np.float32)

        rr.log("/world/camera_pose", rr.Transform3D(
            translation=cam_pose_world[:3, 3], mat3x3=cam_pose_world[:3, :3]))

        if debug:
            rr.log("/world/camera_pose/debug_axes/x", rr.Arrows3D(origins=[0, 0, 0], vectors=[0.1, 0, 0], radii=0.005, colors=[255, 0, 0]))
            rr.log("/world/camera_pose/debug_axes/y", rr.Arrows3D(origins=[0, 0, 0], vectors=[0, 0.1, 0], radii=0.005, colors=[0, 255, 0]))
            rr.log("/world/camera_pose/debug_axes/z", rr.Arrows3D(origins=[0, 0, 0], vectors=[0, 0, 0.1], radii=0.005, colors=[0, 0, 255]))

        rr.log("/world/camera_pose/camera", rr.Pinhole(
            image_from_camera=K, width=color_rgb.shape[1],
            height=color_rgb.shape[0], camera_xyz=rr.ViewCoordinates.RDF))
        rr.log("/world/camera_pose/camera", rr.Image(color_rgb))

        if depth.shape[:2] != color_rgb.shape[:2]:
            depth_resized = cv2.resize(depth, (color_rgb.shape[1], color_rgb.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
        else:
            depth_resized = depth

        pts, cols = get_colored_point_cloud(
            color_rgb, depth_resized, color_rgb.shape[1], color_rgb.shape[0], K,
            depth_scale=depth_scale, min_depth=min_depth, max_depth=max_depth)
        rr.log("/world/camera_pose/point_cloud", rr.Points3D(pts, colors=cols))

        identity_transform = np.eye(4, dtype=np.float32)

        for hand_name in ['left', 'right']:
            if hand_name in gt_hands_data and i < len(gt_hands_data[hand_name]['wrist']):
                viz_gt.log("/world/camera_pose/hands/gt", "/world/camera_pose/camera",
                           hand_name, gt_hands_data[hand_name], i, identity_transform, K)
                if i == 0 and debug:
                    wp = gt_hands_data[hand_name]['wrist'][i]
                    print(f"\n=== Frame {i} - GT {hand_name} wrist ===\nTranslation: {wp[:3, 3]}")
                if motion_type == 'mano' and 'verts' in gt_hands_data[hand_name]:
                    viz_gt.log_mesh("/world/camera_pose/hands/gt", hand_name, gt_hands_data[hand_name], i)

        for hand_name in ['left', 'right']:
            if hand_name in pred_hands_data and i < len(pred_hands_data[hand_name]['wrist']):
                viz_pred.log("/world/camera_pose/hands/pred", "/world/camera_pose/camera",
                             hand_name, pred_hands_data[hand_name], i, identity_transform, K)
                if i == 0 and debug:
                    wp = pred_hands_data[hand_name]['wrist'][i]
                    print(f"\n=== Frame {i} - Pred {hand_name} wrist ===\nTranslation: {wp[:3, 3]}")
                if motion_type == 'mano' and 'verts' in pred_hands_data[hand_name]:
                    viz_pred.log_mesh("/world/camera_pose/hands/pred", hand_name, pred_hands_data[hand_name], i)

        errors = calculate_position_errors(gt_hands_data, pred_hands_data, i)
        for hn in ('left', 'right'):
            all_errors[hn]['wrist'].append(errors[hn]['wrist'])
            for fn in ('Thumb', 'Index', 'Middle', 'Ring', 'Little'):
                all_errors[hn][fn].append(errors[hn][fn])

        lines = [f"Frame {i} Position Errors (meters)", "=" * 50, ""]
        for hn in ('LEFT', 'RIGHT'):
            hn_key = hn.lower()
            lines.append(f"{hn} Hand:")
            lines.append(f"  Wrist:  {errors[hn_key]['wrist']:.6f} m")
            for fn in ('Thumb', 'Index', 'Middle', 'Ring', 'Little'):
                lines.append(f"  {fn:7s} {errors[hn_key][fn]:.6f} m")
        rr.log("/position_errors", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.TEXT))


def main():
    parser = argparse.ArgumentParser(description="Visualize inference results with Rerun")
    parser.add_argument("--inference_zarr_path", type=str, required=True,
                        help="Path to inference results zarr file.")
    parser.add_argument("--origin_zarr_path", type=str, default=None,
                        help="(Optional) Explicit origin dataset zarr path. "
                             "If omitted, resolved automatically from dataset_name.")
    parser.add_argument("--dataset_names", type=str, nargs='+', default=None,
                        help="Filter samples by dataset name(s). "
                             "E.g. --dataset_names taco oakink2")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to visualize (default 1).")
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="Specific sample index (overrides --num_samples / --dataset_names).")
    parser.add_argument("--target_width", type=int, default=1920)
    parser.add_argument("--target_height", type=int, default=1080)
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=1.5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Directory or file path to save .rrd files.")
    args = parser.parse_args()

    if not os.path.exists(args.inference_zarr_path):
        raise FileNotFoundError(f"Inference zarr not found: {args.inference_zarr_path}")

    # Load model config defaults
    config_defaults = load_config_from_inference_config()
    use_relative_action = config_defaults.get("use_relative_action")
    motion_type = config_defaults.get("motion_type")
    print(f"Config: use_relative_action={use_relative_action}, motion_type={motion_type}")

    # Build dataset name → zarr path registry
    dataset_registry = build_dataset_registry(args.inference_zarr_path)
    if dataset_registry:
        print(f"Dataset registry: {list(dataset_registry.keys())}")

    # Select sample indices
    selected = select_sample_indices(
        args.inference_zarr_path,
        dataset_names=args.dataset_names,
        num_samples=args.num_samples,
        sample_idx=args.sample_idx,
    )
    print(f"Will visualize {len(selected)} sample(s): {selected}")

    for seq, sidx in enumerate(selected):
        print(f"\n{'='*60}\n[{seq+1}/{len(selected)}] Loading sample index {sidx}...")
        zarr_data = get_data_from_zarr(
            args.inference_zarr_path,
            origin_zarr_path=args.origin_zarr_path,
            sample_idx=sidx,
            use_relative_action=use_relative_action,
            motion_type=motion_type,
            dataset_registry=dataset_registry,
        )

        visualize_one_sample(
            zarr_data, motion_type=motion_type,
            save_path=args.save_path,
            target_width=args.target_width,
            target_height=args.target_height,
            depth_scale=args.depth_scale,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            debug=args.debug,
        )

    print(f"\nDone. Visualized {len(selected)} sample(s).")


if __name__ == "__main__":
    main()