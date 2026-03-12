import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# TODO(webdataset): Port this workspace to the current IterableDataset /
# WebDataset pipeline. The implementation below still assumes the legacy
# indexed zarr dataset workflow.

import numpy as np

# Fix chumpy compatibility with numpy 2.x
# chumpy tries to import deprecated numpy types that were removed in numpy 2.0
# We need to create aliases for these types before chumpy is imported
if not hasattr(np, 'int'):
    try:
        np.int = np.int64
    except AttributeError:
        np.int = int
if not hasattr(np, 'float'):
    try:
        np.float = np.float64
    except AttributeError:
        np.float = float
if not hasattr(np, 'bool'):
    try:
        np.bool = np.bool_
    except AttributeError:
        np.bool = bool
if not hasattr(np, 'complex'):
    try:
        np.complex = np.complex128
    except AttributeError:
        np.complex = complex
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'unicode'):
    np.unicode = str
if not hasattr(np, 'str'):
    np.str = str

import hydra
import torch
from omegaconf import OmegaConf
import random
import cv2

from src.utils.mano_vis import vis_hand_plot, vis_hand_plot_comparison
from src.utils.mano_utils import rot6d_to_rotmat, sample_to_manovis
from visualize import HandVisualizer, sample_for_vis
from src.utils.geometry import transform_wrist_to_target_frame, transform_hand_points_to_target_frame
from src.utils.mano_utils import invert_extrinsics


class DatasetGTVisualizer:
    """
    Workspace for visualizing dataset ground truth.
    
    Only visualizes ground truth data without loading model or tokenizer.
    Supports MANO and keypoint visualization modes.
    """
    
    def __init__(self, cfg):
        """
        Initialize the visualization workspace.
        
        Args:
            cfg: Configuration object from OmegaConf
        """
        self.cfg = cfg
        
        # Configure dataset
        self.dataset = hydra.utils.instantiate(cfg.dataset)
        
        # Set dataset to return raw samples
        self.dataset.set_return_raw_sample(True)
    
    def resize_image_and_adjust_intrinsics(self, background_img, intrinsic_4d, target_width, target_height, original_size=384):
        """Resize image and adjust camera intrinsics accordingly"""
        if target_width == original_size and target_height == original_size:
            # No resizing needed
            intrinsic = torch.tensor([
                [intrinsic_4d[0], 0, intrinsic_4d[2]],
                [0, intrinsic_4d[1], intrinsic_4d[3]],
                [0, 0, 1]
            ])
            return background_img, intrinsic
        
        is_sequence = background_img.ndim == 4
        
        if is_sequence:
            num_frames, orig_h, orig_w = background_img.shape[:3]
            print(f"Original background image size: {orig_w}x{orig_h} ({num_frames} frames)")
            
            target_size = (target_width, target_height)
            resized_frames = []
            
            for frame_idx in range(num_frames):
                frame_bgr = cv2.cvtColor(background_img[frame_idx], cv2.COLOR_RGB2BGR)
                frame_resized = cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_CUBIC)
                resized_frames.append(frame_resized)
            
            background_img = np.stack(resized_frames, axis=0)
            print(f"Resized background image to: {background_img.shape[2]}x{background_img.shape[1]} ({num_frames} frames)")
        else:
            background_img = cv2.cvtColor(background_img, cv2.COLOR_RGB2BGR)
            original_size_img = background_img.shape[:2]
            target_size = (target_width, target_height)
            
            print(f"Original background image size: {original_size_img[1]}x{original_size_img[0]}")
            background_img = cv2.resize(background_img, target_size, interpolation=cv2.INTER_CUBIC)
            print(f"Resized background image to: {background_img.shape[1]}x{background_img.shape[0]}")
        
        # Scale camera intrinsics
        scale_factor_x = target_width / original_size
        scale_factor_y = target_height / original_size
        
        fx = intrinsic_4d[0] * scale_factor_x
        fy = intrinsic_4d[1] * scale_factor_y
        cx = intrinsic_4d[2] * scale_factor_x
        cy = intrinsic_4d[3] * scale_factor_y
        
        intrinsic = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        return background_img, intrinsic
    
    def convert_bgr_to_rgb(self, background_img, needs_conversion=True):
        """Convert BGR to RGB for visualization"""
        if not needs_conversion:
            return background_img
        
        if background_img.ndim == 4:
            # Multi-frame: convert each frame from BGR to RGB
            background_img = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in background_img], axis=0)
        else:
            # Single frame: convert directly
            background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
        
        return background_img
    
    def convert_action_to_camera_frame(self, action_gt, extrinsic_w2c):
        """Convert ground truth action to camera frame based on motion type"""
        motion_type = self.dataset.vla_dataset.motion_type
        
        if motion_type == 'fingertips':
            # Convert GT action to camera frame
            action_in_camera = action_gt.clone()
            action_in_camera[:, :18] = transform_wrist_to_target_frame(
                action_in_camera[:, :18], extrinsic_w2c)
            action_in_camera[:, 18:] = transform_hand_points_to_target_frame(
                action_in_camera[:, 18:], extrinsic_w2c)
            
        elif motion_type == 'mano':
            action_in_camera = action_gt.clone()
            action_in_camera[:, :18] = transform_wrist_to_target_frame(
                action_in_camera[:, :18], extrinsic_w2c)
        else:
            raise ValueError(f"Unsupported motion_type: {motion_type}")
        
        return action_in_camera
    
    def visualize_mesh_gt(self, raw_sample, background_img, intrinsic, extrinsic_c2w, sample_idx):
        """Visualize ground truth mesh"""
        mano_shape = raw_sample['action_shape']
        action_wrist = raw_sample['action'][:, :18]
        action_hand = raw_sample['action'][:, 18:]
        
        # Convert action to numpy for visualization
        action_gt = raw_sample['action'].cpu().numpy() if isinstance(raw_sample['action'], torch.Tensor) else raw_sample['action']
        
        mano_data = sample_to_manovis(action_gt, action_wrist, action_hand, mano_shape, extrinsic_c2w)
        
        output_dir_base = os.path.join(self.cfg.testing.output_dir, "dataset_gt_visualization")
        os.makedirs(output_dir_base, exist_ok=True)
        output_dir = os.path.join(output_dir_base, f"{sample_idx}_mesh_gt")
        
        # Convert BGR to RGB if needed
        needs_conversion = (self.cfg.testing.target_width != 384 or self.cfg.testing.target_height != 384)
        background_img = self.convert_bgr_to_rgb(background_img, needs_conversion)
        
        # Visualize ground truth only (use GT for both predicted and GT in comparison function)
        vis_hand_plot_comparison(
            mano_data['ground_truth']['rot'], mano_data['ground_truth']['trans'], 
            mano_data['ground_truth']['theta'],
            mano_data['ground_truth']['rot'], mano_data['ground_truth']['trans'], 
            mano_data['ground_truth']['theta'], 
            mano_data['beta'], mano_data['sides'], 
            background_img, intrinsic, extrinsic_c2w, output_dir, fps=30)
        
        print(f"Ground truth mesh visualization saved to {output_dir}")
    
    def visualize_keypoint_gt(self, raw_sample, action_gt_in_camera, background_img, 
                               intrinsic, sample_idx):
        """Visualize ground truth keypoints"""
        mano_root = self.cfg.testing.mano_root_dir
        print(f"Initializing hand visualizer with MANO root: {mano_root}")
        visualizer = HandVisualizer(mano_root=mano_root)
        
        # Prepare data for visualization - use GT data for both pred and GT
        # sample_for_vis expects (pred_action, raw_sample), but we only have GT
        # So we pass GT as both pred and GT
        keypoint_sequence, wrist_sequence, gt_keypoint_sequence, gt_wrist_sequence, \
            intrinsic_matrix, extrinsic_matrices, presence = sample_for_vis(
                action_gt_in_camera, raw_sample)
        
        # Use GT data for visualization
        keypoint_sequence = gt_keypoint_sequence.to(visualizer.device).float()
        wrist_sequence = gt_wrist_sequence.to(visualizer.device).float()
        
        num_frames = keypoint_sequence.shape[0]
        print(f"Detected {num_frames} frames in the ground truth sequence")
        
        # Extract camera intrinsics
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        
        # Process extrinsic matrices
        extrinsic_sequence = None
        if extrinsic_matrices is not None:
            if isinstance(extrinsic_matrices, np.ndarray):
                extrinsic_matrices = torch.from_numpy(extrinsic_matrices).float()
            
            if extrinsic_matrices.shape[0] != num_frames:
                print(f"Warning: Extrinsic sequence frames ({extrinsic_matrices.shape[0]}) does not match GT frames ({num_frames})")
                print(f"Using identity transformation instead")
            else:
                extrinsic_sequence = extrinsic_matrices.to(visualizer.device).float()
                print(f"Loaded camera extrinsics with {extrinsic_sequence.shape[0]} frames")
        else:
            print("No extrinsic data available, using identity transformation")
        
        # Create output directory
        output_dir_base = os.path.join(self.cfg.testing.output_dir, "dataset_gt_visualization")
        os.makedirs(output_dir_base, exist_ok=True)
        output_path_2d = os.path.join(output_dir_base, f"{sample_idx}_keypoint_gt.mp4")
        
        # Generate 2D projection video - only show GT
        print(f"Generating 2D projection video for ground truth...")
        visualizer.generate_2d_video(
            background_img, wrist_sequence, output_path_2d, self.cfg.testing.fps,
            fx=fx, fy=fy, cx=cx, cy=cy,
            extrinsic_sequence=None,
            keypoint_sequence=keypoint_sequence,
            show_mesh=self.cfg.testing.show_mesh, mesh_alpha=self.cfg.testing.mesh_alpha,
            gt_keypoint_sequence=None, gt_wrist_sequence=None,  # Don't show comparison
            presence=presence, mode='keypoint')
        
        print(f"Ground truth keypoint visualization saved to {output_path_2d}")
        return output_path_2d
    
    def filter_valid_samples(self, dataset_size, min_required_frames):
        """Filter samples that have enough frames for visualization"""
        valid_indices = []
        vla_dataset = self.dataset.vla_dataset
        
        for idx in range(dataset_size):
            cumulative_len = 0
            for sampler_idx, sampler_len in enumerate(vla_dataset.sampler_lens):
                if idx < cumulative_len + sampler_len:
                    local_idx = idx - cumulative_len
                    sampler = vla_dataset.samplers[sampler_idx]
                    replay_buffer = sampler.replay_buffer
                    
                    buffer_start_idx, buffer_end_idx, _, _ = sampler.indices[local_idx]
                    episode_ends = replay_buffer.episode_ends
                    episode_idx = np.searchsorted(episode_ends, buffer_end_idx)
                    episode_end = episode_ends[episode_idx]
                    
                    available_frames = episode_end - buffer_start_idx
                    if available_frames >= min_required_frames:
                        valid_indices.append(idx)
                    break
                cumulative_len += sampler_len
        
        return valid_indices
    
    def process_single_sample(self, sample_idx, i):
        """Process a single sample: visualize ground truth"""
        print(f"\nProcessing sample {i+1} (index {sample_idx})")
        
        # Get raw sample
        raw_sample = self.dataset[sample_idx]
        
        # Print sample information
        print("Raw sample keys and data types:")
        for key, value in raw_sample.items():
            if hasattr(value, 'dtype'):
                if hasattr(value, 'shape'):
                    print(f"  {key}: {type(value)} - shape: {value.shape}")
                else:
                    print(f"  {key}: {type(value)} - dtype: {value.dtype}")
            else:
                print(f"  {key}: {type(value)} - value: {value}")
        
        if 'dataset_source' in raw_sample:
            dataset_path = raw_sample['dataset_source']
            dataset_name = os.path.basename(dataset_path).replace('.zarr', '')
            dataset_idx = raw_sample['dataset_idx'].item() if hasattr(raw_sample['dataset_idx'], 'item') else raw_sample['dataset_idx']
            print(f"Dataset source: {dataset_name} (index: {dataset_idx})")
            print(f"Full path: {dataset_path}")
        
        # Prepare image and camera parameters
        background_img = np.array(raw_sample['image'])
        extrinsic_w2c = raw_sample['extrinsic']
        extrinsic_c2w = invert_extrinsics(extrinsic_w2c)
        intrinsic_4d = raw_sample['intrinsic'][0]
        
        # Resize image and adjust intrinsics if needed
        background_img, intrinsic = self.resize_image_and_adjust_intrinsics(
            background_img, intrinsic_4d, 
            self.cfg.testing.target_width, self.cfg.testing.target_height)
        
        # Get ground truth action
        action_gt = raw_sample['action']
        if isinstance(action_gt, torch.Tensor):
            action_gt = action_gt.clone()
        else:
            action_gt = torch.from_numpy(action_gt).clone()
        
        # Convert action to camera frame
        action_gt_in_camera = self.convert_action_to_camera_frame(
            action_gt, extrinsic_w2c)
        
        # Visualize based on type
        if self.cfg.testing.visualize_type == 'mesh':
            self.visualize_mesh_gt(raw_sample, background_img, 
                                   intrinsic, extrinsic_c2w, i)
        elif self.cfg.testing.visualize_type == 'keypoint':
            output_path_2d = self.visualize_keypoint_gt(
                raw_sample, action_gt_in_camera, background_img, intrinsic, i)
            print(f"Sample {i+1} (index {sample_idx}) completed:")
            print(f"  Ground truth keypoint video: {output_path_2d}")
        else:
            raise ValueError(f"Unsupported visualize_type: {self.cfg.testing.visualize_type}. "
                           f"Must be 'mesh' or 'keypoint'")
    
    def run(self):
        """
        Main execution method to run the visualization workflow.
        
        Handles dataset preparation, sample filtering, and processing all samples.
        """
        # Get dataset information
        dataset_size = len(self.dataset)
        print(f"Total dataset size: {dataset_size}")
        
        # Display dataset composition
        print("\nDataset composition:")
        for i, zarr_path in enumerate(self.cfg.vla_dataset_paths):
            dataset_name = os.path.basename(zarr_path).replace('.zarr', '')
            if hasattr(self.dataset.vla_dataset, 'sampler_lens') and i < len(self.dataset.vla_dataset.sampler_lens):
                component_size = self.dataset.vla_dataset.sampler_lens[i]
                print(f"  {dataset_name}: {component_size} samples")
            else:
                print(f"  {dataset_name}: size unknown")
        
        # Filter valid samples
        num_samples = self.cfg.testing.num_samples
        print(f"\nSampling {num_samples} random samples from dataset...")
        
        min_required_frames = self.dataset.vla_dataset.history + self.dataset.vla_dataset.horizon
        valid_indices = self.filter_valid_samples(dataset_size, min_required_frames)
        
        print(f"Found {len(valid_indices)} valid samples (with at least {min_required_frames} frames available)")
        print(f"Filtered out {dataset_size - len(valid_indices)} samples without enough frames")
        
        # Sample from valid indices
        sample_indices = random.sample(valid_indices, min(num_samples, len(valid_indices)))
        
        # Process all samples
        print(f"\nProcessing {len(sample_indices)} samples...")
        for i, sample_idx in enumerate(sample_indices):
            self.process_single_sample(sample_idx, i)


OmegaConf.register_new_resolver("eval", eval, replace=True)

if __name__ == "__main__":
    # Load config
    config_path = "/home/zengfanlian/EgoVLA/src/config/experiment/test_legendvla.yaml"
    cfg = OmegaConf.load(config_path)
    
    # Initialize visualizer and run
    visualizer = DatasetGTVisualizer(cfg)
    visualizer.run()
