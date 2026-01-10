import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# Fix chumpy numpy imports before importing other modules
import numpy as np
# Add the missing numpy types that chumpy expects
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'float'):
    np.float = np.float_
if not hasattr(np, 'complex'):
    np.complex = np.complex_
if not hasattr(np, 'object'):
    np.object = np.object_
if not hasattr(np, 'unicode'):
    np.unicode = np.unicode_
if not hasattr(np, 'str'):
    np.str = np.str_

import os
import hydra
import torch
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from mpl_toolkits.mplot3d import Axes3D
import zarr
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import tqdm
import numpy as np
import pickle
import argparse
from torchvision.transforms import v2
from PIL import Image

from src.dataset.paligemma_processing import PaliGemmaVLAProcessor, PaliGemmaProcessor
from transformers import AutoTokenizer
from src.model.action.fast_tokenizer import UniversalActionProcessor
from src.model.action.vq_tokenizer import VQActionProcessor
from src.utils.mano_vis import mano_forward, vis_hand_plot, vis_hand_plot_comparison
from src.utils.mano_utils import rot6d_to_rotmat, sample_to_manovis
from visualize import HandVisualizer, sample_for_vis
from src.dataset.legendvla_dataset import get_absolute_action, transform_wrist_to_target_frame
from src.utils.geometry import transform_hand_points_from_wrist_to_camera_frame, transform_hand_points_to_wrist_frame, transform_hand_points_to_target_frame
import cv2
from src.utils.mano_utils import invert_extrinsics
from src.utils.monitor import log_execution_time


class TestLegendVLAWorkspace:
    """
    Workspace for testing LegendVLA model.
    
    Handles model loading, dataset preparation, inference, and visualization.
    Similar structure to LegendVLA class for consistency.
    """
    
    def __init__(self, cfg):
        """
        Initialize the test workspace.
        
        Args:
            cfg: Configuration object from OmegaConf
        """
        self.cfg = cfg
        
        # Configure device
        if cfg.testing.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(cfg.testing.device)
        
        # Load model and checkpoint
        self.model = self._load_model_and_checkpoint()
        
        # Configure dataset
        self.dataset = hydra.utils.instantiate(cfg.dataset)
        
        # Load processors
        self._load_processors()
        
        # Load normalizer
        print("Loading normalizer...")
        self.normalizer = self._load_normalizer()
        
        # Initialize tokenizers (will be set in _load_processors)
        self.fast_tokenizer = None
        self.vq_tokenizer = None
    
    def _load_model_and_checkpoint(self):
        """Load model and checkpoint"""
        model = hydra.utils.instantiate(self.cfg.policy)
        
        if self.cfg.testing.checkpoint_path is not None:
            print(f"Loading checkpoint from {self.cfg.testing.checkpoint_path}")
            model.load_state_dict(
                torch.load(self.cfg.testing.checkpoint_path, map_location='cpu', weights_only=False)['module']
            )
            print("Checkpoint loaded successfully!")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _load_processors(self):
        """Load tokenizers and processors"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.policy.cfg.pretrained_model_path, padding_side="right"
        )
        
        self.fast_tokenizer = {
            "states": UniversalActionProcessor.from_pretrained(
                os.path.join(self.cfg.processor.fast_tokenizer_path, "states")
            ),
            "actions": UniversalActionProcessor.from_pretrained(
                os.path.join(self.cfg.processor.fast_tokenizer_path, "actions")
            )
        }
        
        self.vq_tokenizer = {
            "states": VQActionProcessor.from_pretrained(
                self.cfg.processor.hand_states_tokenizer_path
            ),
            "actions": VQActionProcessor.from_pretrained(
                self.cfg.processor.hand_actions_tokenizer_path
            )
        }
        
        vla_processor = PaliGemmaVLAProcessor(
            tokenizer,
            motion_tokenizer=self.vq_tokenizer if self.cfg.testing.tokenizer_type == 'vq' else self.fast_tokenizer,
            num_image_tokens=self.cfg.policy.vision_tower.config.num_image_tokens,
            max_seq_len=self.cfg.policy.cfg.max_vlm_tokens,
            ignore_index=self.cfg.ignore_index,
            image_size=self.cfg.policy.vision_tower.config.image_size,
            tokenizer_padding=self.cfg.tokenizer_padding,
        )
        
        vlm_processor = PaliGemmaProcessor(
            tokenizer,
            num_image_tokens=self.cfg.policy.vision_tower.config.num_image_tokens,
            max_seq_len=self.cfg.policy.cfg.max_vlm_tokens,
            ignore_index=self.cfg.ignore_index,
            image_size=self.cfg.policy.vision_tower.config.image_size,
            tokenizer_padding=self.cfg.tokenizer_padding,
        )
        
        self.dataset.vla_dataset.set_preprocessor(vla_processor)
        if self.dataset.vlm_dataset is not None:
            self.dataset.vlm_dataset.set_preprocessor(vlm_processor)
    
    def _load_normalizer(self):
        """Load or compute normalizer"""
        normalizer_path = self.cfg.testing.normalizer_path
        
        if os.path.exists(normalizer_path):
            normalizer = pickle.load(open(normalizer_path, 'rb'))
            print(f"Normalizer loaded from {normalizer_path}")
        else:
            print(f"Normalizer file not found at {normalizer_path}")
            print("Computing normalizer from dataset...")
            normalizer = self.dataset.vla_dataset.get_normalizer()
            print("Normalizer computed successfully")
            
            os.makedirs(os.path.dirname(normalizer_path), exist_ok=True)
            with open(normalizer_path, 'wb') as f:
                pickle.dump(normalizer, f)
            print(f"Normalizer saved to {normalizer_path}")
        
        self.dataset.vla_dataset.set_normalizer(normalizer)
        return normalizer
    
    def preprocess_batch_for_inference(self, batch, dtype=torch.float32, sample_fm_time=False):
        """Preprocess batch for inference, similar to training script"""
        # Extract data from batch and move to device
        pixel_values = batch["pixel_values"].to(self.device)
        actions = batch["actions"].to(self.device)
        actions_valid_mask = batch["actions_valid_mask"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        answer_start_idx = batch["answer_start_idx"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Get unwrapped model for mask building
        if hasattr(self.model, 'module'):
            unwrapped_model = self.model.module
        else:
            unwrapped_model = self.model
        
        # Build causal mask and position ids
        causal_mask, vlm_position_ids, action_position_ids = (
            unwrapped_model.build_causal_mask_and_position_ids(   
                attention_mask, answer_start_idx, dtype
            )
        )
        max_vlm_tokens = input_ids.shape[-1]
        # Split mask for inference
        vlm_mask, action_mask = (
            unwrapped_model.split_full_mask_into_submasks(causal_mask, max_vlm_tokens)
        )

        inputs = {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values.to(dtype),
            "vlm_position_ids": vlm_position_ids,
            "action_position_ids": action_position_ids,
            "vlm_mask": vlm_mask,
            "action_mask": action_mask,
            "causal_mask": causal_mask,
            "actions": actions.to(dtype),
            "actions_valid_mask": actions_valid_mask,
        }
        
        # Sample flow matching timesteps
        if sample_fm_time:
            # Simple uniform sampling for testing
            bsz = len(input_ids)
            t = torch.rand(bsz, device=input_ids.device, dtype=dtype)
            inputs["t"] = t

        return inputs
    
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
    
    def convert_action_to_camera_frame(self, raw_sample, action_pred_unnormalized, extrinsic_w2c):
        """Convert predicted action to camera frame based on motion type"""
        motion_type = self.dataset.vla_dataset.motion_type
        
        if motion_type == 'fingertips':
            # Step 1: Transform hand points to wrist frame (using world frame wrist)
            state_world = raw_sample['state'].clone()
            state_world[:, 18:] = transform_hand_points_to_wrist_frame(
                state_world[:, 18:], state_world[:, :18])
            
            # Step 2: Transform wrist to camera frame
            state_in_camera = state_world.clone()
            state_in_camera[:, :18] = transform_wrist_to_target_frame(
                state_in_camera[:, :18], extrinsic_w2c[0])
            
            # Get absolute action
            action_pred = get_absolute_action(
                state_in_camera.squeeze(0), 
                action_pred_unnormalized["actions"].squeeze(0))
            
            # Convert hand points from wrist frame to camera frame for visualization
            hand_points_in_camera = transform_hand_points_from_wrist_to_camera_frame(
                action_pred[:, 18:], action_pred[:, :18])
            action_pred_in_camera_0 = torch.cat([action_pred[:, :18], hand_points_in_camera], dim=1)

            extrinsic_c2w_first = invert_extrinsics(extrinsic_w2c[0])  # [1, 4, 4]
            
            # Convert wrist from first frame camera to each frame's camera
            # First convert to world frame, then to each frame's camera frame
            wrist_in_world = transform_wrist_to_target_frame(action_pred_in_camera_0[:, :18], extrinsic_c2w_first)  # [T, 18] in world frame
            wrist_in_each_frame_cam = transform_wrist_to_target_frame(wrist_in_world, extrinsic_w2c)  # [T, 18] in each frame's camera frame
            hand_in_world = transform_hand_points_to_target_frame(action_pred_in_camera_0[:, 18:], extrinsic_c2w_first)
            hand_in_each_frame_cam = transform_hand_points_to_target_frame(hand_in_world, extrinsic_w2c)
            action_pred_in_camera = torch.cat([wrist_in_each_frame_cam, hand_in_each_frame_cam], dim=1)
            # Convert GT action to camera frame
            action_in_camera = raw_sample['action'].clone()
            action_in_camera[:, :18] = transform_wrist_to_target_frame(
                action_in_camera[:, :18], extrinsic_w2c)
            action_in_camera[:, 18:] = transform_hand_points_to_target_frame(
                action_in_camera[:, 18:], extrinsic_w2c)
            
        elif motion_type == 'mano':
            state_in_camera = raw_sample['state'].clone()
            state_in_camera[:, :18] = transform_wrist_to_target_frame(
                state_in_camera[:, :18], extrinsic_w2c[0])
            
            # Get absolute action (based on first frame extrinsic)
            action_pred = get_absolute_action(
                state_in_camera.squeeze(0), 
                action_pred_unnormalized["actions"].squeeze(0))
            
            # Convert action_pred from first frame extrinsic to each frame's extrinsic
            # action_pred[:, :18] is in first frame camera frame, need to convert to each frame's camera frame
            extrinsic_c2w_first = invert_extrinsics(extrinsic_w2c[0])  # [1, 4, 4]
            
            # Convert wrist from first frame camera to each frame's camera
            # First convert to world frame, then to each frame's camera frame
            wrist_in_world = transform_wrist_to_target_frame(action_pred[:, :18], extrinsic_c2w_first)  # [T, 18] in world frame
            wrist_in_each_frame_cam = transform_wrist_to_target_frame(wrist_in_world, extrinsic_w2c)  # [T, 18] in each frame's camera frame
            
            # For MANO mode, only wrist needs conversion, hand params stay the same
            action_pred_in_camera = torch.cat([wrist_in_each_frame_cam, action_pred[:, 18:]], dim=1)
            
            action_in_camera = raw_sample['action'].clone()
            action_in_camera[:, :18] = transform_wrist_to_target_frame(
                action_in_camera[:, :18], extrinsic_w2c)
        else:
            raise ValueError(f"Unsupported motion_type: {motion_type}")
        
        return action_pred, action_pred_in_camera, action_in_camera, state_in_camera
    
    def run_model_inference(self, batch):
        """Run model inference and unnormalize predictions"""
        inputs = self.preprocess_batch_for_inference(batch, sample_fm_time=True)
        
        with torch.no_grad():
            action_pred_normalized = self.model("infer_action", inputs)
            loss = self.model("train_flow", inputs)
        
        print("loss:", loss)
        action_pred_cpu = action_pred_normalized.cpu()
        action_pred_dict = {"actions": action_pred_cpu}
        action_pred_unnormalized = self.normalizer.unnormalize(action_pred_dict)
        
        return action_pred_unnormalized, inputs
    
    def compute_tokenizer_errors(self, decoded_action_gt, gt_action):
        """Compute and print tokenizer reconstruction errors"""
        if isinstance(gt_action, torch.Tensor):
            gt_action = gt_action.cpu().numpy()
        else:
            gt_action = np.array(gt_action)
        
        # Left wrist translation error (0:3)
        left_wrist_trans_error = np.sqrt(np.mean((decoded_action_gt[:, 0:3] - gt_action[:, 0:3]) ** 2, axis=0))
        print(f"Left wrist translation error (X, Y, Z): {left_wrist_trans_error}, Mean: {np.mean(left_wrist_trans_error):.6f}")
        
        # Left wrist rotation error (6:12)
        left_wrist_rot_6d_decoded = torch.from_numpy(decoded_action_gt[:, 6:12]).float()
        left_wrist_rot_6d_gt = torch.from_numpy(gt_action[:, 6:12]).float()
        left_wrist_rot_mat_decoded = rot6d_to_rotmat(left_wrist_rot_6d_decoded).numpy()
        left_wrist_rot_mat_gt = rot6d_to_rotmat(left_wrist_rot_6d_gt).numpy()
        R_rel_left = np.matmul(left_wrist_rot_mat_decoded.transpose(0, 2, 1), left_wrist_rot_mat_gt)
        trace_left = np.trace(R_rel_left, axis1=1, axis2=2)
        trace_left = np.clip(trace_left, -1, 3)
        left_wrist_rot_error_deg = np.arccos((trace_left - 1) / 2) * 180 / np.pi
        print(f"Left wrist rotation error (degrees): Mean: {np.mean(left_wrist_rot_error_deg):.6f}, Std: {np.std(left_wrist_rot_error_deg):.6f}")
        
        # Right wrist translation error (3:6)
        right_wrist_trans_error = np.sqrt(np.mean((decoded_action_gt[:, 3:6] - gt_action[:, 3:6]) ** 2, axis=0))
        print(f"Right wrist translation error (X, Y, Z): {right_wrist_trans_error}, Mean: {np.mean(right_wrist_trans_error):.6f}")
        
        # Right wrist rotation error (12:18)
        right_wrist_rot_6d_decoded = torch.from_numpy(decoded_action_gt[:, 12:18]).float()
        right_wrist_rot_6d_gt = torch.from_numpy(gt_action[:, 12:18]).float()
        right_wrist_rot_mat_decoded = rot6d_to_rotmat(right_wrist_rot_6d_decoded).numpy()
        right_wrist_rot_mat_gt = rot6d_to_rotmat(right_wrist_rot_6d_gt).numpy()
        R_rel_right = np.matmul(right_wrist_rot_mat_decoded.transpose(0, 2, 1), right_wrist_rot_mat_gt)
        trace_right = np.trace(R_rel_right, axis1=1, axis2=2)
        trace_right = np.clip(trace_right, -1, 3)
        right_wrist_rot_error_deg = np.arccos((trace_right - 1) / 2) * 180 / np.pi
        print(f"Right wrist rotation error (degrees): Mean: {np.mean(right_wrist_rot_error_deg):.6f}, Std: {np.std(right_wrist_rot_error_deg):.6f}")
        
        # Left mano error (18:33)
        left_mano_error = np.sqrt(np.mean((decoded_action_gt[:, 18:33] - gt_action[:, 18:33]) ** 2, axis=0))
        print(f"Left mano error (15 dims): Mean: {np.mean(left_mano_error):.6f}, Std: {np.std(left_mano_error):.6f}")
        
        # Right mano error (33:48)
        right_mano_error = np.sqrt(np.mean((decoded_action_gt[:, 33:48] - gt_action[:, 33:48]) ** 2, axis=0))
        print(f"Right mano error (15 dims): Mean: {np.mean(right_mano_error):.6f}, Std: {np.std(right_mano_error):.6f}")
    
    def plot_action_comparison(self, pred_action, gt_action, output_path, sample_idx=0):
        """
        Plot 2D comparison between predicted action and ground truth action for keypoint mode
        Directly compares xyz coordinates of wrist and keypoints
        
        Args:
            pred_action: torch.Tensor or np.ndarray, shape: [H, 48] - predicted action
                         Format: [left_wrist(9), right_wrist(9), left_keypoints(15), right_keypoints(15)]
            gt_action: torch.Tensor or np.ndarray, shape: [H, 48] - ground truth action
            output_path: str - path to save the plot
            sample_idx: int - sample index for title
        """
        # Convert to numpy if needed
        if isinstance(pred_action, torch.Tensor):
            pred_action = pred_action.cpu().numpy()
        if isinstance(gt_action, torch.Tensor):
            gt_action = gt_action.cpu().numpy()
        
        horizon, action_dim = pred_action.shape
        time_steps = np.arange(horizon)
        
        # Use a clean, professional style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure with better layout - 3x2 grid
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    
        # Color schemes
        gt_color = '#2E86AB'  # Blue for GT
        pred_color = '#A23B72'  # Purple/Magenta for Prediction
        coords = ['X', 'Y', 'Z']
        colors_coord = ['#E63946', '#F77F00', '#06A77D']  # Red, Orange, Green
        
        # Finger names for keypoints
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        
        # === 1. Wrist Translation - Split into Left and Right ===
        # Left hand translation
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('Left Wrist Translation', fontsize=13, fontweight='bold', pad=10)
        for i in range(3):
            ax1.plot(time_steps, gt_action[:, i], '--', color=colors_coord[i], 
                    alpha=0.6, linewidth=2.5, label=f'GT {coords[i]}')
            ax1.plot(time_steps, pred_action[:, i], '-', color=colors_coord[i], 
                    alpha=0.95, linewidth=2, label=f'Pred {coords[i]}')
        ax1.set_xlabel('Time Step', fontsize=11)
        ax1.set_ylabel('Translation (m)', fontsize=11)
        ax1.legend(loc='best', ncol=2, fontsize=9, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Right hand translation
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title('Right Wrist Translation', fontsize=13, fontweight='bold', pad=10)
        for i in range(3, 6):
            coord_idx = i - 3
            ax2.plot(time_steps, gt_action[:, i], '--', color=colors_coord[coord_idx], 
                    alpha=0.6, linewidth=2.5, label=f'GT {coords[coord_idx]}')
            ax2.plot(time_steps, pred_action[:, i], '-', color=colors_coord[coord_idx], 
                    alpha=0.95, linewidth=2, label=f'Pred {coords[coord_idx]}')
        ax2.set_xlabel('Time Step', fontsize=11)
        ax2.set_ylabel('Translation (m)', fontsize=11)
        ax2.legend(loc='best', ncol=2, fontsize=9, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # === 2. Wrist Rotation - Show rotation error using geodesic distance ===
        ax3 = fig.add_subplot(gs[1, :])
        ax3.set_title('Wrist Rotation Error (Geodesic Distance in Degrees)', 
                     fontsize=13, fontweight='bold', pad=10)
        
        # Convert 6D rotation to rotation matrices
        gt_left_rot_6d = torch.from_numpy(gt_action[:, 6:12]).float()
        pred_left_rot_6d = torch.from_numpy(pred_action[:, 6:12]).float()
        gt_right_rot_6d = torch.from_numpy(gt_action[:, 12:18]).float()
        pred_right_rot_6d = torch.from_numpy(pred_action[:, 12:18]).float()
        
        gt_left_rot_mat = rot6d_to_rotmat(gt_left_rot_6d).numpy()  # [H, 3, 3]
        pred_left_rot_mat = rot6d_to_rotmat(pred_left_rot_6d).numpy()
        gt_right_rot_mat = rot6d_to_rotmat(gt_right_rot_6d).numpy()
        pred_right_rot_mat = rot6d_to_rotmat(pred_right_rot_6d).numpy()
        
        # Compute geodesic distance (rotation error in degrees)
        def rotation_error_degrees(R_pred, R_gt):
            """
            Compute geodesic distance between rotation matrices in degrees.
            Formula: theta = arccos((trace(R_pred^T @ R_gt) - 1) / 2)
            """
            # Compute relative rotation: R_rel = R_pred^T @ R_gt
            # This gives the rotation needed to go from pred to gt
            R_rel = np.matmul(R_pred.transpose(0, 2, 1), R_gt)
            
            # The angle of rotation is: theta = arccos((trace(R_rel) - 1) / 2)
            trace = np.trace(R_rel, axis1=1, axis2=2)
            
            # Clamp to avoid numerical issues with arccos (valid range: [-1, 1])
            # trace should be in range [−1, 3] for valid rotation matrices
            cos_theta = (trace - 1.0) / 2.0
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            
            theta_rad = np.arccos(cos_theta)
            theta_deg = np.degrees(theta_rad)
            
            return theta_deg
        
        left_rot_error = rotation_error_degrees(pred_left_rot_mat, gt_left_rot_mat)
        right_rot_error = rotation_error_degrees(pred_right_rot_mat, gt_right_rot_mat)
        
        # Compute wrist translation errors
        left_wrist_trans_error = np.sqrt(np.sum((pred_action[:, 0:3] - gt_action[:, 0:3]) ** 2, axis=1))  # [H]
        right_wrist_trans_error = np.sqrt(np.sum((pred_action[:, 3:6] - gt_action[:, 3:6]) ** 2, axis=1))  # [H]
        
        # Print statistics
        print(f"Left wrist translation error - Mean: {np.mean(left_wrist_trans_error)*1000:.2f}mm, "
              f"Median: {np.median(left_wrist_trans_error)*1000:.2f}mm, "
              f"Max: {np.max(left_wrist_trans_error)*1000:.2f}mm")
        print(f"Right wrist translation error - Mean: {np.mean(right_wrist_trans_error)*1000:.2f}mm, "
              f"Median: {np.median(right_wrist_trans_error)*1000:.2f}mm, "
              f"Max: {np.max(right_wrist_trans_error)*1000:.2f}mm")
        print(f"Left rotation error - Mean: {np.mean(left_rot_error):.2f}°, "
              f"Median: {np.median(left_rot_error):.2f}°, "
              f"Max: {np.max(left_rot_error):.2f}°")
        print(f"Right rotation error - Mean: {np.mean(right_rot_error):.2f}°, "
              f"Median: {np.median(right_rot_error):.2f}°, "
              f"Max: {np.max(right_rot_error):.2f}°")
        
        # Plot rotation errors
        ax3.plot(time_steps, left_rot_error, '-', color='#0077B6', 
                linewidth=2.5, alpha=0.9, 
                label=f'Left Hand (Mean: {np.mean(left_rot_error):.1f}°)')
        ax3.plot(time_steps, right_rot_error, '-', color='#D62828', 
                linewidth=2.5, alpha=0.9, 
                label=f'Right Hand (Mean: {np.mean(right_rot_error):.1f}°)')
        
        ax3.set_xlabel('Time Step', fontsize=11)
        ax3.set_ylabel('Rotation Error (degrees)', fontsize=11)
        ax3.legend(loc='best', fontsize=10, framealpha=0.9)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_ylim(bottom=0)  # Error is always non-negative
        
        # === 3. Keypoints Error - Left Hand ===
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.set_title('Left Hand Keypoints Error (L2 Distance)', fontsize=13, fontweight='bold', pad=10)
        
        # Extract left hand keypoints: [H, 15] = 5 fingers × 3 coords
        left_keypoints_gt = gt_action[:, 18:33]  # [H, 15]
        left_keypoints_pred = pred_action[:, 18:33]  # [H, 15]
        
        # Reshape to [H, 5, 3] for easier indexing
        left_keypoints_gt = left_keypoints_gt.reshape(horizon, 5, 3)
        left_keypoints_pred = left_keypoints_pred.reshape(horizon, 5, 3)
        
        # Compute L2 error for each finger (3D position error)
        left_keypoints_error = np.sqrt(np.sum((left_keypoints_pred - left_keypoints_gt) ** 2, axis=2))  # [H, 5]
        
        # Use different colors for each finger
        finger_colors = ['#E63946', '#F77F00', '#FFB703', '#06A77D', '#219EBC']  # Red, Orange, Yellow, Green, Blue
        
        # Plot L2 error for each finger
        for finger_idx, finger_name in enumerate(finger_names):
            ax4.plot(time_steps, left_keypoints_error[:, finger_idx], 
                    '-', color=finger_colors[finger_idx], 
                    linewidth=2.5, alpha=0.9, label=finger_name)
        
        # Compute and print statistics
        left_mean_errors = np.mean(left_keypoints_error, axis=0)
        left_max_errors = np.max(left_keypoints_error, axis=0)
        print(f"Left hand keypoints error (L2, mm):")
        for finger_idx, finger_name in enumerate(finger_names):
            print(f"  {finger_name}: Mean={left_mean_errors[finger_idx]*1000:.2f}mm, Max={left_max_errors[finger_idx]*1000:.2f}mm")
        
        ax4.set_xlabel('Time Step', fontsize=11)
        ax4.set_ylabel('Error (m)', fontsize=11)
        ax4.legend(loc='best', fontsize=10, framealpha=0.9)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_ylim(bottom=0)  # Error is always non-negative
        
        # === 4. Keypoints Error - Right Hand ===
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.set_title('Right Hand Keypoints Error (L2 Distance)', fontsize=13, fontweight='bold', pad=10)
        
        # Extract right hand keypoints: [H, 15] = 5 fingers × 3 coords
        right_keypoints_gt = gt_action[:, 33:48]  # [H, 15]
        right_keypoints_pred = pred_action[:, 33:48]  # [H, 15]
        
        # Reshape to [H, 5, 3] for easier indexing
        right_keypoints_gt = right_keypoints_gt.reshape(horizon, 5, 3)
        right_keypoints_pred = right_keypoints_pred.reshape(horizon, 5, 3)
        
        # Compute L2 error for each finger (3D position error)
        right_keypoints_error = np.sqrt(np.sum((right_keypoints_pred - right_keypoints_gt) ** 2, axis=2))  # [H, 5]
        
        # Plot L2 error for each finger
        for finger_idx, finger_name in enumerate(finger_names):
            ax5.plot(time_steps, right_keypoints_error[:, finger_idx], 
                    '-', color=finger_colors[finger_idx], 
                    linewidth=2.5, alpha=0.9, label=finger_name)
        
        # Compute and print statistics
        right_mean_errors = np.mean(right_keypoints_error, axis=0)
        right_max_errors = np.max(right_keypoints_error, axis=0)
        print(f"Right hand keypoints error (L2, mm):")
        for finger_idx, finger_name in enumerate(finger_names):
            print(f"  {finger_name}: Mean={right_mean_errors[finger_idx]*1000:.2f}mm, Max={right_max_errors[finger_idx]*1000:.2f}mm")
        
        ax5.set_xlabel('Time Step', fontsize=11)
        ax5.set_ylabel('Error (m)', fontsize=11)
        ax5.legend(loc='best', fontsize=10, framealpha=0.9)
        ax5.grid(True, alpha=0.3, linestyle='--')
        ax5.set_ylim(bottom=0)  # Error is always non-negative
        
        # Add main title
        fig.suptitle(f'Action Prediction vs Ground Truth (Keypoint Mode) - Sample {sample_idx}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save with high quality
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Reset style
        plt.style.use('default')
        
        print(f"Action comparison plot saved to {output_path}")

    def visualize_mesh(self, raw_sample, action_pred, background_img, intrinsic, extrinsic_c2w, sample_idx):
        """Visualize mesh comparison"""
        mano_shape = raw_sample['action_shape']
        action_wrist = raw_sample['action'][:, :18]
        action_hand = raw_sample['action'][:, 18:]
        mano_data = sample_to_manovis(action_pred, action_wrist, action_hand, mano_shape, extrinsic_c2w)
        
        output_dir_base = os.path.join(self.cfg.testing.output_dir, "test_legendvla_workspace")
        os.makedirs(output_dir_base, exist_ok=True)
        output_dir = os.path.join(output_dir_base, f"{sample_idx}_mesh")
        
        # Convert BGR to RGB if needed
        needs_conversion = (self.cfg.testing.target_width != 384 or self.cfg.testing.target_height != 384)
        background_img = self.convert_bgr_to_rgb(background_img, needs_conversion)
        
        vis_hand_plot_comparison(
            mano_data['predicted']['rot'], mano_data['predicted']['trans'], 
            mano_data['predicted']['theta'],
            mano_data['ground_truth']['rot'], mano_data['ground_truth']['trans'], 
            mano_data['ground_truth']['theta'], 
            mano_data['beta'], mano_data['sides'], 
            background_img, intrinsic, extrinsic_c2w, output_dir, fps=30)
    
    def visualize_tokenizer(self, raw_sample, inputs, state_in_camera, action_in_camera, 
                             background_img, intrinsic, extrinsic_c2w, sample_idx):
        """Visualize tokenizer reconstruction"""
        mano_shape = raw_sample['action_shape']
        action_wrist = raw_sample['action'][:, :18]
        action_hand = raw_sample['action'][:, 18:]
        
        output_dir_base = os.path.join(self.cfg.testing.output_dir, "test_tokenizer_workspace")
        os.makedirs(output_dir_base, exist_ok=True)
        output_dir = os.path.join(output_dir_base, str(sample_idx))
        
        # Decode using tokenizer
        if self.cfg.testing.tokenizer_type == 'vq':
            encoded_action_gt = self.vq_tokenizer['actions'](inputs['actions'])
            decoded_action_normalized = self.vq_tokenizer['actions'].decode(encoded_action_gt)[0]
        elif self.cfg.testing.tokenizer_type == 'fast':
            encoded_action_gt = self.fast_tokenizer['actions'](inputs['actions'])
            decoded_action_normalized = self.fast_tokenizer['actions'].decode(encoded_action_gt)[0]
        else:
            raise ValueError(f"Unknown tokenizer type: {self.cfg.testing.tokenizer_type}")
        
        # Unnormalize
        decoded_dict = {"actions": torch.from_numpy(decoded_action_normalized).unsqueeze(0)}
        decoded_action_gt = self.normalizer.unnormalize(decoded_dict)['actions'].squeeze(0).numpy()
        decoded_action_gt = get_absolute_action(state_in_camera.squeeze(0), decoded_action_gt)
        decoded_action_gt = decoded_action_gt.cpu().numpy()
        
        # Compute errors
        self.compute_tokenizer_errors(decoded_action_gt, action_in_camera)
        
        # Visualize
        mano_data_decoded = sample_to_manovis(decoded_action_gt, action_wrist, action_hand, mano_shape, extrinsic_c2w)
        
        # Convert BGR to RGB if needed
        needs_conversion = (self.cfg.testing.target_width != 384 or self.cfg.testing.target_height != 384)
        background_img = self.convert_bgr_to_rgb(background_img, needs_conversion)
        
        vis_hand_plot_comparison(
            mano_data_decoded['predicted']['rot'], mano_data_decoded['predicted']['trans'], 
            mano_data_decoded['predicted']['theta'],
            mano_data_decoded['predicted']['rot'], mano_data_decoded['ground_truth']['trans'], 
            mano_data_decoded['ground_truth']['theta'], 
            mano_data_decoded['beta'], mano_data_decoded['sides'], 
            background_img, intrinsic, extrinsic_c2w, output_dir, fps=30)
    
    def visualize_keypoint(self, raw_sample, action_pred_in_camera, background_img, 
                           intrinsic, sample_idx):
        """Visualize keypoint predictions"""
        mano_root = self.cfg.testing.mano_root_dir
        print(f"Initializing hand visualizer with MANO root: {mano_root}")
        visualizer = HandVisualizer(mano_root=mano_root)
        
        # Prepare data for visualization
        keypoint_sequence, wrist_sequence, gt_keypoint_sequence, gt_wrist_sequence, \
            intrinsic_matrix, extrinsic_matrices, presence = sample_for_vis(
                action_pred_in_camera, raw_sample)
        
        # Move to device
        keypoint_sequence = keypoint_sequence.to(visualizer.device).float()
        wrist_sequence = wrist_sequence.to(visualizer.device).float()
        if self.cfg.testing.show_gt:
            gt_keypoint_sequence = gt_keypoint_sequence.to(visualizer.device).float()
            gt_wrist_sequence = gt_wrist_sequence.to(visualizer.device).float()
        
        num_frames = keypoint_sequence.shape[0]
        print(f"Detected {num_frames} frames in the prediction sequences")
        
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
                print(f"Warning: Extrinsic sequence frames ({extrinsic_matrices.shape[0]}) does not match prediction frames ({num_frames})")
                print(f"Using identity transformation instead")
            else:
                extrinsic_sequence = extrinsic_matrices.to(visualizer.device).float()
                print(f"Loaded camera extrinsics with {extrinsic_sequence.shape[0]} frames")
        else:
            print("No extrinsic data available, using identity transformation")
        
        # Create output directory
        output_dir_base = os.path.join(self.cfg.testing.output_dir, "test_keypoint_workspace")
        os.makedirs(output_dir_base, exist_ok=True)
        output_path_2d = os.path.join(output_dir_base, f"{sample_idx}_keypoint.mp4")
        
        # Prepare ground truth data
        gt_keypoint_seq = gt_keypoint_sequence if self.cfg.testing.show_gt else None
        gt_wrist_seq = gt_wrist_sequence if self.cfg.testing.show_gt else None
        
        # Generate 2D projection video
        print(f"Generating 2D projection video...")
        visualizer.generate_2d_video(
            background_img, wrist_sequence, output_path_2d, self.cfg.testing.fps,
            fx=fx, fy=fy, cx=cx, cy=cy,
            extrinsic_sequence=None,
            keypoint_sequence=keypoint_sequence,
            show_mesh=self.cfg.testing.show_mesh, mesh_alpha=self.cfg.testing.mesh_alpha,
            gt_keypoint_sequence=gt_keypoint_seq, gt_wrist_sequence=gt_wrist_seq,
            presence=presence, mode='keypoint')
        
        return output_path_2d
    
    def process_single_sample(self, sample_idx, i):
        """Process a single sample: inference, visualization, and saving"""
        print(f"\nProcessing sample {i+1} (index {sample_idx})")
        
        # Get raw sample
        self.dataset.set_return_raw_sample(True)
        raw_sample = self.dataset[sample_idx]
        self.dataset.set_return_raw_sample(False)
        
        # Print sample information
        print("Raw sample keys and data types:")
        for key, value in raw_sample.items():
            if hasattr(value, 'dtype'):
                print(f"  {key}: {type(value)} - dtype: {value.shape}")
            else:
                print(f"  {key}: {type(value)} - value: {value}")
        
        if 'dataset_source' in raw_sample:
            dataset_path = raw_sample['dataset_source']
            dataset_name = os.path.basename(dataset_path).replace('.zarr', '')
            dataset_idx = raw_sample['dataset_idx'].item() if hasattr(raw_sample['dataset_idx'], 'item') else raw_sample['dataset_idx']
            print(f"Dataset source: {dataset_name} (index: {dataset_idx})")
            print(f"Full path: {dataset_path}")
        
        # Get processed sample and run inference
        sample = self.dataset[sample_idx]
        collate_fn = self.dataset.get_collator()
        batch = collate_fn([sample])
        
        action_pred_unnormalized, inputs = self.run_model_inference(batch)
        
        # Prepare image and camera parameters
        background_img = np.array(raw_sample['image'])
        extrinsic_w2c = raw_sample['extrinsic']
        extrinsic_c2w = invert_extrinsics(extrinsic_w2c)
        intrinsic_4d = raw_sample['intrinsic'][0]
        
        # Resize image and adjust intrinsics if needed
        background_img, intrinsic = self.resize_image_and_adjust_intrinsics(
            background_img, intrinsic_4d, 
            self.cfg.testing.target_width, self.cfg.testing.target_height)
        
        # Convert action to camera frame
        action_pred, action_pred_in_camera, action_in_camera, state_in_camera = \
            self.convert_action_to_camera_frame(raw_sample, action_pred_unnormalized, extrinsic_w2c)
        
        # Plot action comparison
        plot_output_path = os.path.join(self.cfg.testing.output_dir, f"action_comparison_sample_{i+1:03d}.png")
        
        
        self.plot_action_comparison(
            pred_action=action_pred_in_camera,
            gt_action=action_in_camera,
            output_path=plot_output_path,
            sample_idx=i+1
        )
        
        # Visualize based on type
        if self.cfg.testing.visualize_type == 'mesh':
            self.visualize_mesh(raw_sample, action_pred, background_img, 
                                intrinsic, extrinsic_c2w, i)
        elif self.cfg.testing.visualize_type == 'tokenizer':
            self.visualize_tokenizer(raw_sample, inputs, state_in_camera, action_in_camera,
                                     background_img, intrinsic, extrinsic_c2w, i)
        elif self.cfg.testing.visualize_type == 'keypoint':
            output_path_2d = self.visualize_keypoint(raw_sample, action_pred_in_camera, 
                                                     background_img, intrinsic, i)
            print(f"Sample {i+1} (index {sample_idx}) completed:")
            print(f"  2D projection video: {output_path_2d}")
    
    def run(self):
        """
        Main execution method to run the test workflow.
        
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
        
        self.dataset.set_return_raw_sample(True)
        min_required_frames = self.dataset.vla_dataset.history + self.dataset.vla_dataset.horizon
        valid_indices = self.filter_valid_samples(dataset_size, min_required_frames)
        
        print(f"Found {len(valid_indices)} valid samples (with at least {min_required_frames} frames available)")
        print(f"Filtered out {dataset_size - len(valid_indices)} samples without enough frames")
        
        # Sample from valid indices
        sample_indices = random.sample(valid_indices, min(num_samples, len(valid_indices)))
        self.dataset.set_return_raw_sample(False)
        
        # Process all samples
        print(f"\nProcessing {len(sample_indices)} samples...")
        for i, sample_idx in enumerate(sample_indices):
            self.process_single_sample(sample_idx, i)


OmegaConf.register_new_resolver("eval", eval, replace=True)
# %%
if __name__ == "__main__":
    # Load config
    config_path = "/home/zengfanlian/EgoVLA/src/config/experiment/test_legendvla.yaml"
    cfg = OmegaConf.load(config_path)
    
    # Initialize workspace and run
    workspace = TestLegendVLAWorkspace(cfg)
    workspace.run()

