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
from src.workspace.base_workspace import BaseWorkspace
from src.policy.legendvla import LegendVLA
from src.dataset.base_dataset import BaseImageDataset
from src.dataset.paligemma_processing import PaliGemmaVLAProcessor, PaliGemmaProcessor
from transformers import AutoTokenizer
from src.model.action.fast_tokenizer import UniversalActionProcessor
from src.model.action.vq_tokenizer import VQActionProcessor
from src.utils.mano_vis import mano_forward, vis_hand_plot, vis_hand_plot_comparison
from src.utils.mano_utils import rot6d_to_rotmat, sample_to_manovis
from visualize import HandVisualizer, sample_for_vis
from src.dataset.legendvla_dataset import get_absolute_action, transform_wrist_to_target_frame
import cv2
from src.utils.mano_utils import invert_extrinsics

def plot_action_comparison(pred_action, gt_action, output_path, sample_idx=0):
    """
    Plot 2D comparison between predicted action and ground truth action with improved visualization
    
    Args:
        pred_action: torch.Tensor or np.ndarray, shape: [H, action_dim] - predicted action
        gt_action: torch.Tensor or np.ndarray, shape: [H, action_dim] - ground truth action
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
    
    # Create figure with better layout - 2x2 grid
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    # Color schemes
    gt_color = '#2E86AB'  # Blue for GT
    pred_color = '#A23B72'  # Purple/Magenta for Prediction
    
    # === 1. Wrist Translation - Split into Left and Right ===
    # Left hand translation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Left Wrist Translation', fontsize=13, fontweight='bold', pad=10)
    coords = ['X', 'Y', 'Z']
    colors_coord = ['#E63946', '#F77F00', '#06A77D']  # Red, Orange, Green
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
    
    # Debug: print statistics
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
    
    # === 3. Hand Parameters - Show mean and std envelope ===
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_title('Hand Parameters (Left Hand - Dims 18-32)', fontsize=13, fontweight='bold', pad=10)
    
    # Compute mean and std across all hand parameter dimensions
    gt_left_mean = np.mean(gt_action[:, 18:33], axis=1)
    gt_left_std = np.std(gt_action[:, 18:33], axis=1)
    pred_left_mean = np.mean(pred_action[:, 18:33], axis=1)
    pred_left_std = np.std(pred_action[:, 18:33], axis=1)
    
    ax4.plot(time_steps, gt_left_mean, '--', color=gt_color, 
            alpha=0.8, linewidth=2.5, label='GT Mean')
    ax4.fill_between(time_steps, gt_left_mean - gt_left_std, gt_left_mean + gt_left_std, 
                     color=gt_color, alpha=0.2, label='GT Std')
    ax4.plot(time_steps, pred_left_mean, '-', color=pred_color, 
            alpha=0.95, linewidth=2, label='Pred Mean')
    ax4.fill_between(time_steps, pred_left_mean - pred_left_std, pred_left_mean + pred_left_std, 
                     color=pred_color, alpha=0.2, label='Pred Std')
    
    ax4.set_xlabel('Time Step', fontsize=11)
    ax4.set_ylabel('Parameter Value', fontsize=11)
    ax4.legend(loc='best', fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # Right hand parameters
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_title('Hand Parameters (Right Hand - Dims 33-47)', fontsize=13, fontweight='bold', pad=10)
    
    gt_right_mean = np.mean(gt_action[:, 33:48], axis=1)
    gt_right_std = np.std(gt_action[:, 33:48], axis=1)
    pred_right_mean = np.mean(pred_action[:, 33:48], axis=1)
    pred_right_std = np.std(pred_action[:, 33:48], axis=1)
    
    ax5.plot(time_steps, gt_right_mean, '--', color=gt_color, 
            alpha=0.8, linewidth=2.5, label='GT Mean')
    ax5.fill_between(time_steps, gt_right_mean - gt_right_std, gt_right_mean + gt_right_std, 
                     color=gt_color, alpha=0.2, label='GT Std')
    ax5.plot(time_steps, pred_right_mean, '-', color=pred_color, 
            alpha=0.95, linewidth=2, label='Pred Mean')
    ax5.fill_between(time_steps, pred_right_mean - pred_right_std, pred_right_mean + pred_right_std, 
                     color=pred_color, alpha=0.2, label='Pred Std')
    
    ax5.set_xlabel('Time Step', fontsize=11)
    ax5.set_ylabel('Parameter Value', fontsize=11)
    ax5.legend(loc='best', fontsize=9, framealpha=0.9)
    ax5.grid(True, alpha=0.3, linestyle='--')
    
    # Add main title
    fig.suptitle(f'Action Prediction vs Ground Truth - Sample {sample_idx}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save with high quality
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Reset style
    plt.style.use('default')
    
    print(f"Action comparison plot saved to {output_path}")


def preprocess_batch_for_inference(batch, model, device, dtype=torch.float32, sample_fm_time=False):
    """Preprocess batch for inference, similar to training script"""
    # Extract data from batch and move to device
    pixel_values = batch["pixel_values"].to(device)
    actions = batch["actions"].to(device)
    actions_valid_mask = batch["actions_valid_mask"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    answer_start_idx = batch["answer_start_idx"].to(device)
    labels = batch["labels"].to(device)

    # Get unwrapped model for mask building
    if hasattr(model, 'module'):
        unwrapped_model = model.module
    else:
        unwrapped_model = model
    
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
OmegaConf.register_new_resolver("eval", eval, replace=True)
# %%
if __name__ == "__main__":
    # Parse command line arguments
    config_path = "/home/fanlian/EgoVLA/src/config/experiment/test_legendvla.yaml"
    # Load config
    cfg = OmegaConf.load(config_path)
        
    # set seed
    # seed = cfg.testing.seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    # configure model
    model: LegendVLA
    model = hydra.utils.instantiate(cfg.policy)
    
    # Load checkpoint if specified
    if cfg.testing.checkpoint_path is not None:
        print(f"Loading checkpoint from {cfg.testing.checkpoint_path}")
        # Create a temporary workspace to use load_checkpoint method
        temp_workspace = BaseWorkspace(cfg)
        temp_workspace.model = model
        model.load_state_dict(torch.load(cfg.testing.checkpoint_path, map_location='cpu', weights_only=False)['module'])
        print("Checkpoint loaded successfully!")

    # Configure device
    if cfg.testing.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(cfg.testing.device)
    model = model.to(device)
    
    # Convert entire model to bfloat16 for consistency (VLM uses bfloat16)
    model.eval()

    # configure dataset
    dataset: BaseImageDataset
    dataset = hydra.utils.instantiate(cfg.dataset)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.policy.cfg.pretrained_model_path, padding_side="right"
    )
    fast_tokenizer = {
        "states": UniversalActionProcessor.from_pretrained(
            os.path.join(cfg.processor.fast_tokenizer_path, "states")
        ),
        "actions": UniversalActionProcessor.from_pretrained(
            os.path.join(cfg.processor.fast_tokenizer_path, "actions")
        )
    }
    vq_tokenizer = {
        "states": VQActionProcessor.from_pretrained(
            cfg.processor.hand_states_tokenizer_path
        ),
        "actions": VQActionProcessor.from_pretrained(
            cfg.processor.hand_actions_tokenizer_path
        )
    }
    vla_processor = PaliGemmaVLAProcessor(
        tokenizer,
        vq_tokenizer,
        num_image_tokens=cfg.policy.vision_tower.config.num_image_tokens,
        max_seq_len=cfg.policy.cfg.max_vlm_tokens,
        ignore_index=cfg.ignore_index,
        image_size=cfg.policy.vision_tower.config.image_size,
        tokenizer_padding=cfg.tokenizer_padding,
        hand_tokenizer_type=cfg.processor.hand_tokenizer_type,
    )
    vlm_processor = PaliGemmaProcessor(
        tokenizer,
        num_image_tokens=cfg.policy.vision_tower.config.num_image_tokens,
        max_seq_len=cfg.policy.cfg.max_vlm_tokens,
        ignore_index=cfg.ignore_index,
        image_size=cfg.policy.vision_tower.config.image_size,
        tokenizer_padding=cfg.tokenizer_padding,
    )
    dataset.vla_dataset.set_preprocessor(vla_processor)
    if dataset.vlm_dataset is not None:
        dataset.vlm_dataset.set_preprocessor(vlm_processor)
    
    print("Loading normalizer...")
    # Load normalizer from checkpoint

    # Check for normalizer file
    normalizer_path = cfg.testing.normalizer_path 
    if os.path.exists(normalizer_path):
        normalizer = pickle.load(open(normalizer_path, 'rb'))
        print(f"Normalizer loaded from {normalizer_path}")
    else:
        print(f"Normalizer file not found at {normalizer_path}") 
        print("Computing normalizer from dataset...")
        normalizer = dataset.vla_dataset.get_normalizer()
        print("Normalizer computed successfully")
        
        # Save the normalizer for future use
        os.makedirs(os.path.dirname(normalizer_path), exist_ok=True)
        with open(normalizer_path, 'wb') as f:
            pickle.dump(normalizer, f)
        print(f"Normalizer saved to {normalizer_path}")

    dataset.vla_dataset.set_normalizer(normalizer)


    # Get dataset size and sample indices
    dataset_size = len(dataset)
    print(f"Total dataset size: {dataset_size}")
    
    # Display dataset composition
    print("\nDataset composition:")
    for i, zarr_path in enumerate(cfg.vla_dataset_paths):
        dataset_name = os.path.basename(zarr_path).replace('.zarr', '')
        # Get the length of each dataset component
        if hasattr(dataset.vla_dataset, 'sampler_lens') and i < len(dataset.vla_dataset.sampler_lens):
            component_size = dataset.vla_dataset.sampler_lens[i]
            print(f"  {dataset_name}: {component_size} samples")
        else:
            print(f"  {dataset_name}: size unknown")
    
    num_samples = cfg.testing.num_samples
    print(f"\nSampling {num_samples} random samples from dataset...")
    
    # Enable raw sample mode first to ensure samplers load enough frames
    dataset.set_return_raw_sample(True)
    
    # Filter out samples that don't have enough frames for visualization
    # We need history + horizon frames total
    min_required_frames = dataset.vla_dataset.history + dataset.vla_dataset.horizon
    valid_indices = []
    
    for idx in range(dataset_size):
        # Get the sample's indices from the dataset
        vla_dataset = dataset.vla_dataset
        # Find which sampler this index belongs to
        cumulative_len = 0
        for sampler_idx, sampler_len in enumerate(vla_dataset.sampler_lens):
            if idx < cumulative_len + sampler_len:
                # This index belongs to this sampler
                local_idx = idx - cumulative_len
                sampler = vla_dataset.samplers[sampler_idx]
                replay_buffer = sampler.replay_buffer
                
                # Get the buffer indices for this sample
                buffer_start_idx, buffer_end_idx, _, _ = sampler.indices[local_idx]
                
                # Find which episode this sample belongs to
                episode_ends = replay_buffer.episode_ends
                episode_idx = np.searchsorted(episode_ends, buffer_end_idx)
                episode_end = episode_ends[episode_idx]
                
                # Check if we can load enough frames from buffer_start_idx
                # We need at least min_required_frames from buffer_start_idx
                available_frames = episode_end - buffer_start_idx
                if available_frames >= min_required_frames:
                    valid_indices.append(idx)
                break
            cumulative_len += sampler_len
    
    print(f"Found {len(valid_indices)} valid samples (with at least {min_required_frames} frames available)")
    print(f"Filtered out {dataset_size - len(valid_indices)} samples without enough frames")
    
    # Sample from valid indices only
    sample_indices = random.sample(valid_indices, min(num_samples, len(valid_indices)))
    
    # Disable raw sample mode temporarily
    dataset.set_return_raw_sample(False)
    
    # Process all samples
    all_sample_data = []
    
    print(f"\nProcessing {len(sample_indices)} samples...")
    
    for i, sample_idx in enumerate(sample_indices):
        print(f"\nProcessing sample {i+1}/{len(sample_indices)} (index {sample_idx})")
        
        # Get the sample from dataset for model prediction
        dataset.set_return_raw_sample(True)
        raw_sample = dataset[sample_idx]
        dataset.set_return_raw_sample(False)
        
        # Check raw sample data types
        print("Raw sample keys and data types:")
        for key, value in raw_sample.items():
            if hasattr(value, 'dtype'):
                print(f"  {key}: {type(value)} - dtype: {value.shape}")
            else:
                print(f"  {key}: {type(value)} - value: {value}")
        
        # Display dataset source information
        if 'dataset_source' in raw_sample:
            dataset_path = raw_sample['dataset_source']
            dataset_name = os.path.basename(dataset_path).replace('.zarr', '')
            dataset_idx = raw_sample['dataset_idx'].item() if hasattr(raw_sample['dataset_idx'], 'item') else raw_sample['dataset_idx']
            print(f"Dataset source: {dataset_name} (index: {dataset_idx})")
            print(f"Full path: {dataset_path}")
        else:
            print("Dataset source information not available")
        
        sample = dataset[sample_idx]

        collate_fn = dataset.get_collator()
        batch = collate_fn([sample])
        
        # Preprocess batch like in training script
        inputs = preprocess_batch_for_inference(batch, model, device, sample_fm_time=True)

        # Run prediction
        with torch.no_grad():
            # torch.FloatTensor: [B, horizon_steps, human_action_dim] Generated human action sequence [1, 30, 48] for each sample
            action_pred_normalized = model("infer_action", inputs)
            loss = model("train_flow", inputs)
        print("loss:", loss)
        action_pred_cpu = action_pred_normalized.cpu()
        action_pred_dict = {"actions": action_pred_cpu}
        action_pred_unnormalized = normalizer.unnormalize(action_pred_dict)


        background_img = np.array(raw_sample['image'])         # [N, H, W, 3]

        extrinsic_w2c = raw_sample['extrinsic'] # [N, 4, 4]
        extrinsic_c2w = invert_extrinsics(extrinsic_w2c)
        intrinsic_4d = raw_sample['intrinsic'][0] # [N, 4]
        intrinsic = torch.tensor([
                    [intrinsic_4d[0], 0, intrinsic_4d[2]],
                    [0, intrinsic_4d[1], intrinsic_4d[3]],
                    [0, 0, 1]
                ])

        if cfg.testing.target_width != 384 or cfg.testing.target_height != 384:
            # Check if background_img is a sequence [N, H, W, 3] or single frame [H, W, 3]
            is_sequence = background_img.ndim == 4
            
            if is_sequence:
                # Process each frame in the sequence
                num_frames, orig_h, orig_w = background_img.shape[:3]
                print(f"Original background image size: {orig_w}x{orig_h} ({num_frames} frames)")
                
                target_size = (cfg.testing.target_width, cfg.testing.target_height)  # (width, height) for cv2.resize
                resized_frames = []
                
                for frame_idx in range(num_frames):
                    # Convert from RGB to BGR for OpenCV compatibility
                    frame_bgr = cv2.cvtColor(background_img[frame_idx], cv2.COLOR_RGB2BGR)
                    # Resize frame
                    frame_resized = cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_CUBIC)
                    resized_frames.append(frame_resized)
                
                background_img = np.stack(resized_frames, axis=0)  # [N, H, W, 3]
                print(f"Resized background image to: {background_img.shape[2]}x{background_img.shape[1]} ({num_frames} frames)")
            else:
                # Single frame processing (original code)
                # Convert from RGB to BGR for OpenCV compatibility
                background_img = cv2.cvtColor(background_img, cv2.COLOR_RGB2BGR)
                
                # Resize image from 384x384 to target resolution
                original_size = background_img.shape[:2]  # (height, width)
                target_size = (cfg.testing.target_width, cfg.testing.target_height)  # (width, height) for cv2.resize
                
                print(f"Original background image size: {original_size[1]}x{original_size[0]}")
                background_img = cv2.resize(background_img, target_size, interpolation=cv2.INTER_CUBIC)
                print(f"Resized background image to: {background_img.shape[1]}x{background_img.shape[0]}")
            
            # Extract camera intrinsics

            # Original camera intrinsics for 384x384 image
            fx_original = intrinsic_4d[0]
            fy_original = intrinsic_4d[1]
            cx_original = intrinsic_4d[2]
            cy_original = intrinsic_4d[3]


            # Calculate scaling factors for resizing from 384x384 to target resolution
            # Different scaling factors for width and height due to aspect ratio change
            scale_factor_x = cfg.testing.target_width / 384.0  # Scale factor for width
            scale_factor_y = cfg.testing.target_height / 384.0  # Scale factor for height
            
            # Scale camera intrinsics accordingly
            fx = fx_original * scale_factor_x  # Scale focal length in x direction
            fy = fy_original * scale_factor_y  # Scale focal length in y direction
            cx = cx_original * scale_factor_x  # Scale principal point x coordinate
            cy = cy_original * scale_factor_y  # Scale principal point y coordinate
            
            intrinsic = torch.tensor([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]
                    ])        


        state_in_camera = raw_sample['state'].clone()
        state_in_camera[:,:18] = transform_wrist_to_target_frame(state_in_camera[:,:18], extrinsic_w2c[0])

        action_in_camera = raw_sample['action'].clone()
        action_in_camera[:,:18] = transform_wrist_to_target_frame(action_in_camera[:,:18], extrinsic_w2c[0])
        
        print(state_in_camera.squeeze(0).shape, action_pred_unnormalized["actions"].squeeze(0).shape)
        action_pred = get_absolute_action(state_in_camera.squeeze(0), action_pred_unnormalized["actions"].squeeze(0))

        # Plot action comparison (2D curves)
        plot_output_path = os.path.join(cfg.testing.output_dir, f"action_comparison_sample_{i+1:03d}.png")
        plot_action_comparison(
            pred_action=action_pred,
            gt_action=action_in_camera,
            output_path=plot_output_path,
            sample_idx=i+1
        )

        if cfg.testing.visualize_type == 'mesh':
            mano_shape = raw_sample['action_shape'] # [N, 20]
            action_wrist = raw_sample['action'][:,:18]
            action_hand = raw_sample['action'][:,18:]
            mano_data = sample_to_manovis(action_pred, action_wrist, action_hand, mano_shape, extrinsic_c2w)
            output_dir = f"/data/fanlian/outputs/test_legendvla_workspace/{i}"
       
            # Convert single frame [H, W, 3] to sequence format [N, H, W, 3] for vis_hand_plot
            # Since we have multiple frames in the sequence, we need to repeat the image for each frame

            # num_frames = 30
            # image_sequence = np.repeat(background_img[np.newaxis, :, :, :], num_frames, axis=0)  # [N, H, W, 3]

            # Convert BGR to RGB for visualization
            # Handle both single frame [H, W, 3] and multi-frame [N, H, W, 3] cases
            if cfg.testing.target_width != 384 or cfg.testing.target_height != 384:
                if background_img.ndim == 4:
                    # Multi-frame: convert each frame from BGR to RGB
                    background_img = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in background_img], axis=0)
                else:
                    # Single frame: convert directly
                    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
            
            vis_hand_plot_comparison(mano_data['predicted']['rot'], mano_data['predicted']['trans'], mano_data['predicted']['theta'],mano_data['ground_truth']['rot']
            , mano_data['ground_truth']['trans'], mano_data['ground_truth']['theta'], mano_data['beta'], mano_data['sides'], background_img, intrinsic, extrinsic_c2w, output_dir, fps=30)
        elif cfg.testing.visualize_type == 'tokenizer':
            mano_shape = raw_sample['action_shape'] # [N, 20]
            action_wrist = raw_sample['action'][:,:18]
            action_hand = raw_sample['action'][:,18:]
            mano_data = sample_to_manovis(action_pred, action_wrist, action_hand, mano_shape, extrinsic_c2w)
            output_dir = f"/data/fanlian/outputs/test_tokenizer_workspace/{i}"
            encoded_action_gt = vq_tokenizer['actions'](raw_sample['action'])[0]
            decoded_action_gt = vq_tokenizer['actions'].decode(encoded_action_gt)
            mano_data_decoded = sample_to_manovis(decoded_action_gt, action_wrist, action_hand, mano_shape, extrinsic_c2w)
            # Convert single frame [H, W, 3] to sequence format [N, H, W, 3] for vis_hand_plot
            # Since we have multiple frames in the sequence, we need to repeat the image for each frame

            # num_frames = 30
            # image_sequence = np.repeat(background_img[np.newaxis, :, :, :], num_frames, axis=0)  # [N, H, W, 3]

            # Convert BGR to RGB for visualization
            # Handle both single frame [H, W, 3] and multi-frame [N, H, W, 3] cases
            if cfg.testing.target_width != 384 or cfg.testing.target_height != 384:
                if background_img.ndim == 4:
                    # Multi-frame: convert each frame from BGR to RGB
                    background_img = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in background_img], axis=0)
                else:
                    # Single frame: convert directly
                    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
            
            vis_hand_plot_comparison(mano_data_decoded['ground_truth']['rot'], mano_data_decoded['ground_truth']['trans'], mano_data_decoded['ground_truth']['theta'],mano_data['ground_truth']['rot']
            , mano_data['ground_truth']['trans'], mano_data['ground_truth']['theta'], mano_data['beta'], mano_data['sides'], background_img, intrinsic, extrinsic_c2w, output_dir, fps=30)            
        elif cfg.testing.visualize_type == 'skeleton':
            
            mano_root = cfg.testing.mano_root_dir
            print(f"Initializing hand visualizer with MANO root: {mano_root}")
            visualizer = HandVisualizer(mano_root=mano_root)

            # all states and actions are in camera frame after preprocessing
            mano_sequence, wrist_sequence, gt_mano_sequence, gt_wrist_sequence, intrinsic_matrix, extrinsic_matrices, presence = sample_for_vis(
                action_pred, raw_sample)

            # Move to device and convert to float32 to avoid dtype issues
            mano_sequence = mano_sequence.to(visualizer.device).float()
            wrist_sequence = wrist_sequence.to(visualizer.device).float()
            if cfg.testing.show_gt:
                gt_mano_sequence = gt_mano_sequence.to(visualizer.device).float()
                gt_wrist_sequence = gt_wrist_sequence.to(visualizer.device).float()
            
            
            num_frames = mano_sequence.shape[0]
            print(f"Detected {num_frames} frames in the prediction sequences")
            
            # Scale camera intrinsics accordingly
            fx = intrinsic[0,0]
            fy = intrinsic[1,1]
            cx = intrinsic[0,2]
            cy = intrinsic[1,2]
            
            # Process extrinsic matrices
            extrinsic_sequence = None
            if extrinsic_matrices is not None:
                # Convert numpy array to torch tensor if necessary
                if isinstance(extrinsic_matrices, np.ndarray):
                    extrinsic_matrices = torch.from_numpy(extrinsic_matrices).float()
                
                # Validate extrinsic sequence shape matches frame count
                if extrinsic_matrices.shape[0] != num_frames:
                    print(f"Warning: Extrinsic sequence frames ({extrinsic_matrices.shape[0]}) does not match prediction frames ({num_frames})")
                    print(f"Using identity transformation instead")
                else:
                    extrinsic_sequence = extrinsic_matrices.to(visualizer.device).float()
                    print(f"Loaded camera extrinsics with {extrinsic_sequence.shape[0]} frames")
            else:
                print("No extrinsic data available, using identity transformation")
            
            # Display presence information
            presence_desc = {1: "left hand only", 2: "right hand only", 3: "both hands"}
            print(f"  - Hand visibility: {presence_desc.get(presence, 'unknown')} (presence={presence})")
            
            # Create output directory if it doesn't exist
            os.makedirs(cfg.testing.output_dir, exist_ok=True)
            
            # Construct full output paths with sample index to avoid overwriting
            # Extract base name and extension from video names
            video_2d_base = os.path.splitext(cfg.testing.video_name)[0]
            video_3d_base = os.path.splitext(cfg.testing.video_3d_name)[0]
            
            output_path_2d = os.path.join(cfg.testing.output_dir, f"{video_2d_base}_sample_{i+1:03d}.mp4")
            output_path_3d = os.path.join(cfg.testing.output_dir, f"{video_3d_base}_sample_{i+1:03d}.mp4")
            
            # Prepare ground truth data if show_gt is enabled
            gt_mano_seq = gt_mano_sequence if cfg.testing.show_gt else None
            gt_wrist_seq = gt_wrist_sequence if cfg.testing.show_gt else None
            
            # Generate 2D projection video (original functionality)
            print(f"Generating 2D projection video...")
            visualizer.generate_2d_projection_video(background_img, mano_sequence, wrist_sequence, 
                                        output_path_2d, cfg.testing.fps, fx=fx, fy=fy, cx=cx, cy=cy,
                                        extrinsic_sequence = None,
                                        show_mesh=cfg.testing.show_mesh, mesh_alpha=cfg.testing.mesh_alpha,
                                        gt_mano_sequence=gt_mano_seq, gt_wrist_sequence=gt_wrist_seq,
                                        presence=presence)
            
            # Generate 3D mesh and skeleton video in 3D coordinate space
            # print(f"Generating 3D mesh + skeleton video...")
            # visualizer.generate_3d_mesh_skeleton_video(mano_sequence, wrist_sequence,
            #                                           output_path_3d, cfg.testing.fps, 
            #                                           extrinsic_sequence=None,
            #                                           gt_mano_sequence=gt_mano_seq, gt_wrist_sequence=gt_wrist_seq,
            #                                           presence=presence)
            
            print(f"Sample {i+1}/{len(sample_indices)} (index {sample_idx}) completed:")
            print(f"  2D projection video: {output_path_2d}")
            # print(f"  3D mesh + skeleton video: {output_path_3d}")
        

