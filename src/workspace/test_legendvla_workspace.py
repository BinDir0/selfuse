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
from src.utils.mano_vis import mano_forward, vis_hand_plot, vis_hand_plot_comparison
from src.utils.mano_utils import rot6d_to_rotmat, sample_to_manovis
from visualize import HandVisualizer, sample_for_vis
from src.dataset.legendvla_dataset import get_absolute_action, transform_wrist_to_target_frame
import cv2
from src.utils.mano_utils import invert_extrinsics

def preprocess_batch_for_inference(batch, model, device, dtype=torch.float32, sample_fm_time=False):
    """Preprocess batch for inference, similar to training script"""
    # Extract data from batch and move to device
    pixel_values = batch["pixel_values"].to(device)
    human_actions = batch["human_actions"].to(device)
    human_actions_valid_mask = batch["human_actions_valid_mask"].to(device)
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
    causal_mask, vlm_position_ids, human_action_position_ids = (
        unwrapped_model.build_causal_mask_and_position_ids(   
            attention_mask, answer_start_idx, dtype
        )
    )
    max_vlm_tokens = input_ids.shape[-1]
    # Split mask for inference
    vlm_mask, human_action_mask = (
        unwrapped_model.split_full_mask_into_submasks(causal_mask, max_vlm_tokens)
    )

    inputs = {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": pixel_values.to(dtype),
        "vlm_position_ids": vlm_position_ids,
        "human_action_position_ids": human_action_position_ids,
        "vlm_mask": vlm_mask,
        "human_action_mask": human_action_mask,
        "causal_mask": causal_mask,
        "human_actions": human_actions.to(dtype),
        "human_actions_valid_mask": human_actions_valid_mask,
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
        "human_actions": UniversalActionProcessor.from_pretrained(
            os.path.join(cfg.processor.fast_tokenizer_path, "human_actions")
        )
    }
    
    vla_processor = PaliGemmaVLAProcessor(
        tokenizer,
        fast_tokenizer,
        num_image_tokens=cfg.policy.vision_tower.config.num_image_tokens,
        max_seq_len=cfg.policy.cfg.max_vlm_tokens,
        ignore_index=cfg.ignore_index,
        image_size=cfg.policy.vision_tower.config.image_size,
        tokenizer_padding=cfg.tokenizer_padding,
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

    if cfg.testing.normalizer_path is not None:
        normalizer = pickle.load(open(cfg.testing.normalizer_path, 'rb'))
        print(f"Normalizer loaded from {cfg.testing.normalizer_path}")
    else:
        print("Computing normalizer...")
        normalizer = dataset.vla_dataset.get_normalizer()
        print("Normalizer computed from dataset")

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
    sample_indices = random.sample(range(dataset_size), num_samples)
    
    # Sample random indices, ensuring enough future frames
    # Process all samples
    all_sample_data = []
    
    # Create augmentation transforms
    # use_augmentation = cfg.testing.use_augmentation
    # use_color_jitter = cfg.testing.use_color_jitter
    # use_gaussian_noise = cfg.testing.use_gaussian_noise
    # gaussian_noise_sigma = cfg.testing.gaussian_noise_sigma

    # if use_augmentation:
    #     if use_color_jitter:
    #         color_jitter = v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    #         print(f"ColorJitter augmentation enabled: brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1")
        
    #     if use_gaussian_noise:
    #         gaussian_noise = v2.GaussianNoise(sigma=gaussian_noise_sigma)
    #         print(f"GaussianNoise augmentation enabled: sigma={gaussian_noise_sigma}")
        
    #     enabled_augs = []
    #     if use_color_jitter: enabled_augs.append("ColorJitter")
    #     if use_gaussian_noise: enabled_augs.append("GaussianNoise")
    #     print(f"Enabled augmentations: {', '.join(enabled_augs)}")
    # else:
    #     print(f"All augmentations disabled")

    
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

        #     print(f"  Batch image shape: {batch['image'].shape}")
        #     enabled_augs = []
        #     if use_color_jitter: enabled_augs.append("ColorJitter")
        #     if use_gaussian_noise: enabled_augs.append("GaussianNoise")
        #     print(f"  Applying {', '.join(enabled_augs)} to all {batch_seq_len} frames in sequence")
            
        #     # Apply augmentation to each frame in the sequence
        #     augmented_frames = []
        #     for t in range(batch_seq_len):
        #         # Get the original frame
        #         original_frame = batch['image'][0, t]  # (C, H, W) with values [0, 1]
        #         augmented_frame = original_frame.clone()
                
        #         # Apply ColorJitter augmentation if enabled
        #         if use_color_jitter:
        #             # Convert from (C, H, W) to (H, W, C) for PIL processing
        #             frame_hwc = original_frame.permute(1, 2, 0).numpy()  # (H, W, C)
        #             frame_uint8 = (frame_hwc * 255).astype(np.uint8)
                    
        #             # Apply ColorJitter
        #             frame_pil = Image.fromarray(frame_uint8)
        #             augmented_pil = color_jitter(frame_pil)
        #             augmented_array = np.array(augmented_pil)
                    
        #             # Convert back to tensor format (C, H, W)
        #             augmented_frame = torch.from_numpy(augmented_array).float() / 255.0
        #             augmented_frame = augmented_frame.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                
        #         # Apply GaussianNoise if enabled (v2 transforms work on tensors directly)
        #         if use_gaussian_noise:
        #             augmented_frame = gaussian_noise(augmented_frame)
                
        #         augmented_frames.append(augmented_frame)
            
        #     # Stack all augmented frames and replace in batch
        #     augmented_sequence = torch.stack(augmented_frames)  # (T, C, H, W)
        #     batch['image'][0] = augmented_sequence
            
        #     print(f"  Successfully applied {', '.join(enabled_augs)} to all {batch_seq_len} frames in sequence")
        
        # Save the batch data that was used for model prediction
        # This ensures perfect correspondence between prediction and ground truth
        
        # Run prediction
        with torch.no_grad():
            # torch.FloatTensor: [B, horizon_steps, human_action_dim] Generated human action sequence [1, 30, 48] for each sample
            action_pred_normalized = model("infer_human_action", inputs)
            loss = model("train_flow", inputs)
        print("loss:", loss)
        action_pred_cpu = action_pred_normalized.cpu()
        action_pred_dict = {"human_actions": action_pred_cpu}
        action_pred_unnormalized = normalizer.unnormalize(action_pred_dict)


        background_img = np.array(raw_sample['image'].squeeze(0))         # [H, W, 3]

        extrinsic_w2c = raw_sample['extrinsic'] # [N, 4, 4]
        extrinsic_c2w = invert_extrinsics(extrinsic_w2c)
        intrinsic_4d = raw_sample['intrinsic'][0] # [N, 4]
        intrinsic = torch.tensor([
                    [intrinsic_4d[0], 0, intrinsic_4d[2]],
                    [0, intrinsic_4d[1], intrinsic_4d[3]],
                    [0, 0, 1]
                ])

        if cfg.testing.target_width != 384 or cfg.testing.target_height != 384:
            # Convert from RGB to BGR for OpenCV compatibility (assuming input is RGB)
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


        state_in_camera = torch.zeros_like(raw_sample['state'])
        state_in_camera[:18] = transform_wrist_to_target_frame(raw_sample['state'][:18], extrinsic_w2c[0])
        state_in_camera[18:] = raw_sample['state'][18:]
        print(state_in_camera.squeeze(0).shape, action_pred_unnormalized["human_actions"].squeeze(0).shape)
        action_pred = get_absolute_action(state_in_camera.squeeze(0), action_pred_unnormalized["human_actions"].squeeze(0))

        if cfg.testing.visualize_type == 'mesh':
            mano_shape = raw_sample['action_shape'] # [N, 20]
            action_wrist = raw_sample['action'][:,:18]
            action_hand = raw_sample['action'][:,18:]
            mano_data = sample_to_manovis(action_pred, action_wrist, action_hand, mano_shape, extrinsic_c2w)
            output_dir = f"/data/fanlian/outputs/test_legendvla_workspace/{i}"
       
            # Convert single frame [H, W, 3] to sequence format [N, H, W, 3] for vis_hand_plot
            # Since we have multiple frames in the sequence, we need to repeat the image for each frame
            num_frames = 30
            background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
            image_sequence = np.repeat(background_img[np.newaxis, :, :, :], num_frames, axis=0)  # [N, H, W, 3]
                    
            vis_hand_plot_comparison(mano_data['predicted']['rot'], mano_data['predicted']['trans'], mano_data['predicted']['theta'],mano_data['ground_truth']['rot']
            , mano_data['ground_truth']['trans'], mano_data['ground_truth']['theta'], mano_data['beta'], mano_data['sides'], image_sequence, intrinsic, extrinsic_c2w, output_dir, fps=30)

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
        

