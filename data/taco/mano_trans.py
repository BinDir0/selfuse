#!/usr/bin/env python3
"""
hand pose data to 15dPCA
support multi-frame data processing, format similar to mano_trans.py
python taco_mano_trans.py --data_root /share_data/datasets/taco/Hand_Poses --output_root /share_data/datasets/taco/Mano_Poses --mano_root /home/guantianrui/manopth/mano/models --multi_level
"""

import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import glob
from pathlib import Path
import argparse
from tqdm import tqdm
from manopth.manolayer import ManoLayer

# Initialize PCA mode MANO layers (for computing PCA components)
manolayer_pca_right = None
manolayer_pca_left = None

def init_mano_layers(mano_root=None):
    """Initialize MANO layers"""
    global manolayer_pca_right, manolayer_pca_left
    
    if mano_root is None:
        mano_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manopth/mano/models')
    
    manolayer_pca_right = ManoLayer(
        mano_root=mano_root, 
        use_pca=True, 
        ncomps=15, 
        flat_hand_mean=True, 
        side='right'
    )
    
    manolayer_pca_left = ManoLayer(
        mano_root=mano_root, 
        use_pca=True, 
        ncomps=15, 
        flat_hand_mean=True, 
        side='left'
    )

# load TACO hand pose data
def load_taco_hand_pose(pkl_path):
    """
    Load TACO format hand pose data
    
    Args:
        pkl_path: TACO hand pose file path
        
    Returns:
        hand_info: Dictionary containing multi-frame hand data
    """
    with open(pkl_path, 'rb') as f:
        hand_info = pickle.load(f)
    return hand_info

# Load TACO hand shape data
def load_taco_hand_shape(shape_path):
    """
    Load TACO format hand shape data
    
    Args:
        shape_path: TACO hand shape file path
        
    Returns:
        hand_shape: 10-dimensional shape parameters
    """
    with open(shape_path, 'rb') as f:
        shape_data = pickle.load(f)
    return shape_data["hand_shape"].reshape(10).detach().cpu().numpy()

# Convert to MANO hand model (supports left/right hands and multi-frame)
def convert_taco_to_mano(hand_info, hand_shape, hand_side='right'):
    """
    Convert TACO hand pose data to MANO format
    
    Args:
        hand_info: TACO format hand data dictionary
        hand_shape: 10-dimensional shape parameters
        hand_side: 'left' or 'right'
        
    Returns:
        result: Dictionary containing multi-frame MANO parameters
    """
    # Select the corresponding MANO layer
    manolayer_pca = manolayer_pca_right if hand_side == 'right' else manolayer_pca_left
    
    # Get all frame keys and sort them
    keys = list(hand_info.keys())
    keys.sort()
    
    # Batch extract data from all frames
    num_frames = len(keys)
    
    # Pre-allocate arrays
    all_pose_coeffs = np.zeros((num_frames, 48))  # [num_frames, 48]
    all_trans = np.zeros((num_frames, 3))         # [num_frames, 3]
    
    # Batch extract data
    for i, frame_key in enumerate(keys):
        frame_data = hand_info[frame_key]
        all_pose_coeffs[i] = frame_data['hand_pose'].detach().cpu().numpy()  # [48]
        all_trans[i] = frame_data['hand_trans'].detach().cpu().numpy()        # [3]
    
    # Separate global rotation and joint parameters
    global_rots = all_pose_coeffs[:, 0:3].copy()  # [num_frames, 3]
    joint_params = all_pose_coeffs[:, 3:]          # [num_frames, 45]
    
    # Convert to tensor for batch processing
    joint_params_tensor = torch.FloatTensor(joint_params)  # [num_frames, 45]
    
    # Batch compute PCA components
    pca_components = manolayer_pca.th_selected_comps  # [15, 45] - PCA component matrix
    pca_components_tensor = torch.FloatTensor(pca_components)  # [15, 45]
    
    # Batch compute PCA components: joint_params × pca_components^T
    pca_coeffs = torch.matmul(joint_params_tensor, pca_components_tensor.t())  # [num_frames, 15]
    
    # Create time series format result
    result = {
        'pose_coeff': pca_coeffs.detach().numpy(),      # [time, 15]
        'global_rot': global_rots,                      # [time, 3]
        'trans': all_trans,                             # [time, 3]
        'beta': hand_shape,                             # [10] - shape parameters
        'num_frames': num_frames,
        'hand_side': hand_side
    }
    
    return result

def process_multi_level_dataset(data_root, output_root, which_hand='right'):
    """
    Process TACO dataset with multi-level folder structure
    
    Args:
        data_root: Dataset root directory
        output_root: Output directory
        which_hand: 'left' or 'right'
    """
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    processed_count = 0
    error_count = 0
    
    # Recursively find all directories containing 4 .pkl files
    def find_sequence_dirs(root_dir):
        """Recursively find sequence directories containing hand data"""
        sequence_dirs = []
        
        for root, dirs, files in os.walk(root_dir):
            # Check if current directory contains the required 4 files
            required_files = [
                f"{which_hand}_hand.pkl",
                f"{which_hand}_hand_shape.pkl"
            ]
            
            if all(os.path.exists(os.path.join(root, f)) for f in required_files):
                sequence_dirs.append(root)
        
        return sequence_dirs
    
    # Find all sequence directories
    print(f"Searching for directories containing {which_hand} hand data...")
    sequence_dirs = find_sequence_dirs(data_root)
    print(f"Found {len(sequence_dirs)} sequence directories")
    
    if len(sequence_dirs) == 0:
        print(f"Warning: No directories containing {which_hand} hand data found in {data_root}")
        return
    
    # Process each sequence directory
    for sequence_dir in tqdm(sequence_dirs, desc=f"Processing {which_hand} hand data"):
        try:
            # Build file paths
            hand_pose_file = os.path.join(sequence_dir, f"{which_hand}_hand.pkl")
            hand_shape_file = os.path.join(sequence_dir, f"{which_hand}_hand_shape.pkl")
            
            # Load data
            hand_info = load_taco_hand_pose(hand_pose_file)
            hand_shape = load_taco_hand_shape(hand_shape_file)
            
            # Convert to MANO format
            result = convert_taco_to_mano(hand_info, hand_shape, which_hand)
            
            # Generate output file path
            # Use relative path to maintain directory structure
            rel_path = os.path.relpath(sequence_dir, data_root)
            safe_path = rel_path.replace('/', '_').replace('\\', '_')
            output_path = os.path.join(output_root, f"{safe_path}_{which_hand}_mano.npy")
            
            # Save result
            np.save(output_path, result)
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} files")
            
        except Exception as e:
            error_count += 1
            print(f"Error processing directory {sequence_dir}: {e}")
            continue
    
    print(f"Processing completed!")
    print(f"  - Successfully processed: {processed_count} files")
    print(f"  - Failed: {error_count} files")

def process_taco_dataset(data_root, output_root, which_hand='right'):
    """
    Process TACO dataset, convert hand data to MANO format
    
    Args:
        data_root: TACO dataset root directory
        output_root: Output directory
        which_hand: 'left' or 'right'
    """
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    # Find all triplet directories
    hand_poses_dir = os.path.join(data_root, "Hand_Poses")
    if not os.path.exists(hand_poses_dir):
        print(f"Error: Cannot find Hand_Poses directory: {hand_poses_dir}")
        return
    
    triplets = [d for d in os.listdir(hand_poses_dir) if os.path.isdir(os.path.join(hand_poses_dir, d))]
    print(f"Found {len(triplets)} triplet directories")
    
    processed_count = 0
    
    for triplet in tqdm(triplets, desc=f"Processing {which_hand} hand data"):
        triplet_dir = os.path.join(hand_poses_dir, triplet)
        sequences = [d for d in os.listdir(triplet_dir) if os.path.isdir(os.path.join(triplet_dir, d))]
        
        for sequence in sequences:
            sequence_dir = os.path.join(triplet_dir, sequence)
            
            # Build file paths - TACO format has 4 files
            hand_pose_file = os.path.join(sequence_dir, f"{which_hand}_hand.pkl")
            hand_shape_file = os.path.join(sequence_dir, f"{which_hand}_hand_shape.pkl")
            
            if os.path.exists(hand_pose_file) and os.path.exists(hand_shape_file):
                try:
                    # Load hand pose data
                    hand_info = load_taco_hand_pose(hand_pose_file)
                    
                    # Load hand shape data
                    hand_shape = load_taco_hand_shape(hand_shape_file)
                    
                    # Convert to MANO format
                    result = convert_taco_to_mano(hand_info, hand_shape, which_hand)
                    
                    # Generate output file path
                    output_path = os.path.join(output_root, f"{triplet}_{sequence}_{which_hand}_mano.npy")
                    
                    # Save result
                    np.save(output_path, result)
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} files")
                    
                except Exception as e:
                    print(f"Error processing file {hand_pose_file}: {e}")
                    continue
    
    print(f"Processing completed! Processed {processed_count} {which_hand} hand files")

def process_single_sequence(sequence_dir, which_hand='right'):
    """
    Process a single TACO sequence directory
    
    Args:
        sequence_dir: Sequence directory path
        which_hand: 'left' or 'right'
        
    Returns:
        result: Converted MANO format data
    """
    # Build file paths
    hand_pose_file = os.path.join(sequence_dir, f"{which_hand}_hand.pkl")
    hand_shape_file = os.path.join(sequence_dir, f"{which_hand}_hand_shape.pkl")
    
    if not os.path.exists(hand_pose_file):
        raise FileNotFoundError(f"Cannot find hand pose file: {hand_pose_file}")
    
    if not os.path.exists(hand_shape_file):
        raise FileNotFoundError(f"Cannot find hand shape file: {hand_shape_file}")
    
    # Load data
    hand_info = load_taco_hand_pose(hand_pose_file)
    hand_shape = load_taco_hand_shape(hand_shape_file)
    
    # Convert to MANO format
    result = convert_taco_to_mano(hand_info, hand_shape, which_hand)
    
    return result

def process_single_file(hand_pose_file, hand_side='right'):
    """
    Process a single TACO hand pose file (compatible with old interface)
    
    Args:
        hand_pose_file: Hand pose file path
        hand_side: 'left' or 'right'
        
    Returns:
        result: Converted MANO format data
    """
    # Infer shape file path from pose file path
    hand_shape_file = hand_pose_file.replace('_hand.pkl', '_hand_shape.pkl')
    
    if not os.path.exists(hand_shape_file):
        raise FileNotFoundError(f"Cannot find corresponding shape file: {hand_shape_file}")
    
    # Load data
    hand_info = load_taco_hand_pose(hand_pose_file)
    hand_shape = load_taco_hand_shape(hand_shape_file)
    
    # Convert to MANO format
    result = convert_taco_to_mano(hand_info, hand_shape, hand_side)
    
    return result

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Convert TACO hand pose to 15-dimensional PCA components')
    parser.add_argument('--data_root', type=str, required=True,
                       help='TACO dataset root directory')
    parser.add_argument('--output_root', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--mano_root', type=str, default=None,
                       help='MANO model path')
    parser.add_argument('--hand_side', type=str, default='both',
                       choices=['left', 'right', 'both'],
                       help='Hand side to process: left, right, or both')
    parser.add_argument('--multi_level', action='store_true',
                       help='Use multi-level folder processing mode (for arbitrary depth folder structures)')
    
    args = parser.parse_args()
    
    # Initialize MANO layers
    try:
        init_mano_layers(args.mano_root)
        
    except Exception as e:
        print(f"Error initializing MANO layers: {e}")
        exit(1)
    
    # Process dataset
    print(f"Starting to process TACO dataset: {args.data_root}")
    print(f"Output directory: {args.output_root}")
    print(f"Hand side to process: {args.hand_side}")
    print(f"Processing mode: {'Multi-level folders' if args.multi_level else 'Standard TACO format'}")
    
    if args.hand_side in ['left', 'right']:
        if args.multi_level:
            process_multi_level_dataset(args.data_root, args.output_root, args.hand_side)
        else:
            process_taco_dataset(args.data_root, args.output_root, args.hand_side)
    elif args.hand_side == 'both':
        if args.multi_level:
            # Process left hand
            left_output_dir = os.path.join(args.output_root, 'left_hand')
            process_multi_level_dataset(args.data_root, left_output_dir, 'left')
            
            # Process right hand
            right_output_dir = os.path.join(args.output_root, 'right_hand')
            process_multi_level_dataset(args.data_root, right_output_dir, 'right')
        else:
            # Process left hand
            left_output_dir = os.path.join(args.output_root, 'left_hand')
            process_taco_dataset(args.data_root, left_output_dir, 'left')
            
            # Process right hand
            right_output_dir = os.path.join(args.output_root, 'right_hand')
            process_taco_dataset(args.data_root, right_output_dir, 'right')
    
    print("Dataset processing completed!")

if __name__ == '__main__':
    main() 