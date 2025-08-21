from manopth.manolayer import ManoLayer
import pickle
import torch
import torch.nn as nn
import numpy as np
import os
import glob
import argparse

# Initialize PCA mode MANO layer (for computing PCA components)
manolayer_pca_right = ManoLayer(
    mano_root='/home/guantianrui/manopth/mano/models', 
    use_pca=True, 
    ncomps=15, 
    flat_hand_mean=True, 
    side='right',
    center_idx=0
)

# Initialize left hand PCA mode MANO layer
manolayer_pca_left = ManoLayer(
    mano_root='/home/guantianrui/manopth/mano/models', 
    use_pca=True, 
    ncomps=15, 
    flat_hand_mean=True, 
    side='left',
    center_idx=0
)

# Load hand_pose data
def load_hand_pose(pkl_path):
    with open(pkl_path, 'rb') as f:
        hand_info = pickle.load(f, encoding='latin1')
    return hand_info

# Convert to MANO hand model (supports left/right hands)
def convert_to_mano(hand_info, hand_side='right'):
    # Select the corresponding MANO layer
    manolayer_pca = manolayer_pca_right if hand_side == 'right' else manolayer_pca_left
    
    # Extract parameters
    pose_coeff = hand_info['poseCoeff']  
    beta = hand_info['beta']             
    trans = hand_info['trans']           
    
    rot = pose_coeff[0:3].copy()
    pose_coeff[0:3] = 0
    # Convert to tensor
    theta = nn.Parameter(torch.FloatTensor(pose_coeff).unsqueeze(0))  # [1, 48]
    # Do not apply global rotation, set the first 3 global rotation parameters to 0
    beta = nn.Parameter(torch.FloatTensor(beta).unsqueeze(0))         # [1, 10]
    trans = nn.Parameter(torch.FloatTensor(trans).unsqueeze(0))       # [1, 3]
    global_rot = nn.Parameter(torch.FloatTensor(rot).unsqueeze(0))       # [1, 3]
    
    # Compute PCA components
    # Extract the first 15 PCA components from 45-dimensional original axis-angle parameters
    joint_params = theta[:, 3:]  # [1, 45] - 45-dimensional original axis-angle parameters
    
    # Convert 45-dimensional axis-angle parameters to 15-dimensional PCA components
    # Use inverse transform of PCA component matrix: axis_angles × PCA_components^T = PCA_coeffs
    pca_components = manolayer_pca.th_selected_comps  # [15, 45] - PCA component matrix
    # print(f"{hand_side} hand PCA components shape: {pca_components.shape}")
    # Compute PCA components: joint_params × pca_components^T
    pca_coeffs = torch.matmul(joint_params, pca_components.t())  # [1, 15]
    
    return {
        'pose_coeff': pca_coeffs.squeeze(0).detach().numpy(),   # Return 15-dimensional PCA components
        'global_rot': global_rot.squeeze(0).detach().numpy(),  # Return 3-dimensional global rotation
        'trans': trans.squeeze(0).detach().numpy(),              # Return 3-dimensional translation
        'beta': beta.squeeze(0).detach().numpy()                 # Return 10-dimensional shape parameters
    }

def process_hoi4d_dataset(data_root, output_root, which_hand):
    """
    Process HOI4D dataset, convert hand data to MANO format
    
    Args:
        data_root: HOI4D dataset root directory
        output_root: Output directory
    """
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    # Find all pkl files (hand pose data)
    pkl_files = glob.glob(os.path.join(data_root, "**/*.pickle"), recursive=True)
    print(f"Found {len(pkl_files)} pickle files")
    
    for pkl_file in pkl_files:
        try:
            # Load hand data
            hand_info = load_hand_pose(pkl_file)
            
            # Check data format
            if 'poseCoeff' in hand_info and 'beta' in hand_info and 'trans' in hand_info:
                result = convert_to_mano(hand_info, which_hand)
                
                # Generate output file path
                rel_path = os.path.relpath(pkl_file, data_root)
                output_path = os.path.join(output_root, rel_path.replace('.pickle', '_mano.npy'))
                
                # Create output directory
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save result
                np.save(output_path, result)
                print(f"Processing completed: {pkl_file} -> {output_path}")
                
            else:
                print(f"Skipping file {pkl_file}: incorrect data format")
                continue
            
        except Exception as e:
            print(f"Error processing file {pkl_file}: {e}")
            continue

if __name__ == '__main__':
    # Set data paths
    parser = argparse.ArgumentParser(description='Convert HOI4D hand pose to 15-dimensional PCA components')
    parser.add_argument('--data_root', type=str, default="/share_data/datasets/hoi4d/Hand_pose/",
                       help='HOI4D dataset root directory')
    parser.add_argument('--output_root', type=str, default="/share_data/datasets/hoi4d/mano_hand_pose/",
                       help='Output directory')
    parser.add_argument('--hand_side', type=str, default='right',
                       choices=['left', 'right', 'both'],
                       help='Hand side to process: left, right, or both')
    
    args = parser.parse_args()
    
    
    # Process dataset
    if args.hand_side == 'right' or args.hand_side == 'both':
        data_root = os.path.join(args.data_root, "handpose_right_hand")
        output_root = os.path.join(args.output_root, "right_hand")
        process_hoi4d_dataset(data_root, output_root, "right")
    if args.hand_side == 'left' or args.hand_side == 'both':
        data_root = os.path.join(args.data_root, "handpose_left_hand")
        output_root = os.path.join(args.output_root, "left_hand")
        process_hoi4d_dataset(data_root, output_root, "left")
    print("Dataset processing completed!")