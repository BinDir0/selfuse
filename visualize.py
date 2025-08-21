import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as patches
from typing import Tuple, List, Optional
import argparse

# manopth.manolayer will be imported dynamically in main()

# Add transformation utilities (moved to HandVisualizer class)

class HandVisualizer:
    """
    Hand motion visualizer for inference with direct tensor inputs
    """
    
    def __init__(self, mano_root: str = None):
        """
        Initialize the hand visualizer for inference
        
        Args:
            mano_root: Path to MANO model files
        """
        if mano_root is None:
            mano_root = '../manopth/mano/models'  # Default fallback
        self.mano_root = mano_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Import ManoLayer dynamically
        from manopth.manolayer import ManoLayer
        
        # Initialize MANO layers for both hands
        self.mano_left = ManoLayer(
            mano_root=mano_root,
            side='left',
            use_pca=True,
            ncomps=15,  # PCA components
            flat_hand_mean=True,
            center_idx=0
        ).to(self.device)
        
        self.mano_right = ManoLayer(
            mano_root=mano_root,
            side='right', 
            use_pca=True,
            ncomps=15,  # PCA components
            flat_hand_mean=True,
            center_idx=0
        ).to(self.device)
        
        # Hand joint connections (parent-child relationships)
        # MANO joint order: 0-wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky
        self.joint_connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger  
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        # Colors for left and right hands - light sky blue for joint connections (BGR format)
        self.left_color = (255, 150, 0)     # Light sky blue for left hand joints
        self.right_color = (255, 150, 0)    # Light sky blue for right hand joints
        
        # Get MANO faces for mesh rendering
        self.faces = self.mano_left.th_faces.cpu().numpy().astype(np.int32)
        
    def decode_mano_params(self, pca_params: torch.Tensor, global_rotation: torch.Tensor, 
                          translation: torch.Tensor, is_left: bool = True, return_verts: bool = False):
        """
        Decode MANO PCA parameters to get hand joints and optionally vertices in world coordinates
        
        Args:
            pca_params: PCA parameters [batch_size, 15]
            global_rotation: Global rotation in axis-angle format [batch_size, 3]
            translation: Translation [batch_size, 3]
            is_left: Whether this is left hand
            return_verts: Whether to return vertices as well
            
        Returns:
            joints: Hand joints [batch_size, 21, 3] in world coordinates
            verts: (optional) Hand vertices [batch_size, 778, 3] in world coordinates
        """
        batch_size = pca_params.shape[0]
        
        # Combine global rotation and PCA parameters
        # MANO expects [batch_size, ncomps + 3] where first 3 are global rotation
        full_pose = torch.zeros(batch_size, 18, device=self.device)
        full_pose[:, :3] = global_rotation  # Global rotation (axis-angle)
        full_pose[:, 3:] = pca_params  # PCA components
        
        # Get MANO layer
        mano_layer = self.mano_left if is_left else self.mano_right
        
        # Forward pass through MANO
        verts, joints = mano_layer(full_pose)
        
        # Convert from millimeters to meters
        joints = joints / 1000.0
        verts = verts / 1000.0
        
        # Apply translation to get world coordinates
        # print("joints:", joints)
        # print("translation:", translation)
        joints = joints + translation.unsqueeze(1)
        verts = verts + translation.unsqueeze(1)
        
        if return_verts:
            return joints, verts
        else:
            return joints
    
    def rot6d_to_rotmat(self, rot6d: torch.Tensor) -> torch.Tensor:
        '''
        Convert 6D rotation representation to 3x3 rotation matrix.
    
        The 6D representation uses two 3D vectors a and b, where:
        - The first vector a represents the first column of the rotation matrix
        - The second vector b represents the second column of the rotation matrix
        - The third column is computed as the cross product of a and b, then normalized
        
        Args:
            rot: torch.Tensor or np.ndarray, shape: [..., 6] or [6], where the first 3 elements are vector a,
                and the last 3 elements are vector b. Supports arbitrary dimensions.
        Returns:
            rot_matrix: torch.Tensor or np.ndarray, shape: [..., 3, 3] or [3, 3]
        '''
        if isinstance(rot6d, np.ndarray):
            is_numpy = True
            rot6d = torch.from_numpy(rot6d)
        else:
            is_numpy = False
    
        # Store original shape for later restoration
        original_shape = rot6d.shape
    
        # Handle single vector case (shape: [6])
        # Reshape to 2D for easier processing: [..., 6] -> [N, 6]
        rot6d = rot6d.reshape(-1, 6)
    
        # Extract the two 3D vectors
        a = rot6d[..., :3]  # First 3 elements: [N, 3]
        b = rot6d[..., 3:]  # Last 3 elements: [N, 3]
    
        # Schmidth orthogonalization
        a = F.normalize(a, dim=-1)
        b = b - torch.sum(a * b, dim=-1, keepdim=True) * a
        b = F.normalize(b, dim=-1)
        c = torch.cross(a, b, dim=-1)
    
        # Stack to form the rotation matrix
        rot_matrix = torch.stack([a, b, c], dim=-1)  # [N, 3, 3]
    
        # Reshape back to original dimensions if needed
        if len(original_shape) > 1:
            # Remove the last dimension (6) and add [3, 3] at the end
            new_shape = list(original_shape[:-1]) + [3, 3]
        rot_matrix = rot_matrix.reshape(*new_shape)

        if is_numpy:
            rot_matrix = rot_matrix.numpy()
    
        return rot_matrix
    
    def rotmat_to_rot6d(self, rot_matrix):
        '''
        Convert 3x3 rotation matrix to 6D rotation representation.
        
        Args:
            rot_matrix: torch.Tensor or np.ndarray, shape: [..., 3, 3] or [3, 3]. Supports arbitrary dimensions.
        Returns:
            rot_6d: torch.Tensor or np.ndarray, shape: [..., 6] or [6]
        '''
        if isinstance(rot_matrix, np.ndarray):
            is_numpy = True
            rot_matrix = torch.from_numpy(rot_matrix)
        else:
            is_numpy = False
        
        # Store original shape for later restoration
        original_shape = rot_matrix.shape
        
        # Handle single rotation matrix case (shape: [3, 3])
        # Reshape to 3D for easier processing: [..., 3, 3] -> [N, 3, 3]
        rot_matrix = rot_matrix.reshape(-1, 3, 3)
        
        # Extract the first two columns
        a = rot_matrix[..., :, 0]  # First column: [N, 3]
        b = rot_matrix[..., :, 1]  # Second column: [N, 3]
        
        # Concatenate to form 6D representation
        rot_6d = torch.cat([a, b], dim=-1)  # [N, 6]
        
        # Reshape back to original dimensions if needed
        if len(original_shape) > 2:
            # Remove the last two dimensions (3, 3) and add [6] at the end
            new_shape = list(original_shape[:-2]) + [6]
            rot_6d = rot_6d.reshape(*new_shape)
        else:
            rot_6d = rot_6d.squeeze(0)
        
        if is_numpy:
            rot_6d = rot_6d.numpy()
        
        return rot_6d

    def rotmat_to_axisangle(self, R: torch.Tensor) -> torch.Tensor:
        """
        Convert rotation matrix to axis-angle representation
        
        Args:
            R: Rotation matrix [batch_size, 3, 3]
            
        Returns:
            axis_angle: Axis-angle representation [batch_size, 3]
        """
        batch_size = R.shape[0]
        axis_angles = []
        
        for i in range(batch_size):
            R_np = R[i].cpu().numpy().astype(np.float32)
            cos_theta = (np.trace(R_np) - 1.0) / 2.0
            cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
            theta = np.arccos(cos_theta)
            
            if theta < 1e-8:
                axis_angle = np.zeros((3,), dtype=np.float32)
            else:
                rx = R_np[2, 1] - R_np[1, 2]
                ry = R_np[0, 2] - R_np[2, 0]
                rz = R_np[1, 0] - R_np[0, 1]
                axis = np.array([rx, ry, rz], dtype=np.float32)
                axis = axis / (np.linalg.norm(axis) + 1e-8)
                axis_angle = (axis * theta).astype(np.float32)
            
            axis_angles.append(axis_angle)
        
        return torch.from_numpy(np.stack(axis_angles)).to(R.device)
    
    def draw_hand_skeleton(self, img: np.ndarray, joints_2d: np.ndarray, 
                          color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
        """
        Draw hand skeleton on image
        
        Args:
            img: Input image
            joints_2d: 2D joint coordinates [21, 2] or None
            color: Line color (B, G, R)
            thickness: Line thickness
            
        Returns:
            img_with_skeleton: Image with skeleton drawn
        """
        if joints_2d is None:
            return img
            
        img_copy = img.copy()
        
        # Draw connections
        for parent_idx, child_idx in self.joint_connections:
            parent_pos = tuple(joints_2d[parent_idx].astype(int))
            child_pos = tuple(joints_2d[child_idx].astype(int))
            
            # Check if points are within image bounds
            h, w = img.shape[:2]
            if (0 <= parent_pos[0] < w and 0 <= parent_pos[1] < h and
                0 <= child_pos[0] < w and 0 <= child_pos[1] < h):
                cv2.line(img_copy, parent_pos, child_pos, color, thickness)
        
        # Draw joints as circles
        for joint_pos in joints_2d:
            pos = tuple(joint_pos.astype(int))
            if 0 <= pos[0] < w and 0 <= pos[1] < h:
                cv2.circle(img_copy, pos, 3, color, -1)
        
        return img_copy
    
    def draw_hand_mesh(self, img: np.ndarray, vertices_2d: np.ndarray, vertices_3d: np.ndarray,
                      color: Tuple[int, int, int], alpha: float = 0.3) -> np.ndarray:
        """
        Draw hand mesh (faces) on image using 2D projected vertices
        
        Args:
            img: Input image
            vertices_2d: 2D vertex coordinates [N, 2]
            vertices_3d: 3D vertex coordinates [N, 3] (for depth sorting)
            color: Fill color (B, G, R)
            alpha: Transparency (0.0 to 1.0)
            
        Returns:
            img_with_mesh: Image with mesh drawn
        """
        img_copy = img.copy()
        h, w = img.shape[:2]
        
        # Create overlay for transparency
        overlay = img_copy.copy()
        
        # Sort faces by depth (z-coordinate) for proper rendering
        face_depths = []
        valid_faces = []
        
        for face in self.faces:
            # Check if all vertices of the face are valid
            if (face[0] < len(vertices_2d) and face[1] < len(vertices_2d) and face[2] < len(vertices_2d)):
                # Get 2D coordinates
                v0_2d = vertices_2d[face[0]]
                v1_2d = vertices_2d[face[1]]
                v2_2d = vertices_2d[face[2]]
                
                # Check if all vertices are within image bounds
                if (0 <= v0_2d[0] < w and 0 <= v0_2d[1] < h and
                    0 <= v1_2d[0] < w and 0 <= v1_2d[1] < h and
                    0 <= v2_2d[0] < w and 0 <= v2_2d[1] < h):
                    
                    # Calculate average depth for sorting
                    avg_depth = (vertices_3d[face[0], 2] + vertices_3d[face[1], 2] + vertices_3d[face[2], 2]) / 3
                    face_depths.append(avg_depth)
                    valid_faces.append(face)
        
        # Sort faces by depth (furthest first)
        if len(valid_faces) > 0:
            sorted_indices = np.argsort(face_depths)[::-1]  # Sort in descending order
            
            for idx in sorted_indices:
                face = valid_faces[idx]
                
                # Get 2D triangle vertices
                triangle_2d = np.array([
                    vertices_2d[face[0]].astype(np.int32),
                    vertices_2d[face[1]].astype(np.int32),
                    vertices_2d[face[2]].astype(np.int32)
                ], dtype=np.int32)
                
                # Fill triangle
                cv2.fillPoly(overlay, [triangle_2d], color)
        
        # Blend with original image
        img_with_mesh = cv2.addWeighted(img_copy, 1 - alpha, overlay, alpha, 0)
        
        return img_with_mesh
    
    def project_3d_to_2d(self, joints_3d: torch.Tensor, img_size: Tuple[int, int] = (640, 480),
                        fx: float = None, fy: float = None, cx: float = None, cy: float = None,
                        extrinsic_matrix: torch.Tensor = None) -> np.ndarray:
        """
        Project 3D joints to 2D image coordinates using camera intrinsics and extrinsics
        
        Args:
            joints_3d: 3D joints [21, 3] in world coordinates
            img_size: Image size (width, height)
            fx, fy: Focal lengths (if None, use default based on image size)
            cx, cy: Principal point (if None, use image center)
            extrinsic_matrix: 4x4 world-to-camera transformation matrix (if None, assume joints are already in camera frame)
            
        Returns:
            joints_2d: 2D joint coordinates [21, 2]
        """
        # Set default camera intrinsics
        W, H = img_size
        if fx is None or fy is None:
            f = max(W, H)
            fx = fy = f
        if cx is None:
            cx = W / 2.0
        if cy is None:
            cy = H / 2.0
        
        # Convert to numpy for processing
        joints_3d_np = joints_3d.cpu().numpy()
        
        # Apply extrinsic transformation if provided (world -> camera coordinates)
        if extrinsic_matrix is not None:
            extrinsic_np = extrinsic_matrix.cpu().numpy()
            R_wc = extrinsic_np[:3, :3]
            t_wc = extrinsic_np[:3, 3]
            
            # Transform to camera coordinates: P_camera = R_wc * P_world + t_wc
            joints_cam = (R_wc @ joints_3d_np.T).T + t_wc.reshape(1, 3)
        else:
            joints_cam = joints_3d_np
        # print("joints_cam:", joints_cam)
        
        # Apply perspective projection with intrinsic parameters
        x_cam = joints_cam[:, 0]
        y_cam = joints_cam[:, 1]
        z_cam = joints_cam[:, 2]
        
        # Avoid division by zero and ensure positive depth
        z_cam = np.clip(z_cam, 1e-6, None)
        
        # Project to image plane using camera intrinsics
        u = fx * (x_cam / z_cam) + cx
        v = fy * (y_cam / z_cam) + cy
        
        joints_2d = np.stack([u, v], axis=1)
        return joints_2d
    
    def transform_wrist_from_camera_to_world(self, wrist_action, camera_extrinsic):
        '''
        Transform the wrist action from camera frame to world frame.
        This is the inverse of transform_wrist_to_target_frame.
        
        Args:
            wrist_action: torch.Tensor, shape: [18] or [B, 18] - wrist data in camera frame
            camera_extrinsic: torch.Tensor, shape: [4, 4] or [B, 4, 4] - camera extrinsic matrix
        Returns:
            wrist_action: torch.Tensor, shape: [18] or [B, 18] - wrist data in world frame
        '''
        if isinstance(wrist_action, np.ndarray):
            is_numpy = True
            wrist_action = torch.from_numpy(wrist_action)
            camera_extrinsic = torch.from_numpy(camera_extrinsic)
        else:
            is_numpy = False

        # Handle single sample case
        single_sample = False
        if len(wrist_action.shape) == 1:
            wrist_action = wrist_action.unsqueeze(0)
            single_sample = True
        
        if len(camera_extrinsic.shape) == 2:
            camera_extrinsic = camera_extrinsic.unsqueeze(0)

        B = wrist_action.shape[0]

        # Extract components from 18-dim wrist action
        # [0:3] left wrist translation, [3:6] right wrist translation
        # [6:12] left wrist 6D rotation, [12:18] right wrist 6D rotation
        left_trans = wrist_action[..., :3]      # [B, 3]
        right_trans = wrist_action[..., 3:6]    # [B, 3]
        left_rot6d = wrist_action[..., 6:12]    # [B, 6]
        right_rot6d = wrist_action[..., 12:18]  # [B, 6]
        
        # Convert 6D rotations to rotation matrices
        left_rotmat = self.rot6d_to_rotmat(left_rot6d)   # [B, 3, 3]
        right_rotmat = self.rot6d_to_rotmat(right_rot6d) # [B, 3, 3]
        
        # Create 4x4 pose matrices for left and right wrists
        left_pose = torch.zeros(B, 4, 4, device=wrist_action.device, dtype=wrist_action.dtype)
        right_pose = torch.zeros(B, 4, 4, device=wrist_action.device, dtype=wrist_action.dtype)
        
        left_pose[:, :3, :3] = left_rotmat
        left_pose[:, :3, 3] = left_trans
        left_pose[:, 3, 3] = 1.0
        
        right_pose[:, :3, :3] = right_rotmat
        right_pose[:, :3, 3] = right_trans
        right_pose[:, 3, 3] = 1.0
        
        # Stack poses for batch processing
        wrist_poses = torch.stack([left_pose, right_pose], dim=1)  # [B, 2, 4, 4]
        wrist_poses = wrist_poses.view(B * 2, 4, 4)  # [B*2, 4, 4]
        
        # Transform from camera to world coordinates: P_world = T_world_camera × P_camera
        # camera_extrinsic represents T_world_camera (camera pose in world frame)
        camera_extrinsic_expanded = camera_extrinsic.unsqueeze(1).repeat(1, 2, 1, 1).view(B * 2, 4, 4)
        world_poses = torch.matmul(camera_extrinsic_expanded, wrist_poses)  # [B*2, 4, 4]
        
        # Reshape back
        world_poses = world_poses.view(B, 2, 4, 4)  # [B, 2, 4, 4]
        left_pose_world = world_poses[:, 0]  # [B, 4, 4]
        right_pose_world = world_poses[:, 1] # [B, 4, 4]
        
        # Extract transformed components
        left_trans_world = left_pose_world[:, :3, 3]    # [B, 3]
        right_trans_world = right_pose_world[:, :3, 3]  # [B, 3]
        left_rotmat_world = left_pose_world[:, :3, :3]  # [B, 3, 3]
        right_rotmat_world = right_pose_world[:, :3, :3] # [B, 3, 3]
        
        # Convert rotation matrices back to 6D representation
        left_rot6d_world = self.rotmat_to_rot6d(left_rotmat_world)   # [B, 6]
        right_rot6d_world = self.rotmat_to_rot6d(right_rotmat_world) # [B, 6]
        
        # Reconstruct 18-dim wrist action
        wrist_action_world = torch.cat([
            left_trans_world,      # [B, 3]
            right_trans_world,     # [B, 3]
            left_rot6d_world,      # [B, 6]
            right_rot6d_world      # [B, 6]
        ], dim=-1)  # [B, 18]
        
        # Handle single sample case
        if single_sample:
            wrist_action_world = wrist_action_world.squeeze(0)
        
        if is_numpy:
            wrist_action_world = wrist_action_world.numpy()
        
        return wrist_action_world
    
    def visualize_frame(self, frame_img: np.ndarray, mano_params: torch.Tensor, 
                       wrist_params: torch.Tensor, fx: float = None, fy: float = None,
                       cx: float = None, cy: float = None, extrinsic_matrix: torch.Tensor = None,
                       show_mesh: bool = False, mesh_alpha: float = 1.0,
                       gt_mano_params: torch.Tensor = None, gt_wrist_params: torch.Tensor = None,
                       presence: int = 3) -> np.ndarray:
        """
        Visualize a single frame with hand predictions and optionally ground truth
        
        Args:
            frame_img: Current frame image
            mano_params: MANO PCA parameters [30] (15 left + 15 right) for prediction
            wrist_params: Wrist parameters [18] (3+3 translation + 6+6 rotation) in camera frame for prediction
            fx, fy: Camera focal lengths (optional)
            cx, cy: Camera principal point (optional)
            extrinsic_matrix: 4x4 world-to-camera transformation matrix (optional)
            show_mesh: Whether to show hand mesh (faces) (optional)
            mesh_alpha: Transparency of mesh (0.0 to 1.0) (optional)
            gt_mano_params: Ground truth MANO PCA parameters [30] (optional)
            gt_wrist_params: Ground truth wrist parameters [18] (optional)
            presence: Hand visibility flag (1=left only, 2=right only, 3=both visible) (optional)
            
        Returns:
            vis_img: Visualization image with prediction (and optionally ground truth)
        """
        # Transform wrist parameters from camera frame to world frame if extrinsic matrix is available
        if extrinsic_matrix is not None:
            wrist_params_world = self.transform_wrist_from_camera_to_world(wrist_params, extrinsic_matrix)
            # Transform ground truth wrist parameters if provided
            if gt_wrist_params is not None:
                gt_wrist_params_world = self.transform_wrist_from_camera_to_world(gt_wrist_params, extrinsic_matrix)
            else:
                gt_wrist_params_world = None
        else:
            print("No extrinsic matrix provided, assuming wrist parameters are already in world frame")
            wrist_params_world = wrist_params
            gt_wrist_params_world = gt_wrist_params
        
        # Split parameters
        left_pca = mano_params[:15].unsqueeze(0)  # [1, 15]
        right_pca = mano_params[15:].unsqueeze(0)  # [1, 15]
        
        left_trans = wrist_params_world[:3].unsqueeze(0)  # [1, 3]
        right_trans = wrist_params_world[3:6].unsqueeze(0)  # [1, 3]
        left_rot6d = wrist_params_world[6:12].unsqueeze(0)  # [1, 6]
        right_rot6d = wrist_params_world[12:18].unsqueeze(0)  # [1, 6]
        
        # Convert rot6d to rotation matrices, then to axis-angle
        left_rotmat = self.rot6d_to_rotmat(left_rot6d)  # [1, 3, 3]
        right_rotmat = self.rot6d_to_rotmat(right_rot6d)  # [1, 3, 3]
        
        # Convert rotation matrices to axis-angle representation
        left_axis_angle = self.rotmat_to_axisangle(left_rotmat)  # [1, 3]
        right_axis_angle = self.rotmat_to_axisangle(right_rotmat)  # [1, 3]
        # print("left_axis_angle:", left_axis_angle)
        # print("right_axis_angle:", right_axis_angle)
        # print("left_trans:", left_trans)
        # print("right_trans:", right_trans)
        
        # Determine which hands to visualize based on presence value
        # 1: left hand only, 2: right hand only, 3: both hands
        show_left_hand = presence in [1, 3]
        show_right_hand = presence in [2, 3]
        
        # Decode MANO parameters with proper global rotation and translation (only for visible hands)
        left_joints_world = None
        right_joints_world = None
        left_verts_world = None 
        right_verts_world = None
        
        if show_left_hand:
            if show_mesh:
                left_joints_world, left_verts_world = self.decode_mano_params(left_pca, left_axis_angle, left_trans, is_left=True, return_verts=True)  # [1, 21, 3], [1, 778, 3]
            else:
                left_joints_world = self.decode_mano_params(left_pca, left_axis_angle, left_trans, is_left=True)  # [1, 21, 3]
        
        if show_right_hand:
            if show_mesh:
                right_joints_world, right_verts_world = self.decode_mano_params(right_pca, right_axis_angle, right_trans, is_left=False, return_verts=True)  # [1, 21, 3], [1, 778, 3]
            else:
                right_joints_world = self.decode_mano_params(right_pca, right_axis_angle, right_trans, is_left=False)  # [1, 21, 3]
        
        # Project to 2D using improved projection method (only for visible hands)
        img_size = (frame_img.shape[1], frame_img.shape[0])
        left_joints_2d = None
        right_joints_2d = None
        
        if show_left_hand and left_joints_world is not None:
            left_joints_2d = self.project_3d_to_2d(left_joints_world[0], img_size, 
                                                  fx=fx, fy=fy, cx=cx, cy=cy, 
                                                  extrinsic_matrix=extrinsic_matrix)
        
        if show_right_hand and right_joints_world is not None:
            right_joints_2d = self.project_3d_to_2d(right_joints_world[0], img_size,
                                                   fx=fx, fy=fy, cx=cx, cy=cy,
                                                   extrinsic_matrix=extrinsic_matrix)
        # Draw visualization layers
        vis_img = frame_img.copy()
        
        # Layer 1: Draw ground truth first (if provided) - in green
        if gt_mano_params is not None and gt_wrist_params_world is not None:
            # Split ground truth parameters
            gt_left_pca = gt_mano_params[:15].unsqueeze(0)
            gt_right_pca = gt_mano_params[15:].unsqueeze(0)
            
            gt_left_trans = gt_wrist_params_world[:3].unsqueeze(0)
            gt_right_trans = gt_wrist_params_world[3:6].unsqueeze(0)
            gt_left_rot6d = gt_wrist_params_world[6:12].unsqueeze(0)
            gt_right_rot6d = gt_wrist_params_world[12:18].unsqueeze(0)
            
            # Convert ground truth rot6d to rotation matrices, then to axis-angle
            gt_left_rotmat = self.rot6d_to_rotmat(gt_left_rot6d)
            gt_right_rotmat = self.rot6d_to_rotmat(gt_right_rot6d)
            
            gt_left_axis_angle = self.rotmat_to_axisangle(gt_left_rotmat)
            gt_right_axis_angle = self.rotmat_to_axisangle(gt_right_rotmat)
            
            # Decode ground truth MANO parameters (only for visible hands)
            gt_left_joints_world = None
            gt_right_joints_world = None
            gt_left_verts_world = None
            gt_right_verts_world = None
            
            if show_left_hand:
                if show_mesh:
                    gt_left_joints_world, gt_left_verts_world = self.decode_mano_params(gt_left_pca, gt_left_axis_angle, gt_left_trans, is_left=True, return_verts=True)
                else:
                    gt_left_joints_world = self.decode_mano_params(gt_left_pca, gt_left_axis_angle, gt_left_trans, is_left=True)
            
            if show_right_hand:
                if show_mesh:
                    gt_right_joints_world, gt_right_verts_world = self.decode_mano_params(gt_right_pca, gt_right_axis_angle, gt_right_trans, is_left=False, return_verts=True)
                else:
                    gt_right_joints_world = self.decode_mano_params(gt_right_pca, gt_right_axis_angle, gt_right_trans, is_left=False)
            
            # Project ground truth to 2D (only for visible hands)
            gt_left_joints_2d = None
            gt_right_joints_2d = None
            
            if show_left_hand and gt_left_joints_world is not None:
                gt_left_joints_2d = self.project_3d_to_2d(gt_left_joints_world[0], img_size, 
                                                          fx=fx, fy=fy, cx=cx, cy=cy, 
                                                          extrinsic_matrix=extrinsic_matrix)
            
            if show_right_hand and gt_right_joints_world is not None:
                gt_right_joints_2d = self.project_3d_to_2d(gt_right_joints_world[0], img_size,
                                                           fx=fx, fy=fy, cx=cx, cy=cy,
                                                           extrinsic_matrix=extrinsic_matrix)
            
            # Draw ground truth mesh (green) - only for visible hands
            if show_mesh:
                # Light grass green mesh colors (BGR format)
                gt_mesh_color = (200, 255, 200)  # Very light grass green for ground truth mesh
                
                if show_left_hand and gt_left_verts_world is not None:
                    gt_left_verts_2d = self.project_3d_to_2d(gt_left_verts_world[0], img_size, 
                                                             fx=fx, fy=fy, cx=cx, cy=cy, 
                                                             extrinsic_matrix=extrinsic_matrix)
                    vis_img = self.draw_hand_mesh(vis_img, gt_left_verts_2d, gt_left_verts_world[0].cpu().numpy(), 
                                                gt_mesh_color, mesh_alpha)
                
                if show_right_hand and gt_right_verts_world is not None:
                    gt_right_verts_2d = self.project_3d_to_2d(gt_right_verts_world[0], img_size,
                                                              fx=fx, fy=fy, cx=cx, cy=cy,
                                                              extrinsic_matrix=extrinsic_matrix)
                    vis_img = self.draw_hand_mesh(vis_img, gt_right_verts_2d, gt_right_verts_world[0].cpu().numpy(), 
                                                gt_mesh_color, mesh_alpha)
            
            # Draw ground truth skeleton (light grass green, thicker lines) - only for visible hands
            gt_color = (50, 200, 50)  # Light grass green for ground truth skeleton
            if show_left_hand and gt_left_joints_2d is not None:
                vis_img = self.draw_hand_skeleton(vis_img, gt_left_joints_2d, gt_color, thickness=1)
            if show_right_hand and gt_right_joints_2d is not None:
                vis_img = self.draw_hand_skeleton(vis_img, gt_right_joints_2d, gt_color, thickness=1)
        
        # Layer 2: Draw prediction mesh (if enabled) - only for visible hands
        if show_mesh:
            # Mesh colors for prediction
            left_mesh_color = (250, 240, 220)   # Light blue for prediction mesh
            right_mesh_color = (250, 240, 220)  # Light blue for prediction mesh
            
            if show_left_hand and left_verts_world is not None:
                left_verts_2d = self.project_3d_to_2d(left_verts_world[0], img_size, 
                                                     fx=fx, fy=fy, cx=cx, cy=cy, 
                                                     extrinsic_matrix=extrinsic_matrix)
                vis_img = self.draw_hand_mesh(vis_img, left_verts_2d, left_verts_world[0].cpu().numpy(), 
                                            left_mesh_color, mesh_alpha)
            
            if show_right_hand and right_verts_world is not None:
                right_verts_2d = self.project_3d_to_2d(right_verts_world[0], img_size,
                                                      fx=fx, fy=fy, cx=cx, cy=cy,
                                                      extrinsic_matrix=extrinsic_matrix)
                vis_img = self.draw_hand_mesh(vis_img, right_verts_2d, right_verts_world[0].cpu().numpy(), 
                                            right_mesh_color, mesh_alpha)
        
        # Layer 3: Draw prediction skeletons on top - only for visible hands
        if show_left_hand and left_joints_2d is not None:
            vis_img = self.draw_hand_skeleton(vis_img, left_joints_2d, self.left_color, thickness=1)
        if show_right_hand and right_joints_2d is not None:
            vis_img = self.draw_hand_skeleton(vis_img, right_joints_2d, self.right_color, thickness=1)
        
        return vis_img
    
    def generate_video_with_predictions(self, background_img: np.ndarray, 
                                      mano_sequence: torch.Tensor, wrist_sequence: torch.Tensor,
                                      output_path: str, fps: int = 30, fx: float = None, fy: float = None,
                                      cx: float = None, cy: float = None, extrinsic_sequence: torch.Tensor = None,
                                      show_mesh: bool = False, mesh_alpha: float = 1.0,
                                      gt_mano_sequence: torch.Tensor = None, gt_wrist_sequence: torch.Tensor = None,
                                      presence: int = 3) -> None:
        """
        Generate video where each frame shows hand motion prediction overlay on static background image
        
        Args:
            background_img: Static background image to use for all frames
            mano_sequence: MANO parameters sequence [n, 30] where n is the number of future frames
            wrist_sequence: Wrist parameters sequence [n, 18] where n is the number of future frames
            output_path: Output video path
            fps: Video frame rate
            fx, fy: Camera focal lengths (optional)
            cx, cy: Camera principal point (optional)
            extrinsic_sequence: Sequence of 4x4 extrinsic matrices [n, 4, 4] (optional)
            show_mesh: Whether to show hand mesh (faces) (optional)
            mesh_alpha: Transparency of mesh (0.0 to 1.0) (optional)
            gt_mano_sequence: Ground truth MANO parameters sequence [n, 30] (optional)
            gt_wrist_sequence: Ground truth wrist parameters sequence [n, 18] (optional)
            presence: Hand visibility flag (1=left only, 2=right only, 3=both visible) (optional)
        """
        # Get video properties from background image
        height, width = background_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create video writer
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        num_frames = len(mano_sequence)
        print(f"Generating video with static background image...")
        print(f"Hand motion prediction will be overlaid on the same background for {num_frames} frames")
        
        # Generate frames using static background and prediction sequence
        for i in range(num_frames):
            print(f"Processing frame {i+1}/{num_frames}")
            
            # Get extrinsic matrix for this frame if available
            extrinsic_matrix = extrinsic_sequence[i] if extrinsic_sequence is not None else None
            
            # Get ground truth for this frame if available
            gt_mano_frame = gt_mano_sequence[i] if gt_mano_sequence is not None else None
            gt_wrist_frame = gt_wrist_sequence[i] if gt_wrist_sequence is not None else None
            
            # Create visualization by overlaying prediction on static background image
            vis_frame = self.visualize_frame(background_img, mano_sequence[i], wrist_sequence[i],
                                           fx=fx, fy=fy, cx=cx, cy=cy, extrinsic_matrix=extrinsic_matrix,
                                           show_mesh=show_mesh, mesh_alpha=mesh_alpha,
                                           gt_mano_params=gt_mano_frame, gt_wrist_params=gt_wrist_frame,
                                           presence=presence)
            
            # Write frame to video
            video_writer.write(vis_frame)
        
        # Release video writer
        video_writer.release()
        print(f"Video with predictions saved to: {output_path}")

def find_worst_sample(path: str) -> int:
    """
    Find the sample with the highest loss from all samples in the data file
    
    Args:
        path: Path to egovla_predictions_complete.pt file
        
    Returns:
        worst_sample_id: Index of sample with highest loss
    """
    try:
        # Load complete data
        data = torch.load(path, map_location='cpu')
        
        if 'predictions' not in data or 'metadata' not in data:
            print(f"Expected data to contain 'predictions' and 'metadata' keys")
            return 0
        
        predictions = data['predictions']
        
        if not isinstance(predictions, list) or len(predictions) == 0:
            print(f"Expected predictions to be a list with at least 1 sample")
            return 0
        
        print(f"Searching through {len(predictions)} samples for the one with highest loss...")
        
        worst_sample_id = 0
        worst_loss = float('-inf')  # Start with negative infinity to find maximum
        
        for i, sample_data in enumerate(predictions):
            if 'loss' in sample_data:
                loss = sample_data['loss']
                # Handle different loss formats
                if isinstance(loss, torch.Tensor):
                    loss_value = loss.item()
                elif isinstance(loss, (list, tuple)) and len(loss) > 0:
                    loss_value = loss[0] if isinstance(loss[0], (int, float)) else loss[0].item()
                else:
                    loss_value = float(loss)
                
                print(f"Sample {i}: loss = {loss_value:.6f}")
                
                if loss_value > worst_loss:  # Find maximum instead of minimum
                    worst_loss = loss_value
                    worst_sample_id = i
            else:
                print(f"Sample {i}: no loss found")
        
        print(f"\nWorst sample found:")
        print(f"  Sample ID: {worst_sample_id}")
        print(f"  Loss: {worst_loss:.6f}")
        
        return worst_sample_id
        
    except Exception as e:
        print(f"Failed to find worst sample from {path}: {e}")
        import traceback
        traceback.print_exc()
        return 0

def load_complete_data(path: str, sample_id: int = 0) -> tuple:
    """
    Load complete data from egovla_predictions_complete.pt file
    
    Args:
        path: Path to egovla_predictions_complete.pt file
        sample_idx: Index of sample to load (default: 0)
        
    Returns:
        (mano_sequence, wrist_sequence, gt_mano_sequence, gt_wrist_sequence, intrinsic_matrix, extrinsic_matrices, background_image, presence) 
        or (None, None, None, None, None, None, None, None) if failed to load
    """
    try:
        # Load complete data
        data = torch.load(path, map_location='cpu')
        
        if 'predictions' not in data or 'metadata' not in data:
            print(f"Expected data to contain 'predictions' and 'metadata' keys")
            print(f"Available keys: {list(data.keys())}")
            return None, None, None, None, None, None, None
        
        predictions = data['predictions']
        metadata = data['metadata']
        
        if not isinstance(predictions, list) or len(predictions) == 0:
            print(f"Expected predictions to be a list with at least 1 sample, got {type(predictions)}")
            return None, None, None, None, None, None, None
        
        if sample_id >= len(predictions):
            print(f"Sample index {sample_id} out of range, available samples: {len(predictions)}")
            return None, None, None, None, None, None, None
        
        sample_data = predictions[sample_id]
        print(f"Loaded complete data file with {len(predictions)} samples, using sample {sample_id}")
        print(f"Sample keys: {list(sample_data.keys())}")
        print(sample_data['episode_info']['current_frame_idx'])
        print("loss:", sample_data['loss'])
        # Extract action prediction data
        if 'action_pred' not in sample_data:
            print(f"Key 'action_pred' not found in sample data")
            return None, None, None, None, None, None, None
        
        action_pred = sample_data['action_pred']
        if 'hand' not in action_pred or 'wrist' not in action_pred:
            print(f"Keys 'hand' or 'wrist' not found in action_pred")
            print(f"Available keys in action_pred: {list(action_pred.keys())}")
            return None, None, None, None, None, None, None
        
        mano_sequence = action_pred['hand'][0]  # Expected shape: [30, 30]
        wrist_sequence = action_pred['wrist'][0]  # Expected shape: [30, 18]
        print("mano_sequence:", mano_sequence[0])
        print("wrist_sequence:", wrist_sequence)
        
        # Extract camera parameters from original_data
        if 'batch_data' not in sample_data:
            print(f"Key 'batch_data' not found in sample data")
            return None, None, None, None, None, None, None

        batch_data = sample_data['batch_data']
        if 'original_data' not in batch_data:
            print(f"Key 'original_data' not found in batch_data")
            return None, None, None, None, None, None, None

        original_data = batch_data['original_data']

        # if 'data' not in original_data:
        #     print(f"Key 'data' not found in original_data")
        #     return None, None, None, None, None
        # Extract ground truth data
        gt_mano_sequence = torch.from_numpy(batch_data['batch']['action/hand'][0]).float()  # [30, 30]
        gt_wrist_sequence = torch.from_numpy(batch_data['batch']['action/wrist'][0]).float()  # [30, 18]
        print("Ground truth MANO shape:", gt_mano_sequence.shape)
        print("Ground truth wrist shape:", gt_wrist_sequence.shape)
        print("Prediction vs GT wrist difference:", (gt_wrist_sequence - wrist_sequence).abs().mean().item())
        
        # Extract presence data (single integer for all frames)
        presence = 3  # Default to both hands visible
        if 'presence' in batch_data['batch']:
            presence = int(batch_data['batch']['presence'])
            print(f"Hand presence: {presence} (1=left only, 2=right only, 3=both visible)")
        
        # Extract intrinsic matrix
        if 'intrinsic' not in original_data:
            print(f"Key 'intrinsic' not found in data")
            return None, None, None, None, None, None, None
        
        intrinsic_matrix = original_data['intrinsic']  # Expected shape: [4]

        # Extract extrinsic parameters
        if 'extrinsic' not in original_data:
            print(f"Key 'extrinsic' not found in data")
            return None, None, None, None, None, None, None
        
        extrinsic_data = original_data['extrinsic']  # Expected shape: [16] (flattened 4x4 matrix)
        
        # Convert flattened extrinsic to 4x4 matrix and replicate for all frames
        if isinstance(extrinsic_data, torch.Tensor):
            extrinsic_flat = extrinsic_data
        else:
            extrinsic_flat = torch.tensor(extrinsic_data, dtype=torch.float32)
        
        if extrinsic_flat.numel() != 16:
            print(f"Unexpected extrinsic data size: {extrinsic_flat.numel()}, expected 16")
            return None, None, None, None, None, None, None
        
        # Reshape to 4x4 matrix
        extrinsic_matrix = extrinsic_flat.reshape(4, 4)
        
        # Replicate for all frames (30 frames)
        num_frames = mano_sequence.shape[0]
        extrinsic_matrices = extrinsic_matrix.unsqueeze(0).repeat(num_frames, 1, 1)  # [30, 4, 4]
        
        print("extrinsic_matrix:", extrinsic_matrix)
        print("extrinsic_matrices shape:", extrinsic_matrices.shape)
        
        # Extract background image
        if 'image' not in original_data:
            print(f"Key 'image' not found in data")
            return None, None, None, None, None, None, None
        
        background_image = original_data['image']  # Expected shape: [384, 384, 3]
        
        # Validate tensor shapes
        if len(mano_sequence.shape) != 2 or mano_sequence.shape[1] != 30:
            print(f"Unexpected MANO sequence shape: {mano_sequence.shape}, expected [n, 30]")
            return None, None, None, None, None, None, None
        if len(wrist_sequence.shape) != 2 or wrist_sequence.shape[1] != 18:
            print(f"Unexpected wrist sequence shape: {wrist_sequence.shape}, expected [n, 18]")
            return None, None, None, None, None, None, None
        # if len(intrinsic_matrix.shape) != 1 or intrinsic_matrix.shape[0] != 4:
        #     print(f"Unexpected intrinsic matrix shape: {intrinsic_matrix.shape}, expected [4]")
            return None, None, None, None, None
        if len(extrinsic_matrices.shape) != 3 or extrinsic_matrices.shape[1] != 4 or extrinsic_matrices.shape[2] != 4:
            print(f"Unexpected extrinsic matrices shape: {extrinsic_matrices.shape}, expected [n, 4, 4]")
            return None, None, None, None, None, None, None
        if len(background_image.shape) != 3 or background_image.shape[2] != 3:
            print(f"Unexpected background image shape: {background_image.shape}, expected [H, W, 3]")
            return None, None, None, None, None, None, None
        
        print(f"Loaded complete data for sample {sample_id}:")
        print(f"  MANO sequence shape: {mano_sequence.shape}")
        print(f"  Wrist sequence shape: {wrist_sequence.shape}")
        # print(f"  Intrinsic matrix shape: {intrinsic_matrix.shape}")
        print(f"  Extrinsic matrices shape: {extrinsic_matrices.shape}")
        print(f"  Background image shape: {background_image.shape}")
        print(f"  MANO data range: [{mano_sequence.min().item():.3f}, {mano_sequence.max().item():.3f}]")
        print(f"  Wrist data range: [{wrist_sequence.min().item():.3f}, {wrist_sequence.max().item():.3f}]")
        
        return mano_sequence, wrist_sequence, gt_mano_sequence, gt_wrist_sequence, intrinsic_matrix, extrinsic_matrices, background_image, presence
        
    except Exception as e:
        print(f"Failed to load complete data from {path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None, None

def main():
    """
    Main function for inference visualization
    """
    parser = argparse.ArgumentParser(description='Hand Motion Visualizer for Inference')
    parser.add_argument('--data_path', type=str, 
                       default='/share_data/yeyuyao/egovla/egovla_predictions_complete.pt',
                       help='Path to complete data file containing predictions, camera params, and images')
    parser.add_argument('--mano_dir', type=str, default='/home/yeyuyao',
                       help='Directory containing manopth repository')
    parser.add_argument('--sample_id', type=int, default=0,
                       help='Index of sample to visualize (default: 0)')
    parser.add_argument('--find_worst', action='store_true', default=False,
                       help='Automatically find and use the sample with highest loss')
    parser.add_argument('--video_name', type=str, default='inference_hand_motion.mp4',
                       help='Output video name')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for saving videos')
    parser.add_argument('--fps', type=int, default=30, help='Output video frame rate')
    parser.add_argument('--show_mesh', action='store_true', default=False,
                       help='Show hand mesh (faces) for better visual appearance')
    parser.add_argument('--mesh_alpha', type=float, default=0.9,
                       help='Transparency of mesh (0.0 to 1.0)')
    parser.add_argument('--show_gt', action='store_true', default=False,
                       help='Show ground truth annotations (green) along with predictions')
    
    args = parser.parse_args()
    
    try:
        # Add manopth to path dynamically
        manopth_path = os.path.join(args.mano_dir, 'manopth')
        if manopth_path not in sys.path:
            sys.path.append(manopth_path)
        
        # Initialize visualizer with mano model path
        mano_root = os.path.join(args.mano_dir, 'manopth', 'mano', 'models')
        print(f"Initializing hand visualizer for inference with MANO root: {mano_root}")
        visualizer = HandVisualizer(mano_root=mano_root)
        
        # Determine which sample to use
        if args.find_worst:
            print(f"Finding sample with highest loss from: {args.data_path}")
            sample_id = find_worst_sample(args.data_path)
            print(f"Selected sample with highest loss: {sample_id}")
        else:
            sample_id = args.sample_id
            print(f"Using specified sample index: {sample_id}")
        
        # Load complete data from single file
        print(f"Loading complete data from: {args.data_path}")
        mano_sequence, wrist_sequence, gt_mano_sequence, gt_wrist_sequence, intrinsic_matrix, extrinsic_matrices, background_image, presence = load_complete_data(
            args.data_path, sample_id)
        
        if mano_sequence is None:
            raise ValueError(f"Failed to load complete data from file")
        
        # Process background image
        # Convert from tensor to numpy array if necessary
        if isinstance(background_image, torch.Tensor):
            background_img = background_image.cpu().numpy()
        else:
            background_img = background_image
        
        # Ensure image is in uint8 format
        if background_img.dtype != np.uint8:
            if background_img.max() <= 1.0:
                background_img = (background_img * 255).astype(np.uint8)
            else:
                background_img = background_img.astype(np.uint8)
        
        # Convert from RGB to BGR for OpenCV compatibility (assuming input is RGB)
        background_img = cv2.cvtColor(background_img, cv2.COLOR_RGB2BGR)
        print(f"Background image processed: {background_img.shape[1]}x{background_img.shape[0]}")
        
        # Move to device and convert to float32 to avoid dtype issues
        mano_sequence = mano_sequence.to(visualizer.device).float()
        wrist_sequence = wrist_sequence.to(visualizer.device).float()
        if args.show_gt:
            gt_mano_sequence = gt_mano_sequence.to(visualizer.device).float()
            gt_wrist_sequence = gt_wrist_sequence.to(visualizer.device).float()
        
        # Validate tensor shapes
        if len(mano_sequence.shape) != 2 or mano_sequence.shape[1] != 30:
            raise ValueError(f"MANO parameters should be [30, 30], got {mano_sequence.shape}")
        if len(wrist_sequence.shape) != 2 or wrist_sequence.shape[1] != 18:
            raise ValueError(f"Wrist parameters should be [30, 18], got {wrist_sequence.shape}")
        if mano_sequence.shape[0] != wrist_sequence.shape[0]:
            raise ValueError(f"MANO and wrist sequences must have same number of frames: {mano_sequence.shape[0]} vs {wrist_sequence.shape[0]}")
        
        num_frames = mano_sequence.shape[0]
        print(f"Detected {num_frames} frames in the prediction sequences")
        
        # Extract camera intrinsics
        # Convert numpy array to torch tensor if necessary for .item() method
        if isinstance(intrinsic_matrix, np.ndarray):
            intrinsic_matrix = torch.from_numpy(intrinsic_matrix).float()
        
        # fx = intrinsic_matrix[0].item()
        # fy = intrinsic_matrix[1].item()
        # cx = intrinsic_matrix[2].item()
        # cy = intrinsic_matrix[3].item()
        fx = 275.350244
        fy = 489.329905
        cx = 193.537170
        cy = 187.312717
        
        print(f"Loaded camera intrinsics:")
        print(f"  fx={fx:.3f}, fy={fy:.3f}")
        print(f"  cx={cx:.3f}, cy={cy:.3f}")
        
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
        
        # Generate n-frame prediction video with static background
        print(f"Generating {num_frames}-frame prediction video (sample {sample_id}) with static background...")
        if args.show_mesh:
            print(f"  - Showing hand mesh (faces) with alpha {args.mesh_alpha}")
        if args.show_gt:
            print(f"  - Showing ground truth annotations (green) underneath predictions")
        
        # Display presence information
        presence_desc = {1: "left hand only", 2: "right hand only", 3: "both hands"}
        print(f"  - Hand visibility: {presence_desc.get(presence, 'unknown')} (presence={presence})")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Construct full output path
        output_path = os.path.join(args.output_dir, args.video_name)
        
        # Prepare ground truth data if show_gt is enabled
        gt_mano_seq = gt_mano_sequence if args.show_gt else None
        gt_wrist_seq = gt_wrist_sequence if args.show_gt else None
        
        visualizer.generate_video_with_predictions(background_img, mano_sequence, wrist_sequence, 
                                                 output_path, args.fps, fx=fx, fy=fy, cx=cx, cy=cy,
                                                 extrinsic_sequence=extrinsic_sequence,
                                                 show_mesh=args.show_mesh, mesh_alpha=args.mesh_alpha,
                                                 gt_mano_sequence=gt_mano_seq, gt_wrist_sequence=gt_wrist_seq,
                                                 presence=presence)
        
        print(f"Done! {num_frames}-frame prediction video (sample {sample_id}) saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    main()
