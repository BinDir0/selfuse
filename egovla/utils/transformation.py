import torch
import torch.nn.functional as F
import numpy as np

def rot_matrix_from_6drot(rot):
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
    if isinstance(rot, np.ndarray):
        is_numpy = True
        rot = torch.from_numpy(rot)
    else:
        is_numpy = False
    
    # Store original shape for later restoration
    original_shape = rot.shape
    
    # Handle single vector case (shape: [6])
    # Reshape to 2D for easier processing: [..., 6] -> [N, 6]
    rot = rot.reshape(-1, 6)
    
    # Extract the two 3D vectors
    a = rot[..., :3]  # First 3 elements: [N, 3]
    b = rot[..., 3:]  # Last 3 elements: [N, 3]
    
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
    else:
        rot_matrix = rot_matrix.squeeze(0)

    if is_numpy:
        rot_matrix = rot_matrix.numpy()
    
    return rot_matrix

def rot_matrix_to_6drot(rot_matrix):
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

def transform_to_target_frame(pose, target_extrinsic):
    '''
    Transform the pose to the target frame.
    Args:
        pose: torch.Tensor or np.ndarray, shape: [T, 4, 4] or [B, T, 4, 4] in world frame
        target_extrinsic: torch.Tensor or np.ndarray, shape: [4, 4] or [B, 4, 4] in world frame
    Returns:
        pose: torch.Tensor, shape: [T, 4, 4] or [B, T, 4, 4] in target frame
    '''
    assert pose.dtype == target_extrinsic.dtype, "pose and target_extrinsic must have the same dtype"
    # print(f"pose.shape: {pose.shape}, target_extrinsic.shape: {target_extrinsic.shape}")
    if isinstance(pose, np.ndarray):
        is_numpy = True
        pose = torch.from_numpy(pose)
        target_extrinsic = torch.from_numpy(target_extrinsic)
    else:
        is_numpy = False
    target_extrinsic = target_extrinsic.unsqueeze(-3)

    # use pseudo-inverse to avoid NaN
    target_extrinsic_inv = torch.linalg.pinv(target_extrinsic)
    pose = torch.matmul(target_extrinsic_inv, pose)
    
    # check if the result contains NaN and handle it
    if torch.isnan(pose).any():
        print(f"Warning: NaN detected in pose after transformation")
        pose = torch.where(torch.isnan(pose), torch.zeros_like(pose), pose)
    
    if is_numpy:
        pose = pose.numpy()
    return pose

def transform_wrist_to_target_frame(wrist_action, target_extrinsic):
    '''
    Transform the wrist action to the target frame.
    Args:
        wrist_action: torch.Tensor or np.ndarray, shape: [T, 18] or [B, T, 18]
        target_extrinsic: torch.Tensor or np.ndarray, shape: [4, 4] or [B, 4, 4]
    Returns:
        wrist_action: torch.Tensor, shape: [T, 18] or [B, T, 18]
    '''
    assert wrist_action.dtype == target_extrinsic.dtype, "wrist_action and target_extrinsic must have the same dtype"
    if isinstance(wrist_action, np.ndarray):
        is_numpy = True
        wrist_action = torch.from_numpy(wrist_action)
        target_extrinsic = torch.from_numpy(target_extrinsic)
    else:
        is_numpy = False

    T = wrist_action.shape[-2]

    # left wrist rotation is the first 6 elements, right wrist rotation is the last 6 elements
    wrist_rot_6d = torch.cat([wrist_action[..., 6:12], wrist_action[..., 12:18]], dim=-2)
    wrist_pose = torch.zeros(wrist_rot_6d.shape[:-1] + (4, 4))
    wrist_pose[..., :3, 3] = torch.cat([wrist_action[..., :3], wrist_action[..., 3:6]], dim=-2)
    wrist_pose[..., :3, :3] = rot_matrix_from_6drot(wrist_rot_6d)
    wrist_pose[..., 3, 3] = 1

    wrist_pose = transform_to_target_frame(wrist_pose, target_extrinsic)

    wrist_rot_6d = rot_matrix_to_6drot(wrist_pose[..., :3, :3])
    # print(wrist_action[..., :3].shape, wrist_pose[..., 0:T, :3, 3].shape, wrist_pose.shape)
    wrist_action[..., :3] = wrist_pose[..., 0:T, :3, 3]
    wrist_action[..., 3:6] = wrist_pose[..., T:2*T, :3, 3]
    wrist_action[..., 6:12] = wrist_rot_6d[..., 0:T, :]
    wrist_action[..., 12:18] = wrist_rot_6d[..., T:2*T, :]

    if is_numpy:
        wrist_action = wrist_action.numpy()
    
    return wrist_action
    
