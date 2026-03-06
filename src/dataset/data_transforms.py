'''
Data transformation functions for LegendVLA datasets.

Extracted from legendvla_dataset.py for modularity.
'''

from typing import Optional
import numpy as np
import torch
from PIL import Image

from src.utils.geometry import (
    transform_wrist_to_target_frame,
    homo_matrix_from_trans_6drot,
    homo_matrix_to_trans_6drot,
    transform_hand_points_to_wrist_frame,
    transform_hand_points_to_target_frame,
    transform_hand_points_from_wrist_to_camera_frame,
)
from src.model.common.normalizer import LinearNormalizer


def get_relative_action(state, action):
    '''
    Args:
        state: np.ndarray, shape: [wrist_dim + hand_dim]
        action: np.ndarray, shape: [H, wrist_dim + hand_dim]
    Returns:
        action: np.ndarray, shape: [H, wrist_dim + hand_dim]
    '''
    action = action.copy() # avoid modifying the original action
    for idx in range(2):
        wrist_action_homo_mat = homo_matrix_from_trans_6drot(action[..., idx*3 : idx*3+3], action[..., 6+idx*6 : 6+idx*6+6])
        wrist_state_homo_mat = homo_matrix_from_trans_6drot(state[idx*3 : idx*3+3], state[6+idx*6 : 6+idx*6+6])
        wrist_action_homo_mat = np.linalg.pinv(wrist_state_homo_mat) @ wrist_action_homo_mat
        trans, rot_6d = homo_matrix_to_trans_6drot(wrist_action_homo_mat)
        action[..., idx*3 : idx*3+3] = trans
        action[..., 6+idx*6 : 6+idx*6+6] = rot_6d

    action[..., 18:] = action[..., 18:] - state[18:]
    return action

def get_absolute_action(state, relative_action):
    '''
    Convert relative action back to absolute action.
    This is the inverse operation of get_relative_action.

    Args:
        state: torch.Tensor or np.ndarray, shape: [wrist_dim + hand_dim] - current state
            state is in the first frame's camera coordinate system, where wrist is in cam frame and hand is in wrist frame
        relative_action: torch.Tensor or np.ndarray, shape: [H, wrist_dim + hand_dim] - relative action

    Returns:
        absolute_action: torch.Tensor or np.ndarray, shape: [H, wrist_dim + hand_dim] - absolute action
    '''
    # Create a copy of relative_action to store absolute_action
    if isinstance(relative_action, torch.Tensor):
        absolute_action = relative_action.clone()
    else:
        absolute_action = relative_action.copy()

    # For wrist parameters, absolute action = state @ relative action
    # This is the inverse of: relative = pinv(state) @ action
    for idx in range(2):
        wrist_relative_action_homo_mat = homo_matrix_from_trans_6drot(relative_action[..., idx*3 : idx*3+3], relative_action[..., 6+idx*6 : 6+idx*6+6])
        wrist_state_homo_mat = homo_matrix_from_trans_6drot(state[idx*3 : idx*3+3], state[6+idx*6 : 6+idx*6+6])
        wrist_action_homo_mat = wrist_state_homo_mat @ wrist_relative_action_homo_mat
        trans, rot_6d = homo_matrix_to_trans_6drot(wrist_action_homo_mat)
        absolute_action[..., idx*3 : idx*3+3] = trans
        absolute_action[..., 6+idx*6 : 6+idx*6+6] = rot_6d

    # For hand parameters, absolute action = relative action + state
    # This is the inverse of: relative = action - state
    absolute_action[..., 18:] = relative_action[..., 18:] + state[18:]

    return absolute_action

def transform_hand_from_wrist_to_camera(absolute_action, extrinsic):
    '''
    Transform hand points from wrist frame to camera coordinate system.
    This function transforms hand points from wrist frame to first frame's camera coordinate,
    then to target frames' camera coordinates.

    Args:
        absolute_action: torch.Tensor or np.ndarray, shape: [H, wrist_dim + hand_dim] or [wrist_dim + hand_dim]
            absolute action where hand is in wrist frame
        extrinsic: torch.Tensor or np.ndarray, shape: [H, 4, 4] or [4, 4] - camera extrinsic (world2cam) for each frame
    Returns:
        absolute_action: torch.Tensor or np.ndarray, shape: [H, wrist_dim + hand_dim] or [wrist_dim + hand_dim]
            absolute action where hand is in camera frame
    '''
    # Create a copy to avoid modifying the input
    if isinstance(absolute_action, torch.Tensor):
        absolute_action = absolute_action.clone()
    else:
        absolute_action = absolute_action.copy()

    hand_points_wrist = absolute_action[..., 18:]  # (H, 30) or (30,) - hand in wrist frame
    wrist_action_initial = absolute_action[..., :18]  # (H, 18) or (18,) - wrist in first frame's cam coordinate

    # Transform hand points from wrist frame to first frame's cam coordinate
    if absolute_action.ndim > 1:
        # Multiple frames
        hand_points_cam_initial = transform_hand_points_from_wrist_to_camera_frame(hand_points_wrist, wrist_action_initial)
    else:
        # Single frame
        hand_points_cam_initial = transform_hand_points_from_wrist_to_camera_frame(hand_points_wrist.reshape(1, -1), wrist_action_initial.reshape(1, -1))
        hand_points_cam_initial = hand_points_cam_initial.reshape(-1)

    # Now both wrist and hand are in first frame's cam coordinate
    absolute_action[..., 18:] = hand_points_cam_initial
    initial_extrinsic_inv = np.linalg.inv(extrinsic[0])
    # Transform wrist from first frame's cam coordinate to world, then to target frames' cam coordinates
    wrist_action_world = transform_wrist_to_target_frame(wrist_action_initial, initial_extrinsic_inv)  # Transform to world
    wrist_action_target = transform_wrist_to_target_frame(wrist_action_world, extrinsic)  # Transform to target frames
    absolute_action[..., :18] = wrist_action_target

    # Transform hand points from first frame's cam coordinate to world, then to target frames' cam coordinates
    hand_points_world = transform_hand_points_to_target_frame(hand_points_cam_initial, initial_extrinsic_inv)  # Transform to world
    hand_points_target = transform_hand_points_to_target_frame(hand_points_world, extrinsic)  # Transform to target frames
    absolute_action[..., 18:] = hand_points_target

    return absolute_action

def process_state_action(
    wrist_state,
    hand_state,
    wrist_action,
    hand_action,
    extrinsic,
    hand_ndim,
    normalizer : Optional[LinearNormalizer] = None,
    motion_type = 'mano',
    use_relative_action = False,
):
    '''
    Args:
        wrist_state: np.ndarray, shape: [N_state, wrist_dim]
        hand_state: np.ndarray, shape: [N_state, all_hand_dim]
        wrist_action: np.ndarray, shape: [N_action, wrist_dim]
        hand_action: np.ndarray, shape: [N_action, all_hand_dim]
        extrinsic: np.ndarray, shape: [4, 4]
        hand_ndim: int
        normalizer: Optional[LinearNormalizer]
        motion_type: str, 'mano' or 'keypoint'
        use_relative_action: bool
    Returns:
        state: np.ndarray, shape: [N_state, wrist_dim + hand_dim]
        action: np.ndarray, shape: [N_action, wrist_dim + hand_dim]
    '''
    # use first self.hand_ndim components of hand state and action
    all_hand_ndim = hand_state.shape[-1] // 2 # per hand dims, i.e. 45 in MANO hand params
    hand_state = np.concatenate([
        hand_state[:, :hand_ndim],
        hand_state[:, all_hand_ndim:all_hand_ndim + hand_ndim]
    ], axis=-1)
    hand_action = np.concatenate([
        hand_action[:, :hand_ndim],
        hand_action[:, all_hand_ndim:all_hand_ndim + hand_ndim]
    ], axis=-1)

    if motion_type == 'fingertips':
        # TODO: We can try transform the fingertips to the camera coordinate system or wrist frame coordinate system
        processed_hand_state = transform_hand_points_to_wrist_frame(hand_state, wrist_state)
        processed_hand_state = processed_hand_state.reshape(hand_state.shape)
        processed_hand_action = transform_hand_points_to_wrist_frame(hand_action, wrist_action)
        processed_hand_action = processed_hand_action.reshape(hand_action.shape)
    elif motion_type == 'mano':
        processed_hand_state = hand_state
        processed_hand_action = hand_action
    else:
        raise ValueError(f"Unsupported motion type: {motion_type}")

    # transform the wrist state and action to the camera coordinate system
    processed_wrist_state = transform_wrist_to_target_frame(wrist_state, extrinsic)
    processed_wrist_action = transform_wrist_to_target_frame(wrist_action, extrinsic)

    # use delta of wrist translation and hand mano params as action
    processed_state = np.concatenate([processed_wrist_state, processed_hand_state], axis=-1)
    processed_action = np.concatenate([processed_wrist_action, processed_hand_action], axis=-1)
    if use_relative_action:
        processed_action = get_relative_action(processed_state[-1], processed_action)

    if normalizer is not None:
        if not use_relative_action: # Use unified normalizer for both state and action
            state = normalizer['motions'](processed_state)
            action = normalizer['motions'](processed_action)
        else: # Use separate normalizers for state and action
            state = normalizer['states'](processed_state)
            action = normalizer['actions'](processed_action)
    else: # No normalizer
        state = processed_state
        action = processed_action
    return state, action

# TODO: maybe we need to use the same augmentation for all images in the action chunk
# TODO: we can try more advanced augmentation techniques, notably, we should care about the depth image augmentation
def process_image(image, depth_image = None, aug_transform = None, depth_clip_range = None):
    '''
    Args:
        image: np.ndarray, shape: [N, H, W, 3]
        depth_image: np.ndarray, shape: [N, H, W]
        aug_transform: Optional[Callable]
    Returns:
        image: np.ndarray, shape: [N, H, W, 3]
        depth_image: np.ndarray, shape: [N, H, W]
    '''
    images_to_process = image
    depth_images_to_process = depth_image
    if aug_transform is not None:
        augmented_images = []
        for img_np in images_to_process:
            # convert NumPy array (H, W, C) to PIL Image
            img_pil = Image.fromarray(img_np)
            augmented_pil = aug_transform(img_pil)
            augmented_np = np.array(augmented_pil, dtype=np.uint8)
            augmented_images.append(augmented_np)
        images_to_process = np.stack(augmented_images, dtype=np.uint8)

    return images_to_process, depth_images_to_process
