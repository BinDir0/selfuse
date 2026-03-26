'''
Data transformation functions for LegendVLA datasets.
'''

import random
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF

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

def random_resized_crop(images, depth_images, intrinsic, scale_range=(0.9, 1.0)):
    '''Random crop then resize back. Same crop for all frames (temporal consistency).'''
    N, H, W = images.shape[:3]
    scale = random.uniform(*scale_range)
    crop_h, crop_w = int(H * scale), int(W * scale)
    if crop_h >= H and crop_w >= W:
        return images, depth_images, intrinsic

    y0 = random.randint(0, H - crop_h)
    x0 = random.randint(0, W - crop_w)
    sx, sy = W / crop_w, H / crop_h

    # Crop + resize RGB (bilinear)
    images = np.stack([
        np.array(Image.fromarray(f).crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((W, H), Image.BILINEAR), dtype=np.uint8)
        for f in images
    ])

    # Crop + resize depth (nearest to avoid interpolation artifacts at edges)
    if depth_images is not None:
        depth_images = np.stack([
            np.array(Image.fromarray(f, mode='F').crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((W, H), Image.NEAREST), dtype=np.float32)
            for f in depth_images
        ])

    # Update intrinsic [fx, fy, cx, cy] for the crop-then-resize transform
    if intrinsic is not None:
        intrinsic = intrinsic.copy()
        intrinsic[0] *= sx                      # fx
        intrinsic[1] *= sy                      # fy
        intrinsic[2] = (intrinsic[2] - x0) * sx # cx
        intrinsic[3] = (intrinsic[3] - y0) * sy # cy

    return images, depth_images, intrinsic


def augment_color(images):
    '''Color jitter + Gaussian blur. Params sampled once for temporal consistency.'''
    fn_idx, brightness, contrast, saturation, hue = ColorJitter.get_params(
        brightness=(0.7, 1.3), contrast=(0.7, 1.3),
        saturation=(0.7, 1.3), hue=(-0.1, 0.1),
    )
    sigma = random.uniform(0.1, 2.0)

    jitter = [
        lambda img: TF.adjust_brightness(img, brightness),
        lambda img: TF.adjust_contrast(img, contrast),
        lambda img: TF.adjust_saturation(img, saturation),
        lambda img: TF.adjust_hue(img, hue),
    ]
    ops = [jitter[i] for i in fn_idx]
    ops.append(lambda img: TF.gaussian_blur(img, kernel_size=[5, 5], sigma=sigma))

    def apply(frame):
        pil = Image.fromarray(frame)
        for op in ops:
            pil = op(pil)
        return np.array(pil, dtype=np.uint8)

    return np.stack([apply(f) for f in images])


def augment_depth(depth_images, noise_scale=0.005, dropout_prob=0.5):
    '''Depth-dependent Gaussian noise + random rectangular dropout (shared across frames).
    Noise sigma = noise_scale * depth; at 1m ≈ 5mm. Dropout area ≤ ~16%.'''
    depth_images = depth_images.copy()
    N, H, W = depth_images.shape

    # Depth-dependent Gaussian noise: only on valid (>0) pixels
    valid = depth_images > 0
    noise = np.random.randn(N, H, W).astype(np.float32)
    depth_images[valid] += noise[valid] * noise_scale * depth_images[valid]
    np.maximum(depth_images, 0, out=depth_images)

    # Random rectangular dropout
    if random.random() < dropout_prob:
        rh = random.randint(1, max(1, int(H * 0.4)))
        rw = random.randint(1, max(1, int(W * 0.4)))
        ry = random.randint(0, H - rh)
        rx = random.randint(0, W - rw)
        depth_images[:, ry:ry + rh, rx:rx + rw] = 0

    return depth_images


def process_image(image, depth_image=None, intrinsic=None, aug_transform=None, depth_clip_range=None):
    '''
    Args:
        image: np.ndarray, shape: [N, H, W, 3], uint8
        depth_image: np.ndarray or None, shape: [N, H, W]
        intrinsic: np.ndarray or None, shape: [4] — [fx, fy, cx, cy]
        aug_transform: truthy value enables augmentation (the object itself is not called)
        depth_clip_range: [min, max] in meters, or None
    Returns:
        image: np.ndarray, shape: [N, H, W, 3]
        depth_image: np.ndarray or None, shape: [N, H, W], float32 (meters)
        intrinsic: np.ndarray or None, shape: [4]
    '''
    # Depth stored as uint16 in millimeters; convert to float32 meters.
    # If already float, assume meters and skip conversion.
    if depth_image is not None and depth_image.dtype == np.uint16:
        depth_image = depth_image.astype(np.float32) / 1000.0

    if aug_transform is not None:
        image, depth_image, intrinsic = random_resized_crop(image, depth_image, intrinsic)
        image = augment_color(image)
        if depth_image is not None:
            depth_image = augment_depth(depth_image)

    if depth_clip_range is not None and depth_image is not None:
        depth_image = np.clip(depth_image, depth_clip_range[0], depth_clip_range[1])

    return image, depth_image, intrinsic
