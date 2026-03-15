#!/usr/bin/env python3
"""
Simple offline visualization for HaWoR - works on any headless server.

Uses OpenCV for simple 3D projection rendering - no OpenGL/pyrender required.
Generates mp4 video with basic hand mesh visualization.

Installation:
    pip install opencv-python numpy torch
    # No special dependencies needed!

Usage:
    python demo_offline.py --video_path video.mp4 --vis_mode world
    python demo_offline.py --video_path video.mp4 --vis_mode cam
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_slam import hawor_slam
from scripts.scripts_test_video.hawor_video import hawor_infiller, hawor_motion_estimation


def project_3d_to_2d(vertices, camera_pose, focal_length, width, height):
    """Project 3D vertices to 2D image coordinates using camera intrinsics."""
    # Transform vertices to camera space
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    vertices_cam = vertices @ R.T + t

    # Project to 2D
    x = vertices_cam[:, 0] / (vertices_cam[:, 2] + 1e-8) * focal_length + width / 2
    y = vertices_cam[:, 1] / (vertices_cam[:, 2] + 1e-8) * focal_length + height / 2
    z = vertices_cam[:, 2]

    return np.stack([x, y, z], axis=1)


def render_frame_simple(vertices_left, vertices_right, faces_left, faces_right,
                        bg_image, camera_pose, focal_length, width, height):
    """Render a single frame using simple OpenCV drawing - works everywhere!"""
    result = bg_image.copy()
    overlay = np.zeros_like(result)
    mask = np.zeros(result.shape[:2], dtype=np.uint8)

    # Draw right hand (blue) to overlay
    if vertices_right is not None and len(vertices_right) > 0:
        pts_2d = project_3d_to_2d(vertices_right, camera_pose, focal_length, width, height)

        for face in faces_right:
            # Only draw faces facing camera (z > 0)
            if pts_2d[face, 2].mean() > 0:
                pts = pts_2d[face, :2].astype(np.int32)
                cv2.fillPoly(overlay, [pts], color=(202, 152, 53))  # BGR: blue
                cv2.fillPoly(mask, [pts], color=255)

    # Draw left hand (purple) to overlay
    if vertices_left is not None and len(vertices_left) > 0:
        pts_2d = project_3d_to_2d(vertices_left, camera_pose, focal_length, width, height)

        for face in faces_left:
            # Only draw faces facing camera (z > 0)
            if pts_2d[face, 2].mean() > 0:
                pts = pts_2d[face, :2].astype(np.int32)
                cv2.fillPoly(overlay, [pts], color=(200, 100, 128))  # BGR: purple
                cv2.fillPoly(mask, [pts], color=255)

    # Blend overlay with background once (much faster!)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    result = (overlay * mask_3ch * 0.7 + result * (1 - mask_3ch * 0.7)).astype(np.uint8)

    return result

def create_video_world_mode(left_verts, right_verts, faces_left, faces_right,
                            frame_source, frame_indices, R_c2w, t_c2w, focal_length,
                            output_path, fps=30):
    """Create video in world coordinate mode."""
    print("Rendering video in world mode...")

    # Get video dimensions
    first_img = frame_source.get_frame(int(frame_indices[0]), rgb=False)
    height, width = first_img.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    num_frames = left_verts.shape[0]
    for frame_idx in tqdm(range(num_frames), desc="Rendering frames"):
        # Load background image
        bg_img = frame_source.get_frame(int(frame_indices[frame_idx]), rgb=True)

        # Camera pose for this frame
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R_c2w[frame_idx]
        camera_pose[:3, 3] = t_c2w[frame_idx]

        # Render frame
        frame = render_frame_simple(
            left_verts[frame_idx], right_verts[frame_idx],
            faces_left, faces_right,
            bg_img, camera_pose,
            focal_length, width, height
        )

        # Write to video
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"✓ Video saved to: {output_path}")
    return output_path


def create_video_cam_mode(left_verts, right_verts, faces_left, faces_right,
                         frame_source, frame_indices, R_w2c, t_w2c, focal_length,
                         output_path, fps=30):
    """Create video in camera coordinate mode."""
    print("Rendering video in camera mode...")

    # Get video dimensions
    first_img = frame_source.get_frame(int(frame_indices[0]), rgb=False)
    height, width = first_img.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    num_frames = left_verts.shape[0]
    for frame_idx in tqdm(range(num_frames), desc="Rendering frames"):
        # Load background image
        bg_img = frame_source.get_frame(int(frame_indices[frame_idx]), rgb=True)

        # Camera pose for this frame (world to camera)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R_w2c[frame_idx]
        camera_pose[:3, 3] = t_w2c[frame_idx]

        # Render frame
        frame = render_frame_simple(
            left_verts[frame_idx], right_verts[frame_idx],
            faces_left, faces_right,
            bg_img, camera_pose,
            focal_length, width, height
        )

        # Write to video
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"✓ Video saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Offline HaWoR visualization")
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--input_type", type=str, default='file')
    parser.add_argument("--checkpoint", type=str, default='./weights/hawor/checkpoints/hawor.ckpt')
    parser.add_argument("--infiller_weight", type=str, default='./weights/hawor/checkpoints/infiller.pt')
    parser.add_argument("--vis_mode", type=str, default='world', choices=['world', 'cam'],
                       help='Visualization mode: world or cam')
    parser.add_argument("--fps", type=int, default=30, help='FPS for video output')
    parser.add_argument("--max_frames", type=int, default=None,
                       help='Maximum number of frames to visualize (default: all frames)')
    args = parser.parse_args()

    # Run inference pipeline (reuse existing results if available)
    print("=== Running HaWoR inference ===")
    start_idx, end_idx, seq_folder, frame_source = detect_track_video(args)
    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(args, start_idx, end_idx)

    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
        args, start_idx, end_idx, frame_chunks_all
    )

    # Prepare meshes
    print("\n=== Preparing hand meshes ===")
    vis_start = 0
    vis_end = pred_trans.shape[1] - 1

    # Limit frames if max_frames is specified
    if args.max_frames is not None:
        vis_end = min(vis_start + args.max_frames - 1, vis_end)
        print(f"Limiting visualization to first {args.max_frames} frames (0 to {vis_end})")

    faces = get_mano_faces()
    faces_new = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239],
                         [239, 122, 279], [122, 118, 279], [279, 118, 215],
                         [118, 117, 215], [215, 117, 214], [117, 119, 214],
                         [214, 119, 121], [119, 120, 121], [121, 120, 78],
                         [120, 108, 78], [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]

    # Get right hand vertices
    pred_glob_r = run_mano(
        pred_trans[1:2, vis_start:vis_end],
        pred_rot[1:2, vis_start:vis_end],
        pred_hand_pose[1:2, vis_start:vis_end],
        betas=pred_betas[1:2, vis_start:vis_end]
    )
    right_verts = pred_glob_r['vertices'][0].cpu().numpy()

    # Get left hand vertices
    pred_glob_l = run_mano_left(
        pred_trans[0:1, vis_start:vis_end],
        pred_rot[0:1, vis_start:vis_end],
        pred_hand_pose[0:1, vis_start:vis_end],
        betas=pred_betas[0:1, vis_start:vis_end]
    )
    left_verts = pred_glob_l['vertices'][0].cpu().numpy()

    # Transform coordinates
    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
    R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)

    left_verts = np.einsum('ij,tnj->tni', R_x.numpy(), left_verts)
    right_verts = np.einsum('ij,tnj->tni', R_x.numpy(), right_verts)

    # Output path
    output_dir = Path(seq_folder) / f"vis_{args.vis_mode}_{vis_start}_{vis_end}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "visualization.mp4"

    frame_indices = np.arange(vis_start, vis_end, dtype=np.int64)

    # Render video
    print(f"\n=== Rendering video ({args.vis_mode} mode) ===")
    if args.vis_mode == 'world':
        create_video_world_mode(
            left_verts, right_verts, faces_left, faces_right,
            frame_source, frame_indices, R_c2w_sla_all[vis_start:vis_end].cpu().numpy(),
            t_c2w_sla_all[vis_start:vis_end].cpu().numpy(),
            img_focal, output_video, fps=args.fps
        )
    else:  # cam mode
        create_video_cam_mode(
            left_verts, right_verts, faces_left, faces_right,
            frame_source, frame_indices, R_w2c_sla_all[vis_start:vis_end].cpu().numpy(),
            t_w2c_sla_all[vis_start:vis_end].cpu().numpy(),
            img_focal, output_video, fps=args.fps
        )

    print(f"\n✓ Done! Video saved to: {output_video}")
    print(f"\nTo view: scp user@server:{output_video} ./")


if __name__ == '__main__':
    main()
