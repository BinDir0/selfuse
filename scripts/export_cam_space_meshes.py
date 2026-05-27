"""Export HaWoR motion-stage (camera-space) hand meshes to a self-contained npz.

The motion stage writes per-hand, per-chunk camera-space MANO parameters to
``{seq_folder}/cam_space/{hand_idx}/{first}_{last}.json`` (hand_idx 0=left, 1=right).
This script turns those parameters into camera-space hand *vertices* and stores them,
together with the MANO faces and the original video frame indices, in a single npz so
downstream viewers (e.g. the VGGT-Omega demo) can overlay the hands without importing
HaWoR or loading MANO.

Camera convention of the exported vertices is OpenCV (x-right, y-down, z-forward),
matching how the motion stage projects them (u = x/z*focal + cx).

Usage:
    python scripts/export_cam_space_meshes.py --video_path example/video_0.mp4 \
        --out example/video_0/hand_meshes.npz
    # or point directly at the sequence folder that contains cam_space/
    python scripts/export_cam_space_meshes.py --seq_folder example/video_0 --out hand_meshes.npz
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from hawor.utils.rotation import rotation_matrix_to_angle_axis

# Same extra faces the pipeline appends to close the MANO wrist (see hawor_common.MANO_FACE_EXTRA).
MANO_FACE_EXTRA = np.array([
    [92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
    [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
    [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
    [120, 108, 78], [78, 108, 79],
])


def _build_mano_models(device):
    from lib.models.mano_wrapper import MANO

    mano_right = MANO(
        data_dir="_DATA/data/", model_path="_DATA/data/mano", gender="neutral",
        num_hand_joints=15, create_body_pose=False,
    ).to(device)
    mano_left = MANO(
        data_dir="_DATA/data_left/", model_path="_DATA/data_left/mano_left", gender="neutral",
        num_hand_joints=15, create_body_pose=False, is_rhand=False,
    ).to(device)
    mano_left.shapedirs[:, 0, :] *= -1
    return mano_right, mano_left


def _hand_faces():
    faces_right = np.concatenate([get_mano_faces(), MANO_FACE_EXTRA], axis=0).astype(np.int32)
    faces_left = faces_right[:, [0, 2, 1]]
    return faces_left, faces_right


def _export_hand(seq_folder, hand_idx, mano_left, mano_right, use_cuda):
    """Return (frames (N,), vertices (N,778,3)) in camera space for one hand."""
    cam_dir = os.path.join(seq_folder, "cam_space", str(hand_idx))
    do_flip = hand_idx == 0  # 0 = left
    frame_to_verts = {}

    for json_path in sorted(glob.glob(os.path.join(cam_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(json_path))[0]
        try:
            first, last = (int(x) for x in stem.split("_", 1))
        except ValueError:
            print(f"  skip non-chunk file {json_path}")
            continue

        with open(json_path) as handle:
            pred = json.load(handle)
        data_out = {k: torch.tensor(np.asarray(v, dtype=np.float32)) for k, v in pred.items()}

        # Saved orient/pose are rotation matrices; run_mano expects axis-angle.
        root_aa = rotation_matrix_to_angle_axis(data_out["init_root_orient"])      # (1,T,3)
        hand_aa = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])        # (1,T,15,3)
        run_fn = run_mano_left if do_flip else run_mano
        mano_model = mano_left if do_flip else mano_right
        outputs = run_fn(
            data_out["init_trans"], root_aa, hand_aa,
            betas=data_out["init_betas"], use_cuda=use_cuda, mano_model=mano_model,
        )
        verts = outputs["vertices"][0].detach().float().cpu().numpy()  # (T,778,3)

        frames = list(range(first, last + 1))
        if len(frames) != verts.shape[0]:
            raise ValueError(
                f"{json_path}: frame span {len(frames)} != T {verts.shape[0]}"
            )
        for f, v in zip(frames, verts):
            frame_to_verts[f] = v  # later chunks win on overlap

    if not frame_to_verts:
        return np.zeros((0,), np.int64), np.zeros((0, 778, 3), np.float32)
    frames = np.array(sorted(frame_to_verts), dtype=np.int64)
    vertices = np.stack([frame_to_verts[f] for f in frames]).astype(np.float32)
    return frames, vertices


def main():
    parser = argparse.ArgumentParser(description="Export camera-space hand meshes to npz.")
    parser.add_argument("--seq_folder", type=str, default=None,
                        help="Folder containing cam_space/. Defaults to dirname(video)/stem.")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Video path used to derive seq_folder when --seq_folder is omitted.")
    parser.add_argument("--out", type=str, required=True, help="Output .npz path.")
    args = parser.parse_args()

    seq_folder = args.seq_folder
    if seq_folder is None:
        if args.video_path is None:
            parser.error("Provide --seq_folder or --video_path.")
        stem = os.path.splitext(os.path.basename(args.video_path))[0]
        seq_folder = os.path.join(os.path.dirname(args.video_path), stem)
    if not os.path.isdir(os.path.join(seq_folder, "cam_space")):
        parser.error(f"No cam_space/ under {seq_folder}; run the motion stage first.")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    mano_right, mano_left = _build_mano_models(device)
    faces_left, faces_right = _hand_faces()

    print("Exporting left hand (idx 0) ...")
    left_frames, left_vertices = _export_hand(seq_folder, 0, mano_left, mano_right, use_cuda)
    print("Exporting right hand (idx 1) ...")
    right_frames, right_vertices = _export_hand(seq_folder, 1, mano_left, mano_right, use_cuda)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        left_frames=left_frames, left_vertices=left_vertices,
        right_frames=right_frames, right_vertices=right_vertices,
        faces_left=faces_left, faces_right=faces_right,
    )
    print(
        f"Saved {args.out}: left {left_vertices.shape} (frames {left_frames.min() if len(left_frames) else '-'}.."
        f"{left_frames.max() if len(left_frames) else '-'}), right {right_vertices.shape} "
        f"(frames {right_frames.min() if len(right_frames) else '-'}..{right_frames.max() if len(right_frames) else '-'})"
    )


if __name__ == "__main__":
    main()
