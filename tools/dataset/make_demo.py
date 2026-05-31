#!/usr/bin/env python3
"""Produce a 7-image dataset demo for one processed clip.

Outputs (into --out_dir, default <seq_folder>/demo/):
  frame_0_t0.0s.jpg ... frame_5_t5.0s.jpg   -> 6 input frames, one per second
                                                (dataset is 30 fps -> indices 0,30,60,90,120,150)
  hand_cam_3d.png                            -> HaWoR-Figure-1-style world-space composite
                                                (MANO hands trail + camera frustums + ground),
                                                rendered by scripts/render_world_traj.py

This does NOT re-run any inference. It reuses what a clip already has after the
motion + SLAM stages:
  <seq_folder>/world_space_res.pth
  <seq_folder>/SLAM/hawor_slam_w_scale_*.npz
and the extracted RGB frames sitting next to it (the pipeline's frames/<clip_id>/).

The 3D render needs aitviewer with headless EGL, so run this on a GPU machine:
  python tools/dataset/make_demo.py --seq_folder /path/to/<clip>.hawor_pipeline/stage_outputs/<clip_id>

If the frames live somewhere non-standard, pass --frames_dir explicitly.
Extra render knobs (view/azim/elev/num_samples/hands/...) are forwarded to
render_world_traj.py; tune them for a nicer figure.
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seq_folder", required=True,
                   help="clip stage-output dir with world_space_res.pth and SLAM/")
    p.add_argument("--frames_dir", default=None,
                   help="dir with extracted RGB frames (NNNNNN.jpg). "
                        "Default: <seq_folder>/../../frames/<clip_id>/")
    p.add_argument("--out_dir", default=None,
                   help="where to write the 7 images (default <seq_folder>/demo)")
    p.add_argument("--fps", type=float, default=30.0,
                   help="dataset frame rate (default 30) -> 1 frame/sec stride")
    p.add_argument("--num_frames", type=int, default=6,
                   help="how many 1-fps frames to pull (default 6 -> seconds 0..5)")
    p.add_argument("--frame_ext", default="jpg", help="extracted-frame extension")
    p.add_argument("--skip_frames", action="store_true",
                   help="only (re)render the 3D figure, do not copy frames")
    p.add_argument("--skip_3d", action="store_true",
                   help="only copy the 6 frames, skip the 3D render")
    # ---- forwarded to render_world_traj.py ----
    p.add_argument("--num_samples", type=int, default=8,
                   help="hand/cam trail samples in the 3D figure (render_world_traj)")
    p.add_argument("--frame_start", type=int, default=0,
                   help="restrict the 3D trail to video frames >= this")
    p.add_argument("--frame_end", type=int, default=-1,
                   help="restrict the 3D trail to video frames <= this (-1 = last)")
    p.add_argument("--rrd", default=None,
                   help="render the 3D trail to a rerun .rrd instead of a PNG "
                        "(no GL/EGL needed; open in the rerun web viewer)")
    p.add_argument("--hands", choices=["both", "left", "right"], default="both")
    p.add_argument("--view", choices=["auto", "hawor"], default="auto")
    p.add_argument("--cam_azim", type=float, default=60.0)
    p.add_argument("--cam_elev", type=float, default=18.0)
    p.add_argument("--cam_dist_scale", type=float, default=2.6)
    p.add_argument("--width", type=int, default=2000)
    p.add_argument("--height", type=int, default=1500)
    p.add_argument("--rebase", action="store_true")
    p.add_argument("--align", action="store_true")
    p.add_argument("--no_fade", action="store_true")
    p.add_argument("--no_ground", action="store_true")
    return p.parse_args()


def default_frames_dir(seq_folder):
    """A pipeline clip is .../<X>.hawor_pipeline/stage_outputs/<clip_id>; the
    extracted frames sit at .../<X>.hawor_pipeline/frames/<clip_id>/."""
    seq_folder = os.path.abspath(seq_folder.rstrip("/"))
    clip_id = os.path.basename(seq_folder)
    pipeline_root = os.path.dirname(os.path.dirname(seq_folder))  # drop stage_outputs/<clip_id>
    return os.path.join(pipeline_root, "frames", clip_id)


def copy_frames(frames_dir, out_dir, fps, num_frames, ext):
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(
            f"frames dir not found: {frames_dir}\n"
            f"  pass --frames_dir explicitly to the dir holding NNNNNN.{ext}")
    stride = int(round(fps))
    written = []
    for k in range(num_frames):
        idx = k * stride
        # frames are usually 6-digit zero-padded; try that, then other widths
        src = next(
            (c for c in (os.path.join(frames_dir, f"{idx:0{w}d}.{ext}") for w in (6, 5, 4, 0))
             if os.path.exists(c)),
            None,
        )
        if src is None:
            print(f"  [warn] frame {idx} missing under {frames_dir}, skipping")
            continue
        dst = os.path.join(out_dir, f"frame_{k}_t{k * stride / fps:.1f}s.{ext}")
        shutil.copy(src, dst)
        written.append(dst)
        print(f"  frame {idx:>6d}  ({k * stride / fps:.1f}s) -> {os.path.basename(dst)}")
    return written


def render_3d(args, out_png):
    cmd = [
        sys.executable, os.path.join(HERE, "render_world_traj.py"),
        "--seq_folder", args.seq_folder,
        "--out", out_png,
        "--num_samples", str(args.num_samples),
        "--frame_start", str(args.frame_start),
        "--frame_end", str(args.frame_end),
        "--hands", args.hands,
        "--view", args.view,
        "--cam_azim", str(args.cam_azim),
        "--cam_elev", str(args.cam_elev),
        "--cam_dist_scale", str(args.cam_dist_scale),
        "--width", str(args.width),
        "--height", str(args.height),
    ]
    if args.rebase:
        cmd.append("--rebase")
    if args.align:
        cmd.append("--align")
    if args.no_fade:
        cmd.append("--no_fade")
    if args.no_ground:
        cmd.append("--no_ground")
    if args.rrd:
        cmd += ["--rrd", args.rrd]
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    seq = os.path.abspath(args.seq_folder.rstrip("/"))
    if not os.path.exists(os.path.join(seq, "world_space_res.pth")):
        raise FileNotFoundError(f"no world_space_res.pth under {seq}")
    out_dir = args.out_dir or os.path.join(seq, "demo")
    os.makedirs(out_dir, exist_ok=True)

    if not args.skip_frames:
        frames_dir = args.frames_dir or default_frames_dir(seq)
        print(f"[frames] from {frames_dir}")
        copy_frames(frames_dir, out_dir, args.fps, args.num_frames, args.frame_ext)

    if not args.skip_3d:
        out_png = os.path.join(out_dir, "hand_cam_3d.png")
        print(f"[3d] -> {args.rrd if args.rrd else out_png}")
        render_3d(args, out_png)

    print(f"\ndone. demo images in {out_dir}")


if __name__ == "__main__":
    main()
