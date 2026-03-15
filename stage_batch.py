#!/usr/bin/env python3
"""
Stage-based batch processing for HaWoR - Simple and efficient.

Process all videos stage by stage:
1. Run detection on ALL videos (8 GPUs in parallel)
2. Run motion estimation on ALL videos (8 GPUs in parallel)
3. Run SLAM on ALL videos (8 GPUs in parallel)
4. Run infiller on ALL videos (8 GPUs in parallel)

Each GPU processes one video at a time, but 8 GPUs work simultaneously.
Much simpler than pipeline approach and easier to resume.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool, Manager
from pathlib import Path
from functools import partial


def check_stage_complete(video_path, stage):
    """Check if a stage is complete for a video."""
    video_root = os.path.dirname(video_path)
    video = os.path.basename(video_path).split('.')[0]
    seq_folder = os.path.join(video_root, video)

    # Get frame count
    img_folder = f'{seq_folder}/extracted_images'
    if not os.path.exists(img_folder):
        return False

    from glob import glob
    from natsort import natsorted
    imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))
    if len(imgfiles) == 0:
        return False

    start_idx = 0
    end_idx = len(imgfiles)

    if stage == 'detection':
        return os.path.exists(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy')
    elif stage == 'motion':
        return os.path.exists(f'{seq_folder}/tracks_{start_idx}_{end_idx}/frame_chunks_all.npy')
    elif stage == 'slam':
        return os.path.exists(f'{seq_folder}/SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz')
    elif stage == 'infiller':
        # Check if infiller output exists
        return os.path.exists(f'{seq_folder}/tracks_{start_idx}_{end_idx}/pred_trans.npy')

    return False


def process_detection(video_path, gpu_id, stats):
    """Process detection stage for one video."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        # Check if already done
        if check_stage_complete(video_path, 'detection'):
            print(f"[Detection-GPU{gpu_id}] Skip (done): {video_path}")
            stats['detection_skip'] = stats.get('detection_skip', 0) + 1
            return True

        print(f"[Detection-GPU{gpu_id}] Processing: {video_path}")
        start_time = time.time()

        sys.path.insert(0, os.path.dirname(__file__))
        from scripts.scripts_test_video.detect_track_video import detect_track_video

        class Args:
            def __init__(self, video_path):
                self.video_path = video_path
                self.img_focal = None
                self.input_type = 'file'

        args = Args(video_path)
        detect_track_video(args)

        elapsed = time.time() - start_time
        print(f"[Detection-GPU{gpu_id}] ✓ Done in {elapsed:.1f}s: {video_path}")
        stats['detection_done'] = stats.get('detection_done', 0) + 1
        return True

    except Exception as e:
        print(f"[Detection-GPU{gpu_id}] ✗ Error: {video_path}")
        print(f"  {e}")
        stats['detection_fail'] = stats.get('detection_fail', 0) + 1
        return False


def process_motion(video_path, gpu_id, stats, checkpoint_path):
    """Process motion estimation stage for one video."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        # Check if already done
        if check_stage_complete(video_path, 'motion'):
            print(f"[Motion-GPU{gpu_id}] Skip (done): {video_path}")
            stats['motion_skip'] = stats.get('motion_skip', 0) + 1
            return True

        print(f"[Motion-GPU{gpu_id}] Processing: {video_path}")
        start_time = time.time()

        sys.path.insert(0, os.path.dirname(__file__))
        from scripts.scripts_test_video.hawor_video import hawor_motion_estimation
        from glob import glob
        from natsort import natsorted

        class Args:
            def __init__(self, video_path, checkpoint):
                self.video_path = video_path
                self.img_focal = None
                self.checkpoint = checkpoint

        video_root = os.path.dirname(video_path)
        video = os.path.basename(video_path).split('.')[0]
        seq_folder = os.path.join(video_root, video)
        img_folder = f'{seq_folder}/extracted_images'
        imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))
        start_idx = 0
        end_idx = len(imgfiles)

        args = Args(video_path, checkpoint_path)
        hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

        elapsed = time.time() - start_time
        print(f"[Motion-GPU{gpu_id}] ✓ Done in {elapsed:.1f}s: {video_path}")
        stats['motion_done'] = stats.get('motion_done', 0) + 1
        return True

    except Exception as e:
        print(f"[Motion-GPU{gpu_id}] ✗ Error: {video_path}")
        print(f"  {e}")
        stats['motion_fail'] = stats.get('motion_fail', 0) + 1
        return False


def process_slam(video_path, gpu_id, stats):
    """Process SLAM stage for one video."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        # Check if already done
        if check_stage_complete(video_path, 'slam'):
            print(f"[SLAM-GPU{gpu_id}] Skip (done): {video_path}")
            stats['slam_skip'] = stats.get('slam_skip', 0) + 1
            return True

        print(f"[SLAM-GPU{gpu_id}] Processing: {video_path}")
        start_time = time.time()

        sys.path.insert(0, os.path.dirname(__file__))
        from scripts.scripts_test_video.hawor_slam import hawor_slam
        from glob import glob
        from natsort import natsorted

        class Args:
            def __init__(self, video_path):
                self.video_path = video_path
                self.img_focal = None

        video_root = os.path.dirname(video_path)
        video = os.path.basename(video_path).split('.')[0]
        seq_folder = os.path.join(video_root, video)
        img_folder = f'{seq_folder}/extracted_images'
        imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))
        start_idx = 0
        end_idx = len(imgfiles)

        args = Args(video_path)
        hawor_slam(args, start_idx, end_idx)

        elapsed = time.time() - start_time
        print(f"[SLAM-GPU{gpu_id}] ✓ Done in {elapsed:.1f}s: {video_path}")
        stats['slam_done'] = stats.get('slam_done', 0) + 1
        return True

    except Exception as e:
        print(f"[SLAM-GPU{gpu_id}] ✗ Error: {video_path}")
        print(f"  {e}")
        stats['slam_fail'] = stats.get('slam_fail', 0) + 1
        return False


def process_infiller(video_path, gpu_id, stats, infiller_weight):
    """Process infiller stage for one video."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        # Check if already done
        if check_stage_complete(video_path, 'infiller'):
            print(f"[Infiller-GPU{gpu_id}] Skip (done): {video_path}")
            stats['infiller_skip'] = stats.get('infiller_skip', 0) + 1
            return True

        print(f"[Infiller-GPU{gpu_id}] Processing: {video_path}")
        start_time = time.time()

        sys.path.insert(0, os.path.dirname(__file__))
        from scripts.scripts_test_video.hawor_video import hawor_infiller
        from glob import glob
        from natsort import natsorted
        import joblib

        class Args:
            def __init__(self, video_path, infiller_weight):
                self.video_path = video_path
                self.img_focal = None
                self.infiller_weight = infiller_weight

        video_root = os.path.dirname(video_path)
        video = os.path.basename(video_path).split('.')[0]
        seq_folder = os.path.join(video_root, video)
        img_folder = f'{seq_folder}/extracted_images'
        imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))
        start_idx = 0
        end_idx = len(imgfiles)

        frame_chunks_all = joblib.load(f'{seq_folder}/tracks_{start_idx}_{end_idx}/frame_chunks_all.npy')

        args = Args(video_path, infiller_weight)
        hawor_infiller(args, start_idx, end_idx, frame_chunks_all)

        elapsed = time.time() - start_time
        print(f"[Infiller-GPU{gpu_id}] ✓ Done in {elapsed:.1f}s: {video_path}")
        stats['infiller_done'] = stats.get('infiller_done', 0) + 1
        return True

    except Exception as e:
        print(f"[Infiller-GPU{gpu_id}] ✗ Error: {video_path}")
        print(f"  {e}")
        stats['infiller_fail'] = stats.get('infiller_fail', 0) + 1
        return False


def worker_wrapper(args):
    """Wrapper to unpack arguments for pool.map."""
    func, video_path, gpu_id, stats, extra_args = args
    return func(video_path, gpu_id, stats, *extra_args)


def run_stage(stage_name, videos, gpu_ids, process_func, stats, extra_args=()):
    """Run a stage on all videos using multiple GPUs."""
    print(f"\n{'='*60}")
    print(f"Stage: {stage_name}")
    print(f"Videos: {len(videos)}")
    print(f"GPUs: {gpu_ids}")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Create tasks: each video gets assigned to a GPU (round-robin)
    tasks = []
    for i, video in enumerate(videos):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        tasks.append((process_func, video, gpu_id, stats, extra_args))

    # Process in parallel
    with Pool(len(gpu_ids)) as pool:
        results = pool.map(worker_wrapper, tasks)

    elapsed = time.time() - start_time
    success = sum(results)

    print(f"\n{'='*60}")
    print(f"{stage_name} Complete")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Success: {success}/{len(videos)}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Stage-based batch processing")
    parser.add_argument("--video_list", required=True)
    parser.add_argument("--checkpoint", default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stages", default="detection,motion,slam,infiller",
                       help="Comma-separated stages to run")

    args = parser.parse_args()

    # Read videos
    with open(args.video_list) as f:
        videos = [line.strip() for line in f if line.strip()]

    if args.end is not None:
        videos = videos[args.start:args.end]
    else:
        videos = videos[args.start:]

    print(f"Total videos: {len(videos)}")

    # Parse GPUs
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    print(f"GPUs: {gpu_ids}")

    # Shared stats
    manager = Manager()
    stats = manager.dict()

    # Run stages
    stages_to_run = args.stages.split(",")

    if 'detection' in stages_to_run:
        run_stage("Detection", videos, gpu_ids, process_detection, stats)

    if 'motion' in stages_to_run:
        run_stage("Motion Estimation", videos, gpu_ids, process_motion, stats,
                 extra_args=(args.checkpoint,))

    if 'slam' in stages_to_run:
        run_stage("SLAM", videos, gpu_ids, process_slam, stats)

    if 'infiller' in stages_to_run:
        run_stage("Infiller", videos, gpu_ids, process_infiller, stats,
                 extra_args=(args.infiller_weight,))

    # Final summary
    print(f"\n{'='*60}")
    print("Final Summary")
    print(f"{'='*60}")
    for key, value in sorted(stats.items()):
        print(f"{key}: {value}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
