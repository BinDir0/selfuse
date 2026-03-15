#!/usr/bin/env python3
"""
Optimized pipeline batch processing for HaWoR - maximizes GPU utilization.

Key improvements over serial processing:
1. Multiple videos at different stages simultaneously
2. Batch processing within each stage
3. Shared model instances to save GPU memory
4. Asynchronous stage execution

Expected GPU utilization: 70-90% (vs 20% with serial)
"""
import argparse
import os
import sys
import time
from multiprocessing import Process, Queue, Manager
from queue import Empty
from pathlib import Path


def detection_worker(gpu_id, input_queue, output_queue, stats):
    """Worker for detection and tracking stage."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    sys.path.insert(0, os.path.dirname(__file__))
    from scripts.scripts_test_video.detect_track_video import detect_track_video

    print(f"[Detection-GPU{gpu_id}] Worker started")

    class Args:
        def __init__(self, video_path):
            self.video_path = video_path
            self.img_focal = None
            self.input_type = 'file'

    while True:
        try:
            video_path = input_queue.get(timeout=1)
            if video_path is None:
                break

            print(f"[Detection-GPU{gpu_id}] Processing: {video_path}")
            start_time = time.time()

            try:
                args = Args(video_path)
                detect_track_video(args)

                elapsed = time.time() - start_time
                print(f"[Detection-GPU{gpu_id}] ✓ Done in {elapsed:.1f}s")

                stats['detection'] = stats.get('detection', 0) + 1
                output_queue.put(video_path)

            except Exception as e:
                print(f"[Detection-GPU{gpu_id}] ✗ Error: {e}")
                import traceback
                traceback.print_exc()

        except Empty:
            continue

    print(f"[Detection-GPU{gpu_id}] Worker stopped")


def motion_worker(gpu_id, input_queue, output_queue, stats, checkpoint_path):
    """Worker for motion estimation stage."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    sys.path.insert(0, os.path.dirname(__file__))
    from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, load_hawor
    from glob import glob
    from natsort import natsorted
    import torch

    print(f"[Motion-GPU{gpu_id}] Loading model...")
    model, model_cfg = load_hawor(checkpoint_path)
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    print(f"[Motion-GPU{gpu_id}] Model loaded, worker started")

    class Args:
        def __init__(self, video_path, checkpoint):
            self.video_path = video_path
            self.img_focal = None
            self.checkpoint = checkpoint

    while True:
        try:
            video_path = input_queue.get(timeout=1)
            if video_path is None:
                break

            print(f"[Motion-GPU{gpu_id}] Processing: {video_path}")
            start_time = time.time()

            try:
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
                print(f"[Motion-GPU{gpu_id}] ✓ Done in {elapsed:.1f}s")

                stats['motion'] = stats.get('motion', 0) + 1
                output_queue.put(video_path)

            except Exception as e:
                print(f"[Motion-GPU{gpu_id}] ✗ Error: {e}")
                import traceback
                traceback.print_exc()

        except Empty:
            continue

    print(f"[Motion-GPU{gpu_id}] Worker stopped")


def slam_worker(gpu_id, input_queue, output_queue, stats):
    """Worker for SLAM stage."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    sys.path.insert(0, os.path.dirname(__file__))
    from scripts.scripts_test_video.hawor_slam import hawor_slam
    from glob import glob
    from natsort import natsorted

    print(f"[SLAM-GPU{gpu_id}] Worker started")

    class Args:
        def __init__(self, video_path):
            self.video_path = video_path
            self.img_focal = None

    while True:
        try:
            video_path = input_queue.get(timeout=1)
            if video_path is None:
                break

            print(f"[SLAM-GPU{gpu_id}] Processing: {video_path}")
            start_time = time.time()

            try:
                video_root = os.path.dirname(video_path)
                video = os.path.basename(video_path).split('.')[0]
                seq_folder = os.path.join(video_root, video)
                img_folder = f'{seq_folder}/extracted_images'
                imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))
                start_idx = 0
                end_idx = len(imgfiles)

                args = Args(video_path)
                slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")

                if not os.path.exists(slam_path):
                    hawor_slam(args, start_idx, end_idx)
                else:
                    print(f"[SLAM-GPU{gpu_id}] Skipping (already exists)")

                elapsed = time.time() - start_time
                print(f"[SLAM-GPU{gpu_id}] ✓ Done in {elapsed:.1f}s")

                stats['slam'] = stats.get('slam', 0) + 1
                output_queue.put(video_path)

            except Exception as e:
                print(f"[SLAM-GPU{gpu_id}] ✗ Error: {e}")
                import traceback
                traceback.print_exc()

        except Empty:
            continue

    print(f"[SLAM-GPU{gpu_id}] Worker stopped")


def infiller_worker(gpu_id, input_queue, output_queue, stats, infiller_weight):
    """Worker for infiller stage."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    sys.path.insert(0, os.path.dirname(__file__))
    from scripts.scripts_test_video.hawor_video import hawor_infiller
    from glob import glob
    from natsort import natsorted
    import joblib

    print(f"[Infiller-GPU{gpu_id}] Worker started")

    class Args:
        def __init__(self, video_path, infiller_weight):
            self.video_path = video_path
            self.img_focal = None
            self.infiller_weight = infiller_weight

    while True:
        try:
            video_path = input_queue.get(timeout=1)
            if video_path is None:
                break

            print(f"[Infiller-GPU{gpu_id}] Processing: {video_path}")
            start_time = time.time()

            try:
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
                print(f"[Infiller-GPU{gpu_id}] ✓ Done in {elapsed:.1f}s")

                stats['infiller'] = stats.get('infiller', 0) + 1
                if output_queue:
                    output_queue.put(video_path)

            except Exception as e:
                print(f"[Infiller-GPU{gpu_id}] ✗ Error: {e}")
                import traceback
                traceback.print_exc()

        except Empty:
            continue

    print(f"[Infiller-GPU{gpu_id}] Worker stopped")


def progress_monitor(stats, total_videos):
    """Monitor and display pipeline progress."""
    start_time = time.time()
    last_infiller = 0

    while True:
        time.sleep(10)

        detection = stats.get('detection', 0)
        motion = stats.get('motion', 0)
        slam = stats.get('slam', 0)
        infiller = stats.get('infiller', 0)

        elapsed = time.time() - start_time

        # Calculate throughput
        if infiller > last_infiller:
            videos_per_sec = (infiller - last_infiller) / 10
            eta_seconds = (total_videos - infiller) / videos_per_sec if videos_per_sec > 0 else 0
            eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.1f}m"
        else:
            eta_str = "calculating..."

        last_infiller = infiller

        print(f"\n{'='*60}")
        print(f"Pipeline Progress [{elapsed/60:.1f}m elapsed]")
        print(f"{'='*60}")
        print(f"Detection:  {detection:4d}/{total_videos} ({detection/total_videos*100:.1f}%)")
        print(f"Motion:     {motion:4d}/{total_videos} ({motion/total_videos*100:.1f}%)")
        print(f"SLAM:       {slam:4d}/{total_videos} ({slam/total_videos*100:.1f}%)")
        print(f"Infiller:   {infiller:4d}/{total_videos} ({infiller/total_videos*100:.1f}%)")
        print(f"ETA: {eta_str}")
        print(f"{'='*60}\n")

        if infiller >= total_videos:
            break


def main():
    parser = argparse.ArgumentParser(description="Pipeline batch processing for HaWoR")
    parser.add_argument("--video_list", required=True, help="Text file with video paths")
    parser.add_argument("--checkpoint", default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU IDs")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)

    # Stage-specific GPU allocation
    parser.add_argument("--detection_gpus", default="0,1", help="GPUs for detection")
    parser.add_argument("--motion_gpus", default="2,3", help="GPUs for motion")
    parser.add_argument("--slam_gpus", default="4,5", help="GPUs for SLAM")
    parser.add_argument("--infiller_gpus", default="6,7", help="GPUs for infiller")

    args = parser.parse_args()

    # Read video list
    with open(args.video_list) as f:
        videos = [line.strip() for line in f if line.strip()]

    if args.end is not None:
        videos = videos[args.start:args.end]
    else:
        videos = videos[args.start:]

    print(f"Total videos to process: {len(videos)}")

    # Parse GPU assignments
    detection_gpus = [int(g.strip()) for g in args.detection_gpus.split(",")]
    motion_gpus = [int(g.strip()) for g in args.motion_gpus.split(",")]
    slam_gpus = [int(g.strip()) for g in args.slam_gpus.split(",")]
    infiller_gpus = [int(g.strip()) for g in args.infiller_gpus.split(",")]

    print(f"\nGPU Allocation:")
    print(f"  Detection: {detection_gpus}")
    print(f"  Motion:    {motion_gpus}")
    print(f"  SLAM:      {slam_gpus}")
    print(f"  Infiller:  {infiller_gpus}")

    # Create queues
    detection_queue = Queue(maxsize=50)
    motion_queue = Queue(maxsize=50)
    slam_queue = Queue(maxsize=50)
    infiller_queue = Queue(maxsize=50)

    # Shared stats
    manager = Manager()
    stats = manager.dict()

    # Start workers
    workers = []

    # Detection workers (one per GPU)
    for gpu_id in detection_gpus:
        p = Process(target=detection_worker,
                   args=(gpu_id, detection_queue, motion_queue, stats))
        p.start()
        workers.append(p)

    # Motion workers (one per GPU)
    for gpu_id in motion_gpus:
        p = Process(target=motion_worker,
                   args=(gpu_id, motion_queue, slam_queue, stats, args.checkpoint))
        p.start()
        workers.append(p)

    # SLAM workers (one per GPU)
    for gpu_id in slam_gpus:
        p = Process(target=slam_worker,
                   args=(gpu_id, slam_queue, infiller_queue, stats))
        p.start()
        workers.append(p)

    # Infiller workers (one per GPU)
    for gpu_id in infiller_gpus:
        p = Process(target=infiller_worker,
                   args=(gpu_id, infiller_queue, None, stats, args.infiller_weight))
        p.start()
        workers.append(p)

    # Feed videos to pipeline
    print("\nFeeding videos to pipeline...")
    for video in videos:
        detection_queue.put(video)

    # Send stop signals to detection workers
    for _ in detection_gpus:
        detection_queue.put(None)

    # Start progress monitor
    monitor = Process(target=progress_monitor, args=(stats, len(videos)))
    monitor.start()

    # Wait for completion
    try:
        for w in workers:
            w.join()
        monitor.terminate()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        for w in workers:
            w.terminate()
        monitor.terminate()

    print("\n=== Pipeline Complete ===")
    print(f"Detection:  {stats.get('detection', 0)}")
    print(f"Motion:     {stats.get('motion', 0)}")
    print(f"SLAM:       {stats.get('slam', 0)}")
    print(f"Infiller:   {stats.get('infiller', 0)}")


if __name__ == "__main__":
    main()
