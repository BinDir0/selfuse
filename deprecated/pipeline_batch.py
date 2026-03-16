#!/usr/bin/env python3
"""
Pipeline-based batch processing for HaWoR - maximizes GPU utilization.

Instead of processing each video serially through all stages, this pipeline
allows multiple videos to be at different stages simultaneously, keeping
the GPU busy at all times.

Architecture:
    Video Queue -> Stage 1 (Detection) -> Stage 2 (Motion) ->
    Stage 3 (SLAM) -> Stage 4 (Infiller) -> Stage 5 (Vis) -> Done

Each stage can process multiple videos in parallel using batching.
"""
import argparse
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Process, Queue, Manager
from pathlib import Path
from queue import Empty

import torch


class PipelineStage:
    """Base class for a pipeline stage."""

    def __init__(self, stage_name, gpu_ids, input_queue, output_queue,
                 batch_size=1, stats_dict=None):
        self.stage_name = stage_name
        self.gpu_ids = gpu_ids
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.batch_size = batch_size
        self.stats_dict = stats_dict if stats_dict is not None else {}

    def process_video(self, video_path, gpu_id):
        """Process a single video. Override in subclasses."""
        raise NotImplementedError

    def run(self):
        """Main worker loop."""
        print(f"[{self.stage_name}] Worker started with GPUs: {self.gpu_ids}")

        while True:
            try:
                # Get video from input queue
                video_path = self.input_queue.get(timeout=1)

                if video_path is None:  # Stop signal
                    print(f"[{self.stage_name}] Received stop signal")
                    break

                # Select GPU (round-robin)
                gpu_id = self.gpu_ids[0] if len(self.gpu_ids) == 1 else \
                         self.gpu_ids[hash(video_path) % len(self.gpu_ids)]

                print(f"[{self.stage_name}] Processing: {video_path} on GPU {gpu_id}")
                start_time = time.time()

                try:
                    # Process the video
                    result = self.process_video(video_path, gpu_id)

                    elapsed = time.time() - start_time
                    print(f"[{self.stage_name}] ✓ Completed {video_path} in {elapsed:.1f}s")

                    # Update stats
                    if self.stats_dict is not None:
                        self.stats_dict[self.stage_name] = \
                            self.stats_dict.get(self.stage_name, 0) + 1

                    # Pass to next stage
                    if self.output_queue:
                        self.output_queue.put(video_path)

                except Exception as e:
                    print(f"[{self.stage_name}] ✗ Error processing {video_path}: {e}")
                    import traceback
                    traceback.print_exc()

            except Empty:
                continue
            except Exception as e:
                print(f"[{self.stage_name}] Worker error: {e}")
                break

        print(f"[{self.stage_name}] Worker stopped")


class DetectionStage(PipelineStage):
    """Stage 1: Detection and Tracking"""

    def process_video(self, video_path, gpu_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # Import here to avoid loading models in main process
        sys.path.insert(0, os.path.dirname(__file__))
        from scripts.scripts_test_video.detect_track_video import detect_track_video

        class Args:
            def __init__(self, video_path):
                self.video_path = video_path
                self.img_focal = None
                self.input_type = 'file'

        args = Args(video_path)
        detect_track_video(args)
        return True


class MotionEstimationStage(PipelineStage):
    """Stage 2: Motion Estimation"""

    def __init__(self, *args, checkpoint_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_path = checkpoint_path

    def process_video(self, video_path, gpu_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        sys.path.insert(0, os.path.dirname(__file__))
        from scripts.scripts_test_video.hawor_video import hawor_motion_estimation
        from glob import glob
        from natsort import natsorted

        class Args:
            def __init__(self, video_path, checkpoint):
                self.video_path = video_path
                self.img_focal = None
                self.checkpoint = checkpoint

        # Get start_idx and end_idx
        video_root = os.path.dirname(video_path)
        video = os.path.basename(video_path).split('.')[0]
        seq_folder = os.path.join(video_root, video)
        img_folder = f'{seq_folder}/extracted_images'
        imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))
        start_idx = 0
        end_idx = len(imgfiles)

        args = Args(video_path, self.checkpoint_path)
        hawor_motion_estimation(args, start_idx, end_idx, seq_folder)
        return True


class SLAMStage(PipelineStage):
    """Stage 3: SLAM"""

    def process_video(self, video_path, gpu_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        sys.path.insert(0, os.path.dirname(__file__))
        from scripts.scripts_test_video.hawor_slam import hawor_slam
        from glob import glob
        from natsort import natsorted

        class Args:
            def __init__(self, video_path):
                self.video_path = video_path
                self.img_focal = None

        # Get start_idx and end_idx
        video_root = os.path.dirname(video_path)
        video = os.path.basename(video_path).split('.')[0]
        seq_folder = os.path.join(video_root, video)
        img_folder = f'{seq_folder}/extracted_images'
        imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))
        start_idx = 0
        end_idx = len(imgfiles)

        args = Args(video_path)

        # Check if already done
        slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        if not os.path.exists(slam_path):
            hawor_slam(args, start_idx, end_idx)

        return True


class InfillerStage(PipelineStage):
    """Stage 4: Infiller"""

    def __init__(self, *args, infiller_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.infiller_weight = infiller_weight

    def process_video(self, video_path, gpu_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

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

        # Get start_idx, end_idx, and frame_chunks
        video_root = os.path.dirname(video_path)
        video = os.path.basename(video_path).split('.')[0]
        seq_folder = os.path.join(video_root, video)
        img_folder = f'{seq_folder}/extracted_images'
        imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))
        start_idx = 0
        end_idx = len(imgfiles)

        # Load frame_chunks from motion estimation
        frame_chunks_all = joblib.load(f'{seq_folder}/tracks_{start_idx}_{end_idx}/frame_chunks_all.npy')

        args = Args(video_path, self.infiller_weight)
        hawor_infiller(args, start_idx, end_idx, frame_chunks_all)
        return True


def pipeline_worker(stage_class, stage_name, gpu_ids, input_queue,
                   output_queue, stats_dict, **kwargs):
    """Worker process for a pipeline stage."""
    stage = stage_class(stage_name, gpu_ids, input_queue, output_queue,
                       stats_dict=stats_dict, **kwargs)
    stage.run()


def main():
    parser = argparse.ArgumentParser(description="Pipeline batch processing for HaWoR")
    parser.add_argument("--video_list", required=True, help="Text file with video paths")
    parser.add_argument("--checkpoint", default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU IDs")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)

    # Pipeline configuration
    parser.add_argument("--detection_gpus", default=None, help="GPUs for detection (default: all)")
    parser.add_argument("--motion_gpus", default=None, help="GPUs for motion estimation")
    parser.add_argument("--slam_gpus", default=None, help="GPUs for SLAM")
    parser.add_argument("--infiller_gpus", default=None, help="GPUs for infiller")

    parser.add_argument("--detection_workers", type=int, default=2)
    parser.add_argument("--motion_workers", type=int, default=2)
    parser.add_argument("--slam_workers", type=int, default=2)
    parser.add_argument("--infiller_workers", type=int, default=2)

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
    all_gpus = [int(g.strip()) for g in args.gpus.split(",")]
    detection_gpus = [int(g) for g in args.detection_gpus.split(",")] if args.detection_gpus else all_gpus
    motion_gpus = [int(g) for g in args.motion_gpus.split(",")] if args.motion_gpus else all_gpus
    slam_gpus = [int(g) for g in args.slam_gpus.split(",")] if args.slam_gpus else all_gpus
    infiller_gpus = [int(g) for g in args.infiller_gpus.split(",")] if args.infiller_gpus else all_gpus

    print(f"Detection GPUs: {detection_gpus}")
    print(f"Motion GPUs: {motion_gpus}")
    print(f"SLAM GPUs: {slam_gpus}")
    print(f"Infiller GPUs: {infiller_gpus}")

    # Create queues
    detection_queue = Queue(maxsize=100)
    motion_queue = Queue(maxsize=100)
    slam_queue = Queue(maxsize=100)
    infiller_queue = Queue(maxsize=100)

    # Shared stats
    manager = Manager()
    stats = manager.dict()

    # Start workers for each stage
    workers = []

    # Detection workers
    for i in range(args.detection_workers):
        p = Process(target=pipeline_worker,
                   args=(DetectionStage, f"Detection-{i}", detection_gpus,
                        detection_queue, motion_queue, stats))
        p.start()
        workers.append(p)

    # Motion estimation workers
    for i in range(args.motion_workers):
        p = Process(target=pipeline_worker,
                   args=(MotionEstimationStage, f"Motion-{i}", motion_gpus,
                        motion_queue, slam_queue, stats),
                   kwargs={'checkpoint_path': args.checkpoint})
        p.start()
        workers.append(p)

    # SLAM workers
    for i in range(args.slam_workers):
        p = Process(target=pipeline_worker,
                   args=(SLAMStage, f"SLAM-{i}", slam_gpus,
                        slam_queue, infiller_queue, stats))
        p.start()
        workers.append(p)

    # Infiller workers
    for i in range(args.infiller_workers):
        p = Process(target=pipeline_worker,
                   args=(InfillerStage, f"Infiller-{i}", infiller_gpus,
                        infiller_queue, None, stats),
                   kwargs={'infiller_weight': args.infiller_weight})
        p.start()
        workers.append(p)

    # Feed videos to detection queue
    print("Feeding videos to pipeline...")
    for video in videos:
        detection_queue.put(video)

    # Send stop signals
    for _ in range(args.detection_workers):
        detection_queue.put(None)

    # Monitor progress
    print("\nMonitoring pipeline progress...")
    try:
        while any(w.is_alive() for w in workers):
            time.sleep(5)
            print(f"Progress: Detection={stats.get('Detection-0', 0)}, "
                  f"Motion={stats.get('Motion-0', 0)}, "
                  f"SLAM={stats.get('SLAM-0', 0)}, "
                  f"Infiller={stats.get('Infiller-0', 0)}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    # Wait for all workers
    for w in workers:
        w.join()

    print("\n=== Pipeline Complete ===")
    print(f"Total videos processed: {len(videos)}")


if __name__ == "__main__":
    main()

