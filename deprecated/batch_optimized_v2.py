#!/usr/bin/env python3
"""
Optimized batch processing - Focus on maximizing GPU utilization.

Strategy:
1. Multi-GPU parallel: 8 GPUs process 8 videos simultaneously
2. Larger batch size within each video: Process more frames/chunks at once
3. Async data loading: Overlap I/O with computation

This is more practical than cross-video batching due to:
- YOLO tracking requires sequential processing per video
- Variable video lengths make cross-video batching complex
"""
import argparse
import os
import sys
from multiprocessing import Pool, Manager
from functools import partial

# Import the stage processing functions from stage_batch.py
sys.path.insert(0, os.path.dirname(__file__))


def process_detection_optimized(video_path, gpu_id, stats, frame_batch_size=32):
    """
    Optimized detection with larger batch size.

    Key optimization: Process multiple frames in one YOLO forward pass.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        from scripts.scripts_test_video.detect_track_video import detect_track_video
        from glob import glob
        from natsort import natsorted
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO
        from tqdm import tqdm

        video_root = os.path.dirname(video_path)
        video = os.path.basename(video_path).split('.')[0]
        seq_folder = os.path.join(video_root, video)
        img_folder = f'{seq_folder}/extracted_images'

        if not os.path.exists(img_folder):
            return False

        imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))
        start_idx = 0
        end_idx = len(imgfiles)

        # Check if already done
        if os.path.exists(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy'):
            print(f"[Detection-GPU{gpu_id}] Skip (done): {video_path}")
            stats['detection_skip'] = stats.get('detection_skip', 0) + 1
            return True

        print(f"[Detection-GPU{gpu_id}] Processing: {video_path} ({len(imgfiles)} frames)")
        os.makedirs(f"{seq_folder}/tracks_{start_idx}_{end_idx}", exist_ok=True)

        # Load model
        hand_det_model = YOLO('./weights/external/detector.pt')

        # Process with larger batch size
        boxes_all = []
        tracks = {}
        fallback_counter = 0
        track_last_seen = {}

        # Process frames in batches
        for batch_start in tqdm(range(0, len(imgfiles), frame_batch_size),
                               desc=f"GPU{gpu_id}", leave=False):
            batch_end = min(batch_start + frame_batch_size, len(imgfiles))
            batch_imgfiles = imgfiles[batch_start:batch_end]

            # Load batch of images
            batch_images = [cv2.imread(str(f)) for f in batch_imgfiles]

            # Batch inference - YOLO can process multiple images at once
            with torch.no_grad():
                results_batch = hand_det_model.track(
                    batch_images,
                    conf=0.35,
                    persist=True,
                    verbose=False,
                    # Optimize YOLO settings for batch processing
                    half=True,  # Use FP16 for faster inference
                    device=0
                )

                # Process results
                for t_local, results in enumerate(results_batch):
                    t = batch_start + t_local
                    img_h, img_w = batch_images[t_local].shape[:2]

                    boxes = results.boxes.xyxy.cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()
                    handedness = results.boxes.cls.cpu().numpy()

                    if results.boxes.id is not None:
                        track_id = results.boxes.id.cpu().numpy()
                    else:
                        track_id = [-1] * len(boxes)

                    if len(boxes) == 0:
                        continue

                    boxes = np.hstack([boxes, confs[:, None]])

                    # Edge detection parameters
                    edge_margin_ratio = 0.1
                    min_edge_conf = 0.4
                    edge_left = img_w * edge_margin_ratio
                    edge_right = img_w * (1 - edge_margin_ratio)
                    edge_top = img_h * edge_margin_ratio
                    edge_bottom = img_h * (1 - edge_margin_ratio)

                    find_right = False
                    find_left = False

                    for idx in range(len(boxes)):
                        x1, y1, x2, y2, conf = boxes[idx]
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        is_near_edge = (cx < edge_left or cx > edge_right or
                                       cy < edge_top or cy > edge_bottom)

                        if is_near_edge and conf < min_edge_conf:
                            continue

                        if track_id[idx] == -1:
                            if handedness[idx] > 0:
                                id = int(10000 + fallback_counter)
                            else:
                                id = int(5000 + fallback_counter)
                            fallback_counter += 1
                        else:
                            id = int(track_id[idx])

                        if id in track_last_seen:
                            frames_since_last = t - track_last_seen[id]
                            if frames_since_last > 10 and conf < min_edge_conf:
                                continue

                        subj = {
                            'frame': t,
                            'det': True,
                            'det_box': boxes[[idx]],
                            'det_handedness': handedness[[idx]],
                            'is_near_edge': is_near_edge
                        }

                        if (not find_right and handedness[idx] > 0) or \
                           (not find_left and handedness[idx] == 0):
                            if id in tracks:
                                tracks[id].append(subj)
                            else:
                                tracks[id] = [subj]

                            track_last_seen[id] = t

                            if handedness[idx] > 0:
                                find_right = True
                            elif handedness[idx] == 0:
                                find_left = True

        # Save results
        tracks_array = np.array(tracks, dtype=object)
        boxes_array = np.array(boxes_all, dtype=object)
        np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy', boxes_array)
        np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_tracks.npy', tracks_array)

        print(f"[Detection-GPU{gpu_id}] ✓ Done: {video_path}")
        stats['detection_done'] = stats.get('detection_done', 0) + 1
        return True

    except Exception as e:
        print(f"[Detection-GPU{gpu_id}] ✗ Error: {video_path}")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        stats['detection_fail'] = stats.get('detection_fail', 0) + 1
        return False


def worker_wrapper_detection(args):
    """Wrapper for detection worker."""
    video_path, gpu_id, stats, frame_batch_size = args
    return process_detection_optimized(video_path, gpu_id, stats, frame_batch_size)


def run_detection_stage(videos, gpu_ids, stats, frame_batch_size=32):
    """Run detection stage with optimized batching."""
    print(f"\n{'='*60}")
    print(f"Optimized Detection Stage")
    print(f"Videos: {len(videos)}")
    print(f"GPUs: {gpu_ids}")
    print(f"Frame batch size: {frame_batch_size}")
    print(f"{'='*60}\n")

    # Create tasks
    tasks = []
    for i, video in enumerate(videos):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        tasks.append((video, gpu_id, stats, frame_batch_size))

    # Process in parallel
    with Pool(len(gpu_ids)) as pool:
        results = pool.map(worker_wrapper_detection, tasks)

    success = sum(results)
    print(f"\n✓ Detection complete: {success}/{len(videos)} successful\n")


def main():
    parser = argparse.ArgumentParser(description="Optimized batch processing")
    parser.add_argument("--video_list", required=True)
    parser.add_argument("--checkpoint", default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stages", default="detection,motion,slam,infiller")

    # Batch size parameters
    parser.add_argument("--detection_batch_size", type=int, default=32,
                       help="Number of frames to process in one batch (detection)")
    parser.add_argument("--motion_batch_size", type=int, default=64,
                       help="Batch size for motion estimation")

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

    stages_to_run = args.stages.split(",")

    # Run detection with optimized batching
    if 'detection' in stages_to_run:
        run_detection_stage(videos, gpu_ids, stats,
                           frame_batch_size=args.detection_batch_size)

    # For other stages, use the original stage_batch.py functions
    if 'motion' in stages_to_run:
        print("\nRunning motion estimation...")
        from stage_batch import process_motion, worker_wrapper, run_stage
        run_stage("Motion Estimation", videos, gpu_ids, process_motion, stats,
                 extra_args=(args.checkpoint,))

    if 'slam' in stages_to_run:
        print("\nRunning SLAM...")
        from stage_batch import process_slam, worker_wrapper, run_stage
        run_stage("SLAM", videos, gpu_ids, process_slam, stats)

    if 'infiller' in stages_to_run:
        print("\nRunning infiller...")
        from stage_batch import process_infiller, worker_wrapper, run_stage
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
