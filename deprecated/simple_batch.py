#!/usr/bin/env python3
"""
Simple batch processing script - runs demo_offline.py on multiple videos across 8 GPUs.
No fancy features, just simple parallel processing.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from multiprocessing import Process, Queue


def worker(gpu_id, video_queue, result_queue):
    """Worker process that processes videos on a specific GPU."""
    while True:
        try:
            video_path = video_queue.get(timeout=1)
        except:
            break

        if video_path is None:
            break

        print(f"[GPU {gpu_id}] Processing: {video_path}")

        # Run demo_offline.py for this video
        cmd = [
            sys.executable,
            "demo_offline.py",
            "--video", video_path,
            "--vis_mode", "world"
        ]

        env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}

        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout per video
            )

            if result.returncode == 0:
                print(f"[GPU {gpu_id}] ✓ Success: {video_path}")
                result_queue.put((video_path, True))
            else:
                print(f"[GPU {gpu_id}] ✗ Failed: {video_path}")
                print(f"Error: {result.stderr[:200]}")
                result_queue.put((video_path, False))
        except subprocess.TimeoutExpired:
            print(f"[GPU {gpu_id}] ✗ Timeout: {video_path}")
            result_queue.put((video_path, False))
        except Exception as e:
            print(f"[GPU {gpu_id}] ✗ Error: {video_path} - {e}")
            result_queue.put((video_path, False))


def main():
    parser = argparse.ArgumentParser(description="Simple batch processing for HaWoR")
    parser.add_argument("--video_list", required=True, help="Text file with video paths")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU IDs")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    args = parser.parse_args()

    # Read video list
    with open(args.video_list) as f:
        videos = [line.strip() for line in f if line.strip()]

    # Apply start/end slicing
    if args.end is not None:
        videos = videos[args.start:args.end]
    else:
        videos = videos[args.start:]

    print(f"Total videos to process: {len(videos)}")

    # Parse GPU IDs
    gpus = [int(g.strip()) for g in args.gpus.split(",")]
    print(f"Using GPUs: {gpus}")

    # Create queues
    video_queue = Queue()
    result_queue = Queue()

    # Add videos to queue
    for video in videos:
        video_queue.put(video)

    # Add stop signals
    for _ in gpus:
        video_queue.put(None)

    # Start workers
    workers = []
    for gpu in gpus:
        p = Process(target=worker, args=(gpu, video_queue, result_queue))
        p.start()
        workers.append(p)

    # Wait for all workers to finish
    for w in workers:
        w.join()

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Print summary
    success = sum(1 for _, s in results if s)
    failed = len(results) - success

    print(f"\n=== Summary ===")
    print(f"Total: {len(results)}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
