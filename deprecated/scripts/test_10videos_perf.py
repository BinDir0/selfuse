#!/usr/bin/env python3
"""
Quick performance + correctness test for HaWoR batch inference.

Default test protocol:
- 10 videos
- 1 GPU
- configurable chunk_batch_size

It runs batch_infer once, then reports:
- total elapsed time
- avg sec/video
- success/failed counts from status.json
- output correctness check (world_space_res.pth existence)
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def read_video_list(path: Path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def write_video_subset(videos, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for v in videos:
            f.write(v + "\n")


def video_seq_folder(video_path: str) -> Path:
    p = Path(video_path)
    return p.parent / p.stem


def check_outputs(videos):
    missing = []
    for v in videos:
        world_file = video_seq_folder(v) / "world_space_res.pth"
        if not world_file.exists():
            missing.append(str(world_file))
    return missing


def load_status(status_file: Path):
    if not status_file.exists():
        return None
    with open(status_file, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="10-video 1-GPU perf test")
    parser.add_argument("--video_list", required=True, help="Path to full video list txt")
    parser.add_argument("--num_videos", type=int, default=10, help="How many videos to test")
    parser.add_argument("--gpu", type=int, default=0, help="Single GPU id")
    parser.add_argument("--chunk_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16, help="Number of DataLoader workers for parallel frame loading")
    parser.add_argument("--render_batch_size", type=int, default=8, help="Rendering batch size")
    parser.add_argument("--checkpoint", default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--stages", default="detect_track,motion,slam,infiller")
    parser.add_argument("--run_root", default="./batch_runs/perf_tests")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun by disabling resume in batch_infer",
    )
    args = parser.parse_args()

    all_videos = read_video_list(Path(args.video_list))
    if len(all_videos) < args.num_videos:
        raise ValueError(f"video_list has only {len(all_videos)} videos, need {args.num_videos}")

    test_videos = all_videos[: args.num_videos]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / f"gpu{args.gpu}_n{args.num_videos}_chunk{args.chunk_batch_size}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    subset_file = run_dir / "videos_subset.txt"
    write_video_subset(test_videos, subset_file)

    cmd = [
        args.python_bin,
        "scripts/batch_infer.py",
        "--video_list", str(subset_file),
        "--gpus", str(args.gpu),
        "--stages", args.stages,
        "--chunk_batch_size", str(args.chunk_batch_size),
        "--render_batch_size", str(args.render_batch_size),
        "--checkpoint", args.checkpoint,
        "--infiller_weight", args.infiller_weight,
        "--run_dir", str(run_dir / "batch_run"),
    ]

    if args.force:
        cmd.append("--no-resume")

    print("=== Running perf test ===")
    print("Command:", " ".join(cmd))

    start = time.time()
    proc = subprocess.run(cmd)
    elapsed = time.time() - start

    status_file = run_dir / "batch_run" / "status.json"
    status = load_status(status_file)

    completed = 0
    failed = 0
    if status and "tasks" in status:
        for _, task in status["tasks"].items():
            stage_status = task.get("stage_status", {})
            if stage_status.get("infiller") == "completed":
                completed += 1
            elif any(v == "failed" for v in stage_status.values()):
                failed += 1

    missing_outputs = check_outputs(test_videos)

    report = {
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
        "avg_sec_per_video": round(elapsed / args.num_videos, 3),
        "num_videos": args.num_videos,
        "gpu": args.gpu,
        "chunk_batch_size": args.chunk_batch_size,
        "force": args.force,
        "stages": args.stages,
        "completed_from_status": completed,
        "failed_from_status": failed,
        "missing_world_outputs": len(missing_outputs),
        "missing_world_output_paths": missing_outputs,
        "run_dir": str(run_dir),
        "status_file": str(status_file),
    }

    report_file = run_dir / "report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Test Report ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved: {report_file}")


if __name__ == "__main__":
    main()
