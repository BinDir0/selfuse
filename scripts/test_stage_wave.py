#!/usr/bin/env python3
"""
Stage-wave scheduling test script.

Tests the new stage-wave mode against legacy mode with a small video set.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_batch_infer(video_list, gpus, mode, run_dir, force=False):
    """Run batch inference with specified mode."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "batch_infer.py"),
        "--video_list", str(video_list),
        "--gpus", gpus,
        "--scheduler_mode", mode,
        "--run_dir", str(run_dir),
    ]

    if mode == "wave":
        cmd.append("--persistent_worker")

    if force:
        cmd.append("--no-resume")

    print(f"\n{'='*60}")
    print(f"Running: {mode.upper()} mode")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start_time = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - start_time

    return result.returncode == 0, elapsed


def parse_events(events_file):
    """Parse events.jsonl and extract metrics."""
    if not events_file.exists():
        return None

    events = []
    with open(events_file) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    metrics = {
        "total_videos": 0,
        "success": 0,
        "failed": 0,
        "elapsed_sec": 0,
    }

    for event in events:
        if event.get("event") == "batch_start":
            metrics["total_videos"] = event.get("total_videos", 0)
        elif event.get("event") == "batch_end":
            metrics["success"] = event.get("success", 0)
            metrics["failed"] = event.get("failed", 0)

    # Find batch start and end times
    if events:
        start_time = datetime.fromisoformat(events[0]["time"])
        end_time = datetime.fromisoformat(events[-1]["time"])
        metrics["elapsed_sec"] = (end_time - start_time).total_seconds()

    return metrics


def check_outputs(video_list, mode_name):
    """Check if all expected outputs exist."""
    with open(video_list) as f:
        video_paths = [line.strip() for line in f if line.strip()]

    missing = []
    for vp in video_paths:
        vp_path = Path(vp)
        seq_folder = vp_path.parent / vp_path.stem
        world_res = seq_folder / "world_space_res.pth"

        if not world_res.exists():
            missing.append(vp)

    print(f"\n{mode_name} Output Check:")
    print(f"  Total videos: {len(video_paths)}")
    print(f"  Missing world_space_res.pth: {len(missing)}")

    if missing:
        print(f"  Missing for: {missing[:3]}{'...' if len(missing) > 3 else ''}")

    return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(description="Test stage-wave scheduling")
    parser.add_argument(
        "--video_list",
        type=str,
        required=True,
        help="Path to video list file (recommend 3-5 videos for quick test)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="GPU IDs (e.g., '0' or '0,1')",
    )
    parser.add_argument(
        "--skip_legacy",
        action="store_true",
        help="Skip legacy mode test (only test wave)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun (--no-resume)",
    )

    args = parser.parse_args()

    video_list = Path(args.video_list)
    if not video_list.exists():
        print(f"Error: Video list not found: {video_list}")
        sys.exit(1)

    # Count videos
    with open(video_list) as f:
        num_videos = sum(1 for line in f if line.strip())

    print(f"\n{'='*60}")
    print(f"Stage-Wave Scheduling Test")
    print(f"{'='*60}")
    print(f"Video list: {video_list}")
    print(f"Number of videos: {num_videos}")
    print(f"GPUs: {args.gpus}")
    print(f"Force rerun: {args.force}")
    print(f"{'='*60}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    # Test legacy mode
    if not args.skip_legacy:
        legacy_dir = PROJECT_ROOT / "batch_runs" / f"test_legacy_{timestamp}"
        legacy_success, legacy_time = run_batch_infer(
            video_list, args.gpus, "legacy", legacy_dir, force=args.force
        )

        legacy_metrics = parse_events(legacy_dir / "events.jsonl")
        legacy_outputs_ok = check_outputs(video_list, "LEGACY")

        results["legacy"] = {
            "success": legacy_success,
            "elapsed_sec": legacy_time,
            "metrics": legacy_metrics,
            "outputs_ok": legacy_outputs_ok,
        }

    # Test wave mode
    wave_dir = PROJECT_ROOT / "batch_runs" / f"test_wave_{timestamp}"
    wave_success, wave_time = run_batch_infer(
        video_list, args.gpus, "wave", wave_dir, force=args.force
    )

    wave_metrics = parse_events(wave_dir / "events.jsonl")
    wave_outputs_ok = check_outputs(video_list, "WAVE")

    results["wave"] = {
        "success": wave_success,
        "elapsed_sec": wave_time,
        "metrics": wave_metrics,
        "outputs_ok": wave_outputs_ok,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}\n")

    if not args.skip_legacy:
        print(f"LEGACY Mode:")
        print(f"  Success: {results['legacy']['success']}")
        print(f"  Elapsed: {results['legacy']['elapsed_sec']:.1f}s")
        print(f"  Outputs OK: {results['legacy']['outputs_ok']}")
        if results['legacy']['metrics']:
            m = results['legacy']['metrics']
            print(f"  Videos: {m['success']}/{m['total_videos']} succeeded")
        print()

    print(f"WAVE Mode:")
    print(f"  Success: {results['wave']['success']}")
    print(f"  Elapsed: {results['wave']['elapsed_sec']:.1f}s")
    print(f"  Outputs OK: {results['wave']['outputs_ok']}")
    if results['wave']['metrics']:
        m = results['wave']['metrics']
        print(f"  Videos: {m['success']}/{m['total_videos']} succeeded")
    print()

    if not args.skip_legacy and results['legacy']['success'] and results['wave']['success']:
        speedup = results['legacy']['elapsed_sec'] / results['wave']['elapsed_sec']
        print(f"Speedup: {speedup:.2f}x")
        print()

    # Test resume
    if results['wave']['success']:
        print(f"Testing WAVE resume (should skip all)...")
        wave_resume_dir = PROJECT_ROOT / "batch_runs" / f"test_wave_resume_{timestamp}"
        resume_success, resume_time = run_batch_infer(
            video_list, args.gpus, "wave", wave_dir, force=False
        )
        print(f"  Resume elapsed: {resume_time:.1f}s (should be fast)")
        print()

    # Final verdict
    print(f"{'='*60}")
    if results['wave']['success'] and results['wave']['outputs_ok']:
        print("✓ WAVE mode test PASSED")
        if not args.skip_legacy:
            if results['legacy']['success'] and results['legacy']['outputs_ok']:
                print("✓ LEGACY mode test PASSED")
            else:
                print("✗ LEGACY mode test FAILED")
    else:
        print("✗ WAVE mode test FAILED")
    print(f"{'='*60}\n")

    sys.exit(0 if results['wave']['success'] else 1)


if __name__ == "__main__":
    main()
