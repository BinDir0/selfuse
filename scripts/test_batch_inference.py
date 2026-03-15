#!/usr/bin/env python3
"""
Validation script for batch inference system.

Tests:
1. Smoke test: 2-3 short videos on 2 GPUs
2. Resume test: verify skip behavior on re-run
3. Partial failure test: verify recovery from missing outputs
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    success = result.returncode == 0

    if success:
        print(f"✓ {description} - SUCCESS")
    else:
        print(f"✗ {description} - FAILED")

    return success


def check_output_exists(video_path, stage):
    """Check if stage output exists for a video."""
    video_path = Path(video_path)
    seq_folder = video_path.parent / video_path.stem

    if stage == "detect_track":
        return (seq_folder / "extracted_images").exists()
    elif stage == "motion":
        tracks_dirs = list(seq_folder.glob("tracks_*_*"))
        if not tracks_dirs:
            return False
        return any((d / "frame_chunks_all.npy").exists() for d in tracks_dirs)
    elif stage == "slam":
        slam_dir = seq_folder / "SLAM"
        return slam_dir.exists() and any(slam_dir.glob("hawor_slam_w_scale_*.npz"))
    elif stage == "infiller":
        return (seq_folder / "world_space_res.pth").exists()
    return False


def test_smoke(video_list, gpus, run_dir):
    """Test 1: Basic smoke test with full pipeline."""
    print("\n" + "="*60)
    print("TEST 1: SMOKE TEST - Full pipeline on sample videos")
    print("="*60)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "batch_infer.py"),
        "--video_list", video_list,
        "--gpus", gpus,
        "--run_dir", str(run_dir),
        "--retries", "1",
    ]

    success = run_command(cmd, "Full pipeline execution")

    # Verify outputs
    with open(video_list) as f:
        videos = [line.strip() for line in f if line.strip()]

    all_complete = True
    for video in videos:
        if not check_output_exists(video, "infiller"):
            print(f"✗ Missing final output for {video}")
            all_complete = False
        else:
            print(f"✓ Output verified for {Path(video).name}")

    return success and all_complete


def test_resume(video_list, gpus, run_dir):
    """Test 2: Resume behavior - should skip completed stages."""
    print("\n" + "="*60)
    print("TEST 2: RESUME TEST - Verify skip behavior")
    print("="*60)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "batch_infer.py"),
        "--video_list", video_list,
        "--gpus", gpus,
        "--run_dir", str(run_dir),
        "--resume",
    ]

    success = run_command(cmd, "Resume from existing outputs")

    # Check events log for skip events
    events_file = Path(run_dir) / "events.jsonl"
    if events_file.exists():
        skip_count = 0
        with open(events_file) as f:
            for line in f:
                event = json.loads(line)
                if event.get("event") == "stage_skip":
                    skip_count += 1

        print(f"\nFound {skip_count} stage skip events")
        if skip_count > 0:
            print("✓ Resume logic working correctly")
            return success
        else:
            print("✗ Expected skip events but found none")
            return False

    print("✗ Events file not found")
    return False


def test_partial_recovery(video_list, gpus, run_dir):
    """Test 3: Partial failure recovery - delete one stage output and verify recovery."""
    print("\n" + "="*60)
    print("TEST 3: PARTIAL RECOVERY - Delete stage output and recover")
    print("="*60)

    # Read first video
    with open(video_list) as f:
        videos = [line.strip() for line in f if line.strip()]

    if not videos:
        print("✗ No videos in list")
        return False

    test_video = videos[0]
    video_path = Path(test_video)
    seq_folder = video_path.parent / video_path.stem
    world_file = seq_folder / "world_space_res.pth"

    # Delete infiller output
    if world_file.exists():
        print(f"Deleting {world_file} to simulate partial failure")
        world_file.unlink()
    else:
        print(f"✗ Expected output file not found: {world_file}")
        return False

    # Re-run with resume
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "batch_infer.py"),
        "--video_list", video_list,
        "--gpus", gpus,
        "--run_dir", str(run_dir),
        "--resume",
    ]

    success = run_command(cmd, "Recovery from partial failure")

    # Verify output was recreated
    if world_file.exists():
        print(f"✓ Output successfully recovered: {world_file}")
        return success
    else:
        print(f"✗ Output not recovered: {world_file}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test batch inference system")
    parser.add_argument(
        "--video_list",
        required=True,
        help="Path to text file with 2-3 short test videos",
    )
    parser.add_argument(
        "--gpus",
        default="0,1",
        help="Comma-separated GPU IDs for testing (default: 0,1)",
    )
    parser.add_argument(
        "--run_dir",
        help="Custom run directory (default: temp directory)",
    )
    parser.add_argument(
        "--keep_outputs",
        action="store_true",
        help="Keep test outputs after completion",
    )

    args = parser.parse_args()

    # Setup run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = Path(tempfile.mkdtemp(prefix="batch_test_"))

    print(f"\nTest run directory: {run_dir}")

    # Run tests
    results = {}

    try:
        results["smoke"] = test_smoke(args.video_list, args.gpus, run_dir)
        results["resume"] = test_resume(args.video_list, args.gpus, run_dir)
        results["partial_recovery"] = test_partial_recovery(
            args.video_list, args.gpus, run_dir
        )

        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)

        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{test_name:20s}: {status}")

        all_passed = all(results.values())
        print("\n" + "="*60)
        if all_passed:
            print("ALL TESTS PASSED ✓")
        else:
            print("SOME TESTS FAILED ✗")
        print("="*60)

        return 0 if all_passed else 1

    finally:
        if not args.keep_outputs and not args.run_dir:
            print(f"\nCleaning up test directory: {run_dir}")
            shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
