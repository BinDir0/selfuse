#!/usr/bin/env python3
"""
Test script to diagnose frame source issue
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.pipeline.frame_source import build_frame_source_auto

# Test with a problematic video
test_video = "/share_data/lvjianan/datasets/BuildAI-processed/factory_007/worker_016/processed/factory007_worker016_00005_crop002.mp4"

print(f"Testing with video: {test_video}")
print(f"Video file exists: {Path(test_video).exists()}")

# Check extracted_images
video_path_obj = Path(test_video)
video_dir = video_path_obj.parent
video_stem = video_path_obj.stem
extracted_dir = video_dir / video_stem / "extracted_images"

print(f"\nExpected extracted_dir: {extracted_dir}")
print(f"extracted_dir exists: {extracted_dir.exists()}")

if extracted_dir.exists():
    import glob
    from natsort import natsorted
    jpg_files = natsorted(glob.glob(str(extracted_dir / "*.jpg")))
    png_files = natsorted(glob.glob(str(extracted_dir / "*.png")))
    print(f"Found {len(jpg_files)} .jpg files")
    print(f"Found {len(png_files)} .png files")

    if jpg_files:
        print(f"First jpg: {jpg_files[0]}")
        print(f"Last jpg: {jpg_files[-1]}")

# Now test build_frame_source_auto
print("\n" + "="*60)
print("Calling build_frame_source_auto()...")
print("="*60)

os.environ["HAWOR_QUIET"] = "0"  # Force verbose output
frame_source, source_type = build_frame_source_auto(test_video, backend="decord")

print(f"\nResult:")
print(f"  source_type: {source_type}")
print(f"  frame_source type: {type(frame_source).__name__}")
print(f"  len(frame_source): {len(frame_source)}")
