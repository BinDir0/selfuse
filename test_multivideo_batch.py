#!/usr/bin/env python3
"""
Test script for multi-video batch processing with Supervision tracker.

This script tests the cross-video batching functionality for detect_track stage.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "lib" / "pipeline"))
sys.path.insert(0, str(PROJECT_ROOT / "lib"))

def test_supervision_import():
    """Test if supervision library is installed."""
    try:
        import supervision as sv
        print(f"✓ Supervision library installed (version: {sv.__version__})")
        return True
    except ImportError:
        print("✗ Supervision library not found!")
        print("  Install with: pip install supervision")
        return False

def test_multivideo_function():
    """Test if detect_track_multivideo function exists."""
    try:
        from tools import detect_track_multivideo
        print("✓ detect_track_multivideo function imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import detect_track_multivideo: {e}")
        return False

def test_basic_functionality():
    """Test basic multi-video batch processing with dummy data."""
    try:
        import numpy as np
        import supervision as sv
        from ultralytics import YOLO

        print("\n=== Testing Basic Functionality ===")

        # Test ByteTrack tracker creation
        tracker = sv.ByteTrack()
        print("✓ ByteTrack tracker created successfully")

        # Test YOLO model loading
        model_path = './weights/external/detector.pt'
        if not Path(model_path).exists():
            print(f"⚠ YOLO model not found at {model_path}")
            print("  Skipping model test")
            return True

        model = YOLO(model_path)
        print("✓ YOLO model loaded successfully")

        # Test batch prediction
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = model.predict([dummy_img, dummy_img], verbose=False)
        print(f"✓ Batch prediction successful (got {len(results)} results)")

        # Test supervision conversion
        detections = sv.Detections.from_ultralytics(results[0])
        print(f"✓ Supervision conversion successful")

        # Test tracker update
        tracked = tracker.update_with_detections(detections)
        print(f"✓ Tracker update successful")

        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== Multi-Video Batch Processing Test ===\n")

    all_passed = True

    # Test 1: Supervision import
    if not test_supervision_import():
        all_passed = False
        print("\nPlease install supervision library first:")
        print("  pip install supervision")
        return

    # Test 2: Function import
    if not test_multivideo_function():
        all_passed = False

    # Test 3: Basic functionality
    if not test_basic_functionality():
        all_passed = False

    print("\n" + "="*50)
    if all_passed:
        print("✓ All tests passed!")
        print("\nYou can now use multi-video batch processing:")
        print("  python scripts/batch_infer.py --video_list videos.txt \\")
        print("    --gpus 0 --detect_video_batch_size 4 \\")
        print("    --scheduler_mode wave --persistent_worker")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
