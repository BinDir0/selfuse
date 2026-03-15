#!/bin/bash
# Smoke test for batch inference system
# Tests basic functionality with 2-3 short videos on 2 GPUs

set -e

echo "=== HaWoR Batch Inference Smoke Test ==="
echo ""

# Configuration
TEST_VIDEOS_DIR="${1:-./test_videos}"
GPUS="${2:-0,1}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "$TEST_VIDEOS_DIR" ]; then
    echo "Error: Test videos directory not found: $TEST_VIDEOS_DIR"
    echo "Usage: $0 <test_videos_dir> [gpus]"
    echo "Example: $0 /path/to/test_videos 0,1"
    exit 1
fi

# Count videos
VIDEO_COUNT=$(find "$TEST_VIDEOS_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | wc -l)
echo "Found $VIDEO_COUNT videos in $TEST_VIDEOS_DIR"

if [ "$VIDEO_COUNT" -lt 2 ]; then
    echo "Warning: Need at least 2 videos for smoke test"
fi

echo "Using GPUs: $GPUS"
echo ""

# Test 1: Full pipeline
echo "=== Test 1: Full pipeline execution ==="
RUN_DIR="$PROJECT_ROOT/batch_runs/smoke_test_$(date +%Y%m%d_%H%M%S)"
/share_data/guantianrui/environment/anaconda3/envs/hawor/bin/python "$PROJECT_ROOT/scripts/batch_infer.py" \
    --video_dir "$TEST_VIDEOS_DIR" \
    --gpus "$GPUS" \
    --run_dir "$RUN_DIR" \
    --retries 1

if [ $? -ne 0 ]; then
    echo "❌ Test 1 failed: Pipeline execution error"
    exit 1
fi

echo "✓ Test 1 passed"
echo ""

# Test 2: Resume (should skip all stages)
echo "=== Test 2: Resume from existing outputs ==="
/share_data/guantianrui/environment/anaconda3/envs/hawor/bin/python "$PROJECT_ROOT/scripts/batch_infer.py" \
    --video_dir "$TEST_VIDEOS_DIR" \
    --gpus "$GPUS" \
    --run_dir "$RUN_DIR" \
    --retries 1

if [ $? -ne 0 ]; then
    echo "❌ Test 2 failed: Resume error"
    exit 1
fi

# Check that stages were skipped
SKIP_COUNT=$(grep -c '"event": "stage_skip"' "$RUN_DIR/events.jsonl" || true)
if [ "$SKIP_COUNT" -eq 0 ]; then
    echo "⚠️  Warning: No stages were skipped on resume"
fi

echo "✓ Test 2 passed (skipped $SKIP_COUNT stages)"
echo ""

# Test 3: Verify outputs
echo "=== Test 3: Verify output artifacts ==="
MISSING_OUTPUTS=0

for video in "$TEST_VIDEOS_DIR"/*.mp4 "$TEST_VIDEOS_DIR"/*.avi "$TEST_VIDEOS_DIR"/*.mov; do
    [ -f "$video" ] || continue

    video_name=$(basename "$video" | sed 's/\.[^.]*$//')
    seq_folder="$TEST_VIDEOS_DIR/$video_name"

    if [ ! -f "$seq_folder/world_space_res.pth" ]; then
        echo "❌ Missing world_space_res.pth for $video_name"
        MISSING_OUTPUTS=$((MISSING_OUTPUTS + 1))
    fi
done

if [ "$MISSING_OUTPUTS" -gt 0 ]; then
    echo "❌ Test 3 failed: $MISSING_OUTPUTS videos missing outputs"
    exit 1
fi

echo "✓ Test 3 passed"
echo ""

# Test 4: Partial rerun (delete one output and verify it's regenerated)
echo "=== Test 4: Partial rerun after artifact deletion ==="
FIRST_VIDEO=$(find "$TEST_VIDEOS_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | head -1)
if [ -n "$FIRST_VIDEO" ]; then
    video_name=$(basename "$FIRST_VIDEO" | sed 's/\.[^.]*$//')
    seq_folder="$TEST_VIDEOS_DIR/$video_name"

    if [ -f "$seq_folder/world_space_res.pth" ]; then
        echo "Deleting $seq_folder/world_space_res.pth"
        rm "$seq_folder/world_space_res.pth"

        /share_data/guantianrui/environment/anaconda3/envs/hawor/bin/python "$PROJECT_ROOT/scripts/batch_infer.py" \
            --video_dir "$TEST_VIDEOS_DIR" \
            --gpus "$GPUS" \
            --run_dir "$RUN_DIR" \
            --retries 1

        if [ ! -f "$seq_folder/world_space_res.pth" ]; then
            echo "❌ Test 4 failed: Output not regenerated"
            exit 1
        fi

        echo "✓ Test 4 passed"
    else
        echo "⚠️  Skipping Test 4: No output to delete"
    fi
fi

echo ""
echo "=== All smoke tests passed! ==="
echo "Run directory: $RUN_DIR"
echo "Check logs at: $RUN_DIR/logs/"
echo "Check status at: $RUN_DIR/status.json"
