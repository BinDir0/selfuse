#!/bin/bash
# Production-scale test for 8-GPU batch inference
# Tests with 20+ videos to verify full GPU utilization

set -e

echo "=== HaWoR 8-GPU Production Test ==="
echo ""

# Configuration
VIDEO_LIST="${1:-videos.txt}"
GPUS="${2:-0,1,2,3,4,5,6,7}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "$VIDEO_LIST" ]; then
    echo "Error: Video list file not found: $VIDEO_LIST"
    echo "Usage: $0 <video_list.txt> [gpus]"
    echo ""
    echo "Create a video list file with one video path per line:"
    echo "  /path/to/video1.mp4"
    echo "  /path/to/video2.mp4"
    echo "  ..."
    exit 1
fi

VIDEO_COUNT=$(wc -l < "$VIDEO_LIST")
echo "Video list: $VIDEO_LIST"
echo "Total videos: $VIDEO_COUNT"
echo "GPUs: $GPUS"
echo ""

if [ "$VIDEO_COUNT" -lt 20 ]; then
    echo "⚠️  Warning: Recommended 20+ videos for production test (found $VIDEO_COUNT)"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start batch inference
RUN_DIR="$PROJECT_ROOT/batch_runs/prod_test_$(date +%Y%m%d_%H%M%S)"
echo "Starting batch inference..."
echo "Run directory: $RUN_DIR"
echo ""

# Run in background and monitor
python "$PROJECT_ROOT/scripts/batch_infer.py" \
    --video_list "$VIDEO_LIST" \
    --gpus "$GPUS" \
    --run_dir "$RUN_DIR" \
    --retries 2 &

BATCH_PID=$!

# Monitor progress
echo "Monitoring progress (Ctrl+C to stop monitoring, batch will continue)..."
echo ""

monitor_progress() {
    while kill -0 $BATCH_PID 2>/dev/null; do
        if [ -f "$RUN_DIR/status.json" ]; then
            echo "=== Progress Update $(date +%H:%M:%S) ==="

            # Count completed/failed videos
            COMPLETED=$(grep -c '"completed"' "$RUN_DIR/status.json" || echo 0)
            FAILED=$(grep -c '"failed"' "$RUN_DIR/status.json" || echo 0)
            RUNNING=$(grep -c '"running"' "$RUN_DIR/status.json" || echo 0)

            echo "Completed: $COMPLETED / $VIDEO_COUNT"
            echo "Failed: $FAILED"
            echo "Running: $RUNNING"

            # Show recent events
            if [ -f "$RUN_DIR/events.jsonl" ]; then
                echo ""
                echo "Recent events:"
                tail -5 "$RUN_DIR/events.jsonl" | jq -r '"\(.time | split("T")[1] | split(".")[0]) [\(.event)] \(.video // "" | split("/")[-1]) \(.stage // "")"' 2>/dev/null || tail -5 "$RUN_DIR/events.jsonl"
            fi

            echo ""
        fi
        sleep 30
    done
}

trap 'echo "Monitoring stopped. Batch inference continues in background (PID: $BATCH_PID)"' INT

monitor_progress

# Wait for completion
wait $BATCH_PID
EXIT_CODE=$?

echo ""
echo "=== Batch Inference Complete ==="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All videos processed successfully"
else
    echo "⚠️  Some videos failed (exit code: $EXIT_CODE)"
fi

# Final statistics
if [ -f "$RUN_DIR/status.json" ]; then
    echo ""
    echo "=== Final Statistics ==="
    python3 -c "
import json
with open('$RUN_DIR/status.json') as f:
    data = json.load(f)

tasks = data.get('tasks', {})
total = len(tasks)
completed = sum(1 for t in tasks.values() if all(s == 'completed' for s in t['stage_status'].values()))
failed = sum(1 for t in tasks.values() if any(s == 'failed' for s in t['stage_status'].values()))

print(f'Total videos: {total}')
print(f'Completed: {completed}')
print(f'Failed: {failed}')
print(f'Success rate: {completed/total*100:.1f}%')

# Stage statistics
stages = ['detect_track', 'motion', 'slam', 'infiller']
print()
print('Per-stage completion:')
for stage in stages:
    stage_completed = sum(1 for t in tasks.values() if t['stage_status'].get(stage) == 'completed')
    print(f'  {stage}: {stage_completed}/{total}')
" 2>/dev/null || cat "$RUN_DIR/status.json"
fi

echo ""
echo "Run directory: $RUN_DIR"
echo "Logs: $RUN_DIR/logs/"
echo "Events: $RUN_DIR/events.jsonl"

exit $EXIT_CODE
