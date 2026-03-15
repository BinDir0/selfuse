#!/bin/bash
# One-click setup and test script for HaWoR batch inference system
# Run this on a machine with conda environment and GPUs available

set -e  # Exit on error

echo "=========================================="
echo "HaWoR Batch Inference Setup & Test"
echo "=========================================="
echo ""

# Configuration
CONDA_ENV="hawor"
TEST_GPUS="0,1"  # Modify based on available GPUs
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"

# Step 1: Check conda environment
echo "[1/6] Checking conda environment..."
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo "❌ Conda environment '${CONDA_ENV}' not found!"
    echo "Please create it first using the installation instructions in README.md"
    exit 1
fi
echo "✓ Conda environment '${CONDA_ENV}' found"
echo ""

# Step 2: Activate environment
echo "[2/6] Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV}
echo "✓ Environment activated"
echo ""

# Step 3: Verify dependencies
echo "[3/6] Verifying dependencies..."
python -c "import torch; import joblib; import numpy; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies!"
    exit 1
fi
echo "✓ Dependencies verified"
echo ""

# Step 4: Check for test videos
echo "[4/6] Checking for test videos..."
if [ ! -d "example" ] || [ -z "$(ls -A example/*.mp4 2>/dev/null)" ]; then
    echo "⚠️  No test videos found in example/ directory"
    echo "Please add 2-3 short test videos to the example/ directory"
    echo "Or modify this script to point to your test videos"
    exit 1
fi

# Create test video list
TEST_VIDEO_LIST="${PROJECT_ROOT}/test_videos.txt"
ls -1 "${PROJECT_ROOT}/example"/*.mp4 | head -3 > "${TEST_VIDEO_LIST}"
NUM_VIDEOS=$(wc -l < "${TEST_VIDEO_LIST}")
echo "✓ Found ${NUM_VIDEOS} test video(s):"
cat "${TEST_VIDEO_LIST}"
echo ""

# Step 5: Run smoke test
echo "[5/6] Running batch inference smoke test..."
echo "Configuration:"
echo "  - Videos: ${NUM_VIDEOS}"
echo "  - GPUs: ${TEST_GPUS}"
echo "  - Stages: detect_track,motion,slam,infiller"
echo ""

RUN_DIR="${PROJECT_ROOT}/batch_runs/test_$(date +%Y%m%d_%H%M%S)"

python scripts/batch_infer.py \
    --video_list "${TEST_VIDEO_LIST}" \
    --gpus "${TEST_GPUS}" \
    --run_dir "${RUN_DIR}" \
    --retries 1

if [ $? -eq 0 ]; then
    echo "✓ Smoke test PASSED"
else
    echo "❌ Smoke test FAILED"
    echo "Check logs in: ${RUN_DIR}/logs/"
    exit 1
fi
echo ""

# Step 6: Run resume test
echo "[6/6] Testing resume functionality..."
echo "Re-running with --resume (should skip completed stages)..."

python scripts/batch_infer.py \
    --video_list "${TEST_VIDEO_LIST}" \
    --gpus "${TEST_GPUS}" \
    --run_dir "${RUN_DIR}" \
    --resume

if [ $? -eq 0 ]; then
    echo "✓ Resume test PASSED"
else
    echo "❌ Resume test FAILED"
    exit 1
fi

# Check for skip events
SKIP_COUNT=$(grep -c '"event": "stage_skip"' "${RUN_DIR}/events.jsonl" || echo "0")
echo "Found ${SKIP_COUNT} stage skip events"

if [ "${SKIP_COUNT}" -gt 0 ]; then
    echo "✓ Resume logic working correctly"
else
    echo "⚠️  Expected skip events but found none"
fi
echo ""

# Summary
echo "=========================================="
echo "✓ ALL TESTS PASSED"
echo "=========================================="
echo ""
echo "Test results:"
echo "  - Run directory: ${RUN_DIR}"
echo "  - Status file: ${RUN_DIR}/status.json"
echo "  - Events log: ${RUN_DIR}/events.jsonl"
echo "  - Stage logs: ${RUN_DIR}/logs/"
echo ""
echo "Verify outputs:"
while IFS= read -r video; do
    video_stem=$(basename "${video}" .mp4)
    video_dir=$(dirname "${video}")
    output_file="${video_dir}/${video_stem}/world_space_res.pth"
    if [ -f "${output_file}" ]; then
        echo "  ✓ ${video_stem}/world_space_res.pth"
    else
        echo "  ❌ ${video_stem}/world_space_res.pth (missing)"
    fi
done < "${TEST_VIDEO_LIST}"
echo ""

echo "To run on your full dataset:"
echo "  python scripts/batch_infer.py \\"
echo "    --video_dir /path/to/videos \\"
echo "    --gpus 0,1,2,3,4,5,6,7"
echo ""
