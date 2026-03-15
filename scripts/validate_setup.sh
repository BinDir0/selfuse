#!/bin/bash
# Quick validation script to check batch inference setup
# Run this first to verify environment before running full tests

set -e

echo "=== HaWoR Batch Inference Setup Validation ==="
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Check Python
echo "Checking Python environment..."
if ! command -v python &> /dev/null; then
    echo "❌ Python not found"
    exit 1
fi
echo "✓ Python: $(python --version)"

# Check required packages
echo ""
echo "Checking required packages..."
MISSING_PACKAGES=()

for pkg in torch numpy joblib; do
    if ! python -c "import $pkg" 2>/dev/null; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "❌ Missing packages: ${MISSING_PACKAGES[*]}"
    echo "Install with: pip install ${MISSING_PACKAGES[*]}"
    exit 1
fi
echo "✓ Required packages installed"

# Check CUDA
echo ""
echo "Checking CUDA availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.device_count()} GPUs')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('❌ CUDA not available')
    exit(1)
"

# Check model weights
echo ""
echo "Checking model weights..."
MISSING_WEIGHTS=()

if [ ! -f "./weights/hawor/checkpoints/hawor.ckpt" ]; then
    MISSING_WEIGHTS+=("hawor.ckpt")
fi

if [ ! -f "./weights/hawor/checkpoints/infiller.pt" ]; then
    MISSING_WEIGHTS+=("infiller.pt")
fi

if [ ! -f "./weights/external/droid.pth" ]; then
    MISSING_WEIGHTS+=("droid.pth")
fi

if [ ${#MISSING_WEIGHTS[@]} -gt 0 ]; then
    echo "❌ Missing weights: ${MISSING_WEIGHTS[*]}"
    echo "Download weights as described in README.md"
    exit 1
fi
echo "✓ Model weights found"

# Check batch scripts
echo ""
echo "Checking batch inference scripts..."
if [ ! -f "./scripts/batch_worker.py" ]; then
    echo "❌ batch_worker.py not found"
    exit 1
fi

if [ ! -f "./scripts/batch_infer.py" ]; then
    echo "❌ batch_infer.py not found"
    exit 1
fi
echo "✓ Batch scripts found"

# Test batch_worker.py help
echo ""
echo "Testing batch_worker.py..."
if ! python scripts/batch_worker.py --help &>/dev/null; then
    echo "❌ batch_worker.py failed"
    exit 1
fi
echo "✓ batch_worker.py OK"

# Test batch_infer.py help
echo ""
echo "Testing batch_infer.py..."
if ! python scripts/batch_infer.py --help &>/dev/null; then
    echo "❌ batch_infer.py failed"
    exit 1
fi
echo "✓ batch_infer.py OK"

# Check DROID-SLAM
echo ""
echo "Checking DROID-SLAM installation..."
if ! python -c "import droid_backends" 2>/dev/null; then
    echo "⚠️  Warning: DROID-SLAM not installed or not importable"
    echo "   Install with: cd thirdparty/DROID-SLAM && python setup.py install"
else
    echo "✓ DROID-SLAM installed"
fi

echo ""
echo "=== Setup Validation Complete ==="
echo ""
echo "Next steps:"
echo "1. Prepare test videos (2-3 short videos for smoke test)"
echo "2. Run smoke test: bash scripts/test_smoke.sh /path/to/test_videos 0,1"
echo "3. For production: bash scripts/test_production.sh videos.txt 0,1,2,3,4,5,6,7"
echo ""
echo "Quick start example:"
echo "  # Create video list"
echo "  find /path/to/videos -name '*.mp4' > videos.txt"
echo ""
echo "  # Run batch inference"
echo "  python scripts/batch_infer.py --video_list videos.txt --gpus 0,1,2,3,4,5,6,7"
