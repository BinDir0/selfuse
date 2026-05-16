#!/bin/bash
# Validate the two-env runtime contract used by the official dataset pipeline.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

HAWOR_ENV="${HAWOR_ENV:-hawor}"
ANY4D_ENV="${ANY4D_ENV:-any4d}"

echo "=== RoWaH / HaWoR Setup Validation ==="
echo "Project root : $PROJECT_ROOT"
echo "HaWoR env    : $HAWOR_ENV"
echo "Any4D env    : $ANY4D_ENV"
echo ""

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not on PATH. The pipeline runtime resolver expects conda, mamba, or micromamba env discovery."
    exit 1
fi

echo "Checking conda environments..."
if ! conda env list | awk '{print $1}' | grep -qx "$HAWOR_ENV"; then
    echo "ERROR: conda env '$HAWOR_ENV' was not found."
    echo "Create it with the README Environment Setup instructions."
    exit 1
fi
if ! conda env list | awk '{print $1}' | grep -qx "$ANY4D_ENV"; then
    echo "ERROR: conda env '$ANY4D_ENV' was not found."
    echo "Create it with the README Environment Setup instructions."
    exit 1
fi
echo "OK: required conda env names exist"
echo ""

echo "Checking HaWoR Python packages..."
conda run -n "$HAWOR_ENV" python - <<'PY'
import sys

missing = []
for module in (
    "torch",
    "numpy",
    "cv2",
    "joblib",
    "webdataset",
    "natsort",
    "ultralytics",
):
    try:
        __import__(module)
    except Exception as exc:
        missing.append(f"{module}: {exc}")

if missing:
    print("ERROR: missing HaWoR packages:")
    for item in missing:
        print("  -", item)
    raise SystemExit(1)

import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
PY
echo ""

echo "Checking Any4D Python packages and configured paths..."
conda run -n "$ANY4D_ENV" python - <<'PY'
import sys
from pathlib import Path

missing = []
for module in (
    "torch",
    "cv2",
    "hydra",
    "natsort",
    "PIL",
    "tqdm",
    "joblib",
    "pycocotools",
    "evo",
    "torchmin",
):
    try:
        __import__(module)
    except Exception as exc:
        missing.append(f"{module}: {exc}")

try:
    import any4d  # noqa: F401
except Exception as exc:
    missing.append(f"any4d: {exc}")

if missing:
    print("ERROR: missing Any4D packages:")
    for item in missing:
        print("  -", item)
    raise SystemExit(1)

import torch
from lib.pipeline.any4d_depth import resolve_any4d_paths

repo_root, checkpoint_path, resolution, use_amp = resolve_any4d_paths()
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
print("any4d_repo_root:", repo_root)
print("any4d_checkpoint:", checkpoint_path)
print("any4d_resolution:", resolution)
print("any4d_use_amp:", use_amp)

if not Path(repo_root).is_dir():
    raise SystemExit(f"ERROR: Any4D repo root not found: {repo_root}")
if not Path(checkpoint_path).is_file():
    raise SystemExit(f"ERROR: Any4D checkpoint not found: {checkpoint_path}")
PY
echo ""

echo "Checking SLAM CLI imports in Any4D env..."
conda run -n "$ANY4D_ENV" python scripts/batch_infer.py --help >/dev/null
conda run -n "$ANY4D_ENV" python - <<'PY'
import lib.pipeline.stages.slam  # noqa: F401
from dpvo.config import cfg  # noqa: F401
print("OK: slam stage imports")
PY
echo ""

echo "Checking model weights..."
missing_weights=()
for weight in \
    "./weights/hawor/checkpoints/hawor.ckpt" \
    "./weights/hawor/checkpoints/infiller.pt" \
    "./weights/hawor/model_config.yaml" \
    "./weights/external/detector.pt"; do
    if [ ! -f "$weight" ]; then
        missing_weights+=("$weight")
    fi
done

if [ "${#missing_weights[@]}" -gt 0 ]; then
    echo "ERROR: missing required HaWoR/WiLoR weights:"
    printf '  - %s\n' "${missing_weights[@]}"
    echo "Download them as described in README.md."
    exit 1
fi
echo "OK: required HaWoR/WiLoR weights found"
echo ""

echo "Checking pipeline CLI imports in HaWoR env..."
conda run -n "$HAWOR_ENV" python scripts/run_dataset_pipeline.py --help >/dev/null
conda run -n "$HAWOR_ENV" python scripts/batch_infer.py --help >/dev/null
echo "OK: pipeline CLIs import"
echo ""

echo "Checking optional DROID-SLAM backend in HaWoR env..."
if conda run -n "$HAWOR_ENV" python -c "import droid_backends" >/dev/null 2>&1; then
    echo "OK: DROID-SLAM backend importable"
else
    echo "WARN: DROID-SLAM backend is not importable. Install it if you use --slam_backend droid."
    echo "      cd thirdparty/DROID-SLAM && python setup.py install"
fi
echo ""

echo "=== Setup Validation Complete ==="
echo "Next smoke test:"
echo "  conda run -n $HAWOR_ENV python scripts/run_dataset_pipeline.py --config configs/my_video.yaml --stages prepare"
echo "  conda run -n $HAWOR_ENV python scripts/run_dataset_pipeline.py --config configs/my_video.yaml"
