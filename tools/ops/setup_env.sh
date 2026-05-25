#!/bin/bash
# One-command setup for the two conda environments used by the pipeline.
#
#   tools/ops/setup_env.sh hawor   # frame prep / detect / motion / infiller / build
#   tools/ops/setup_env.sh any4d   # SLAM + Any4D dense depth (any4d + dpvo)
#   tools/ops/setup_env.sh both
#
# The two envs are intentionally separate: HaWoR pins torch 2.5 (cu121) and
# Any4D pins torch 2.6 (cu124). They cannot share one env. This script is
# idempotent -- re-run it to resume a half-finished install.
#
# Override defaults via env vars, e.g.:
#   HAWOR_CUDA=cu118 ANY4D_CUDA=cu126 tools/ops/setup_env.sh both
#   HAWOR_ENV=hawor2 tools/ops/setup_env.sh hawor
#
# CUDA_HOME must point at a CUDA toolkit (DPVO compiles from source). Set it
# before running, e.g. export CUDA_HOME=/usr/local/cuda-12.4

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

HAWOR_ENV="${HAWOR_ENV:-hawor}"
ANY4D_ENV="${ANY4D_ENV:-any4d}"
HAWOR_CUDA="${HAWOR_CUDA:-cu121}"   # torch 2.5.1 index for HaWoR env
ANY4D_CUDA="${ANY4D_CUDA:-cu124}"   # torch 2.6 index for Any4D env (no cu121 wheels exist)

log()  { echo -e "\n=== $* ==="; }
have_conda() { command -v conda >/dev/null 2>&1; }

env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

create_env() {  # create_env <name> <python_version>
    local name="$1" py="$2"
    if env_exists "$name"; then
        log "conda env '$name' already exists -- skipping create"
    else
        log "Creating conda env '$name' (python=$py)"
        conda create -n "$name" "python=$py" -y
    fi
}

ensure_eigen() {  # DPVO's setup.py expects Eigen 3.4.0 here (not vendored)
    if [ -f thirdparty/DPVO/thirdparty/eigen-3.4.0/Eigen/Core ]; then
        log "Eigen 3.4.0 already present -- skipping fetch"
        return
    fi
    log "Fetching Eigen 3.4.0 for DPVO"
    ( cd thirdparty/DPVO && mkdir -p thirdparty \
      && wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip -O thirdparty/eigen-3.4.0.zip \
      && unzip -q -o thirdparty/eigen-3.4.0.zip -d thirdparty/ )
}

check_cuda_home() {
    if [ -z "${CUDA_HOME:-}" ]; then
        echo "WARNING: CUDA_HOME is not set. DPVO/DROID compile from source and need a CUDA toolkit."
        echo "         export CUDA_HOME=/usr/local/cuda-XX.X before running if the build fails."
    fi
}

install_hawor() {
    local R="conda run --no-capture-output -n $HAWOR_ENV"
    create_env "$HAWOR_ENV" 3.10
    log "Installing torch 2.5.1 ($HAWOR_CUDA) + xformers into '$HAWOR_ENV'"
    $R pip install torch==2.5.1 torchvision torchaudio --index-url "https://download.pytorch.org/whl/$HAWOR_CUDA"
    $R pip install -U xformers --index-url "https://download.pytorch.org/whl/$HAWOR_CUDA"
    log "Installing pipeline requirements"
    $R pip install -r requirements.txt
    $R pip install pytorch-lightning==2.2.4 --no-deps
    $R pip install lightning-utilities torchmetrics==1.4.0
    log "Installing build-isolated extras (mmcv / pytorch3d / chumpy)"
    $R pip install --no-build-isolation mmcv
    $R pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@stable
    $R pip install --no-build-isolation git+https://github.com/mattloper/chumpy
    ensure_eigen
    log "Building DPVO + DROID-SLAM into '$HAWOR_ENV'"
    $R bash -c "cd thirdparty/DPVO && pip install . --no-build-isolation"
    $R bash -c "cd thirdparty/DROID-SLAM && python setup.py install"
    log "HaWoR env ready. Viewers/demo (optional): conda run -n $HAWOR_ENV pip install --no-deps -r requirements-viewer.txt"
}

install_any4d() {
    local R="conda run --no-capture-output -n $ANY4D_ENV"
    create_env "$ANY4D_ENV" 3.12
    log "Installing torch 2.6 ($ANY4D_CUDA) into '$ANY4D_ENV'"
    $R pip install "torch==2.6.*" torchvision torchaudio --index-url "https://download.pytorch.org/whl/$ANY4D_CUDA"
    log "Installing Any4D (thirdparty/Any4D)"
    $R bash -c "cd thirdparty/Any4D && pip install -e ."
    # The SLAM subprocess imports HaWoR pipeline code (smplx, pytorch-lightning,
    # ultralytics, ...), so install the full pipeline requirements -- not just a
    # minimal subset, which leaves the run failing on missing imports. Skip
    # torch-scatter (no pinned torch-2.6 wheel) and opencv-python (Any4D already
    # pins the headless build); lightning installs with --no-deps so it cannot
    # pull a different torch.
    log "Installing HaWoR pipeline deps the SLAM subprocess imports"
    $R bash -c "grep -vE '^(torch-scatter|opencv-python)' requirements.txt | pip install -r /dev/stdin"
    $R pip install pytorch-lightning==2.2.4 --no-deps
    $R pip install lightning-utilities torchmetrics==1.4.0
    ensure_eigen
    log "Building DPVO into '$ANY4D_ENV' against torch 2.6"
    $R bash -c "cd thirdparty/DPVO && pip install . --no-build-isolation"
    log "Any4D env ready. Fetch the checkpoint if you have not:"
    echo "  mkdir -p checkpoints && wget -P checkpoints \\"
    echo "    https://huggingface.co/airlabshare/any4d-checkpoint/resolve/main/any4d_4v_combined.pth"
}

main() {
    local target="${1:-}"
    if ! have_conda; then
        echo "ERROR: conda is not on PATH." >&2
        exit 1
    fi
    check_cuda_home
    case "$target" in
        hawor) install_hawor ;;
        any4d) install_any4d ;;
        both)  install_hawor; install_any4d ;;
        *)
            echo "Usage: $0 {hawor|any4d|both}" >&2
            echo "  hawor  HaWoR env (torch 2.5.1 / $HAWOR_CUDA)"
            echo "  any4d  Any4D + DPVO env (torch 2.6 / $ANY4D_CUDA)"
            echo "  both   both envs"
            exit 2
            ;;
    esac
    log "Done. Verify with: bash tools/ops/validate_setup.sh"
}

main "$@"
