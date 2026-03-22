#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-/share_data/jixinhao/RoWaH/checkpoints}"
mkdir -p "${TARGET_DIR}"

CKPT_URL="https://huggingface.co/airlabshare/any4d-checkpoint/resolve/main/any4d_4v_combined.pth"
CKPT_PATH="${TARGET_DIR}/any4d_4v_combined.pth"

if [[ -f "${CKPT_PATH}" ]]; then
  echo "[Any4D] checkpoint already exists: ${CKPT_PATH}"
  exit 0
fi

echo "[Any4D] downloading checkpoint to: ${CKPT_PATH}"
if command -v wget >/dev/null 2>&1; then
  wget -O "${CKPT_PATH}" "${CKPT_URL}"
else
  echo "Error: wget not found. Please install wget or download manually from:"
  echo "${CKPT_URL}"
  exit 1
fi

echo "[Any4D] done."

