#!/bin/bash

# Single-node Accelerate FSDP2 launcher for the container image environment.
# Usage: bash scripts/pretrain_legendvla_fsdp2_single_node.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# ---------------- CONFIGURATION ----------------
ACC_CONFIG="src/config/acc_config.yaml"
SCRIPT="train.py"
ARGS="experiment=legendvla_qwen3_vl"
MASTER_ADDR="172.18.1.150"
MASTER_PORT="18276"
GPU_COUNT="8"
# -----------------------------------------------

if ! command -v accelerate >/dev/null 2>&1; then
    echo "accelerate is not installed in the current environment."
    exit 1
fi

cd "$PROJECT_DIR"

export NCCL_SOCKET_FAMILY="AF_INET"
export GLOO_SOCKET_IFNAME="eth1"
export TP_SOCKET_IFNAME="eth1"
export NCCL_SOCKET_IFNAME="eth1"
export NCCL_DEBUG="INFO"
export NCCL_TIMEOUT="3600000"
export NCCL_ASYNC_ERROR_HANDLING="1"
export NCCL_IB_DISABLE="0"
export NCCL_IB_HCA="=mlx5_1,=mlx5_2,=mlx5_3,=mlx5_4"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED=1

echo "Launching single-node FSDP2 training..."
echo "Project Directory: $PROJECT_DIR"
echo "GPU Count: $GPU_COUNT"
echo "Socket Interface: eth1"
echo "IB Devices: =mlx5_1,=mlx5_2,=mlx5_3,=mlx5_4"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

exec accelerate launch \
    --config_file "$ACC_CONFIG" \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --num_processes "$GPU_COUNT" \
    "$SCRIPT" \
    $ARGS
