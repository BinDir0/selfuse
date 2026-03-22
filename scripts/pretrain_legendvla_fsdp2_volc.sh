#!/bin/bash

# Volcengine MLP multi-node Accelerate FSDP2 launcher.
# Usage: bash scripts/pretrain_legendvla_fsdp2_volc.sh
#
# Volcengine console example:
#   Command: bash
#   Arguments: scripts/pretrain_legendvla_fsdp2_volc.sh
#
# Required platform environment variables injected by Volcengine MLP:
#   MLP_WORKER_0_HOST: rank-0 worker address
#   MLP_WORKER_0_PORT: rank-0 worker port
#   MLP_ROLE_INDEX: current worker rank
#   MLP_WORKER_NUM: total number of workers
#   MLP_WORKER_GPU: number of GPUs per worker
#
# This script assumes the container image already contains Python,
# Accelerate, PyTorch, and all training dependencies in the root environment.
# No Conda activation is required.
#
# Network assumptions validated on the training container:
#   Socket interface: eth1
#   RDMA devices: mlx5_1, mlx5_2, mlx5_3, mlx5_4
#   GID index: left unset for NCCL auto selection

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# ---------------- CONFIGURATION ----------------
ACC_CONFIG="src/config/acc_config.yaml"
SCRIPT="train.py"
ARGS="experiment=legendvla_qwen3_vl"
# -----------------------------------------------

MASTER_ADDR="$MLP_WORKER_0_HOST"
MASTER_PORT="$MLP_WORKER_0_PORT"
MACHINE_RANK="$MLP_ROLE_INDEX"
NNODES="$MLP_WORKER_NUM"
GPUS_PER_NODE="$MLP_WORKER_GPU"

if ! command -v accelerate >/dev/null 2>&1; then
    echo "accelerate is not installed in the current environment."
    exit 1
fi

TOTAL_PROCESSES=$((NNODES * GPUS_PER_NODE))

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

echo "Launching Volcengine FSDP2 training..."
echo "Project Directory: $PROJECT_DIR"
echo "Socket Interface: eth1"
echo "IB Devices: =mlx5_1,=mlx5_2,=mlx5_3,=mlx5_4"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Machine Rank: $MACHINE_RANK"
echo "Num Machines: $NNODES"
echo "GPUs Per Node: $GPUS_PER_NODE"
echo "Total Processes: $TOTAL_PROCESSES"

exec accelerate launch \
    --config_file "$ACC_CONFIG" \
    --num_machines "$NNODES" \
    --machine_rank "$MACHINE_RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --num_processes "$TOTAL_PROCESSES" \
    "$SCRIPT" \
    $ARGS
