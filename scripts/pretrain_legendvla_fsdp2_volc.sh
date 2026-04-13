#!/bin/bash

# Volcengine MLP multi-node FSDP2 launcher using torchrun.
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
# PyTorch, and all training dependencies in the root environment.
# No Conda activation is required.
#
# Network assumptions validated on the training container:
#   Socket interface: eth1 (override via RDMA_IFNAME env var)
#   RDMA devices: mlx5_1, mlx5_2, mlx5_3, mlx5_4
#   GID index: left unset for NCCL auto selection

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# ---------------- CONFIGURATION ----------------
SCRIPT="train.py"
ARGS="experiment=legendvla_qwen3_vl"
# -----------------------------------------------

MASTER_ADDR="$MLP_WORKER_0_HOST"
MASTER_PORT="$MLP_WORKER_0_PORT"
MACHINE_RANK="$MLP_ROLE_INDEX"
NNODES="$MLP_WORKER_NUM"
GPUS_PER_NODE="$MLP_WORKER_GPU"

cd "$PROJECT_DIR"

RDMA_IFNAME="${RDMA_IFNAME:-eth1}"

export NCCL_SOCKET_FAMILY="AF_INET"
export GLOO_SOCKET_IFNAME="$RDMA_IFNAME"
export TP_SOCKET_IFNAME="$RDMA_IFNAME"
export NCCL_SOCKET_IFNAME="$RDMA_IFNAME"
export NCCL_DEBUG="INFO"
export NCCL_TIMEOUT="3600000"
export NCCL_ASYNC_ERROR_HANDLING="1"
export NCCL_IB_DISABLE="0"
export NCCL_IB_HCA="=mlx5_1,=mlx5_2,=mlx5_3,=mlx5_4"
export MALLOC_TRIM_THRESHOLD_="0"
export MALLOC_MMAP_THRESHOLD_="65536"
export MALLOC_ARENA_MAX="2"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS="1"
export PYTHONUNBUFFERED=1

echo "Launching Volcengine FSDP2 training..."
echo "Project Directory: $PROJECT_DIR"
echo "Socket Interface: $RDMA_IFNAME"
echo "IB Devices: =mlx5_1,=mlx5_2,=mlx5_3,=mlx5_4"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Machine Rank: $MACHINE_RANK"
echo "Num Machines: $NNODES"
echo "GPUs Per Node: $GPUS_PER_NODE"

exec torchrun \
    --nnodes="$NNODES" \
    --node_rank="$MACHINE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$GPUS_PER_NODE" \
    "$SCRIPT" \
    $ARGS
