#!/bin/bash

# Single-node FSDP2 launcher using torchrun.
# Usage: bash scripts/pretrain_legendvla_fsdp2_single_node.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# ---------------- CONFIGURATION ----------------
SCRIPT="train.py"
ARGS="experiment=legendvla_qwen3_vl"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-18276}"
GPU_COUNT="8"
# -----------------------------------------------

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

echo "Launching single-node FSDP2 training..."
echo "Project Directory: $PROJECT_DIR"
echo "GPU Count: $GPU_COUNT"
echo "Socket Interface: $RDMA_IFNAME"
echo "IB Devices: =mlx5_1,=mlx5_2,=mlx5_3,=mlx5_4"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

exec torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$GPU_COUNT" \
    "$SCRIPT" \
    $ARGS
