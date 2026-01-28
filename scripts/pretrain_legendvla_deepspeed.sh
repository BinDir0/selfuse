#!/bin/bash

# Multi-node DeepSpeed training script for LegendVLA

set -e

# Set NCCL environment variables for better multi-node performance
export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_P2P_DISABLE=0
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_TIMEOUT=3600000 
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=600
export NCCL_PXN_DISABLE=0
export NCCL_MIN_CTAS=4

# Enable expandable segments for better memory utilization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

rm -f .deepspeed_env

# Launch training
accelerate launch \
    --config_file src/config/acc_config.yaml \
    train.py \
    experiment=pretrain_legendvla_deepspeed \
    2>&1 | tee training.log
