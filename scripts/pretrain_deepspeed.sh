#!/bin/bash

# Multi-node DeepSpeed training script for EgoVLA

set -e

# Set NCCL environment variables for better multi-node performance
export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_P2P_DISABLE=0
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=23

rm -f .deepspeed_env

# Launch training
accelerate launch \
    --config_file src/config/acc_config.yaml \
    train.py \
    experiment=pretrain_deepspeed 
    