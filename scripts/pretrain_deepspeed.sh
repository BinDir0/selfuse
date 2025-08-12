#!/bin/bash

# Multi-node DeepSpeed training script for EgoVLA

set -e

# Set NCCL environment variables for better multi-node performance
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# Launch training
accelerate launch \
    --config_file egovla/config/acc_node0.yaml \
    train_deepspeed.py \
    experiment=pretrain_deepspeed \
    > progress.log