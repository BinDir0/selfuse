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

nsys profile -t cuda,mpi,nvtx,cudnn -o rname.%p python train.py experiment=pretrain_deepspeed 

# accelerate launch --config_file egovla/config/acc_node0.yaml --no_python ./scripts/pretrain_deepspeed_nsys.sh
