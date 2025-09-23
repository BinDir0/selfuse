#!/bin/bash

# Multi-node DeepSpeed training script for LegendVLA

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

mpirun -np 16 --hostfile /home/chenzhang/project/EgoVLA/src/config/hostfile /home/chenzhang/project/nccl-tests/build/all_reduce_perf_mpi -b 8 -e 1G -f 2 -g 1
