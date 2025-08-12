#!/bin/bash

# Multi-node DeepSpeed training script for EgoVLA
# Usage: ./run_multi_node_deepspeed.sh [config_name] [master_ip] [master_port] [num_nodes] [num_gpus_per_node] [node_rank]

set -e

# Default values
CONFIG_NAME=${1:-"train_egovla_deepspeed_workspace"}
MASTER_IP=${2:-"localhost"}
MASTER_PORT=${3:-11451}
NUM_NODES=${4:-1}
NUM_GPUS_PER_NODE=${5:-8}
NODE_RANK=${6:-0}

# DeepSpeed configuration
DEEPSPEED_CONFIG="/home/chenzhang/project/EgoVLA/egovla/config/ds_config.json"


# Set NCCL environment variables for better multi-node performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

echo "Starting DeepSpeed training with:"
echo "  Config: $CONFIG_NAME"
echo "  Master IP: $MASTER_IP"
echo "  Master Port: $MASTER_PORT"
echo "  Node Rank: $NODE_RANK"
echo "  Total Nodes: $NUM_NODES"
echo "  GPUs per node: $NUM_GPUS_PER_NODE"
echo "  Total GPUs: $((NUM_NODES * NUM_GPUS_PER_NODE))"
echo "  DeepSpeed config: $DEEPSPEED_CONFIG"

# Launch training
accelerate launch \
    --config_file /home/chenzhang/project/EgoVLA/egovla/config/acc_node0.yaml \
    train.py \
    experiment=pretrain_deepspeed