#!/bin/bash

# Multi-node DeepSpeed training script for EgoVLA
# Usage: ./run_multi_node_deepspeed.sh [config_name] [master_ip] [master_port] [num_nodes] [num_gpus_per_node] [node_rank]

set -e

# Default values
CONFIG_NAME=${1:-"train_egovla_deepspeed_workspace"}
MASTER_IP=${2:-"localhost"}
MASTER_PORT=${3:-29500}
NUM_NODES=${4:-1}
NUM_GPUS_PER_NODE=${5:-8}
NODE_RANK=${6:-0}

# DeepSpeed configuration
DEEPSPEED_CONFIG="$(pwd)/../egovla/workspace/ds_config.json"

# Environment variables for DeepSpeed
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_PLUGIN_CONFIG_FILE=$DEEPSPEED_CONFIG
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust based on your GPU setup

# Set NCCL environment variables for better multi-node performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# Fix BF16 LayerNorm compatibility
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Starting DeepSpeed training with:"
echo "  Config: $CONFIG_NAME"
echo "  Master IP: $MASTER_IP"
echo "  Master Port: $MASTER_PORT"
echo "  Node Rank: $NODE_RANK"
echo "  Total Nodes: $NUM_NODES"
echo "  GPUs per node: $NUM_GPUS_PER_NODE"
echo "  Total GPUs: $((NUM_NODES * NUM_GPUS_PER_NODE))"
echo "  DeepSpeed config: $DEEPSPEED_CONFIG"

# Create output directory
OUTPUT_DIR="outputs/${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Launch training
accelerate launch \
    --config_file ../egovla/config/acc_node0.yaml \
    ../egovla/workspace/train_egovla_deepspeed_workspace.py \
    hydra.run.dir=$OUTPUT_DIR \
    hydra.sweep.dir=$OUTPUT_DIR \
    hydra.job.name=${CONFIG_NAME}