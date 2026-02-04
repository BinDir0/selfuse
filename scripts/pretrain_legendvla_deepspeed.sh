#!/bin/bash

# # 清除 VS Code 注入的调试变量
# unset VSCODE_IPC_HOOK_CLI
# unset VSCODE_PID
# unset VSCODE_INJECTION
# unset DEBUGPY_PROCESS_SPAWN

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
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=600
export NCCL_PXN_DISABLE=0
export NCCL_MIN_CTAS=4

# Enable expandable segments for better memory utilization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Enable graph cache and CUDA code cache for torch.compile
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_FORCE_CUDA_CODE_CACHE=1

# # 设置 NCCL 超时为 1 小时 (单位毫秒: 3600000)
# export NCCL_TIMEOUT=3600000 
# export NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_COMPILE_DISABLE=1

rm -f .deepspeed_env

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Launch training
accelerate launch \
    --config_file src/config/acc_config.yaml \
    train.py \
    experiment=pretrain_legendvla_deepspeed 
