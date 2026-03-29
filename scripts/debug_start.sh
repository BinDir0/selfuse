#!/bin/bash
# 调试脚本：尽可能还原正式训练环境

# ============ 1. 硬件与环境配置 ============
PYTHON_PATH="/share_data/chenzhang/miniconda3/envs/legendvla/bin/python3.10"
# 指定你调试想用的卡（例如 4,5,6,7）
export CUDA_VISIBLE_DEVICES="1,2,3"

# 计算显卡数量
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_ARRAY[@]}

# 调试器配置
export DEBUGPY_PORT=5679
export ENABLE_DEBUGPY=1
export DEBUGPY_WAIT=1

# 禁用 WandB（调试时不需要上传日志）
export WANDB_MODE=disabled

# ============ 2. 还原正式训练的 NCCL 参数 ============
# 保留这些参数可以确保通信协议与正式训练一致
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600000
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
# 注意：如果调试机器的网卡名字不同，可以注释掉下面几行
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3
export NCCL_NET_GDR_LEVEL=2

# 内存管理优化（正式脚本有的，调试也要带上）
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "正在使用 GPU: $CUDA_VISIBLE_DEVICES (共 $NUM_GPUS 张卡)"

# ============ 3. 启动命令 ============
# 关键点：
# 1. 使用 --config_file 还原正式逻辑
# 2. 使用 --num_processes 覆盖 YAML 里的卡数，强制匹配当前可见卡数
# 3. 使用 --num_machines 1 强制单机模式（调试通常是单机）
# 4. 不要传 --gpu_ids，让环境变量生效

# WARNING: do not insert inline comments between backslash-continued lines;
# bash treats '#' as a literal argument, breaking the command silently.
$PYTHON_PATH -m accelerate.commands.launch \
    --config_file src/config/acc_config.yaml \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_port 29501 \
    train.py \
    experiment=legendvla_qwen3_vl \
    training.max_train_steps=10 \
    training.eval_every=5 \
    2>&1 | tee debug_training.log