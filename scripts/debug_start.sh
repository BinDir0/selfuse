#!/bin/bash
# 调试脚本：尽可能还原正式训练环境

# ============ 1. 硬件与环境配置 ============
# 指定你调试想用的卡（例如 4,5,6,7）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# 计算显卡数量
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_ARRAY[@]}
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-18276}"
GPU_COUNT="${GPU_COUNT:-$NUM_GPUS}"
if (( GPU_COUNT > NUM_GPUS )); then
    echo "错误：GPU_COUNT=$GPU_COUNT 大于 CUDA_VISIBLE_DEVICES 中的 GPU 数量 NUM_GPUS=$NUM_GPUS" >&2
    echo "请减少 GPU_COUNT，或增加 CUDA_VISIBLE_DEVICES 中的可见 GPU。" >&2
    exit 1
fi

# 调试器配置
export DEBUGPY_PORT=5679
export ENABLE_DEBUGPY=1
export DEBUGPY_WAIT=1
export PYTHONUNBUFFERED=1
# HF Processor 的 __repr__ 很重；VS Code/PyDev 在停点展示变量时会自动 repr，
# 默认 0.5s 会频繁弹 "Computing repr ... was slow" warning。
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT="${PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT:-5}"
export EGOVLA_FAST_DEBUG_REPR="${EGOVLA_FAST_DEBUG_REPR:-1}"

# 禁用 WandB（调试时不需要上传日志）
export WANDB_MODE=disabled

# ============ 2. 还原正式训练的 NCCL 参数 ============
# 保留这些参数可以确保通信协议与正式训练一致
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
# NVIDIA InfiniBand / RDMA 专用参数。Alibaba PPU 用厂商自带 CCL、网卡命名不同，
# 这些 mlx5 / eth0 / GDR 设置会让通信库初始化挂起。默认在 NVIDIA 主机上启用；
# 在 PPU 上设 EGOVLA_DISABLE_NV_NCCL=1 跳过，让厂商 CCL 自己选传输层。
if [ "${EGOVLA_DISABLE_NV_NCCL:-0}" != "1" ]; then
    export NCCL_IB_DISABLE=0
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_IB_GID_INDEX=3
    # 注意：如果调试机器的网卡名字不同，可以注释掉下面几行
    export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3
    export NCCL_NET_GDR_LEVEL=2
fi

# 内存管理优化（正式脚本有的，调试也要带上）
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "正在使用 GPU: $CUDA_VISIBLE_DEVICES (共 $NUM_GPUS 张卡，torchrun 进程数 $GPU_COUNT)"

# ============ 3. 启动命令 ============
# 关键点：
# 1. 使用 torchrun 还原当前训练主链路
# 2. 使用 --nproc_per_node 匹配当前可见卡数
# 3. 使用 --nnodes 1 强制单机模式（调试通常是单机）
# 4. 不要传具体 GPU id，让 CUDA_VISIBLE_DEVICES 生效

# WARNING: do not insert inline comments between backslash-continued lines;
# bash treats '#' as a literal argument, breaking the command silently.
exec torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$GPU_COUNT" \
    train.py \
    experiment=legendvla_qwen3_vl \
    training.max_train_steps=10 \
    training.eval_every=5 \
    dataloader.loader.num_workers=0 \
    dataloader.loader.persistent_workers=False \
    dataloader.loader.prefetch_factor=null \
    2>&1 | tee debug_training.log
# NOTE: num_workers=0 forces data loading in the main process so debugpy
# breakpoints in __getitem__ / collator / transforms will be hit.
# persistent_workers and prefetch_factor must be disabled when num_workers=0,
# otherwise PyTorch raises ValueError at DataLoader init.
