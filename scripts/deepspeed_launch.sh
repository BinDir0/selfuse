#!/bin/bash

# Multi-node DeepSpeed training script for LegendVLA - Integrated Version
set -e

# 1. 强制使用 SSH 模式以兼容别名
export PDSH_RCMD_TYPE=ssh
PYTHON_PATH="/share_data/chenzhang/miniconda3/envs/legendvla/bin/python3.10"
CONDA_BIN="/home/chenzhang/miniconda3/envs/legendvla/bin"

# 2. 准备 hostfile
cat > hostfile <<EOF
pro-10 slots=8
pro-01 slots=8
EOF

# 3. 设置环境变量（在本地和远程节点都生效）
# 先 export 到当前 shell（本地节点）- 这些对 DeepSpeed 识别很关键！
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_CONFIG_FILE=src/config/ds_config.json
export ACCELERATE_MIXED_PRECISION=bf16
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600000
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=600
export NCCL_PXN_DISABLE=0
export NCCL_MIN_CTAS=4
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_FORCE_CUDA_CODE_CACHE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PATH=$CONDA_BIN:/usr/local/bin:/usr/bin:/bin:$PATH

# 准备 .deepspeed_env (远程节点会读取这个文件)
cat > .deepspeed_env <<EOF
PATH=$CONDA_BIN:/usr/local/bin:/usr/bin:/bin:\$PATH
ACCELERATE_USE_DEEPSPEED=true
ACCELERATE_DEEPSPEED_CONFIG_FILE=src/config/ds_config.json
ACCELERATE_MIXED_PRECISION=bf16
NCCL_DEBUG=INFO
NCCL_TIMEOUT=3600000
NCCL_ASYNC_ERROR_HANDLING=1
TORCH_NCCL_ASYNC_ERROR_HANDLING=1
NCCL_SOCKET_IFNAME=eth0
NCCL_IB_GID_INDEX=3
NCCL_IB_DISABLE=0
NCCL_NET_GDR_LEVEL=2
NCCL_IB_QPS_PER_CONNECTION=4
NCCL_IB_TC=160
NCCL_IB_TIMEOUT=600
NCCL_PXN_DISABLE=0
NCCL_MIN_CTAS=4
NCCL_SOCKET_FAMILY=AF_INET
GLOO_SOCKET_FAMILY=AF_INET
GLOO_SOCKET_IFNAME=eth0
TP_SOCKET_IFNAME=eth0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TORCHINDUCTOR_FX_GRAPH_CACHE=1
TORCHINDUCTOR_FORCE_CUDA_CODE_CACHE=1
TOKENIZERS_PARALLELISM=false
PYTHONUNBUFFERED=1
EOF

# 4. 设定分布式主节点信息 (master node 的 eth0 IP)
MASTER_ADDR="172.18.0.110"
MASTER_PORT=29501

# 5. 启动命令
echo "正在启动分布式训练..."
echo "Master Node IP: $MASTER_ADDR"

deepspeed --hostfile hostfile \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --no_local_rank \
    --include "pro-10:4,5,6,7@pro-01:0,1,2,3,4,5,6,7" \
    train.py \
    experiment=legendvla_qwen3_vl \
    2>&1 | tee multi_node_train.log