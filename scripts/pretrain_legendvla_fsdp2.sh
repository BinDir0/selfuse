#!/bin/bash

# Master-only multi-node Accelerate FSDP launcher using pdsh.
# Usage: bash scripts/pretrain_legendvla_fsdp2.sh

set -euo pipefail

# ---------------- CONFIGURATION ----------------
NODES=(
    "172.18.1.150"
    "172.18.1.151"
)

SSH_USER=""
PROJECT_DIR="/home/zengfanlian/Projects/legendvla"
GPUS_PER_NODE=8
MASTER_PORT=18276

ACC_CONFIG="src/config/acc_config.yaml"
SCRIPT="train.py"
ARGS="experiment=legendvla_qwen3_vl"
# -----------------------------------------------

if ! command -v pdsh >/dev/null 2>&1; then
    echo "pdsh is not installed on this node."
    exit 1
fi

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

MASTER_ADDR=${NODES[0]}
NNODES=${#NODES[@]}
TOTAL_PROCESSES=$((GPUS_PER_NODE * NNODES))
HOSTLIST=$(IFS=,; echo "${NODES[*]}")

export PDSH_RCMD_TYPE=ssh

ssh_target_prefix() {
    if [ -n "$SSH_USER" ]; then
        printf "%s@" "$SSH_USER"
    fi
}

remote_cleanup() {
    pdsh -S -R exec -w "$HOSTLIST" \
        ssh -o BatchMode=yes "$(ssh_target_prefix)%h" \
        "pkill -f 'accelerate launch' || true; pkill -f 'train.py' || true" || true
}

cleanup() {
    local exit_code="${1:-130}"
    trap - INT TERM EXIT

    echo
    echo "Stopping remote training processes on all nodes..."
    remote_cleanup
    exit "$exit_code"
}

trap 'cleanup 130' INT TERM
trap 'cleanup $?' EXIT

echo "Launching training on $NNODES nodes from master..."
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Total Processes: $TOTAL_PROCESSES"
echo "Hostlist: $HOSTLIST"

echo "Cleaning up previous runs on all nodes..."
remote_cleanup
echo "Cleanup complete."

read -r -d '' REMOTE_SCRIPT <<EOF || true
cd "$PROJECT_DIR"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate legendvla
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_FORCE_CUDA_CODE_CACHE=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
exec accelerate launch \\
    --config_file "$ACC_CONFIG" \\
    --num_machines "$NNODES" \\
    --machine_rank __NODE_RANK__ \\
    --main_process_ip "$MASTER_ADDR" \\
    --main_process_port "$MASTER_PORT" \\
    --num_processes "$TOTAL_PROCESSES" \\
    "$SCRIPT" \\
    $ARGS
EOF

PDSH_COMMAND=${REMOTE_SCRIPT//__NODE_RANK__/%n}

echo "Starting pdsh launcher. Press Ctrl+C to stop all nodes."
pdsh -S -R exec -w "$HOSTLIST" \
    ssh -tt -o BatchMode=yes "$(ssh_target_prefix)%h" \
    "bash -lc '$PDSH_COMMAND'"

trap - EXIT
echo "Training finished."
