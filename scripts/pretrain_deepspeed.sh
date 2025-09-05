<<<<<<< HEAD
#!/usr/bin/env bash
set -euo pipefail

# DeepSpeed launcher for EgoVLA (single or multi-node)
# Usage (env-driven, all optional):
#   GPUS=8 HOSTFILE=/abs/path/hostfile OUTPUT_DIR=/abs/path/out EXPERIMENT=pretrain_deepspeed WANDB_MODE=online \
#   ./pretrain_deepspeed.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

TRAIN_SCRIPT="${PROJECT_ROOT}/egovla/workspace/train_egovla_deepspeed_workspace.py"
DS_CONFIG="/home/zengfanlian/EgoVLA/egovla/config/ds_config.json"

# Defaults (can be overridden by env)
GPUS=${GPUS:-8}
HOSTFILE=${HOSTFILE:-"/home/zengfanlian/EgoVLA/egovla/config/hostfile"}  # Default hostfile path
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/pretrain_deepspeed_$(date +%Y%m%d_%H%M%S)"}
EXPERIMENT=${EXPERIMENT:-}         # e.g. pretrain_deepspeed (Hydra config group under config/experiment)
WANDB_MODE=${WANDB_MODE:-online}   # online|offline|disabled

# NCCL/Perf settings
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}

# Optional: control W&B from env (script also supports Hydra logging config)
export WANDB_MODE="${WANDB_MODE}"

echo "Launching DeepSpeed with:"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Train script: ${TRAIN_SCRIPT}"
echo "  DS config:    ${DS_CONFIG} (loaded by training script)"
if [[ -f "${HOSTFILE}" ]]; then
  echo "  Hostfile:     ${HOSTFILE} (found)"
  echo "  GPUs:         (inferred from hostfile)"
else
  echo "  Hostfile:     ${HOSTFILE} (not found, single node mode)"
  echo "  GPUs:         ${GPUS}"
fi
echo "  Output dir:   ${OUTPUT_DIR}"
if [[ -n "${EXPERIMENT}" ]]; then
  echo "  Experiment:   ${EXPERIMENT} (Hydra config group)"
fi
echo "  W&B mode:     ${WANDB_MODE}"

mkdir -p "${OUTPUT_DIR}"

EXTRA_OPTS=(
  "hydra.run.dir=${OUTPUT_DIR}"
  "hydra.sweep.dir=${OUTPUT_DIR}"
  "hydra.job.name=pretrain_deepspeed"
)

if [[ -n "${EXPERIMENT}" ]]; then
  # Require a config group named `experiment` under egovla/config/experiment
  EXTRA_OPTS+=("experiment=${EXPERIMENT}")
fi

# Build deepspeed command
if [[ -f "${HOSTFILE}" ]]; then
  # When hostfile exists, DeepSpeed infers GPU allocation from hostfile
  DS_CMD=(deepspeed --hostfile "${HOSTFILE}" "${TRAIN_SCRIPT}" "${EXTRA_OPTS[@]}")
else
  # Single node mode (or no hostfile found)
  DS_CMD=(deepspeed --num_gpus "${GPUS}" "${TRAIN_SCRIPT}" "${EXTRA_OPTS[@]}")
fi

echo "Running: ${DS_CMD[*]}"
"${DS_CMD[@]}"

=======
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

# Launch training
accelerate launch \
    --config_file src/config/acc_config.yaml \
    train.py \
    experiment=pretrain_deepspeed 
    >>>>>>> 5c010b0f792275a214b29d8004106ea92fdfc589

