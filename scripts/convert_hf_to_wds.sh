#!/bin/bash
#
# Multi-node parallel HuggingFace(arrow/parquet) -> WebDataset conversion.
#
# Splits the hf_list evenly across nodes and launches conversion
# on each node via SSH. All nodes must share the same CFS mount
# for both input (hf dataset files) and output (wds shards).
# All datasets in hf_list are merged into one WebDataset shard set.
#
# Usage:
#   bash scripts/convert_hf_to_wds.sh \
#       --hf_list    data/hf_paths.txt \
#       --output_dir /cfs/data/vlm_wds \
#       --split      train \
#       --nodes      "pro-10 pro-01" \
#       --shard_prefix shard \
#       --workers    32
#
set -euo pipefail

# ─── defaults ───────────────────────────────────────────────────
HF_LIST="hf_list.txt"
OUTPUT_DIR="/share_data/guantianrui/datasets/VLM/Webdataset/ShareRobot/test"
SPLIT="test"
NODES="pro-04"
WORKERS_PER_NODE=96
MAXCOUNT=20000
MAXSIZE=1000000000
IMAGE_QUALITY=95
SHARD_PREFIX="shard"
PYTHON_PATH="/share_data/chenzhang/miniconda3/envs/legendvla/bin/python3.10"
PROJECT_DIR="/home/chenzhang/projects/diffloss-ar"
LOG_DIR="outputs/convert_hf_wds"

# ─── parse args ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf_list)      HF_LIST="$2"; shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --split)        SPLIT="$2"; shift 2 ;;
        --nodes)        NODES="$2"; shift 2 ;;
        --workers)      WORKERS_PER_NODE="$2"; shift 2 ;;
        --maxcount)     MAXCOUNT="$2"; shift 2 ;;
        --maxsize)      MAXSIZE="$2"; shift 2 ;;
        --image_quality) IMAGE_QUALITY="$2"; shift 2 ;;
        --shard_prefix) SHARD_PREFIX="$2"; shift 2 ;;
        --python)       PYTHON_PATH="$2"; shift 2 ;;
        --project_dir)  PROJECT_DIR="$2"; shift 2 ;;
        --log_dir)      LOG_DIR="$2"; shift 2 ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$HF_LIST" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: --hf_list and --output_dir are required."
    exit 1
fi

# Resolve hf_list to absolute path
HF_LIST="$(cd "$(dirname "$HF_LIST")" && pwd)/$(basename "$HF_LIST")"

# ─── read & filter hf list ──────────────────────────────────────
mapfile -t ALL_LINES < <(grep -v '^\s*#' "$HF_LIST" | grep -v '^\s*$')
TOTAL=${#ALL_LINES[@]}

if [[ $TOTAL -eq 0 ]]; then
    echo "Error: no valid entries in $HF_LIST"
    exit 1
fi

# ─── split across nodes ─────────────────────────────────────────
read -ra NODE_ARR <<< "$NODES"
NUM_NODES=${#NODE_ARR[@]}
PER_NODE=$(( (TOTAL + NUM_NODES - 1) / NUM_NODES ))

echo "============================================"
echo " HF -> WebDataset multi-node conversion"
echo "============================================"
echo " hf_list:      $HF_LIST ($TOTAL datasets)"
echo " output_dir:   $OUTPUT_DIR"
echo " split:        $SPLIT"
echo " nodes:        ${NODE_ARR[*]} ($NUM_NODES)"
echo " workers/node: $WORKERS_PER_NODE"
echo " maxcount:     $MAXCOUNT"
echo " maxsize:      $MAXSIZE"
echo " image_quality:$IMAGE_QUALITY"
echo " shard_prefix: $SHARD_PREFIX"
echo " python:       $PYTHON_PATH"
echo " project_dir:  $PROJECT_DIR"
echo "============================================"

mkdir -p "$LOG_DIR"

# Write per-node split files and launch via SSH
PIDS=()
NODE_USED=()
for (( i=0; i<NUM_NODES; i++ )); do
    NODE="${NODE_ARR[$i]}"
    START=$(( i * PER_NODE ))
    END=$(( START + PER_NODE ))
    if [[ $END -gt $TOTAL ]]; then
        END=$TOTAL
    fi
    if [[ $START -ge $TOTAL ]]; then
        echo "[node $NODE] no datasets to process, skipping."
        continue
    fi

    SPLIT_FILE="${LOG_DIR}/hf_split_${NODE}.txt"
    > "$SPLIT_FILE"
    for (( j=START; j<END; j++ )); do
        echo "${ALL_LINES[$j]}" >> "$SPLIT_FILE"
    done
    COUNT=$(( END - START ))

    echo "[node $NODE] assigned $COUNT datasets (index $START..$((END-1)))"

    LOG_FILE="${LOG_DIR}/convert_${NODE}.log"
    ssh -o StrictHostKeyChecking=no "$NODE" bash -lc "
        cd $PROJECT_DIR && \
        $PYTHON_PATH data/convert_hf_to_wds.py \
            --hf_list $SPLIT_FILE \
            --output_dir $OUTPUT_DIR \
            --split $SPLIT \
            --num_workers $WORKERS_PER_NODE \
            --maxcount $MAXCOUNT \
            --maxsize $MAXSIZE \
            --image_quality $IMAGE_QUALITY \
            --shard_prefix ${SHARD_PREFIX}-${NODE} \
    " > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    NODE_USED+=("$NODE")
    echo "[node $NODE] launched (pid $!, log: $LOG_FILE)"
done

# ─── wait for all nodes ─────────────────────────────────────────
echo ""
echo "Waiting for all nodes to finish..."
FAILED=0
for (( i=0; i<${#PIDS[@]}; i++ )); do
    NODE="${NODE_USED[$i]}"
    if wait "${PIDS[$i]}"; then
        echo "[node $NODE] done."
    else
        echo "[node $NODE] FAILED (exit code $?)."
        echo "  check log: ${LOG_DIR}/convert_${NODE}.log"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [[ $FAILED -eq 0 ]]; then
    echo "All nodes finished successfully."
else
    echo "$FAILED node(s) failed. Check logs in $LOG_DIR/"
    exit 1
fi
