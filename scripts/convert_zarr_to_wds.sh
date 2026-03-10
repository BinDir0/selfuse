#!/bin/bash
#
# Multi-node parallel Zarr -> WebDataset conversion.
#
# Splits the zarr_list evenly across nodes and launches conversion
# on each node via SSH. All nodes must share the same CFS mount
# for both input (zarr) and output (wds shards).
#
# Usage:
#   bash scripts/convert_zarr_to_wds.sh \
#       --zarr_list  data/zarr_paths.txt \
#       --output_dir /cfs/data/wds \
#       --nodes      "pro-10 pro-01" \.0
#       --workers    120
#
set -euo pipefail

# ─── defaults ───────────────────────────────────────────────────
ZARR_LIST="zarr_list.txt"
OUTPUT_DIR="/share_data/guantianrui/datasets/Webdataset_val"
NODES="pro-10"
WORKERS_PER_NODE=160
PYTHON_PATH="/share_data/chenzhang/miniconda3/envs/legendvla/bin/python3.10"
PROJECT_DIR="/home/chenzhang/projects/diffloss-ar"
LOG_DIR="outputs/convert_wds"

# ─── parse args ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --zarr_list)   ZARR_LIST="$2";   shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --nodes)       NODES="$2";       shift 2 ;;
        --workers)     WORKERS_PER_NODE="$2"; shift 2 ;;
        --python)      PYTHON_PATH="$2"; shift 2 ;;
        --project_dir) PROJECT_DIR="$2"; shift 2 ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$ZARR_LIST" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: --zarr_list and --output_dir are required."
    exit 1
fi

# Resolve zarr_list to absolute path
ZARR_LIST="$(cd "$(dirname "$ZARR_LIST")" && pwd)/$(basename "$ZARR_LIST")"

# ─── read & filter zarr list ────────────────────────────────────
mapfile -t ALL_LINES < <(grep -v '^\s*#' "$ZARR_LIST" | grep -v '^\s*$')
TOTAL=${#ALL_LINES[@]}

if [[ $TOTAL -eq 0 ]]; then
    echo "Error: no valid entries in $ZARR_LIST"
    exit 1
fi

# ─── split across nodes ────────────────────────────────────────
read -ra NODE_ARR <<< "$NODES"
NUM_NODES=${#NODE_ARR[@]}
PER_NODE=$(( (TOTAL + NUM_NODES - 1) / NUM_NODES ))

echo "============================================"
echo " Zarr -> WebDataset multi-node conversion"
echo "============================================"
echo " zarr_list:   $ZARR_LIST ($TOTAL datasets)"
echo " output_dir:  $OUTPUT_DIR"
echo " nodes:       ${NODE_ARR[*]} ($NUM_NODES)"
echo " workers/node: $WORKERS_PER_NODE"
echo " python:      $PYTHON_PATH"
echo " project_dir: $PROJECT_DIR"
echo "============================================"

mkdir -p "$LOG_DIR"

# Write per-node split files and launch via SSH
PIDS=()
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

    # Write a temporary split file on CFS (accessible from all nodes)
    SPLIT_FILE="${LOG_DIR}/zarr_split_${NODE}.txt"
    > "$SPLIT_FILE"
    for (( j=START; j<END; j++ )); do
        echo "${ALL_LINES[$j]}" >> "$SPLIT_FILE"
    done
    COUNT=$(( END - START ))

    echo "[node $NODE] assigned $COUNT datasets (index $START..$((END-1)))"

    LOG_FILE="${LOG_DIR}/convert_${NODE}.log"

    # Launch on remote node via SSH (nohup so it survives SSH disconnect)
    ssh -o StrictHostKeyChecking=no "$NODE" bash -lc "
        cd $PROJECT_DIR && \
        $PYTHON_PATH data/convert_zarr_to_wds.py \
            --zarr_list $SPLIT_FILE \
            --output_dir $OUTPUT_DIR \
            --num_workers $WORKERS_PER_NODE \
    " > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    echo "[node $NODE] launched (pid $!, log: $LOG_FILE)"
done

# ─── wait for all nodes ─────────────────────────────────────────
echo ""
echo "Waiting for all nodes to finish..."
FAILED=0
for (( i=0; i<${#PIDS[@]}; i++ )); do
    NODE="${NODE_ARR[$i]}"
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
