#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-/share_data/guantianrui/environment/anaconda3/envs/hawor/bin/python3.10}"

source_shard_dir="${SOURCE_SHARD_DIR:-/share_data/guantianrui/datasets/BuildAI-100K-part1-0401.with_instruction_v2}"
processed_root="${PROCESSED_ROOT:-/share_data/guantianrui/datasets/Egocentric-100K/processed_v9_test_jpg}"

output_root="${OUTPUT_ROOT:-/DATA/guantianrui/buildai_v2_rewrite_batches_full}"
report_root="${REPORT_ROOT:-/DATA/guantianrui/buildai_v2_rewrite_reports_full}"
mano_dir="${MANO_DIR:-/share_data/guantianrui/manopth/mano/models}"

shard_list_file="${SHARD_LIST_FILE:?SHARD_LIST_FILE is required}"
machine_tag="${MACHINE_TAG:-$(hostname -s)}"
slots_per_gpu="${SLOTS_PER_GPU:-1}"
infiller_window_batch_size="${INFILLER_WINDOW_BATCH_SIZE:-64}"
verify_sample_count="${VERIFY_SAMPLE_COUNT:-0}"
visualize_sample_count="${VISUALIZE_SAMPLE_COUNT:-0}"

base_gpus=(0 1 2 3 4 5 6 7)
worker_gpus=()
for gpu in "${base_gpus[@]}"; do
    for ((slot_copy=0; slot_copy<slots_per_gpu; slot_copy++)); do
        worker_gpus+=("$gpu")
    done
done

log_root="${report_root}/${machine_tag}_rebalance_logs"
mkdir -p "$output_root" "$report_root" "$log_root"

mapfile -t shard_names < <(grep -v '^[[:space:]]*$' "$shard_list_file" | grep -v '^[[:space:]]*#' || true)
if (( ${#shard_names[@]} == 0 )); then
    echo "${machine_tag}: no shards in ${shard_list_file}"
    exit 0
fi

cd "$repo_root"

echo "${machine_tag}: shard_count=${#shard_names[@]} slots=${#worker_gpus[@]} shard_list=${shard_list_file}"

run_worker() {
    local slot="$1"
    local gpu="$2"
    local log_file="${log_root}/${machine_tag}.gpu${gpu}.slot${slot}.log"

    echo "${machine_tag}: launch gpu=${gpu} slot=${slot} log=${log_file}"

    (
        set -euo pipefail
        local idx shard_name shard_idx
        for ((idx=slot; idx<${#shard_names[@]}; idx+=${#worker_gpus[@]})); do
            shard_name="${shard_names[$idx]}"
            shard_name="${shard_name%.tar}"
            shard_idx="${shard_name#shard-}"
            shard_idx=$((10#$shard_idx))

            echo "[$(date '+%F %T')] ${machine_tag} gpu=${gpu} shard=${shard_name}"
            sudo "$python_bin" -u scripts/rerun_and_rewrite_buildai_shards.py \
                --source_shard_dir "$source_shard_dir" \
                --buildai_processed_root "$processed_root" \
                --output_dir "$output_root" \
                --report_root "$report_root" \
                --shard_start "$shard_idx" \
                --shard_end "$((shard_idx + 1))" \
                --python_bin "$python_bin" \
                --gpu "$gpu" \
                --infiller_window_batch_size "$infiller_window_batch_size" \
                --rewrite_workers 1 \
                --mano_device "cuda:${gpu}" \
                --mano_gpus "$gpu" \
                --mano_dir "$mano_dir" \
                --verify_device "cuda:${gpu}" \
                --verify_sample_count "$verify_sample_count" \
                --visualize_sample_count "$visualize_sample_count"
        done
    ) 2>&1 | tee "$log_file"
}

pids=()
for slot in "${!worker_gpus[@]}"; do
    gpu="${worker_gpus[$slot]}"
    run_worker "$slot" "$gpu" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

if (( failed != 0 )); then
    echo "${machine_tag}: failed"
    exit 1
fi

echo "${machine_tag}: all workers done"
