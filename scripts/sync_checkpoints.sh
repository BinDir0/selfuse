#!/bin/bash
set -euo pipefail

# Sync FSDP2 sharded checkpoints from remote training server to local machine.
#
# Features:
#   - Consolidate DCP shards into a single .pt file on the remote (via SSH)
#   - Parallel rsync to saturate bandwidth
#   - Filter by time (--since) and auto-discover latest run (--latest-run)
#   - Watch mode for continuous polling
#
# Dependencies:
#   Local  : rsync, ssh (apt-get install rsync openssh-client / yum install rsync openssh-clients)
#   Remote : PyTorch >= 2.3 (torch.distributed.checkpoint.format_utils)
#            Already satisfied if the training env uses FSDP2 (requires PyTorch >= 2.4).
#            Specify the conda env via -c / --conda (e.g. -c legendvla).
#
# Typical usage:
#   bash scripts/sync_checkpoints.sh -H node0 -r /path/to/outputs/2026.04.02/10.30_run --conda legendvla
#   bash scripts/sync_checkpoints.sh -H node0 --latest-run --watch 300 --conda legendvla

# ========================= Colors & Logging =========================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${BLUE}[STEP]${NC} $1"; }

# ========================= Dependency Check =========================

for cmd in rsync ssh; do
    if ! command -v "$cmd" &>/dev/null; then
        log_error "'${cmd}' is not installed. Install it first:"
        log_error "  Ubuntu/Debian:  sudo apt-get install ${cmd}"
        log_error "  CentOS/RHEL:    sudo yum install ${cmd}"
        log_error "  Arch:           sudo pacman -S ${cmd}"
        exit 1
    fi
done

# ========================= Defaults =========================

REMOTE_HOST=""
REMOTE_DIR=""
LOCAL_DIR="./outputs_synced"
CONDA_ENV=""
DRY_RUN=false
STEP_CKPT_COUNT=0
SINCE=""
NO_CONSOLIDATE=false
DELETE=false
DELETE_BEFORE=""
WATCH_INTERVAL=""
LATEST_RUN=false
MAX_PARALLEL=4

# Remote output base for --latest-run discovery
REMOTE_OUTPUT_BASE=""

# ========================= Usage =========================

usage() {
    cat <<'USAGE'
Usage: sync_checkpoints.sh [OPTIONS]

Options:
  -H, --host HOST             Remote SSH host (required)
  -r, --remote-dir DIR        Remote run output directory (full path)
  -l, --local-dir DIR         Local destination (default: ./outputs_synced/)
  -c, --conda ENV             Remote conda env name for consolidation (e.g. legendvla)
  -n, --dry-run               Preview without transferring
  -s, --step-checkpoints N    Sync the latest N step checkpoints (default: 0)
  --since DATETIME            Only sync checkpoints newer than this time
                              (format: "2026-04-01" or "2026-04-01 14:00")
  --no-consolidate            Skip remote consolidation, pull raw DCP shards
  --delete                    Remove local checkpoints deleted on remote (top-k rotation)
  --delete-before DATETIME    Delete local checkpoints older than this time
  --watch INTERVAL            Re-sync every INTERVAL seconds (run in tmux)
  --latest-run                Auto-detect the most recent run on remote
  --max-parallel N            Max parallel rsync processes (default: 4)

Examples:
  # One-shot sync of a specific run
  sync_checkpoints.sh -H node0 -r /path/to/outputs/2026.04.02/10.30_run -c legendvla

  # Auto-detect latest run, watch every 5 min
  sync_checkpoints.sh -H node0 --latest-run --watch 300 -c legendvla

  # Only sync checkpoints after April 1st
  sync_checkpoints.sh -H node0 -r /path/to/run --since "2026-04-01" -c legendvla

  # Cron job (every 10 min):
  # */10 * * * * cd /path/to/EgoVLA && bash scripts/sync_checkpoints.sh -H node0 --latest-run -c legendvla >> /tmp/ckpt_sync.log 2>&1
USAGE
    exit 0
}

# ========================= Arg Parsing =========================

while [[ $# -gt 0 ]]; do
    case "$1" in
        -H|--host)             REMOTE_HOST="$2";       shift 2 ;;
        -r|--remote-dir)       REMOTE_DIR="$2";        shift 2 ;;
        -l|--local-dir)        LOCAL_DIR="$2";         shift 2 ;;
        -c|--conda)            CONDA_ENV="$2";         shift 2 ;;
        -n|--dry-run)          DRY_RUN=true;           shift   ;;
        -s|--step-checkpoints) STEP_CKPT_COUNT="$2";   shift 2 ;;
        --since)               SINCE="$2";             shift 2 ;;
        --no-consolidate)      NO_CONSOLIDATE=true;    shift   ;;
        --delete)              DELETE=true;             shift   ;;
        --delete-before)       DELETE_BEFORE="$2";     shift 2 ;;
        --watch)               WATCH_INTERVAL="$2";    shift 2 ;;
        --latest-run)          LATEST_RUN=true;        shift   ;;
        --max-parallel)        MAX_PARALLEL="$2";      shift 2 ;;
        -h|--help)             usage ;;
        *) log_error "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$REMOTE_HOST" ]]; then
    log_error "Missing required option: -H / --host"
    exit 1
fi

if [[ -z "$REMOTE_DIR" ]] && [[ "$LATEST_RUN" == false ]]; then
    log_error "Must specify -r / --remote-dir or --latest-run"
    exit 1
fi

# ========================= Helper: Remote Command =========================

# Build the python command prefix based on --conda setting.
# With conda: conda run -n ENV --no-banner python
# Without:    python
remote_python_cmd() {
    if [[ -n "$CONDA_ENV" ]]; then
        echo "conda run -n ${CONDA_ENV} --no-banner python"
    else
        echo "python"
    fi
}

remote_exec() {
    ssh -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE_HOST" "$@"
}

# ========================= SSH Connectivity Check =========================

log_step "Checking SSH connectivity to ${REMOTE_HOST}..."
if ! remote_exec "true" 2>/dev/null; then
    log_error "Cannot connect to ${REMOTE_HOST} via SSH"
    exit 1
fi
log_info "SSH connection OK"

# ========================= Discover Latest Run =========================

discover_latest_run() {
    # Hydra output dir format: outputs/{YYYY.MM.DD}/{HH.MM}_{name}_{experiment}
    # We find the remote-dir's parent (or use a known base) and pick the newest.
    local base
    if [[ -n "$REMOTE_DIR" ]]; then
        # Use parent of parent as base (go up from run dir to outputs/)
        base=$(dirname "$(dirname "$REMOTE_DIR")")
    else
        log_error "--latest-run requires either -r (to infer base) or a known outputs path"
        log_error "Provide -r /path/to/outputs/any_date/any_run to let the script infer the base"
        exit 1
    fi

    log_step "Discovering latest run under ${base}/ ..."
    local latest
    latest=$(remote_exec "find ${base} -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-")
    if [[ -z "$latest" ]]; then
        log_error "No run directories found under ${base}/"
        exit 1
    fi
    log_info "Latest run: ${latest}"
    REMOTE_DIR="$latest"
}

if [[ "$LATEST_RUN" == true ]]; then
    discover_latest_run
fi

# ========================= Collect Checkpoint Dirs =========================

# List checkpoint directories under checkpoints/ and optionally step_checkpoints/.
# Applies --since filtering.
collect_checkpoint_dirs() {
    local remote_run="$1"
    local dirs=()

    # Top-K checkpoints (always collected)
    local topk_dirs
    if [[ -n "$SINCE" ]]; then
        topk_dirs=$(remote_exec "find '${remote_run}/checkpoints' -mindepth 1 -maxdepth 1 -type d -newermt '${SINCE}' 2>/dev/null | sort")
    else
        topk_dirs=$(remote_exec "find '${remote_run}/checkpoints' -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort")
    fi
    while IFS= read -r d; do
        [[ -n "$d" ]] && dirs+=("$d")
    done <<< "$topk_dirs"

    # Step checkpoints (latest N, optionally filtered by --since)
    if [[ "$STEP_CKPT_COUNT" -gt 0 ]]; then
        local step_dirs
        if [[ -n "$SINCE" ]]; then
            step_dirs=$(remote_exec "find '${remote_run}/step_checkpoints' -mindepth 1 -maxdepth 1 -type d -newermt '${SINCE}' 2>/dev/null | sort -t_ -k3 -n | tail -n ${STEP_CKPT_COUNT}")
        else
            step_dirs=$(remote_exec "find '${remote_run}/step_checkpoints' -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -t_ -k3 -n | tail -n ${STEP_CKPT_COUNT}")
        fi
        while IFS= read -r d; do
            [[ -n "$d" ]] && dirs+=("$d")
        done <<< "$step_dirs"
    fi

    echo "${dirs[@]}"
}

# ========================= Remote Consolidation =========================

# Consolidate DCP shards into a single file on the remote.
# Idempotent: skips if consolidated.pt already exists.
# Ref: https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
consolidate_remote() {
    local ckpt_dir="$1"
    local py_cmd
    py_cmd=$(remote_python_cmd)

    # Check if already consolidated
    if remote_exec "test -f '${ckpt_dir}/consolidated.pt'" 2>/dev/null; then
        log_info "Already consolidated: $(basename "$ckpt_dir")"
        return 0
    fi

    # Check if DCP shards exist
    if ! remote_exec "test -d '${ckpt_dir}/pytorch_model_fsdp_0'" 2>/dev/null; then
        log_warn "No pytorch_model_fsdp_0/ in $(basename "$ckpt_dir"), skipping consolidation"
        return 1
    fi

    log_info "Consolidating shards: $(basename "$ckpt_dir") ..."
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[dry-run] Would run: ${py_cmd} -m torch.distributed.checkpoint.format_utils dcp_to_torch ..."
        return 0
    fi

    if ! remote_exec "${py_cmd} -m torch.distributed.checkpoint.format_utils dcp_to_torch '${ckpt_dir}/pytorch_model_fsdp_0' '${ckpt_dir}/consolidated.pt'" 2>&1; then
        log_error "Consolidation failed for $(basename "$ckpt_dir")"
        return 1
    fi
    log_info "Consolidation done: $(basename "$ckpt_dir")"
}

# ========================= Sync Functions =========================

# Sync metadata: .hydra/ config and training.log
sync_metadata() {
    local remote_run="$1"
    local local_run="$2"

    log_step "Syncing metadata..."
    mkdir -p "$local_run"

    local rsync_opts=(-avz -e ssh)
    [[ "$DRY_RUN" == true ]] && rsync_opts+=(--dry-run)

    # .hydra/ directory
    rsync "${rsync_opts[@]}" "${REMOTE_HOST}:${remote_run}/.hydra/" "${local_run}/.hydra/" 2>/dev/null || true

    # training.log
    rsync "${rsync_opts[@]}" "${REMOTE_HOST}:${remote_run}/training.log" "${local_run}/training.log" 2>/dev/null || true
}

# Sync a single checkpoint (consolidated.pt + custom_checkpoint_*.pkl).
# If --no-consolidate, pulls the raw DCP shards instead.
sync_single_checkpoint() {
    local remote_ckpt="$1"
    local local_ckpt="$2"

    mkdir -p "$local_ckpt"

    local rsync_opts=(-avz --progress -e ssh)
    [[ "$DRY_RUN" == true ]] && rsync_opts+=(--dry-run)
    [[ "$DELETE" == true ]] && rsync_opts+=(--delete)

    if [[ "$NO_CONSOLIDATE" == true ]]; then
        # Pull everything except optimizer/scheduler/random_states
        rsync "${rsync_opts[@]}" \
            --exclude='optimizer_*/' \
            --exclude='scheduler.bin' \
            --exclude='random_states_*.pkl' \
            --exclude='scaler.pt' \
            "${REMOTE_HOST}:${remote_ckpt}/" "${local_ckpt}/"
    else
        # Pull only consolidated.pt + custom checkpoints
        rsync "${rsync_opts[@]}" \
            --include='consolidated.pt' \
            --include='custom_checkpoint_*.pkl' \
            --exclude='*' \
            "${REMOTE_HOST}:${remote_ckpt}/" "${local_ckpt}/"
    fi
}

# Sync all collected checkpoints in parallel (up to MAX_PARALLEL).
sync_checkpoints_parallel() {
    local remote_run="$1"
    local local_run="$2"
    shift 2
    local ckpt_dirs=("$@")

    if [[ ${#ckpt_dirs[@]} -eq 0 ]]; then
        log_warn "No checkpoints found to sync"
        return 0
    fi

    log_step "Syncing ${#ckpt_dirs[@]} checkpoint(s) (max ${MAX_PARALLEL} parallel)..."

    # Phase 1: consolidate all (sequential, remote CPU-bound)
    if [[ "$NO_CONSOLIDATE" == false ]]; then
        for ckpt_dir in "${ckpt_dirs[@]}"; do
            consolidate_remote "$ckpt_dir" || true
        done
    fi

    # Phase 2: parallel rsync
    local running=0
    local pids=()
    for ckpt_dir in "${ckpt_dirs[@]}"; do
        # Determine local subpath: preserve checkpoints/ vs step_checkpoints/ structure
        local rel_path="${ckpt_dir#"${remote_run}/"}"
        local local_ckpt="${local_run}/${rel_path}"

        log_info "Syncing: ${rel_path}"
        sync_single_checkpoint "$ckpt_dir" "$local_ckpt" &
        pids+=($!)
        running=$((running + 1))

        # Throttle: wait if we hit MAX_PARALLEL
        if [[ $running -ge $MAX_PARALLEL ]]; then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")
            running=$((running - 1))
        fi
    done

    # Wait for remaining
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    log_info "All checkpoints synced"
}

# ========================= Local Cleanup =========================

delete_before() {
    local local_run="$1"
    local cutoff="$2"

    if [[ -z "$cutoff" ]]; then
        return 0
    fi

    log_step "Deleting local checkpoints older than ${cutoff}..."
    local cutoff_epoch
    # GNU date (-d) and BSD/macOS date (-j -f) have different syntax;
    # try GNU first, then BSD as fallback.
    cutoff_epoch=$(date -d "${cutoff}" "+%s" 2>/dev/null || \
                   date -j -f "%Y-%m-%d" "${cutoff}" "+%s" 2>/dev/null || \
                   date -j -f "%Y-%m-%d %H:%M" "${cutoff}" "+%s" 2>/dev/null || \
                   echo "")

    if [[ -z "$cutoff_epoch" ]]; then
        log_error "Cannot parse --delete-before date: ${cutoff}"
        return 1
    fi

    local deleted=0
    for dir_type in checkpoints step_checkpoints; do
        local base="${local_run}/${dir_type}"
        [[ -d "$base" ]] || continue
        for d in "$base"/*/; do
            [[ -d "$d" ]] || continue
            local dir_epoch
            # GNU stat (-c) and BSD/macOS stat (-f) have different syntax
            dir_epoch=$(stat -c "%Y" "$d" 2>/dev/null || stat -f "%m" "$d" 2>/dev/null || echo "0")
            if [[ "$dir_epoch" -lt "$cutoff_epoch" ]]; then
                if [[ "$DRY_RUN" == true ]]; then
                    log_info "[dry-run] Would delete: $d"
                else
                    rm -rf "$d"
                    log_info "Deleted: $d"
                fi
                deleted=$((deleted + 1))
            fi
        done
    done

    log_info "Deleted ${deleted} checkpoint(s)"
}

# ========================= Main Sync Orchestrator =========================

do_sync() {
    local local_run="${LOCAL_DIR}/$(basename "$(dirname "$REMOTE_DIR")")/$(basename "$REMOTE_DIR")"

    log_step "=== Sync started at $(date) ==="
    log_info "Remote: ${REMOTE_HOST}:${REMOTE_DIR}"
    log_info "Local:  ${local_run}"

    # Collect checkpoint dirs
    local ckpt_dirs_str
    ckpt_dirs_str=$(collect_checkpoint_dirs "$REMOTE_DIR")
    local ckpt_dirs=()
    read -ra ckpt_dirs <<< "$ckpt_dirs_str"

    # Sync metadata
    sync_metadata "$REMOTE_DIR" "$local_run"

    # Sync checkpoints
    sync_checkpoints_parallel "$REMOTE_DIR" "$local_run" "${ckpt_dirs[@]}"

    # Local cleanup
    if [[ -n "$DELETE_BEFORE" ]]; then
        delete_before "$local_run" "$DELETE_BEFORE"
    fi

    log_step "=== Sync completed at $(date) ==="
}

# ========================= Entry Point =========================

if [[ -n "$WATCH_INTERVAL" ]]; then
    log_info "Watch mode: syncing every ${WATCH_INTERVAL}s (Ctrl+C to stop)"
    while true; do
        # Re-discover latest run each iteration if --latest-run
        if [[ "$LATEST_RUN" == true ]]; then
            discover_latest_run
        fi
        do_sync
        log_info "Next sync in ${WATCH_INTERVAL}s..."
        sleep "$WATCH_INTERVAL"
    done
else
    do_sync
fi
