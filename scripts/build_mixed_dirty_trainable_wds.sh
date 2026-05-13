#!/usr/bin/env bash
# Build a "mixed dirty, error-excluded" trainable WDS from a mixed-source WDS.
#
# This wrapper assumes the source WDS already contains the desired mixed dirty
# injection (for example via repair_legacy_rot6d_wds.py with legacy rot6d kept
# for some episodes and generic instructions injected for others).
#
# Output:
#   - <WORK>/bad_keys.jsonl                       full checker findings
#   - <WORK>/bad_keys.error_only.jsonl            unloadable/error samples only
#   - <WORK>/bad_keys.dirty_only.jsonl            dirty analysis only
#   - <WORK>/bad_keys_split_report.json           per-reason summary
#   - <DST>                                      trainable WDS with only error samples removed
#
# Pipeline:
#   1. scan <SRC> with filter_and_check_datasets.py
#   2. split bad keys into error_only / dirty_only
#   3. hardlink <SRC> shards into <DST>
#   4. drop error_only samples from <DST>
#   5. optional sanity re-scan of changed shards in <DST>
#
# Usage:
#   scripts/build_mixed_dirty_trainable_wds.sh \
#     --src   /data/<mixed_source_wds> \
#     --dst   /data/<mixed_trainable_wds> \
#     --work  /data/<mixed_trainable_wds>.work \
#     [--workers 8] [--shard-start 0] [--shard-end 1000] [--skip-sanity] [--check-media]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

SRC=""
DST=""
WORK=""
WORKERS=4
CHECK_MEDIA=0
SKIP_SANITY=0
FULL_SANITY=0
SCAN_PROGRESS_INTERVAL=1000
SHARD_START=0
SHARD_END=""

usage() {
  tail -n +2 "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while (( "$#" )); do
  case "$1" in
    --src) SRC="$2"; shift 2 ;;
    --dst) DST="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --scan-progress-interval) SCAN_PROGRESS_INTERVAL="$2"; shift 2 ;;
    --shard-start) SHARD_START="$2"; shift 2 ;;
    --shard-end) SHARD_END="$2"; shift 2 ;;
    --check-media) CHECK_MEDIA=1; shift ;;
    --no-check-media) CHECK_MEDIA=0; shift ;;
    --skip-sanity) SKIP_SANITY=1; shift ;;
    --full-sanity) FULL_SANITY=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown argument: $1" >&2; usage 1 ;;
  esac
done

[[ -n "${SRC}"  ]] || { echo "--src is required"  >&2; exit 1; }
[[ -n "${DST}"  ]] || { echo "--dst is required"  >&2; exit 1; }
[[ -n "${WORK}" ]] || { echo "--work is required" >&2; exit 1; }
[[ -d "${SRC}"  ]] || { echo "--src not found: ${SRC}" >&2; exit 1; }

mkdir -p "${DST}" "${WORK}"

BAD_KEYS="${WORK}/bad_keys.jsonl"
ERR_KEYS="${WORK}/bad_keys.error_only.jsonl"
DIRTY_KEYS="${WORK}/bad_keys.dirty_only.jsonl"
FILTER_SUMMARY="${WORK}/filter_summary.json"
SPLIT_REPORT="${WORK}/bad_keys_split_report.json"
SCAN_PROGRESS="${WORK}/scan_progress.jsonl"
DROP_REPORT="${WORK}/drop_bad_wds_frames_report.json"
POST_BAD_KEYS="${WORK}/bad_keys.post_check.jsonl"
POST_SUMMARY="${WORK}/filter_summary.post_check.json"
POST_SPLIT_REPORT="${WORK}/bad_keys_split_report.post_check.json"
POST_SCAN_PROGRESS="${WORK}/scan_progress.post_check.jsonl"
SELECTED_SHARDS="${WORK}/selected_shards.txt"

log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*"; }

FILTER_EXTRA=()
if (( CHECK_MEDIA == 1 )); then
  FILTER_EXTRA+=(--check-media)
fi

FILTER_RANGE=(--shard-start "${SHARD_START}")
if [[ -n "${SHARD_END}" ]]; then
  FILTER_RANGE+=(--shard-end "${SHARD_END}")
fi

find "${SRC}" -maxdepth 1 -name 'shard-*.tar' \
  | sort \
  | awk -v start="${SHARD_START}" -v end="${SHARD_END}" 'NR > start && (end == "" || NR <= end)' \
  > "${SELECTED_SHARDS}"

if [[ ! -s "${SELECTED_SHARDS}" ]]; then
  echo "No shards selected by --shard-start ${SHARD_START} --shard-end ${SHARD_END:-<end>}" >&2
  exit 1
fi

log "Selected $(wc -l <"${SELECTED_SHARDS}") shard(s) from ${SRC} [${SHARD_START}, ${SHARD_END:-end})"

log "Step 1: scan ${SRC} for error and dirty keys"
"${PY}" "${SCRIPT_DIR}/filter_and_check_datasets.py" wds \
  --shards "${SRC}/shard-*.tar" \
  "${FILTER_RANGE[@]}" \
  --workers "${WORKERS}" \
  --bad-keys-output "${BAD_KEYS}" \
  --report "${FILTER_SUMMARY}" \
  --shard-progress-output "${SCAN_PROGRESS}" \
  --shard-progress-interval "${SCAN_PROGRESS_INTERVAL}" \
  "${FILTER_EXTRA[@]}"

log "Step 2: split bad keys into error_only and dirty_only"
"${PY}" "${SCRIPT_DIR}/split_bad_keys_by_reason.py" \
  --bad-keys "${BAD_KEYS}" \
  --error-output "${ERR_KEYS}" \
  --dirty-output "${DIRTY_KEYS}" \
  --report "${SPLIT_REPORT}"

log "  dirty analysis is kept in ${DIRTY_KEYS} and ${SPLIT_REPORT}"

log "Step 3: hardlink selected mixed-source shards into ${DST}"
while IFS= read -r shard; do
  ln -f "${shard}" "${DST}/"
done < "${SELECTED_SHARDS}"

log "Step 4: drop error-only samples from ${DST}"
if [[ -s "${ERR_KEYS}" ]]; then
  "${PY}" "${SCRIPT_DIR}/drop_bad_wds_frames.py" \
    --bad-keys "${ERR_KEYS}" \
    --dst-dir "${DST}" \
    --report "${DROP_REPORT}" \
    --overwrite
else
  log "  no error-only keys; skipping drop"
fi

if (( SKIP_SANITY == 1 )); then
  log "Skipping sanity re-scan (per --skip-sanity)"
  log "OK: ${DST} is ready as mixed-dirty trainable WDS."
  exit 0
fi

SANITY_SHARDS_FILE="${WORK}/sanity_shards.txt"
if (( FULL_SANITY == 1 )); then
  log "Step 5: FULL sanity re-scan of selected shards in ${DST}"
  while IFS= read -r shard; do
    printf '%s/%s\n' "${DST}" "$(basename "${shard}")"
  done < "${SELECTED_SHARDS}" > "${SANITY_SHARDS_FILE}"
else
  log "Step 5: incremental sanity re-scan of rewritten shards"
  : > "${SANITY_SHARDS_FILE}"
  if [[ -f "${DROP_REPORT}" ]]; then
    "${PY}" -c "import json,sys
report=json.load(open(sys.argv[1]))
for entry in report.get('shards', []):
    dest=entry.get('destination')
    if dest: print(dest)" "${DROP_REPORT}" | sort >> "${SANITY_SHARDS_FILE}"
  fi
fi

if [[ ! -s "${SANITY_SHARDS_FILE}" ]]; then
  log "  nothing to re-scan (no rewritten shards). Skipping Step 5."
  log "OK: ${DST} is ready as mixed-dirty trainable WDS."
  exit 0
fi

log "  re-scanning $(wc -l <"${SANITY_SHARDS_FILE}") shard(s)"
xargs -r -a "${SANITY_SHARDS_FILE}" "${PY}" "${SCRIPT_DIR}/filter_and_check_datasets.py" wds \
  --workers "${WORKERS}" \
  --bad-keys-output "${POST_BAD_KEYS}" \
  --report "${POST_SUMMARY}" \
  --shard-progress-output "${POST_SCAN_PROGRESS}" \
  --shard-progress-interval "${SCAN_PROGRESS_INTERVAL}" \
  "${FILTER_EXTRA[@]}" \
  --shards

"${PY}" "${SCRIPT_DIR}/split_bad_keys_by_reason.py" \
  --bad-keys "${POST_BAD_KEYS}" \
  --error-output "${WORK}/bad_keys.post_check.error_only.jsonl" \
  --dirty-output "${WORK}/bad_keys.post_check.dirty_only.jsonl" \
  --report "${POST_SPLIT_REPORT}" >/dev/null

ERR_LEFT=$("${PY}" -c "import json,sys; print(json.load(open(sys.argv[1]))['error_records'])" "${POST_SPLIT_REPORT}")
DIRTY_LEFT=$("${PY}" -c "import json,sys; print(json.load(open(sys.argv[1]))['dirty_records'])" "${POST_SPLIT_REPORT}")
log "Post-scan error-class records remaining: ${ERR_LEFT}"
log "Post-scan dirty-class records remaining: ${DIRTY_LEFT}"
if (( ERR_LEFT != 0 )); then
  echo "FAIL: ${ERR_LEFT} error-class records still present in ${DST}. Investigate before training." >&2
  exit 2
fi

log "OK: ${DST} is ready as mixed-dirty trainable WDS."
