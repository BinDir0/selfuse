#!/usr/bin/env bash
# Build a "dirty-included, error-excluded" WDS variant for ablation experiments.
#
# Given a pre-filter WDS directory, this wrapper produces a sibling directory
# with:
#   - unloadable (error) samples dropped so training does not crash
#   - quality-rejected (dirty) samples retained
#   - dirty samples optionally upsampled by a small integer multiplier
#
# Pipeline (each step is a standalone script; see dirty-data ablation plan):
#   1. scripts/filter_and_check_datasets.py wds   -- scan + emit bad-keys JSONL
#   2. scripts/split_bad_keys_by_reason.py        -- split into error/dirty
#   3. hardlink pre-filter shards into DST_DIRTY
#   4. scripts/drop_bad_wds_frames.py             -- drop error samples only
#   5. scripts/duplicate_dirty_samples.py         -- optional dirty upsampling
#   6. scripts/filter_and_check_datasets.py wds   -- sanity re-scan
#
# Usage:
#   scripts/build_dirty_ablation_wds.sh \
#     --src   /data/<dataset>_pre_filter \
#     --dst   /data/<dataset>_dirty_ablation \
#     --work  /data/<dataset>_dirty_ablation.work \
#     [--multiplier 2] [--workers 8] [--skip-sanity]
#
# Intended to run on the production machine. On the office machine use it as a
# recipe, not an executor.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${PYTHON:-python3}"

SRC=""
DST=""
WORK=""
MULTIPLIER=2
WORKERS=4
SAMPLES_PER_DUP_SHARD=1000
SKIP_SANITY=0
ALLOW_LARGE_MULTIPLIER=0
FULL_SANITY=0

usage() {
  grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while (( "$#" )); do
  case "$1" in
    --src) SRC="$2"; shift 2 ;;
    --dst) DST="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --multiplier) MULTIPLIER="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --samples-per-dup-shard) SAMPLES_PER_DUP_SHARD="$2"; shift 2 ;;
    --allow-large-multiplier) ALLOW_LARGE_MULTIPLIER=1; shift ;;
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
DUP_REPORT="${WORK}/dirty_duplication_report.json"
DROP_REPORT="${DST}/drop_bad_wds_frames_report.json"
POST_BAD_KEYS="${WORK}/bad_keys.post_check.jsonl"
POST_SUMMARY="${WORK}/filter_summary.post_check.json"

log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*"; }

log "Step 1: scan ${SRC} for bad keys"
"${PY}" "${SCRIPT_DIR}/filter_and_check_datasets.py" wds \
  --shards "${SRC}"/shard-*.tar \
  --workers "${WORKERS}" \
  --check-media \
  --bad-keys-output "${BAD_KEYS}" \
  --summary "${FILTER_SUMMARY}"

log "Step 2: split bad keys by reason"
"${PY}" "${SCRIPT_DIR}/split_bad_keys_by_reason.py" \
  --bad-keys "${BAD_KEYS}" \
  --error-output "${ERR_KEYS}" \
  --dirty-output "${DIRTY_KEYS}" \
  --report "${SPLIT_REPORT}"

log "Step 3: hardlink pre-filter shards into ${DST}"
find "${SRC}" -maxdepth 1 -name 'shard-*.tar' -exec ln -f {} "${DST}"/ \;

log "Step 4: drop error samples (keep dirty)"
if [[ -s "${ERR_KEYS}" ]]; then
  "${PY}" "${SCRIPT_DIR}/drop_bad_wds_frames.py" \
    --bad-keys "${ERR_KEYS}" \
    --dst-dir "${DST}" \
    --report "${DROP_REPORT}" \
    --overwrite
else
  log "  no error-only keys; skipping drop"
fi

log "Step 5: duplicate dirty samples (multiplier=${MULTIPLIER})"
DUP_EXTRA=()
if (( ALLOW_LARGE_MULTIPLIER == 1 )); then
  DUP_EXTRA+=(--allow-large-multiplier)
fi
"${PY}" "${SCRIPT_DIR}/duplicate_dirty_samples.py" \
  --dst-dir "${DST}" \
  --dirty-keys "${DIRTY_KEYS}" \
  --multiplier "${MULTIPLIER}" \
  --samples-per-shard "${SAMPLES_PER_DUP_SHARD}" \
  --report "${DUP_REPORT}" \
  --overwrite \
  "${DUP_EXTRA[@]}"

if (( SKIP_SANITY == 1 )); then
  log "Skipping sanity re-scan (per --skip-sanity)"
  exit 0
fi

SANITY_SHARDS_FILE="${WORK}/sanity_shards.txt"
if (( FULL_SANITY == 1 )); then
  log "Step 6: FULL sanity re-scan of ${DST}"
  find "${DST}" -maxdepth 1 -name 'shard-*.tar' | sort > "${SANITY_SHARDS_FILE}"
else
  log "Step 6: incremental sanity re-scan (dup shards + shards rewritten in Step 4)"
  # Only shards that actually changed in Steps 4–5 need re-checking. Unchanged
  # shards in <DST_DIRTY> are hardlinks of <SRC> and were already scanned in
  # Step 1; re-decoding their images is pure cost. Use --full-sanity to override.
  : > "${SANITY_SHARDS_FILE}"
  find "${DST}" -maxdepth 1 -name 'shard-dup-*.tar' | sort >> "${SANITY_SHARDS_FILE}"
  if [[ -f "${DROP_REPORT}" ]]; then
    "${PY}" -c "import json,sys
report=json.load(open(sys.argv[1]))
for entry in report.get('shards', []):
    dest=entry.get('destination')
    if dest: print(dest)" "${DROP_REPORT}" | sort >> "${SANITY_SHARDS_FILE}"
  fi
fi

if [[ ! -s "${SANITY_SHARDS_FILE}" ]]; then
  log "  nothing to re-scan (no dup shards, no rewritten shards). Skipping Step 6."
  log "OK: ${DST} is ready for dirty-ablation training."
  exit 0
fi

log "  re-scanning $(wc -l <"${SANITY_SHARDS_FILE}") shard(s)"
xargs -r -a "${SANITY_SHARDS_FILE}" "${PY}" "${SCRIPT_DIR}/filter_and_check_datasets.py" wds \
  --workers "${WORKERS}" \
  --check-media \
  --bad-keys-output "${POST_BAD_KEYS}" \
  --summary "${POST_SUMMARY}" \
  --shards

POST_SPLIT_REPORT="${WORK}/bad_keys_split_report.post_check.json"
"${PY}" "${SCRIPT_DIR}/split_bad_keys_by_reason.py" \
  --bad-keys "${POST_BAD_KEYS}" \
  --error-output "${WORK}/bad_keys.post_check.error_only.jsonl" \
  --dirty-output "${WORK}/bad_keys.post_check.dirty_only.jsonl" \
  --report "${POST_SPLIT_REPORT}" >/dev/null
ERR_LEFT=$("${PY}" -c "import json,sys; print(json.load(open(sys.argv[1]))['error_records'])" "${POST_SPLIT_REPORT}")
log "Post-scan error-class records remaining: ${ERR_LEFT}"
if (( ERR_LEFT != 0 )); then
  echo "FAIL: ${ERR_LEFT} error-class records still present in ${DST}. Investigate before training." >&2
  exit 2
fi
log "OK: ${DST} is ready for dirty-ablation training."