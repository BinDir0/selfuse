#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_fpha_strict_postprocess.sh \
    --run-dir /share_data/louwenjie/dataset_pipeline_logs/20260420_115248 \
    --output-dir /share_data/louwenjie/FPHA/final_vla_dataset/fpha_20260420_strict

Optional:
  --python PATH
  --annotation-root PATH
  --filter-workers N
EOF
}

RUN_DIR=""
OUTPUT_DIR=""
PYTHON_BIN="/share_data/guantianrui/environment/anaconda3/envs/hawor/bin/python3.10"
ANNOTATION_ROOT="/share_data/louwenjie/FPHA/fpha_annotations"
FILTER_WORKERS="8"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --annotation-root)
      ANNOTATION_ROOT="$2"
      shift 2
      ;;
    --filter-workers)
      FILTER_WORKERS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$RUN_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "--run-dir and --output-dir are required" >&2
  usage
  exit 1
fi

MANIFEST="$RUN_DIR/clip_manifest.motion.completed.slam.completed.infiller.completed.jsonl"
FILTERED="$RUN_DIR/clip_manifest.filtered.jsonl"
FILTER_REPORT="$RUN_DIR/filter_report.json"
VALIDATE_LOG="$RUN_DIR/validate_strict.log"
WDS_SANITY_REPORT="$RUN_DIR/wds_sanity.json"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing completed infiller manifest: $MANIFEST" >&2
  exit 1
fi

if [[ ! -d "$ANNOTATION_ROOT" ]]; then
  echo "Annotation root not found: $ANNOTATION_ROOT" >&2
  exit 1
fi

echo "[1/4] Strict manifest filter"
sudo nice -n -15 "$PYTHON_BIN" scripts/filter_manifest_by_quality.py \
  --input_manifest "$MANIFEST" \
  --output_manifest "$FILTERED" \
  --report_out "$FILTER_REPORT" \
  --stages detect_track,motion,slam,infiller \
  --workers "$FILTER_WORKERS" \
  --min_instruction_num 1 \
  --annotation_root "$ANNOTATION_ROOT" \
  --require_annotation

FILTERED_COUNT="$(wc -l < "$FILTERED" | tr -d ' ')"
if [[ "$FILTERED_COUNT" == "0" ]]; then
  echo >&2
  echo "Strict filter produced zero clips; aborting before build." >&2
  echo "  annotation_root: $ANNOTATION_ROOT" >&2
  echo "  filtered_manifest: $FILTERED" >&2
  echo "  filter_report: $FILTER_REPORT" >&2
  echo "This usually means one of these:" >&2
  echo "  1) annotation_root points to the wrong directory" >&2
  echo "  2) annotation files are missing for these clip_ids" >&2
  echo "  3) annotation JSON exists but status/instruction is invalid" >&2
  exit 1
fi

echo "[2/4] Build final dataset"
sudo nice -n -15 "$PYTHON_BIN" scripts/build_vla_from_manifest.py \
  --descriptor_manifest "$FILTERED" \
  --output_dir "$OUTPUT_DIR" \
  --annotation_root "$ANNOTATION_ROOT" \
  --require_annotation \
  --preprocess_workers 8 \
  --writer_workers 4 \
  --frames_per_shard 10000 \
  --mano_device cuda:0 \
  --source_fps 30 \
  --target_fps 30

echo "[3/4] Full validate"
sudo nice -n -15 "$PYTHON_BIN" scripts/validate_pipeline_run.py \
  --descriptor_manifest "$FILTERED" \
  --annotation_root "$ANNOTATION_ROOT" \
  --dataset_dir "$OUTPUT_DIR" \
  --stages detect_track,motion,slam,infiller \
  --max_clips 0 \
  --dataset_sample_checks 0 | tee "$VALIDATE_LOG"

echo "[4/4] Full WDS analyze"
sudo nice -n -15 "$PYTHON_BIN" scripts/sanity_check_webdataset.py \
  --source_shard_dir "$OUTPUT_DIR" \
  --report_out "$WDS_SANITY_REPORT"

echo
echo "Done."
echo "  Filter report: $FILTER_REPORT"
echo "  Validate log: $VALIDATE_LOG"
echo "  WDS sanity report: $WDS_SANITY_REPORT"
