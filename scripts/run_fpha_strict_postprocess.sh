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
  --check-workers N
EOF
}

RUN_DIR=""
OUTPUT_DIR=""
PYTHON_BIN="/share_data/guantianrui/environment/anaconda3/envs/hawor/bin/python3.10"
ANNOTATION_ROOT="/share_data/louwenjie/FPHA/fpha_annotations"
FILTER_WORKERS="8"
CHECK_WORKERS="16"

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
    --check-workers)
      CHECK_WORKERS="$2"
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
WDS_INSTR_REPORT="$RUN_DIR/wds_instruction_check.json"
WDS_SANITY_REPORT="$RUN_DIR/wds_sanity.json"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing completed infiller manifest: $MANIFEST" >&2
  exit 1
fi

echo "[1/5] Strict manifest filter"
sudo nice -n -15 "$PYTHON_BIN" scripts/filter_manifest_by_quality.py \
  --input_manifest "$MANIFEST" \
  --output_manifest "$FILTERED" \
  --report_out "$FILTER_REPORT" \
  --stages detect_track,motion,slam,infiller \
  --workers "$FILTER_WORKERS" \
  --drop_nonfinite_lowdim \
  --min_instruction_num 1 \
  --annotation_root "$ANNOTATION_ROOT" \
  --require_annotation

echo "[2/5] Build final dataset"
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

echo "[3/5] Full validate"
sudo nice -n -15 "$PYTHON_BIN" scripts/validate_pipeline_run.py \
  --descriptor_manifest "$FILTERED" \
  --annotation_root "$ANNOTATION_ROOT" \
  --dataset_dir "$OUTPUT_DIR" \
  --stages detect_track,motion,slam,infiller \
  --max_clips 0 \
  --dataset_sample_checks 0 | tee "$VALIDATE_LOG"

echo "[4/5] Full instruction scan"
sudo nice -n -15 "$PYTHON_BIN" scripts/check_wds_instructions.py \
  --source_shard_dir "$OUTPUT_DIR" \
  --workers "$CHECK_WORKERS" \
  --summary_only \
  --max_examples 0 \
  --report_out "$WDS_INSTR_REPORT"

echo "[5/5] Full WDS sanity"
sudo nice -n -15 "$PYTHON_BIN" scripts/sanity_check_webdataset.py \
  --source_shard_dir "$OUTPUT_DIR" \
  --report_out "$WDS_SANITY_REPORT"

echo
echo "Done."
echo "  Filter report: $FILTER_REPORT"
echo "  Validate log: $VALIDATE_LOG"
echo "  Instruction report: $WDS_INSTR_REPORT"
echo "  WDS sanity report: $WDS_SANITY_REPORT"
