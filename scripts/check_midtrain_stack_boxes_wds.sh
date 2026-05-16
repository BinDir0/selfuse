#!/bin/bash

# Strict-ish preflight for the stack-box mid-training WDS config.
# It checks configured train/val VLA shards, then exports real collated train
# and val batches for HTML/manual inspection.
#
# Usage:
#   CONFIG=src/config/experiment/legendvla_qwen3_vl_midtrain_stack_boxes_smoke.yaml \
#     bash scripts/check_midtrain_stack_boxes_wds.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

CONFIG="${CONFIG:-src/config/experiment/legendvla_qwen3_vl_midtrain_stack_boxes_smoke.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/preflight/midtrain_stack_boxes}"
MAX_SCAN_SAMPLES="${MAX_SCAN_SAMPLES:-200}"
SAMPLE_COUNT="${SAMPLE_COUNT:-9}"
NUM_WORKERS="${NUM_WORKERS:-0}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

python - "$CONFIG" "$OUTPUT_DIR" "$MAX_SCAN_SAMPLES" <<'PY'
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

from omegaconf import OmegaConf

from data.filter_and_check_datasets import scan_wds, write_report
from src.tests.debug_dataloader_batch import load_config

config_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
max_samples = int(sys.argv[3])

cfg = load_config(config_path)
reports = {}

for split, key in (("train", "vla_wds_datasets"), ("val", "val_vla_wds_datasets")):
    reports[split] = {}
    datasets = cfg.get(key, [])
    if not datasets:
        raise SystemExit(f"{key} is empty in {config_path}")
    for ds in datasets:
        name = str(ds.get("name", "unnamed"))
        shard_urls = OmegaConf.to_container(ds["shard_urls"], resolve=True)
        shards = [shard_urls] if isinstance(shard_urls, str) else list(shard_urls)
        report_path = output_dir / f"{split}_{name}_wds_report.json"
        bad_path = output_dir / f"{split}_{name}_bad_keys.jsonl"
        args = Namespace(
            kind="vla",
            shards=shards,
            max_samples=max_samples,
            check_media=True,
            check_depth=False,
            check_image_quality=False,
            target_image_size=None,
            max_shard_fail_rate=0.0,
            report=str(report_path),
            good_keys_output=None,
            bad_keys_output=str(bad_path),
            filtered_shards_output=None,
        )
        report = scan_wds(args)
        write_report(report, str(report_path))
        reports[split][name] = report
        if report["failed"] > 0:
            raise SystemExit(
                f"{split}/{name} has {report['failed']} bad samples; see {bad_path}"
            )

summary_path = output_dir / "wds_scan_summary.json"
summary_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"WDS scan reports written under {output_dir}")
PY

normalizer_args=()
if [[ -n "${NORMALIZER_PATH:-}" ]]; then
    normalizer_args=(--normalizer_path "$NORMALIZER_PATH")
fi

python -m src.tests.codex.run_real_batch_report \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR/real_batch_train" \
    --sample_count "$SAMPLE_COUNT" \
    --split train \
    --dataset_kind unified \
    --num_workers "$NUM_WORKERS" \
    "${normalizer_args[@]}"

python -m src.tests.codex.run_real_batch_report \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR/real_batch_val" \
    --sample_count "$SAMPLE_COUNT" \
    --split val \
    --dataset_kind unified \
    --num_workers "$NUM_WORKERS" \
    "${normalizer_args[@]}"

echo "Mid-train WDS preflight complete: $OUTPUT_DIR"
