#!/usr/bin/env python3
"""Validate manifest, stage outputs, annotations, and final dataset shards."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_parser():
    parser = argparse.ArgumentParser(description="Validate a whole-pipeline run")
    parser.add_argument("--descriptor_manifest", type=str, required=True, help="Frozen clip manifest JSONL")
    parser.add_argument("--annotation_root", type=str, default=None, help="Clip annotation sidecar directory")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Final dataset shard directory")
    parser.add_argument(
        "--stages",
        type=str,
        default="detect_track,motion,slam,infiller",
        help="Comma-separated stages to validate",
    )
    parser.add_argument("--max_clips", type=int, default=None, help="Limit manifest clips for quick validation")
    parser.add_argument("--dataset_sample_checks", type=int, default=10, help="How many output samples to inspect")
    return parser


def validate_manifest_outputs(records, stages):
    from lib.pipeline.stage_api import get_track_range, validate_stage_output_fast

    stats = {
        "clips_total": len(records),
        "clips_ok": 0,
        "clips_failed": 0,
        "stage_failures": {stage: 0 for stage in stages},
    }

    for record in records:
        seq_folder = Path(record.descriptor.seq_folder)
        clip_ok = True
        try:
            start_idx, end_idx = get_track_range(seq_folder, fast=True)
        except Exception:
            for stage in stages:
                stats["stage_failures"][stage] += 1
            stats["clips_failed"] += 1
            continue
        for stage in stages:
            if not validate_stage_output_fast(stage, seq_folder, start_idx, end_idx):
                stats["stage_failures"][stage] += 1
                clip_ok = False
        if clip_ok:
            stats["clips_ok"] += 1
        else:
            stats["clips_failed"] += 1
    return stats


def validate_annotations(records, annotation_root):
    from lib.pipeline.annotation_protocol import load_clip_annotation

    if not annotation_root:
        return None

    stats = {
        "valid": 0,
        "missing_annotation": 0,
        "invalid_json": 0,
        "invalid_status": 0,
        "empty_instruction": 0,
    }
    for record in records:
        annotation, error_code, _ = load_clip_annotation(annotation_root, record.clip_id)
        if annotation is not None:
            stats["valid"] += 1
        else:
            stats[error_code] = stats.get(error_code, 0) + 1
    return stats


def validate_dataset(dataset_dir, sample_checks):
    from lib.pipeline.exporters.webdataset_rewriter import iter_shard_samples

    if not dataset_dir:
        return None

    root = Path(dataset_dir)
    shard_paths = sorted(root.glob("*.tar"))
    if not shard_paths:
        raise RuntimeError(f"No dataset shards found in {dataset_dir}")

    checked = 0
    errors = []
    for shard_path in shard_paths:
        for sample in iter_shard_samples(str(shard_path)):
            checked += 1
            meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            required = ["clip_id", "instruction", "instruction_num", "presence"]
            missing = [key for key in required if key not in meta]
            if missing:
                errors.append({"shard": shard_path.name, "sample": sample["key"], "missing_meta_keys": missing})
            if checked >= sample_checks:
                return {
                    "shards": len(shard_paths),
                    "samples_checked": checked,
                    "errors": errors,
                }
    return {
        "shards": len(shard_paths),
        "samples_checked": checked,
        "errors": errors,
    }


def main():
    args = get_parser().parse_args()
    from lib.pipeline.clip_manifest import load_clip_manifest

    records = load_clip_manifest(args.descriptor_manifest)
    if args.max_clips is not None:
        records = records[: args.max_clips]

    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    summary = {
        "manifest": {
            "path": str(Path(args.descriptor_manifest).resolve()),
            "clips_checked": len(records),
        },
        "stages": validate_manifest_outputs(records, stages),
        "annotations": validate_annotations(records, args.annotation_root),
        "dataset": validate_dataset(args.dataset_dir, args.dataset_sample_checks),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
