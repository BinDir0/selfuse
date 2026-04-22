#!/usr/bin/env python3
"""Build a balanced descriptor manifest from clips with completed upstream stages."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.clip_manifest import load_clip_manifest, write_clip_manifest


@dataclass
class Partition:
    part_id: int
    records: list
    total_weight: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a manifest down to clips whose required stages are already complete, "
            "and optionally split the result into balanced partitions."
        )
    )
    parser.add_argument(
        "--source_manifest",
        type=str,
        action="append",
        default=[],
        help="Input clip manifest JSONL. May be passed multiple times.",
    )
    parser.add_argument(
        "--source_manifest_glob",
        type=str,
        action="append",
        default=[],
        help="Glob pattern for input manifests. May be passed multiple times.",
    )
    parser.add_argument(
        "--required_stages",
        type=str,
        default="detect_track,motion",
        help="Comma-separated upstream stages that must be complete for a clip to be kept.",
    )
    parser.add_argument(
        "--output_manifest",
        type=str,
        default=None,
        help="Output manifest path when writing a single filtered manifest.",
    )
    parser.add_argument(
        "--split_count",
        type=int,
        default=1,
        help="If >1, write balanced partitions instead of a single manifest.",
    )
    parser.add_argument(
        "--split_prefix",
        type=str,
        default=None,
        help="Output prefix for partitioned manifests. Files are named <prefix>.partXXXX.jsonl.",
    )
    parser.add_argument(
        "--report_out",
        type=str,
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--full_check",
        action="store_true",
        help="Run full stage validation instead of done-marker/fast validation.",
    )
    return parser.parse_args()


def _parse_required_stages(raw: str) -> list[str]:
    stages = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not stages:
        raise ValueError("--required_stages must not be empty")
    return stages


def _record_weight(record) -> int:
    return max(1, int(getattr(record.descriptor, "frame_count", 0) or 0))


def _resolve_source_manifests(args: argparse.Namespace) -> list[Path]:
    manifest_paths = []
    seen = set()

    for raw_path in args.source_manifest:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"source manifest not found: {path}")
        text = str(path)
        if text not in seen:
            seen.add(text)
            manifest_paths.append(path)

    for pattern in args.source_manifest_glob:
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            raise FileNotFoundError(f"source manifest glob matched nothing: {pattern}")
        for raw_path in matches:
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                continue
            text = str(path)
            if text not in seen:
                seen.add(text)
                manifest_paths.append(path)

    if not manifest_paths:
        raise ValueError("at least one --source_manifest or --source_manifest_glob is required")
    return manifest_paths


def _choose_record(current, candidate):
    current_seq_exists = Path(current.descriptor.seq_folder).exists()
    candidate_seq_exists = Path(candidate.descriptor.seq_folder).exists()
    if candidate_seq_exists and not current_seq_exists:
        return candidate
    if current_seq_exists and not candidate_seq_exists:
        return current
    if _record_weight(candidate) > _record_weight(current):
        return candidate
    return current


def _load_merged_records(manifest_paths: list[Path]):
    merged = {}
    duplicate_clip_ids = set()
    duplicate_conflicts = []

    for manifest_path in manifest_paths:
        for record in load_clip_manifest(manifest_path):
            existing = merged.get(record.clip_id)
            if existing is None:
                merged[record.clip_id] = record
                continue
            duplicate_clip_ids.add(record.clip_id)
            if (
                existing.descriptor.seq_folder != record.descriptor.seq_folder
                or existing.descriptor.frame_count != record.descriptor.frame_count
            ):
                if len(duplicate_conflicts) < 32:
                    duplicate_conflicts.append(
                        {
                            "clip_id": record.clip_id,
                            "kept_seq_folder": existing.descriptor.seq_folder,
                            "candidate_seq_folder": record.descriptor.seq_folder,
                            "kept_frame_count": existing.descriptor.frame_count,
                            "candidate_frame_count": record.descriptor.frame_count,
                            "chosen_seq_folder": _choose_record(existing, record).descriptor.seq_folder,
                        }
                    )
            merged[record.clip_id] = _choose_record(existing, record)

    return list(merged.values()), sorted(duplicate_clip_ids), duplicate_conflicts


def _get_stage_done_marker(seq_folder: Path, stage: str) -> Path:
    return seq_folder / f".{stage}.done"


def _stage_complete(stage: str, seq_folder: Path, *, full_check: bool) -> bool:
    if not seq_folder.exists():
        return False
    if not full_check:
        return _get_stage_done_marker(seq_folder, stage).exists()

    from lib.pipeline.stage_api import is_stage_complete

    return bool(is_stage_complete(stage, seq_folder, fast_check=False))


def _assign_balanced(records: list, split_count: int) -> list[Partition]:
    partitions = [Partition(part_id=index, records=[]) for index in range(split_count)]
    ordered = sorted(records, key=_record_weight, reverse=True)
    for record in ordered:
        target = min(partitions, key=lambda item: (item.total_weight, len(item.records), item.part_id))
        target.records.append(record)
        target.total_weight += _record_weight(record)
    return partitions


def main() -> None:
    args = parse_args()
    required_stages = _parse_required_stages(args.required_stages)
    source_manifests = _resolve_source_manifests(args)
    if args.split_count < 1:
        raise ValueError("--split_count must be >= 1")

    if args.split_count == 1 and not args.output_manifest:
        raise ValueError("--output_manifest is required when --split_count=1")
    if args.split_count > 1 and not args.split_prefix:
        raise ValueError("--split_prefix is required when --split_count>1")

    records, duplicate_clip_ids, duplicate_conflicts = _load_merged_records(source_manifests)
    kept_records = []
    dropped_examples = []
    stage_missing_counts = {stage: 0 for stage in required_stages}

    for record in records:
        seq_folder = Path(record.descriptor.seq_folder)
        missing = []
        for stage in required_stages:
            if not _stage_complete(stage, seq_folder, full_check=args.full_check):
                stage_missing_counts[stage] += 1
                missing.append(stage)
        if missing:
            if len(dropped_examples) < 32:
                dropped_examples.append(
                    {
                        "clip_id": record.clip_id,
                        "seq_folder": str(seq_folder),
                        "missing_stages": missing,
                        "frame_count": _record_weight(record),
                    }
                )
            continue
        kept_records.append(record)

    report = {
        "source_manifests": [str(path) for path in source_manifests],
        "required_stages": required_stages,
        "total_records": len(records),
        "kept_records": len(kept_records),
        "dropped_records": len(records) - len(kept_records),
        "duplicate_clip_id_count": len(duplicate_clip_ids),
        "duplicate_clip_ids_preview": duplicate_clip_ids[:32],
        "duplicate_conflicts": duplicate_conflicts,
        "stage_missing_counts": stage_missing_counts,
        "dropped_examples": dropped_examples,
    }

    if args.split_count == 1:
        output_manifest = Path(args.output_manifest).expanduser().resolve()
        write_clip_manifest(kept_records, output_manifest)
        report["output_manifests"] = [str(output_manifest)]
        report["partitions"] = [
            {
                "part_id": 0,
                "manifest_path": str(output_manifest),
                "clip_count": len(kept_records),
                "total_weight": sum(_record_weight(record) for record in kept_records),
            }
        ]
    else:
        prefix = Path(args.split_prefix).expanduser().resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        partitions = _assign_balanced(kept_records, args.split_count)
        output_manifests = []
        partition_summaries = []
        for partition in partitions:
            manifest_path = prefix.parent / f"{prefix.name}.part{partition.part_id:04d}.jsonl"
            write_clip_manifest(partition.records, manifest_path)
            output_manifests.append(str(manifest_path))
            partition_summaries.append(
                {
                    "part_id": partition.part_id,
                    "manifest_path": str(manifest_path),
                    "clip_count": len(partition.records),
                    "total_weight": partition.total_weight,
                }
            )
        report["output_manifests"] = output_manifests
        report["partitions"] = partition_summaries

    report_text = json.dumps(report, ensure_ascii=False, indent=2)
    print(report_text)
    if args.report_out:
        report_path = Path(args.report_out).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
