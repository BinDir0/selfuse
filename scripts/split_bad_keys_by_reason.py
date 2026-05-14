#!/usr/bin/env python3
"""Split a bad-keys JSONL from filter_and_check_datasets.py by reason class.

filter_and_check_datasets.py emits one JSONL record per failing sample with a
``reason`` field that is the exception class name. This script bucketizes those
records into:

  - error_only: samples that are unloadable (would crash training). These are
    ``MissingOrInvalidFilesError``, ``NonFiniteDataError``, plus any
    ``Unexpected*`` reason produced when filter_and_check_datasets.py catches
    a non-DataSkipError exception.
  - dirty_only: samples that passed loading but failed quality thresholds.

The two output files feed the dirty-data ablation pipeline:
  - error_only -> drop_bad_wds_frames.py (remove so training does not crash)
  - dirty_only -> duplicate_dirty_samples.py (optional upsampling)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else ()


ERROR_REASONS = frozenset({"MissingOrInvalidFilesError", "NonFiniteDataError"})
ERROR_REASON_PREFIXES = ("Unexpected",)


def is_error_reason(reason: str) -> bool:
    if reason in ERROR_REASONS:
        return True
    return any(reason.startswith(prefix) for prefix in ERROR_REASON_PREFIXES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split bad-keys JSONL into error_only and dirty_only buckets by reason.",
    )
    parser.add_argument(
        "--bad-keys",
        required=True,
        help="Input JSONL produced by filter_and_check_datasets.py --bad-keys-output.",
    )
    parser.add_argument(
        "--error-output",
        required=True,
        help="Output JSONL for training-unsafe samples (MissingOrInvalidFilesError, NonFiniteDataError, Unexpected*).",
    )
    parser.add_argument(
        "--dirty-output",
        required=True,
        help="Output JSONL for quality-rejected samples (all other DataSkipError subclasses).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON report path summarising per-reason counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.bad_keys)
    error_path = Path(args.error_output)
    dirty_path = Path(args.dirty_output)

    error_path.parent.mkdir(parents=True, exist_ok=True)
    dirty_path.parent.mkdir(parents=True, exist_ok=True)

    counts_total: Counter[str] = Counter()
    counts_error: Counter[str] = Counter()
    counts_dirty: Counter[str] = Counter()
    missing_reason = 0

    with (
        src.open("r", encoding="utf-8") as src_file,
        error_path.open("w", encoding="utf-8") as error_file,
        dirty_path.open("w", encoding="utf-8") as dirty_file,
    ):
        iterator = tqdm(src_file, desc="Splitting bad keys", unit=" rec")
        for line in iterator:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            reason = record.get("reason")
            if not reason:
                missing_reason += 1
                continue
            counts_total[reason] += 1
            if is_error_reason(reason):
                error_file.write(stripped + "\n")
                counts_error[reason] += 1
            else:
                dirty_file.write(stripped + "\n")
                counts_dirty[reason] += 1

    total = sum(counts_total.values())
    total_error = sum(counts_error.values())
    total_dirty = sum(counts_dirty.values())

    print(f"Input: {src} ({total} records)")
    if missing_reason:
        print(f"  !! skipped {missing_reason} records without a 'reason' field")
    print(f"  error_only -> {error_path} ({total_error} records)")
    print(f"  dirty_only -> {dirty_path} ({total_dirty} records)")
    print("Per-reason counts:")
    for reason, count in sorted(counts_total.items(), key=lambda kv: -kv[1]):
        bucket = "error" if is_error_reason(reason) else "dirty"
        print(f"  [{bucket:5s}] {reason}: {count}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "source": str(src),
            "error_output": str(error_path),
            "dirty_output": str(dirty_path),
            "total_records": total,
            "error_records": total_error,
            "dirty_records": total_dirty,
            "missing_reason_records": missing_reason,
            "reason_counts": dict(counts_total),
            "error_reason_counts": dict(counts_error),
            "dirty_reason_counts": dict(counts_dirty),
            "error_reasons": sorted(ERROR_REASONS),
            "error_reason_prefixes": list(ERROR_REASON_PREFIXES),
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
