#!/usr/bin/env python3
"""Rewrite WebDataset shards while dropping bad sample keys.

The intended workflow is:
  1. Hardlink/copy all original shards into a new dataset directory.
  2. Run this script with the bad-keys JSONL produced by
     data/filter_and_check_datasets.py.
  3. Only shards containing bad samples are rewritten in the new directory.

Input JSONL records must contain at least:
  {"key": "...", "shard": "/path/to/original/shard.tar", ...}
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop bad WebDataset sample keys by rewriting affected shards."
    )
    parser.add_argument(
        "--bad-keys",
        required=True,
        help="JSONL file with records containing 'key' and 'shard'.",
    )
    parser.add_argument(
        "--dst-dir",
        required=True,
        help="Destination dataset directory that already contains copied/hardlinked shards.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON report path. Defaults to <dst-dir>/drop_bad_wds_frames_report.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be rewritten; do not write tar files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing destination shard files.",
    )
    return parser.parse_args()


def load_bad_keys(path: str | Path) -> dict[str, set[str]]:
    bad_by_shard: dict[str, set[str]] = defaultdict(set)
    with open(path, "r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, 1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            key = record.get("key")
            shard = record.get("shard")
            if not key or not shard:
                raise ValueError(f"{path}:{line_no}: record must contain key and shard")
            bad_by_shard[str(shard)].add(str(key))
    return dict(bad_by_shard)


def dropped_key_for_member(member_name: str, bad_keys: set[str], bad_prefixes: tuple[str, ...]) -> str | None:
    """Return the bad sample key owning this member, if it should be dropped.

    WebDataset stores one sample as multiple tar members named like:
      <key>.image.jpg, <key>.lowdim.npy, <key>.meta.json, ...

    Matching by ``bad_key + "."`` intentionally drops every field for the
    sample, including uncommon sidecar extensions, without trying to enumerate
    all possible suffixes.
    """
    if member_name in bad_keys:
        return member_name
    if not member_name.startswith(bad_prefixes):
        return None
    for key in bad_keys:
        if member_name.startswith(key + "."):
            return key
    return None


def rewrite_shard(
    *,
    src_path: Path,
    dst_path: Path,
    bad_keys: set[str],
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if not src_path.exists():
        raise FileNotFoundError(f"source shard not found: {src_path}")
    if dst_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(
            f"destination shard exists: {dst_path}. "
            "Pass --overwrite after confirming this is the copied/hardlinked directory."
        )

    stats: dict[str, Any] = {
        "source": str(src_path),
        "destination": str(dst_path),
        "bad_keys_requested": len(bad_keys),
        "members_total": 0,
        "members_kept": 0,
        "members_dropped": 0,
        "sample_keys_dropped": [],
        "dry_run": dry_run,
    }

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_name(dst_path.name + ".tmp")
    if tmp_path.exists() and not dry_run:
        tmp_path.unlink()

    dropped_sample_keys: set[str] = set()
    bad_prefixes = tuple(key + "." for key in bad_keys)
    started = time.time()

    with tarfile.open(src_path, "r:*") as src:
        dst = None if dry_run else tarfile.open(tmp_path, "w")
        try:
            for member in src:
                stats["members_total"] += 1
                dropped_key = dropped_key_for_member(member.name, bad_keys, bad_prefixes)
                if dropped_key is not None:
                    stats["members_dropped"] += 1
                    dropped_sample_keys.add(dropped_key)
                    continue

                stats["members_kept"] += 1
                if dry_run:
                    continue
                file_obj = src.extractfile(member) if member.isfile() else None
                dst.addfile(member, file_obj)
        finally:
            if dst is not None:
                dst.close()

    stats["sample_keys_dropped"] = sorted(dropped_sample_keys)
    stats["elapsed_sec"] = time.time() - started
    missing = sorted(set(bad_keys) - dropped_sample_keys)
    stats["bad_keys_missing_in_shard"] = missing

    if not dry_run:
        os.replace(tmp_path, dst_path)

    return stats


def main() -> None:
    args = parse_args()
    dst_dir = Path(args.dst_dir)
    report_path = Path(args.report) if args.report else dst_dir / "drop_bad_wds_frames_report.json"
    bad_by_shard = load_bad_keys(args.bad_keys)

    report: dict[str, Any] = {
        "bad_keys": args.bad_keys,
        "dst_dir": str(dst_dir),
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "shards_to_rewrite": len(bad_by_shard),
        "shards": [],
    }

    for idx, (src, bad_keys) in enumerate(sorted(bad_by_shard.items()), 1):
        src_path = Path(src)
        dst_path = dst_dir / src_path.name
        print(
            f"[{idx}/{len(bad_by_shard)}] {src_path.name}: "
            f"drop {len(bad_keys)} sample keys",
            flush=True,
        )
        stats = rewrite_shard(
            src_path=src_path,
            dst_path=dst_path,
            bad_keys=bad_keys,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        report["shards"].append(stats)
        print(
            f"  members kept={stats['members_kept']} "
            f"dropped={stats['members_dropped']} "
            f"missing_keys={len(stats['bad_keys_missing_in_shard'])} "
            f"elapsed={stats['elapsed_sec']:.1f}s",
            flush=True,
        )

    report["members_total"] = sum(item["members_total"] for item in report["shards"])
    report["members_dropped"] = sum(item["members_dropped"] for item in report["shards"])
    report["sample_keys_dropped"] = sum(len(item["sample_keys_dropped"]) for item in report["shards"])
    report["missing_bad_keys"] = sum(len(item["bad_keys_missing_in_shard"]) for item in report["shards"])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")
    if report["missing_bad_keys"]:
        raise SystemExit(f"Some bad keys were not found in their shards: {report['missing_bad_keys']}")


if __name__ == "__main__":
    main()
