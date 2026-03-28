#!/usr/bin/env python3
"""Build a frozen clip manifest snapshot from shard directories."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_parser():
    parser = argparse.ArgumentParser(description="Build clip manifest from shard directories")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--shard_root", type=str, help="Root directory containing shard-group directories")
    source_group.add_argument("--shard_dir_list", type=str, help="Text file with one shard directory per line")
    parser.add_argument("--source_id", type=str, required=True, help="Logical source identifier, e.g. buildai")
    parser.add_argument("--split", type=str, default="train", help="Dataset split label")
    parser.add_argument("--include_dirs", type=str, nargs="*", default=None, help="Optional shard-group directory names to include")
    parser.add_argument("--manifest_out", type=str, required=True, help="Output JSONL manifest path")
    parser.add_argument("--shard_dirs_out", type=str, default=None, help="Optional output text file listing resolved shard dirs")
    return parser


def main():
    from lib.pipeline.clip_manifest import (
        build_clip_manifest_records,
        discover_shard_dirs,
        write_clip_manifest,
        write_shard_dir_list,
    )

    args = get_parser().parse_args()

    if args.shard_dir_list:
        with open(args.shard_dir_list, "r", encoding="utf-8") as handle:
            shard_dirs = [line.strip() for line in handle if line.strip()]
    else:
        shard_dirs = discover_shard_dirs(args.shard_root, include_dirs=args.include_dirs)

    if not shard_dirs:
        raise RuntimeError("No shard directories found")

    records = build_clip_manifest_records(
        shard_dirs,
        source_id=args.source_id,
        split=args.split,
    )
    if not records:
        raise RuntimeError("No clips found while building manifest")

    write_clip_manifest(records, args.manifest_out)
    if args.shard_dirs_out:
        write_shard_dir_list(shard_dirs, args.shard_dirs_out)

    summary = {
        "source_id": args.source_id,
        "split": args.split,
        "shard_dir_count": len(shard_dirs),
        "clip_count": len(records),
        "manifest_out": str(Path(args.manifest_out).resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
