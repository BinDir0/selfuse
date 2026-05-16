#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from buildai_v2_full_status import (  # noqa: E402
    classify_shards,
    shard_tar_name_from_index,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage completed BuildAI rewritten shards into a flat directory for fpsync.")
    parser.add_argument("--source_shard_dir", default="/share_data/guantianrui/datasets/BuildAI-100K-part1-0401.with_instruction_v2")
    parser.add_argument("--report_root", default="/DATA/guantianrui/buildai_v2_rewrite_reports_full")
    parser.add_argument("--stage_dir", required=True, help="Flat output dir containing shard-XXXXXX.tar")
    parser.add_argument(
        "--link_mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="hardlink is cheap but requires same filesystem; copy is slower but always works",
    )
    parser.add_argument(
        "--prune",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove staged shard files that are no longer marked completed",
    )
    return parser


def _materialize(src: Path, dst: Path, mode: str) -> str:
    if dst.exists():
        if dst.is_file():
            try:
                if src.samefile(dst):
                    return "reused"
            except FileNotFoundError:
                pass
            dst.unlink()
        else:
            raise RuntimeError(f"Stage path is not a file: {dst}")

    if mode == "hardlink":
        os.link(src, dst)
        return "linked"

    shutil.copy2(src, dst)
    return "copied"


def main() -> None:
    args = build_parser().parse_args()
    source_shard_dir = Path(args.source_shard_dir).expanduser().resolve()
    report_root = Path(args.report_root).expanduser().resolve()
    stage_dir = Path(args.stage_dir).expanduser().resolve()

    if not source_shard_dir.is_dir():
        raise FileNotFoundError(f"Source shard dir not found: {source_shard_dir}")
    if not report_root.is_dir():
        raise FileNotFoundError(f"Report root not found: {report_root}")

    snapshot = classify_shards(source_shard_dir, report_root)
    records = snapshot["records"]
    stage_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    copied = 0
    reused = 0
    missing_outputs = []
    staged_names = set()

    for shard_idx in snapshot["done"]:
        record = records.get(shard_idx)
        if record is None or record.output_path is None or not record.output_path.is_file():
            missing_outputs.append(shard_tar_name_from_index(shard_idx))
            continue
        dst = stage_dir / shard_tar_name_from_index(shard_idx)
        result = _materialize(record.output_path, dst, args.link_mode)
        staged_names.add(dst.name)
        if result == "linked":
            linked += 1
        elif result == "copied":
            copied += 1
        else:
            reused += 1

    pruned = 0
    if bool(args.prune):
        for path in stage_dir.glob("shard-*.tar"):
            if path.name not in staged_names:
                path.unlink()
                pruned += 1

    summary = {
        "source_shard_dir": str(source_shard_dir),
        "report_root": str(report_root),
        "stage_dir": str(stage_dir),
        "link_mode": args.link_mode,
        "done_count": len(snapshot["done"]),
        "active_count": len(snapshot["active"]),
        "pending_count": len(snapshot["pending"]),
        "linked": linked,
        "copied": copied,
        "reused": reused,
        "pruned": pruned,
        "missing_outputs": missing_outputs,
    }
    (stage_dir / "stage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage_dir / "completed_shards.txt").write_text(
        "".join(f"{shard_tar_name_from_index(shard_idx)}\n" for shard_idx in snapshot["done"]),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
