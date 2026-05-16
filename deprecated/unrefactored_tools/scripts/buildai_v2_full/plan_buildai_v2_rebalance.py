#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from buildai_v2_full_status import (  # noqa: E402
    classify_shards,
    list_source_shard_indices,
    parse_shard_index_from_name,
    shard_name_from_index,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan unfinished BuildAI shard redistribution.")
    parser.add_argument("--source_shard_dir", default="/share_data/guantianrui/datasets/BuildAI-100K-part1-0401.with_instruction_v2")
    parser.add_argument(
        "--report_root",
        default=None,
        help="Optional report root; use when you want active/pending states from local report metadata",
    )
    parser.add_argument(
        "--completed_shard_dir",
        default=None,
        help="Optional flat or nested dir containing completed shard-XXXXXX.tar files; treated as source-of-truth done set",
    )
    parser.add_argument("--plan_dir", required=True, help="Directory to write shard lists and summary")
    parser.add_argument("--machine_count", type=int, required=True, help="How many target machines to split across")
    parser.add_argument(
        "--mode",
        choices=("pending_only", "not_done"),
        default="not_done",
        help="When using report_root, pending_only is safe while old workers keep running; with completed_shard_dir only, pending_only==not_done",
    )
    parser.add_argument(
        "--strategy",
        choices=("round_robin", "contiguous"),
        default="round_robin",
        help="How to split selected shards across target machines",
    )
    return parser


def _write_lines(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{shard_name_from_index(shard_idx)}\n" for shard_idx in values),
        encoding="utf-8",
    )


def _split_round_robin(values: list[int], machine_count: int) -> list[list[int]]:
    buckets = [[] for _ in range(machine_count)]
    for idx, value in enumerate(values):
        buckets[idx % machine_count].append(value)
    return buckets


def _split_contiguous(values: list[int], machine_count: int) -> list[list[int]]:
    total = len(values)
    buckets = []
    for machine_id in range(machine_count):
        start = machine_id * total // machine_count
        end = (machine_id + 1) * total // machine_count
        buckets.append(values[start:end])
    return buckets


def _snapshot_from_completed_dir(source_shard_dir: Path, completed_shard_dir: Path) -> dict:
    source_indices = list_source_shard_indices(source_shard_dir)
    done = []
    done_set = set()
    for path in completed_shard_dir.rglob("shard-*.tar"):
        shard_idx = parse_shard_index_from_name(path.stem)
        if shard_idx is None:
            continue
        done_set.add(shard_idx)
    for shard_idx in source_indices:
        if shard_idx in done_set:
            done.append(shard_idx)
    pending = [shard_idx for shard_idx in source_indices if shard_idx not in done_set]
    return {
        "source_indices": source_indices,
        "done": done,
        "active": [],
        "pending": pending,
        "not_done": pending,
    }


def main() -> None:
    args = build_parser().parse_args()
    source_shard_dir = Path(args.source_shard_dir).expanduser().resolve()
    plan_dir = Path(args.plan_dir).expanduser().resolve()
    report_root = Path(args.report_root).expanduser().resolve() if args.report_root else None
    completed_shard_dir = Path(args.completed_shard_dir).expanduser().resolve() if args.completed_shard_dir else None

    if not source_shard_dir.is_dir():
        raise FileNotFoundError(f"Source shard dir not found: {source_shard_dir}")
    if int(args.machine_count) <= 0:
        raise ValueError("--machine_count must be > 0")
    if report_root is None and completed_shard_dir is None:
        raise ValueError("Provide at least one of --report_root or --completed_shard_dir")
    if report_root is not None and not report_root.is_dir():
        raise FileNotFoundError(f"Report root not found: {report_root}")
    if completed_shard_dir is not None and not completed_shard_dir.is_dir():
        raise FileNotFoundError(f"Completed shard dir not found: {completed_shard_dir}")

    if completed_shard_dir is not None:
        snapshot = _snapshot_from_completed_dir(source_shard_dir, completed_shard_dir)
    else:
        snapshot = classify_shards(source_shard_dir, report_root)
    selected = snapshot["pending"] if args.mode == "pending_only" else snapshot["not_done"]
    splitter = _split_round_robin if args.strategy == "round_robin" else _split_contiguous
    buckets = splitter(selected, int(args.machine_count))

    plan_dir.mkdir(parents=True, exist_ok=True)
    _write_lines(plan_dir / "done_shards.txt", snapshot["done"])
    _write_lines(plan_dir / "active_shards.txt", snapshot["active"])
    _write_lines(plan_dir / "pending_shards.txt", snapshot["pending"])
    _write_lines(plan_dir / "not_done_shards.txt", snapshot["not_done"])

    assignments = {}
    for machine_id, shard_indices in enumerate(buckets, start=1):
        file_name = f"machine{machine_id:02d}.{args.mode}.txt"
        _write_lines(plan_dir / file_name, shard_indices)
        assignments[f"machine{machine_id:02d}"] = {
            "count": len(shard_indices),
            "file": str((plan_dir / file_name).resolve()),
            "first": shard_name_from_index(shard_indices[0]) if shard_indices else None,
            "last": shard_name_from_index(shard_indices[-1]) if shard_indices else None,
        }

    summary = {
        "source_shard_dir": str(source_shard_dir),
        "report_root": str(report_root) if report_root is not None else None,
        "completed_shard_dir": str(completed_shard_dir) if completed_shard_dir is not None else None,
        "plan_dir": str(plan_dir),
        "mode": args.mode,
        "strategy": args.strategy,
        "machine_count": int(args.machine_count),
        "total_shards": len(snapshot["source_indices"]),
        "done_count": len(snapshot["done"]),
        "active_count": len(snapshot["active"]),
        "pending_count": len(snapshot["pending"]),
        "not_done_count": len(snapshot["not_done"]),
        "selected_count": len(selected),
        "assignments": assignments,
    }
    (plan_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
