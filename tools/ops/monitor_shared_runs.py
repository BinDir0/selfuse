#!/usr/bin/env python3
"""Monitor multiple pipeline run directories from a shared filesystem."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.shared_run_monitor import format_stage_progress, summarize_runs


def _format_stage_progress(item: dict, max_stages: int) -> str:
    return format_stage_progress(item, max_stages)


def _print_table(rows: list[dict], *, max_stages: int, show_errors: bool) -> None:
    headers = ["RUN", "SRC", "DONE", "FAIL", "PART", "RUN", "PEND", "LAST", "STAGES", "CONFIG"]
    widths = [24, 6, 7, 7, 7, 7, 7, 7, 36, 28]

    def fmt(text, width):
        value = str(text)
        return value if len(value) <= width else value[: width - 1] + "…"

    print(" ".join(fmt(header, width).ljust(width) for header, width in zip(headers, widths)))
    print(" ".join("-" * width for width in widths))
    for item in rows:
        total = max(1, int(item["total"]))
        values = [
            item["run_tag"],
            item["status_source"],
            f"{item['completed']}/{total}",
            f"{item['failed']}/{total}",
            f"{item['partial']}/{total}",
            item["running"],
            item["pending"],
            item["last_event_age"],
            _format_stage_progress(item, max_stages),
            item["config"],
        ]
        print(" ".join(fmt(value, width).ljust(width) for value, width in zip(values, widths)))
        if show_errors and item["status_errors"]:
            for error in item["status_errors"]:
                print(f"  warning[{item['run_tag']}]: {error}")


def _print_json(rows: list[dict]) -> None:
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor multiple pipeline run directories on a shared filesystem")
    parser.add_argument("--log_root", required=True, help="Shared dataset_pipeline_logs root")
    parser.add_argument("--run_tags", default=None, help="Comma-separated run tags to inspect")
    parser.add_argument("--pattern", default="20*", help="Glob pattern under log_root when --run_tags is not set")
    parser.add_argument("--limit", type=int, default=12, help="Only show the last N matching run dirs")
    parser.add_argument("--watch", type=float, default=0.0, help="Refresh every N seconds; 0 disables watch mode")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a text table")
    parser.add_argument("--max_stages", type=int, default=4, help="How many stage progress columns to display inside STAGES")
    parser.add_argument("--show_errors", action="store_true", help="Print status recovery warnings under the table")
    parser.add_argument("--stall_seconds", type=int, default=1800, help="Mark active runs as stalled if no events arrive for this long")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    log_root = Path(args.log_root).resolve()
    if not log_root.is_dir():
        raise SystemExit(f"log_root not found: {log_root}")

    run_tags = None
    if args.run_tags:
        run_tags = [item.strip() for item in str(args.run_tags).split(",") if item.strip()]

    while True:
        rows = summarize_runs(
            log_root,
            run_tags=run_tags,
            pattern=args.pattern,
            limit=args.limit,
            stall_seconds=args.stall_seconds,
        )

        if args.watch > 0:
            print("\033[2J\033[H", end="")
            print(f"Shared Run Monitor | root={log_root} | updated={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()

        if args.json:
            _print_json(rows)
        else:
            _print_table(rows, max_stages=max(1, int(args.max_stages)), show_errors=bool(args.show_errors))

        if args.watch <= 0:
            break
        time.sleep(float(args.watch))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
