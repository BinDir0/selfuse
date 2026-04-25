#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

SHARD_DIR_RE = re.compile(r"^shard-(\d{6})$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor BuildAI shard-list rebalance runs.")
    parser.add_argument(
        "--plan_dir",
        required=True,
        help="Directory containing machineXX.not_done.txt files",
    )
    parser.add_argument(
        "--report_root_pattern",
        default="/share_data/guantianrui/buildai_v2_rewrite_reports_rebalance_{machine}",
        help="Pattern for each machine report root; {machine} will be replaced with machine01, machine02, ...",
    )
    parser.add_argument("--refresh_sec", type=float, default=20.0)
    parser.add_argument("--once", action="store_true")
    return parser


def _machine_files(plan_dir: Path) -> list[tuple[str, Path]]:
    items = []
    for path in sorted(plan_dir.glob("machine*.not_done.txt")):
        machine = path.name.split(".", 1)[0]
        items.append((machine, path))
    return items


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _state_of(shard_dir: Path) -> str:
    if (shard_dir / "shard_summary.json").is_file():
        return "done"
    if (shard_dir / "rewrite_report.json").is_file():
        return "rewrite"
    if (shard_dir / "rerun_report.json").is_file() or (shard_dir / "scan_report.json").is_file():
        return "rerun"
    return "started"


def _shard_sort_key(shard_dir: Path) -> tuple[int, str]:
    match = SHARD_DIR_RE.match(shard_dir.name)
    if match is None:
        return (-1, shard_dir.name)
    return (int(match.group(1)), shard_dir.name)


def _last_nonempty_line(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return None


def _machine_snapshot(machine: str, plan_file: Path, report_root: Path) -> dict:
    assigned = _count_lines(plan_file)
    shard_dirs = [path for path in report_root.glob("shards_*_*/per_shard/shard-*") if path.is_dir()]

    done = 0
    rewrite = 0
    rerun = 0
    started = 0
    active = []

    for shard_dir in sorted(shard_dirs, key=_shard_sort_key):
        state = _state_of(shard_dir)
        if state == "done":
            done += 1
        elif state == "rewrite":
            rewrite += 1
            active.append((shard_dir.name, state))
        elif state == "rerun":
            rerun += 1
            active.append((shard_dir.name, state))
        else:
            started += 1
            active.append((shard_dir.name, state))

    latest_logs = []
    for log_path in sorted(report_root.glob("*_rebalance_logs/*.log")):
        last_line = _last_nonempty_line(log_path)
        if last_line:
            latest_logs.append((log_path.name, last_line))

    return {
        "machine": machine,
        "plan_file": plan_file,
        "report_root": report_root,
        "assigned": assigned,
        "done": done,
        "rewrite": rewrite,
        "rerun": rerun,
        "started": started,
        "pending": max(0, assigned - done - rewrite - rerun - started),
        "active": active[:8],
        "logs": latest_logs[-8:],
    }


def snapshot(plan_dir: Path, report_root_pattern: str) -> list[dict]:
    machines = []
    for machine, plan_file in _machine_files(plan_dir):
        report_root = Path(report_root_pattern.format(machine=machine)).expanduser().resolve()
        machines.append(_machine_snapshot(machine, plan_file, report_root))
    return machines


def print_snapshot(plan_dir: Path, report_root_pattern: str) -> None:
    machines = snapshot(plan_dir, report_root_pattern)
    total_assigned = sum(item["assigned"] for item in machines)
    total_done = sum(item["done"] for item in machines)
    total_rewrite = sum(item["rewrite"] for item in machines)
    total_rerun = sum(item["rerun"] for item in machines)
    total_started = sum(item["started"] for item in machines)
    total_pending = sum(item["pending"] for item in machines)

    print(f"plan_dir : {plan_dir}")
    print(f"pattern  : {report_root_pattern}")
    print(
        f"overall  : assigned={total_assigned} done={total_done} rewrite={total_rewrite} "
        f"rerun={total_rerun} started={total_started} pending={total_pending}"
    )
    print()

    for item in machines:
        pct = 0.0 if item["assigned"] <= 0 else (100.0 * item["done"] / item["assigned"])
        print(
            f"[{item['machine']}] progress={pct:6.2f}% done={item['done']}/{item['assigned']} "
            f"rewrite={item['rewrite']} rerun={item['rerun']} started={item['started']} pending={item['pending']}"
        )
        print(f"  report_root: {item['report_root']}")
        if item["active"]:
            active_text = " ".join(f"{name}({state})" for name, state in item["active"])
            print(f"  active: {active_text}")
        if item["logs"]:
            for log_name, last_line in item["logs"][-4:]:
                print(f"  {log_name}: {last_line}")
        print()


def main() -> None:
    args = build_parser().parse_args()
    plan_dir = Path(args.plan_dir).expanduser().resolve()
    if not plan_dir.is_dir():
        raise FileNotFoundError(f"Plan dir not found: {plan_dir}")

    if args.once:
        print_snapshot(plan_dir, args.report_root_pattern)
        return

    while True:
        print("\033[2J\033[H", end="")
        print_snapshot(plan_dir, args.report_root_pattern)
        sys.stdout.flush()
        time.sleep(float(args.refresh_sec))


if __name__ == "__main__":
    main()
