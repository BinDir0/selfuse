#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

SHARD_RE = re.compile(r"^shard-(\d{6})$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor BuildAI shard rerun/rewrite progress.")
    parser.add_argument("--source_shard_dir", default="/share_data/guantianrui/datasets/BuildAI-100K-part1-0401.with_instruction_v2")
    parser.add_argument("--report_root", default="/DATA/guantianrui/buildai_v2_rewrite_reports_full")
    parser.add_argument("--machine_count", type=int, default=5)
    parser.add_argument("--refresh_sec", type=float, default=20.0)
    parser.add_argument("--once", action="store_true")
    return parser


def _shard_num_from_name(name: str) -> int | None:
    m = SHARD_RE.match(name)
    return None if m is None else int(m.group(1))


def _collect_shard_dirs(report_root: Path) -> dict[int, Path]:
    shard_dirs: dict[int, Path] = {}
    for path in report_root.glob("shards_*_*/per_shard/shard-*"):
        if not path.is_dir():
            continue
        num = _shard_num_from_name(path.name)
        if num is None:
            continue
        shard_dirs[num] = path
    return shard_dirs


def _load_scan_report(shard_dir: Path) -> dict | None:
    path = shard_dir / "scan_report.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _clip_progress(shard_dir: Path, global_seq_progress_dir: Path) -> tuple[int, int]:
    scan_report = _load_scan_report(shard_dir)
    if scan_report is None:
        return 0, 0
    clip_file = Path(scan_report["clip_id_list"])
    if not clip_file.is_file():
        return 0, int(scan_report.get("clip_count", 0))
    clip_ids = [line.strip() for line in clip_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = 0
    for clip_id in clip_ids:
        if (global_seq_progress_dir / f"{clip_id}.success.json").is_file():
            done += 1
    return done, len(clip_ids)


def _state_of(shard_dir: Path) -> str:
    if (shard_dir / "shard_summary.json").is_file():
        return "done"
    if (shard_dir / "rewrite_report.json").is_file():
        return "rewrite"
    if (shard_dir / "rerun_report.json").is_file() or (shard_dir / "scan_report.json").is_file():
        return "rerun"
    return "started"


def snapshot(source_shard_dir: Path, report_root: Path, machine_count: int):
    source_shards = sorted(path for path in source_shard_dir.glob("shard-*.tar"))
    total_shards = len(source_shards)
    if total_shards <= 0:
        raise RuntimeError(f"No shards found under {source_shard_dir}")
    shard_dirs = _collect_shard_dirs(report_root)
    global_seq_progress_dir = report_root / "_seq_progress"

    overall = {
        "total_shards": total_shards,
        "done": 0,
        "rewrite": 0,
        "rerun": 0,
        "started": 0,
        "pending": 0,
        "approx_done_equiv": 0.0,
    }
    machines = []

    for machine_id in range(machine_count):
        machine_start = machine_id * total_shards // machine_count
        machine_end = (machine_id + 1) * total_shards // machine_count
        m = {
            "tag": f"machine{machine_id + 1:02d}",
            "start": machine_start,
            "end": machine_end,
            "total_shards": machine_end - machine_start,
            "done": 0,
            "rewrite": 0,
            "rerun": 0,
            "started": 0,
            "pending": 0,
            "approx_done_equiv": 0.0,
            "active": [],
        }
        for shard_idx in range(machine_start, machine_end):
            shard_dir = shard_dirs.get(shard_idx)
            if shard_dir is None:
                m["pending"] += 1
                overall["pending"] += 1
                continue
            state = _state_of(shard_dir)
            m[state] += 1
            overall[state] += 1
            if state == "done":
                m["approx_done_equiv"] += 1.0
                overall["approx_done_equiv"] += 1.0
            else:
                clip_done, clip_total = _clip_progress(shard_dir, global_seq_progress_dir)
                frac = 0.0 if clip_total <= 0 else float(clip_done / clip_total)
                m["approx_done_equiv"] += frac
                overall["approx_done_equiv"] += frac
                m["active"].append({
                    "shard": f"shard-{shard_idx:06d}",
                    "state": state,
                    "clip_done": clip_done,
                    "clip_total": clip_total,
                    "frac": frac,
                })
        machines.append(m)
    return machines, overall


def print_once(source_shard_dir: Path, report_root: Path, machine_count: int) -> None:
    machines, overall = snapshot(source_shard_dir, report_root, machine_count)
    print(f"source_shard_dir: {source_shard_dir}")
    print(f"report_root     : {report_root}")
    print(f"total_shards    : {overall['total_shards']}")
    print(f"machine_count   : {machine_count}")
    print()
    for m in machines:
        pct = 0.0 if m["total_shards"] == 0 else 100.0 * m["approx_done_equiv"] / m["total_shards"]
        print(
            f"[{m['tag']}] range=[{m['start']}, {m['end']}) approx_progress={pct:6.2f}% "
            f"done={m['done']} rewrite={m['rewrite']} rerun={m['rerun']} started={m['started']} pending={m['pending']}"
        )
        for item in m["active"][:8]:
            print(
                f"  {item['shard']} state={item['state']} clip_progress={item['clip_done']}/{item['clip_total']} "
                f"({item['frac'] * 100.0:5.1f}%)"
            )
        if len(m["active"]) > 8:
            print(f"  ... ({len(m['active']) - 8} more active shards)")
    print()
    pct = 0.0 if overall["total_shards"] == 0 else 100.0 * overall["approx_done_equiv"] / overall["total_shards"]
    print(
        f"overall approx_progress={pct:6.2f}% done={overall['done']} rewrite={overall['rewrite']} "
        f"rerun={overall['rerun']} started={overall['started']} pending={overall['pending']}"
    )


def main() -> None:
    args = build_parser().parse_args()
    source_shard_dir = Path(args.source_shard_dir).expanduser().resolve()
    report_root = Path(args.report_root).expanduser().resolve()

    if args.once:
        print_once(source_shard_dir, report_root, int(args.machine_count))
        return

    try:
        from tqdm import tqdm
    except Exception:
        while True:
            print_once(source_shard_dir, report_root, int(args.machine_count))
            sys.stdout.flush()
            time.sleep(float(args.refresh_sec))

    machines, overall = snapshot(source_shard_dir, report_root, int(args.machine_count))
    pbar = tqdm(total=overall["total_shards"], desc="BuildAI Shard Progress", dynamic_ncols=True)
    pbar.n = int(round(overall["approx_done_equiv"]))
    pbar.set_postfix(done=overall["done"], rewrite=overall["rewrite"], rerun=overall["rerun"], pending=overall["pending"])
    pbar.refresh()

    while True:
        time.sleep(float(args.refresh_sec))
        machines, overall = snapshot(source_shard_dir, report_root, int(args.machine_count))
        pbar.n = int(round(overall["approx_done_equiv"]))
        pbar.set_postfix(done=overall["done"], rewrite=overall["rewrite"], rerun=overall["rerun"], pending=overall["pending"])
        pbar.refresh()

        lines = []
        for m in machines:
            pct = 0.0 if m["total_shards"] == 0 else 100.0 * m["approx_done_equiv"] / m["total_shards"]
            lines.append(
                f"{m['tag']} {pct:6.2f}% done={m['done']} rewrite={m['rewrite']} rerun={m['rerun']} started={m['started']} pending={m['pending']}"
            )
            for item in m["active"][:4]:
                lines.append(
                    f"  {item['shard']} {item['state']} clip={item['clip_done']}/{item['clip_total']} ({item['frac'] * 100.0:5.1f}%)"
                )
        tqdm.write("\n".join(lines))

        if overall["done"] >= overall["total_shards"]:
            pbar.close()
            break


if __name__ == "__main__":
    main()
