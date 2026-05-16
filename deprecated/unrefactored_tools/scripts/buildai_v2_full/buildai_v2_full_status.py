#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

SHARD_NAME_RE = re.compile(r"^shard-(\d{6})$")


@dataclass
class ShardRecord:
    shard_idx: int
    shard_name: str
    shard_dir: Path
    state: str
    output_path: Path | None
    has_scan_report: bool
    has_rerun_report: bool
    has_rewrite_report: bool
    has_shard_summary: bool


def shard_name_from_index(shard_idx: int) -> str:
    return f"shard-{int(shard_idx):06d}"


def shard_tar_name_from_index(shard_idx: int) -> str:
    return f"{shard_name_from_index(shard_idx)}.tar"


def parse_shard_index_from_name(name: str) -> int | None:
    match = SHARD_NAME_RE.match(name)
    if match is None:
        return None
    return int(match.group(1))


def list_source_shard_indices(source_shard_dir: Path) -> list[int]:
    shard_indices = []
    for path in sorted(source_shard_dir.glob("shard-*.tar")):
        idx = parse_shard_index_from_name(path.stem)
        if idx is not None:
            shard_indices.append(idx)
    return shard_indices


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_output_path(shard_dir: Path, shard_name: str) -> Path | None:
    summary = _load_json(shard_dir / "shard_summary.json")
    if summary is not None:
        batch_output_dir = summary.get("batch_output_dir")
        if batch_output_dir:
            return Path(batch_output_dir) / shard_name

    scan_report = _load_json(shard_dir / "scan_report.json")
    if scan_report is not None:
        batch_output_dir = scan_report.get("batch_output_dir")
        if batch_output_dir:
            return Path(batch_output_dir) / shard_name

    return None


def _state_rank(state: str) -> int:
    order = {
        "done": 4,
        "rewrite": 3,
        "rerun": 2,
        "started": 1,
        "pending": 0,
    }
    return order.get(state, -1)


def _record_from_dir(shard_dir: Path) -> ShardRecord | None:
    shard_idx = parse_shard_index_from_name(shard_dir.name)
    if shard_idx is None:
        return None

    output_path = _resolve_output_path(shard_dir, f"{shard_dir.name}.tar")
    has_scan_report = (shard_dir / "scan_report.json").is_file()
    has_rerun_report = (shard_dir / "rerun_report.json").is_file()
    has_rewrite_report = (shard_dir / "rewrite_report.json").is_file()
    has_shard_summary = (shard_dir / "shard_summary.json").is_file()
    output_exists = output_path is not None and output_path.is_file()

    if output_exists or has_shard_summary:
        state = "done"
    elif has_rewrite_report:
        state = "rewrite"
    elif has_rerun_report or has_scan_report:
        state = "rerun"
    else:
        state = "started"

    return ShardRecord(
        shard_idx=shard_idx,
        shard_name=shard_dir.name,
        shard_dir=shard_dir,
        state=state,
        output_path=output_path,
        has_scan_report=has_scan_report,
        has_rerun_report=has_rerun_report,
        has_rewrite_report=has_rewrite_report,
        has_shard_summary=has_shard_summary,
    )


def collect_best_shard_records(report_root: Path) -> dict[int, ShardRecord]:
    best: dict[int, ShardRecord] = {}
    for shard_dir in report_root.glob("shards_*_*/per_shard/shard-*"):
        if not shard_dir.is_dir():
            continue
        record = _record_from_dir(shard_dir)
        if record is None:
            continue
        existing = best.get(record.shard_idx)
        if existing is None:
            best[record.shard_idx] = record
            continue
        if _state_rank(record.state) > _state_rank(existing.state):
            best[record.shard_idx] = record
            continue
        if _state_rank(record.state) == _state_rank(existing.state):
            if record.shard_dir.stat().st_mtime > existing.shard_dir.stat().st_mtime:
                best[record.shard_idx] = record
    return best


def classify_shards(source_shard_dir: Path, report_root: Path) -> dict:
    source_indices = list_source_shard_indices(source_shard_dir)
    records = collect_best_shard_records(report_root)

    done = []
    active = []
    pending = []
    for shard_idx in source_indices:
        record = records.get(shard_idx)
        if record is None:
            pending.append(shard_idx)
        elif record.state == "done":
            done.append(shard_idx)
        else:
            active.append(shard_idx)

    return {
        "source_indices": source_indices,
        "records": records,
        "done": done,
        "active": active,
        "pending": pending,
        "not_done": [idx for idx in source_indices if idx not in set(done)],
    }
