from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def parse_iso8601(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def format_age(ts: datetime | None) -> str:
    if ts is None:
        return "-"
    now = datetime.now(timezone.utc)
    delta = now - ts.astimezone(timezone.utc)
    total = int(max(0, delta.total_seconds()))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m"
    if total < 86400:
        return f"{total // 3600}h"
    return f"{total // 86400}d"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _status_backup_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.bak")


def _init_recovered_task(video_path: str, stages: list[str]) -> dict[str, Any]:
    return {
        "video_path": video_path,
        "video_name": Path(video_path).stem,
        "stage_status": {stage: "pending" for stage in stages},
        "retry_count": {stage: 0 for stage in stages},
        "start_time": None,
        "end_time": None,
    }


def _build_status_payload_from_events(
    events_path: Path,
    *,
    run_dir: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    stage_order: list[str] = []
    stage_seen = set()
    malformed_lines = 0
    parsed_lines = 0
    recovered_results = 0
    last_error = None

    with events_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                malformed_lines += 1
                last_error = f"line {line_no}: {error}"
                continue
            parsed_lines += 1
            event = payload.get("event")
            if event not in {"stage_success", "stage_failure"}:
                continue
            video_path = payload.get("video")
            stage = payload.get("stage")
            if not video_path or not stage:
                continue
            if stage not in stage_seen:
                stage_order.append(stage)
                stage_seen.add(stage)
                for task in tasks.values():
                    task["stage_status"].setdefault(stage, "pending")
                    task["retry_count"].setdefault(stage, 0)
            task = tasks.setdefault(video_path, _init_recovered_task(video_path, stage_order))
            task["stage_status"][stage] = "completed" if event == "stage_success" else "failed"
            task["retry_count"].setdefault(stage, 0)
            recovered_results += 1

    recovered = {
        "run_dir": str(run_dir or events_path.parent),
        "gpus": [],
        "stages": stage_order,
        "tasks": tasks,
    }
    meta = {
        "source": "events",
        "events_path": str(events_path),
        "parsed_lines": parsed_lines,
        "malformed_lines": malformed_lines,
        "recovered_results": recovered_results,
        "last_malformed_error": last_error,
    }
    return recovered, meta


def _load_status_payload_with_fallback(status_path: Path, *, events_path: Path | None = None) -> tuple[dict | None, dict[str, Any]]:
    backup_path = _status_backup_path(status_path)
    load_errors = []

    for candidate, source in ((status_path, "status"), (backup_path, "backup")):
        if not candidate.exists():
            continue
        try:
            return _read_json(candidate), {"source": source, "path": str(candidate), "load_errors": load_errors}
        except (OSError, json.JSONDecodeError, ValueError) as error:
            load_errors.append(f"{source}:{candidate}: {error}")

    if events_path is not None and events_path.exists():
        payload, meta = _build_status_payload_from_events(events_path, run_dir=str(status_path.parent))
        meta["load_errors"] = load_errors
        return payload, meta

    if not status_path.exists() and not backup_path.exists():
        return None, {"source": "missing", "path": str(status_path), "load_errors": load_errors}

    raise RuntimeError(
        f"Failed to load batch status from {status_path}"
        + (f" (backup: {backup_path})" if backup_path != status_path else "")
        + (f". Errors: {' | '.join(load_errors)}" if load_errors else "")
    )


def _tail_last_event_time(events_path: Path) -> datetime | None:
    if not events_path.exists():
        return None
    try:
        with events_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            block = 4096
            data = b""
            pos = size
            while pos > 0:
                read_size = min(block, pos)
                pos -= read_size
                handle.seek(pos)
                data = handle.read(read_size) + data
                lines = data.splitlines()
                if len(lines) >= 2 or pos == 0:
                    for line in reversed(lines):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line.decode("utf-8"))
                        except Exception:
                            continue
                        return parse_iso8601(payload.get("time"))
    except OSError:
        return None
    return None


def _load_run_summary(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return {}, []
    try:
        return _read_json(summary_path), []
    except Exception as error:
        return {}, [f"run_summary:{summary_path}: {error}"]


def _compute_health(
    *,
    total: int,
    completed: int,
    failed: int,
    partial: int,
    running: int,
    pending: int,
    last_event_time: datetime | None,
    stall_seconds: int,
) -> tuple[str, str | None]:
    if total <= 0:
        return "empty", None
    if completed == total and failed == 0:
        return "done", None
    if failed == total and completed == 0 and partial == 0:
        return "failed", None

    age_seconds = None
    if last_event_time is not None:
        age_seconds = max(
            0.0,
            (datetime.now(timezone.utc) - last_event_time.astimezone(timezone.utc)).total_seconds(),
        )

    active = partial > 0 or running > 0 or pending > 0
    if active and age_seconds is not None and age_seconds > max(1, int(stall_seconds)):
        return "stalled", f"no new events for {int(age_seconds)}s"
    if active:
        return "running", None
    if failed > 0 and completed > 0:
        return "partial", None
    if failed > 0:
        return "failed", None
    return "partial", None


def discover_run_dirs(
    log_root: Path,
    run_tags: list[str] | None,
    pattern: str,
    limit: int | None,
) -> list[Path]:
    if run_tags:
        run_dirs = [log_root / tag for tag in run_tags]
    else:
        run_dirs = sorted(path for path in log_root.glob(pattern) if path.is_dir())
    if limit is not None:
        run_dirs = run_dirs[-int(limit) :]
    return [path for path in run_dirs if path.exists()]


def compute_run_summary(run_dir: Path, *, stall_seconds: int = 1800) -> dict[str, Any]:
    run_summary, summary_errors = _load_run_summary(run_dir)
    status_path = run_dir / "status.json"
    events_path = run_dir / "events.jsonl"

    try:
        status_payload, status_meta = _load_status_payload_with_fallback(status_path, events_path=events_path)
    except Exception as error:
        return {
            "run_tag": run_dir.name,
            "run_dir": str(run_dir),
            "config": Path(run_summary.get("config", "")).name if run_summary.get("config") else "-",
            "health": "error",
            "health_reason": str(error),
            "status_source": "error",
            "status_errors": summary_errors + [str(error)],
            "total": 0,
            "completed": 0,
            "failed": 0,
            "partial": 0,
            "running": 0,
            "pending": 0,
            "stages": list(run_summary.get("expanded_internal_stages") or run_summary.get("stages") or []),
            "stage_completed": {},
            "stage_counts": {},
            "last_event_time": _tail_last_event_time(events_path),
            "last_event_age": format_age(_tail_last_event_time(events_path)),
            "active_manifest_path": run_summary.get("active_manifest_path"),
            "annotation_root": run_summary.get("annotation_root"),
        }

    tasks = {} if status_payload is None else (status_payload.get("tasks") or {})
    stages = list(status_payload.get("stages") or [])
    if not stages:
        stages = list(run_summary.get("expanded_internal_stages") or run_summary.get("stages") or [])

    total = len(tasks)
    completed = 0
    failed = 0
    partial = 0
    pending = 0
    running = 0
    stage_completed: dict[str, int] = {stage: 0 for stage in stages}
    stage_counts: dict[str, dict[str, int]] = {
        stage: {"completed": 0, "failed": 0, "running": 0, "pending": 0} for stage in stages
    }

    for task in tasks.values():
        stage_status = task.get("stage_status") or {}
        if stages and all(stage_status.get(stage) == "completed" for stage in stages):
            completed += 1
        elif any(stage_status.get(stage) == "failed" for stage in stages):
            failed += 1
        else:
            partial += 1

        if any(stage_status.get(stage) == "running" for stage in stages):
            running += 1
        if any(stage_status.get(stage, "pending") == "pending" for stage in stages):
            pending += 1

        for stage in stages:
            state = stage_status.get(stage, "pending")
            if state == "completed":
                stage_completed[stage] += 1
            if state not in stage_counts[stage]:
                state = "pending"
            stage_counts[stage][state] += 1

    last_event_time = _tail_last_event_time(events_path)
    health, health_reason = _compute_health(
        total=total,
        completed=completed,
        failed=failed,
        partial=partial,
        running=running,
        pending=pending,
        last_event_time=last_event_time,
        stall_seconds=stall_seconds,
    )
    config_name = Path(run_summary.get("config", "")).name if run_summary.get("config") else "-"
    status_errors = list(summary_errors)
    status_errors.extend(status_meta.get("load_errors") or [])

    return {
        "run_tag": run_dir.name,
        "run_dir": str(run_dir),
        "config": config_name,
        "health": health,
        "health_reason": health_reason,
        "status_source": status_meta.get("source", "missing"),
        "status_errors": status_errors,
        "total": total,
        "completed": completed,
        "failed": failed,
        "partial": partial,
        "running": running,
        "pending": pending,
        "stages": stages,
        "stage_completed": stage_completed,
        "stage_counts": stage_counts,
        "last_event_time": last_event_time,
        "last_event_age": format_age(last_event_time),
        "active_manifest_path": run_summary.get("active_manifest_path"),
        "annotation_root": run_summary.get("annotation_root"),
        "requested_stage_tokens": list(run_summary.get("requested_stage_tokens") or []),
        "expanded_internal_stages": list(run_summary.get("expanded_internal_stages") or []),
    }


def summarize_runs(
    log_root: Path,
    *,
    run_tags: list[str] | None = None,
    pattern: str = "20*",
    limit: int | None = 12,
    stall_seconds: int = 1800,
) -> list[dict[str, Any]]:
    rows = [compute_run_summary(run_dir, stall_seconds=stall_seconds) for run_dir in discover_run_dirs(log_root, run_tags, pattern, limit)]
    rows.sort(key=lambda item: (item["last_event_time"] or datetime.fromtimestamp(0, timezone.utc)), reverse=True)
    return rows


def format_stage_progress(item: dict[str, Any], max_stages: int) -> str:
    total = max(1, int(item["total"]))
    parts = []
    for stage in item["stages"][:max_stages]:
        parts.append(f"{stage}={item['stage_completed'].get(stage, 0)}/{total}")
    return " ".join(parts) if parts else "-"
