#!/usr/bin/env python3
"""Analyze a batch run produced by the current wave scheduler."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_STAGES = ["detect_track", "motion", "slam", "infiller"]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def summarize_tasks(tasks: dict, stages: list[str]):
    completed = []
    failed = []
    partial = []
    stage_counts = {stage: Counter() for stage in stages}

    for video_path, task in tasks.items():
        video_name = task.get("video_name") or Path(video_path).name
        stage_status = task.get("stage_status", {})
        for stage in stages:
            stage_counts[stage][stage_status.get(stage, "missing")] += 1

        all_completed = all(stage_status.get(stage) == "completed" for stage in stages)
        failed_stages = [stage for stage in stages if stage_status.get(stage) == "failed"]
        running = [stage for stage in stages if stage_status.get(stage) == "running"]
        pending = [stage for stage in stages if stage_status.get(stage) == "pending"]

        if all_completed:
            completed.append(video_name)
        elif failed_stages:
            failed.append((video_name, failed_stages, task.get("retry_count", {})))
        else:
            partial.append((video_name, running, pending))

    return completed, failed, partial, stage_counts


def summarize_events(events_path: Path):
    counts = Counter()
    wave_failures = defaultdict(list)
    stage_failures = Counter()
    stage_success = Counter()

    if not events_path.exists():
        return counts, wave_failures, stage_failures, stage_success

    for event in iter_jsonl(events_path):
        event_type = event.get("event", "unknown")
        counts[event_type] += 1

        if event_type == "wave_retry":
            wave_failures[event.get("stage", "unknown")].append(event.get("failed", 0))
        elif event_type == "stage_failure":
            stage_failures[event.get("stage", "unknown")] += 1
        elif event_type == "stage_success":
            stage_success[event.get("stage", "unknown")] += 1

    return counts, wave_failures, stage_failures, stage_success


def analyze_run(run_dir: Path) -> int:
    status_file = run_dir / "status.json"
    events_file = run_dir / "events.jsonl"

    if not status_file.exists():
        print(f"Error: {status_file} not found")
        return 1

    status = load_json(status_file)
    tasks = status.get("tasks", {})
    total_videos = len(tasks)
    stages = list(status.get("stages") or DEFAULT_STAGES)

    completed, failed, partial, stage_counts = summarize_tasks(tasks, stages)
    event_counts, wave_failures, stage_failures, stage_success = summarize_events(events_file)

    print("=" * 60)
    print(f"Batch Run Analysis: {run_dir.name}")
    print("=" * 60)
    print()
    print(f"Run dir: {run_dir}")
    print(f"Total videos: {total_videos}")
    if total_videos > 0:
        print(f"Completed: {len(completed)} ({len(completed) / total_videos * 100:.1f}%)")
        print(f"Failed: {len(failed)} ({len(failed) / total_videos * 100:.1f}%)")
        print(f"Partial: {len(partial)} ({len(partial) / total_videos * 100:.1f}%)")
    else:
        print("Completed: 0")
        print("Failed: 0")
        print("Partial: 0")
    print()

    print("Per-stage status:")
    for stage in stages:
        counts = stage_counts[stage]
        pieces = [f"{name}={counts[name]}" for name in ("completed", "failed", "running", "pending") if counts[name] > 0]
        if counts["missing"] > 0:
            pieces.append(f"missing={counts['missing']}")
        print(f"  {stage:15s}: " + (", ".join(pieces) if pieces else "no entries"))
    print()

    if failed:
        print("Failed videos:")
        for video_name, failed_stages, retry_count in failed:
            retries = ", ".join(f"{stage}={retry_count.get(stage, 0)}" for stage in failed_stages)
            print(f"  {video_name}: stages={','.join(failed_stages)} retries=({retries})")
        print()

    if partial:
        print("Partial videos:")
        for video_name, running, pending in partial:
            fragments = []
            if running:
                fragments.append("running=" + ",".join(running))
            if pending:
                fragments.append("pending=" + ",".join(pending))
            print(f"  {video_name}: " + ", ".join(fragments))
        print()

    if event_counts:
        print("Event counts:")
        for event_name in sorted(event_counts):
            print(f"  {event_name:20s}: {event_counts[event_name]}")
        print()

    if stage_success or stage_failures:
        print("Scheduler stage results:")
        for stage in stages:
            print(
                f"  {stage:15s}: success={stage_success.get(stage, 0)} "
                f"failure={stage_failures.get(stage, 0)}"
            )
        print()

    if wave_failures:
        print("Wave retries:")
        for stage in stages:
            failed_counts = wave_failures.get(stage)
            if not failed_counts:
                continue
            joined = ", ".join(str(value) for value in failed_counts)
            print(f"  {stage:15s}: retry_failed_counts=[{joined}]")
        print()

    print("Recommendations:")
    if failed:
        print(f"  - Resume this run with: python scripts/batch_infer.py --run_dir {run_dir} --resume ...")
        print("  - Inspect failures in status.json and events.jsonl before retrying.")
    elif partial:
        print(f"  - Resume this run with: python scripts/batch_infer.py --run_dir {run_dir} --resume ...")
    else:
        print("  - No action needed; all tracked stages are completed.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Analyze batch inference run results")
    parser.add_argument("run_dir", type=str, help="Path to batch run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Directory not found: {run_dir}")
        return 1

    return analyze_run(run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
