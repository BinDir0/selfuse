#!/usr/bin/env python3
"""Create missing `.stage.done` markers for already-complete sequence folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional CLI nicety
    def tqdm(iterable, **_kwargs):
        return iterable


STAGES = ["detect_track", "motion", "slam", "infiller"]


def find_seq_folders(root_dir: Path, max_depth: int):
    seq_folders = []

    def walk(current: Path, depth: int):
        if depth > max_depth:
            return
        try:
            children = list(current.iterdir())
        except OSError:
            return

        if any(child.is_dir() and child.name.startswith("tracks_") for child in children):
            seq_folders.append(current)
            return

        for child in children:
            if child.is_dir() and not child.name.startswith("."):
                walk(child, depth + 1)

    walk(root_dir, 0)
    return seq_folders


def create_done_markers(root_dir: Path, stages: list[str], max_depth: int, dry_run: bool, verbose: bool):
    from lib.pipeline.stage_api import get_stage_done_marker, is_stage_complete

    seq_folders = find_seq_folders(root_dir, max_depth=max_depth)
    print(f"Found {len(seq_folders)} candidate sequence folder(s) under {root_dir}")

    stats = {
        stage: {"existing": 0, "checked": 0, "complete": 0, "created": 0}
        for stage in stages
    }

    for seq_folder in tqdm(seq_folders, desc="Sequence folders"):
        for stage in stages:
            done_marker = get_stage_done_marker(seq_folder, stage)
            if done_marker.exists():
                stats[stage]["existing"] += 1
                continue

            stats[stage]["checked"] += 1
            if not is_stage_complete(stage, seq_folder, fast_check=True):
                if verbose:
                    print(f"[incomplete] {seq_folder} :: {stage}")
                continue

            stats[stage]["complete"] += 1
            if dry_run:
                if verbose:
                    print(f"[dry-run] {done_marker}")
                continue

            done_marker.touch()
            stats[stage]["created"] += 1
            if verbose:
                print(f"[created] {done_marker}")

    print("\nSummary")
    print("=" * 60)
    for stage in stages:
        stage_stats = stats[stage]
        print(
            f"{stage:15s} existing={stage_stats['existing']:5d} "
            f"checked={stage_stats['checked']:5d} "
            f"complete={stage_stats['complete']:5d} "
            f"created={stage_stats['created']:5d}"
        )

    if dry_run:
        print("\nDry run only; no files were created.")


def main():
    parser = argparse.ArgumentParser(description="Create .done markers for already-complete sequence folders")
    parser.add_argument("root_dir", type=Path, help="Root directory containing sequence folders")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES), help="Stages to validate")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum recursive search depth")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be created without touching files")
    parser.add_argument("--verbose", action="store_true", help="Print per-folder details")
    args = parser.parse_args()

    if not args.root_dir.exists():
        raise SystemExit(f"Root directory not found: {args.root_dir}")

    create_done_markers(
        args.root_dir,
        stages=args.stages,
        max_depth=args.max_depth,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
