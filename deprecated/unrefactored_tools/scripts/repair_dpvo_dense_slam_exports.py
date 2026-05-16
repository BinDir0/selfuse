#!/usr/bin/env python3
"""Repair DPVO-based hawor_slam_w_scale exports using cached dense raw trajectories."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **_kwargs):
        return iterable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replace sparse exported SLAM traj with dense traj from cached dpvo_raw_*.npz"
    )
    parser.add_argument(
        "--seq_folder",
        action="append",
        default=[],
        help="Sequence folder to repair. Can be passed multiple times.",
    )
    parser.add_argument(
        "--seq_folder_list",
        default=None,
        help="Optional text file with one seq_folder per line.",
    )
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Report planned repairs without writing files.",
    )
    parser.add_argument(
        "--report_out",
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(64, (os.cpu_count() or 1) * 4)),
        help="Thread workers for parallel seq_folder repair.",
    )
    parser.add_argument(
        "--skip_content_compare",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip expensive elementwise traj comparison. When enabled, the script decides whether to rewrite "
            "primarily from traj length mismatch, which is the common sparse->dense repair case."
        ),
    )
    return parser


def _collect_seq_folders(args: argparse.Namespace) -> list[Path]:
    seq_folders: list[Path] = []
    seen: set[Path] = set()

    def _add(path_str: str) -> None:
        path = Path(path_str).resolve()
        if path in seen:
            return
        seen.add(path)
        seq_folders.append(path)

    for item in args.seq_folder:
        if item:
            _add(item)

    if args.seq_folder_list:
        for line in Path(args.seq_folder_list).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                _add(line)

    if not seq_folders:
        raise SystemExit("No seq_folder provided. Use --seq_folder or --seq_folder_list.")
    return seq_folders


def _rewrite_npz(export_path: Path, payload: dict[str, np.ndarray | float]) -> None:
    tmp_path = export_path.with_suffix(export_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez(handle, **payload)
    tmp_path.replace(export_path)


def repair_seq_folder(seq_folder: Path, *, dry_run: bool, skip_content_compare: bool) -> dict:
    slam_dir = seq_folder / "SLAM"
    if not slam_dir.is_dir():
        return {"seq_folder": str(seq_folder), "status": "missing_slam_dir"}

    repaired = []
    skipped = []
    for raw_path in sorted(slam_dir.glob("dpvo_raw_*.npz")):
        suffix = raw_path.stem[len("dpvo_raw_") :]
        export_path = slam_dir / f"hawor_slam_w_scale_{suffix}.npz"
        if not export_path.is_file():
            skipped.append(
                {
                    "suffix": suffix,
                    "reason": "missing_export",
                    "raw_path": str(raw_path),
                    "export_path": str(export_path),
                }
            )
            continue

        with np.load(raw_path, allow_pickle=False) as raw_data:
            dense_traj = np.asarray(raw_data["traj"], dtype=np.float32)
        with np.load(export_path, allow_pickle=False) as export_data:
            payload = {key: export_data[key] for key in export_data.files if key != "traj"}
            old_traj = np.asarray(export_data["traj"], dtype=np.float32)

        if skip_content_compare:
            changed = bool(old_traj.shape != dense_traj.shape)
        else:
            changed = bool(old_traj.shape != dense_traj.shape or not np.array_equal(old_traj, dense_traj))
        payload["traj"] = dense_traj
        if changed and not dry_run:
            _rewrite_npz(export_path, payload)

        repaired.append(
            {
                "suffix": suffix,
                "changed": changed,
                "traj_old_frames": int(old_traj.shape[0]),
                "traj_new_frames": int(dense_traj.shape[0]),
                "raw_path": str(raw_path),
                "export_path": str(export_path),
            }
        )

    status = "ok" if repaired else "no_dpvo_raw"
    return {
        "seq_folder": str(seq_folder),
        "status": status,
        "repaired": repaired,
        "skipped": skipped,
    }


def _summarize_report(report: list[dict]) -> dict:
    status_counts: dict[str, int] = {}
    repaired_exports = 0
    changed_exports = 0
    skipped_exports = 0

    for item in report:
        status = str(item.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        repaired = item.get("repaired") or []
        skipped = item.get("skipped") or []
        repaired_exports += len(repaired)
        changed_exports += sum(1 for entry in repaired if entry.get("changed"))
        skipped_exports += len(skipped)

    return {
        "seq_folders_total": len(report),
        "status_counts": status_counts,
        "repaired_exports": int(repaired_exports),
        "changed_exports": int(changed_exports),
        "skipped_exports": int(skipped_exports),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    seq_folders = _collect_seq_folders(args)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    print(
        f"Repairing DPVO dense SLAM exports: seq_folders={len(seq_folders)} workers={int(args.workers)} "
        f"dry_run={bool(args.dry_run)}",
        flush=True,
    )

    def _iter_report():
        if int(args.workers) == 1:
            for seq_folder in seq_folders:
                yield repair_seq_folder(
                    seq_folder,
                    dry_run=bool(args.dry_run),
                    skip_content_compare=bool(args.skip_content_compare),
                )
            return
        with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
            yield from executor.map(
                lambda seq_folder: repair_seq_folder(
                    seq_folder,
                    dry_run=bool(args.dry_run),
                    skip_content_compare=bool(args.skip_content_compare),
                ),
                seq_folders,
                chunksize=64,
            )

    report = list(
        tqdm(
            _iter_report(),
            total=len(seq_folders),
            desc="Repair DPVO exports",
            unit="seq",
            dynamic_ncols=True,
        )
    )
    summary = _summarize_report(report)
    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"summary": summary, "items": report}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps({"summary": summary, "items": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
