#!/usr/bin/env python3
"""Audit BuildAI manifest clips for sparse DPVO slam exports and emit repair lists."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.clip_manifest import load_clip_manifest  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **_kwargs):
        return iterable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit manifest seq_folders for sparse DPVO hawor_slam_w_scale exports."
    )
    parser.add_argument("--descriptor_manifest", required=True, help="Input clip manifest JSONL")
    parser.add_argument(
        "--report_out",
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--repair_seq_folder_list_out",
        default=None,
        help="Optional text file with one seq_folder per line for repairable sparse exports.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Thread workers for per-clip filesystem checks.",
    )
    parser.add_argument(
        "--max_clips",
        type=int,
        default=None,
        help="Optional clip cap for smoke testing.",
    )
    return parser


def _load_slam_backend(seq_folder: Path) -> str | None:
    path = seq_folder / "SLAM" / "slam_backend.txt"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8").strip().lower() or None
    except Exception:
        return None


def _inspect_record(record) -> dict:
    seq_folder = Path(record.descriptor.seq_folder)
    clip_id = record.clip_id
    expected_frames = int(getattr(record.descriptor, "frame_count", 0) or 0)
    slam_dir = seq_folder / "SLAM"

    result = {
        "clip_id": clip_id,
        "seq_folder": str(seq_folder),
        "expected_frames": expected_frames,
        "status": "ok",
        "slam_backend": _load_slam_backend(seq_folder),
        "repairable": False,
        "needs_repair": False,
        "slam_files": [],
        "issues": [],
    }

    if not slam_dir.is_dir():
        result["status"] = "missing_slam_dir"
        result["issues"].append("missing_slam_dir")
        return result

    export_paths = sorted(slam_dir.glob("hawor_slam_w_scale_*.npz"))
    if not export_paths:
        result["status"] = "missing_export"
        result["issues"].append("missing_export")
        return result

    for export_path in export_paths:
        file_info = {
            "name": export_path.name,
            "traj_len": None,
            "tstamp_len": None,
            "dense_by_traj_len": None,
            "repairable": False,
            "dpvo_raw_path": None,
        }
        try:
            with np.load(export_path, allow_pickle=False) as payload:
                traj = np.asarray(payload["traj"])
                tstamp = np.asarray(payload["tstamp"]).reshape(-1) if "tstamp" in payload.files else None
        except Exception as error:
            file_info["error"] = str(error)
            result["status"] = "invalid_export"
            result["issues"].append(f"invalid_export:{export_path.name}")
            result["slam_files"].append(file_info)
            continue

        traj_len = int(traj.shape[0])
        tstamp_len = None if tstamp is None else int(tstamp.shape[0])
        dense_by_traj_len = expected_frames > 0 and traj_len == expected_frames
        suffix = export_path.stem[len("hawor_slam_w_scale_") :]
        dpvo_raw_path = slam_dir / f"dpvo_raw_{suffix}.npz"
        repairable = bool(dpvo_raw_path.is_file())

        file_info.update(
            {
                "traj_len": traj_len,
                "tstamp_len": tstamp_len,
                "dense_by_traj_len": bool(dense_by_traj_len),
                "repairable": repairable,
                "dpvo_raw_path": str(dpvo_raw_path) if repairable else None,
            }
        )
        result["slam_files"].append(file_info)

        if expected_frames > 0 and traj_len != expected_frames:
            result["needs_repair"] = True
            result["repairable"] = result["repairable"] or repairable
            result["issues"].append(f"sparse_export:{export_path.name}")

    if result["needs_repair"]:
        result["status"] = "needs_repair"
    elif result["status"] == "ok" and result["issues"]:
        result["status"] = "warning"
    return result


def _summarize(results: list[dict], *, manifest_path: Path) -> dict:
    status_counts: dict[str, int] = {}
    repairable_seq_folders: list[str] = []
    sparse_examples: list[dict] = []
    invalid_examples: list[dict] = []

    for item in results:
        status = str(item["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if item["needs_repair"] and item["repairable"]:
            repairable_seq_folders.append(item["seq_folder"])
        if item["needs_repair"] and len(sparse_examples) < 64:
            sparse_examples.append(item)
        if item["status"] == "invalid_export" and len(invalid_examples) < 32:
            invalid_examples.append(item)

    unique_repairable = sorted(set(repairable_seq_folders))
    return {
        "descriptor_manifest": str(manifest_path.resolve()),
        "total_clips": len(results),
        "status_counts": status_counts,
        "needs_repair_count": int(sum(1 for item in results if item["needs_repair"])),
        "repairable_count": len(unique_repairable),
        "repairable_seq_folders": unique_repairable,
        "sparse_examples": sparse_examples,
        "invalid_examples": invalid_examples,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    manifest_path = Path(args.descriptor_manifest).expanduser().resolve()
    records = load_clip_manifest(manifest_path)
    if args.max_clips is not None and int(args.max_clips) > 0:
        records = records[: int(args.max_clips)]

    def _iter_results():
        if args.workers == 1:
            for record in records:
                yield _inspect_record(record)
            return
        with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
            yield from executor.map(_inspect_record, records, chunksize=64)

    results = list(
        tqdm(
            _iter_results(),
            total=len(records),
            desc="Audit dense SLAM",
            unit="clip",
            dynamic_ncols=True,
        )
    )
    report = _summarize(results, manifest_path=manifest_path)

    if args.repair_seq_folder_list_out:
        out_path = Path(args.repair_seq_folder_list_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "".join(f"{path}\n" for path in report["repairable_seq_folders"]),
            encoding="utf-8",
        )

    if args.report_out:
        report_path = Path(args.report_out).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
