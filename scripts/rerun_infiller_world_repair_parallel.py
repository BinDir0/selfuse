#!/usr/bin/env python3
"""Run rerun_infiller_world_repair.py with multiple subprocess workers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RERUN_SCRIPT = PROJECT_ROOT / "scripts" / "rerun_infiller_world_repair.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch multiple infiller rerun workers with unified logs and a merged report."
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
        "--worker_count",
        type=int,
        default=16,
        help="Number of subprocess workers to launch.",
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated GPU ids assigned round-robin across workers. Use empty string for CPU-only workers.",
    )
    parser.add_argument(
        "--python_bin",
        default=sys.executable,
        help="Python executable used to invoke worker subprocesses.",
    )
    parser.add_argument(
        "--checkpoint",
        default="./weights/hawor/checkpoints/hawor.ckpt",
        help="Motion checkpoint path passed through to each worker.",
    )
    parser.add_argument(
        "--infiller_weight",
        default="./weights/hawor/checkpoints/infiller.pt",
        help="Infiller checkpoint path passed through to each worker.",
    )
    parser.add_argument(
        "--infiller_window_batch_size",
        type=int,
        default=64,
        help="Batch size for infiller windows.",
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=None,
        help="Optional cap for CPU math threads per worker.",
    )
    parser.add_argument(
        "--interop_threads",
        type=int,
        default=None,
        help="Optional cap for Torch inter-op CPU threads per worker.",
    )
    parser.add_argument(
        "--rebuild_cam_space_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rebuild cam_space cache before rerunning infiller.",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force rerun even if .infiller.done exists.",
    )
    parser.add_argument(
        "--backup_dir",
        default=None,
        help="Optional directory where existing world_space_res.pth files are copied before rerun.",
    )
    parser.add_argument(
        "--progress_dir",
        default=None,
        help="Optional directory with per-seq success markers for resume.",
    )
    parser.add_argument(
        "--report_out",
        required=True,
        help="Merged JSON report output path.",
    )
    parser.add_argument(
        "--worker_tmp_dir",
        default=None,
        help="Optional directory for per-worker seq lists and child reports.",
    )
    parser.add_argument(
        "--filter_completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip seq_folders that already have progress markers before sharding workers.",
    )
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print planned workers without executing them.",
    )
    return parser


def _append_bool_flag(cmd: list[str], flag: str, value: bool) -> None:
    cmd.append(flag if value else flag.replace("--", "--no-", 1))


def _collect_seq_folders(args: argparse.Namespace) -> list[Path]:
    seq_folders: list[Path] = []
    seen: set[Path] = set()

    def _add(path_str: str) -> None:
        path = Path(path_str).expanduser().resolve()
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


def _parse_gpu_list(raw: str) -> list[str]:
    parts = [part.strip() for part in str(raw).split(",")]
    cleaned = [part for part in parts if part != ""]
    return cleaned or [""]


def _progress_marker_path(progress_dir: Path, seq_folder: Path) -> Path:
    return progress_dir / f"{seq_folder.name}.success.json"


def _filter_completed_seq_folders(seq_folders: list[Path], progress_dir: Path | None) -> tuple[list[Path], int]:
    if progress_dir is None:
        return seq_folders, 0
    remaining: list[Path] = []
    skipped = 0
    for seq_folder in seq_folders:
        if _progress_marker_path(progress_dir, seq_folder).is_file():
            skipped += 1
            continue
        remaining.append(seq_folder)
    return remaining, skipped


def _bucketize_round_robin(seq_folders: list[Path], worker_count: int) -> list[list[Path]]:
    buckets: list[list[Path]] = [[] for _ in range(worker_count)]
    for idx, seq_folder in enumerate(seq_folders):
        buckets[idx % worker_count].append(seq_folder)
    return buckets


def _write_seq_list(path: Path, seq_folders: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{seq_folder}\n" for seq_folder in seq_folders), encoding="utf-8")


def _stream_worker_output(pipe, prefix: str) -> None:
    assert pipe is not None
    try:
        for line in pipe:
            print(f"{prefix} {line.rstrip()}", flush=True)
    finally:
        pipe.close()


def _load_child_report(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise RuntimeError(f"Unexpected child report payload in {path}")


def main() -> int:
    args = build_parser().parse_args()
    if args.worker_count < 1:
        raise SystemExit("--worker_count must be >= 1")

    seq_folders = _collect_seq_folders(args)
    progress_dir = Path(args.progress_dir).expanduser().resolve() if args.progress_dir else None
    report_out = Path(args.report_out).expanduser().resolve()
    worker_tmp_dir = (
        Path(args.worker_tmp_dir).expanduser().resolve()
        if args.worker_tmp_dir
        else report_out.parent / f"{report_out.stem}_workers"
    )
    worker_tmp_dir.mkdir(parents=True, exist_ok=True)
    worker_lists_dir = worker_tmp_dir / "seq_lists"
    worker_reports_dir = worker_tmp_dir / "reports"
    worker_lists_dir.mkdir(parents=True, exist_ok=True)
    worker_reports_dir.mkdir(parents=True, exist_ok=True)

    prefiltered_total = len(seq_folders)
    skipped_prefilter = 0
    if args.filter_completed:
        seq_folders, skipped_prefilter = _filter_completed_seq_folders(seq_folders, progress_dir)

    gpu_ids = _parse_gpu_list(args.gpus)
    buckets = _bucketize_round_robin(seq_folders, int(args.worker_count))

    worker_specs = []
    for worker_idx, bucket in enumerate(buckets):
        if not bucket:
            continue
        seq_list_path = worker_lists_dir / f"worker_{worker_idx:02d}.txt"
        child_report_path = worker_reports_dir / f"worker_{worker_idx:02d}.json"
        _write_seq_list(seq_list_path, bucket)
        worker_specs.append(
            {
                "worker_index": int(worker_idx),
                "gpu": gpu_ids[worker_idx % len(gpu_ids)],
                "seq_count": int(len(bucket)),
                "seq_list_path": str(seq_list_path),
                "child_report_path": str(child_report_path),
            }
        )

    print(
        json.dumps(
            {
                "workers_planned": len(worker_specs),
                "worker_count_requested": int(args.worker_count),
                "gpus": gpu_ids,
                "seq_total_input": int(prefiltered_total),
                "seq_skipped_prefilter": int(skipped_prefilter),
                "seq_remaining": int(len(seq_folders)),
                "report_out": str(report_out),
                "worker_tmp_dir": str(worker_tmp_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    if args.dry_run:
        return 0

    child_processes = []
    child_threads = []
    for spec in worker_specs:
        cmd = [
            str(args.python_bin),
            "-u",
            str(RERUN_SCRIPT),
            "--seq_folder_list",
            spec["seq_list_path"],
            "--gpu",
            spec["gpu"],
            "--checkpoint",
            str(args.checkpoint),
            "--infiller_weight",
            str(args.infiller_weight),
            "--infiller_window_batch_size",
            str(int(args.infiller_window_batch_size)),
            "--report_out",
            spec["child_report_path"],
        ]
        if args.backup_dir:
            cmd.extend(["--backup_dir", str(args.backup_dir)])
        if args.progress_dir:
            cmd.extend(["--progress_dir", str(args.progress_dir)])
        if args.cpu_threads is not None:
            cmd.extend(["--cpu_threads", str(int(args.cpu_threads))])
        if args.interop_threads is not None:
            cmd.extend(["--interop_threads", str(int(args.interop_threads))])
        _append_bool_flag(cmd, "--rebuild_cam_space_cache", bool(args.rebuild_cam_space_cache))
        _append_bool_flag(cmd, "--force", bool(args.force))
        _append_bool_flag(cmd, "--dry_run", False)

        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        prefix = f"[worker {spec['worker_index']:02d} gpu={spec['gpu'] or 'cpu'}]"
        thread = threading.Thread(
            target=_stream_worker_output,
            args=(proc.stdout, prefix),
            daemon=True,
        )
        thread.start()
        spec["pid"] = int(proc.pid)
        child_processes.append((spec, proc))
        child_threads.append(thread)

    exit_code = 0
    for spec, proc in child_processes:
        returncode = proc.wait()
        spec["returncode"] = int(returncode)
        if returncode != 0:
            exit_code = 1
    for thread in child_threads:
        thread.join()

    merged_items: list[dict] = []
    status_counts: Counter[str] = Counter()
    for spec, _proc in child_processes:
        child_items = _load_child_report(Path(spec["child_report_path"]))
        spec["report_items"] = int(len(child_items))
        for item in child_items:
            item = dict(item)
            item["worker_index"] = int(spec["worker_index"])
            item["assigned_gpu"] = spec["gpu"]
            merged_items.append(item)
            status_counts[str(item.get("status", "unknown"))] += 1

    report_payload = {
        "launcher": {
            "worker_count_requested": int(args.worker_count),
            "workers_started": int(len(worker_specs)),
            "gpus": gpu_ids,
            "python_bin": str(args.python_bin),
            "seq_total_input": int(prefiltered_total),
            "seq_skipped_prefilter": int(skipped_prefilter),
            "seq_remaining_after_prefilter": int(len(seq_folders)),
            "worker_tmp_dir": str(worker_tmp_dir),
        },
        "workers": worker_specs,
        "summary": {
            "items_total": int(len(merged_items)),
            "status_counts": dict(sorted(status_counts.items())),
            "success": int(status_counts.get("success", 0)),
            "skipped_completed": int(status_counts.get("skipped_completed", 0)),
            "nonzero_worker_exit_count": int(sum(1 for spec in worker_specs if int(spec.get("returncode", 0)) != 0)),
        },
        "items": merged_items,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "report_out": str(report_out),
                "items_total": int(len(merged_items)),
                "status_counts": dict(sorted(status_counts.items())),
                "workers_nonzero_exit": int(
                    sum(1 for spec in worker_specs if int(spec.get("returncode", 0)) != 0)
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
