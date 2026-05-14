#!/usr/bin/env python3
"""Small measurement lab for legacy rot6d WDS repair.

The lab builds a tiny synthetic legacy WDS, runs the real repair CLI, then
checks the repaired output with the same VLA sample checker used by
filter_and_check_datasets.py.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.webdataset_rewriter import iter_shard_samples, write_sample_to_tar  # noqa: E402
from scripts.filter_and_check_datasets import DataChecker, DataSkipError, check_vla_wds_sample  # noqa: E402


REPAIR_SCRIPT = PROJECT_ROOT / "scripts" / "repair_legacy_rot6d_wds.py"
LEGACY_IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tiny measurable legacy-rot6d WDS repair lab.")
    parser.add_argument(
        "--case",
        choices=("all", "clean", "all-legacy", "all-empty", "all-generic", "all-state-action-scale", "mixed"),
        default="all",
        help="Which lab case to run.",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Synthetic episodes per lab case.")
    parser.add_argument("--frames-per-episode", type=int, default=2, help="Synthetic frames per episode.")
    parser.add_argument("--shards", type=int, default=2, help="Synthetic shard count.")
    parser.add_argument("--workers", type=int, default=2, help="Workers passed to repair_legacy_rot6d_wds.py.")
    parser.add_argument("--executor", choices=("process", "thread"), default="process")
    parser.add_argument("--keep-workdir", action="store_true", help="Keep the temporary lab directory.")
    return parser


def encode_npy(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return buffer.getvalue()


def legacy_lowdim() -> np.ndarray:
    lowdim = np.zeros(116, dtype=np.float32)
    for start in (6, 12, 54, 60):
        lowdim[start:start + 6] = LEGACY_IDENTITY_ROT6D
    lowdim[96:112] = np.eye(4, dtype=np.float32).reshape(-1)
    lowdim[112:116] = np.asarray([100.0, 100.0, 64.0, 64.0], dtype=np.float32)
    return lowdim


def build_fixture(source_dir: Path, *, episodes: int, frames_per_episode: int, shards: int) -> int:
    source_dir.mkdir(parents=True, exist_ok=True)
    shard_writers: list[tarfile.TarFile] = []
    try:
        for shard_idx in range(shards):
            shard_writers.append(tarfile.open(source_dir / f"shard-{shard_idx:06d}.tar", "w"))

        sample_count = 0
        for episode_idx in range(episodes):
            clip_id = f"lab_ep{episode_idx:05d}"
            for frame_idx in range(frames_per_episode):
                sample_key = f"{clip_id}_f{frame_idx:06d}"
                meta = {
                    "clip_id": clip_id,
                    "episode_id": clip_id,
                    "instruction": ["pick up the object"],
                    "instruction_num": 1,
                    "cameras": ["head"],
                }
                tar_writer = shard_writers[sample_count % shards]
                write_sample_to_tar(
                    tar_writer,
                    sample_key,
                    b"synthetic-jpeg-not-decoded",
                    encode_npy(legacy_lowdim()),
                    json.dumps(meta, ensure_ascii=False).encode("utf-8"),
                )
                sample_count += 1
        return sample_count
    finally:
        for tar_writer in shard_writers:
            tar_writer.close()


def case_args(case_name: str) -> list[str]:
    if case_name == "clean":
        return []
    if case_name == "all-legacy":
        return ["--keep-legacy-rot6d-episode-fraction", "1.0"]
    if case_name == "all-empty":
        return [
            "--dirty-instruction-episode-fraction",
            "1.0",
            "--dirty-instruction-mode",
            "empty",
        ]
    if case_name == "all-generic":
        return [
            "--dirty-instruction-episode-fraction",
            "1.0",
            "--dirty-instruction-mode",
            "generic",
            "--generic-instruction",
            "do something useful",
        ]
    if case_name == "all-state-action-scale":
        return [
            "--dirty-state-action-scale-episode-fraction",
            "1.0",
            "--dirty-state-action-scale-min",
            "0.9",
            "--dirty-state-action-scale-max",
            "1.1",
        ]
    if case_name == "mixed":
        return [
            "--dirty-seed",
            "legacy-rot6d-lab",
            "--keep-legacy-rot6d-episode-fraction",
            "0.35",
            "--dirty-instruction-episode-fraction",
            "0.35",
            "--dirty-instruction-mode",
            "generic",
            "--generic-instruction",
            "do something useful",
            "--dirty-state-action-scale-episode-fraction",
            "0.35",
        ]
    raise ValueError(f"unknown case: {case_name}")


def run_repair_case(
    *,
    case_name: str,
    root: Path,
    source_dir: Path,
    workers: int,
    executor: str,
) -> dict[str, Any]:
    output_dir = root / f"out_{case_name}"
    report_path = root / f"{case_name}_report.json"
    progress_path = root / f"{case_name}_progress.jsonl"
    cmd = [
        sys.executable,
        str(REPAIR_SCRIPT),
        "--source-shard-dir",
        str(source_dir),
        "--output-dir",
        str(output_dir),
        "--workers",
        str(workers),
        "--executor",
        executor,
        "--progress-out",
        str(progress_path),
        "--progress-interval",
        "1",
        "--report-out",
        str(report_path),
        "--no-resume",
        *case_args(case_name),
    ]

    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(
            f"repair case {case_name} failed with exit {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    validation = validate_output(output_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    progress_events = Counter(
        json.loads(line)["event"]
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    samples = int(validation["samples"])
    return {
        "case": case_name,
        "elapsed_sec": elapsed,
        "samples": samples,
        "samples_per_sec": samples / elapsed if elapsed > 0 else 0.0,
        "reason_counts": validation["reason_counts"],
        "meta_counts": validation["meta_counts"],
        "repair_summary": report["summary"],
        "progress_events": dict(progress_events),
    }


def validate_output(output_dir: Path) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    meta_counts: Counter[str] = Counter()
    samples = 0
    for shard_path in sorted(output_dir.glob("shard-*.tar")):
        for sample in iter_shard_samples(str(shard_path)):
            samples += 1
            meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            marker = meta.get("lowdim_rot6d_repair", "<missing>")
            meta_counts[f"rot6d_marker:{marker}"] += 1
            for flag in meta.get("dirty_ablation_flags", []):
                meta_counts[f"dirty_flag:{flag}"] += 1

            checker = DataChecker()
            checker_sample = {
                "__key__": sample["key"],
                "__url__": str(shard_path),
                "meta.json": sample["meta_bytes"],
                "lowdim.npy": sample["lowdim_bytes"],
            }
            try:
                check_vla_wds_sample(
                    checker_sample,
                    checker,
                    check_media=False,
                    check_depth=False,
                )
            except DataSkipError as exc:
                reason_counts[type(exc).__name__] += 1
            except Exception as exc:
                reason_counts[f"Unexpected{type(exc).__name__}"] += 1
            else:
                reason_counts["OK"] += 1
    return {
        "samples": samples,
        "reason_counts": dict(reason_counts),
        "meta_counts": dict(meta_counts),
    }


def assert_case(summary: dict[str, Any]) -> None:
    case_name = summary["case"]
    reasons = Counter(summary["reason_counts"])
    samples = int(summary["samples"])
    meta = Counter(summary["meta_counts"])
    if case_name == "clean":
        if reasons != Counter({"OK": samples}):
            raise AssertionError(f"clean case should be fully OK, got {dict(reasons)}")
    elif case_name == "all-legacy":
        if reasons != Counter({"Rot6DInvalidError": samples}):
            raise AssertionError(f"all-legacy should fail rot6d, got {dict(reasons)}")
    elif case_name == "all-empty":
        if reasons != Counter({"InstructionInvalidError": samples}):
            raise AssertionError(f"all-empty should fail instruction, got {dict(reasons)}")
    elif case_name == "all-generic":
        if reasons != Counter({"OK": samples}):
            raise AssertionError(f"all-generic should remain checker-clean, got {dict(reasons)}")
        if meta.get("dirty_flag:generic_instruction", 0) != samples:
            raise AssertionError(f"all-generic should mark all samples generic, got {dict(meta)}")
    elif case_name == "all-state-action-scale":
        if reasons != Counter({"OK": samples}):
            raise AssertionError(f"all-state-action-scale should remain checker-clean, got {dict(reasons)}")
        if meta.get("dirty_flag:state_action_scale", 0) != samples:
            raise AssertionError(f"all-state-action-scale should mark all samples scaled, got {dict(meta)}")
    elif case_name == "mixed":
        if not (0 < meta.get("dirty_flag:legacy_rot6d", 0) < samples):
            raise AssertionError(f"mixed should mark some but not all legacy rot6d samples, got {dict(meta)}")
        if not (0 < meta.get("dirty_flag:generic_instruction", 0) < samples):
            raise AssertionError(f"mixed should mark some but not all generic instruction samples, got {dict(meta)}")
        if not (0 < meta.get("dirty_flag:state_action_scale", 0) < samples):
            raise AssertionError(f"mixed should mark some but not all state/action scale samples, got {dict(meta)}")


def main() -> None:
    args = build_parser().parse_args()
    requested_cases = (
        ["clean", "all-legacy", "all-empty", "all-generic", "all-state-action-scale", "mixed"]
        if args.case == "all"
        else [args.case]
    )

    temp_ctx = tempfile.TemporaryDirectory(prefix="legacy-rot6d-lab-")
    root = Path(temp_ctx.name)
    try:
        source_dir = root / "source"
        sample_count = build_fixture(
            source_dir,
            episodes=int(args.episodes),
            frames_per_episode=int(args.frames_per_episode),
            shards=int(args.shards),
        )
        summaries = []
        for case_name in requested_cases:
            summary = run_repair_case(
                case_name=case_name,
                root=root,
                source_dir=source_dir,
                workers=int(args.workers),
                executor=str(args.executor),
            )
            assert_case(summary)
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

        print(
            json.dumps(
                {
                    "status": "ok",
                    "workdir": str(root),
                    "source_samples": sample_count,
                    "cases": [item["case"] for item in summaries],
                },
                ensure_ascii=False,
            )
        )
    finally:
        if args.keep_workdir:
            print(f"Kept lab workdir: {root}", file=sys.stderr)
        else:
            temp_ctx.cleanup()


if __name__ == "__main__":
    main()
