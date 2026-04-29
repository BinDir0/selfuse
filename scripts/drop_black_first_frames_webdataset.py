#!/usr/bin/env python3
"""Drop black first frames from WebDataset episodes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
from io import BytesIO
from multiprocessing import get_context
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.webdataset_rewriter import (  # noqa: E402
    REQUIRED_SAMPLE_FIELDS,
    iter_shard_paths,
    iter_shard_samples,
    split_sample_member_name,
    validate_sample_record,
    write_sample_to_tar,
)

LOWDIM_INTRINSIC_SLICE = slice(112, 116)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite a WebDataset shard directory while dropping the first frame "
            "of each episode when that frame is black."
        )
    )
    parser.add_argument("--source_shard_dir", required=True, help="Source directory containing shard tar files")
    parser.add_argument("--output_dir", default=None, help="Output directory for rewritten shard tar files")
    parser.add_argument("--report_out", default=None, help="Optional JSON report path")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report black first frames; do not write output shards",
    )
    parser.add_argument(
        "--audit_zero_intrinsic",
        action="store_true",
        help="Only count frames whose lowdim intrinsic slice has zero values; do not write output shards",
    )
    parser.add_argument(
        "--max_mean",
        type=float,
        default=2.0,
        help="Deprecated compatibility option; black-head detection now uses lowdim intrinsic == 0",
    )
    parser.add_argument(
        "--max_pixel",
        type=int,
        default=10,
        help="Deprecated compatibility option; black-head detection now uses lowdim intrinsic == 0",
    )
    parser.add_argument(
        "--min_dark_ratio",
        type=float,
        default=0.999,
        help="Deprecated compatibility option; black-head detection now uses lowdim intrinsic == 0",
    )
    parser.add_argument(
        "--detail_limit",
        type=int,
        default=1000,
        help="Maximum number of dropped-frame details to keep in the JSON report",
    )
    parser.add_argument(
        "--progress_every_samples",
        type=int,
        default=100000,
        help="Deprecated compatibility option; current fast path reports by prepass/detect/rewrite shard",
    )
    parser.add_argument(
        "--progress_every_shards",
        type=int,
        default=1,
        help="Print a completion line every N shards; 0 disables shard completion lines",
    )
    parser.add_argument(
        "--checkpoint_every_samples",
        type=int,
        default=0,
        help=(
            "Deprecated compatibility option; when --report_out is set, the in-progress report "
            "is refreshed after each shard result"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Shard workers for the main scan/rewrite. Values >1 use a correctness-preserving "
            "prepass to find each episode's global first frame."
        ),
    )
    return parser


def _is_same_or_nested(path_a: Path, path_b: Path) -> bool:
    if path_a == path_b:
        return True
    try:
        path_a.relative_to(path_b)
        return True
    except ValueError:
        return False


def validate_io_dirs(source_dir: Path, output_dir: Path) -> None:
    source_resolved = source_dir.resolve()
    output_resolved = output_dir.resolve()
    if source_resolved == output_resolved:
        raise ValueError("--output_dir must be different from --source_shard_dir")
    if _is_same_or_nested(output_resolved, source_resolved):
        raise ValueError("--output_dir must not be inside --source_shard_dir")
    if _is_same_or_nested(source_resolved, output_resolved):
        raise ValueError("--source_shard_dir must not be inside --output_dir")


def sample_clip_id(sample: dict, meta: dict | None) -> str:
    if meta is not None:
        clip_id = meta.get("clip_id")
        if clip_id:
            return str(clip_id)
    return sample["key"].rsplit("_f", 1)[0]


def decode_meta(meta_bytes: bytes | None) -> dict | None:
    if meta_bytes is None:
        return None
    try:
        return json.loads(meta_bytes.decode("utf-8"))
    except Exception:
        return None


def lowdim_has_zero_intrinsic(lowdim_bytes: bytes) -> tuple[bool, dict]:
    array = np.load(BytesIO(lowdim_bytes), allow_pickle=False)
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    if flat.shape[0] < LOWDIM_INTRINSIC_SLICE.stop:
        raise ValueError(f"lowdim too short for intrinsic slice: shape={array.shape}")
    intrinsic = flat[LOWDIM_INTRINSIC_SLICE]
    has_zero = bool(np.any(intrinsic == 0.0))
    return has_zero, {
        "intrinsic": [float(value) for value in intrinsic.tolist()],
    }


def _new_zero_intrinsic_counts() -> dict:
    return {
        "lowdim_frames_total": 0,
        "any_zero_intrinsic_frames": 0,
        "all_zero_intrinsic_frames": 0,
        "zero_fx_frames": 0,
        "zero_fy_frames": 0,
        "zero_cx_frames": 0,
        "zero_cy_frames": 0,
        "nonfinite_intrinsic_frames": 0,
        "decode_errors": 0,
    }


def _audit_zero_intrinsic_for_shard(task: tuple[int, str, int]) -> dict:
    shard_index, shard_path_str, detail_limit = task
    shard_path = Path(shard_path_str)
    counts = _new_zero_intrinsic_counts()
    details = []

    with tarfile.open(shard_path, "r|") as tar_reader:
        for member in tar_reader:
            if not member.isfile():
                continue
            sample_key, _, field_name = split_sample_member_name(member.name)
            if sample_key is None:
                raise ValueError(f"Unsupported shard member: {member.name}")
            if field_name != "lowdim_bytes":
                continue

            counts["lowdim_frames_total"] += 1
            member_file = tar_reader.extractfile(member)
            if member_file is None:
                counts["decode_errors"] += 1
                if len(details) < detail_limit:
                    details.append(
                        {
                            "shard_name": shard_path.name,
                            "sample_key": sample_key,
                            "frame_idx": _frame_index_from_key(sample_key),
                            "error": "extractfile_none",
                        }
                    )
                continue

            try:
                array = np.load(BytesIO(member_file.read()), allow_pickle=False)
                flat = np.asarray(array, dtype=np.float32).reshape(-1)
                if flat.shape[0] < LOWDIM_INTRINSIC_SLICE.stop:
                    raise ValueError(f"lowdim too short for intrinsic slice: shape={array.shape}")
                intrinsic = flat[LOWDIM_INTRINSIC_SLICE]
            except Exception as error:  # noqa: BLE001
                counts["decode_errors"] += 1
                if len(details) < detail_limit:
                    details.append(
                        {
                            "shard_name": shard_path.name,
                            "sample_key": sample_key,
                            "frame_idx": _frame_index_from_key(sample_key),
                            "error": f"{error.__class__.__name__}: {error}",
                        }
                    )
                continue

            zero_mask = intrinsic == 0.0
            any_zero = bool(np.any(zero_mask))
            all_zero = bool(np.all(zero_mask))
            nonfinite = bool(not np.isfinite(intrinsic).all())

            counts["any_zero_intrinsic_frames"] += int(any_zero)
            counts["all_zero_intrinsic_frames"] += int(all_zero)
            counts["zero_fx_frames"] += int(bool(zero_mask[0]))
            counts["zero_fy_frames"] += int(bool(zero_mask[1]))
            counts["zero_cx_frames"] += int(bool(zero_mask[2]))
            counts["zero_cy_frames"] += int(bool(zero_mask[3]))
            counts["nonfinite_intrinsic_frames"] += int(nonfinite)

            if (any_zero or nonfinite) and len(details) < detail_limit:
                details.append(
                    {
                        "shard_name": shard_path.name,
                        "sample_key": sample_key,
                        "frame_idx": _frame_index_from_key(sample_key),
                        "intrinsic": [float(value) for value in intrinsic.tolist()],
                        "zero_mask": [bool(value) for value in zero_mask.tolist()],
                        "all_zero": all_zero,
                        "nonfinite": nonfinite,
                    }
                )

    return {
        "shard_index": int(shard_index),
        "shard_name": shard_path.name,
        "summary": counts,
        "details": details,
    }


def _merge_zero_intrinsic_audit_result(report: dict, result: dict, detail_limit: int) -> None:
    shard_name = result["shard_name"]
    report["shards"][shard_name] = dict(result["summary"])
    for key, value in result["summary"].items():
        report["summary"][key] += int(value)
    remaining_details = max(0, detail_limit - len(report["details"]))
    if remaining_details > 0:
        report["details"].extend(result["details"][:remaining_details])


def audit_zero_intrinsic_frames(
    source_dir: Path,
    *,
    workers: int = 1,
    detail_limit: int = 1000,
    progress_every_shards: int = 1,
) -> dict:
    shard_paths = list(iter_shard_paths(str(source_dir)))
    if not shard_paths:
        raise RuntimeError(f"No shard tar files found in {source_dir}")

    detail_limit = max(0, int(detail_limit))
    per_shard_detail_limit = min(detail_limit, 100) if detail_limit > 0 else 0
    summary = _new_zero_intrinsic_counts()
    summary["source_shards"] = len(shard_paths)
    report = {
        "source_shard_dir": str(source_dir.resolve()),
        "mode": "audit_zero_intrinsic",
        "thresholds": {
            "intrinsic_slice": [LOWDIM_INTRINSIC_SLICE.start, LOWDIM_INTRINSIC_SLICE.stop],
            "zero_criterion": "any intrinsic value equals 0.0",
        },
        "summary": summary,
        "shards": {},
        "details": [],
        "progress": {
            "completed_shards": 0,
            "current_shard_index": None,
            "current_shard_name": None,
        },
    }

    workers = max(1, int(workers))
    tasks = [(index, shard_path, per_shard_detail_limit) for index, shard_path in enumerate(shard_paths, start=1)]
    completed = 0
    _progress(f"Auditing zero lowdim intrinsics with {workers} worker(s)...")
    if workers <= 1:
        iterator = map(_audit_zero_intrinsic_for_shard, tasks)
        close_pool = None
    else:
        context = get_context("fork")
        pool = context.Pool(processes=workers)
        iterator = pool.imap_unordered(_audit_zero_intrinsic_for_shard, tasks, chunksize=1)
        close_pool = pool

    try:
        for result in iterator:
            completed += 1
            _merge_zero_intrinsic_audit_result(report, result, detail_limit)
            report["progress"]["completed_shards"] = int(completed)
            report["progress"]["current_shard_index"] = int(result["shard_index"])
            report["progress"]["current_shard_name"] = result["shard_name"]
            if progress_every_shards > 0 and completed % progress_every_shards == 0:
                shard_summary = result["summary"]
                _progress(
                    f"[audit {completed}/{len(shard_paths)}] {result['shard_name']}: "
                    f"frames={shard_summary['lowdim_frames_total']} "
                    f"any_zero={shard_summary['any_zero_intrinsic_frames']} "
                    f"all_zero={shard_summary['all_zero_intrinsic_frames']} "
                    f"total_any_zero={report['summary']['any_zero_intrinsic_frames']}"
                )
    finally:
        if close_pool is not None:
            close_pool.close()
            close_pool.join()

    report["shards"] = {key: report["shards"][key] for key in sorted(report["shards"])}
    return report


def _open_output_tar(output_dir: Path, shard_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / ".tmp_drop_black_first_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / shard_name
    tmp_path = tmp_dir / f"{shard_name}.tmp"
    return tarfile.open(tmp_path, "w"), output_path, tmp_path


def _write_sample(tar_writer: tarfile.TarFile, sample: dict, sample_key: str | None = None) -> None:
    write_sample_to_tar(
        tar_writer,
        sample["key"] if sample_key is None else sample_key,
        sample["image_bytes"],
        sample["lowdim_bytes"],
        sample["meta_bytes"],
        mano_bytes=sample.get("mano_bytes"),
        depth_bytes=sample.get("depth_bytes"),
    )


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _progress_report_path(report_out: str | None) -> Path | None:
    if not report_out:
        return None
    report_path = Path(report_out).expanduser()
    return report_path.with_name(f"{report_path.name}.in_progress")


def _new_shard_stats() -> dict:
    return {
        "samples_total": 0,
        "frames_written": 0,
        "episodes_started": 0,
        "black_first_frames_dropped": 0,
        "incomplete_samples": 0,
        "decode_errors": 0,
        "shard_written": False,
    }


def _frame_index_from_key(sample_key: str) -> int | None:
    try:
        return int(sample_key.rsplit("_f", 1)[1])
    except Exception:
        return None


def _candidate_sort_key(sample_key: str, shard_index: int, sample_order: int) -> tuple[int, int, int]:
    frame_index = _frame_index_from_key(sample_key)
    if frame_index is None:
        frame_index = sample_order
    return int(frame_index), int(shard_index), int(sample_order)


def _iter_shard_sample_light(shard_path: str):
    current_key = None
    current_meta = None
    current_fields = set()

    def flush():
        if current_key is None:
            return None
        return current_key, current_meta, set(current_fields)

    with tarfile.open(shard_path, "r") as tar_reader:
        for member in tar_reader:
            if not member.isfile():
                continue

            sample_key, _, field_name = split_sample_member_name(member.name)
            if sample_key is None:
                raise ValueError(f"Unsupported shard member: {member.name}")

            if current_key is None:
                current_key = sample_key
                current_meta = None
                current_fields = set()
            elif current_key != sample_key:
                item = flush()
                if item is not None:
                    yield item
                current_key = sample_key
                current_meta = None
                current_fields = set()

            current_fields.add(field_name)
            if field_name == "meta_bytes":
                member_file = tar_reader.extractfile(member)
                if member_file is None:
                    raise ValueError(f"Failed to extract shard member: {member.name}")
                current_meta = member_file.read()

    item = flush()
    if item is not None:
        yield item


def _find_shard_first_candidates(task: tuple[int, str]) -> dict:
    shard_index, shard_path = task
    first_by_clip = {}
    clip_ids = set()
    samples_total = 0
    incomplete_samples = 0
    for sample_order, (sample_key, meta_bytes, fields) in enumerate(_iter_shard_sample_light(shard_path)):
        samples_total += 1
        if any(field_name not in fields for field_name in REQUIRED_SAMPLE_FIELDS):
            incomplete_samples += 1
            continue
        meta = decode_meta(meta_bytes)
        clip_id = sample_clip_id({"key": sample_key}, meta)
        clip_ids.add(clip_id)
        sort_key = _candidate_sort_key(sample_key, shard_index, sample_order)
        previous = first_by_clip.get(clip_id)
        if previous is None or sort_key < previous["sort_key"]:
            first_by_clip[clip_id] = {
                "sample_key": sample_key,
                "sort_key": sort_key,
            }
    return {
        "shard_index": int(shard_index),
        "shard_name": Path(shard_path).name,
        "samples_total": int(samples_total),
        "incomplete_samples": int(incomplete_samples),
        "clip_ids": sorted(clip_ids),
        "first_by_clip": first_by_clip,
    }


def _build_global_first_sample_keys(shard_paths: list[str], workers: int) -> tuple[dict[str, str], dict[str, dict]]:
    _progress(f"Building global first-frame index with {workers} worker(s)...")
    tasks = [(index, shard_path) for index, shard_path in enumerate(shard_paths, start=1)]
    first_by_clip = {}
    shard_stats = {}
    completed = 0
    if workers <= 1:
        iterator = map(_find_shard_first_candidates, tasks)
        close_pool = None
    else:
        context = get_context("fork")
        pool = context.Pool(processes=workers)
        iterator = pool.imap_unordered(_find_shard_first_candidates, tasks, chunksize=1)
        close_pool = pool

    try:
        for result in iterator:
            completed += 1
            shard_stats[result["shard_name"]] = {
                "samples_total": int(result["samples_total"]),
                "incomplete_samples": int(result["incomplete_samples"]),
                "clip_ids": list(result["clip_ids"]),
            }
            for clip_id, candidate in result["first_by_clip"].items():
                previous = first_by_clip.get(clip_id)
                if previous is None or candidate["sort_key"] < previous["sort_key"]:
                    first_by_clip[clip_id] = candidate
            _progress(
                f"[prepass {completed}/{len(tasks)}] {result['shard_name']}: "
                f"samples={result['samples_total']} clips_seen={len(result['first_by_clip'])} "
                f"global_clips={len(first_by_clip)}"
            )
    finally:
        if close_pool is not None:
            close_pool.close()
            close_pool.join()

    return {clip_id: candidate["sample_key"] for clip_id, candidate in first_by_clip.items()}, shard_stats


_WORKER_FIRST_SAMPLE_BY_CLIP = None
_WORKER_FIRST_CLIP_BY_SAMPLE = None
_WORKER_FIRST_SAMPLE_KEYS = None
_WORKER_CONFIG = None


def _init_process_shard_worker(first_sample_by_clip: dict[str, str], config: dict) -> None:
    global _WORKER_FIRST_SAMPLE_BY_CLIP, _WORKER_FIRST_CLIP_BY_SAMPLE, _WORKER_FIRST_SAMPLE_KEYS, _WORKER_CONFIG
    _WORKER_FIRST_SAMPLE_BY_CLIP = first_sample_by_clip
    _WORKER_FIRST_CLIP_BY_SAMPLE = {sample_key: clip_id for clip_id, sample_key in first_sample_by_clip.items()}
    _WORKER_FIRST_SAMPLE_KEYS = set(_WORKER_FIRST_CLIP_BY_SAMPLE)
    _WORKER_CONFIG = config


def _link_unchanged_shard(source_path: Path, output_dir: Path, shard_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / ".tmp_drop_black_first_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / shard_name
    tmp_path = tmp_dir / f"{shard_name}.linktmp"
    if tmp_path.exists():
        tmp_path.unlink()
    os.link(source_path, tmp_path)
    os.replace(tmp_path, output_path)


def _copy_unchanged_shard(source_path: Path, output_dir: Path, shard_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / ".tmp_drop_black_first_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / shard_name
    tmp_path = tmp_dir / f"{shard_name}.copytmp"
    if tmp_path.exists():
        tmp_path.unlink()
    shutil.copy2(source_path, tmp_path)
    os.replace(tmp_path, output_path)


def _scan_first_frame_lowdims_for_drop_clips(
    shard_path: Path,
    first_sample_keys: set[str],
    first_clip_by_sample: dict[str, str],
    config: dict,
) -> tuple[dict[str, dict], list[dict], int, int, int]:
    drop_by_clip = {}
    details = []
    first_frames_checked = 0
    black_first_frames_dropped = 0
    decode_errors = 0
    detail_limit = int(config["detail_limit"])

    with tarfile.open(shard_path, "r") as tar_reader:
        for member in tar_reader:
            if not member.isfile():
                continue
            sample_key, _, field_name = split_sample_member_name(member.name)
            if sample_key is None:
                raise ValueError(f"Unsupported shard member: {member.name}")
            if field_name != "lowdim_bytes" or sample_key not in first_sample_keys:
                continue

            first_frames_checked += 1
            member_file = tar_reader.extractfile(member)
            if member_file is None:
                decode_errors += 1
                continue
            try:
                should_drop, lowdim_stats = lowdim_has_zero_intrinsic(member_file.read())
            except Exception as error:
                decode_errors += 1
                lowdim_stats = {"error": f"{error.__class__.__name__}: {error}"}
                should_drop = False

            if should_drop:
                clip_id = first_clip_by_sample.get(sample_key, sample_key.rsplit("_f", 1)[0])
                drop_frame_idx = _frame_index_from_key(sample_key)
                if drop_frame_idx is None:
                    drop_frame_idx = 0
                drop_by_clip[clip_id] = {
                    "sample_key": sample_key,
                    "drop_frame_idx": int(drop_frame_idx),
                }
                black_first_frames_dropped += 1
                if len(details) < detail_limit:
                    details.append(
                        {
                            "clip_id": clip_id,
                            "sample_key": sample_key,
                            "drop_frame_idx": int(drop_frame_idx),
                            "shard_name": shard_path.name,
                            **lowdim_stats,
                        }
                    )

    return drop_by_clip, details, first_frames_checked, black_first_frames_dropped, decode_errors


def _detect_drop_clips_for_shard(task: tuple[int, str]) -> dict:
    if _WORKER_FIRST_SAMPLE_KEYS is None or _WORKER_FIRST_CLIP_BY_SAMPLE is None or _WORKER_CONFIG is None:
        raise RuntimeError("Parallel worker was not initialized")
    shard_index, shard_path_str = task
    shard_path = Path(shard_path_str)
    drop_by_clip, details, first_checked, dropped, decode_errors = _scan_first_frame_lowdims_for_drop_clips(
        shard_path,
        _WORKER_FIRST_SAMPLE_KEYS,
        _WORKER_FIRST_CLIP_BY_SAMPLE,
        _WORKER_CONFIG,
    )
    return {
        "shard_index": int(shard_index),
        "shard_name": shard_path.name,
        "drop_by_clip": drop_by_clip,
        "details": details,
        "first_frames_checked": int(first_checked),
        "black_first_frames_dropped": int(dropped),
        "decode_errors": int(decode_errors),
    }


def _renumber_sample_key_after_drop(sample_key: str, drop_frame_idx: int) -> str | None:
    frame_idx = _frame_index_from_key(sample_key)
    if frame_idx is None:
        return sample_key
    if frame_idx == drop_frame_idx:
        return None
    if frame_idx < drop_frame_idx:
        return sample_key
    prefix, frame_text = sample_key.rsplit("_f", 1)
    return f"{prefix}_f{frame_idx - 1:0{len(frame_text)}d}"


def _rewrite_shard_for_dropped_clips(
    shard_path: Path,
    output_dir: Path,
    shard_name: str,
    drop_by_clip: dict[str, dict],
) -> int:
    tar_writer, output_path, tmp_path = _open_output_tar(output_dir, shard_name)
    frames_written = 0
    try:
        for sample in iter_shard_samples(str(shard_path)):
            validate_sample_record(sample)
            meta = decode_meta(sample.get("meta_bytes"))
            clip_id = sample_clip_id(sample, meta)
            drop_info = drop_by_clip.get(clip_id)
            sample_key = sample["key"]
            if drop_info is not None:
                sample_key = _renumber_sample_key_after_drop(sample_key, int(drop_info["drop_frame_idx"]))
                if sample_key is None:
                    continue
            _write_sample(tar_writer, sample, sample_key=sample_key)
            frames_written += 1
        tar_writer.close()
        tar_writer = None
        if frames_written > 0:
            os.replace(tmp_path, output_path)
        elif tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        if tar_writer is not None:
            tar_writer.close()
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return frames_written


def _process_shard_with_global_first(task: tuple[int, int, str, dict]) -> dict:
    if _WORKER_CONFIG is None:
        raise RuntimeError("Parallel worker was not initialized")

    shard_index, total_shards, shard_path_str, prepass_stats = task
    shard_path = Path(shard_path_str)
    shard_name = shard_path.name
    config = _WORKER_CONFIG
    dry_run = bool(config["dry_run"])
    output_dir = Path(config["output_dir"]) if config.get("output_dir") else None
    drop_by_clip = config.get("drop_by_clip", {})
    shard_stats = _new_shard_stats()
    shard_stats["samples_total"] = int(prepass_stats.get("samples_total", 0))
    shard_stats["incomplete_samples"] = int(prepass_stats.get("incomplete_samples", 0))
    shard_clip_ids = set(prepass_stats.get("clip_ids", []))
    affected_clip_ids = shard_clip_ids.intersection(drop_by_clip)
    summary = {
        "samples_total": int(prepass_stats.get("samples_total", 0)),
        "frames_written": 0,
        "episodes_seen": int(prepass_stats.get("first_frames_checked", 0)),
        "first_frames_checked": int(prepass_stats.get("first_frames_checked", 0)),
        "black_first_frames_dropped": int(prepass_stats.get("black_first_frames_dropped", 0)),
        "incomplete_samples": int(prepass_stats.get("incomplete_samples", 0)),
        "decode_errors": int(prepass_stats.get("decode_errors", 0)),
    }
    details = list(prepass_stats.get("details", []))
    shard_stats["episodes_started"] = int(prepass_stats.get("first_frames_checked", 0))
    shard_stats["black_first_frames_dropped"] = int(prepass_stats.get("black_first_frames_dropped", 0))
    shard_stats["decode_errors"] = int(prepass_stats.get("decode_errors", 0))

    try:
        if dry_run:
            summary["frames_written"] = max(0, shard_stats["samples_total"] - summary["black_first_frames_dropped"])
            shard_stats["frames_written"] = summary["frames_written"]
        elif output_dir is not None:
            if affected_clip_ids:
                frames_written = _rewrite_shard_for_dropped_clips(shard_path, output_dir, shard_name, drop_by_clip)
                summary["frames_written"] = int(frames_written)
                shard_stats["frames_written"] = int(frames_written)
                shard_stats["shard_written"] = True
                shard_stats["renumbered_clips"] = len(affected_clip_ids)
            else:
                try:
                    _link_unchanged_shard(shard_path, output_dir, shard_name)
                    shard_stats["hardlinked"] = True
                except OSError:
                    _copy_unchanged_shard(shard_path, output_dir, shard_name)
                    shard_stats["copied"] = True
                summary["frames_written"] = int(shard_stats["samples_total"])
                shard_stats["frames_written"] = int(shard_stats["samples_total"])
                shard_stats["shard_written"] = True
    except Exception:
        raise

    return {
        "shard_index": int(shard_index),
        "total_shards": int(total_shards),
        "shard_name": shard_name,
        "shard_stats": shard_stats,
        "summary": summary,
        "details": details,
    }


def _merge_shard_result(report: dict, result: dict, detail_limit: int) -> None:
    shard_name = result["shard_name"]
    shard_stats = result["shard_stats"]
    report["shards"][shard_name] = shard_stats
    for key, value in result["summary"].items():
        report["summary"][key] += int(value)
    if shard_stats.get("shard_written"):
        report["summary"]["shards_written"] += 1
    remaining_details = max(0, detail_limit - len(report["dropped_first_frames"]))
    if remaining_details > 0:
        report["dropped_first_frames"].extend(result["details"][:remaining_details])


def _new_report(
    source_dir: Path,
    output_dir: Path | None,
    dry_run: bool,
    shard_count: int,
    max_mean: float,
    max_pixel: int,
    min_dark_ratio: float,
) -> dict:
    return {
        "source_shard_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()) if output_dir is not None else None,
        "dry_run": bool(dry_run),
        "thresholds": {
            "drop_criterion": "first_frame_lowdim_intrinsic_any_zero",
            "intrinsic_slice": [LOWDIM_INTRINSIC_SLICE.start, LOWDIM_INTRINSIC_SLICE.stop],
        },
        "summary": {
            "source_shards": int(shard_count),
            "shards_written": 0,
            "samples_total": 0,
            "frames_written": 0,
            "episodes_seen": 0,
            "first_frames_checked": 0,
            "black_first_frames_dropped": 0,
            "incomplete_samples": 0,
            "decode_errors": 0,
        },
        "shards": {},
        "dropped_first_frames": [],
        "progress": {
            "completed_shards": 0,
            "current_shard_index": None,
            "current_shard_name": None,
        },
    }


def _drop_black_first_frames_parallel(
    source_dir: Path,
    output_dir: Path | None,
    *,
    dry_run: bool,
    max_mean: float,
    max_pixel: int,
    min_dark_ratio: float,
    detail_limit: int,
    progress_every_shards: int,
    checkpoint_path: Path | None,
    workers: int,
) -> dict:
    shard_paths = list(iter_shard_paths(str(source_dir)))
    if not shard_paths:
        raise RuntimeError(f"No shard tar files found in {source_dir}")

    if not dry_run and output_dir is None:
        raise ValueError("--output_dir is required unless --dry_run is set")

    report = _new_report(source_dir, output_dir, dry_run, len(shard_paths), max_mean, max_pixel, min_dark_ratio)
    first_sample_by_clip, prepass_stats_by_shard = _build_global_first_sample_keys(shard_paths, workers)
    report["summary"]["episodes_seen"] = 0
    report["summary"]["first_frames_checked"] = 0

    detection_config = {
        "max_mean": float(max_mean),
        "max_pixel": int(max_pixel),
        "min_dark_ratio": float(min_dark_ratio),
        "detail_limit": int(detail_limit),
    }
    drop_by_clip = {}
    detect_tasks = [(index, shard_path) for index, shard_path in enumerate(shard_paths, start=1)]
    context = get_context("fork")
    _progress(f"Scanning global first-frame lowdim intrinsics with {workers} worker(s)...")
    with context.Pool(
        processes=workers,
        initializer=_init_process_shard_worker,
        initargs=(first_sample_by_clip, detection_config),
    ) as pool:
        for result in pool.imap_unordered(_detect_drop_clips_for_shard, detect_tasks, chunksize=1):
            drop_by_clip.update(result["drop_by_clip"])
            shard_stats = prepass_stats_by_shard.setdefault(result["shard_name"], {})
            shard_stats["first_frames_checked"] = int(result["first_frames_checked"])
            shard_stats["black_first_frames_dropped"] = int(result["black_first_frames_dropped"])
            shard_stats["decode_errors"] = int(result["decode_errors"])
            shard_stats["details"] = list(result["details"])
            _progress(
                f"[detect {len([s for s in prepass_stats_by_shard.values() if 'first_frames_checked' in s])}/"
                f"{len(shard_paths)}] {result['shard_name']}: "
                f"first_frames={result['first_frames_checked']} "
                f"dropped={result['black_first_frames_dropped']} "
                f"total_dropped={len(drop_by_clip)}"
            )

    _progress(
        f"Found {len(shard_paths)} shard(s) under {source_dir}. "
        f"Mode={'dry-run' if dry_run else 'rewrite'}; workers={workers}; "
        f"indexed_clips={len(first_sample_by_clip)}; drop_clips={len(drop_by_clip)}."
    )

    config = {
        "dry_run": bool(dry_run),
        "output_dir": str(output_dir) if output_dir is not None else None,
        "max_mean": float(max_mean),
        "max_pixel": int(max_pixel),
        "min_dark_ratio": float(min_dark_ratio),
        "detail_limit": int(detail_limit),
        "drop_by_clip": drop_by_clip,
    }
    tasks = [
        (index, len(shard_paths), shard_path, prepass_stats_by_shard.get(Path(shard_path).name, {}))
        for index, shard_path in enumerate(shard_paths, start=1)
    ]
    context = get_context("fork")
    completed = 0
    with context.Pool(
        processes=workers,
        initializer=_init_process_shard_worker,
        initargs=(first_sample_by_clip, config),
    ) as pool:
        for result in pool.imap_unordered(_process_shard_with_global_first, tasks, chunksize=1):
            completed += 1
            _merge_shard_result(report, result, detail_limit)
            report["progress"]["completed_shards"] = int(completed)
            report["progress"]["current_shard_index"] = int(result["shard_index"])
            report["progress"]["current_shard_name"] = result["shard_name"]
            if checkpoint_path is not None:
                _write_json_atomic(checkpoint_path, report)
            if progress_every_shards > 0 and completed % progress_every_shards == 0:
                _progress(
                    f"[parallel {completed}/{len(shard_paths)}] done {result['shard_name']}: "
                    f"samples={result['shard_stats']['samples_total']} "
                    f"episodes={result['shard_stats']['episodes_started']} "
                    f"dropped_first_frames={result['shard_stats']['black_first_frames_dropped']} "
                    f"total_episodes={report['summary']['episodes_seen']} "
                    f"total_dropped={report['summary']['black_first_frames_dropped']}"
                )

    report["shards"] = {key: report["shards"][key] for key in sorted(report["shards"])}
    return report


def drop_black_first_frames(
    source_dir: Path,
    output_dir: Path | None,
    *,
    dry_run: bool,
    max_mean: float,
    max_pixel: int,
    min_dark_ratio: float,
    detail_limit: int,
    progress_every_samples: int = 0,
    progress_every_shards: int = 0,
    checkpoint_path: Path | None = None,
    checkpoint_every_samples: int = 0,
    workers: int = 1,
) -> dict:
    shard_paths = list(iter_shard_paths(str(source_dir)))
    if not shard_paths:
        raise RuntimeError(f"No shard tar files found in {source_dir}")

    if not dry_run and output_dir is None:
        raise ValueError("--output_dir is required unless --dry_run is set")

    return _drop_black_first_frames_parallel(
        source_dir,
        output_dir,
        dry_run=dry_run,
        max_mean=max_mean,
        max_pixel=max_pixel,
        min_dark_ratio=min_dark_ratio,
        detail_limit=detail_limit,
        progress_every_shards=progress_every_shards,
        checkpoint_path=checkpoint_path,
        workers=max(1, int(workers)),
    )

def main() -> None:
    args = build_parser().parse_args()
    if args.max_pixel < 0 or args.max_pixel > 255:
        raise ValueError("--max_pixel must be in [0, 255]")
    if not (0.0 <= args.min_dark_ratio <= 1.0):
        raise ValueError("--min_dark_ratio must be in [0, 1]")

    source_dir = Path(args.source_shard_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source shard dir not found: {source_dir}")

    if args.audit_zero_intrinsic:
        report = audit_zero_intrinsic_frames(
            source_dir,
            workers=max(1, int(args.workers)),
            detail_limit=max(0, int(args.detail_limit)),
            progress_every_shards=max(0, int(args.progress_every_shards)),
        )
        if args.report_out:
            report_path = Path(args.report_out).expanduser()
            _write_json_atomic(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    if not args.dry_run and output_dir is not None:
        validate_io_dirs(source_dir, output_dir)

    report = drop_black_first_frames(
        source_dir,
        output_dir,
        dry_run=bool(args.dry_run),
        max_mean=float(args.max_mean),
        max_pixel=int(args.max_pixel),
        min_dark_ratio=float(args.min_dark_ratio),
        detail_limit=max(0, int(args.detail_limit)),
        progress_every_samples=max(0, int(args.progress_every_samples)),
        progress_every_shards=max(0, int(args.progress_every_shards)),
        checkpoint_path=_progress_report_path(args.report_out),
        checkpoint_every_samples=max(0, int(args.checkpoint_every_samples)),
        workers=max(1, int(args.workers)),
    )

    if args.report_out:
        report_path = Path(args.report_out).expanduser()
        _write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
