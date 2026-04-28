#!/usr/bin/env python3
"""Drop black first frames from WebDataset episodes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from io import BytesIO
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from PIL import Image

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
        "--max_mean",
        type=float,
        default=2.0,
        help="Maximum decoded RGB mean for a first frame to be considered black",
    )
    parser.add_argument(
        "--max_pixel",
        type=int,
        default=10,
        help="Per-channel pixel threshold used for the dark-pixel ratio check",
    )
    parser.add_argument(
        "--min_dark_ratio",
        type=float,
        default=0.999,
        help="Minimum fraction of pixels whose RGB channels are all <= --max_pixel",
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
        help="Print an in-shard progress line every N samples; 0 disables sample-level progress",
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
            "When --report_out is set, also refresh the in-progress report every N samples "
            "inside a shard; 0 only writes it after each shard"
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


def is_black_image_bytes(
    image_bytes: bytes,
    *,
    max_mean: float,
    max_pixel: int,
    min_dark_ratio: float,
) -> tuple[bool, dict]:
    with Image.open(BytesIO(image_bytes)) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)

    mean_value = float(array.mean())
    dark_pixels = np.all(array <= int(max_pixel), axis=2)
    dark_ratio = float(dark_pixels.mean())
    is_black = mean_value <= float(max_mean) and dark_ratio >= float(min_dark_ratio)
    return is_black, {
        "mean": mean_value,
        "dark_ratio": dark_ratio,
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
    }


def _open_output_tar(output_dir: Path, shard_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / ".tmp_drop_black_first_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / shard_name
    tmp_path = tmp_dir / f"{shard_name}.tmp"
    return tarfile.open(tmp_path, "w"), output_path, tmp_path


def _write_sample(tar_writer: tarfile.TarFile, sample: dict) -> None:
    write_sample_to_tar(
        tar_writer,
        sample["key"],
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
    samples_total = 0
    incomplete_samples = 0
    for sample_order, (sample_key, meta_bytes, fields) in enumerate(_iter_shard_sample_light(shard_path)):
        samples_total += 1
        if any(field_name not in fields for field_name in REQUIRED_SAMPLE_FIELDS):
            incomplete_samples += 1
            continue
        meta = decode_meta(meta_bytes)
        clip_id = sample_clip_id({"key": sample_key}, meta)
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
        "first_by_clip": first_by_clip,
    }


def _build_global_first_sample_keys(shard_paths: list[str], workers: int) -> dict[str, str]:
    _progress(f"Building global first-frame index with {workers} worker(s)...")
    tasks = [(index, shard_path) for index, shard_path in enumerate(shard_paths, start=1)]
    first_by_clip = {}
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

    return {clip_id: candidate["sample_key"] for clip_id, candidate in first_by_clip.items()}


_WORKER_FIRST_SAMPLE_BY_CLIP = None
_WORKER_CONFIG = None


def _init_process_shard_worker(first_sample_by_clip: dict[str, str], config: dict) -> None:
    global _WORKER_FIRST_SAMPLE_BY_CLIP, _WORKER_CONFIG
    _WORKER_FIRST_SAMPLE_BY_CLIP = first_sample_by_clip
    _WORKER_CONFIG = config


def _process_shard_with_global_first(task: tuple[int, int, str]) -> dict:
    if _WORKER_FIRST_SAMPLE_BY_CLIP is None or _WORKER_CONFIG is None:
        raise RuntimeError("Parallel worker was not initialized")

    shard_index, total_shards, shard_path_str = task
    shard_path = Path(shard_path_str)
    shard_name = shard_path.name
    config = _WORKER_CONFIG
    dry_run = bool(config["dry_run"])
    output_dir = Path(config["output_dir"]) if config.get("output_dir") else None
    detail_limit = int(config["detail_limit"])
    shard_stats = _new_shard_stats()
    details = []
    summary = {
        "samples_total": 0,
        "frames_written": 0,
        "episodes_seen": 0,
        "first_frames_checked": 0,
        "black_first_frames_dropped": 0,
        "incomplete_samples": 0,
        "decode_errors": 0,
    }

    tar_writer = None
    output_path = None
    tmp_path = None
    try:
        if not dry_run and output_dir is not None:
            tar_writer, output_path, tmp_path = _open_output_tar(output_dir, shard_name)

        for sample in iter_shard_samples(str(shard_path)):
            summary["samples_total"] += 1
            shard_stats["samples_total"] += 1
            try:
                validate_sample_record(sample)
            except ValueError:
                summary["incomplete_samples"] += 1
                shard_stats["incomplete_samples"] += 1
                continue

            meta = decode_meta(sample.get("meta_bytes"))
            clip_id = sample_clip_id(sample, meta)
            is_first_frame = _WORKER_FIRST_SAMPLE_BY_CLIP.get(clip_id) == sample["key"]
            drop_sample = False

            if is_first_frame:
                summary["episodes_seen"] += 1
                summary["first_frames_checked"] += 1
                shard_stats["episodes_started"] += 1
                try:
                    is_black, image_stats = is_black_image_bytes(
                        sample["image_bytes"],
                        max_mean=float(config["max_mean"]),
                        max_pixel=int(config["max_pixel"]),
                        min_dark_ratio=float(config["min_dark_ratio"]),
                    )
                except Exception as error:
                    summary["decode_errors"] += 1
                    shard_stats["decode_errors"] += 1
                    image_stats = {"error": f"{error.__class__.__name__}: {error}"}
                    is_black = False

                if is_black:
                    drop_sample = True
                    summary["black_first_frames_dropped"] += 1
                    shard_stats["black_first_frames_dropped"] += 1
                    if len(details) < detail_limit:
                        details.append(
                            {
                                "clip_id": clip_id,
                                "sample_key": sample["key"],
                                "shard_name": shard_name,
                                **image_stats,
                            }
                        )

            if drop_sample:
                continue

            if not dry_run and tar_writer is not None:
                _write_sample(tar_writer, sample)
            summary["frames_written"] += 1
            shard_stats["frames_written"] += 1

        if tar_writer is not None:
            tar_writer.close()
            tar_writer = None
            if shard_stats["frames_written"] > 0:
                os.replace(tmp_path, output_path)
                shard_stats["shard_written"] = True
            elif tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
    except Exception:
        if tar_writer is not None:
            tar_writer.close()
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
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
            "max_mean": float(max_mean),
            "max_pixel": int(max_pixel),
            "min_dark_ratio": float(min_dark_ratio),
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
    first_sample_by_clip = _build_global_first_sample_keys(shard_paths, workers)
    report["summary"]["episodes_seen"] = 0
    report["summary"]["first_frames_checked"] = 0

    _progress(
        f"Found {len(shard_paths)} shard(s) under {source_dir}. "
        f"Mode={'dry-run' if dry_run else 'rewrite'}; workers={workers}; "
        f"indexed_clips={len(first_sample_by_clip)}."
    )

    config = {
        "dry_run": bool(dry_run),
        "output_dir": str(output_dir) if output_dir is not None else None,
        "max_mean": float(max_mean),
        "max_pixel": int(max_pixel),
        "min_dark_ratio": float(min_dark_ratio),
        "detail_limit": int(detail_limit),
    }
    tasks = [(index, len(shard_paths), shard_path) for index, shard_path in enumerate(shard_paths, start=1)]
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

    workers = int(workers)
    if workers > 1:
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
            workers=workers,
        )

    seen_clip_ids: set[str] = set()
    report = {
        "source_shard_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()) if output_dir is not None else None,
        "dry_run": bool(dry_run),
        "thresholds": {
            "max_mean": float(max_mean),
            "max_pixel": int(max_pixel),
            "min_dark_ratio": float(min_dark_ratio),
        },
        "summary": {
            "source_shards": len(shard_paths),
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

    _progress(
        f"Found {len(shard_paths)} shard(s) under {source_dir}. "
        f"Mode={'dry-run' if dry_run else 'rewrite'}."
    )

    for shard_index, shard_path_str in enumerate(tqdm(shard_paths, desc="Scan shards"), start=1):
        shard_path = Path(shard_path_str)
        shard_name = shard_path.name
        shard_stats = {
            "samples_total": 0,
            "frames_written": 0,
            "episodes_started": 0,
            "black_first_frames_dropped": 0,
            "incomplete_samples": 0,
            "decode_errors": 0,
            "shard_written": False,
        }
        report["shards"][shard_name] = shard_stats

        tar_writer = None
        output_path = None
        tmp_path = None
        try:
            report["progress"]["current_shard_index"] = int(shard_index)
            report["progress"]["current_shard_name"] = shard_name
            _progress(f"[{shard_index}/{len(shard_paths)}] start {shard_name}")
            if not dry_run and output_dir is not None:
                tar_writer, output_path, tmp_path = _open_output_tar(output_dir, shard_name)

            for sample in iter_shard_samples(str(shard_path)):
                report["summary"]["samples_total"] += 1
                shard_stats["samples_total"] += 1
                if (
                    progress_every_samples > 0
                    and shard_stats["samples_total"] % progress_every_samples == 0
                ):
                    _progress(
                        f"[{shard_index}/{len(shard_paths)}] {shard_name}: "
                        f"samples={shard_stats['samples_total']} "
                        f"episodes={shard_stats['episodes_started']} "
                        f"dropped_first_frames={shard_stats['black_first_frames_dropped']} "
                        f"total_episodes={report['summary']['episodes_seen']} "
                        f"total_dropped={report['summary']['black_first_frames_dropped']}"
                    )
                if (
                    checkpoint_path is not None
                    and checkpoint_every_samples > 0
                    and report["summary"]["samples_total"] % checkpoint_every_samples == 0
                ):
                    _write_json_atomic(checkpoint_path, report)

                try:
                    validate_sample_record(sample)
                except ValueError:
                    report["summary"]["incomplete_samples"] += 1
                    shard_stats["incomplete_samples"] += 1
                    continue

                meta = decode_meta(sample.get("meta_bytes"))
                clip_id = sample_clip_id(sample, meta)
                is_first_frame = clip_id not in seen_clip_ids
                drop_sample = False

                if is_first_frame:
                    seen_clip_ids.add(clip_id)
                    report["summary"]["episodes_seen"] += 1
                    report["summary"]["first_frames_checked"] += 1
                    shard_stats["episodes_started"] += 1
                    try:
                        is_black, image_stats = is_black_image_bytes(
                            sample["image_bytes"],
                            max_mean=max_mean,
                            max_pixel=max_pixel,
                            min_dark_ratio=min_dark_ratio,
                        )
                    except Exception as error:
                        report["summary"]["decode_errors"] += 1
                        shard_stats["decode_errors"] += 1
                        image_stats = {"error": f"{error.__class__.__name__}: {error}"}
                        is_black = False

                    if is_black:
                        drop_sample = True
                        report["summary"]["black_first_frames_dropped"] += 1
                        shard_stats["black_first_frames_dropped"] += 1
                        if len(report["dropped_first_frames"]) < detail_limit:
                            report["dropped_first_frames"].append(
                                {
                                    "clip_id": clip_id,
                                    "sample_key": sample["key"],
                                    "shard_name": shard_name,
                                    **image_stats,
                                }
                            )

                if drop_sample:
                    continue

                if not dry_run and tar_writer is not None:
                    _write_sample(tar_writer, sample)
                report["summary"]["frames_written"] += 1
                shard_stats["frames_written"] += 1

            if tar_writer is not None:
                tar_writer.close()
                tar_writer = None
                if shard_stats["frames_written"] > 0:
                    os.replace(tmp_path, output_path)
                    shard_stats["shard_written"] = True
                    report["summary"]["shards_written"] += 1
                elif tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink()

            report["progress"]["completed_shards"] = int(shard_index)
            if checkpoint_path is not None:
                _write_json_atomic(checkpoint_path, report)

            if progress_every_shards > 0 and shard_index % progress_every_shards == 0:
                _progress(
                    f"[{shard_index}/{len(shard_paths)}] done {shard_name}: "
                    f"samples={shard_stats['samples_total']} "
                    f"episodes={shard_stats['episodes_started']} "
                    f"dropped_first_frames={shard_stats['black_first_frames_dropped']} "
                    f"written={shard_stats['frames_written']} "
                    f"total_episodes={report['summary']['episodes_seen']} "
                    f"total_dropped={report['summary']['black_first_frames_dropped']}"
                )
        except Exception:
            if tar_writer is not None:
                tar_writer.close()
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
            raise

    return report


def main() -> None:
    args = build_parser().parse_args()
    if args.max_pixel < 0 or args.max_pixel > 255:
        raise ValueError("--max_pixel must be in [0, 255]")
    if not (0.0 <= args.min_dark_ratio <= 1.0):
        raise ValueError("--min_dark_ratio must be in [0, 1]")

    source_dir = Path(args.source_shard_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source shard dir not found: {source_dir}")

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
