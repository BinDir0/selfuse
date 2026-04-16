#!/usr/bin/env python3
"""Rewrite existing WebDataset shards with instruction metadata from clip annotations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from multiprocessing import get_context
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.annotation_protocol import load_clip_annotation  # noqa: E402
from lib.pipeline.exporters.webdataset_rewriter import (  # noqa: E402
    build_updated_meta,
    iter_shard_paths,
    iter_shard_samples,
    validate_sample_record,
    write_sample_to_tar,
)

_WORKER_ARGS = None


def build_parser():
    parser = argparse.ArgumentParser(description="Reinject instruction metadata into existing WebDataset shards")
    parser.add_argument("--source_shard_dir", required=True, help="Directory containing source shard tar files")
    parser.add_argument("--output_dir", required=True, help="Directory for rewritten shard tar files")
    parser.add_argument("--annotation_root", required=True, help="Root directory containing clip annotations")
    parser.add_argument(
        "--annotation_suffix",
        default=".annotation.json",
        help="Annotation suffix, e.g. .annotation.json",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of shard rewrite workers",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip output shard files that already exist and are non-empty",
    )
    parser.add_argument("--shard_start", type=int, default=0, help="Start shard index (inclusive)")
    parser.add_argument("--shard_end", type=int, default=None, help="End shard index (exclusive)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail a shard on missing/invalid/empty annotations instead of keeping empty instructions",
    )
    return parser


def select_shard_paths(shard_dir: str, *, shard_start: int, shard_end: int | None, output_dir: str, resume: bool) -> list[str]:
    shard_paths = list(iter_shard_paths(shard_dir))
    selected = shard_paths[shard_start:shard_end]
    if not resume:
        return selected

    kept = []
    for shard_path in selected:
        output_path = os.path.join(output_dir, os.path.basename(shard_path))
        if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            continue
        kept.append(shard_path)
    return kept


def rewrite_shard(
    shard_path: str,
    *,
    output_dir: str,
    annotation_root: str,
    annotation_suffix: str,
    strict: bool,
) -> dict:
    output_path = os.path.join(output_dir, os.path.basename(shard_path))
    tmp_path = output_path + ".tmp"
    frames_written = 0
    rewritten_frames = 0
    empty_frames = 0
    clip_stats: dict[str, str] = {}

    tar_writer = None
    try:
        for sample in iter_shard_samples(shard_path):
            validate_sample_record(sample)
            meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            clip_id = meta.get("clip_id")
            if not clip_id:
                raise RuntimeError(f"Sample {sample['key']} missing clip_id in meta")

            annotation, error_code, annotation_path = load_clip_annotation(
                annotation_root,
                clip_id,
                annotation_suffix=annotation_suffix,
            )
            if annotation is None:
                clip_stats.setdefault(clip_id, error_code or "unknown")
                if strict:
                    raise RuntimeError(
                        f"Failed to load annotation for clip_id={clip_id}: {error_code} ({annotation_path})"
                    )
                updated_meta = build_updated_meta(sample["meta_bytes"], [], language=meta.get("language"))
                empty_frames += 1
            else:
                clip_stats.setdefault(clip_id, "updated")
                updated_meta = build_updated_meta(
                    sample["meta_bytes"],
                    annotation.instruction,
                    language=annotation.language,
                )
                rewritten_frames += 1

            if tar_writer is None:
                os.makedirs(output_dir, exist_ok=True)
                tar_writer = tarfile.open(tmp_path, "w")

            write_sample_to_tar(
                tar_writer,
                sample["key"],
                sample["image_bytes"],
                sample["lowdim_bytes"],
                updated_meta,
                mano_bytes=sample.get("mano_bytes"),
            )
            frames_written += 1
    except Exception:
        if tar_writer is not None:
            tar_writer.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if tar_writer is not None:
        tar_writer.close()
        os.replace(tmp_path, output_path)

    counts = {
        "updated": sum(1 for value in clip_stats.values() if value == "updated"),
        "missing_annotation": sum(1 for value in clip_stats.values() if value == "missing_annotation"),
        "invalid_json": sum(1 for value in clip_stats.values() if value == "invalid_json"),
        "invalid_status": sum(1 for value in clip_stats.values() if value == "invalid_status"),
        "empty_instruction": sum(1 for value in clip_stats.values() if value == "empty_instruction"),
    }
    return {
        "shard": os.path.basename(shard_path),
        "output_path": output_path,
        "frames_written": frames_written,
        "rewritten_frames": rewritten_frames,
        "empty_frames": empty_frames,
        "clip_counts": counts,
    }


def _worker_init(args_dict: dict):
    global _WORKER_ARGS
    _WORKER_ARGS = dict(args_dict)


def _worker_rewrite_shard(shard_path: str) -> dict:
    return rewrite_shard(shard_path, **_WORKER_ARGS)


def main():
    args = build_parser().parse_args()
    shard_paths = select_shard_paths(
        args.source_shard_dir,
        shard_start=args.shard_start,
        shard_end=args.shard_end,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    if not shard_paths:
        print("No shard files selected.")
        return

    worker_args = {
        "output_dir": args.output_dir,
        "annotation_root": args.annotation_root,
        "annotation_suffix": args.annotation_suffix,
        "strict": args.strict,
    }
    totals = {
        "shards": 0,
        "frames_written": 0,
        "rewritten_frames": 0,
        "empty_frames": 0,
        "updated_clips": 0,
        "missing_annotation": 0,
        "invalid_json": 0,
        "invalid_status": 0,
        "empty_instruction": 0,
    }

    if args.workers <= 1:
        result_iter = (
            rewrite_shard(
                shard_path,
                output_dir=args.output_dir,
                annotation_root=args.annotation_root,
                annotation_suffix=args.annotation_suffix,
                strict=args.strict,
            )
            for shard_path in shard_paths
        )
    else:
        mp_context = get_context()
        pool = mp_context.Pool(args.workers, initializer=_worker_init, initargs=(worker_args,))
        result_iter = pool.imap_unordered(_worker_rewrite_shard, shard_paths, chunksize=1)

    try:
        for result in tqdm(result_iter, total=len(shard_paths), desc="Rewrite shards"):
            totals["shards"] += 1
            totals["frames_written"] += result["frames_written"]
            totals["rewritten_frames"] += result["rewritten_frames"]
            totals["empty_frames"] += result["empty_frames"]
            for key, value in result["clip_counts"].items():
                if key == "updated":
                    totals["updated_clips"] += value
                else:
                    totals[key] += value
    finally:
        if args.workers > 1:
            pool.close()
            pool.join()

    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
