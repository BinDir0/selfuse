#!/usr/bin/env python3
"""Backfill per-frame depth into an existing WebDataset from seq_folder depth artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from multiprocessing import Pool, get_context
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.clip_manifest import load_clip_manifest  # noqa: E402
from lib.pipeline.depth_artifacts import (  # noqa: E402
    DEPTH_EXPORT_ENCODING,
    DEPTH_EXPORT_SCHEMA,
    encode_depth_npy,
    load_export_depths,
)
from lib.pipeline.exporters.webdataset_rewriter import (  # noqa: E402
    iter_shard_paths,
    iter_shard_samples,
    validate_sample_record,
    write_sample_to_tar,
)


_WORKER_CLIP_LOOKUP = {}
_WORKER_OUTPUT_DIR = None
_WORKER_INPLACE = False
_WORKER_SKIP_EXISTING = False


def get_parser():
    parser = argparse.ArgumentParser(description="Backfill .depth.npy into an existing WebDataset")
    parser.add_argument("--source_dir", required=True, help="Directory containing shard-*.tar files")
    parser.add_argument("--descriptor_manifest", required=True, help="Manifest used to build the dataset")
    parser.add_argument("--output_dir", default=None, help="Output directory for rewritten shards")
    parser.add_argument("--inplace", action="store_true", help="Rewrite shards in place via atomic temp-file replacement")
    parser.add_argument("--workers", type=int, default=8, help="Shard rewrite workers")
    parser.add_argument("--start_shard", type=int, default=None, help="Start shard index in sorted order (inclusive)")
    parser.add_argument("--end_shard", type=int, default=None, help="End shard index in sorted order (exclusive)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip output shards that already exist and are non-empty")
    parser.add_argument("--report_out", default=None, help="Optional JSON report path")
    return parser


def _frame_index_from_sample_key(sample_key: str) -> int:
    try:
        return int(sample_key.rsplit("_f", 1)[1])
    except Exception as error:
        raise ValueError(f"Failed to parse frame index from sample key: {sample_key}") from error


def _clip_id_from_sample(sample: dict, meta: dict | None) -> str:
    if isinstance(meta, dict):
        clip_id = meta.get("clip_id")
        if clip_id:
            return str(clip_id)
    return sample["key"].rsplit("_f", 1)[0]


def _build_clip_lookup(manifest_path: str | Path) -> dict[str, dict]:
    lookup = {}
    for record in load_clip_manifest(manifest_path):
        lookup[record.clip_id] = {
            "seq_folder": str(record.descriptor.seq_folder),
            "frame_count": int(record.descriptor.frame_count),
        }
    return lookup


def _build_meta_with_depth(meta_bytes: bytes) -> bytes:
    meta = json.loads(meta_bytes.decode("utf-8"))
    meta["depth_schema"] = DEPTH_EXPORT_SCHEMA
    meta["depth_encoding"] = DEPTH_EXPORT_ENCODING
    return json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _rewrite_one_shard(shard_path: str) -> dict:
    shard_name = os.path.basename(shard_path)
    target_dir = Path(shard_path).parent if _WORKER_INPLACE else Path(_WORKER_OUTPUT_DIR)
    target_path = target_dir / shard_name
    tmp_path = target_dir / f".{shard_name}.tmp.{os.getpid()}"

    if _WORKER_SKIP_EXISTING and target_path.exists() and target_path.stat().st_size > 0:
        return {
            "shard_name": shard_name,
            "skipped_existing": True,
            "frames_written": 0,
            "frames_augmented": 0,
            "frames_already_had_depth": 0,
            "clips_loaded": 0,
        }

    result = {
        "shard_name": shard_name,
        "skipped_existing": False,
        "frames_written": 0,
        "frames_augmented": 0,
        "frames_already_had_depth": 0,
        "clips_loaded": 0,
    }
    clip_depth_cache: dict[str, object] = {}

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tmp_path, "w") as tar_writer:
            for sample in iter_shard_samples(shard_path):
                validate_sample_record(sample)
                meta = json.loads(sample["meta_bytes"].decode("utf-8"))
                clip_id = _clip_id_from_sample(sample, meta)
                frame_idx = _frame_index_from_sample_key(sample["key"])

                clip_info = _WORKER_CLIP_LOOKUP.get(clip_id)
                if clip_info is None:
                    raise KeyError(f"Clip {clip_id} not found in manifest lookup")

                if clip_id not in clip_depth_cache:
                    clip_depth_cache[clip_id] = load_export_depths(
                        clip_info["seq_folder"],
                        int(clip_info["frame_count"]),
                    )
                    result["clips_loaded"] += 1

                depth_all = clip_depth_cache[clip_id]
                if frame_idx >= len(depth_all):
                    raise IndexError(
                        f"Frame index {frame_idx} out of depth range for {clip_id}: depth_count={len(depth_all)}"
                    )

                if sample.get("depth_bytes") is None:
                    depth_bytes = encode_depth_npy(depth_all[frame_idx])
                    meta_bytes = _build_meta_with_depth(sample["meta_bytes"])
                    result["frames_augmented"] += 1
                else:
                    depth_bytes = sample["depth_bytes"]
                    meta_bytes = sample["meta_bytes"] if b'"depth_schema"' in sample["meta_bytes"] else _build_meta_with_depth(sample["meta_bytes"])
                    result["frames_already_had_depth"] += 1

                write_sample_to_tar(
                    tar_writer,
                    sample["key"],
                    sample["image_bytes"],
                    sample["lowdim_bytes"],
                    meta_bytes,
                    mano_bytes=sample.get("mano_bytes"),
                    depth_bytes=depth_bytes,
                )
                result["frames_written"] += 1
    except Exception as error:
        if tmp_path.exists():
            tmp_path.unlink()
        result["error"] = str(error)
        return result

    os.replace(tmp_path, target_path)
    return result


def _worker_init(clip_lookup: dict[str, dict], output_dir: str | None, inplace: bool, skip_existing: bool):
    global _WORKER_CLIP_LOOKUP, _WORKER_OUTPUT_DIR, _WORKER_INPLACE, _WORKER_SKIP_EXISTING
    _WORKER_CLIP_LOOKUP = clip_lookup
    _WORKER_OUTPUT_DIR = output_dir
    _WORKER_INPLACE = bool(inplace)
    _WORKER_SKIP_EXISTING = bool(skip_existing)


def _slice_shards(shard_paths: list[str], start_shard: int | None, end_shard: int | None) -> list[str]:
    start = 0 if start_shard is None else max(0, int(start_shard))
    end = len(shard_paths) if end_shard is None else min(len(shard_paths), int(end_shard))
    return shard_paths[start:end]


def main():
    args = get_parser().parse_args()
    if bool(args.output_dir) == bool(args.inplace):
        raise ValueError("Specify exactly one of --output_dir or --inplace")

    clip_lookup = _build_clip_lookup(args.descriptor_manifest)
    shard_paths = _slice_shards(list(iter_shard_paths(args.source_dir)), args.start_shard, args.end_shard)
    if not shard_paths:
        raise RuntimeError("No shards selected")

    results = []
    if args.workers <= 1:
        _worker_init(clip_lookup, args.output_dir, args.inplace, args.skip_existing)
        iterator = (_rewrite_one_shard(path) for path in shard_paths)
        for item in tqdm(iterator, total=len(shard_paths), desc="Backfill depth"):
            results.append(item)
    else:
        mp_context = get_context("spawn")
        with mp_context.Pool(
            args.workers,
            initializer=_worker_init,
            initargs=(clip_lookup, args.output_dir, args.inplace, args.skip_existing),
        ) as pool:
            for item in tqdm(pool.imap_unordered(_rewrite_one_shard, shard_paths, chunksize=1), total=len(shard_paths), desc="Backfill depth"):
                results.append(item)

    results.sort(key=lambda item: item["shard_name"])
    summary = {
        "source_dir": str(Path(args.source_dir).resolve()),
        "descriptor_manifest": str(Path(args.descriptor_manifest).resolve()),
        "output_dir": None if args.output_dir is None else str(Path(args.output_dir).resolve()),
        "inplace": bool(args.inplace),
        "workers": int(args.workers),
        "selected_shards": len(shard_paths),
        "shards_ok": sum(1 for item in results if "error" not in item),
        "shards_failed": sum(1 for item in results if "error" in item),
        "frames_written": int(sum(item.get("frames_written", 0) for item in results)),
        "frames_augmented": int(sum(item.get("frames_augmented", 0) for item in results)),
        "frames_already_had_depth": int(sum(item.get("frames_already_had_depth", 0) for item in results)),
        "shards_skipped_existing": int(sum(1 for item in results if item.get("skipped_existing"))),
        "results": results,
    }
    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
