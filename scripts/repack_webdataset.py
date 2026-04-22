#!/usr/bin/env python3
"""Repack an existing WebDataset directory into new shard sizes."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

from tqdm import tqdm

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.webdataset_rewriter import (  # noqa: E402
    iter_shard_paths,
    iter_shard_samples,
    validate_sample_record,
    write_sample_to_tar,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Repack existing WebDataset shards")
    parser.add_argument("--source_shard_dir", required=True, help="Source directory containing shard tar files")
    parser.add_argument("--output_dir", required=True, help="Output directory for repacked shard tar files")
    parser.add_argument("--report_out", default=None, help="Optional JSON report path")
    parser.add_argument("--frames_per_shard", type=int, default=10000, help="Target frame budget for output shards")
    return parser


def _is_same_or_nested(path_a: Path, path_b: Path) -> bool:
    if path_a == path_b:
        return True
    try:
        path_a.relative_to(path_b)
        return True
    except ValueError:
        return False


def validate_io_dirs(source_dir: Path, output_dir: Path):
    source_resolved = source_dir.resolve()
    output_resolved = output_dir.resolve()
    if source_resolved == output_resolved:
        raise ValueError("--output_dir must be different from --source_shard_dir")
    if _is_same_or_nested(output_resolved, source_resolved):
        raise ValueError("--output_dir must not be inside --source_shard_dir")
    if _is_same_or_nested(source_resolved, output_resolved):
        raise ValueError("--source_shard_dir must not be inside --output_dir")


def _sample_clip_id(sample: dict, meta: dict | None) -> str:
    if meta is not None:
        clip_id = meta.get("clip_id")
        if clip_id:
            return str(clip_id)
    return sample["key"].rsplit("_f", 1)[0]


class RepackWriter:
    def __init__(self, output_dir: Path, frames_per_shard: int):
        self.output_dir = output_dir
        self.frames_per_shard = frames_per_shard
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._shard_idx = 0
        self._current_tar = None
        self._current_tmp_path = None
        self._current_output_path = None
        self._current_frame_count = 0
        self._current_clip_count = 0
        self.output_shards = []

    def _open_shard(self):
        output_name = f"shard-{self._shard_idx:06d}.tar"
        self._current_output_path = self.output_dir / output_name
        self._current_tmp_path = self.output_dir / f"{output_name}.tmp"
        self._current_tar = tarfile.open(self._current_tmp_path, "w")
        self._current_frame_count = 0
        self._current_clip_count = 0

    def _close_shard(self):
        if self._current_tar is None:
            return
        self._current_tar.close()
        if self._current_frame_count <= 0:
            if self._current_tmp_path and self._current_tmp_path.exists():
                self._current_tmp_path.unlink()
        else:
            os.replace(self._current_tmp_path, self._current_output_path)
            self.output_shards.append(
                {
                    "name": self._current_output_path.name,
                    "frames": self._current_frame_count,
                    "clips": self._current_clip_count,
                }
            )
            self._shard_idx += 1
        self._current_tar = None
        self._current_tmp_path = None
        self._current_output_path = None
        self._current_frame_count = 0
        self._current_clip_count = 0

    def add_clip(self, clip_samples: list[dict]):
        clip_frame_count = len(clip_samples)
        if clip_frame_count <= 0:
            return
        if self._current_tar is None:
            self._open_shard()
        elif self._current_frame_count > 0 and self._current_frame_count + clip_frame_count > self.frames_per_shard:
            self._close_shard()
            self._open_shard()

        for sample in clip_samples:
            write_sample_to_tar(
                self._current_tar,
                sample["key"],
                sample["image_bytes"],
                sample["lowdim_bytes"],
                sample["meta_bytes"],
                mano_bytes=sample.get("mano_bytes"),
                depth_bytes=sample.get("depth_bytes"),
            )
        self._current_frame_count += clip_frame_count
        self._current_clip_count += 1

    def finish(self):
        self._close_shard()
        return list(self.output_shards)

    def abort(self):
        if self._current_tar is not None:
            self._current_tar.close()
        if self._current_tmp_path and self._current_tmp_path.exists():
            self._current_tmp_path.unlink()
        self._current_tar = None
        self._current_tmp_path = None
        self._current_output_path = None


def repack_dataset(source_dir: Path, output_dir: Path, frames_per_shard: int) -> dict:
    shard_paths = list(iter_shard_paths(str(source_dir)))
    if not shard_paths:
        raise RuntimeError(f"No shard tar files found in {source_dir}")

    writer = RepackWriter(output_dir, frames_per_shard)
    source_shards = {}
    total_samples = 0
    total_clips = 0

    try:
        for shard_path in tqdm(shard_paths, desc="Repack shards"):
            shard_name = os.path.basename(shard_path)
            shard_stats = {
                "samples_total": 0,
                "clips_total": 0,
            }
            source_shards[shard_name] = shard_stats

            current_clip_id = None
            current_clip_samples = []

            for sample in iter_shard_samples(shard_path):
                validate_sample_record(sample)
                shard_stats["samples_total"] += 1
                total_samples += 1

                meta = None
                try:
                    meta = json.loads(sample["meta_bytes"].decode("utf-8"))
                except Exception:
                    meta = None
                clip_id = _sample_clip_id(sample, meta)

                if current_clip_id is None:
                    current_clip_id = clip_id
                elif clip_id != current_clip_id:
                    writer.add_clip(current_clip_samples)
                    shard_stats["clips_total"] += 1
                    total_clips += 1
                    current_clip_id = clip_id
                    current_clip_samples = []

                current_clip_samples.append(sample)

            if current_clip_samples:
                writer.add_clip(current_clip_samples)
                shard_stats["clips_total"] += 1
                total_clips += 1

        output_shards = writer.finish()
    except Exception:
        writer.abort()
        raise

    return {
        "source_shard_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "frames_per_shard": int(frames_per_shard),
        "total_source_shards": len(source_shards),
        "total_samples": total_samples,
        "total_clips": total_clips,
        "source_shards": source_shards,
        "rewrite": {
            "shards_written": len(output_shards),
            "frames_written": int(sum(item["frames"] for item in output_shards)),
            "clips_written": int(sum(item["clips"] for item in output_shards)),
            "output_shards": output_shards,
        },
    }


def main():
    args = build_parser().parse_args()
    if args.frames_per_shard < 1:
        raise ValueError("--frames_per_shard must be >= 1")

    source_dir = Path(args.source_shard_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source shard dir not found: {source_dir}")

    output_dir = Path(args.output_dir)
    validate_io_dirs(source_dir, output_dir)

    report = repack_dataset(source_dir, output_dir, args.frames_per_shard)
    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
