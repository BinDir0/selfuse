#!/usr/bin/env python3
"""Drop black first frames from WebDataset episodes."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.webdataset_rewriter import (  # noqa: E402
    iter_shard_paths,
    iter_shard_samples,
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
    output_path = output_dir / shard_name
    tmp_path = output_dir / f"{shard_name}.tmp"
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


def drop_black_first_frames(
    source_dir: Path,
    output_dir: Path | None,
    *,
    dry_run: bool,
    max_mean: float,
    max_pixel: int,
    min_dark_ratio: float,
    detail_limit: int,
) -> dict:
    shard_paths = list(iter_shard_paths(str(source_dir)))
    if not shard_paths:
        raise RuntimeError(f"No shard tar files found in {source_dir}")

    if not dry_run and output_dir is None:
        raise ValueError("--output_dir is required unless --dry_run is set")

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
    }

    for shard_path_str in tqdm(shard_paths, desc="Scan shards"):
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
            if not dry_run and output_dir is not None:
                tar_writer, output_path, tmp_path = _open_output_tar(output_dir, shard_name)

            for sample in iter_shard_samples(str(shard_path)):
                report["summary"]["samples_total"] += 1
                shard_stats["samples_total"] += 1

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
    )

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
