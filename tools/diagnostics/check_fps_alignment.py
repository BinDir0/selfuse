#!/usr/bin/env python3
"""Check whether 5 FPS stage outputs align with 30 FPS descriptor frames."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Check whether lower-FPS stage outputs align with higher-FPS RGB shards."
    )
    parser.add_argument("--high_fps_shard_root", required=True, help="Shard root containing target RGB frames")
    parser.add_argument("--seq_folder_root", required=True, help="Root containing outputs/<clip_id>/world_space_res.pth")
    parser.add_argument(
        "--low_fps_shard_root",
        default=None,
        help="Optional shard root containing the original lower-FPS RGB frames for image-level alignment checks",
    )
    parser.add_argument("--start_factory_id", type=int, required=True, help="Inclusive factory start")
    parser.add_argument("--end_factory_id", type=int, required=True, help="Inclusive factory end")
    parser.add_argument("--source_fps", type=float, default=5.0, help="FPS of the stage outputs / optional low-FPS shards")
    parser.add_argument("--target_fps", type=float, default=30.0, help="FPS of the target RGB shards")
    parser.add_argument("--sample_clips", type=int, default=100, help="Number of clips to sample")
    parser.add_argument("--image_check_clips", type=int, default=20, help="Subset of sampled clips for image-level checks")
    parser.add_argument("--frames_per_image_check", type=int, default=8, help="Frames per clip for image-level checks")
    parser.add_argument("--max_offset", type=int, default=3, help="Search window around the expected target frame index")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report_out", default=None, help="Optional JSON report path")
    return parser


def _factory_names(start_factory_id: int, end_factory_id: int) -> list[str]:
    return [f"factory{factory_id:03d}" for factory_id in range(start_factory_id, end_factory_id + 1)]


def _load_descriptors(shard_root: str, *, start_factory_id: int, end_factory_id: int):
    from lib.pipeline.clip_manifest import discover_shard_dirs
    from lib.pipeline.video_index import collect_videos_from_factories

    shard_dirs = discover_shard_dirs(
        shard_root,
        include_dirs=_factory_names(start_factory_id, end_factory_id),
    )
    descriptors = collect_videos_from_factories(shard_dirs)
    return {descriptor.clip_id: descriptor for descriptor in descriptors}


def _load_source_frame_count(seq_folder: str) -> int:
    import joblib

    world_res_path = Path(seq_folder) / "world_space_res.pth"
    pred_trans, *_ = joblib.load(world_res_path)
    return int(np.asarray(pred_trans).shape[1])


def _expected_target_count(source_count: int, source_fps: float, target_fps: float) -> int:
    if source_count <= 0:
        return 0
    if source_count == 1:
        return 1
    duration = float(source_count - 1) / float(source_fps)
    return int(round(duration * float(target_fps))) + 1


def _decode_image(image_bytes: bytes) -> np.ndarray:
    import cv2

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode image bytes")
    return image


def _frame_sample_indices(frame_count: int, limit: int) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count <= limit:
        return list(range(frame_count))
    return sorted({int(round(x)) for x in np.linspace(0, frame_count - 1, num=limit)})


def _percentiles(values: list[float]) -> dict | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _check_clip_images(
    clip_id: str,
    low_descriptor,
    high_descriptor,
    *,
    source_fps: float,
    target_fps: float,
    frames_per_check: int,
    max_offset: int,
):
    from lib.pipeline.frame_sources import read_frame_bytes_from_descriptor

    high_frame_cache: dict[int, np.ndarray] = {}
    source_indices = _frame_sample_indices(low_descriptor.frame_count, frames_per_check)
    matches = []

    for source_idx in source_indices:
        expected_idx = int(round(float(source_idx) * float(target_fps) / float(source_fps)))
        expected_idx = min(max(expected_idx, 0), high_descriptor.frame_count - 1)
        search_start = max(0, expected_idx - max_offset)
        search_end = min(high_descriptor.frame_count - 1, expected_idx + max_offset)

        low_image = _decode_image(read_frame_bytes_from_descriptor(low_descriptor, source_idx))
        best = None
        for target_idx in range(search_start, search_end + 1):
            high_image = high_frame_cache.get(target_idx)
            if high_image is None:
                high_image = _decode_image(read_frame_bytes_from_descriptor(high_descriptor, target_idx))
                high_frame_cache[target_idx] = high_image
            if low_image.shape != high_image.shape:
                raise RuntimeError(
                    f"Shape mismatch for clip={clip_id}: low={low_image.shape} high={high_image.shape}"
                )

            diff = np.abs(low_image.astype(np.int16) - high_image.astype(np.int16))
            mae = float(diff.mean())
            pixel_equal = bool(np.array_equal(low_image, high_image))
            candidate = {
                "source_frame_idx": int(source_idx),
                "expected_target_idx": int(expected_idx),
                "target_frame_idx": int(target_idx),
                "offset_from_expected": int(target_idx - expected_idx),
                "mae": mae,
                "pixel_equal": pixel_equal,
            }
            if best is None or candidate["mae"] < best["mae"]:
                best = candidate
                if pixel_equal and target_idx == expected_idx:
                    break
        assert best is not None
        matches.append(best)

    return {
        "clip_id": clip_id,
        "frames_checked": len(matches),
        "matches": matches,
    }


def main():
    args = build_parser().parse_args()
    rng = random.Random(args.seed)

    high_by_clip = _load_descriptors(
        args.high_fps_shard_root,
        start_factory_id=args.start_factory_id,
        end_factory_id=args.end_factory_id,
    )
    if not high_by_clip:
        raise RuntimeError("No clips found under --high_fps_shard_root")

    low_by_clip = None
    if args.low_fps_shard_root:
        low_by_clip = _load_descriptors(
            args.low_fps_shard_root,
            start_factory_id=args.start_factory_id,
            end_factory_id=args.end_factory_id,
        )
        if not low_by_clip:
            raise RuntimeError("No clips found under --low_fps_shard_root")

    high_descriptors = list(high_by_clip.values())
    from lib.pipeline.clip_manifest import remap_descriptor_seq_folders

    remap_descriptor_seq_folders(high_descriptors, args.seq_folder_root)

    sampled_high = high_descriptors if len(high_descriptors) <= args.sample_clips else rng.sample(high_descriptors, args.sample_clips)
    sampled_high.sort(key=lambda item: item.clip_id)

    count_checks = []
    count_deltas = []
    abs_count_deltas = []
    ratio_errors = []
    missing_world_res = []

    for descriptor in sampled_high:
        seq_folder = Path(descriptor.seq_folder)
        world_res_path = seq_folder / "world_space_res.pth"
        if not world_res_path.is_file():
            missing_world_res.append(descriptor.clip_id)
            continue

        try:
            source_count = _load_source_frame_count(str(seq_folder))
        except Exception as exc:
            missing_world_res.append(f"{descriptor.clip_id}: {exc}")
            continue

        actual_target_count = int(descriptor.frame_count)
        expected_target_count = _expected_target_count(source_count, args.source_fps, args.target_fps)
        count_delta = int(actual_target_count - expected_target_count)
        count_deltas.append(float(count_delta))
        abs_count_deltas.append(float(abs(count_delta)))

        actual_ratio = None
        expected_ratio = None
        ratio_error = None
        if source_count > 1 and actual_target_count > 1:
            actual_ratio = float(actual_target_count - 1) / float(source_count - 1)
            expected_ratio = float(args.target_fps) / float(args.source_fps)
            ratio_error = actual_ratio - expected_ratio
            ratio_errors.append(float(ratio_error))

        count_checks.append(
            {
                "clip_id": descriptor.clip_id,
                "seq_folder": str(seq_folder),
                "source_frame_count": int(source_count),
                "actual_target_frame_count": int(actual_target_count),
                "expected_target_frame_count": int(expected_target_count),
                "count_delta": int(count_delta),
                "actual_ratio": actual_ratio,
                "expected_ratio": expected_ratio,
                "ratio_error": ratio_error,
            }
        )

    image_checks = []
    image_maes = []
    image_offsets = []
    exact_pixel_hits = 0
    total_image_matches = 0
    missing_low_fps = []
    if low_by_clip is not None:
        eligible = [item for item in count_checks if item["clip_id"] in low_by_clip]
        if len(eligible) > args.image_check_clips:
            eligible = rng.sample(eligible, args.image_check_clips)
        eligible.sort(key=lambda item: item["clip_id"])

        for item in eligible:
            clip_id = item["clip_id"]
            try:
                image_check = _check_clip_images(
                    clip_id,
                    low_by_clip[clip_id],
                    high_by_clip[clip_id],
                    source_fps=args.source_fps,
                    target_fps=args.target_fps,
                    frames_per_check=args.frames_per_image_check,
                    max_offset=args.max_offset,
                )
            except KeyError:
                missing_low_fps.append(clip_id)
                continue

            for match in image_check["matches"]:
                image_maes.append(float(match["mae"]))
                image_offsets.append(float(match["offset_from_expected"]))
                total_image_matches += 1
                if match["pixel_equal"] and match["offset_from_expected"] == 0:
                    exact_pixel_hits += 1
            image_checks.append(image_check)

    summary = {
        "high_fps_shard_root": str(Path(args.high_fps_shard_root).resolve()),
        "seq_folder_root": str(Path(args.seq_folder_root).resolve()),
        "low_fps_shard_root": str(Path(args.low_fps_shard_root).resolve()) if args.low_fps_shard_root else None,
        "factory_range": {
            "start_factory_id": int(args.start_factory_id),
            "end_factory_id": int(args.end_factory_id),
        },
        "fps": {
            "source_fps": float(args.source_fps),
            "target_fps": float(args.target_fps),
        },
        "sampled_clips": len(sampled_high),
        "count_checks_ok": len(count_checks),
        "missing_world_res": missing_world_res,
        "count_delta_stats": _percentiles(count_deltas),
        "abs_count_delta_stats": _percentiles(abs_count_deltas),
        "ratio_error_stats": _percentiles(ratio_errors),
        "image_check_summary": None,
        "count_checks": count_checks,
        "image_checks": image_checks,
    }

    if low_by_clip is not None:
        summary["image_check_summary"] = {
            "clips_checked": len(image_checks),
            "missing_low_fps_clips": missing_low_fps,
            "frames_checked": total_image_matches,
            "mae_stats": _percentiles(image_maes),
            "offset_stats": _percentiles(image_offsets),
            "exact_pixel_match_at_expected_rate": (
                float(exact_pixel_hits) / float(total_image_matches) if total_image_matches > 0 else None
            ),
        }

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
