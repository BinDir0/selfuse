#!/usr/bin/env python3
"""Validate WDS depth payloads against seq_folder depth artifacts."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.depth_artifacts import (  # noqa: E402
    DEPTH_EXPORT_ENCODING,
    DEPTH_EXPORT_SCHEMA,
    depth_to_uint16_mm,
    load_export_depths,
)
from lib.pipeline.exporters.webdataset_rewriter import (  # noqa: E402
    iter_shard_paths,
    iter_shard_samples,
    validate_sample_record,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check .depth.npy payloads inside WDS against source seq_folder depth artifacts")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wds_shard", default=None, help="Single shard tar path")
    group.add_argument("--wds_dir", default=None, help="Directory containing shard-*.tar")
    parser.add_argument("--buildai_processed_root", required=True, help="Processed root used to resolve BuildAI seq_folders")
    parser.add_argument("--start_shard", type=int, default=None, help="Start shard index in sorted shard order (inclusive)")
    parser.add_argument("--end_shard", type=int, default=None, help="End shard index in sorted shard order (exclusive)")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional max number of frames to check in total")
    parser.add_argument("--report_out", default=None, help="Optional JSON report path")
    return parser


def _resolve_buildai_seq_folder(processed_root: Path, clip_id: str) -> Path:
    parts = clip_id.split("_")
    if len(parts) < 2 or not parts[0].startswith("f") or not parts[1].startswith("w"):
        raise ValueError(f"Unsupported BuildAI clip id format: {clip_id}")
    factory_id = int(parts[0][1:])
    worker_id = int(parts[1][1:])
    candidates = [
        processed_root / f"factory_{factory_id:03d}" / f"worker_{worker_id:03d}" / "processed" / clip_id,
        processed_root / f"factory{factory_id:03d}" / "outputs" / clip_id,
        processed_root / f"factory_{factory_id:03d}" / "outputs" / clip_id,
    ]
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Failed to resolve seq_folder for {clip_id}. Checked: {checked}")


def _infer_frame_count(seq_folder: Path) -> int:
    world_path = seq_folder / "world_space_res.pth"
    if world_path.is_file():
        pred_trans, *_rest = joblib.load(world_path)
        return int(np.asarray(pred_trans).shape[1])

    frame_dir = seq_folder / "extracted_images"
    if frame_dir.is_dir():
        image_count = len(sorted(frame_dir.glob("*.jpg"))) or len(sorted(frame_dir.glob("*.png")))
        if image_count > 0:
            return int(image_count)

    raise RuntimeError(f"Failed to infer frame count for {seq_folder}")


def _iter_selected_shards(args: argparse.Namespace) -> list[Path]:
    if args.wds_shard is not None:
        return [Path(args.wds_shard).expanduser().resolve()]
    shard_paths = [Path(path) for path in iter_shard_paths(str(Path(args.wds_dir).expanduser().resolve()))]
    start = 0 if args.start_shard is None else max(0, int(args.start_shard))
    end = len(shard_paths) if args.end_shard is None else min(len(shard_paths), int(args.end_shard))
    return shard_paths[start:end]


def _frame_idx_from_sample_key(sample_key: str) -> int:
    return int(sample_key.rsplit("_f", 1)[1])


def main() -> None:
    args = build_parser().parse_args()
    processed_root = Path(args.buildai_processed_root).expanduser().resolve()
    shard_paths = _iter_selected_shards(args)
    if not shard_paths:
        raise RuntimeError("No shards selected")

    clip_depth_cache: dict[str, np.ndarray] = {}
    clip_seq_cache: dict[str, str] = {}
    clip_frame_count_cache: dict[str, int] = {}
    per_shard = []
    frames_checked = 0
    frames_missing_depth = 0
    frames_mismatched = 0
    exact_match_frames = 0
    max_abs_diff_mm = 0
    mismatches = []

    for shard_path in shard_paths:
        shard_report = {
            "shard_name": shard_path.name,
            "frames_checked": 0,
            "frames_missing_depth": 0,
            "frames_mismatched": 0,
            "max_abs_diff_mm": 0,
        }
        for sample in iter_shard_samples(str(shard_path)):
            validate_sample_record(sample)
            meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            clip_id = str(meta.get("clip_id") or sample["key"].rsplit("_f", 1)[0])
            frame_idx = _frame_idx_from_sample_key(sample["key"])

            if clip_id not in clip_depth_cache:
                seq_folder = _resolve_buildai_seq_folder(processed_root, clip_id)
                clip_seq_cache[clip_id] = str(seq_folder)
                frame_count = _infer_frame_count(seq_folder)
                clip_frame_count_cache[clip_id] = frame_count
                clip_depth_cache[clip_id] = load_export_depths(seq_folder, frame_count)

            if sample.get("depth_bytes") is None:
                frames_missing_depth += 1
                shard_report["frames_missing_depth"] += 1
                continue

            actual = np.load(io.BytesIO(sample["depth_bytes"]), allow_pickle=False)
            expected = depth_to_uint16_mm(clip_depth_cache[clip_id][frame_idx])

            diff_mm = int(np.max(np.abs(actual.astype(np.int32) - expected.astype(np.int32))))
            frames_checked += 1
            shard_report["frames_checked"] += 1
            max_abs_diff_mm = max(max_abs_diff_mm, diff_mm)
            shard_report["max_abs_diff_mm"] = max(shard_report["max_abs_diff_mm"], diff_mm)

            meta_schema_ok = meta.get("depth_schema") == DEPTH_EXPORT_SCHEMA
            meta_encoding_ok = meta.get("depth_encoding") == DEPTH_EXPORT_ENCODING
            shape_ok = tuple(actual.shape) == tuple(expected.shape)
            exact_ok = shape_ok and diff_mm == 0 and meta_schema_ok and meta_encoding_ok
            if exact_ok:
                exact_match_frames += 1
            else:
                frames_mismatched += 1
                shard_report["frames_mismatched"] += 1
                if len(mismatches) < 50:
                    mismatches.append(
                        {
                            "shard_name": shard_path.name,
                            "clip_id": clip_id,
                            "frame_idx": int(frame_idx),
                            "seq_folder": clip_seq_cache[clip_id],
                            "shape_actual": list(actual.shape),
                            "shape_expected": list(expected.shape),
                            "max_abs_diff_mm": int(diff_mm),
                            "meta_schema": meta.get("depth_schema"),
                            "meta_encoding": meta.get("depth_encoding"),
                            "meta_schema_ok": bool(meta_schema_ok),
                            "meta_encoding_ok": bool(meta_encoding_ok),
                        }
                    )

            if args.max_frames is not None and frames_checked >= int(args.max_frames):
                break

        per_shard.append(shard_report)
        if args.max_frames is not None and frames_checked >= int(args.max_frames):
            break

    summary = {
        "buildai_processed_root": str(processed_root),
        "selected_shards": [str(path) for path in shard_paths],
        "frames_checked": int(frames_checked),
        "frames_missing_depth": int(frames_missing_depth),
        "frames_mismatched": int(frames_mismatched),
        "exact_match_frames": int(exact_match_frames),
        "exact_match_ratio": float(exact_match_frames / frames_checked) if frames_checked > 0 else 0.0,
        "max_abs_diff_mm": int(max_abs_diff_mm),
        "per_shard": per_shard,
        "mismatches": mismatches,
    }
    if args.report_out:
        report_path = Path(args.report_out).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
