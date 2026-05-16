#!/usr/bin/env python3
"""Unified dataset filtering and checking utility for EgoVLA.

This script consolidates the repository's existing dataset gates into one
standalone CLI:

  - wds:       scan VLA/VLM WebDataset shards and report bad samples
  - zarr-list: validate/filter zarr list files by episode count
  - hf-list:   validate/filter HF arrow/parquet dataset directories

The WebDataset checker intentionally reuses ``src.dataset.sanity_checks`` so
the thresholds match training-time data skips.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.zarr_list_utils import VALID_MAPPING_TYPES


class DataSkipError(ValueError):
    """Fallback used before the training sanity module is lazily imported."""


class MissingOrInvalidFilesError(DataSkipError):
    """Fallback used before the training sanity module is lazily imported."""


DataChecker = None
build_lowdim_slices = None


def load_training_checkers() -> None:
    """Load training-time checker classes only for WDS scans.

    Keeping these imports lazy lets --help, zarr-list, and hf-list run in
    lightweight environments that do not have the full training stack installed.
    """
    global DataChecker, DataSkipError, MissingOrInvalidFilesError, build_lowdim_slices
    from src.dataset.sanity_checks import (
        DataChecker as _DataChecker,
        DataSkipError as _DataSkipError,
        MissingOrInvalidFilesError as _MissingOrInvalidFilesError,
    )
    from src.dataset.wds_dataset import build_lowdim_slices as _build_lowdim_slices

    DataChecker = _DataChecker
    DataSkipError = _DataSkipError
    MissingOrInvalidFilesError = _MissingOrInvalidFilesError
    build_lowdim_slices = _build_lowdim_slices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified dataset filtering/checking script for EgoVLA."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    wds_parser = subparsers.add_parser(
        "wds", help="Scan VLA/VLM WebDataset shards."
    )
    wds_parser.add_argument(
        "--kind",
        choices=("auto", "vla", "vlm"),
        default="auto",
        help="Sample schema to check. auto infers from tar member names.",
    )
    wds_parser.add_argument(
        "--shards",
        nargs="+",
        required=True,
        help="Shard paths or glob patterns.",
    )
    wds_parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Stop after this many samples across all shards. 0 scans all.",
    )
    wds_parser.add_argument(
        "--check-media",
        action="store_true",
        help="Decode image/depth payloads and run media checks.",
    )
    wds_parser.add_argument(
        "--check-depth",
        action="store_true",
        help="When --check-media is set, require/check depth.npy for VLA samples.",
    )
    wds_parser.add_argument(
        "--check-image-quality",
        action="store_true",
        help="Run brightness/contrast checks for VLM images too.",
    )
    wds_parser.add_argument(
        "--target-image-size",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        default=None,
        help="Optional resize size used before VLM image finite checks.",
    )
    wds_parser.add_argument(
        "--max-shard-fail-rate",
        type=float,
        default=0.0,
        help=(
            "Shard fail-rate threshold for --filtered-shards-output. "
            "Default keeps only shards with no failures."
        ),
    )
    add_common_outputs(wds_parser)
    wds_parser.add_argument(
        "--good-keys-output",
        default=None,
        help="Optional JSONL file with passing WDS sample keys.",
    )
    wds_parser.add_argument(
        "--bad-keys-output",
        default=None,
        help="Optional JSONL file with failing WDS sample keys and reasons.",
    )
    wds_parser.add_argument(
        "--filtered-shards-output",
        default=None,
        help="Optional text file listing shards whose fail rate is within threshold.",
    )

    zarr_parser = subparsers.add_parser(
        "zarr-list", help="Validate/filter zarr list entries."
    )
    zarr_parser.add_argument("--zarr-list", required=True, help="Input zarr list.")
    zarr_parser.add_argument(
        "--min-episodes",
        type=int,
        default=10,
        help="Minimum required episode count.",
    )
    zarr_parser.add_argument(
        "--filtered-output",
        default=None,
        help="Optional filtered zarr list output.",
    )
    add_common_outputs(zarr_parser)

    hf_parser = subparsers.add_parser(
        "hf-list", help="Validate/filter HF arrow/parquet dataset directories."
    )
    hf_parser.add_argument("--hf-list", required=True, help="Input HF list.")
    hf_parser.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="train",
        help="Split directory/directories to inspect.",
    )
    hf_parser.add_argument(
        "--filtered-output",
        default=None,
        help="Optional filtered HF list output.",
    )
    add_common_outputs(hf_parser)

    return parser.parse_args()


def add_common_outputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON report path. If omitted, report is printed to stdout.",
    )


def expand_paths(patterns: Iterable[str]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            out.extend(matches)
        elif not any(ch in pattern for ch in "*?["):
            out.append(pattern)
    return out


def write_report(report: dict[str, Any], path: str | None) -> None:
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def json_load_maybe(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        return json.loads(value.decode("utf-8"))
    if isinstance(value, dict):
        return value
    raise MissingOrInvalidFilesError("missing or invalid meta.json")


def npy_load_maybe(value: Any) -> np.ndarray:
    if isinstance(value, bytes):
        return np.load(io.BytesIO(value))
    if isinstance(value, np.ndarray):
        return value
    raise MissingOrInvalidFilesError("missing or invalid npy payload")


def decode_image_bytes(value: Any) -> np.ndarray:
    from PIL import Image

    if value is None:
        raise MissingOrInvalidFilesError("missing image payload")
    if isinstance(value, bytes):
        return np.array(Image.open(io.BytesIO(value)).convert("RGB"), dtype=np.uint8)
    return np.array(value, dtype=np.uint8)


def infer_wds_kind(sample: dict[str, Any]) -> str:
    if "lowdim.npy" in sample:
        return "vla"
    if any(key.startswith("image_") and key.endswith(".jpg") for key in sample):
        return "vlm"
    raise MissingOrInvalidFilesError("cannot infer WDS sample kind")


def sample_locator(sample: dict[str, Any], meta: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = meta or {}
    return {
        "key": sample.get("__key__", ""),
        "shard": sample.get("__url__", ""),
        "dataset_name": meta.get("dataset_name", meta.get("source", "")),
        "episode_index": meta.get("episode_index", meta.get("sample_idx", "")),
    }


def lowdim_field(lowdim: np.ndarray, slices: dict[str, tuple[int, int]], key: str) -> np.ndarray:
    start, end = slices[key]
    if lowdim.ndim != 1 or lowdim.shape[0] < end:
        raise MissingOrInvalidFilesError(
            f"lowdim shape={tuple(lowdim.shape)} too short for {key}[{start}:{end}]"
        )
    return lowdim[start:end].astype(np.float32, copy=False)


def check_vla_wds_sample(
    sample: dict[str, Any],
    checker: Any,
    *,
    check_media: bool,
    check_depth: bool,
) -> None:
    meta = json_load_maybe(sample.get("meta.json"))
    lowdim = npy_load_maybe(sample.get("lowdim.npy"))

    cameras = meta.get("cameras", ["head"])
    if build_lowdim_slices is None:
        raise RuntimeError("training checker modules were not loaded")
    slices = build_lowdim_slices(cameras)
    wrist_state = lowdim_field(lowdim, slices, "wrist_state").reshape(1, -1)
    hand_state = lowdim_field(lowdim, slices, "hand_state").reshape(1, -1)
    wrist_action = lowdim_field(lowdim, slices, "wrist_action").reshape(1, -1)
    hand_action = lowdim_field(lowdim, slices, "hand_action").reshape(1, -1)
    extrinsic = lowdim_field(lowdim, slices, "head_extrinsic").reshape(4, 4)
    intrinsic = lowdim_field(lowdim, slices, "head_intrinsic")

    field_sample = {
        "wrist_state": wrist_state,
        "hand_state": hand_state,
        "wrist_action": wrist_action,
        "hand_action": hand_action,
        "extrinsic": extrinsic.reshape(-1),
        "intrinsic": intrinsic,
        "instruction": meta.get("instruction"),
        "instruction_num": meta.get("instruction_num"),
    }
    checker.check(
        sample_schema=(
            field_sample,
            {
                "required_keys": (
                    "wrist_state",
                    "hand_state",
                    "wrist_action",
                    "hand_action",
                    "extrinsic",
                    "intrinsic",
                    "instruction",
                    "instruction_num",
                ),
                "expected_last_dim": {
                    "wrist_state": 18,
                    "hand_state": 30,
                    "wrist_action": 18,
                    "hand_action": 30,
                    "extrinsic": 16,
                    "intrinsic": 4,
                },
            },
        )
    )
    checker.check(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        instruction=(meta.get("instruction"), meta.get("instruction_num")),
    )
    checker.check(
        finite={
            "wrist_state": wrist_state,
            "hand_state": hand_state,
            "wrist_action": wrist_action,
            "hand_action": hand_action,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
        },
        rot6d={"wrist_state": wrist_state, "wrist_action": wrist_action},
        state_action_delta=(wrist_state, hand_state, wrist_action, hand_action),
    )

    if not check_media:
        return

    image = decode_image_bytes(sample.get("image.jpg"))
    checker.check(image=image[None], finite={"image": image})

    if check_depth:
        depth = npy_load_maybe(sample.get("depth.npy"))
        checker.check(depth=(depth, None), finite={"depth": depth})
    elif sample.get("depth.npy") is not None:
        depth = npy_load_maybe(sample.get("depth.npy"))
        checker.check(finite={"depth": depth})


def choose_vlm_text(meta: dict[str, Any], weights: tuple[float, float, float]) -> dict[str, Any]:
    texts = meta.get("texts")
    if not isinstance(texts, list) or not texts:
        raise MissingOrInvalidFilesError("meta.texts must be a non-empty list")

    rating_fields = (
        "formatting_ratings",
        "visual_dependency_ratings",
        "relevance_ratings",
    )
    ratings = []
    for field in rating_fields:
        value = meta.get(field)
        if not isinstance(value, list) or len(value) != len(texts):
            raise MissingOrInvalidFilesError(
                f"meta.{field} must be a list aligned with texts"
            )
        ratings.append(np.array([0 if item is None else item for item in value], dtype=np.float32))

    if len(texts) == 1:
        return texts[0]
    scores = ratings[0] * weights[0] + ratings[1] * weights[1] + ratings[2] * weights[2]
    return texts[int(np.argmax(scores))]


def check_vlm_wds_sample(
    sample: dict[str, Any],
    checker: Any,
    *,
    check_media: bool,
    check_image_quality: bool,
    target_image_size: tuple[int, int] | None,
) -> None:
    meta = json_load_maybe(sample.get("meta.json"))
    checker.check(
        sample_schema=(
            {"meta.json": meta},
            {
                "required_keys": ("meta.json",),
                "required_meta_keys": (
                    "texts",
                    "formatting_ratings",
                    "visual_dependency_ratings",
                    "relevance_ratings",
                ),
            },
        )
    )

    image_keys = sorted(
        key for key in sample if key.startswith("image_") and key.endswith(".jpg")
    )
    if not image_keys:
        raise MissingOrInvalidFilesError("missing required image_*.jpg fields")

    text = choose_vlm_text(meta, weights=(0.5, 0.5, 0.5))
    question = str(text.get("user", ""))
    answer = str(text.get("assistant", ""))
    checker.check(instruction=(question, 1))
    if not answer.strip():
        raise MissingOrInvalidFilesError("selected assistant answer is empty")

    if not check_media:
        return

    from src.dataset.data_transforms import process_image

    images = np.stack([decode_image_bytes(sample[key]) for key in image_keys], axis=0)
    processed, _, _ = process_image(
        images,
        aug_transform=False,
        target_size=target_image_size,
    )
    checker.check(finite={"images_processed": processed})
    if check_image_quality:
        checker.check(image=processed)


def scan_wds(args: argparse.Namespace) -> dict[str, Any]:
    import webdataset as wds

    load_training_checkers()
    if DataChecker is None:
        raise RuntimeError("failed to load training data checker")

    shard_paths = expand_paths(args.shards)
    if not shard_paths:
        raise SystemExit("No shards matched --shards")

    checker = DataChecker()
    target_size = tuple(args.target_image_size) if args.target_image_size else None
    started = time.time()
    total = 0
    passed = 0
    failed = 0
    reason_counts: Counter[str] = Counter()
    shard_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"samples": 0, "passed": 0, "failed": 0, "reasons": Counter()}
    )

    good_file = open(args.good_keys_output, "w", encoding="utf-8") if args.good_keys_output else None
    bad_file = open(args.bad_keys_output, "w", encoding="utf-8") if args.bad_keys_output else None
    try:
        for shard_path in shard_paths:
            if args.max_samples and total >= args.max_samples:
                break
            dataset = wds.WebDataset(str(shard_path), shardshuffle=False, empty_check=False)
            for sample in dataset:
                if args.max_samples and total >= args.max_samples:
                    break

                total += 1
                checker.note_sample_seen()
                stat = shard_stats[str(shard_path)]
                stat["samples"] += 1
                meta: dict[str, Any] | None = None
                try:
                    meta = json_load_maybe(sample.get("meta.json"))
                    kind = args.kind if args.kind != "auto" else infer_wds_kind(sample)
                    if kind == "vla":
                        check_vla_wds_sample(
                            sample,
                            checker,
                            check_media=args.check_media,
                            check_depth=args.check_depth,
                        )
                    elif kind == "vlm":
                        check_vlm_wds_sample(
                            sample,
                            checker,
                            check_media=args.check_media,
                            check_image_quality=args.check_image_quality,
                            target_image_size=target_size,
                        )
                    else:
                        raise MissingOrInvalidFilesError(f"unknown kind: {kind}")
                except DataSkipError as exc:
                    failed += 1
                    stat["failed"] += 1
                    reason = type(exc).__name__
                    reason_counts[reason] += 1
                    stat["reasons"][reason] += 1
                    if bad_file is not None:
                        record = sample_locator(sample, meta)
                        record.update({"reason": reason, "message": str(exc)})
                        bad_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as exc:
                    failed += 1
                    stat["failed"] += 1
                    reason = f"Unexpected{type(exc).__name__}"
                    reason_counts[reason] += 1
                    stat["reasons"][reason] += 1
                    if bad_file is not None:
                        record = sample_locator(sample, meta)
                        record.update({"reason": reason, "message": str(exc)})
                        bad_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    passed += 1
                    stat["passed"] += 1
                    if good_file is not None:
                        good_file.write(
                            json.dumps(sample_locator(sample, meta), ensure_ascii=False) + "\n"
                        )
    finally:
        if good_file is not None:
            good_file.close()
        if bad_file is not None:
            bad_file.close()

    shard_reports = []
    kept_shards = []
    for shard_path in shard_paths:
        stat = shard_stats[str(shard_path)]
        samples = int(stat["samples"])
        failures = int(stat["failed"])
        fail_rate = failures / samples if samples else 0.0
        if samples and fail_rate <= args.max_shard_fail_rate:
            kept_shards.append(str(shard_path))
        shard_reports.append(
            {
                "shard": str(shard_path),
                "samples": samples,
                "passed": int(stat["passed"]),
                "failed": failures,
                "fail_rate": fail_rate,
                "reasons": dict(stat["reasons"]),
            }
        )

    if args.filtered_shards_output:
        Path(args.filtered_shards_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.filtered_shards_output).write_text(
            "\n".join(kept_shards) + ("\n" if kept_shards else ""),
            encoding="utf-8",
        )

    return {
        "command": "wds",
        "kind": args.kind,
        "elapsed_sec": time.time() - started,
        "shards_total": len(shard_paths),
        "samples_total": total,
        "passed": passed,
        "failed": failed,
        "fail_rate": failed / total if total else 0.0,
        "reason_counts": dict(reason_counts),
        "filtered_shards": {
            "threshold": args.max_shard_fail_rate,
            "kept": len(kept_shards),
            "output": args.filtered_shards_output,
        },
        "shards": shard_reports,
    }


def parse_zarr_list_line(raw: str, line_no: int) -> tuple[str, str, str | None]:
    parts = raw.split()
    if len(parts) == 1:
        return parts[0], "human", None
    if len(parts) == 2:
        zarr_path, mapping_type = parts
        if mapping_type not in VALID_MAPPING_TYPES:
            raise ValueError("mapping_type must be 'human' or 'real_world'")
        return zarr_path, mapping_type, None
    if len(parts) == 3:
        zarr_path, mapping_type, target_name = parts
        if mapping_type not in VALID_MAPPING_TYPES:
            raise ValueError("mapping_type must be 'human' or 'real_world'")
        if "/" in target_name or "\\" in target_name:
            raise ValueError("target name must be a simple directory name")
        return zarr_path, mapping_type, target_name
    raise ValueError(
        f"line {line_no}: expected '<zarr_path> [human|real_world] [target_name]'"
    )


def count_zarr_episodes(zarr_path: str) -> int:
    import zarr

    try:
        root = zarr.open(zarr_path, mode="r")
        return int(len(root["meta/episode_ends"]))
    except Exception:
        try:
            root = zarr.open_consolidated(zarr_path, mode="r")
            return int(len(root["meta/episode_ends"]))
        except Exception as exc:
            raise ValueError(f"failed to read meta/episode_ends: {exc}") from exc


def scan_zarr_list(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.zarr_list)
    if not input_path.exists():
        raise SystemExit(f"zarr list not found: {input_path}")

    seen_names: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    kept_lines: list[str] = []
    reason_counts: Counter[str] = Counter()

    for line_no, raw_line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        record: dict[str, Any] = {"line_no": line_no, "raw": stripped}
        try:
            zarr_path, mapping_type, target_name = parse_zarr_list_line(stripped, line_no)
            path = Path(zarr_path)
            if not path.exists():
                raise ValueError(f"zarr path does not exist: {zarr_path}")
            dataset_name = path.stem
            if dataset_name in seen_names:
                raise ValueError(
                    f"duplicate dataset name '{dataset_name}' first seen on line {seen_names[dataset_name]}"
                )
            seen_names[dataset_name] = line_no
            if target_name is None:
                target_name = dataset_name
            episodes = count_zarr_episodes(zarr_path)
            keep = episodes >= args.min_episodes
            reason = "ok" if keep else "too_few_episodes"
            if keep:
                kept_lines.append(f"{zarr_path} {mapping_type} {target_name}")
        except Exception as exc:
            episodes = None
            keep = False
            reason = type(exc).__name__
            record["message"] = str(exc)

        reason_counts[reason] += 1
        record.update(
            {
                "episodes": episodes,
                "keep": keep,
                "reason": reason,
            }
        )
        records.append(record)

    if args.filtered_output:
        Path(args.filtered_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.filtered_output).write_text(
            "\n".join(kept_lines) + ("\n" if kept_lines else ""),
            encoding="utf-8",
        )

    return {
        "command": "zarr-list",
        "input": str(input_path),
        "min_episodes": args.min_episodes,
        "entries_total": len(records),
        "kept": sum(1 for item in records if item["keep"]),
        "dropped": sum(1 for item in records if not item["keep"]),
        "reason_counts": dict(reason_counts),
        "filtered_output": args.filtered_output,
        "entries": records,
    }


def load_hf_list(path: Path) -> list[tuple[str, str | None, str]]:
    entries = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        dataset_path = parts[0]
        dataset_name = parts[1] if len(parts) > 1 else None
        entries.append((dataset_path, dataset_name, stripped))
    return entries


def collect_hf_files(dataset_path: str, split: str) -> tuple[list[str], list[str]]:
    files = []
    empty = []
    for ext in ("arrow", "parquet"):
        for file_path in sorted(glob.glob(f"{dataset_path}/{split}/*.{ext}")):
            if Path(file_path).stat().st_size > 0:
                files.append(file_path)
            else:
                empty.append(file_path)
    return files, empty


def scan_hf_list(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.hf_list)
    if not input_path.exists():
        raise SystemExit(f"HF list not found: {input_path}")

    splits = ["train", "test"] if args.split == "both" else [args.split]
    records = []
    kept_lines = []
    reason_counts: Counter[str] = Counter()

    for dataset_path, dataset_name, raw in load_hf_list(input_path):
        split_info = {}
        total_files = 0
        total_empty = 0
        for split in splits:
            valid_files, empty_files = collect_hf_files(dataset_path, split)
            split_info[split] = {
                "valid_files": len(valid_files),
                "empty_files": len(empty_files),
            }
            total_files += len(valid_files)
            total_empty += len(empty_files)

        exists = Path(dataset_path).exists()
        keep = exists and total_files > 0
        if not exists:
            reason = "path_missing"
        elif total_files == 0:
            reason = "no_nonempty_files"
        elif total_empty > 0:
            reason = "ok_with_empty_files_skipped"
        else:
            reason = "ok"
        reason_counts[reason] += 1
        if keep:
            kept_lines.append(raw)
        records.append(
            {
                "dataset_path": dataset_path,
                "dataset_name": dataset_name or Path(dataset_path).name,
                "keep": keep,
                "reason": reason,
                "splits": split_info,
            }
        )

    if args.filtered_output:
        Path(args.filtered_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.filtered_output).write_text(
            "\n".join(kept_lines) + ("\n" if kept_lines else ""),
            encoding="utf-8",
        )

    return {
        "command": "hf-list",
        "input": str(input_path),
        "split": args.split,
        "entries_total": len(records),
        "kept": sum(1 for item in records if item["keep"]),
        "dropped": sum(1 for item in records if not item["keep"]),
        "reason_counts": dict(reason_counts),
        "filtered_output": args.filtered_output,
        "entries": records,
    }


def main() -> None:
    args = parse_args()
    if args.command == "wds":
        report = scan_wds(args)
    elif args.command == "zarr-list":
        report = scan_zarr_list(args)
    elif args.command == "hf-list":
        report = scan_hf_list(args)
    else:
        raise AssertionError(args.command)
    write_report(report, args.report)


if __name__ == "__main__":
    main()
