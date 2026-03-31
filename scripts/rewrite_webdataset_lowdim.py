#!/usr/bin/env python3
"""Rewrite existing WebDataset shards with corrected lowdim semantics."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import tarfile
from multiprocessing import get_context
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, **kwargs):
        return iterable

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
from lib.pipeline.quality_metrics import parse_frame_index  # noqa: E402


DEFAULT_WORKERS = max(1, min(4, os.cpu_count() or 1))
BUILDAI_CLIP_RE = re.compile(r"^f(\d{3})_")
_WORKER_CLIP_INDEX = None
_WORKER_FEATURE_CACHE_DIR = None
_WORKER_DEVICE = None
_WORKER_MANO_RIGHT = None
_WORKER_MANO_LEFT = None
_WORKER_MANO_DIR = None
_WORKER_BUILDAI_ROOT = None
_WORKER_EPISODE_CACHE = {}
_WORKER_OUTPUT_DIR = None


def build_parser():
    parser = argparse.ArgumentParser(description="Rewrite WebDataset shards with corrected lowdim semantics")
    parser.add_argument("--source_shard_dir", required=True, help="Source directory containing shard tar files")
    parser.add_argument("--output_dir", required=True, help="Output directory for rewritten shards")
    parser.add_argument(
        "--buildai_processed_root",
        required=True,
        help="BuildAI processed root containing factoryXXX/outputs/<clip_id> stage outputs.",
    )
    parser.add_argument("--report_out", default=None, help="Optional JSON report path")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel shard workers")
    parser.add_argument("--mano_device", type=str, default="cuda:0", help="Device for MANO forward pass")
    parser.add_argument("--mano_gpus", type=str, default=None, help="Optional comma-separated GPU list for MANO workers")
    parser.add_argument("--mano_dir", type=str, default=None, help="Optional MANO model directory")
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default=None,
        help="Optional cache dir for corrected episode features; defaults to <output_dir>/_episode_feature_cache",
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


def validate_io_dirs(source_dir: Path, output_dir: Path):
    source_resolved = source_dir.resolve()
    output_resolved = output_dir.resolve()
    if source_resolved == output_resolved:
        raise AssertionError("--output_dir must be different from --source_shard_dir")
    if _is_same_or_nested(output_resolved, source_resolved):
        raise AssertionError("--output_dir must not be inside --source_shard_dir")
    if _is_same_or_nested(source_resolved, output_resolved):
        raise AssertionError("--source_shard_dir must not be inside --output_dir")


def _sample_clip_id(sample: dict, meta: dict | None) -> str:
    if meta is not None:
        clip_id = meta.get("clip_id")
        if clip_id:
            return str(clip_id)
    return sample["key"].rsplit("_f", 1)[0]


def _encode_lowdim(lowdim) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(lowdim, dtype=np.float32), allow_pickle=False)
    return buffer.getvalue()


def _build_updated_meta(meta: dict, clip_info: dict, presence: int) -> bytes:
    updated = dict(meta)
    updated.setdefault("dataset_name", clip_info.get("source_id") or "buildai")
    updated["clip_id"] = clip_info["clip_id"]
    updated.setdefault("split", clip_info.get("split") or "unknown")
    updated["presence"] = int(presence)
    updated["lowdim_schema"] = "hawor_wrist_world_v2"
    updated["wrist_translation_semantics"] = "mano_joint_0_world"
    updated["camera_extrinsic_convention"] = "w2c"
    return json.dumps(updated, ensure_ascii=False).encode("utf-8")


def _ensure_worker_models():
    global _WORKER_MANO_RIGHT, _WORKER_MANO_LEFT
    from lib.pipeline.exporters.webdataset_features import build_mano_models

    if _WORKER_MANO_RIGHT is not None and _WORKER_MANO_LEFT is not None:
        return
    _WORKER_MANO_RIGHT, _WORKER_MANO_LEFT = build_mano_models(_WORKER_DEVICE, mano_dir=_WORKER_MANO_DIR)
    _WORKER_MANO_RIGHT.eval()
    _WORKER_MANO_LEFT.eval()


def _worker_init(
    device_specs,
    mano_dir,
    clip_index: dict[str, dict],
    feature_cache_dir: str | None,
    output_dir: str,
    buildai_root: str,
):
    global _WORKER_CLIP_INDEX, _WORKER_FEATURE_CACHE_DIR, _WORKER_DEVICE
    global _WORKER_MANO_RIGHT, _WORKER_MANO_LEFT, _WORKER_EPISODE_CACHE
    global _WORKER_MANO_DIR, _WORKER_OUTPUT_DIR, _WORKER_BUILDAI_ROOT

    from multiprocessing import current_process
    import torch

    identity = current_process()._identity
    worker_idx = identity[0] - 1 if identity else 0
    device_str = device_specs[worker_idx % len(device_specs)]
    _WORKER_DEVICE = torch.device(device_str)
    _WORKER_CLIP_INDEX = clip_index
    _WORKER_FEATURE_CACHE_DIR = feature_cache_dir
    _WORKER_MANO_RIGHT = None
    _WORKER_MANO_LEFT = None
    _WORKER_MANO_DIR = mano_dir
    _WORKER_EPISODE_CACHE = {}
    _WORKER_OUTPUT_DIR = output_dir
    _WORKER_BUILDAI_ROOT = buildai_root


def _get_episode_data(clip_id: str):
    from lib.pipeline.exporters.manifest_vla import load_descriptor_episode_features

    if clip_id in _WORKER_EPISODE_CACHE:
        return _WORKER_EPISODE_CACHE[clip_id]

    clip_info = _WORKER_CLIP_INDEX.get(clip_id)
    if clip_info is None:
        raise KeyError(f"Clip {clip_id} not found in BuildAI index")

    _ensure_worker_models()
    feature_request = dict(clip_info)
    if feature_request.get("num_valid_frames") is None:
        feature_request.pop("num_valid_frames", None)
    episode_data = load_descriptor_episode_features(
        feature_request,
        _WORKER_MANO_RIGHT,
        _WORKER_MANO_LEFT,
        _WORKER_DEVICE,
        _WORKER_FEATURE_CACHE_DIR,
    )
    if episode_data is None:
        raise RuntimeError(f"Failed to load corrected features for clip {clip_id}")
    _WORKER_EPISODE_CACHE[clip_id] = episode_data
    return episode_data


def _build_buildai_clip_info(clip_id: str) -> dict:
    match = BUILDAI_CLIP_RE.match(clip_id)
    if not match:
        raise ValueError(
            f"Clip id does not look like BuildAI format fXXX_...: {clip_id}"
        )
    factory_id = int(match.group(1))
    seq_folder = Path(_WORKER_BUILDAI_ROOT) / f"factory{factory_id:03d}" / "outputs" / clip_id
    if not seq_folder.is_dir():
        raise FileNotFoundError(
            f"BuildAI seq_folder not found for {clip_id}: {seq_folder}"
        )
    return {
        "clip_id": clip_id,
        "episode_id": clip_id,
        "seq_folder": str(seq_folder.resolve()),
        "source_id": "buildai",
        "split": "unknown",
    }


def _resolve_clip_info(clip_id: str):
    clip_info = _WORKER_CLIP_INDEX.get(clip_id)
    if clip_info is not None:
        return clip_info

    clip_info = _build_buildai_clip_info(clip_id)
    _WORKER_CLIP_INDEX[clip_id] = clip_info
    return clip_info


def process_shard(shard_path: str) -> dict:
    shard_name = os.path.basename(shard_path)
    output_path = os.path.join(_WORKER_OUTPUT_DIR, shard_name)
    tmp_path = f"{output_path}.tmp"
    frames_rewritten = 0
    clips_touched = set()
    tar_writer = None

    try:
        for sample in iter_shard_samples(shard_path):
            validate_sample_record(sample)
            meta = {}
            try:
                meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            except Exception:
                meta = {}

            clip_id = _sample_clip_id(sample, meta)
            frame_idx = parse_frame_index(sample["key"])
            clip_info = _resolve_clip_info(clip_id)

            episode_data = _get_episode_data(clip_id)
            if frame_idx >= int(episode_data["lowdim_all"].shape[0]):
                raise IndexError(
                    f"Sample {sample['key']} requests frame {frame_idx}, "
                    f"but corrected features only have {episode_data['lowdim_all'].shape[0]} frames"
                )

            lowdim_bytes = _encode_lowdim(episode_data["lowdim_all"][frame_idx])
            meta_bytes = _build_updated_meta(meta, clip_info, int(episode_data["presence_per_frame"][frame_idx]))

            if tar_writer is None:
                os.makedirs(_WORKER_OUTPUT_DIR, exist_ok=True)
                tar_writer = tarfile.open(tmp_path, "w")

            write_sample_to_tar(
                tar_writer,
                sample["key"],
                sample["image_bytes"],
                lowdim_bytes,
                meta_bytes,
            )
            frames_rewritten += 1
            clips_touched.add(clip_id)
    except Exception:
        if tar_writer is not None:
            tar_writer.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if tar_writer is not None:
        tar_writer.close()
        os.replace(tmp_path, output_path)

    return {
        "shard_name": shard_name,
        "output_path": output_path,
        "frames_rewritten": frames_rewritten,
        "clips_touched": len(clips_touched),
        "shard_written": 1 if frames_rewritten > 0 else 0,
    }

def _worker_process_shard(shard_path: str) -> dict:
    return process_shard(shard_path)


def build_report(
    source_dir: Path,
    output_dir: Path,
    buildai_processed_root: Path,
    shard_results: list[dict],
    feature_cache_dir: Path,
) -> dict:
    report = {
        "source_shard_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "buildai_processed_root": str(buildai_processed_root.resolve()),
        "feature_cache_dir": str(feature_cache_dir.resolve()),
        "shards_total": len(shard_results),
        "shards_written": int(sum(item["shard_written"] for item in shard_results)),
        "frames_rewritten": int(sum(item["frames_rewritten"] for item in shard_results)),
        "clips_touched": int(sum(item["clips_touched"] for item in shard_results)),
        "shards": {
            item["shard_name"]: {
                "frames_rewritten": item["frames_rewritten"],
                "clips_touched": item["clips_touched"],
            }
            for item in shard_results
        },
    }
    return report


def main():
    args = build_parser().parse_args()
    import torch
    from lib.pipeline.exporters.webdataset_workers import normalize_mano_devices

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    source_dir = Path(args.source_shard_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source shard dir not found: {source_dir}")

    output_dir = Path(args.output_dir)
    validate_io_dirs(source_dir, output_dir)
    buildai_processed_root = Path(args.buildai_processed_root).resolve()
    if not buildai_processed_root.is_dir():
        raise FileNotFoundError(f"BuildAI processed root not found: {buildai_processed_root}")

    feature_cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else output_dir / "_episode_feature_cache"
    feature_cache_dir.mkdir(parents=True, exist_ok=True)

    clip_index = {}
    shard_paths = list(iter_shard_paths(str(source_dir)))
    if not shard_paths:
        raise RuntimeError(f"No shard tar files found in {source_dir}")

    mano_device_obj = torch.device(args.mano_device if torch.cuda.is_available() else "cpu")
    device_specs = normalize_mano_devices(str(mano_device_obj), args.mano_gpus if mano_device_obj.type == "cuda" else None)
    if mano_device_obj.type == "cuda":
        args.workers = min(args.workers, len(device_specs))

    if args.workers <= 1:
        _worker_init(
            device_specs,
            args.mano_dir,
            clip_index,
            str(feature_cache_dir),
            str(output_dir),
            str(buildai_processed_root),
        )
        shard_results = [process_shard(shard_path) for shard_path in tqdm(shard_paths, desc="Rewrite shards")]
    else:
        mp_context = get_context("spawn") if mano_device_obj.type == "cuda" else get_context()
        with mp_context.Pool(
            args.workers,
            initializer=_worker_init,
            initargs=(
                device_specs,
                args.mano_dir,
                clip_index,
                str(feature_cache_dir),
                str(output_dir),
                str(buildai_processed_root),
            ),
        ) as pool:
            shard_results = list(
                tqdm(
                    pool.imap_unordered(_worker_process_shard, shard_paths),
                    total=len(shard_paths),
                    desc="Rewrite shards",
                )
            )

    report = build_report(source_dir, output_dir, buildai_processed_root, shard_results, feature_cache_dir)
    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
