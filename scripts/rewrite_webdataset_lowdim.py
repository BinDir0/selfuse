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
    parse_episode_index,
    validate_sample_record,
    write_sample_to_tar,
)
from lib.pipeline.quality_metrics import parse_frame_index  # noqa: E402


DEFAULT_WORKERS = max(1, min(4, os.cpu_count() or 1))
BUILDAI_CLIP_RE = re.compile(r"^f(\d{3})_")
LEGACY_BUILDAI_EP_RE = re.compile(r"^buildai_ep(\d+)$")
_WORKER_CLIP_INDEX = None
_WORKER_LEGACY_EPISODES = None
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
    parser.add_argument(
        "--legacy_buildai_input_dir",
        default=None,
        help="Optional old builder input_dir for resolving legacy buildai_epXXXX sample keys.",
    )
    parser.add_argument(
        "--legacy_episode_list",
        default=None,
        help="Optional episode_list used by the old builder when producing legacy buildai_epXXXX shards.",
    )
    parser.add_argument(
        "--legacy_factory_range",
        default=None,
        help="Optional factory range like 1-50 used by the old builder when producing legacy buildai_epXXXX shards.",
    )
    parser.add_argument(
        "--legacy_episode_cache",
        default=None,
        help="Optional path to the old builder _vla_episodes_cache.json for resolving buildai_epXXXX keys.",
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


def build_legacy_episode_index(
    input_dir: str,
    *,
    episode_list: str | None,
    factory_range: str | None,
    cache_file: str | None,
) -> dict[int, dict]:
    from lib.pipeline.exporters.webdataset_discovery import discover_episodes

    episodes = discover_episodes(
        input_dir,
        episode_list=episode_list,
        cache_file=cache_file,
        require_world_res=True,
        factory_range=factory_range,
    )
    legacy_index = {}
    for ep in episodes:
        episode_index = int(ep["episode_index"])
        legacy_index[episode_index] = {
            "clip_id": str(ep["episode_id"]),
            "episode_id": str(ep["episode_id"]),
            "seq_folder": str(Path(ep["crop_dir"]).resolve()),
            "source_id": "buildai",
            "split": "unknown",
            "legacy_episode_index": episode_index,
        }
    return legacy_index


def build_legacy_episode_index_from_processed_root(
    processed_root: str,
    *,
    factory_range: str | None,
) -> dict[int, dict]:
    from lib.pipeline.exporters.webdataset_discovery import matches_factory_range

    root = Path(processed_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"BuildAI processed root not found: {root}")

    seq_folders = []
    for world_res in sorted(root.rglob("world_space_res.pth")):
        seq_folder = world_res.parent.resolve()
        parent = seq_folder.parent
        if parent.name != "outputs":
            continue
        factory_dir = parent.parent
        if not factory_dir.name.startswith("factory"):
            continue
        if not matches_factory_range(str(factory_dir), factory_range):
            continue
        seq_folders.append(seq_folder)

    legacy_index = {}
    for episode_index, seq_folder in enumerate(seq_folders):
        clip_id = seq_folder.name
        legacy_index[episode_index] = {
            "clip_id": clip_id,
            "episode_id": clip_id,
            "seq_folder": str(seq_folder),
            "source_id": "buildai",
            "split": "unknown",
            "legacy_episode_index": episode_index,
        }
    if not legacy_index:
        raise RuntimeError(
            "Failed to auto-discover any BuildAI seq_folder under "
            f"{root}. Expected paths like factoryXXX/outputs/<clip_id>/world_space_res.pth"
        )
    return legacy_index


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
    legacy_episodes: dict[int, dict] | None,
    feature_cache_dir: str | None,
    output_dir: str,
    buildai_root: str,
):
    global _WORKER_CLIP_INDEX, _WORKER_LEGACY_EPISODES, _WORKER_FEATURE_CACHE_DIR, _WORKER_DEVICE
    global _WORKER_MANO_RIGHT, _WORKER_MANO_LEFT, _WORKER_EPISODE_CACHE
    global _WORKER_MANO_DIR, _WORKER_OUTPUT_DIR, _WORKER_BUILDAI_ROOT

    from multiprocessing import current_process
    import torch

    identity = current_process()._identity
    worker_idx = identity[0] - 1 if identity else 0
    device_str = device_specs[worker_idx % len(device_specs)]
    _WORKER_DEVICE = torch.device(device_str)
    _WORKER_CLIP_INDEX = clip_index
    _WORKER_LEGACY_EPISODES = legacy_episodes or {}
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
        candidates = sorted(Path(_WORKER_BUILDAI_ROOT).rglob(f"outputs/{clip_id}"))
        candidates = [path.resolve() for path in candidates if path.is_dir()]
        if len(candidates) == 1:
            seq_folder = candidates[0]
        elif len(candidates) > 1:
            raise FileNotFoundError(
                f"Multiple BuildAI seq_folder candidates found for {clip_id}: {candidates[:3]}"
            )
        else:
            raise FileNotFoundError(
                f"BuildAI seq_folder not found for {clip_id} under {_WORKER_BUILDAI_ROOT}"
            )
    return {
        "clip_id": clip_id,
        "episode_id": clip_id,
        "seq_folder": str(seq_folder.resolve()),
        "source_id": "buildai",
        "split": "unknown",
    }


def _build_legacy_buildai_clip_info(sample_key: str, clip_id: str, meta: dict | None) -> dict:
    episode_index = None
    if meta is not None and meta.get("episode_index") is not None:
        episode_index = int(meta["episode_index"])
    else:
        match = LEGACY_BUILDAI_EP_RE.match(clip_id)
        if match:
            episode_index = int(match.group(1))
        else:
            episode_index = int(parse_episode_index(sample_key))

    clip_info = _WORKER_LEGACY_EPISODES.get(episode_index)
    if clip_info is None:
        raise KeyError(
            "Legacy BuildAI episode index "
            f"{episode_index} was not found. Pass --legacy_buildai_input_dir "
            "and, if needed, --legacy_episode_list / --legacy_factory_range / --legacy_episode_cache "
            "to reconstruct the old builder ordering. "
            f"Currently loaded legacy episode count: {len(_WORKER_LEGACY_EPISODES)}"
        )
    return clip_info


def _resolve_clip_info(sample_key: str, clip_id: str, meta: dict | None):
    clip_info = _WORKER_CLIP_INDEX.get(clip_id)
    if clip_info is not None:
        return clip_info

    if BUILDAI_CLIP_RE.match(clip_id):
        clip_info = _build_buildai_clip_info(clip_id)
    else:
        clip_info = _build_legacy_buildai_clip_info(sample_key, clip_id, meta)
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
            clip_info = _resolve_clip_info(sample["key"], clip_id, meta)

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


def source_contains_legacy_buildai_keys(shard_paths: list[str], sample_limit: int = 32) -> bool:
    checked = 0
    for shard_path in shard_paths:
        for sample in iter_shard_samples(shard_path):
            meta = None
            try:
                meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            except Exception:
                meta = None
            clip_id = _sample_clip_id(sample, meta)
            checked += 1
            if LEGACY_BUILDAI_EP_RE.match(clip_id):
                return True
            if checked >= sample_limit:
                return False
    return False


def build_report(
    source_dir: Path,
    output_dir: Path,
    buildai_processed_root: Path,
    legacy_episode_source: str | None,
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
    if legacy_episode_source:
        report["legacy_episode_source"] = legacy_episode_source
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
    legacy_episodes = {}
    legacy_episode_source = None
    legacy_episode_cache = args.legacy_episode_cache
    if legacy_episode_cache:
        legacy_episode_cache = str(Path(legacy_episode_cache).resolve())

    if args.legacy_buildai_input_dir or legacy_episode_cache:
        legacy_input_dir = str(Path(args.legacy_buildai_input_dir or buildai_processed_root).resolve())
        legacy_episodes = build_legacy_episode_index(
            legacy_input_dir,
            episode_list=args.legacy_episode_list,
            factory_range=args.legacy_factory_range,
            cache_file=legacy_episode_cache,
        )
        if legacy_episodes:
            if legacy_episode_cache:
                legacy_episode_source = legacy_episode_cache
            else:
                legacy_episode_source = legacy_input_dir

    if not legacy_episodes:
        legacy_episodes = build_legacy_episode_index_from_processed_root(
            str(buildai_processed_root),
            factory_range=args.legacy_factory_range,
        )
        legacy_episode_source = f"{buildai_processed_root} [auto-scan]"

    shard_paths = list(iter_shard_paths(str(source_dir)))
    if not shard_paths:
        raise RuntimeError(f"No shard tar files found in {source_dir}")
    if source_contains_legacy_buildai_keys(shard_paths) and not legacy_episodes:
        raise RuntimeError(
            "Detected legacy buildai_epXXXX shard keys, but no legacy BuildAI episodes "
            f"were discovered under {buildai_processed_root}. "
            "Check --buildai_processed_root or pass --legacy_buildai_input_dir / --legacy_episode_cache."
        )

    mano_device_obj = torch.device(args.mano_device if torch.cuda.is_available() else "cpu")
    device_specs = normalize_mano_devices(str(mano_device_obj), args.mano_gpus if mano_device_obj.type == "cuda" else None)
    if mano_device_obj.type == "cuda":
        args.workers = min(args.workers, len(device_specs))

    if args.workers <= 1:
        _worker_init(
            device_specs,
            args.mano_dir,
            clip_index,
            legacy_episodes,
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
                legacy_episodes,
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

    report = build_report(
        source_dir,
        output_dir,
        buildai_processed_root,
        legacy_episode_source,
        shard_results,
        feature_cache_dir,
    )
    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
