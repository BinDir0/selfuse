#!/usr/bin/env python3
"""Rewrite only camera-related slices inside existing WebDataset lowdim payloads."""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from multiprocessing import get_context
from pathlib import Path

import joblib
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

from lib.pipeline.exporters.webdataset_geometry import (  # noqa: E402
    interpolate_extrinsics,
    normalize_slam_keyframes,
    quat_to_4x4,
)
from lib.pipeline.exporters.webdataset_rewriter import (  # noqa: E402
    iter_shard_paths,
    iter_shard_samples,
    parse_episode_index,
    validate_sample_record,
    write_sample_to_tar,
)
from lib.pipeline.quality_metrics import (  # noqa: E402
    EXTRINSIC_SLICE,
    LOWDIM_SIZE,
    decode_lowdim,
    parse_frame_index,
)


INTRINSIC_SLICE = slice(112, 116)
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))

_WORKER_OUTPUT_DIR = None
_WORKER_CLIP_INDEX = None
_WORKER_EPISODE_INDEX = None
_WORKER_CAMERA_MODE = None
_WORKER_CAMERA_CACHE = {}


def _log(message: str):
    print(message, flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Rewrite only camera slices in WebDataset lowdim.npy payloads")
    parser.add_argument("--source_shard_dir", required=True, help="Source directory containing shard tar files")
    parser.add_argument("--output_dir", required=True, help="Output directory for rewritten shards")
    parser.add_argument(
        "--processed_root",
        required=True,
        help=(
            "Processed root used to resolve source episodes. Supports both "
            "factory_*/worker_*/processed/<clip_id> and factory*/outputs/<clip_id> layouts."
        ),
    )
    parser.add_argument("--shard_start", type=int, default=0, help="Inclusive shard index in sorted shard order")
    parser.add_argument("--shard_end", type=int, default=None, help="Exclusive shard index in sorted shard order")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel shard workers")
    parser.add_argument(
        "--scan_processed_root",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force an additional processed_root scan even when _vla_episodes_cache.json is available",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip shards whose rewritten output tar already exists",
    )
    parser.add_argument(
        "--camera_mode",
        choices=("direct-traj-only", "direct-traj-plus-image-center"),
        default="direct-traj-only",
        help=(
            "direct-traj-only: use raw traj when traj length matches frame count, leave stored intrinsic untouched; "
            "direct-traj-plus-image-center: additionally replace cx,cy with image center w/2,h/2."
        ),
    )
    parser.add_argument("--report_out", default=None, help="Optional JSON report path")
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


def _load_episode_cache(processed_root: Path):
    cache_path = processed_root / "_vla_episodes_cache.json"
    if not cache_path.is_file():
        _log(f"Episode cache not found under {processed_root}; will scan processed_root directly.")
        return [], None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _log(f"Episode cache is unreadable: {cache_path}; will scan processed_root directly.")
        return [], None
    if not isinstance(payload, list):
        _log(f"Episode cache has unexpected format: {cache_path}; will scan processed_root directly.")
        return [], None
    _log(f"Loaded episode cache: {cache_path} ({len(payload)} entries)")
    return payload, str(cache_path)


def _iter_seq_folders(processed_root: Path):
    for seq_folder in sorted(processed_root.glob("factory_*/*/processed/*")):
        if seq_folder.is_dir() and (seq_folder / "world_space_res.pth").is_file():
            yield seq_folder.resolve()
    for seq_folder in sorted(processed_root.glob("factory*/outputs/*")):
        if seq_folder.is_dir() and (seq_folder / "world_space_res.pth").is_file():
            yield seq_folder.resolve()


def build_episode_indices(processed_root: Path, *, scan_processed_root: bool):
    clip_index = {}
    episode_index = {}
    cached_episodes, cache_path = _load_episode_cache(processed_root)

    _log("Building clip / episode indices...")
    for idx, ep in enumerate(cached_episodes):
        crop_dir = ep.get("crop_dir")
        episode_id = ep.get("episode_id")
        if not crop_dir or not episode_id:
            continue
        record = {
            "clip_id": str(episode_id),
            "episode_id": str(episode_id),
            "seq_folder": str(Path(crop_dir).resolve()),
        }
        clip_index[record["clip_id"]] = record
        ep_idx = ep.get("episode_index", idx)
        try:
            episode_index[int(ep_idx)] = record
        except (TypeError, ValueError):
            pass

    scanned_seq_folders = 0
    should_scan = scan_processed_root or not cached_episodes
    if should_scan:
        if cached_episodes and scan_processed_root:
            _log("scan_processed_root enabled; supplementing cache with seq_folder scan.")
        else:
            _log("No usable episode cache found; scanning processed_root for seq folders.")
        for seq_folder in tqdm(_iter_seq_folders(processed_root), desc="Index seq folders", unit="seq", dynamic_ncols=True):
            record = {
                "clip_id": seq_folder.name,
                "episode_id": seq_folder.name,
                "seq_folder": str(seq_folder),
            }
            clip_index.setdefault(record["clip_id"], record)
            scanned_seq_folders += 1
            if scanned_seq_folders <= 5 or scanned_seq_folders % 5000 == 0:
                _log(
                    f"Index progress: scanned_seq_folders={scanned_seq_folders} "
                    f"clip_index={len(clip_index)} episode_index={len(episode_index)}"
                )
    else:
        _log("Using cached episode index only; skipping processed_root scan.")

    _log(
        f"Finished index build: scanned_seq_folders={scanned_seq_folders} "
        f"clip_index={len(clip_index)} episode_index={len(episode_index)}"
    )
    return clip_index, episode_index, cache_path


def _resolve_clip_info(sample_key: str, clip_id: str, meta: dict | None):
    clip_info = _WORKER_CLIP_INDEX.get(clip_id)
    if clip_info is not None:
        return clip_info

    episode_idx = None
    if meta is not None and meta.get("episode_index") is not None:
        try:
            episode_idx = int(meta["episode_index"])
        except (TypeError, ValueError):
            episode_idx = None
    if episode_idx is None:
        try:
            episode_idx = int(parse_episode_index(sample_key))
        except Exception:
            episode_idx = None

    if episode_idx is not None:
        clip_info = _WORKER_EPISODE_INDEX.get(episode_idx)
        if clip_info is not None:
            return clip_info

    raise KeyError(
        f"Failed to resolve source episode for sample {sample_key} (clip_id={clip_id!r}, episode_index={episode_idx!r})"
    )


def _load_image_center(extracted_dir: Path):
    try:
        from lib.pipeline.exporters.webdataset_discovery import load_or_build_frame_index
        import cv2

        frame_index = load_or_build_frame_index(str(extracted_dir), rescan=False)
        if not frame_index:
            return None
        first_frame_path = frame_index.get(min(frame_index))
        if not first_frame_path or not os.path.exists(first_frame_path):
            return None
        image = cv2.imread(first_frame_path, cv2.IMREAD_COLOR)
        if image is None:
            return None
        height, width = image.shape[:2]
        return np.array([float(width) / 2.0, float(height) / 2.0], dtype=np.float32)
    except Exception:
        return None


def _load_camera_sequence(seq_folder: str, *, camera_mode: str):
    if seq_folder in _WORKER_CAMERA_CACHE:
        return _WORKER_CAMERA_CACHE[seq_folder]

    seq_path = Path(seq_folder)
    world_res_path = seq_path / "world_space_res.pth"
    slam_dir = seq_path / "SLAM"
    pred_trans, *_ = joblib.load(world_res_path)
    num_frames = int(np.asarray(pred_trans).shape[1])

    extrinsics = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
    intrinsic = np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float32)
    slam_files = sorted(slam_dir.glob("hawor_slam_w_scale_*.npz")) if slam_dir.is_dir() else []

    if slam_files:
        slam_data = np.load(str(slam_files[0]), allow_pickle=True)
        tstamps = np.asarray(slam_data["tstamp"], dtype=np.int64)
        traj = np.asarray(slam_data["traj"], dtype=np.float32)
        scale = float(slam_data["scale"])
        img_focal = float(slam_data["img_focal"])
        img_center = np.asarray(slam_data["img_center"], dtype=np.float32)

        intrinsic = np.array([img_focal, img_focal, float(img_center[0]), float(img_center[1])], dtype=np.float32)
        if camera_mode == "direct-traj-plus-image-center":
            image_center = _load_image_center(seq_path / "extracted_images")
            if image_center is not None:
                intrinsic[2:] = image_center

        if traj.shape[0] == num_frames:
            c2w = np.stack([quat_to_4x4(traj_row, scale) for traj_row in traj], axis=0)
            extrinsics = np.linalg.inv(c2w).astype(np.float32)
        else:
            tstamps, traj = normalize_slam_keyframes(tstamps, traj)
            if len(tstamps) > 0:
                extrinsics = interpolate_extrinsics(tstamps, traj, scale, num_frames)

    payload = {
        "num_frames": num_frames,
        "extrinsics": extrinsics,
        "intrinsic": intrinsic,
    }
    _WORKER_CAMERA_CACHE[seq_folder] = payload
    return payload


def _worker_init(output_dir: str, clip_index: dict, episode_index: dict, camera_mode: str):
    global _WORKER_OUTPUT_DIR, _WORKER_CLIP_INDEX, _WORKER_EPISODE_INDEX, _WORKER_CAMERA_MODE, _WORKER_CAMERA_CACHE
    _WORKER_OUTPUT_DIR = output_dir
    _WORKER_CLIP_INDEX = clip_index
    _WORKER_EPISODE_INDEX = episode_index
    _WORKER_CAMERA_MODE = camera_mode
    _WORKER_CAMERA_CACHE = {}


def _append_skip(skip_details: list[dict], *, sample_key: str | None, reason: str, error: Exception | str):
    if len(skip_details) >= 32:
        return
    skip_details.append(
        {
            "sample_key": sample_key,
            "reason": reason,
            "error": str(error),
        }
    )


def process_shard(shard_path: str) -> dict:
    shard_name = os.path.basename(shard_path)
    output_path = os.path.join(_WORKER_OUTPUT_DIR, shard_name)
    tmp_path = f"{output_path}.tmp"
    frames_rewritten = 0
    clips_touched = set()
    skipped_samples = 0
    skip_details = []
    tar_writer = None

    try:
        for sample in iter_shard_samples(shard_path):
            try:
                validate_sample_record(sample)
                meta = json.loads(sample["meta_bytes"].decode("utf-8"))
                clip_id = _sample_clip_id(sample, meta)
                clip_info = _resolve_clip_info(sample["key"], clip_id, meta)
                frame_idx = parse_frame_index(sample["key"])

                camera = _load_camera_sequence(clip_info["seq_folder"], camera_mode=_WORKER_CAMERA_MODE)
                if frame_idx >= int(camera["num_frames"]):
                    raise IndexError(
                        f"Sample {sample['key']} requests frame {frame_idx}, "
                        f"but source episode only has {camera['num_frames']} frames"
                    )

                lowdim = decode_lowdim(sample["lowdim_bytes"])
                if lowdim.shape != (LOWDIM_SIZE,):
                    raise ValueError(f"Unexpected lowdim shape for {sample['key']}: {lowdim.shape}")
                lowdim[EXTRINSIC_SLICE] = camera["extrinsics"][frame_idx].reshape(-1)
                if _WORKER_CAMERA_MODE == "direct-traj-plus-image-center":
                    lowdim[INTRINSIC_SLICE] = camera["intrinsic"]
                lowdim_bytes = _encode_lowdim(lowdim)

                if tar_writer is None:
                    os.makedirs(_WORKER_OUTPUT_DIR, exist_ok=True)
                    tar_writer = tarfile.open(tmp_path, "w")

                write_sample_to_tar(
                    tar_writer,
                    sample["key"],
                    sample["image_bytes"],
                    lowdim_bytes,
                    sample["meta_bytes"],
                    mano_bytes=sample.get("mano_bytes"),
                )
                frames_rewritten += 1
                clips_touched.add(clip_id)
            except Exception as error:
                skipped_samples += 1
                _append_skip(
                    skip_details,
                    sample_key=sample.get("key"),
                    reason="sample_error",
                    error=error,
                )
                if skipped_samples <= 5 or skipped_samples % 100 == 0:
                    _log(
                        f"Skip sample in {shard_name}: key={sample.get('key')} "
                        f"skipped={skipped_samples} error={error}"
                    )
                continue
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
        "skipped_samples": skipped_samples,
        "skip_details": skip_details,
        "shard_error": None,
    }


def _worker_process_shard(shard_path: str) -> dict:
    try:
        return process_shard(shard_path)
    except Exception as error:
        shard_name = os.path.basename(shard_path)
        _log(f"Skip shard {shard_name}: {error}")
        return {
            "shard_name": shard_name,
            "output_path": os.path.join(_WORKER_OUTPUT_DIR, shard_name),
            "frames_rewritten": 0,
            "clips_touched": 0,
            "shard_written": 0,
            "skipped_samples": 0,
            "skip_details": [],
            "shard_error": str(error),
        }


def select_shard_paths(shard_paths: list[str], output_dir: Path, shard_start: int, shard_end: int | None, resume: bool):
    total = len(shard_paths)
    if shard_start < 0:
        raise ValueError("--shard_start must be >= 0")
    if shard_end is not None and shard_end < 0:
        raise ValueError("--shard_end must be >= 0")
    if shard_end is not None and shard_end < shard_start:
        raise ValueError("--shard_end must be >= --shard_start")
    if shard_start > total:
        raise ValueError(f"--shard_start ({shard_start}) exceeds shard count ({total})")

    selected_end = total if shard_end is None else min(shard_end, total)
    selected = shard_paths[shard_start:selected_end]
    pending = []
    reused = []
    for shard_path in selected:
        output_path = output_dir / os.path.basename(shard_path)
        if resume and output_path.is_file():
            reused.append(shard_path)
        else:
            pending.append(shard_path)

    print(
        "Shard selection:"
        f" total={total}"
        f" range=[{shard_start}, {selected_end})"
        f" selected={len(selected)}"
        f" reused={len(reused)}"
        f" pending={len(pending)}"
        ,
        flush=True,
    )
    return pending, reused, selected_end


def build_report(
    source_dir: Path,
    output_dir: Path,
    processed_root: Path,
    shard_results: list[dict],
    *,
    shard_start: int,
    shard_end: int,
    selected_shards: int,
    reused_shards: int,
    camera_mode: str,
    clip_index_size: int,
    episode_index_size: int,
    cache_path: str | None,
):
    report = {
        "source_shard_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "processed_root": str(processed_root.resolve()),
        "camera_mode": camera_mode,
        "shard_range": {
            "start": int(shard_start),
            "end": int(shard_end),
        },
        "selected_shards": int(selected_shards),
        "reused_shards": int(reused_shards),
        "shards_total": len(shard_results),
        "shards_written": int(sum(item["shard_written"] for item in shard_results)),
        "frames_rewritten": int(sum(item["frames_rewritten"] for item in shard_results)),
        "clips_touched": int(sum(item["clips_touched"] for item in shard_results)),
        "skipped_samples": int(sum(item.get("skipped_samples", 0) for item in shard_results)),
        "failed_shards": int(sum(1 for item in shard_results if item.get("shard_error"))),
        "clip_index_size": int(clip_index_size),
        "episode_index_size": int(episode_index_size),
        "episode_cache": cache_path,
        "shards": {
            item["shard_name"]: {
                "frames_rewritten": item["frames_rewritten"],
                "clips_touched": item["clips_touched"],
                "skipped_samples": int(item.get("skipped_samples", 0)),
                "shard_error": item.get("shard_error"),
                "skip_details": item.get("skip_details", []),
            }
            for item in shard_results
        },
    }
    return report


def main():
    args = build_parser().parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    source_dir = Path(args.source_shard_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source shard dir not found: {source_dir}")

    output_dir = Path(args.output_dir)
    validate_io_dirs(source_dir, output_dir)

    processed_root = Path(args.processed_root).resolve()
    if not processed_root.is_dir():
        raise FileNotFoundError(f"Processed root not found: {processed_root}")

    _log(
        "Rewrite camera config:"
        f" source={source_dir}"
        f" output={output_dir}"
        f" processed_root={processed_root}"
        f" camera_mode={args.camera_mode}"
        f" workers={args.workers}"
        f" scan_processed_root={args.scan_processed_root}"
    )
    clip_index, episode_index, cache_path = build_episode_indices(
        processed_root,
        scan_processed_root=args.scan_processed_root,
    )
    if not clip_index and not episode_index:
        raise RuntimeError(f"Failed to discover source episodes under {processed_root}")

    _log(f"Scanning shard list from {source_dir} ...")
    shard_paths = list(iter_shard_paths(str(source_dir)))
    if not shard_paths:
        raise RuntimeError(f"No shard tar files found in {source_dir}")
    _log(f"Discovered {len(shard_paths)} shard(s)")

    shard_paths, reused_shards, selected_end = select_shard_paths(
        shard_paths,
        output_dir,
        args.shard_start,
        args.shard_end,
        args.resume,
    )

    if not shard_paths:
        _log("No pending shards to rewrite.")

    if args.workers <= 1:
        _worker_init(str(output_dir), clip_index, episode_index, args.camera_mode)
        shard_results = []
        for shard_path in tqdm(shard_paths, desc="Rewrite camera", unit="shard", dynamic_ncols=True):
            result = process_shard(shard_path)
            shard_results.append(result)
            _log(
                f"Finished {result['shard_name']}: "
                f"frames={result['frames_rewritten']} clips={result['clips_touched']} "
                f"skipped_samples={result.get('skipped_samples', 0)}"
            )
    else:
        _log(f"Starting worker pool: workers={args.workers}")
        with get_context().Pool(
            args.workers,
            initializer=_worker_init,
            initargs=(str(output_dir), clip_index, episode_index, args.camera_mode),
        ) as pool:
            shard_results = []
            for result in tqdm(
                pool.imap_unordered(_worker_process_shard, shard_paths),
                total=len(shard_paths),
                desc="Rewrite camera",
                unit="shard",
                dynamic_ncols=True,
            ):
                shard_results.append(result)
                if len(shard_results) <= 5 or len(shard_results) % 10 == 0 or len(shard_results) == len(shard_paths):
                    _log(
                        f"Rewrite progress: done={len(shard_results)}/{len(shard_paths)} "
                        f"latest={result['shard_name']} frames={result['frames_rewritten']} "
                        f"clips={result['clips_touched']} skipped_samples={result.get('skipped_samples', 0)} "
                        f"shard_error={result.get('shard_error')}"
                    )

    report = build_report(
        source_dir,
        output_dir,
        processed_root,
        shard_results,
        shard_start=args.shard_start,
        shard_end=selected_end,
        selected_shards=len(shard_paths) + len(reused_shards),
        reused_shards=len(reused_shards),
        camera_mode=args.camera_mode,
        clip_index_size=len(clip_index),
        episode_index_size=len(episode_index),
        cache_path=cache_path,
    )
    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(f"Wrote report to {report_path}")
    _log("Rewrite complete.")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
