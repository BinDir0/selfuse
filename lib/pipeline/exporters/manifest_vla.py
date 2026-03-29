"""Build VLA WebDataset from clip manifest + HaWoR stage outputs."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from multiprocessing import current_process, get_context
from pathlib import Path

import joblib
import numpy as np
import torch
from tqdm import tqdm

from lib.pipeline.annotation_protocol import load_clip_annotation
from lib.pipeline.clip_manifest import ClipManifestRecord, load_clip_manifest
from lib.pipeline.frame_sources import read_frame_bytes_from_descriptor
from lib.pipeline.exporters.webdataset_features import (
    _build_lowdim_features,
    _compute_hand_state,
    _compute_presence_per_frame,
    _compute_wrist_state,
    _load_episode_camera_features,
    _load_world_space_prediction,
    build_mano_models,
)
from lib.pipeline.exporters.webdataset_workers import normalize_mano_devices


_worker_mano_right = None
_worker_mano_left = None
_worker_device = None
_worker_feature_cache_dir = None
_worker_episode_cache = {}
_worker_shard_fd_cache = {}
_worker_shard_tar_cache = {}


def _feature_cache_path(seq_folder: str, feature_cache_dir: str) -> str:
    digest = hashlib.md5(seq_folder.encode("utf-8")).hexdigest()
    return os.path.join(feature_cache_dir, f"{digest}.joblib")


def _load_cached_features(seq_folder: str, frame_count: int, feature_cache_dir: str):
    if not feature_cache_dir:
        return None
    path = _feature_cache_path(seq_folder, feature_cache_dir)
    if not os.path.exists(path):
        return None
    try:
        payload = joblib.load(path)
    except Exception:
        return None
    if payload.get("seq_folder") != seq_folder or payload.get("frame_count") != frame_count:
        return None
    return {
        "frame_count": payload["frame_count"],
        "lowdim_all": payload["lowdim_all"],
        "presence_per_frame": payload["presence_per_frame"],
    }


def _write_cached_features(seq_folder: str, feature_cache_dir: str, episode_data: dict):
    if not feature_cache_dir:
        return
    os.makedirs(feature_cache_dir, exist_ok=True)
    path = _feature_cache_path(seq_folder, feature_cache_dir)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    payload = {
        "seq_folder": seq_folder,
        "frame_count": episode_data["frame_count"],
        "lowdim_all": episode_data["lowdim_all"],
        "presence_per_frame": episode_data["presence_per_frame"].astype(np.uint8),
    }
    try:
        joblib.dump(payload, tmp_path)
        os.replace(tmp_path, path)
    except OSError:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_descriptor_episode_features(ep: dict, mano_right, mano_left, device, feature_cache_dir: str | None):
    seq_folder = ep["seq_folder"]
    frame_count = int(ep.get("num_valid_frames", ep["frame_end"] - ep.get("frame_start", 0)))
    cached = _load_cached_features(seq_folder, frame_count, feature_cache_dir) if feature_cache_dir else None
    if cached is not None:
        return cached

    prediction = _load_world_space_prediction({"episode_id": ep["episode_id"]}, os.path.join(seq_folder, "world_space_res.pth"))
    if prediction is None:
        return None

    pred_trans = prediction["pred_trans"]
    pred_rot = prediction["pred_rot"]
    pred_hand_pose = prediction["pred_hand_pose"]
    pred_betas = prediction["pred_betas"]
    pred_valid = prediction["pred_valid"]
    num_frames = int(pred_trans.shape[1])
    frame_count = min(frame_count, num_frames)
    if frame_count <= 0:
        return None

    wrist_state = _compute_wrist_state(pred_trans, pred_rot)[:frame_count]
    hand_state = _compute_hand_state(
        pred_trans,
        pred_rot,
        pred_hand_pose,
        pred_betas,
        mano_right,
        mano_left,
        device,
    )[:frame_count]
    camera_ep = {"crop_dir": seq_folder, "episode_id": ep["episode_id"]}
    extrinsics, intrinsic = _load_episode_camera_features(camera_ep, num_frames)
    presence_per_frame = _compute_presence_per_frame(pred_valid, num_frames)[:frame_count]
    lowdim_all = _build_lowdim_features(
        wrist_state,
        hand_state,
        extrinsics[:frame_count],
        intrinsic,
    )

    episode_data = {
        "frame_count": frame_count,
        "lowdim_all": lowdim_all[:frame_count],
        "presence_per_frame": presence_per_frame[:frame_count],
    }
    _write_cached_features(seq_folder, feature_cache_dir, episode_data)
    return episode_data


def plan_manifest_shards(episodes: list[dict], frames_per_shard: int, output_dir: str):
    tasks = []
    shard_slices = []
    shard_frame_count = 0
    shard_idx = 0

    def flush_current():
        nonlocal shard_slices, shard_frame_count, shard_idx
        if not shard_slices:
            return
        output_path = os.path.join(output_dir, f"shard-{shard_idx:06d}.tar")
        tasks.append(
            {
                "shard_idx": shard_idx,
                "output_path": output_path,
                "tmp_path": f"{output_path}.tmp",
                "frame_count": shard_frame_count,
                "episode_slices": shard_slices,
            }
        )
        shard_idx += 1
        shard_slices = []
        shard_frame_count = 0

    for ep in episodes:
        num_frames = ep["num_valid_frames"]
        if shard_slices and shard_frame_count + num_frames > frames_per_shard:
            flush_current()
        shard_slices.append(
            {
                "seq_folder": ep["seq_folder"],
                "episode_id": ep["episode_id"],
                "episode_index": ep["episode_index"],
                "clip_id": ep["clip_id"],
                "source_id": ep["source_id"],
                "split": ep["split"],
                "instruction": list(ep.get("instruction", [])),
                "instruction_num": int(ep.get("instruction_num", 0)),
                "language": ep.get("language"),
                "descriptor": ep["descriptor"],
                "frame_start": 0,
                "frame_end": num_frames,
            }
        )
        shard_frame_count += num_frames
        if shard_frame_count >= frames_per_shard:
            flush_current()

    flush_current()
    return tasks


def repeat_manifest_episodes(episodes: list[dict], repeat_count: int) -> list[dict]:
    repeated = []
    for repeat_idx in range(repeat_count):
        for ep in episodes:
            ep_copy = dict(ep)
            ep_copy["source_episode_index"] = ep["episode_index"]
            if repeat_count > 1:
                ep_copy["repeat_index"] = repeat_idx
            ep_copy["episode_index"] = len(repeated)
            repeated.append(ep_copy)
    return repeated


def add_sample_bytes_to_tar(tar_writer, key: str, image_bytes: bytes, lowdim, meta: dict):
    img_info = tarfile.TarInfo(name=f"{key}.image.jpg")
    img_info.size = len(image_bytes)
    tar_writer.addfile(img_info, io.BytesIO(image_bytes))

    lowdim_buf = io.BytesIO()
    np.save(lowdim_buf, lowdim)
    lowdim_bytes = lowdim_buf.getvalue()
    lowdim_info = tarfile.TarInfo(name=f"{key}.lowdim.npy")
    lowdim_info.size = len(lowdim_bytes)
    tar_writer.addfile(lowdim_info, io.BytesIO(lowdim_bytes))

    meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    meta_info = tarfile.TarInfo(name=f"{key}.meta.json")
    meta_info.size = len(meta_bytes)
    tar_writer.addfile(meta_info, io.BytesIO(meta_bytes))


def _worker_init(device_specs, mano_dir, feature_cache_dir):
    global _worker_mano_right, _worker_mano_left, _worker_device
    global _worker_feature_cache_dir, _worker_episode_cache
    global _worker_shard_fd_cache, _worker_shard_tar_cache

    identity = current_process()._identity
    worker_idx = identity[0] - 1 if identity else 0
    device_str = device_specs[worker_idx % len(device_specs)]
    _worker_device = torch.device(device_str)
    _worker_mano_right, _worker_mano_left = build_mano_models(_worker_device, mano_dir=mano_dir)
    _worker_mano_right.eval()
    _worker_mano_left.eval()
    _worker_feature_cache_dir = feature_cache_dir
    _worker_episode_cache = {}
    _worker_shard_fd_cache = {}
    _worker_shard_tar_cache = {}


def _worker_process_shard(task):
    frames_written = 0
    skipped_episodes = 0
    touched_episodes = set()
    tar_writer = None

    try:
        for episode_slice in task["episode_slices"]:
            cache_key = episode_slice["seq_folder"]
            if cache_key not in _worker_episode_cache:
                _worker_episode_cache[cache_key] = load_descriptor_episode_features(
                    episode_slice,
                    _worker_mano_right,
                    _worker_mano_left,
                    _worker_device,
                    _worker_feature_cache_dir,
                )

            episode_data = _worker_episode_cache[cache_key]
            if episode_data is None:
                skipped_episodes += 1
                continue

            descriptor = episode_slice["descriptor"]
            for frame_idx in range(episode_slice["frame_start"], episode_slice["frame_end"]):
                image_bytes = read_frame_bytes_from_descriptor(
                    descriptor,
                    frame_idx,
                    shard_fd_cache=_worker_shard_fd_cache,
                    shard_tar_cache=_worker_shard_tar_cache,
                )
                meta = {
                    "dataset_name": episode_slice["source_id"],
                    "clip_id": episode_slice["clip_id"],
                    "episode_index": episode_slice["episode_index"],
                    "split": episode_slice["split"],
                    "instruction": list(episode_slice.get("instruction", [])),
                    "instruction_num": int(episode_slice.get("instruction_num", 0)),
                    "language": episode_slice.get("language"),
                    "presence": int(episode_data["presence_per_frame"][frame_idx]),
                }
                key = f"{episode_slice['clip_id']}_f{frame_idx:06d}"
                if tar_writer is None:
                    os.makedirs(os.path.dirname(task["output_path"]), exist_ok=True)
                    tar_writer = tarfile.open(task["tmp_path"], "w")
                add_sample_bytes_to_tar(
                    tar_writer,
                    key,
                    image_bytes,
                    episode_data["lowdim_all"][frame_idx],
                    meta,
                )
                frames_written += 1

            touched_episodes.add(cache_key)
    except Exception:
        if tar_writer is not None:
            tar_writer.close()
        if os.path.exists(task["tmp_path"]):
            os.remove(task["tmp_path"])
        raise

    if tar_writer is not None:
        tar_writer.close()

    if frames_written == 0:
        if os.path.exists(task["tmp_path"]):
            os.remove(task["tmp_path"])
    else:
        os.replace(task["tmp_path"], task["output_path"])

    return {
        "shard_idx": task["shard_idx"],
        "frames_written": frames_written,
        "episodes_written": len(touched_episodes),
        "skipped_episodes": skipped_episodes,
        "output_path": task["output_path"],
    }


def _prepare_manifest_episode(record: ClipManifestRecord, require_annotation: bool, annotation_root: str | None):
    seq_folder = Path(record.descriptor.seq_folder)
    world_res_path = seq_folder / "world_space_res.pth"
    if not world_res_path.exists():
        return None, "missing_world_res"

    try:
        pred_trans, *_ = joblib.load(world_res_path)
    except Exception:
        return None, "invalid_world_res"

    num_frames = min(int(np.asarray(pred_trans).shape[1]), record.descriptor.frame_count)
    if num_frames <= 0:
        return None, "empty_frames"

    language = None
    instruction = []
    if annotation_root:
        annotation, error_code, _ = load_clip_annotation(annotation_root, record.clip_id)
        if annotation is None:
            if require_annotation:
                return None, error_code
        else:
            instruction = annotation.instruction
            language = annotation.language

    return {
        "clip_id": record.clip_id,
        "episode_id": record.clip_id,
        "seq_folder": str(seq_folder),
        "source_id": record.source_id,
        "split": record.split,
        "descriptor": record.descriptor,
        "num_valid_frames": num_frames,
        "instruction": instruction,
        "instruction_num": len(instruction),
        "language": language,
    }, None


def prepare_manifest_episodes(
    manifest_path: str,
    *,
    annotation_root: str | None,
    require_annotation: bool,
    max_episodes: int | None,
    preprocess_workers: int,
):
    records = load_clip_manifest(manifest_path)
    if max_episodes is not None:
        records = records[:max_episodes]

    stats = {
        "kept": 0,
        "missing_world_res": 0,
        "invalid_world_res": 0,
        "empty_frames": 0,
        "missing_annotation": 0,
        "invalid_json": 0,
        "invalid_status": 0,
        "empty_instruction": 0,
    }

    if preprocess_workers <= 1:
        iterator = (_prepare_manifest_episode(record, require_annotation, annotation_root) for record in records)
    else:
        mp_context = get_context()
        pool = mp_context.Pool(preprocess_workers)
        iterator = pool.imap(
            _prepare_manifest_episode_star,
            ((record, require_annotation, annotation_root) for record in records),
            chunksize=32,
        )

    episodes = []
    try:
        for episode, error_code in tqdm(iterator, total=len(records), desc="Manifest episodes"):
            if episode is None:
                stats[error_code] = stats.get(error_code, 0) + 1
                continue
            episode["episode_index"] = len(episodes)
            episodes.append(episode)
            stats["kept"] += 1
    finally:
        if preprocess_workers > 1:
            pool.close()
            pool.join()

    return episodes, stats


def _prepare_manifest_episode_star(args):
    return _prepare_manifest_episode(*args)


def run_manifest_build(
    *,
    manifest_path: str,
    output_dir: str,
    annotation_root: str | None,
    require_annotation: bool,
    max_episodes: int | None,
    repeat_episodes: int,
    preprocess_workers: int,
    writer_workers: int,
    frames_per_shard: int,
    mano_device: str,
    mano_gpus: str | None,
    mano_dir: str | None,
):
    episodes, prepare_stats = prepare_manifest_episodes(
        manifest_path,
        annotation_root=annotation_root,
        require_annotation=require_annotation,
        max_episodes=max_episodes,
        preprocess_workers=preprocess_workers,
    )
    if not episodes:
        raise RuntimeError(f"No valid manifest episodes found: {prepare_stats}")

    repeated = repeat_manifest_episodes(episodes, repeat_episodes)
    shard_tasks = plan_manifest_shards(repeated, frames_per_shard, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    feature_cache_dir = None
    if repeat_episodes > 1:
        feature_cache_dir = os.path.join(output_dir, "_episode_feature_cache")
        os.makedirs(feature_cache_dir, exist_ok=True)

    mano_device_obj = torch.device(mano_device if torch.cuda.is_available() else "cpu")
    mano_device_specs = normalize_mano_devices(str(mano_device_obj), mano_gpus if mano_device_obj.type == "cuda" else None)
    if mano_device_obj.type == "cuda" and len(mano_device_specs) == 1:
        writer_workers = min(writer_workers, 1)
    elif mano_device_obj.type == "cuda":
        writer_workers = min(writer_workers, len(mano_device_specs))

    totals = {
        "frames_written": 0,
        "episodes_written": 0,
        "skipped_episodes": 0,
        "shards_written": 0,
    }

    if writer_workers <= 1:
        _worker_init(mano_device_specs, mano_dir, feature_cache_dir)
        result_iter = (_worker_process_shard(task) for task in shard_tasks)
    else:
        mp_context = get_context("spawn") if mano_device_obj.type == "cuda" else get_context()
        pool = mp_context.Pool(
            writer_workers,
            initializer=_worker_init,
            initargs=(mano_device_specs, mano_dir, feature_cache_dir),
        )
        result_iter = pool.imap_unordered(_worker_process_shard, shard_tasks)

    try:
        for result in tqdm(result_iter, total=len(shard_tasks), desc="Build shards"):
            totals["frames_written"] += result["frames_written"]
            totals["episodes_written"] += result["episodes_written"]
            totals["skipped_episodes"] += result["skipped_episodes"]
            totals["shards_written"] += 1 if result["frames_written"] > 0 else 0
    finally:
        if writer_workers > 1:
            pool.close()
            pool.join()

    return {
        "prepare_stats": prepare_stats,
        "totals": totals,
        "planned_shards": len(shard_tasks),
        "planned_frames": sum(ep["num_valid_frames"] for ep in repeated),
    }
