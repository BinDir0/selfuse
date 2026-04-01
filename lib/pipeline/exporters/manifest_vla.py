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
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

from lib.pipeline.annotation_protocol import load_clip_annotation
from lib.pipeline.clip_manifest import ClipManifestRecord, load_clip_manifest
from lib.pipeline.frame_sources import read_frame_bytes_from_descriptor
from lib.pipeline.exporters.webdataset_features import (
    _build_lowdim_features,
    _compute_joint_states,
    _compute_presence_per_frame,
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
MANIFEST_FEATURE_CACHE_VERSION = 3


def _feature_cache_path(seq_folder: str, feature_cache_dir: str) -> str:
    digest = hashlib.md5(seq_folder.encode("utf-8")).hexdigest()
    return os.path.join(feature_cache_dir, f"{digest}.joblib")


def _load_cached_features(
    seq_folder: str,
    frame_count: int,
    feature_cache_dir: str,
    *,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
):
    if not feature_cache_dir:
        return None
    path = _feature_cache_path(seq_folder, feature_cache_dir)
    if not os.path.exists(path):
        return None
    try:
        payload = joblib.load(path)
    except Exception:
        return None
    if (
        payload.get("cache_version") != MANIFEST_FEATURE_CACHE_VERSION
        or payload.get("seq_folder") != seq_folder
        or payload.get("frame_count") != frame_count
        or float(payload.get("source_fps", -1.0)) != float(source_fps)
        or float(payload.get("target_fps", -1.0)) != float(target_fps)
        or bool(payload.get("interpolate_labels", False)) != bool(interpolate_labels)
    ):
        return None
    return {
        "frame_count": payload["frame_count"],
        "lowdim_all": payload["lowdim_all"],
        "presence_per_frame": payload["presence_per_frame"],
    }


def _write_cached_features(
    seq_folder: str,
    feature_cache_dir: str,
    episode_data: dict,
    *,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
):
    if not feature_cache_dir:
        return
    os.makedirs(feature_cache_dir, exist_ok=True)
    path = _feature_cache_path(seq_folder, feature_cache_dir)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    payload = {
        "cache_version": MANIFEST_FEATURE_CACHE_VERSION,
        "seq_folder": seq_folder,
        "frame_count": episode_data["frame_count"],
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "interpolate_labels": bool(interpolate_labels),
        "lowdim_all": episode_data["lowdim_all"],
        "presence_per_frame": episode_data["presence_per_frame"].astype(np.uint8),
    }
    try:
        joblib.dump(payload, tmp_path)
        os.replace(tmp_path, path)
    except OSError:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _build_source_target_times(source_count: int, target_count: int, source_fps: float, target_fps: float):
    if source_count <= 0 or target_count <= 0:
        raise ValueError(f"Invalid counts for resampling: source={source_count}, target={target_count}")
    if source_count == 1:
        return np.zeros((1,), dtype=np.float64), np.zeros((target_count,), dtype=np.float64)

    if source_fps > 0 and target_fps > 0:
        source_times = np.arange(source_count, dtype=np.float64) / float(source_fps)
        target_times = np.arange(target_count, dtype=np.float64) / float(target_fps)
    else:
        source_times = np.arange(source_count, dtype=np.float64)
        target_times = np.linspace(0.0, float(source_count - 1), num=target_count, dtype=np.float64)
    return source_times, np.clip(target_times, source_times[0], source_times[-1])


def _resample_linear_sequence(sequence, target_count: int, source_fps: float, target_fps: float) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float32)
    source_count = int(array.shape[0])
    if target_count == source_count:
        return array.astype(np.float32, copy=False)
    if source_count == 1:
        return np.repeat(array[:1], target_count, axis=0).astype(np.float32, copy=False)

    source_times, target_times = _build_source_target_times(source_count, target_count, source_fps, target_fps)
    flat = array.reshape(source_count, -1)
    output = np.empty((target_count, flat.shape[1]), dtype=np.float32)
    for column_idx in range(flat.shape[1]):
        output[:, column_idx] = np.interp(target_times, source_times, flat[:, column_idx]).astype(np.float32)
    return output.reshape((target_count,) + array.shape[1:])


def _resample_nearest_sequence(sequence, target_count: int, source_fps: float, target_fps: float) -> np.ndarray:
    array = np.asarray(sequence)
    source_count = int(array.shape[0])
    if target_count == source_count:
        return array
    if source_count == 1:
        return np.repeat(array[:1], target_count, axis=0)

    source_times, target_times = _build_source_target_times(source_count, target_count, source_fps, target_fps)
    float_indices = np.interp(target_times, source_times, np.arange(source_count, dtype=np.float64))
    nearest_indices = np.clip(np.rint(float_indices).astype(np.int64), 0, source_count - 1)
    return array[nearest_indices]


def _resample_axis_angle_batch(axis_angle, target_count: int, source_fps: float, target_fps: float) -> np.ndarray:
    array = np.asarray(axis_angle, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected axis-angle batch with shape (N,T,3), got {array.shape}")
    batch_size, source_count, _ = array.shape
    if target_count == source_count:
        return array.astype(np.float32, copy=False)
    if source_count == 1:
        return np.repeat(array[:, :1, :], target_count, axis=1).astype(np.float32, copy=False)

    source_times, target_times = _build_source_target_times(source_count, target_count, source_fps, target_fps)
    output = np.empty((batch_size, target_count, 3), dtype=np.float32)
    for batch_idx in range(batch_size):
        rotations = Rotation.from_rotvec(array[batch_idx])
        slerp = Slerp(source_times, rotations)
        output[batch_idx] = slerp(target_times).as_rotvec().astype(np.float32)
    return output


def _resample_extrinsics_sequence(extrinsics, target_count: int, source_fps: float, target_fps: float) -> np.ndarray:
    mats = np.asarray(extrinsics, dtype=np.float32)
    source_count = int(mats.shape[0])
    if target_count == source_count:
        return mats.astype(np.float32, copy=False)
    if source_count == 1:
        return np.repeat(mats[:1], target_count, axis=0).astype(np.float32, copy=False)

    source_times, target_times = _build_source_target_times(source_count, target_count, source_fps, target_fps)
    rotations = Rotation.from_matrix(mats[:, :3, :3])
    slerp = Slerp(source_times, rotations)
    interp_rot = slerp(target_times).as_matrix().astype(np.float32)
    interp_trans = _resample_linear_sequence(mats[:, :3, 3], target_count, source_fps, target_fps)

    output = np.tile(np.eye(4, dtype=np.float32), (target_count, 1, 1))
    output[:, :3, :3] = interp_rot
    output[:, :3, 3] = interp_trans
    return output


def _resample_episode_features(
    wrist_state,
    hand_state,
    pred_rot,
    extrinsics,
    presence_per_frame,
    target_count: int,
    *,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
):
    source_count = int(wrist_state.shape[0])
    if not interpolate_labels:
        frame_count = min(source_count, target_count)
        return (
            wrist_state[:frame_count],
            hand_state[:frame_count],
            np.asarray(extrinsics[:frame_count], dtype=np.float32),
            np.asarray(presence_per_frame[:frame_count]),
        )

    if target_count <= 0:
        raise ValueError(f"Invalid target_count for resampling: {target_count}")

    wrist_positions = _resample_linear_sequence(wrist_state[:, :6].cpu().numpy(), target_count, source_fps, target_fps)
    hand_state_resampled = _resample_linear_sequence(hand_state.cpu().numpy(), target_count, source_fps, target_fps)
    pred_rot_resampled = _resample_axis_angle_batch(pred_rot.float().cpu().numpy(), target_count, source_fps, target_fps)
    rot6d = axis_angle_to_rot6d(torch.from_numpy(pred_rot_resampled)).cpu().numpy().astype(np.float32)
    wrist_state_resampled = np.concatenate(
        [
            wrist_positions,
            rot6d[0],
            rot6d[1],
        ],
        axis=-1,
    ).astype(np.float32)
    extrinsics_resampled = _resample_extrinsics_sequence(extrinsics, target_count, source_fps, target_fps)
    presence_resampled = _resample_nearest_sequence(presence_per_frame, target_count, source_fps, target_fps)
    return (
        torch.from_numpy(wrist_state_resampled),
        torch.from_numpy(hand_state_resampled.astype(np.float32)),
        extrinsics_resampled.astype(np.float32, copy=False),
        np.asarray(presence_resampled),
    )


def load_descriptor_episode_features(
    ep: dict,
    mano_right,
    mano_left,
    device,
    feature_cache_dir: str | None,
    *,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
):
    seq_folder = ep["seq_folder"]
    requested_frame_count = ep.get("num_valid_frames")
    if requested_frame_count is None and "frame_end" in ep:
        requested_frame_count = int(ep["frame_end"] - ep.get("frame_start", 0))
    if requested_frame_count is not None:
        requested_frame_count = int(requested_frame_count)
        cached = (
            _load_cached_features(
                seq_folder,
                requested_frame_count,
                feature_cache_dir,
                source_fps=source_fps,
                target_fps=target_fps,
                interpolate_labels=interpolate_labels,
            )
            if feature_cache_dir
            else None
        )
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
    source_frame_count = int(pred_trans.shape[1])
    if requested_frame_count is None:
        if interpolate_labels and source_fps > 0 and target_fps > 0 and source_frame_count > 1:
            duration = float(source_frame_count - 1) / float(source_fps)
            requested_frame_count = int(round(duration * float(target_fps))) + 1
        else:
            requested_frame_count = source_frame_count

    frame_count = int(
        requested_frame_count if interpolate_labels else min(int(requested_frame_count), source_frame_count)
    )
    if frame_count <= 0:
        return None

    cached = (
        _load_cached_features(
            seq_folder,
            frame_count,
            feature_cache_dir,
            source_fps=source_fps,
            target_fps=target_fps,
            interpolate_labels=interpolate_labels,
        )
        if feature_cache_dir
        else None
    )
    if cached is not None:
        return cached

    wrist_state, hand_state = _compute_joint_states(
        pred_trans,
        pred_rot,
        pred_hand_pose,
        pred_betas,
        mano_right,
        mano_left,
        device,
    )
    camera_ep = {"crop_dir": seq_folder, "episode_id": ep["episode_id"]}
    extrinsics, intrinsic = _load_episode_camera_features(camera_ep, source_frame_count)
    presence_per_frame = _compute_presence_per_frame(pred_valid, source_frame_count)
    wrist_state, hand_state, extrinsics, presence_per_frame = _resample_episode_features(
        wrist_state[:source_frame_count],
        hand_state[:source_frame_count],
        pred_rot[:, :source_frame_count],
        extrinsics[:source_frame_count],
        presence_per_frame[:source_frame_count],
        frame_count,
        source_fps=source_fps,
        target_fps=target_fps,
        interpolate_labels=interpolate_labels,
    )
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
    _write_cached_features(
        seq_folder,
        feature_cache_dir,
        episode_data,
        source_fps=source_fps,
        target_fps=target_fps,
        interpolate_labels=interpolate_labels,
    )
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
                "source_fps": float(ep.get("source_fps", 5.0)),
                "target_fps": float(ep.get("target_fps", 30.0)),
                "interpolate_labels": bool(ep.get("interpolate_labels", False)),
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
                    source_fps=float(episode_slice.get("source_fps", 5.0)),
                    target_fps=float(episode_slice.get("target_fps", 30.0)),
                    interpolate_labels=bool(episode_slice.get("interpolate_labels", False)),
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
                    "lowdim_schema": "hawor_wrist_world_v2",
                    "wrist_translation_semantics": "mano_joint_0_world",
                    "camera_extrinsic_convention": "w2c",
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


def _prepare_manifest_episode(
    record: ClipManifestRecord,
    require_annotation: bool,
    annotation_root: str | None,
    annotation_suffix: str,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
):
    seq_folder = Path(record.descriptor.seq_folder)
    world_res_path = seq_folder / "world_space_res.pth"
    if not world_res_path.exists():
        return None, "missing_world_res"

    try:
        pred_trans, *_ = joblib.load(world_res_path)
    except Exception:
        return None, "invalid_world_res"

    source_num_frames = int(np.asarray(pred_trans).shape[1])
    target_num_frames = int(record.descriptor.frame_count)
    num_frames = int(target_num_frames if interpolate_labels else min(source_num_frames, target_num_frames))
    if num_frames <= 0:
        return None, "empty_frames"

    language = None
    instruction = []
    if annotation_root:
        annotation, error_code, _ = load_clip_annotation(
            annotation_root,
            record.clip_id,
            annotation_suffix=annotation_suffix,
        )
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
        "source_num_frames": source_num_frames,
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "interpolate_labels": bool(interpolate_labels),
        "instruction": instruction,
        "instruction_num": len(instruction),
        "language": language,
    }, None


def prepare_manifest_episodes(
    manifest_path: str,
    *,
    annotation_root: str | None,
    annotation_suffix: str,
    require_annotation: bool,
    max_episodes: int | None,
    preprocess_workers: int,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
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
        iterator = (
            _prepare_manifest_episode(
                record,
                require_annotation,
                annotation_root,
                annotation_suffix,
                source_fps,
                target_fps,
                interpolate_labels,
            )
            for record in records
        )
    else:
        mp_context = get_context()
        pool = mp_context.Pool(preprocess_workers)
        iterator = pool.imap(
            _prepare_manifest_episode_star,
            (
                (
                    record,
                    require_annotation,
                    annotation_root,
                    annotation_suffix,
                    source_fps,
                    target_fps,
                    interpolate_labels,
                )
                for record in records
            ),
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
    annotation_suffix: str,
    require_annotation: bool,
    max_episodes: int | None,
    repeat_episodes: int,
    preprocess_workers: int,
    writer_workers: int,
    frames_per_shard: int,
    mano_device: str,
    mano_gpus: str | None,
    mano_dir: str | None,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
):
    episodes, prepare_stats = prepare_manifest_episodes(
        manifest_path,
        annotation_root=annotation_root,
        annotation_suffix=annotation_suffix,
        require_annotation=require_annotation,
        max_episodes=max_episodes,
        preprocess_workers=preprocess_workers,
        source_fps=source_fps,
        target_fps=target_fps,
        interpolate_labels=interpolate_labels,
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
