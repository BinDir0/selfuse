"""Episode preparation and feature loading for manifest-based build/export."""

from __future__ import annotations

from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm

from lib.pipeline.annotation_protocol import load_clip_annotation
from lib.pipeline.clip_manifest import ClipManifestRecord, load_clip_manifest
from lib.pipeline.frame_sources import classify_descriptor_storage, validate_descriptor_for_frame_reads
from lib.pipeline.exporters.mano_codec import build_mano_pca_frame_features
from lib.pipeline.exporters.webdataset_features import (
    InvalidCameraDataError,
    _build_lowdim_features,
    _compute_joint_states,
    _compute_presence_per_frame,
    _load_episode_camera_features,
    _load_world_space_prediction,
)
from lib.pipeline.quality_metrics import (
    finalize_clip_quality_metrics,
    new_clip_quality_stats,
    update_clip_quality_stats,
)

from .cache import load_cached_features, write_cached_features
from .resample import resample_episode_features


def load_descriptor_episode_features(
    ep: dict,
    mano_right,
    mano_left,
    device,
    feature_cache_dir: str | None,
    mano_dir: str | None,
    *,
    prediction: dict | None = None,
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
            load_cached_features(
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

    if prediction is None:
        prediction = _load_world_space_prediction({"episode_id": ep["episode_id"]}, str(Path(seq_folder) / "world_space_res.pth"))
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
        load_cached_features(
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
    try:
        extrinsics, intrinsic = _load_episode_camera_features(camera_ep, source_frame_count)
        presence_per_frame = _compute_presence_per_frame(pred_valid, source_frame_count)
        wrist_state, hand_state, pred_rot, pred_hand_pose, pred_betas, extrinsics, presence_per_frame = resample_episode_features(
            wrist_state[:source_frame_count],
            hand_state[:source_frame_count],
            pred_rot[:, :source_frame_count],
            pred_hand_pose[:, :source_frame_count],
            pred_betas[:, :source_frame_count],
            extrinsics[:source_frame_count],
            presence_per_frame[:source_frame_count],
            frame_count,
            source_fps=source_fps,
            target_fps=target_fps,
            interpolate_labels=interpolate_labels,
        )
    except InvalidCameraDataError as error:
        print(f"  Skip {ep['episode_id']}: invalid camera features: {error}")
        return None
    lowdim_all = _build_lowdim_features(
        wrist_state,
        hand_state,
        extrinsics[:frame_count],
        intrinsic,
    )
    mano_all = build_mano_pca_frame_features(
        pred_hand_pose[:, :frame_count].cpu().numpy(),
        pred_betas[:, :frame_count].cpu().numpy(),
        mano_dir=mano_dir,
    )

    episode_data = {
        "frame_count": frame_count,
        "lowdim_all": lowdim_all[:frame_count],
        "mano_all": mano_all[:frame_count],
        "presence_per_frame": presence_per_frame[:frame_count],
    }
    write_cached_features(
        seq_folder,
        feature_cache_dir,
        episode_data,
        source_fps=source_fps,
        target_fps=target_fps,
        interpolate_labels=interpolate_labels,
    )
    return episode_data


def compute_descriptor_episode_quality_metrics(
    ep: dict,
    mano_right,
    mano_left,
    device,
    feature_cache_dir: str | None,
    mano_dir: str | None,
    *,
    prediction: dict | None = None,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
):
    episode_data = load_descriptor_episode_features(
        ep,
        mano_right,
        mano_left,
        device,
        feature_cache_dir,
        mano_dir,
        prediction=prediction,
        source_fps=source_fps,
        target_fps=target_fps,
        interpolate_labels=interpolate_labels,
    )
    if episode_data is None:
        return None

    frame_count = int(episode_data["frame_count"])
    lowdim_all = np.asarray(episode_data["lowdim_all"])
    presence_per_frame = np.asarray(episode_data["presence_per_frame"])
    if (
        frame_count <= 0
        or lowdim_all.ndim != 2
        or lowdim_all.shape[0] < frame_count
        or presence_per_frame.shape[0] < frame_count
    ):
        return None

    stats = new_clip_quality_stats(ep["clip_id"])
    instruction_num = int(ep.get("instruction_num", 0))
    for frame_idx in range(frame_count):
        update_clip_quality_stats(
            stats,
            frame_idx,
            instruction_num,
            int(presence_per_frame[frame_idx]),
            lowdim_all[frame_idx],
        )
    return finalize_clip_quality_metrics(stats)


def _prepare_manifest_episode(
    record: ClipManifestRecord,
    require_annotation: bool,
    annotation_root: str | None,
    annotation_suffix: str,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
):
    try:
        validate_descriptor_for_frame_reads(record.descriptor)
    except Exception:
        return None, "invalid_descriptor"

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


def prepare_manifest_record_for_build(
    record: ClipManifestRecord,
    *,
    require_annotation: bool,
    annotation_root: str | None,
    annotation_suffix: str,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
    prediction: dict | None = None,
):
    if prediction is None:
        return _prepare_manifest_episode(
            record,
            require_annotation,
            annotation_root,
            annotation_suffix,
            source_fps,
            target_fps,
            interpolate_labels,
        )

    try:
        source_num_frames = int(prediction["pred_trans"].shape[1])
    except Exception:
        return None, "invalid_world_res"

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
        "seq_folder": str(Path(record.descriptor.seq_folder)),
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


def load_manifest_record_prediction(record: ClipManifestRecord):
    seq_folder = Path(record.descriptor.seq_folder)
    world_res_path = seq_folder / "world_space_res.pth"
    if not world_res_path.exists():
        return None, "missing_world_res"
    prediction = _load_world_space_prediction({"episode_id": record.clip_id}, str(world_res_path))
    if prediction is None:
        return None, "invalid_world_res"
    return prediction, None


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
        "invalid_descriptor": 0,
        "missing_world_res": 0,
        "invalid_world_res": 0,
        "empty_frames": 0,
        "missing_annotation": 0,
        "invalid_json": 0,
        "invalid_status": 0,
        "empty_instruction": 0,
        "descriptor_paths": {
            "light_tar": 0,
            "heavy_tar": 0,
            "image_sequence": 0,
        },
    }

    for record in records:
        descriptor_kind = classify_descriptor_storage(record.descriptor)
        stats["descriptor_paths"][descriptor_kind] = stats["descriptor_paths"].get(descriptor_kind, 0) + 1

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
