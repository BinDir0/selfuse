"""Shared quality-metric helpers for manifest and WebDataset filtering."""

from __future__ import annotations

import re

import numpy as np


LOWDIM_SIZE = 116
LEFT_HAND_TRANSLATION_SLICE = slice(0, 3)
RIGHT_HAND_TRANSLATION_SLICE = slice(3, 6)
LEFT_FINGERTIPS_SLICE = slice(18, 33)
RIGHT_FINGERTIPS_SLICE = slice(33, 48)
EXTRINSIC_SLICE = slice(96, 112)
FRAME_INDEX_PATTERN = re.compile(r"_f(\d+)$")


def is_finite_array(value) -> bool:
    array = np.asarray(value)
    return bool(np.isfinite(array).all())


def max_translation_step(sequence, valid_mask=None) -> dict:
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected (T, C) sequence, got shape {array.shape}")
    if array.shape[0] < 2:
        return {
            "max_step": 0.0,
            "pair_index": None,
            "valid_pairs": 0,
        }

    diffs = np.linalg.norm(array[1:] - array[:-1], axis=1)
    pair_mask = np.ones(diffs.shape[0], dtype=bool)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid.shape[0] != array.shape[0]:
            raise ValueError(
                f"valid_mask length mismatch: expected {array.shape[0]}, got {valid.shape[0]}"
            )
        pair_mask &= valid[1:] & valid[:-1]

    valid_indices = np.flatnonzero(pair_mask)
    if valid_indices.size == 0:
        return {
            "max_step": 0.0,
            "pair_index": None,
            "valid_pairs": 0,
        }

    candidate_diffs = diffs[valid_indices]
    local_argmax = int(np.argmax(candidate_diffs))
    pair_index = int(valid_indices[local_argmax])
    return {
        "max_step": float(candidate_diffs[local_argmax]),
        "pair_index": pair_index,
        "valid_pairs": int(valid_indices.size),
    }


def max_camera_step(extrinsics, valid_mask=None) -> dict:
    mats = np.asarray(extrinsics, dtype=np.float32)
    if mats.ndim == 2 and mats.shape[1] == 16:
        mats = mats.reshape(-1, 4, 4)
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError(f"Expected (T,4,4) extrinsics, got shape {mats.shape}")
    if mats.shape[0] < 2:
        return {
            "max_translation_step": 0.0,
            "translation_pair_index": None,
            "max_rotation_step": 0.0,
            "rotation_pair_index": None,
            "valid_pairs": 0,
        }

    translations = mats[:, :3, 3]
    rotations = mats[:, :3, :3]
    translation_diffs = np.linalg.norm(translations[1:] - translations[:-1], axis=1)
    rotation_diffs = np.linalg.norm((rotations[1:] - rotations[:-1]).reshape(rotations.shape[0] - 1, -1), axis=1)

    pair_mask = np.ones(translation_diffs.shape[0], dtype=bool)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid.shape[0] != mats.shape[0]:
            raise ValueError(
                f"valid_mask length mismatch: expected {mats.shape[0]}, got {valid.shape[0]}"
            )
        pair_mask &= valid[1:] & valid[:-1]

    valid_indices = np.flatnonzero(pair_mask)
    if valid_indices.size == 0:
        return {
            "max_translation_step": 0.0,
            "translation_pair_index": None,
            "max_rotation_step": 0.0,
            "rotation_pair_index": None,
            "valid_pairs": 0,
        }

    translation_local_argmax = int(np.argmax(translation_diffs[valid_indices]))
    rotation_local_argmax = int(np.argmax(rotation_diffs[valid_indices]))
    translation_pair_index = int(valid_indices[translation_local_argmax])
    rotation_pair_index = int(valid_indices[rotation_local_argmax])
    return {
        "max_translation_step": float(translation_diffs[translation_pair_index]),
        "translation_pair_index": translation_pair_index,
        "max_rotation_step": float(rotation_diffs[rotation_pair_index]),
        "rotation_pair_index": rotation_pair_index,
        "valid_pairs": int(valid_indices.size),
    }


def parse_frame_index(sample_key: str) -> int:
    match = FRAME_INDEX_PATTERN.search(sample_key)
    if not match:
        raise ValueError(f"Failed to parse frame index from sample key: {sample_key}")
    return int(match.group(1))


def decode_lowdim(lowdim_bytes: bytes) -> np.ndarray:
    import io

    array = np.load(io.BytesIO(lowdim_bytes), allow_pickle=False)
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    if array.shape != (LOWDIM_SIZE,):
        raise ValueError(f"Expected lowdim shape {(LOWDIM_SIZE,)}, got {array.shape}")
    return array


def extract_lowdim_components(lowdim: np.ndarray) -> dict:
    array = np.asarray(lowdim, dtype=np.float32).reshape(-1)
    if array.shape != (LOWDIM_SIZE,):
        raise ValueError(f"Expected lowdim shape {(LOWDIM_SIZE,)}, got {array.shape}")
    return {
        "left_translation": array[LEFT_HAND_TRANSLATION_SLICE],
        "right_translation": array[RIGHT_HAND_TRANSLATION_SLICE],
        "left_fingertips": array[LEFT_FINGERTIPS_SLICE].reshape(5, 3),
        "right_fingertips": array[RIGHT_FINGERTIPS_SLICE].reshape(5, 3),
        "extrinsic": array[EXTRINSIC_SLICE].reshape(4, 4),
    }


def transform_points_world_to_camera(points, extrinsic) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    mat = np.asarray(extrinsic, dtype=np.float32).reshape(4, 4)
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    return (pts @ rot.T) + trans


def camera_space_abs_metrics(points_world, extrinsic) -> dict:
    points_cam = transform_points_world_to_camera(points_world, extrinsic)
    abs_points = np.abs(points_cam)
    return {
        "max_abs": float(abs_points.max()) if abs_points.size else 0.0,
        "max_abs_x": float(abs_points[:, 0].max()) if abs_points.size else 0.0,
        "max_abs_y": float(abs_points[:, 1].max()) if abs_points.size else 0.0,
        "max_abs_z": float(abs_points[:, 2].max()) if abs_points.size else 0.0,
    }


def new_clip_quality_stats(clip_id: str) -> dict:
    return {
        "clip_id": clip_id,
        "frames_total": 0,
        "frames_kept_candidate": 0,
        "presence_nonzero_frames": 0,
        "incomplete_sample_frames": 0,
        "nonfinite_lowdim_frames": 0,
        "invalid_meta_frames": 0,
        "invalid_lowdim_frames": 0,
        "instruction_num_max": 0,
        "max_hand_translation_step": 0.0,
        "max_camera_translation_step": 0.0,
        "max_camera_rotation_step": 0.0,
        "max_camera_space_wrist_abs": 0.0,
        "max_camera_space_hand_abs": 0.0,
        "_prev_frame_idx": None,
        "_prev_left": None,
        "_prev_right": None,
        "_prev_extrinsic": None,
        "_prev_finite": False,
    }


def update_clip_quality_stats(
    stats: dict,
    frame_idx: int,
    instruction_num: int,
    presence: int,
    lowdim,
    *,
    count_invalid_lowdim: bool = True,
) -> None:
    stats["frames_total"] += 1
    stats["instruction_num_max"] = max(stats["instruction_num_max"], int(instruction_num))
    if int(presence) > 0:
        stats["presence_nonzero_frames"] += 1

    if lowdim is None:
        if count_invalid_lowdim:
            stats["invalid_lowdim_frames"] += 1
        stats["_prev_frame_idx"] = int(frame_idx)
        stats["_prev_left"] = None
        stats["_prev_right"] = None
        stats["_prev_extrinsic"] = None
        stats["_prev_finite"] = False
        return

    try:
        lowdim_array = np.asarray(lowdim, dtype=np.float32).reshape(-1)
    except Exception:
        lowdim_array = None
    if lowdim_array is None or lowdim_array.shape != (LOWDIM_SIZE,):
        if count_invalid_lowdim:
            stats["invalid_lowdim_frames"] += 1
        stats["_prev_frame_idx"] = int(frame_idx)
        stats["_prev_left"] = None
        stats["_prev_right"] = None
        stats["_prev_extrinsic"] = None
        stats["_prev_finite"] = False
        return

    if not is_finite_array(lowdim_array):
        stats["nonfinite_lowdim_frames"] += 1
        stats["_prev_frame_idx"] = int(frame_idx)
        stats["_prev_left"] = None
        stats["_prev_right"] = None
        stats["_prev_extrinsic"] = None
        stats["_prev_finite"] = False
        return

    stats["frames_kept_candidate"] += 1
    parts = extract_lowdim_components(lowdim_array)
    current_left = parts["left_translation"]
    current_right = parts["right_translation"]
    left_fingertips = parts["left_fingertips"]
    right_fingertips = parts["right_fingertips"]
    current_extrinsic = parts["extrinsic"]

    wrist_camera_metrics = camera_space_abs_metrics(
        np.stack([current_left, current_right], axis=0),
        current_extrinsic,
    )
    hand_camera_metrics = camera_space_abs_metrics(
        np.concatenate([left_fingertips, right_fingertips], axis=0),
        current_extrinsic,
    )
    stats["max_camera_space_wrist_abs"] = max(
        stats["max_camera_space_wrist_abs"],
        wrist_camera_metrics["max_abs"],
    )
    stats["max_camera_space_hand_abs"] = max(
        stats["max_camera_space_hand_abs"],
        hand_camera_metrics["max_abs"],
    )

    prev_idx = stats["_prev_frame_idx"]
    if stats["_prev_finite"] and prev_idx is not None:
        frame_gap = max(1, int(frame_idx) - int(prev_idx))
        left_step = float(np.linalg.norm(current_left - stats["_prev_left"]) / frame_gap)
        right_step = float(np.linalg.norm(current_right - stats["_prev_right"]) / frame_gap)
        prev_rot = stats["_prev_extrinsic"][:3, :3]
        prev_trans = stats["_prev_extrinsic"][:3, 3]
        curr_rot = current_extrinsic[:3, :3]
        curr_trans = current_extrinsic[:3, 3]
        camera_translation_step = float(np.linalg.norm(curr_trans - prev_trans) / frame_gap)
        camera_rotation_step = float(np.linalg.norm((curr_rot - prev_rot).reshape(-1)) / frame_gap)

        stats["max_hand_translation_step"] = max(
            stats["max_hand_translation_step"],
            left_step,
            right_step,
        )
        stats["max_camera_translation_step"] = max(
            stats["max_camera_translation_step"],
            camera_translation_step,
        )
        stats["max_camera_rotation_step"] = max(
            stats["max_camera_rotation_step"],
            camera_rotation_step,
        )

    stats["_prev_frame_idx"] = int(frame_idx)
    stats["_prev_left"] = current_left
    stats["_prev_right"] = current_right
    stats["_prev_extrinsic"] = current_extrinsic
    stats["_prev_finite"] = True


def finalize_clip_quality_metrics(stats: dict) -> dict:
    return {
        "frames_total": int(stats["frames_total"]),
        "frames_kept_candidate": int(stats["frames_kept_candidate"]),
        "presence_ratio": (
            float(stats["presence_nonzero_frames"]) / float(stats["frames_total"])
            if stats["frames_total"] > 0
            else 0.0
        ),
        "instruction_num_max": int(stats["instruction_num_max"]),
        "incomplete_sample_frames": int(stats["incomplete_sample_frames"]),
        "nonfinite_lowdim_frames": int(stats["nonfinite_lowdim_frames"]),
        "invalid_meta_frames": int(stats["invalid_meta_frames"]),
        "invalid_lowdim_frames": int(stats["invalid_lowdim_frames"]),
        "max_hand_translation_step": float(stats["max_hand_translation_step"]),
        "max_camera_translation_step": float(stats["max_camera_translation_step"]),
        "max_camera_rotation_step": float(stats["max_camera_rotation_step"]),
        "max_camera_space_wrist_abs": float(stats["max_camera_space_wrist_abs"]),
        "max_camera_space_hand_abs": float(stats["max_camera_space_hand_abs"]),
    }


def summarize_metric_distribution(values) -> dict | None:
    finite = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if finite.size == 0:
        return None
    return {
        "count": int(finite.size),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(finite.max()),
    }


def resolve_auto_quality_thresholds(clip_metrics: list[dict], criteria: dict) -> dict:
    resolved = {
        "max_camera_space_wrist_abs": criteria["max_camera_space_wrist_abs"],
        "max_camera_space_hand_abs": criteria["max_camera_space_hand_abs"],
    }
    summaries = {}
    percentile = float(criteria["camera_space_abs_percentile"])
    scale = float(criteria["camera_space_abs_scale"])

    candidate_metrics = [metrics for metrics in clip_metrics if metrics["frames_kept_candidate"] > 0]
    wrist_values = [metrics["max_camera_space_wrist_abs"] for metrics in candidate_metrics]
    hand_values = [metrics["max_camera_space_hand_abs"] for metrics in candidate_metrics]
    wrist_summary = summarize_metric_distribution(wrist_values)
    hand_summary = summarize_metric_distribution(hand_values)
    if wrist_summary is not None:
        summaries["max_camera_space_wrist_abs"] = wrist_summary
    if hand_summary is not None:
        summaries["max_camera_space_hand_abs"] = hand_summary

    if resolved["max_camera_space_wrist_abs"] is None and wrist_summary is not None:
        resolved["max_camera_space_wrist_abs"] = float(
            np.percentile(np.asarray(wrist_values, dtype=np.float64), percentile) * scale
        )
    if resolved["max_camera_space_hand_abs"] is None and hand_summary is not None:
        resolved["max_camera_space_hand_abs"] = float(
            np.percentile(np.asarray(hand_values, dtype=np.float64), percentile) * scale
        )

    return {
        "resolved": resolved,
        "distribution": summaries,
        "auto_rule": {
            "percentile": percentile,
            "scale": scale,
        },
    }


def decide_clip_quality(
    metrics: dict,
    criteria: dict,
    *,
    include_incomplete_sample_reason: bool = True,
    include_invalid_meta_reason: bool = True,
) -> tuple[bool, list[str]]:
    reasons = []
    if include_incomplete_sample_reason and metrics["incomplete_sample_frames"] > 0:
        reasons.append("incomplete_sample")
    if include_invalid_meta_reason and metrics["invalid_meta_frames"] > 0:
        reasons.append("invalid_meta")
    if metrics["invalid_lowdim_frames"] > 0:
        reasons.append("invalid_lowdim")
    if criteria["drop_nonfinite_lowdim"] and metrics["nonfinite_lowdim_frames"] > 0:
        reasons.append("nonfinite_lowdim")
    if criteria["min_instruction_num"] is not None and metrics["instruction_num_max"] < criteria["min_instruction_num"]:
        reasons.append("instruction_num_below_min")
    if criteria["min_presence_ratio"] is not None and metrics["presence_ratio"] < criteria["min_presence_ratio"]:
        reasons.append("presence_ratio_below_min")
    if (
        criteria["max_hand_translation_step"] is not None
        and metrics["max_hand_translation_step"] > criteria["max_hand_translation_step"]
    ):
        reasons.append("hand_translation_step_exceeded")
    if (
        criteria["max_camera_translation_step"] is not None
        and metrics["max_camera_translation_step"] > criteria["max_camera_translation_step"]
    ):
        reasons.append("camera_translation_step_exceeded")
    if (
        criteria["max_camera_rotation_step"] is not None
        and metrics["max_camera_rotation_step"] > criteria["max_camera_rotation_step"]
    ):
        reasons.append("camera_rotation_step_exceeded")
    if (
        criteria["max_camera_space_wrist_abs"] is not None
        and metrics["max_camera_space_wrist_abs"] > criteria["max_camera_space_wrist_abs"]
    ):
        reasons.append("camera_space_wrist_abs_exceeded")
    if (
        criteria["max_camera_space_hand_abs"] is not None
        and metrics["max_camera_space_hand_abs"] > criteria["max_camera_space_hand_abs"]
    ):
        reasons.append("camera_space_hand_abs_exceeded")
    return not reasons, reasons
