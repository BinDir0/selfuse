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
CAMERA_AXES = ("x", "y", "z")


def parse_instruction_metadata(meta: dict | None) -> dict:
    """Normalize one frame/episode instruction payload and expose fail-closed flags."""
    if not isinstance(meta, dict):
        return {
            "instruction_num": 0,
            "instructions": [],
            "slots": [],
            "effective_slots": [],
            "missing_instruction": False,
            "empty_instruction": False,
            "instruction_num_mismatch": False,
        }

    raw_instruction_num = meta.get("instruction_num", 0)
    try:
        instruction_num = max(0, int(raw_instruction_num))
    except Exception:
        instruction_num = 0

    raw_instruction = meta.get("instruction", [])
    if isinstance(raw_instruction, str):
        slots = [raw_instruction]
    elif isinstance(raw_instruction, (list, tuple)):
        slots = list(raw_instruction)
    else:
        slots = []

    effective_slots = slots[:instruction_num]
    instructions = [str(item).strip() for item in effective_slots if str(item).strip()]
    missing_instruction = instruction_num <= 0
    empty_instruction = instruction_num > 0 and len(instructions) == 0
    instruction_num_mismatch = instruction_num > 0 and (
        len(slots) < instruction_num or len(instructions) != min(instruction_num, len(effective_slots))
    )
    return {
        "instruction_num": int(instruction_num),
        "instructions": instructions,
        "slots": slots,
        "effective_slots": effective_slots,
        "missing_instruction": bool(missing_instruction),
        "empty_instruction": bool(empty_instruction),
        "instruction_num_mismatch": bool(instruction_num_mismatch),
    }


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


def camera_space_axis_metrics(points_world, extrinsic) -> dict:
    points_cam = transform_points_world_to_camera(points_world, extrinsic)
    if points_cam.size == 0:
        return {
            "min_x": 0.0,
            "max_x": 0.0,
            "min_y": 0.0,
            "max_y": 0.0,
            "min_z": 0.0,
            "max_z": 0.0,
        }
    return {
        "min_x": float(points_cam[:, 0].min()),
        "max_x": float(points_cam[:, 0].max()),
        "min_y": float(points_cam[:, 1].min()),
        "max_y": float(points_cam[:, 1].max()),
        "min_z": float(points_cam[:, 2].min()),
        "max_z": float(points_cam[:, 2].max()),
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
        "missing_instruction_frames": 0,
        "empty_instruction_frames": 0,
        "instruction_num_mismatch_frames": 0,
        "instruction_num_max": 0,
        "max_hand_translation_step": 0.0,
        "max_camera_translation_step": 0.0,
        "max_camera_rotation_step": 0.0,
        "max_camera_space_wrist_abs": 0.0,
        "max_camera_space_hand_abs": 0.0,
        "_camera_space_wrist_min": np.full((3,), np.inf, dtype=np.float32),
        "_camera_space_wrist_max": np.full((3,), -np.inf, dtype=np.float32),
        "_camera_space_hand_min": np.full((3,), np.inf, dtype=np.float32),
        "_camera_space_hand_max": np.full((3,), -np.inf, dtype=np.float32),
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
    missing_instruction: bool = False,
    empty_instruction: bool = False,
    instruction_num_mismatch: bool = False,
    count_invalid_lowdim: bool = True,
    compute_motion_metrics: bool = True,
    compute_camera_space_metrics: bool = True,
) -> None:
    stats["frames_total"] += 1
    stats["instruction_num_max"] = max(stats["instruction_num_max"], int(instruction_num))
    if missing_instruction:
        stats["missing_instruction_frames"] += 1
    if empty_instruction:
        stats["empty_instruction_frames"] += 1
    if instruction_num_mismatch:
        stats["instruction_num_mismatch_frames"] += 1
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
    if not compute_motion_metrics and not compute_camera_space_metrics:
        stats["_prev_frame_idx"] = None
        stats["_prev_left"] = None
        stats["_prev_right"] = None
        stats["_prev_extrinsic"] = None
        stats["_prev_finite"] = False
        return

    parts = extract_lowdim_components(lowdim_array)
    current_left = parts["left_translation"]
    current_right = parts["right_translation"]
    left_fingertips = parts["left_fingertips"]
    right_fingertips = parts["right_fingertips"]
    current_extrinsic = parts["extrinsic"]

    if compute_camera_space_metrics:
        wrist_camera_metrics = camera_space_abs_metrics(
            np.stack([current_left, current_right], axis=0),
            current_extrinsic,
        )
        wrist_axis_metrics = camera_space_axis_metrics(
            np.stack([current_left, current_right], axis=0),
            current_extrinsic,
        )
        hand_camera_metrics = camera_space_abs_metrics(
            np.concatenate([left_fingertips, right_fingertips], axis=0),
            current_extrinsic,
        )
        hand_axis_metrics = camera_space_axis_metrics(
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
        stats["_camera_space_wrist_min"] = np.minimum(
            stats["_camera_space_wrist_min"],
            np.asarray([wrist_axis_metrics["min_x"], wrist_axis_metrics["min_y"], wrist_axis_metrics["min_z"]], dtype=np.float32),
        )
        stats["_camera_space_wrist_max"] = np.maximum(
            stats["_camera_space_wrist_max"],
            np.asarray([wrist_axis_metrics["max_x"], wrist_axis_metrics["max_y"], wrist_axis_metrics["max_z"]], dtype=np.float32),
        )
        stats["_camera_space_hand_min"] = np.minimum(
            stats["_camera_space_hand_min"],
            np.asarray([hand_axis_metrics["min_x"], hand_axis_metrics["min_y"], hand_axis_metrics["min_z"]], dtype=np.float32),
        )
        stats["_camera_space_hand_max"] = np.maximum(
            stats["_camera_space_hand_max"],
            np.asarray([hand_axis_metrics["max_x"], hand_axis_metrics["max_y"], hand_axis_metrics["max_z"]], dtype=np.float32),
        )

    if compute_motion_metrics:
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
    else:
        stats["_prev_frame_idx"] = None
        stats["_prev_left"] = None
        stats["_prev_right"] = None
        stats["_prev_extrinsic"] = None
        stats["_prev_finite"] = False


def finalize_clip_quality_metrics(stats: dict) -> dict:
    def _axis_value(array, axis_idx: int, *, fallback: float) -> float:
        value = float(array[axis_idx])
        return fallback if not np.isfinite(value) else value

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
        "missing_instruction_frames": int(stats["missing_instruction_frames"]),
        "empty_instruction_frames": int(stats["empty_instruction_frames"]),
        "instruction_num_mismatch_frames": int(stats["instruction_num_mismatch_frames"]),
        "max_hand_translation_step": float(stats["max_hand_translation_step"]),
        "max_camera_translation_step": float(stats["max_camera_translation_step"]),
        "max_camera_rotation_step": float(stats["max_camera_rotation_step"]),
        "max_camera_space_wrist_abs": float(stats["max_camera_space_wrist_abs"]),
        "max_camera_space_hand_abs": float(stats["max_camera_space_hand_abs"]),
        "min_camera_space_wrist_x": _axis_value(stats["_camera_space_wrist_min"], 0, fallback=0.0),
        "max_camera_space_wrist_x": _axis_value(stats["_camera_space_wrist_max"], 0, fallback=0.0),
        "min_camera_space_wrist_y": _axis_value(stats["_camera_space_wrist_min"], 1, fallback=0.0),
        "max_camera_space_wrist_y": _axis_value(stats["_camera_space_wrist_max"], 1, fallback=0.0),
        "min_camera_space_wrist_z": _axis_value(stats["_camera_space_wrist_min"], 2, fallback=0.0),
        "max_camera_space_wrist_z": _axis_value(stats["_camera_space_wrist_max"], 2, fallback=0.0),
        "min_camera_space_hand_x": _axis_value(stats["_camera_space_hand_min"], 0, fallback=0.0),
        "max_camera_space_hand_x": _axis_value(stats["_camera_space_hand_max"], 0, fallback=0.0),
        "min_camera_space_hand_y": _axis_value(stats["_camera_space_hand_min"], 1, fallback=0.0),
        "max_camera_space_hand_y": _axis_value(stats["_camera_space_hand_max"], 1, fallback=0.0),
        "min_camera_space_hand_z": _axis_value(stats["_camera_space_hand_min"], 2, fallback=0.0),
        "max_camera_space_hand_z": _axis_value(stats["_camera_space_hand_max"], 2, fallback=0.0),
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


def summarize_iqr_distribution(values, multiplier: float) -> dict | None:
    finite = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if finite.size == 0:
        return None
    q1 = float(np.percentile(finite, 25))
    q3 = float(np.percentile(finite, 75))
    iqr = float(q3 - q1)
    return {
        "count": int(finite.size),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "lower_bound": float(q1 - multiplier * iqr),
        "upper_bound": float(q3 + multiplier * iqr),
    }


def _camera_space_bound_metrics(prefix: str, clip_metrics: list[dict], multiplier: float) -> tuple[dict | None, dict]:
    bounds = {}
    distributions = {}
    has_any = False
    for axis in CAMERA_AXES:
        min_key = f"min_camera_space_{prefix}_{axis}"
        max_key = f"max_camera_space_{prefix}_{axis}"
        min_summary = summarize_iqr_distribution([metrics[min_key] for metrics in clip_metrics], multiplier)
        max_summary = summarize_iqr_distribution([metrics[max_key] for metrics in clip_metrics], multiplier)
        if min_summary is None or max_summary is None:
            continue
        has_any = True
        bounds[axis] = {
            "lower": float(min_summary["lower_bound"]),
            "upper": float(max_summary["upper_bound"]),
        }
        distributions[axis] = {
            "lower_tail": min_summary,
            "upper_tail": max_summary,
        }
    return (bounds if has_any else None), distributions


def _camera_space_axis_abs_cap_bounds(cap: float | None) -> dict | None:
    if cap is None:
        return None
    cap_value = float(cap)
    return {
        axis: {
            "lower": -cap_value,
            "upper": cap_value,
        }
        for axis in CAMERA_AXES
    }


def _merge_camera_space_bounds(primary: dict | None, secondary: dict | None) -> dict | None:
    if primary is None:
        return secondary
    if secondary is None:
        return primary
    merged = {}
    for axis in CAMERA_AXES:
        primary_axis = primary.get(axis) if primary else None
        secondary_axis = secondary.get(axis) if secondary else None
        if primary_axis is None and secondary_axis is None:
            continue
        if primary_axis is None:
            merged[axis] = dict(secondary_axis)
            continue
        if secondary_axis is None:
            merged[axis] = dict(primary_axis)
            continue
        merged[axis] = {
            "lower": max(float(primary_axis["lower"]), float(secondary_axis["lower"])),
            "upper": min(float(primary_axis["upper"]), float(secondary_axis["upper"])),
        }
    return merged or None


def resolve_auto_quality_thresholds(clip_metrics: list[dict], criteria: dict) -> dict:
    resolved = {
        "max_camera_space_wrist_abs": criteria["max_camera_space_wrist_abs"],
        "max_camera_space_hand_abs": criteria["max_camera_space_hand_abs"],
        "camera_space_wrist_bounds": None,
        "camera_space_hand_bounds": None,
    }
    summaries = {}
    auto_method = str(criteria.get("camera_space_auto_method", "iqr_bounds"))
    percentile = float(criteria.get("camera_space_abs_percentile", 99.0))
    scale = float(criteria.get("camera_space_abs_scale", 2.5))
    iqr_multiplier = float(criteria.get("camera_space_iqr_multiplier", 2.5))
    axis_abs_cap = criteria.get("camera_space_axis_abs_cap", 1.5)

    candidate_metrics = [metrics for metrics in clip_metrics if metrics["frames_kept_candidate"] > 0]
    use_manual_abs = (
        resolved["max_camera_space_wrist_abs"] is not None
        or resolved["max_camera_space_hand_abs"] is not None
    )
    if not criteria.get("use_auto_camera_space_thresholds", True) and not use_manual_abs:
        return {
            "resolved": resolved,
            "distribution": summaries,
            "auto_rule": {
                "method": "disabled",
                "percentile": percentile,
                "scale": scale,
                "iqr_multiplier": iqr_multiplier,
                "axis_abs_cap": axis_abs_cap,
            },
        }

    if not use_manual_abs and auto_method == "iqr_bounds":
        wrist_bounds, wrist_distributions = _camera_space_bound_metrics("wrist", candidate_metrics, iqr_multiplier)
        hand_bounds, hand_distributions = _camera_space_bound_metrics("hand", candidate_metrics, iqr_multiplier)
        cap_bounds = _camera_space_axis_abs_cap_bounds(axis_abs_cap)
        resolved["camera_space_wrist_bounds"] = _merge_camera_space_bounds(wrist_bounds, cap_bounds)
        resolved["camera_space_hand_bounds"] = _merge_camera_space_bounds(hand_bounds, cap_bounds)
        if wrist_distributions:
            summaries["camera_space_wrist_bounds"] = wrist_distributions
        if hand_distributions:
            summaries["camera_space_hand_bounds"] = hand_distributions
        if cap_bounds is not None:
            summaries["camera_space_axis_abs_cap"] = {
                "lower": -float(axis_abs_cap),
                "upper": float(axis_abs_cap),
            }
    else:
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

    if use_manual_abs and axis_abs_cap is not None:
        summaries["camera_space_axis_abs_cap"] = {
            "lower": -float(axis_abs_cap),
            "upper": float(axis_abs_cap),
        }

    return {
        "resolved": resolved,
        "distribution": summaries,
        "auto_rule": {
            "method": "manual_abs" if use_manual_abs else auto_method,
            "percentile": percentile,
            "scale": scale,
            "iqr_multiplier": iqr_multiplier,
            "axis_abs_cap": axis_abs_cap,
        },
    }


def _camera_space_bounds_exceeded(metrics: dict, prefix: str, bounds: dict | None) -> bool:
    if not bounds:
        return False
    for axis in CAMERA_AXES:
        axis_bounds = bounds.get(axis)
        if not axis_bounds:
            continue
        min_key = f"min_camera_space_{prefix}_{axis}"
        max_key = f"max_camera_space_{prefix}_{axis}"
        if metrics[min_key] < axis_bounds["lower"] or metrics[max_key] > axis_bounds["upper"]:
            return True
    return False


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
    if metrics["nonfinite_lowdim_frames"] > 0:
        reasons.append("nonfinite_lowdim")
    if metrics.get("missing_instruction_frames", 0) > 0:
        reasons.append("missing_instruction_frame")
    if metrics.get("empty_instruction_frames", 0) > 0:
        reasons.append("empty_instruction_frame")
    if metrics.get("instruction_num_mismatch_frames", 0) > 0:
        reasons.append("instruction_num_mismatch_frame")
    if criteria.get("min_instruction_num") is not None and metrics["instruction_num_max"] < criteria["min_instruction_num"]:
        reasons.append("instruction_num_below_min")
    if criteria.get("min_presence_ratio") is not None and metrics["presence_ratio"] < criteria["min_presence_ratio"]:
        reasons.append("presence_ratio_below_min")
    if (
        criteria.get("max_hand_translation_step") is not None
        and metrics["max_hand_translation_step"] > criteria["max_hand_translation_step"]
    ):
        reasons.append("hand_translation_step_exceeded")
    if (
        criteria.get("max_camera_translation_step") is not None
        and metrics["max_camera_translation_step"] > criteria["max_camera_translation_step"]
    ):
        reasons.append("camera_translation_step_exceeded")
    if (
        criteria.get("max_camera_rotation_step") is not None
        and metrics["max_camera_rotation_step"] > criteria["max_camera_rotation_step"]
    ):
        reasons.append("camera_rotation_step_exceeded")
    if (
        criteria.get("max_camera_space_wrist_abs") is not None
        and metrics["max_camera_space_wrist_abs"] > criteria["max_camera_space_wrist_abs"]
    ):
        reasons.append("camera_space_wrist_abs_exceeded")
    if (
        criteria.get("max_camera_space_hand_abs") is not None
        and metrics["max_camera_space_hand_abs"] > criteria["max_camera_space_hand_abs"]
    ):
        reasons.append("camera_space_hand_abs_exceeded")
    if _camera_space_bounds_exceeded(metrics, "wrist", criteria.get("camera_space_wrist_bounds")):
        reasons.append("camera_space_wrist_iqr_bounds_exceeded")
    if _camera_space_bounds_exceeded(metrics, "hand", criteria.get("camera_space_hand_bounds")):
        reasons.append("camera_space_hand_iqr_bounds_exceeded")
    axis_abs_cap = criteria.get("camera_space_axis_abs_cap")
    if axis_abs_cap is not None:
        cap_bounds = _camera_space_axis_abs_cap_bounds(axis_abs_cap)
        if (
            criteria.get("camera_space_wrist_bounds") is None
            and _camera_space_bounds_exceeded(metrics, "wrist", cap_bounds)
        ):
            reasons.append("camera_space_wrist_axis_abs_cap_exceeded")
        if (
            criteria.get("camera_space_hand_bounds") is None
            and _camera_space_bounds_exceeded(metrics, "hand", cap_bounds)
        ):
            reasons.append("camera_space_hand_axis_abs_cap_exceeded")
    return not reasons, reasons
