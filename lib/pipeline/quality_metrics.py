"""Shared quality-metric helpers for manifest and WebDataset filtering."""

from __future__ import annotations

import re

import numpy as np


LOWDIM_SIZE = 116
LEFT_HAND_TRANSLATION_SLICE = slice(0, 3)
RIGHT_HAND_TRANSLATION_SLICE = slice(3, 6)
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
        "extrinsic": array[EXTRINSIC_SLICE].reshape(4, 4),
    }
