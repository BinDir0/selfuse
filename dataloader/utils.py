"""Utils for webdataset loader."""
from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Dict, Tuple
import io
import json

import numpy as np
from PIL import Image

# 128：wrist(18)+hand(90)+extrinsic(16)+intrinsic(4)。148：中间多 shape(20)。以导出脚本为准。
LOWDIM_DIM_V2 = 128
LOWDIM_DIM_V3 = 148
LOWDIM_DIM = LOWDIM_DIM_V3

STATE_WRIST_SLICE = slice(0, 18)
STATE_HAND_SLICE = slice(18, 108)
# v3 only:
STATE_SHAPE_SLICE = slice(108, 128)
EXTRINSIC_SLICE_V3 = slice(128, 144)
INTRINSIC_SLICE_V3 = slice(144, 148)
EXTRINSIC_SLICE_V2 = slice(108, 124)
INTRINSIC_SLICE_V2 = slice(124, 128)

LEFT_TRANSLATION_SLICE = slice(0, 3)
RIGHT_TRANSLATION_SLICE = slice(3, 6)
LEFT_ROTATION_SLICE = slice(6, 12)
RIGHT_ROTATION_SLICE = slice(12, 18)
LEFT_HAND_SLICE = slice(0, 45)
RIGHT_HAND_SLICE = slice(45, 90)
LEFT_SHAPE_SLICE = slice(0, 10)
RIGHT_SHAPE_SLICE = slice(10, 20)


def is_glob_pattern(path: str) -> bool:
    return any(char in path for char in "*?[]")


def discover_shards(input_arg: str, shard_glob: str) -> List[Path]:
    input_path = Path(input_arg)
    if input_path.is_dir():
        shards = sorted(input_path.glob(shard_glob))
    elif input_path.is_file():
        shards = [input_path] if input_path.suffix == ".tar" else []
    elif is_glob_pattern(input_arg):
        shards = [Path(path) for path in sorted(glob.glob(input_arg))]
    else:
        shards = []

    shards = [path.resolve() for path in shards if path.is_file() and path.suffix == ".tar"]
    if not shards:
        raise FileNotFoundError(f"No tar shards found from input={input_arg!r}")
    return shards


def decode_npy(npy_bytes: bytes) -> np.ndarray:
    return np.load(io.BytesIO(npy_bytes), allow_pickle=False)


def decode_image_jpg(image_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as image:
        return np.asarray(image.convert("RGB"))


def decode_json(json_bytes: bytes) -> Dict:
    return json.loads(json_bytes.decode("utf-8"))


def scalar_int(value, default: int = 3) -> int:
    if value is None:
        return default
    try:
        return int(np.asarray(value).reshape(-1)[0])
    except Exception:
        return default


def sanitize_key(text: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(text))


def normalize_episode_name(meta: Dict) -> str:
    dataset_name = str(meta.get("dataset_name", ""))
    frame_idx = int(meta["frame_idx"])
    episode_name = str(meta["episode_name"])

    frame_suffix = f"_{frame_idx:06d}"
    if episode_name.endswith(frame_suffix): # Hoi4d dataset has some bugs
        return episode_name[: -len(frame_suffix)]
    return episode_name

def split_lowdim(lowdim: np.ndarray) -> Dict[str, np.ndarray]:
    """解析 lowdim 向量；hand 为左右各 45 维 MANO 手指 PCA 系数。"""
    vector = np.asarray(lowdim, dtype=np.float32).reshape(-1)
    n = int(vector.shape[0])
    if n == LOWDIM_DIM_V3:
        state_wrist = vector[STATE_WRIST_SLICE].copy()
        state_hand = vector[STATE_HAND_SLICE].copy()
        state_shape = vector[STATE_SHAPE_SLICE].copy()
        extrinsic_flat = vector[EXTRINSIC_SLICE_V3].copy()
        intrinsic = vector[INTRINSIC_SLICE_V3].copy()
    elif n == LOWDIM_DIM_V2:
        state_wrist = vector[STATE_WRIST_SLICE].copy()
        state_hand = vector[STATE_HAND_SLICE].copy()
        state_shape = np.zeros(20, dtype=np.float32)
        extrinsic_flat = vector[EXTRINSIC_SLICE_V2].copy()
        intrinsic = vector[INTRINSIC_SLICE_V2].copy()
    else:
        raise ValueError(
            f"Invalid lowdim length: expected {LOWDIM_DIM_V2} or {LOWDIM_DIM_V3}, got {vector.shape}"
        )

    return {
        "lowdim": vector,
        "state_wrist": state_wrist,
        "state_hand": state_hand,
        "state_shape": state_shape,
        "extrinsic": extrinsic_flat,
        "extrinsic_4x4": extrinsic_flat.reshape(4, 4).copy(),
        "intrinsic": intrinsic,
        "left_translation": state_wrist[LEFT_TRANSLATION_SLICE].copy(),
        "right_translation": state_wrist[RIGHT_TRANSLATION_SLICE].copy(),
        "left_rot6": state_wrist[LEFT_ROTATION_SLICE].copy(),
        "right_rot6": state_wrist[RIGHT_ROTATION_SLICE].copy(),
        "left_hand_pose45": state_hand[LEFT_HAND_SLICE].copy(),
        "right_hand_pose45": state_hand[RIGHT_HAND_SLICE].copy(),
        "left_shape": state_shape[LEFT_SHAPE_SLICE].copy(),
        "right_shape": state_shape[RIGHT_SHAPE_SLICE].copy(),
    }

# this function provides a unique episode encoding
# used to distinguish between different episodes
def episode_identity(sample: Dict) -> Tuple[str, int, str]:
    return (
        str(sample["dataset_name"]),
        int(sample["episode_idx"]),
        str(sample["episode_name"]),
    )
