"""
Provides a webdataset dataloader that produce:
video (B,T,3,H,W)
existence (B,T,2)
MANO (B,T,·)
    left_translation : B x T x 3
    right_translation : B x T x 3
    left_rot6 : B x T x 6
    right_rot6 : B x T x 6
    left_hand_pose45 : B x T x 45
    right_hand_pose45 : B x T x 45
    left_shape : B x T x 10
    right_shape : B x T x 10

Accepts dataset with the following structure in tar shards:
taco_v2/
  ├── taco_v2-000000.tar
  ├── taco_v2-000001.tar
  └── ...
Each tar shard contains samples with the following files:
{dataset_name}_{episode_name}_f{frame_idx_int:06d}.image.jpg
taco_v2_episodexxx_f00045.lowdim.npy
taco_v2_episodexxx_f00045.meta.json

# taco_ep000123_f00045.meta.json
{
    "dataset_name": "taco",
    "episode_index": 123,
    "frame_index": 45,
    "presence": 3,   # 0 none, 1 left, 2 right, 3 both
}

# taco_ep000123_f00045.lowdim.npy   shape=(148,), dtype=float32
lowdim = concat([
    wrist,   # 18 = [left_trans(3), right_trans(3), left_rot(6), right_rot(6)]
    # 6drot is the first two columns of a 3x3 transformation matrix（wrist2world）.
    hand,    # 90 = [left_hand(45), right_hand(45)]
             # in left: [thumb_fingertips([x, y, z] 3), index_fingertips(3), ...]
    shape,   # 20 = [left_shape(10), right_shape(10)]
    # The coordinates above are in the world coordinate system.
    extrinsic,     # 16, flatten(4x4) homogeneous transformation matrix World2Cam
    intrinsic,     # 4, [fx, fy, cx, cy]
])

"""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, IterableDataset

try:
    from .utils import (
        discover_shards,
        decode_npy,
        decode_image_jpg,
        decode_json,
        scalar_int,
        normalize_episode_name,
        split_lowdim,
        episode_identity,
    )
except ImportError:
    from utils import (
        discover_shards,
        decode_npy,
        decode_image_jpg,
        decode_json,
        scalar_int,
        normalize_episode_name,
        split_lowdim,
        episode_identity,
    )

KNOWN_SUFFIXES = (
    "image.jpg",
    "lowdim.npy",
    "meta.json",
)
# each sample must contain:
REQUIRED_FIELDS = {"image.jpg", "lowdim.npy", "meta.json"}
REQUIRED_META_KEYS = ("dataset_name", "episode_name", "episode_idx", "frame_idx", "presence")


# ---------------------------------- Helper functions for loading and processing ----------------------------------

def parse_member_name(name: str) -> Optional[Tuple[Optional[str], str]]:
    normalized = name.strip().lstrip("./")
    for suffix in KNOWN_SUFFIXES:
        if normalized == suffix:
            return None, suffix
        for separator in (".", "/"):
            marker = f"{separator}{suffix}"
            if normalized.endswith(marker):
                sample_key = normalized[: -len(marker)]
                if sample_key:
                    return sample_key, suffix
    return None


# check if the meta.json is valid
def validate_converted_meta(meta: Dict, sample_key: str) -> Dict:
    if not isinstance(meta, dict):
        raise TypeError(f"meta.json must decode to a dict for key={sample_key!r}")

    missing = [key for key in REQUIRED_META_KEYS if key not in meta]
    if missing:
        raise KeyError(f"Converted lowdim sample key={sample_key!r} missing required meta keys: {missing}")

    validated = dict(meta)
    validated["dataset_name"] = str(meta["dataset_name"])
    validated["episode_name"] = str(meta["episode_name"])
    validated["episode_idx"] = int(meta["episode_idx"])
    validated["frame_idx"] = int(meta["frame_idx"])
    validated["presence"] = scalar_int(meta["presence"], default=-1)

    if validated["presence"] not in (0, 1, 2, 3):
        raise ValueError(
            f"Invalid presence={validated['presence']} for key={sample_key!r}; expected one of [0,1,2,3]"
        )

    validated["episode_name_normalized"] = normalize_episode_name(validated)

    return validated


def decode_lowdim_sample(sample_key: str, fields: Dict[str, bytes]) -> Dict:
    meta = validate_converted_meta(decode_json(fields["meta.json"]), sample_key)
    mano = split_lowdim(decode_npy(fields["lowdim.npy"]).astype(np.float32))
    presence = int(meta["presence"])

    sample = {
        "sample_key": sample_key,
        "image.jpg": fields["image.jpg"],
        "meta": meta,
        "dataset_name": str(meta["dataset_name"]),
        "episode_name": str(meta["episode_name_normalized"]),
        "episode_name_raw": str(meta["episode_name"]),
        "episode_idx": int(meta["episode_idx"]),
        "frame_idx": int(meta["frame_idx"]),
        "presence": presence,
    }
    sample.update(mano)
    return sample


def build_window_batch(episode_name: str, episode_frames: List[Dict], start: int, window_size: int) -> Dict:
    window_frames = episode_frames[start : start + window_size]
    frame_indices = np.asarray([frame["frame_idx"] for frame in window_frames], dtype=np.int32)

    def convert_presence(p: int) -> List[int]:
        return [int(p in (1, 3)), int(p in (2, 3))]

    def concat_imgs(frames: List[Dict]) -> np.ndarray:
        video_thwc = np.stack([decode_image_jpg(frame["image.jpg"]) for frame in frames], axis=0)
        return np.transpose(video_thwc, (0, 3, 1, 2)).copy()

    video = concat_imgs(window_frames)
    existence = np.asarray([convert_presence(frame["presence"]) for frame in window_frames], dtype=np.int8)

    return {
        # Basic information
        "dataset_name": window_frames[0]["dataset_name"],
        "episode_name": episode_name,
        "episode_idx": int(window_frames[0]["episode_idx"]),
        "sample_keys": [frame["sample_key"] for frame in window_frames],
        "frame_indices": frame_indices,
        "meta": [frame["meta"] for frame in window_frames],
        "window_start": int(frame_indices[0]),
        "window_end": int(frame_indices[-1]),
        "window_size": window_size,
        # Video shape: (T,3,H,W); DataLoader batching gives (B,T,3,H,W)
        "video": video,
        # Existence shape: (T,2); DataLoader batching gives (B,T,2)
        "existence": existence,
        # MANO features
        # uncomment these if you want to include raw information for debugging
        # "lowdim": np.stack([frame["lowdim"] for frame in window_frames], axis=0),
        "state_wrist": np.stack([frame["state_wrist"] for frame in window_frames], axis=0),
        "state_hand": np.stack([frame["state_hand"] for frame in window_frames], axis=0),
        "state_shape": np.stack([frame["state_shape"] for frame in window_frames], axis=0),
        # wrist translations
        "left_translation": np.stack([frame["left_translation"] for frame in window_frames], axis=0),
        "right_translation": np.stack([frame["right_translation"] for frame in window_frames], axis=0),
        # wrist rotations (6d representation)
        "left_rot6": np.stack([frame["left_rot6"] for frame in window_frames], axis=0),
        "right_rot6": np.stack([frame["right_rot6"] for frame in window_frames], axis=0),
        # hand pose parameters
        "left_hand_pose45": np.stack([frame["left_hand_pose45"] for frame in window_frames], axis=0),
        "right_hand_pose45": np.stack([frame["right_hand_pose45"] for frame in window_frames], axis=0),
        # hand shape parameters(betas)
        "left_shape": np.stack([frame["left_shape"] for frame in window_frames], axis=0),
        "right_shape": np.stack([frame["right_shape"] for frame in window_frames], axis=0),
        # camera parameters
        "extrinsic": np.stack([frame["extrinsic"] for frame in window_frames], axis=0),
        "extrinsic_4x4": np.stack([frame["extrinsic_4x4"] for frame in window_frames], axis=0),
        "intrinsic": np.stack([frame["intrinsic"] for frame in window_frames], axis=0),
    }


# ------------------------------ Dataloading and episode windowing logic ------------------------------

# read a full sample ("image.jpg", "lowdim.npy", "meta.json") from a tar shard, yielding one sample at a time
def iter_lowdim_samples_in_shard(shard_path: Path) -> Iterator[Tuple[str, Dict[str, bytes]]]:
    current_key: Optional[str] = None
    current_fields: Dict[str, bytes] = {}
    current_unkeyed_index = 0

    def next_unkeyed_key() -> str:
        return f"sample_{current_unkeyed_index:06d}"

    with tarfile.open(shard_path, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parsed = parse_member_name(member.name)
            if parsed is None:
                continue

            sample_key, suffix = parsed
            extracted = tar.extractfile(member)
            if extracted is None:
                continue

            if sample_key is None:
                if current_key is None:
                    current_key = next_unkeyed_key()
                elif suffix in current_fields:
                    if REQUIRED_FIELDS.issubset(current_fields):
                        yield current_key, current_fields
                    current_unkeyed_index += 1
                    current_key = next_unkeyed_key()
                    current_fields = {}
                sample_key = current_key

            if current_key is None:
                current_key = sample_key
            elif sample_key != current_key:
                if REQUIRED_FIELDS.issubset(current_fields):
                    yield current_key, current_fields
                current_key = sample_key
                current_fields = {}

            current_fields[suffix] = extracted.read()

    if current_key is not None and REQUIRED_FIELDS.issubset(current_fields):
        yield current_key, current_fields


def emit_episode_windows(
    episode_name: str,
    episode_frames: List[Dict],
    window_size: int,
    stride: int,
    episode_filter: Optional[str],
) -> Iterator[Dict]:
    if not episode_frames:
        return
    if episode_filter is not None and episode_name != episode_filter:
        return

    ordered_frames = sorted(episode_frames, key=lambda item: item["frame_idx"])
    if len(ordered_frames) < window_size:
        return

    for start in range(0, len(ordered_frames) - window_size + 1, stride):
        yield build_window_batch(episode_name, ordered_frames, start, window_size)


def iter_episode_windows(
    shards: List[Path],
    window_size: int,
    stride: int,
    episode_filter: Optional[str],
) -> Iterator[Dict]:
    """
    Main logic for dataloading and episode windowing.
    This function will handle cross-shard loading, episode boundary detection, and window emission.
    """
    current_episode_id: Optional[Tuple[str, int, str]] = None
    current_frames: List[Dict] = []

    for shard_path in shards:
        for sample_key, fields in iter_lowdim_samples_in_shard(shard_path):
            sample = decode_lowdim_sample(sample_key, fields)
            sample_episode_id = episode_identity(sample) 

            if current_episode_id is None:
                current_episode_id = sample_episode_id

            if sample_episode_id != current_episode_id: # we have reached a new episode 
                yield from emit_episode_windows(
                    current_episode_id[2],
                    current_frames,
                    window_size,
                    stride,
                    episode_filter,
                )
                current_episode_id = sample_episode_id # start tracking the new episode
                current_frames = []

            current_frames.append(sample)

    if current_episode_id is not None: # the last shard is done, emit remaining windows
        yield from emit_episode_windows(
            current_episode_id[2],
            current_frames,
            window_size,
            stride,
            episode_filter,
        )


class EpisodeWindowDataset(IterableDataset):
    def __init__(
        self,
        dataset_path: str | Path,
        *,
        window_size: int,
        stride: int = 1,
        shard_glob: str = "*.tar",
        episode_filter: Optional[str] = None,
    ) -> None:
        super().__init__()
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        self.dataset_path = str(dataset_path)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.shard_glob = shard_glob
        self.episode_filter = episode_filter
        self.shards = discover_shards(self.dataset_path, self.shard_glob)

    def __iter__(self) -> Iterator[Dict]:
        return iter_episode_windows(
            shards=self.shards,
            window_size=self.window_size,
            stride=self.stride,
            episode_filter=self.episode_filter,
        )


class EpisodeWindowDataLoader(DataLoader):
    def __init__(
        self,
        dataset_path: str | Path,
        *,
        window_size: int,
        stride: int = 1,
        shard_glob: str = "*.tar",
        episode_filter: Optional[str] = None,
        **kwargs,
    ) -> None:
        if kwargs.get("shuffle"):
            raise ValueError("shuffle=True is not supported for EpisodeWindowDataLoader")

        dataset = EpisodeWindowDataset(
            dataset_path=dataset_path,
            window_size=window_size,
            stride=stride,
            shard_glob=shard_glob,
            episode_filter=episode_filter,
        )
        super().__init__(dataset, **kwargs)
