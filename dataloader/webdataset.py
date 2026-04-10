"""
Provides a webdataset dataloader that produce:
video (B,T,3,H,W)
existence (B,T,2)
MANO (B,T,·)
    left_translation : B x T x 3
    right_translation : B x T x 3
    left_rot6 : B x T x 6
    right_rot6 : B x T x 6
    left_hand_pose45 : B x T x 45   # MANO 手指 PCA，每手 45
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

# lowdim.npy float32，长度 148 或 128（无 shape 段时为 128）
lowdim = concat([
    wrist,     # 18: left_t3, right_t3, left_rot6, right_rot6（wrist→world）
    hand,      # 90: 左 45 + 右 45，MANO 手指 PCA
    shape,     # 20: 仅 148 维布局；左 beta10 + 右 beta10
    extrinsic, # 16: World→Cam，4x4 展平
    intrinsic, # 4: fx, fy, cx, cy
])

"""

from __future__ import annotations

import multiprocessing as mp
import random
import tarfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

try:
    from .utils import (
        discover_shards,
        decode_npy,
        decode_image_jpg,
        decode_json,
        load_episode_name_set,
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
        load_episode_name_set,
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
DEFAULT_WINDOW_SHUFFLE_WINDOWS = True
DEFAULT_WINDOW_SHUFFLE_BUFFER_SIZE = 1000
DEFAULT_WINDOW_SHUFFLE_SEED = 42


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

    # Keep only fixed meta keys so default_collate can batch mixed datasets safely.
    def compact_meta(frame: Dict) -> Dict[str, object]:
        meta = frame["meta"]
        return {
            "dataset_name": str(meta["dataset_name"]),
            "episode_name": str(meta["episode_name"]),
            "episode_idx": int(meta["episode_idx"]),
            "frame_idx": int(meta["frame_idx"]),
            "presence": int(meta["presence"]),
        }

    video = concat_imgs(window_frames)
    existence = np.asarray([convert_presence(frame["presence"]) for frame in window_frames], dtype=np.int8)

    return {
        # Basic information
        "dataset_name": window_frames[0]["dataset_name"],
        "episode_name": episode_name,
        "episode_idx": int(window_frames[0]["episode_idx"]),
        "sample_keys": [frame["sample_key"] for frame in window_frames],
        "frame_indices": frame_indices,
        "meta": [compact_meta(frame) for frame in window_frames],
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


def iter_normalized_episode_names_in_shard(shard_path: Path) -> Iterator[str]:
    """Yield normalized episode names by reading only meta.json per frame (no images/lowdim)."""
    with tarfile.open(shard_path, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parsed = parse_member_name(member.name)
            if parsed is None:
                continue
            sample_key, suffix = parsed
            if suffix != "meta.json":
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            raw = extracted.read()
            key_for_msg = sample_key if sample_key is not None else member.name
            meta = validate_converted_meta(decode_json(raw), key_for_msg)
            yield str(meta["episode_name_normalized"])


def _episode_keep(
    episode_name: str,
    episode_filter: Optional[str],
    episode_allowlist: Optional[frozenset[str]],
) -> bool:
    if episode_allowlist is not None:
        return episode_name in episode_allowlist
    if episode_filter is not None:
        return episode_name == episode_filter
    return True


def emit_episode_windows(
    episode_name: str,
    episode_frames: List[Dict],
    window_size: int,
    stride: int,
    episode_filter: Optional[str],
    episode_allowlist: Optional[frozenset[str]],
    *,
    shuffle_windows: bool = False,
    rng_windows: Optional[random.Random] = None,
) -> Iterator[Dict]:
    if not episode_frames:
        return
    if not _episode_keep(episode_name, episode_filter, episode_allowlist):
        return

    ordered_frames = sorted(episode_frames, key=lambda item: item["frame_idx"])
    if len(ordered_frames) < window_size:
        return

    starts = list(range(0, len(ordered_frames) - window_size + 1, stride))
    if shuffle_windows and rng_windows is not None:
        rng_windows.shuffle(starts)
    for start in starts:
        yield build_window_batch(episode_name, ordered_frames, start, window_size)


def iter_episode_windows(
    shards: List[Path],
    window_size: int,
    stride: int,
    episode_filter: Optional[str],
    episode_allowlist: Optional[frozenset[str]] = None,
    *,
    shuffle_windows: bool = False,
    rng_windows: Optional[random.Random] = None,
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
                    episode_allowlist,
                    shuffle_windows=shuffle_windows,
                    rng_windows=rng_windows,
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
            episode_allowlist,
            shuffle_windows=shuffle_windows,
            rng_windows=rng_windows,
        )


class EpisodeWindowDataset(IterableDataset):
    def __init__(
        self,
        dataset_path: str | Path,
        *,
        dataset_sources: Optional[List[Dict[str, str]]] = None,
        window_size: int,
        stride: int = 1,
        shard_glob: str = "*.tar",
        episode_filter: Optional[str] = None,
        episode_list_file: Optional[str] = None,
        dist_rank: int = 0,
        dist_world_size: int = 1,
        ddp_read_all_shards: bool = False,
        shuffle_shards: bool = True,
        shuffle_windows: bool = DEFAULT_WINDOW_SHUFFLE_WINDOWS,
        shuffle_buffer_size: int = DEFAULT_WINDOW_SHUFFLE_BUFFER_SIZE,
        shuffle_seed: int = DEFAULT_WINDOW_SHUFFLE_SEED,
    ) -> None:
        super().__init__()
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if episode_list_file and episode_filter is not None:
            raise ValueError("use either episode_list_file or episode_filter, not both")
        if dataset_sources is not None and (episode_list_file or episode_filter):
            raise ValueError("do not pass episode_list_file/episode_filter when dataset_sources is provided")
        if shuffle_buffer_size < 0:
            raise ValueError(f"shuffle_buffer_size must be >= 0, got {shuffle_buffer_size}")
        ws = int(dist_world_size)
        rk = int(dist_rank)
        if ws < 1 or rk < 0 or rk >= ws:
            raise ValueError(f"invalid dist rank/world_size: rank={rk}, world_size={ws}")

        self.window_size = int(window_size)
        self.stride = int(stride)
        self.dist_rank = rk
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_windows = bool(shuffle_windows)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.shuffle_seed = int(shuffle_seed)

        sources: List[Dict[str, object]] = []
        if dataset_sources is not None:
            for idx, src in enumerate(dataset_sources):
                data_path = str(src.get("data_path", "")).strip()
                if not data_path:
                    raise ValueError(f"dataset_sources[{idx}].data_path is required")
                src_glob = str(src.get("shard_glob", "")).strip() or shard_glob
                src_episodes_file = str(src.get("episodes_file", "")).strip()
                src_episode_filter = str(src.get("episode_filter", "")).strip()
                if src_episodes_file and src_episode_filter:
                    raise ValueError(f"dataset_sources[{idx}]: use either episodes_file or episode_filter, not both")

                src_allowlist: Optional[frozenset[str]] = None
                if src_episodes_file:
                    src_allowlist = load_episode_name_set(src_episodes_file)

                shards = discover_shards(data_path, src_glob)
                if ws > 1 and not ddp_read_all_shards:
                    shards = shards[rk::ws]

                sources.append(
                    {
                        "name": str(src.get("name", f"dataset_{idx}")),
                        "episode_filter": src_episode_filter or None,
                        "episode_allowlist": src_allowlist,
                        "shards": shards,
                    }
                )
        else:
            allowlist: Optional[frozenset[str]] = None
            if episode_list_file:
                allowlist = load_episode_name_set(episode_list_file)
            shards = discover_shards(str(dataset_path), shard_glob)
            if ws > 1 and not ddp_read_all_shards:
                shards = shards[rk::ws]
            sources.append(
                {
                    "name": "dataset_0",
                    "episode_filter": episode_filter,
                    "episode_allowlist": allowlist,
                    "shards": shards,
                }
            )

        self.sources = sources
        # Shared across forked DataLoader workers (Linux) so set_epoch() affects all workers.
        self._epoch_val = mp.Value("i", 0)

    def set_epoch(self, epoch: int) -> None:
        """Bumps RNG for shard / window shuffling each training epoch (1-based epoch index ok)."""
        with self._epoch_val.get_lock():
            self._epoch_val.value = int(epoch)

    def _iter_source_windows(
        self,
        source: Dict[str, object],
        source_shards: List[Path],
        rng_windows: Optional[random.Random],
    ) -> Iterator[Dict]:
        current_episode_id: Optional[Tuple[str, int, str]] = None
        current_frames: List[Dict] = []
        source_episode_filter = source["episode_filter"]
        source_episode_allowlist = source["episode_allowlist"]

        for shard_path in source_shards:
            for sample_key, fields in iter_lowdim_samples_in_shard(shard_path):
                sample = decode_lowdim_sample(sample_key, fields)
                sample_episode_id = episode_identity(sample)

                if current_episode_id is None:
                    current_episode_id = sample_episode_id

                if sample_episode_id != current_episode_id:
                    yield from emit_episode_windows(
                        current_episode_id[2],
                        current_frames,
                        self.window_size,
                        self.stride,
                        source_episode_filter,
                        source_episode_allowlist,
                        shuffle_windows=self.shuffle_windows,
                        rng_windows=rng_windows,
                    )
                    current_episode_id = sample_episode_id
                    current_frames = []

                current_frames.append(sample)

        if current_episode_id is not None:
            yield from emit_episode_windows(
                current_episode_id[2],
                current_frames,
                self.window_size,
                self.stride,
                source_episode_filter,
                source_episode_allowlist,
                shuffle_windows=self.shuffle_windows,
                rng_windows=rng_windows,
            )

    @staticmethod
    def _iter_round_robin_sources(source_iters: List[Iterator[Dict]]) -> Iterator[Dict]:
        if not source_iters:
            return

        exhausted = [False] * len(source_iters)
        num_exhausted = 0
        source_idx = 0

        while num_exhausted < len(source_iters):
            if exhausted[source_idx]:
                source_idx = (source_idx + 1) % len(source_iters)
                continue

            try:
                yield next(source_iters[source_idx])
            except StopIteration:
                exhausted[source_idx] = True
                num_exhausted += 1

            source_idx = (source_idx + 1) % len(source_iters)

    def _iter_rank_local_windows(self, epoch_i: int, rng_windows: Optional[random.Random]) -> Iterator[Dict]:
        source_iters: List[Iterator[Dict]] = []
        for source_idx, source in enumerate(self.sources):
            source_shards = list(source["shards"])
            if self.shuffle_shards:
                src_rng = random.Random(self.shuffle_seed + epoch_i * 1_000_003 + source_idx * 9_176)
                src_rng.shuffle(source_shards)
            source_iters.append(self._iter_source_windows(source, source_shards, rng_windows))

        yield from self._iter_round_robin_sources(source_iters)

    @staticmethod
    def _iter_worker_partition(
        windows: Iterator[Dict], worker_id: int, num_workers: int
    ) -> Iterator[Dict]:
        for window_idx, window in enumerate(windows):
            if window_idx % num_workers == worker_id:
                yield window

    @staticmethod
    def _iter_shuffle_buffer(
        windows: Iterator[Dict], buffer_size: int, rng: np.random.Generator
    ) -> Iterator[Dict]:
        if buffer_size <= 1:
            yield from windows
            return

        buffer: list[Dict] = []
        for window in windows:
            if len(buffer) < buffer_size:
                buffer.append(window)
                continue
            pop_idx = int(rng.integers(len(buffer)))
            yield buffer[pop_idx]
            buffer[pop_idx] = window

        while buffer:
            pop_idx = int(rng.integers(len(buffer)))
            yield buffer.pop(pop_idx)

    def __iter__(self) -> Iterator[Dict]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        with self._epoch_val.get_lock():
            epoch_i = int(self._epoch_val.value)

        rng_win: Optional[random.Random] = None
        if self.shuffle_windows:
            rng_win = random.Random(
                self.shuffle_seed + epoch_i * 999_983 + self.dist_rank * 9_176 + worker_id * 50_051
            )

        windows = self._iter_rank_local_windows(epoch_i=epoch_i, rng_windows=rng_win)
        if num_workers > 1:
            windows = self._iter_worker_partition(windows, worker_id, num_workers)

        if self.shuffle_windows and self.shuffle_buffer_size > 1:
            effective_seed = (
                self.shuffle_seed
                + epoch_i * 1_000_003
                + self.dist_rank * 9_176
                + worker_id
            )
            rng = np.random.default_rng(effective_seed)
            windows = self._iter_shuffle_buffer(windows, self.shuffle_buffer_size, rng)

        yield from windows


class EpisodeWindowDataLoader(DataLoader):
    def __init__(
        self,
        dataset_path: str | Path,
        *,
        dataset_sources: Optional[List[Dict[str, str]]] = None,
        window_size: int,
        stride: int = 1,
        shard_glob: str = "*.tar",
        episode_filter: Optional[str] = None,
        episode_list_file: Optional[str] = None,
        dist_rank: int = 0,
        dist_world_size: int = 1,
        ddp_read_all_shards: bool = False,
        shuffle: bool = True,
        shuffle_windows: bool = DEFAULT_WINDOW_SHUFFLE_WINDOWS,
        shuffle_buffer_size: int = DEFAULT_WINDOW_SHUFFLE_BUFFER_SIZE,
        shuffle_seed: int = DEFAULT_WINDOW_SHUFFLE_SEED,
        **kwargs,
    ) -> None:
        _dup = kwargs.pop("shuffle", None)
        if _dup is not None:
            raise ValueError(
                "Pass shuffle= to EpisodeWindowDataLoader(..., shuffle=...); "
                "do not pass DataLoader(shuffle=)."
            )

        dataset = EpisodeWindowDataset(
            dataset_path=dataset_path,
            dataset_sources=dataset_sources,
            window_size=window_size,
            stride=stride,
            shard_glob=shard_glob,
            episode_filter=episode_filter,
            episode_list_file=episode_list_file,
            dist_rank=dist_rank,
            dist_world_size=dist_world_size,
            ddp_read_all_shards=ddp_read_all_shards,
            shuffle_shards=shuffle,
            shuffle_windows=shuffle_windows,
            shuffle_buffer_size=shuffle_buffer_size,
            shuffle_seed=shuffle_seed,
        )
        super().__init__(dataset, **kwargs)
