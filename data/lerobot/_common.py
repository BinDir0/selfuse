from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import numpy as np
import zarr
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DATA_DIR = Path(__file__).resolve().parents[1]
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from zarr_list_utils import parse_zarr_list_entries


VALID_MAPPING_TYPES = {"human", "real_world"}

HUMAN_RAW_KEYS = [
    "image",
    "depth",
    "state/wrist",
    "state/fingertips",
    "action/wrist",
    "action/fingertips",
    "extrinsic",
    "intrinsic",
    "instruction",
    "instruction_num",
    "presence",
]

REAL_WORLD_RAW_KEYS = [
    "image-head",
    "depth-head",
    "state/wrist-head",
    "state/fingertips-head",
    "action/wrist-head",
    "action/fingertips-head",
    "extrinsic",
    "intrinsic/head",
    "instruction",
    "instruction_num",
]

IMAGE_MAX_DIFF = 200
IMAGE_AVG_DIFF = 8.0

_DECORD_IMPORTED = False
_DECORD_MODULE = None
_DECORD_IMPORT_ERROR = None
_DECORD_IMPORT_WARNING_EMITTED = False
_DECORD_GPU_WARNING_EMITTED = False
_DECORD_READERS = {}


@dataclass(frozen=True)
class ManifestEntry:
    zarr_path: str
    mapping_type: str
    dataset_name: str
    target_dataset: str


@dataclass(frozen=True)
class FeatureSpec:
    raw_key: str | None
    feature_name: str
    dtype: str
    shape: tuple[int, ...]
    names: tuple[str, ...] | None
    is_visual: bool = False
    is_string: bool = False

    def as_lerobot_feature(self) -> dict:
        return {
            "dtype": self.dtype,
            "shape": self.shape,
            "names": list(self.names) if self.names is not None else None,
        }


METADATA_FEATURE_SPECS = OrderedDict(
    {
        "source_dataset": FeatureSpec(
            raw_key=None,
            feature_name="source_dataset",
            dtype="string",
            shape=(1,),
            names=None,
            is_string=True,
        ),
        "source_episode_index": FeatureSpec(
            raw_key=None,
            feature_name="source_episode_index",
            dtype="int64",
            shape=(1,),
            names=None,
        ),
        "source_frame_index": FeatureSpec(
            raw_key=None,
            feature_name="source_frame_index",
            dtype="int64",
            shape=(1,),
            names=None,
        ),
    }
)


def parse_zarr_list(zarr_list_path: str) -> list[ManifestEntry]:
    return [
        ManifestEntry(
            zarr_path=entry.zarr_path,
            mapping_type=entry.mapping_type,
            dataset_name=entry.dataset_name,
            target_dataset=entry.target_name,
        )
        for entry in parse_zarr_list_entries(zarr_list_path, "target_dataset")
    ]


def open_zarr_root(zarr_path: str):
    try:
        return zarr.open_consolidated(zarr_path, mode="r")
    except Exception:
        return zarr.open(zarr_path, mode="r")


def get_data_root(src):
    return src["data"] if "data" in src else src


def get_array(data_root, raw_key: str):
    node = data_root
    for part in raw_key.split("/"):
        node = node[part]
    return node


def raw_keys_for_mapping(mapping_type: str) -> list[str]:
    return HUMAN_RAW_KEYS if mapping_type == "human" else REAL_WORLD_RAW_KEYS


def normalize_instruction(value) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        else:
            value = value.tolist()
    if isinstance(value, np.bytes_):
        value = value.tobytes()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        value = str(value)
    return value


def feature_name_from_raw_key(raw_key: str) -> str:
    return raw_key.replace("/", ".")


def infer_entry_feature_specs(entry: ManifestEntry, use_videos: bool) -> OrderedDict[str, FeatureSpec]:
    src = open_zarr_root(entry.zarr_path)
    data_root = get_data_root(src)

    specs: OrderedDict[str, FeatureSpec] = OrderedDict()
    for raw_key in raw_keys_for_mapping(entry.mapping_type):
        try:
            arr = get_array(data_root, raw_key)
        except Exception:
            if raw_key == "presence":
                continue
            raise

        sample = np.asarray(arr[0])
        feature_name = feature_name_from_raw_key(raw_key)

        if raw_key in {"instruction"}:
            specs[feature_name] = FeatureSpec(
                raw_key=raw_key,
                feature_name=feature_name,
                dtype="string",
                shape=(1,),
                names=None,
                is_string=True,
            )
            continue

        if sample.ndim == 3 and sample.dtype == np.uint8:
            height, width, channels = sample.shape
            specs[feature_name] = FeatureSpec(
                raw_key=raw_key,
                feature_name=feature_name,
                dtype="video" if use_videos else "image",
                shape=(channels, height, width),
                names=("channels", "height", "width"),
                is_visual=True,
            )
            continue

        if sample.ndim == 0:
            shape = (1,)
        else:
            shape = tuple(sample.shape)

        specs[feature_name] = FeatureSpec(
            raw_key=raw_key,
            feature_name=feature_name,
            dtype=sample.dtype.name,
            shape=shape,
            names=None,
        )

    return specs


def build_group_feature_specs(
    entries: list[ManifestEntry], use_videos: bool
) -> tuple[OrderedDict[str, FeatureSpec], dict[str, OrderedDict[str, FeatureSpec]]]:
    group_specs: OrderedDict[str, FeatureSpec] = OrderedDict()
    entry_specs: dict[str, OrderedDict[str, FeatureSpec]] = {}

    for entry in entries:
        specs = infer_entry_feature_specs(entry, use_videos)
        entry_specs[entry.dataset_name] = specs
        for feature_name, spec in specs.items():
            existing = group_specs.get(feature_name)
            if existing is None:
                group_specs[feature_name] = spec
                continue

            if (
                existing.dtype != spec.dtype
                or existing.shape != spec.shape
                or existing.names != spec.names
                or existing.is_visual != spec.is_visual
                or existing.is_string != spec.is_string
            ):
                raise ValueError(
                    f"Incompatible feature '{feature_name}' inside target_dataset '{entries[0].target_dataset}': "
                    f"{existing.dtype}/{existing.shape} vs {spec.dtype}/{spec.shape}"
                )

    for feature_name, spec in METADATA_FEATURE_SPECS.items():
        group_specs[feature_name] = spec

    return group_specs, entry_specs


def episode_ranges(src) -> list[tuple[int, int, int]]:
    episode_ends = src["meta/episode_ends"][:]
    ranges = []
    for ep_idx, ep_end in enumerate(episode_ends):
        ep_start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
        ranges.append((ep_start, int(ep_end), ep_idx))
    return ranges


def default_value_for_spec(spec: FeatureSpec):
    if spec.is_visual:
        channels, height, width = spec.shape
        return np.zeros((height, width, channels), dtype=np.uint8)
    if spec.is_string:
        return ""
    return np.zeros(spec.shape, dtype=np.dtype(spec.dtype))


def to_frame_value(value, spec: FeatureSpec):
    if spec.is_visual:
        arr = np.asarray(value)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr
    if spec.is_string:
        return normalize_instruction(value)

    arr = np.asarray(value, dtype=np.dtype(spec.dtype))
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def repo_id_for_dataset(dataset_dir: Path, repo_id_prefix: str) -> str:
    prefix = repo_id_prefix.strip("/")
    return f"{prefix}/{dataset_dir.name}" if prefix else dataset_dir.name


def resolve_dataset_dirs(root_dir: str | Path) -> list[Path]:
    root_dir = Path(root_dir)
    if (root_dir / "meta" / "info.json").is_file():
        return [root_dir]

    dataset_dirs = sorted(
        path for path in root_dir.iterdir() if path.is_dir() and (path / "meta" / "info.json").is_file()
    )
    if not dataset_dirs:
        raise ValueError(f"No LeRobot dataset found under {root_dir}")
    return dataset_dirs


def load_lerobot_dataset(dataset_dir: Path, repo_id_prefix: str) -> LeRobotDataset:
    return LeRobotDataset(
        repo_id_for_dataset(dataset_dir, repo_id_prefix),
        root=dataset_dir,
        download_videos=False,
        video_backend="pyav",
    )


def _load_decord_module():
    global _DECORD_IMPORTED, _DECORD_MODULE, _DECORD_IMPORT_ERROR

    if not _DECORD_IMPORTED:
        _DECORD_IMPORTED = True
        try:
            import decord

            _DECORD_MODULE = decord
        except Exception as exc:
            _DECORD_IMPORT_ERROR = exc
    return _DECORD_MODULE


def _warn_decord_missing(fallback_backend: str):
    global _DECORD_IMPORT_WARNING_EMITTED

    if _DECORD_IMPORT_WARNING_EMITTED:
        return

    detail = f": {_DECORD_IMPORT_ERROR}" if _DECORD_IMPORT_ERROR is not None else ""
    warnings.warn(
        f"decord is unavailable{detail}; falling back to LeRobot video backend '{fallback_backend}'.",
        stacklevel=2,
    )
    _DECORD_IMPORT_WARNING_EMITTED = True


def _warn_decord_gpu_fallback(exc: Exception):
    global _DECORD_GPU_WARNING_EMITTED

    if _DECORD_GPU_WARNING_EMITTED:
        return

    warnings.warn(f"decord GPU decode unavailable ({exc}); falling back to CPU.", stacklevel=2)
    _DECORD_GPU_WARNING_EMITTED = True


def _get_decord_reader(video_path: Path):
    decord = _load_decord_module()
    if decord is None:
        return None, None

    gpu_key = (str(video_path), "gpu")
    if gpu_key in _DECORD_READERS:
        return _DECORD_READERS[gpu_key], "gpu"

    try:
        reader = decord.VideoReader(str(video_path), ctx=decord.gpu(0))
        _DECORD_READERS[gpu_key] = reader
        return reader, "gpu"
    except Exception as exc:
        _warn_decord_gpu_fallback(exc)

    cpu_key = (str(video_path), "cpu")
    if cpu_key not in _DECORD_READERS:
        _DECORD_READERS[cpu_key] = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    return _DECORD_READERS[cpu_key], "cpu"


def _decode_video_frame_with_decord(video_path: Path, timestamp_s: float, fps: float):
    reader, device = _get_decord_reader(video_path)
    if reader is None:
        return None

    frame_index = int(round(timestamp_s * fps))
    frame_index = max(0, min(frame_index, len(reader) - 1))

    try:
        frame = reader[frame_index]
    except Exception as exc:
        if device != "gpu":
            raise
        _warn_decord_gpu_fallback(exc)
        decord = _load_decord_module()
        cpu_key = (str(video_path), "cpu")
        if cpu_key not in _DECORD_READERS:
            _DECORD_READERS[cpu_key] = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        frame = _DECORD_READERS[cpu_key][frame_index]

    if hasattr(frame, "asnumpy"):
        return frame.asnumpy()
    return np.asarray(frame)


def resolve_task_name(tasks, task_idx: int) -> str:
    if tasks is None:
        raise KeyError(f"Cannot resolve task_index={task_idx}: tasks metadata is missing")

    columns = getattr(tasks, "columns", None)
    if columns is not None and "task_index" in columns:
        matches = tasks[tasks["task_index"] == task_idx]
        if len(matches) == 1:
            if "task" in matches.columns:
                return str(matches.iloc[0]["task"])
            return str(matches.index[0])

    if columns is not None and "task" in columns:
        if task_idx in getattr(tasks, "index", []):
            value = tasks.loc[task_idx, "task"]
            if hasattr(value, "iloc"):
                value = value.iloc[0]
            return str(value)
        if 0 <= task_idx < len(tasks):
            return str(tasks.iloc[task_idx]["task"])

    index = getattr(tasks, "index", None)
    if index is not None and 0 <= task_idx < len(index):
        return str(index[task_idx])

    raise KeyError(f"Cannot resolve task_index={task_idx} from tasks metadata")


def get_lerobot_item(dataset: LeRobotDataset, idx: int) -> dict:
    if len(dataset.meta.video_keys) == 0:
        return dataset[idx]

    if _load_decord_module() is None:
        _warn_decord_missing(dataset.video_backend or "pyav")
        return dataset[idx]

    dataset._ensure_hf_dataset_loaded()
    base_item = dict(dataset.hf_dataset[idx])
    ep_idx = to_python_int(base_item["episode_index"])
    current_ts = to_python_float(base_item["timestamp"])

    try:
        video_frames = {}
        for vid_key in dataset.meta.video_keys:
            video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, vid_key)
            episode = dataset.meta.episodes[ep_idx]
            from_timestamp = float(episode[f"videos/{vid_key}/from_timestamp"])
            shifted_ts = from_timestamp + current_ts
            video_frames[vid_key] = _decode_video_frame_with_decord(video_path, shifted_ts, dataset.fps)
    except Exception as exc:
        warnings.warn(
            f"decord video decode failed ({exc}); falling back to LeRobot video backend '{dataset.video_backend}'.",
            stacklevel=2,
        )
        return dataset[idx]

    item = {**video_frames, **base_item}
    task_idx = to_python_int(item["task_index"])
    item["task"] = resolve_task_name(dataset.meta.tasks, task_idx)
    return item


def to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Image.Image):
        return np.asarray(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
        return value
    return np.asarray(value)


def to_python_int(value) -> int:
    arr = to_numpy(value)
    if np.asarray(arr).ndim == 0:
        return int(arr)
    return int(np.asarray(arr).reshape(-1)[0])


def to_python_float(value) -> float:
    arr = to_numpy(value)
    if np.asarray(arr).ndim == 0:
        return float(arr)
    return float(np.asarray(arr).reshape(-1)[0])


def to_python_str(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    arr = to_numpy(value)
    if arr.ndim == 0:
        return str(arr.item())
    return str(arr.tolist())


def image_to_hwc_uint8(value) -> np.ndarray:
    arr = to_numpy(value)
    if arr.ndim != 3:
        raise ValueError(f"Expected image tensor/array with 3 dims, got shape {arr.shape}")
    if arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.5:
            arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def numeric_array_to_shape(value, spec: FeatureSpec) -> np.ndarray:
    arr = to_numpy(value)
    arr = np.asarray(arr, dtype=np.dtype(spec.dtype))
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def actual_source_names_for_dataset_dirs(dataset_dirs: list[Path], repo_id_prefix: str) -> set[str]:
    names = set()
    for dataset_dir in dataset_dirs:
        dataset = load_lerobot_dataset(dataset_dir, repo_id_prefix)
        for row in dataset.hf_dataset:
            names.add(row["source_dataset"])
    return names


def build_source_registry(entries: list[ManifestEntry]) -> dict[str, dict]:
    registry = {}
    for entry in entries:
        src = open_zarr_root(entry.zarr_path)
        data_root = get_data_root(src)
        episode_ends = src["meta/episode_ends"][:]
        episode_starts = np.zeros_like(episode_ends)
        episode_starts[1:] = episode_ends[:-1]
        feature_specs = infer_entry_feature_specs(entry, use_videos=False)
        registry[entry.dataset_name] = {
            "entry": entry,
            "src": src,
            "data_root": data_root,
            "episode_starts": episode_starts,
            "episode_ends": episode_ends,
            "feature_specs": feature_specs,
        }
    return registry


def compare_images(actual, expected) -> tuple[bool, int, float]:
    actual_img = image_to_hwc_uint8(actual)
    expected_img = image_to_hwc_uint8(expected)
    if actual_img.shape != expected_img.shape:
        return False, -1, -1.0
    img_abs = np.abs(actual_img.astype(int) - expected_img.astype(int))
    max_diff = int(img_abs.max())
    avg_diff = float(img_abs.mean())
    ok = max_diff <= IMAGE_MAX_DIFF and avg_diff <= IMAGE_AVG_DIFF
    return ok, max_diff, avg_diff
