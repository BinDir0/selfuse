"""Video discovery and indexing for WebDataset factory directories.

Scans tar shards in a factory directory, groups frames by video,
and provides VideoDescriptor objects for pipeline consumption.
"""

import json
import os
import re
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional

from tqdm import tqdm
from lib.pipeline.datasets.descriptors import ClipDescriptor as VideoDescriptor


# Regex to split frame filename into video_key + frame_number
# e.g. "f001_w012_v00029_i000_f000000.jpg" -> key="f001_w012_v00029_i000", frame="f000000"
_FRAME_RE = re.compile(r'^(.+)_(f\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)
DEFAULT_INDEX_WORKERS = max(1, min(16, os.cpu_count() or 1))
VIDEO_INDEX_FORMAT_VERSION = 2


def _list_tar_shards(factory_dir: str) -> list[str]:
    return sorted(
        entry.name
        for entry in os.scandir(factory_dir)
        if entry.is_file() and entry.name.endswith(".tar")
    )


def _parse_frame_name(name: str):
    """Parse a frame filename into (video_key, frame_sort_key, extension).

    Returns None if the filename doesn't match the expected pattern.
    """
    m = _FRAME_RE.match(name)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def _new_video_summary(shard_file: str, video_key: str, frame_sort: str, ext: str) -> dict:
    return {
        "shard": shard_file,
        "video_name": video_key,
        "frame_count": 0,
        "frame_ext": f".{ext.lower()}",
        "frame_index_width": max(1, len(frame_sort) - 1),
        "frame_start_idx": int(frame_sort[1:]),
    }


def build_video_index(factory_dir: str, *, show_progress: bool = True) -> dict:
    """Scan all tar shards in a factory directory and group frames by video.

    Args:
        factory_dir: Path to a factory directory containing *_shard*.tar files.

    Returns:
        dict with keys:
            "videos": {video_key: {"shard": filename, "frame_count": N, ...}}
            "shards": [list of shard filenames]
            "num_videos": int
            "num_shards": int
    """
    factory_dir = str(factory_dir)
    factory_path = Path(factory_dir)

    # Find all tar shards
    shard_files = _list_tar_shards(factory_dir)

    if not shard_files:
        raise FileNotFoundError(f"No tar shards found in {factory_dir}")

    # Group frames by video_key.
    # Keep this index summary-only: clip -> shard + frame_count (+ naming metadata).
    # Storing every frame name/offset blows up both cold-start time and JSON size.
    videos: Dict[str, dict] = {}

    n_frames_total = 0
    pbar = tqdm(shard_files, desc=f"Scanning {factory_path.name}", unit="shard", disable=not show_progress)
    for shard_file in pbar:
        shard_path = str(factory_path / shard_file)
        with tarfile.open(shard_path, 'r|') as tar:  # streaming mode 'r|'
            for member in tar:
                name = member.name
                if name.endswith(('.jpg', '.jpeg', '.png')):
                    parsed = _parse_frame_name(name)
                    if parsed is None:
                        continue
                    video_key, frame_sort, ext = parsed

                    if video_key not in videos:
                        videos[video_key] = _new_video_summary(shard_file, video_key, frame_sort, ext)
                    videos[video_key]["frame_count"] += 1
                    n_frames_total += 1
                    if show_progress and n_frames_total % 500 == 0:
                        pbar.set_postfix(videos=len(videos), frames=n_frames_total)

    if show_progress:
        pbar.set_postfix(videos=len(videos), frames=n_frames_total)
    pbar.close()

    return {
        "format_version": VIDEO_INDEX_FORMAT_VERSION,
        "videos": videos,
        "shards": shard_files,
        "num_videos": len(videos),
        "num_shards": len(shard_files),
    }


def _index_cache_path(factory_dir: str) -> str:
    return os.path.join(factory_dir, "_video_index.json")


def _is_index_stale(index: dict, factory_dir: str) -> bool:
    """Check if cached index is stale by comparing shard file list or schema."""
    current_shards = _list_tar_shards(factory_dir)
    if current_shards != index.get("shards", []):
        return True
    if int(index.get("format_version", 0)) != VIDEO_INDEX_FORMAT_VERSION:
        return True
    videos = index.get("videos", {})
    if videos:
        first_video = next(iter(videos.values()))
        if "frame_count" not in first_video or "shard" not in first_video:
            return True
    return False


def load_or_build_index(factory_dir: str, force_rebuild: bool = False, *, show_progress: bool = True) -> tuple[dict, dict]:
    """Load cached video index, or build and cache it if missing/stale.

    Args:
        factory_dir: Path to factory directory.
        force_rebuild: Force rebuilding even if cache exists.

    Returns:
        Tuple of (video index dict, summary dict).
    """
    cache_path = _index_cache_path(factory_dir)
    summary = {
        "factory_dir": str(Path(factory_dir).resolve()),
        "factory_name": Path(factory_dir).name,
        "cache_hit": False,
        "built": False,
        "elapsed_sec": 0.0,
        "num_videos": 0,
        "num_shards": 0,
    }
    start_time = time.monotonic()

    if not force_rebuild and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                index = json.load(f)
            if not _is_index_stale(index, factory_dir):
                summary["cache_hit"] = True
                summary["elapsed_sec"] = time.monotonic() - start_time
                summary["num_videos"] = int(index.get("num_videos", len(index.get("videos", {}))))
                summary["num_shards"] = int(index.get("num_shards", len(index.get("shards", []))))
                return index, summary
        except (json.JSONDecodeError, KeyError):
            pass

    # Build fresh index
    index = build_video_index(factory_dir, show_progress=show_progress)
    summary["built"] = True

    # Cache it
    try:
        with open(cache_path, 'w') as f:
            json.dump(index, f, ensure_ascii=False)
    except OSError:
        pass  # Non-fatal: can't write cache (e.g. read-only filesystem)

    summary["elapsed_sec"] = time.monotonic() - start_time
    summary["num_videos"] = int(index.get("num_videos", len(index.get("videos", {}))))
    summary["num_shards"] = int(index.get("num_shards", len(index.get("shards", []))))
    return index, summary


def collect_videos_from_factory(
    factory_dir: str,
    *,
    force_rebuild: bool = False,
    show_progress: bool = True,
) -> List[VideoDescriptor]:
    """Collect all videos from a factory directory as VideoDescriptors.

    Args:
        factory_dir: Path to factory directory containing tar shards.

    Returns:
        List of VideoDescriptor, one per video found.
    """
    factory_dir = str(Path(factory_dir).resolve())
    index, _summary = load_or_build_index(factory_dir, force_rebuild=force_rebuild, show_progress=show_progress)

    descriptors = []
    for video_key, info in sorted(index["videos"].items()):
        shard_path = os.path.join(factory_dir, info["shard"])
        seq_folder = os.path.join(factory_dir, "outputs", video_key)

        desc = VideoDescriptor(
            clip_id=video_key,
            clip_name=info["video_name"],
            storage_kind="tar_shard",
            root_dir=factory_dir,
            shard_path=shard_path,
            frame_names=[],
            seq_folder=seq_folder,
            frame_offsets=None,
            frame_count_override=int(info.get("frame_count", 0)),
            extra={
                "frame_ext": info.get("frame_ext", ".jpg"),
                "frame_index_width": int(info.get("frame_index_width", 6)),
                "frame_start_idx": int(info.get("frame_start_idx", 0)),
            },
        )
        descriptors.append(desc)

    return descriptors


def _load_factory_descriptors(factory_dir: str, *, force_rebuild: bool, show_progress: bool):
    factory_dir = str(Path(factory_dir).resolve())
    index, summary = load_or_build_index(factory_dir, force_rebuild=force_rebuild, show_progress=show_progress)

    descriptors = []
    for video_key, info in sorted(index["videos"].items()):
        shard_path = os.path.join(factory_dir, info["shard"])
        seq_folder = os.path.join(factory_dir, "outputs", video_key)

        desc = VideoDescriptor(
            clip_id=video_key,
            clip_name=info.get("video_name") or video_key,
            storage_kind="tar_shard",
            root_dir=factory_dir,
            shard_path=shard_path,
            frame_names=[],
            seq_folder=seq_folder,
            frame_offsets=None,
            frame_count_override=int(info.get("frame_count", 0)),
            extra={
                "frame_ext": info.get("frame_ext", ".jpg"),
                "frame_index_width": int(info.get("frame_index_width", 6)),
                "frame_start_idx": int(info.get("frame_start_idx", 0)),
            },
        )
        descriptors.append(desc)

    summary["num_descriptors"] = len(descriptors)
    return descriptors, summary


def _resolve_index_workers(factory_dirs: List[str], workers: int | None) -> int:
    if not factory_dirs:
        return 1
    if workers is not None:
        return max(1, min(int(workers), len(factory_dirs)))
    env_value = os.getenv("HAWOR_SCAN_WORKERS", "").strip()
    if env_value:
        try:
            return max(1, min(int(env_value), len(factory_dirs)))
        except ValueError:
            pass
    return max(1, min(DEFAULT_INDEX_WORKERS, len(factory_dirs)))


def collect_videos_from_factories(
    factory_dirs: List[str],
    *,
    workers: int | None = None,
    force_rebuild: bool = False,
) -> List[VideoDescriptor]:
    """Collect videos from multiple factory directories.

    Results are cached per-factory at ``<factory_dir>/_video_index.json``.
    This helper loads/builds those indexes in parallel to reduce manifest scan latency.
    """
    if not factory_dirs:
        return []

    if len(factory_dirs) == 1:
        descriptors, summary = _load_factory_descriptors(
            factory_dirs[0],
            force_rebuild=force_rebuild,
            show_progress=True,
        )
        tqdm.write(
            f"[index] {summary['factory_name']}: "
            f"{'cache' if summary['cache_hit'] else 'built'} "
            f"videos={summary['num_videos']} shards={summary['num_shards']} "
            f"elapsed={summary['elapsed_sec']:.1f}s"
        )
        return descriptors

    resolved_workers = _resolve_index_workers(factory_dirs, workers)
    results_by_idx: Dict[int, List[VideoDescriptor]] = {}
    cache_hits = 0
    built = 0
    total_videos = 0
    with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
        future_to_idx = {
            executor.submit(
                _load_factory_descriptors,
                factory_dir,
                force_rebuild=force_rebuild,
                show_progress=False,
            ): idx
            for idx, factory_dir in enumerate(factory_dirs)
        }
        with tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Load shard indexes", unit="factory") as pbar:
            for future in pbar:
                idx = future_to_idx[future]
                descriptors, summary = future.result()
                results_by_idx[idx] = descriptors
                cache_hits += int(summary["cache_hit"])
                built += int(summary["built"])
                total_videos += int(summary["num_videos"])
                pbar.set_postfix(
                    cache=cache_hits,
                    built=built,
                    videos=total_videos,
                    workers=resolved_workers,
                )

    all_descriptors = []
    for idx in range(len(factory_dirs)):
        all_descriptors.extend(results_by_idx[idx])
    return all_descriptors
