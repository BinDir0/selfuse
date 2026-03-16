"""Video discovery and indexing for WebDataset factory directories.

Scans tar shards in a factory directory, groups frames by video,
and provides VideoDescriptor objects for pipeline consumption.
"""

import json
import os
import re
import tarfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional


@dataclass
class VideoDescriptor:
    """Describes a single video within a WebDataset factory directory."""
    video_key: str          # e.g. "f001_w012_v00029_i000" — unique ID
    video_name: str         # e.g. "factory_001_worker_012_0029" — from JSON metadata
    factory_dir: str        # e.g. "/share_data/.../factory001"
    shard_path: str         # absolute path to the tar containing this video
    frame_names: List[str]  # sorted list of JPEG filenames within the tar
    seq_folder: str         # output directory for this video

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> 'VideoDescriptor':
        d = json.loads(s)
        return cls(**d)

    @classmethod
    def from_dict(cls, d: dict) -> 'VideoDescriptor':
        return cls(**d)


# Regex to split frame filename into video_key + frame_number
# e.g. "f001_w012_v00029_i000_f000000.jpg" -> key="f001_w012_v00029_i000", frame="f000000"
_FRAME_RE = re.compile(r'^(.+)_(f\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)


def _parse_frame_name(name: str):
    """Parse a frame filename into (video_key, frame_sort_key, extension).

    Returns None if the filename doesn't match the expected pattern.
    """
    m = _FRAME_RE.match(name)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def build_video_index(factory_dir: str) -> dict:
    """Scan all tar shards in a factory directory and group frames by video.

    Args:
        factory_dir: Path to a factory directory containing *_shard*.tar files.

    Returns:
        dict with keys:
            "videos": {video_key: {"shard": filename, "frames": [...], "num_frames": N, "video_name": str}}
            "shards": [list of shard filenames]
            "num_videos": int
            "num_shards": int
    """
    factory_dir = str(factory_dir)
    factory_path = Path(factory_dir)

    # Find all tar shards
    shard_files = sorted([
        f for f in os.listdir(factory_dir)
        if f.endswith('.tar')
    ])

    if not shard_files:
        raise FileNotFoundError(f"No tar shards found in {factory_dir}")

    # Group frames by video_key
    videos: Dict[str, dict] = {}

    for shard_file in shard_files:
        shard_path = str(factory_path / shard_file)
        # Single pass: iterate members streaming (no getnames() full scan),
        # and extract JSON metadata inline to avoid re-opening the tar.
        # json_meta: buffer for video_name read from JSON that appeared
        # before the first frame of that video in the tar stream.
        json_meta: Dict[str, str] = {}
        with tarfile.open(shard_path, 'r|') as tar:  # streaming mode 'r|'
            for member in tar:
                name = member.name
                if name.endswith(('.jpg', '.jpeg', '.png')):
                    parsed = _parse_frame_name(name)
                    if parsed is None:
                        continue
                    video_key, frame_sort, ext = parsed

                    if video_key not in videos:
                        videos[video_key] = {
                            "shard": shard_file,
                            "frames": [],
                            "video_name": json_meta.pop(video_key, ""),
                        }
                    videos[video_key]["frames"].append(name)

                elif name.endswith('.json') and member.isreg():
                    base = name[:-5]  # remove .json
                    parsed = _parse_frame_name(base + '.jpg')  # fake ext for parsing
                    if parsed:
                        video_key = parsed[0]
                        try:
                            f = tar.extractfile(member)
                            if f:
                                meta = json.loads(f.read())
                                vn = meta.get("video_name", "")
                                if vn:
                                    if video_key in videos:
                                        # Video already seen, fill directly
                                        if not videos[video_key]["video_name"]:
                                            videos[video_key]["video_name"] = vn
                                    else:
                                        # Video not seen yet, buffer for later
                                        json_meta[video_key] = vn
                        except Exception:
                            pass

    # Sort frames within each video and set num_frames
    for video_key, info in videos.items():
        info["frames"] = sorted(info["frames"])
        info["num_frames"] = len(info["frames"])
        if not info["video_name"]:
            info["video_name"] = video_key

    return {
        "videos": videos,
        "shards": shard_files,
        "num_videos": len(videos),
        "num_shards": len(shard_files),
    }


def _index_cache_path(factory_dir: str) -> str:
    return os.path.join(factory_dir, "_video_index.json")


def _is_index_stale(index: dict, factory_dir: str) -> bool:
    """Check if cached index is stale by comparing shard file list."""
    current_shards = sorted([
        f for f in os.listdir(factory_dir)
        if f.endswith('.tar')
    ])
    return current_shards != index.get("shards", [])


def load_or_build_index(factory_dir: str, force_rebuild: bool = False) -> dict:
    """Load cached video index, or build and cache it if missing/stale.

    Args:
        factory_dir: Path to factory directory.
        force_rebuild: Force rebuilding even if cache exists.

    Returns:
        Video index dict.
    """
    cache_path = _index_cache_path(factory_dir)

    if not force_rebuild and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                index = json.load(f)
            if not _is_index_stale(index, factory_dir):
                return index
        except (json.JSONDecodeError, KeyError):
            pass

    # Build fresh index
    index = build_video_index(factory_dir)

    # Cache it
    try:
        with open(cache_path, 'w') as f:
            json.dump(index, f, ensure_ascii=False)
    except OSError:
        pass  # Non-fatal: can't write cache (e.g. read-only filesystem)

    return index


def collect_videos_from_factory(factory_dir: str) -> List[VideoDescriptor]:
    """Collect all videos from a factory directory as VideoDescriptors.

    Args:
        factory_dir: Path to factory directory containing tar shards.

    Returns:
        List of VideoDescriptor, one per video found.
    """
    factory_dir = str(Path(factory_dir).resolve())
    index = load_or_build_index(factory_dir)

    descriptors = []
    for video_key, info in sorted(index["videos"].items()):
        shard_path = os.path.join(factory_dir, info["shard"])
        seq_folder = os.path.join(factory_dir, "outputs", video_key)

        desc = VideoDescriptor(
            video_key=video_key,
            video_name=info["video_name"],
            factory_dir=factory_dir,
            shard_path=shard_path,
            frame_names=info["frames"],
            seq_folder=seq_folder,
        )
        descriptors.append(desc)

    return descriptors


def collect_videos_from_factories(factory_dirs: List[str]) -> List[VideoDescriptor]:
    """Collect videos from multiple factory directories."""
    all_descriptors = []
    for factory_dir in factory_dirs:
        descriptors = collect_videos_from_factory(factory_dir)
        all_descriptors.extend(descriptors)
    return all_descriptors
