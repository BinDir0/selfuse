"""Video discovery and indexing for WebDataset factory directories.

Scans tar shards in a factory directory, groups frames by video,
and provides VideoDescriptor objects for pipeline consumption.
"""

import json
import os
import re
import tarfile
from pathlib import Path
from typing import List, Dict, Optional

from tqdm import tqdm
from lib.pipeline.datasets.descriptors import ClipDescriptor as VideoDescriptor


# Regex to split frame filename into video_key + frame_number
# e.g. "f001_w012_v00029_i000_f000000.jpg" -> key="f001_w012_v00029_i000", frame="f000000"
_FRAME_RE = re.compile(r"^(.+)_(f\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


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
    shard_files = sorted([f for f in os.listdir(factory_dir) if f.endswith(".tar")])

    if not shard_files:
        raise FileNotFoundError(f"No tar shards found in {factory_dir}")

    # Group frames by video_key
    videos: Dict[str, dict] = {}

    n_frames_total = 0
    pbar = tqdm(shard_files, desc="Scanning shards", unit="shard")
    for shard_file in pbar:
        shard_path = str(factory_path / shard_file)
        json_meta: Dict[str, str] = {}
        with tarfile.open(shard_path, "r|") as tar:  # streaming mode 'r|'
            for member in tar:
                name = member.name
                if name.endswith((".jpg", ".jpeg", ".png")):
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
                    videos[video_key]["frames"].append(
                        {
                            "name": name,
                            "offset": member.offset_data,
                            "size": member.size,
                        }
                    )
                    n_frames_total += 1
                    if n_frames_total % 500 == 0:
                        pbar.set_postfix(videos=len(videos), frames=n_frames_total)

                elif name.endswith(".json") and member.isreg():
                    base = name[:-5]  # remove .json
                    parsed = _parse_frame_name(base + ".jpg")  # fake ext for parsing
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

    pbar.set_postfix(videos=len(videos), frames=n_frames_total)
    pbar.close()

    # Sort frames within each video and set num_frames
    for video_key, info in videos.items():
        info["frames"] = sorted(info["frames"], key=lambda f: f["name"])
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
    """Check if cached index is stale by comparing shard file list or missing offsets."""
    current_shards = sorted([f for f in os.listdir(factory_dir) if f.endswith(".tar")])
    if current_shards != index.get("shards", []):
        return True
    # Check if index has frame offsets (new format with dicts instead of strings)
    videos = index.get("videos", {})
    if videos:
        first_video = next(iter(videos.values()))
        frames = first_video.get("frames", [])
        if frames and isinstance(frames[0], str):
            return True  # Old format without offsets — rebuild
    return False


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
            with open(cache_path, "r") as f:
                index = json.load(f)
            if not _is_index_stale(index, factory_dir):
                return index
        except (json.JSONDecodeError, KeyError):
            pass

    # Build fresh index
    index = build_video_index(factory_dir)

    # Cache it
    try:
        with open(cache_path, "w") as f:
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

        frames = info["frames"]
        # frames is List[dict] with keys: name, offset, size
        frame_names = [f["name"] for f in frames]
        frame_offsets = [[f["offset"], f["size"]] for f in frames]

        desc = VideoDescriptor(
            clip_id=video_key,
            clip_name=info["video_name"],
            storage_kind="tar_shard",
            root_dir=factory_dir,
            shard_path=shard_path,
            frame_names=frame_names,
            seq_folder=seq_folder,
            frame_offsets=frame_offsets,
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
