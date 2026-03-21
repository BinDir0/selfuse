"""Episode discovery and indexing helpers for WebDataset export."""

import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm


def discover_episodes(input_dir, episode_list=None, max_episodes=None, cache_file=None, require_world_res=True):
    """Discover episode directories under BuildAI processed output."""
    if cache_file is None:
        cache_file = os.path.join(input_dir, "_vla_episodes_cache.json")

    if require_world_res and os.path.exists(cache_file):
        print(f"Loading cached episode list from {cache_file}")
        with open(cache_file) as f:
            episodes = json.load(f)
        for i, ep in enumerate(episodes):
            ep["episode_index"] = i
        if max_episodes:
            episodes = episodes[:max_episodes]
        print(f"  {len(episodes)} episodes from cache")
        return episodes

    print("Scanning for episodes...")
    episodes = []

    if episode_list and os.path.exists(episode_list):
        with open(episode_list) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                crop_dir = line[:-4] if line.endswith(".mp4") else line
                if require_world_res and not os.path.exists(os.path.join(crop_dir, "world_space_res.pth")):
                    continue
                episodes.append({"crop_dir": crop_dir, "episode_id": Path(crop_dir).name})
    else:
        input_path = Path(input_dir)
        for extracted_dir in sorted(input_path.glob("*/*/processed/*/extracted_images")):
            crop_dir = str(extracted_dir.parent)
            if require_world_res and not os.path.exists(os.path.join(crop_dir, "world_space_res.pth")):
                continue
            episodes.append({"crop_dir": crop_dir, "episode_id": Path(crop_dir).name})

    for i, ep in enumerate(episodes):
        ep["episode_index"] = i

    if require_world_res:
        try:
            with open(cache_file, "w") as f:
                json.dump(episodes, f, ensure_ascii=False)
            print(f"  Cached {len(episodes)} episodes to {cache_file}")
        except OSError as e:
            print(f"  Warning: failed to write episode cache: {e}")

    if max_episodes:
        episodes = episodes[:max_episodes]
    return episodes


def load_or_build_frame_index(extracted_dir, rescan=False):
    """Load cached frame index or build it from extracted_images."""
    cache_path = os.path.join(extracted_dir, "_frame_index.json")

    if not rescan and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            return {int(k): os.path.join(extracted_dir, v) for k, v in cached.items()}
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    index = {}
    try:
        files = os.listdir(extracted_dir)
    except OSError:
        return index

    for name in files:
        if not name.endswith(".jpg"):
            continue
        stem = name[:-4]
        try:
            frame_idx = int(stem)
        except ValueError:
            continue
        index[frame_idx] = os.path.join(extracted_dir, name)

    try:
        with open(cache_path, "w") as f:
            json.dump({str(k): os.path.basename(v) for k, v in sorted(index.items())}, f)
    except OSError:
        pass

    return index


def load_episode_stats(ep, rescan_frame_index=False):
    """Augment one episode with sequence length and valid frame ids."""
    extracted_dir = os.path.join(ep["crop_dir"], "extracted_images")
    frame_index = load_or_build_frame_index(extracted_dir, rescan=rescan_frame_index)
    if not frame_index:
        return None

    world_res_path = os.path.join(ep["crop_dir"], "world_space_res.pth")
    try:
        pred_trans, *_ = joblib.load(world_res_path)
    except Exception as e:
        print(f"  Skip {ep['episode_id']}: failed to load world_space_res.pth for stats: {e}")
        return None

    seq_len = int(np.array(pred_trans).shape[1])
    frame_ids = sorted(frame_idx for frame_idx in frame_index if frame_idx < seq_len)
    if not frame_ids:
        return None

    ep_with_stats = dict(ep)
    ep_with_stats["sequence_length"] = seq_len
    ep_with_stats["frame_ids"] = frame_ids
    ep_with_stats["num_valid_frames"] = len(frame_ids)
    return ep_with_stats


def discover_episode_stats(episodes, rescan_frame_index=False):
    """Collect stats for all valid episodes."""
    stats = []
    for ep in tqdm(episodes, desc="Episode stats"):
        ep_stats = load_episode_stats(ep, rescan_frame_index=rescan_frame_index)
        if ep_stats is None:
            continue
        stats.append(ep_stats)
    return stats


def repeat_episode_stats(episodes, repeat_count):
    """Repeat episodes as [1..N, 1..N, ...] with fresh episode indices."""
    if repeat_count <= 1:
        repeated = []
        for new_index, ep in enumerate(episodes):
            ep_copy = dict(ep)
            ep_copy["source_episode_index"] = ep["episode_index"]
            ep_copy["episode_index"] = new_index
            repeated.append(ep_copy)
        return repeated

    repeated = []
    for repeat_idx in range(repeat_count):
        for ep in episodes:
            ep_copy = dict(ep)
            ep_copy["source_episode_index"] = ep["episode_index"]
            ep_copy["repeat_index"] = repeat_idx
            ep_copy["episode_index"] = len(repeated)
            repeated.append(ep_copy)
    return repeated


def get_episode_feature_cache_path(ep, feature_cache_dir):
    """Return the per-episode feature cache path."""
    crop_hash = hashlib.md5(ep["crop_dir"].encode("utf-8")).hexdigest()
    return os.path.join(feature_cache_dir, f"{crop_hash}.joblib")
