"""Multiprocessing worker helpers for WebDataset export."""

import os
import tarfile
from multiprocessing import current_process

import torch

from .webdataset_features import build_mano_models, load_episode_features
from .webdataset_writer import add_sample_to_tar, iter_episode_samples

_worker_mano_right = None
_worker_mano_left = None
_worker_device = None
_worker_rescan_frame_index = False
_worker_feature_cache_dir = None
_worker_episode_cache = {}


def normalize_mano_devices(mano_device, mano_gpus):
    """Normalize MANO worker device list."""
    if mano_gpus:
        devices = []
        for gpu in mano_gpus.split(","):
            gpu = gpu.strip()
            if not gpu:
                continue
            if gpu.startswith("cuda:"):
                devices.append(gpu)
            else:
                devices.append(f"cuda:{gpu}")
        if devices:
            return devices
    return [mano_device]


def _worker_init(device_specs, mano_dir, rescan_frame_index, feature_cache_dir):
    global _worker_mano_right, _worker_mano_left, _worker_device
    global _worker_rescan_frame_index, _worker_feature_cache_dir, _worker_episode_cache

    identity = current_process()._identity
    worker_idx = identity[0] - 1 if identity else 0
    device_str = device_specs[worker_idx % len(device_specs)]
    _worker_device = torch.device(device_str)
    _worker_mano_right, _worker_mano_left = build_mano_models(_worker_device, mano_dir=mano_dir)
    _worker_mano_right.eval()
    _worker_mano_left.eval()
    _worker_rescan_frame_index = rescan_frame_index
    _worker_feature_cache_dir = feature_cache_dir
    _worker_episode_cache = {}


def _worker_process_shard(task):
    """Build one shard in a worker process and write directly to disk."""
    frames_written = 0
    skipped_episodes = 0
    touched_episodes = set()
    tar_writer = None
    output_path = task["output_path"]
    tmp_path = task["tmp_path"]

    try:
        for episode_slice in task["episode_slices"]:
            cache_key = episode_slice["crop_dir"]
            if cache_key not in _worker_episode_cache:
                _worker_episode_cache[cache_key] = load_episode_features(
                    episode_slice,
                    _worker_mano_right,
                    _worker_mano_left,
                    _worker_device,
                    rescan_frame_index=_worker_rescan_frame_index,
                    feature_cache_dir=_worker_feature_cache_dir,
                )

            episode_data = _worker_episode_cache[cache_key]
            if episode_data is None:
                skipped_episodes += 1
                continue

            sample_iter = iter_episode_samples(
                episode_slice,
                episode_data,
                episode_slice["frame_start"],
                episode_slice["frame_end"],
            )
            for key, frame_path, lowdim, meta in sample_iter:
                if tar_writer is None:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    tar_writer = tarfile.open(tmp_path, "w")
                add_sample_to_tar(tar_writer, key, frame_path, lowdim, meta)
                frames_written += 1

            touched_episodes.add(cache_key)
    except Exception:
        if tar_writer is not None:
            tar_writer.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if tar_writer is not None:
        tar_writer.close()

    if frames_written == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    else:
        os.replace(tmp_path, output_path)

    return {
        "shard_idx": task["shard_idx"],
        "frames_written": frames_written,
        "episodes_written": len(touched_episodes),
        "skipped_episodes": skipped_episodes,
        "output_path": output_path,
    }
