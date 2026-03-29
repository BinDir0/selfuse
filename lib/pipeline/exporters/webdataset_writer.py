"""Shard planning and tar writing helpers for WebDataset export."""

import io
import json
import os
import tarfile

import numpy as np


def plan_shards(episodes, frames_per_shard, output_dir):
    """Pack whole episodes into shards near the target frame count."""
    tasks = []
    shard_slices = []
    shard_frame_count = 0
    shard_idx = 0

    def flush_current():
        nonlocal shard_slices, shard_frame_count, shard_idx
        if not shard_slices:
            return
        output_path = os.path.join(output_dir, f"shard-{shard_idx:06d}.tar")
        tmp_path = output_path + ".tmp"
        tasks.append(
            {
                "shard_idx": shard_idx,
                "output_path": output_path,
                "tmp_path": tmp_path,
                "frame_count": shard_frame_count,
                "episode_slices": shard_slices,
            }
        )
        shard_idx += 1
        shard_slices = []
        shard_frame_count = 0

    for ep in episodes:
        num_frames = ep["num_valid_frames"]
        if shard_slices and shard_frame_count + num_frames > frames_per_shard:
            flush_current()

        shard_slices.append(
            {
                "crop_dir": ep["crop_dir"],
                "episode_id": ep["episode_id"],
                "episode_index": ep["episode_index"],
                "instruction": list(ep.get("instruction", [])),
                "frame_start": 0,
                "frame_end": num_frames,
            }
        )
        shard_frame_count += num_frames

        if shard_frame_count >= frames_per_shard:
            flush_current()

    flush_current()
    return tasks


def iter_episode_samples(ep, episode_data, frame_start, frame_end):
    """Yield frame samples for one planned episode slice."""
    frame_ids = episode_data["frame_ids"][frame_start:frame_end]
    for frame_idx in frame_ids:
        frame_path = episode_data["frame_index"].get(frame_idx)
        if frame_path is None:
            continue
        meta = {
            "dataset_name": "buildai",
            "episode_index": ep["episode_index"],
            "instruction": list(ep.get("instruction", [])),
            "instruction_num": len(ep.get("instruction", [])),
            "presence": int(episode_data["presence_per_frame"][frame_idx]),
        }
        sample_key = f"buildai_ep{ep['episode_index']:06d}_f{frame_idx:05d}"
        yield sample_key, frame_path, episode_data["lowdim_all"][frame_idx], meta


def add_sample_to_tar(tar_writer, key, frame_path, lowdim, meta):
    """Write one WebDataset sample to a tar file."""
    with open(frame_path, "rb") as image_file:
        img_info = tarfile.TarInfo(name=f"{key}.image.jpg")
        img_info.size = os.fstat(image_file.fileno()).st_size
        tar_writer.addfile(img_info, image_file)

    lowdim_buf = io.BytesIO()
    np.save(lowdim_buf, lowdim, allow_pickle=False)
    lowdim_bytes = lowdim_buf.getvalue()
    lowdim_info = tarfile.TarInfo(name=f"{key}.lowdim.npy")
    lowdim_info.size = len(lowdim_bytes)
    tar_writer.addfile(lowdim_info, io.BytesIO(lowdim_bytes))

    meta_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    meta_info = tarfile.TarInfo(name=f"{key}.meta.json")
    meta_info.size = len(meta_bytes)
    tar_writer.addfile(meta_info, io.BytesIO(meta_bytes))
