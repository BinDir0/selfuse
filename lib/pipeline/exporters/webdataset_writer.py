"""Shard planning and tar writing helpers for WebDataset export."""

import io
import json
import os
import tarfile

import numpy as np


LOWDIM_SIZE = 116
LOWDIM_DTYPE = np.dtype(np.float32)
_LOWDIM_SAMPLE = np.zeros((LOWDIM_SIZE,), dtype=LOWDIM_DTYPE)
_LOWDIM_BUF = io.BytesIO()
np.save(_LOWDIM_BUF, _LOWDIM_SAMPLE, allow_pickle=False)
_LOWDIM_NPY_HEADER = _LOWDIM_BUF.getvalue()[: -_LOWDIM_SAMPLE.nbytes]


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
    instruction = list(ep.get("instruction", []))
    meta_prefix = (
        json.dumps(
            {
                "dataset_name": "buildai",
                "episode_index": ep["episode_index"],
                "instruction": instruction,
                "instruction_num": len(instruction),
                "lowdim_schema": "hawor_wrist_world_v2",
                "wrist_translation_semantics": "mano_joint_0_world",
                "camera_extrinsic_convention": "w2c",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )[:-1]
        + ',"presence":'
    )
    frame_ids = episode_data["frame_ids"][frame_start:frame_end]
    for frame_idx in frame_ids:
        frame_path = episode_data["frame_index"].get(frame_idx)
        if frame_path is None:
            continue
        presence = int(episode_data["presence_per_frame"][frame_idx])
        meta_bytes = f"{meta_prefix}{presence}}}".encode("utf-8")
        sample_key = f"buildai_ep{ep['episode_index']:06d}_f{frame_idx:05d}"
        yield sample_key, frame_path, episode_data["lowdim_all"][frame_idx], meta_bytes


def _encode_lowdim_npy(lowdim):
    array = np.asarray(lowdim, dtype=LOWDIM_DTYPE)
    if array.shape == (LOWDIM_SIZE,):
        return _LOWDIM_NPY_HEADER + np.ascontiguousarray(array).tobytes()

    lowdim_buf = io.BytesIO()
    np.save(lowdim_buf, array, allow_pickle=False)
    return lowdim_buf.getvalue()


def prepare_sample_payload(key, frame_path, lowdim, meta_bytes):
    with open(frame_path, "rb") as image_file:
        image_bytes = image_file.read()
    lowdim_bytes = _encode_lowdim_npy(lowdim)
    return key, image_bytes, lowdim_bytes, meta_bytes


def add_sample_to_tar(tar_writer, key, frame_path, lowdim, meta_bytes):
    key, image_bytes, lowdim_bytes, meta_bytes = prepare_sample_payload(key, frame_path, lowdim, meta_bytes)
    add_prepared_sample_to_tar(tar_writer, key, image_bytes, lowdim_bytes, meta_bytes)


def add_prepared_sample_to_tar(tar_writer, key, image_bytes, lowdim_bytes, meta_bytes):
    img_info = tarfile.TarInfo(name=f"{key}.image.jpg")
    img_info.size = len(image_bytes)
    tar_writer.addfile(img_info, io.BytesIO(image_bytes))

    lowdim_info = tarfile.TarInfo(name=f"{key}.lowdim.npy")
    lowdim_info.size = len(lowdim_bytes)
    tar_writer.addfile(lowdim_info, io.BytesIO(lowdim_bytes))

    meta_info = tarfile.TarInfo(name=f"{key}.meta.json")
    meta_info.size = len(meta_bytes)
    tar_writer.addfile(meta_info, io.BytesIO(meta_bytes))
