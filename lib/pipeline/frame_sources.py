"""Helpers for building frame sources and reading raw frame bytes from descriptors."""

from __future__ import annotations

import os
import tarfile
import threading
from pathlib import Path

from lib.pipeline.datasets.descriptors import (
    ClipDescriptor,
    STORAGE_IMAGE_SEQUENCE,
    STORAGE_TAR_SHARD,
)


_IMAGE_MEMBER_SUFFIXES = (".jpg", ".jpeg", ".png")
_TAR_CLIP_MEMBER_CACHE = {}
_TAR_CLIP_MEMBER_CACHE_LOCK = threading.Lock()


def _frame_member_sort_key(member_name: str):
    stem = Path(member_name).stem
    frame_suffix = stem.rsplit("_", 1)[-1]
    if frame_suffix.startswith("f") and frame_suffix[1:].isdigit():
        return int(frame_suffix[1:])
    return frame_suffix


def _build_tar_clip_member_cache(tar: tarfile.TarFile):
    clip_members = {}
    for member in tar.getmembers():
        if not member.isfile():
            continue
        lower_name = member.name.lower()
        if not lower_name.endswith(_IMAGE_MEMBER_SUFFIXES):
            continue
        stem = Path(member.name).stem
        clip_id, sep, frame_suffix = stem.rpartition("_")
        if not sep or not frame_suffix.startswith("f") or not frame_suffix[1:].isdigit():
            continue
        clip_members.setdefault(clip_id, []).append(member.name)

    for names in clip_members.values():
        names.sort(key=_frame_member_sort_key)
    return clip_members


def _get_tar_clip_members(tar_path: str, tar: tarfile.TarFile, clip_id: str) -> list[str]:
    with _TAR_CLIP_MEMBER_CACHE_LOCK:
        clip_map = _TAR_CLIP_MEMBER_CACHE.get(tar_path)
        if clip_map is None:
            clip_map = _build_tar_clip_member_cache(tar)
            _TAR_CLIP_MEMBER_CACHE[tar_path] = clip_map
    return list(clip_map.get(clip_id, ()))


def _ensure_descriptor_tar_members(descriptor: ClipDescriptor, *, shard_tar_cache: dict | None = None) -> list[str]:
    if descriptor.frame_names:
        return descriptor.frame_names
    if descriptor.storage_kind != STORAGE_TAR_SHARD or descriptor.shard_path is None:
        raise ValueError(f"Descriptor {descriptor.clip_id} does not support lazy tar resolution")

    with _TAR_CLIP_MEMBER_CACHE_LOCK:
        clip_map = _TAR_CLIP_MEMBER_CACHE.get(descriptor.shard_path)
    if clip_map is not None:
        frame_names = list(clip_map.get(descriptor.clip_id, ()))
        if not frame_names:
            raise RuntimeError(f"No image members found for clip {descriptor.clip_id} in shard {descriptor.shard_path}")
        descriptor.frame_names = frame_names
        descriptor.frame_count_override = len(frame_names)
        return descriptor.frame_names

    tar_reader = None if shard_tar_cache is None else shard_tar_cache.get(descriptor.shard_path)
    opened_locally = False
    if tar_reader is None:
        tar_reader = tarfile.open(descriptor.shard_path, "r")
        opened_locally = True
        if shard_tar_cache is not None:
            shard_tar_cache[descriptor.shard_path] = tar_reader

    try:
        frame_names = _get_tar_clip_members(descriptor.shard_path, tar_reader, descriptor.clip_id)
    finally:
        if opened_locally and shard_tar_cache is None:
            tar_reader.close()
    if not frame_names:
        raise RuntimeError(f"No image members found for clip {descriptor.clip_id} in shard {descriptor.shard_path}")

    descriptor.frame_names = frame_names
    descriptor.frame_count_override = len(frame_names)
    return descriptor.frame_names


def build_frame_source_from_descriptor(descriptor: ClipDescriptor):
    from lib.pipeline.frame_source import ImageFolderFrameSource, ShardVideoFrameSource

    if descriptor.storage_kind == STORAGE_TAR_SHARD:
        if not descriptor.shard_path:
            raise ValueError(f"Descriptor {descriptor.clip_id} missing shard_path")
        frame_names = descriptor.frame_names or _ensure_descriptor_tar_members(descriptor)
        return ShardVideoFrameSource(
            descriptor.shard_path,
            frame_names,
            frame_offsets=descriptor.frame_offsets,
        )

    if descriptor.storage_kind == STORAGE_IMAGE_SEQUENCE:
        if not descriptor.frame_dir:
            raise ValueError(f"Descriptor {descriptor.clip_id} missing frame_dir")
        image_paths = [str((Path(descriptor.frame_dir) / frame_name).resolve()) for frame_name in descriptor.frame_names]
        return ImageFolderFrameSource(image_paths)

    raise ValueError(f"Unsupported descriptor storage_kind: {descriptor.storage_kind}")


def read_frame_bytes_from_descriptor(
    descriptor: ClipDescriptor,
    frame_idx: int,
    *,
    shard_fd_cache: dict | None = None,
    shard_tar_cache: dict | None = None,
) -> bytes:
    if descriptor.storage_kind == STORAGE_TAR_SHARD:
        if descriptor.shard_path is None:
            raise ValueError(f"Descriptor {descriptor.clip_id} missing shard_path")
        frame_names = descriptor.frame_names or _ensure_descriptor_tar_members(
            descriptor,
            shard_tar_cache=shard_tar_cache,
        )
        if descriptor.frame_offsets is not None:
            offset, size = descriptor.frame_offsets[frame_idx]
            fd = None if shard_fd_cache is None else shard_fd_cache.get(descriptor.shard_path)
            if fd is None:
                fd = os.open(descriptor.shard_path, os.O_RDONLY)
                if shard_fd_cache is not None:
                    shard_fd_cache[descriptor.shard_path] = fd
            payload = os.pread(fd, size, offset)
            if len(payload) != size:
                raise RuntimeError(
                    f"Short read from shard {descriptor.shard_path} frame {frame_names[frame_idx]}"
                )
            return payload

        tar_reader = None if shard_tar_cache is None else shard_tar_cache.get(descriptor.shard_path)
        if tar_reader is None:
            tar_reader = tarfile.open(descriptor.shard_path, "r")
            if shard_tar_cache is not None:
                shard_tar_cache[descriptor.shard_path] = tar_reader
        member_name = frame_names[frame_idx]
        member = tar_reader.getmember(member_name)
        extracted = tar_reader.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Failed to extract {member_name} from {descriptor.shard_path}")
        return extracted.read()

    if descriptor.storage_kind == STORAGE_IMAGE_SEQUENCE:
        if descriptor.frame_dir is None:
            raise ValueError(f"Descriptor {descriptor.clip_id} missing frame_dir")
        frame_path = Path(descriptor.frame_dir) / descriptor.frame_names[frame_idx]
        return frame_path.read_bytes()

    raise ValueError(f"Unsupported descriptor storage_kind: {descriptor.storage_kind}")
