"""Helpers for building frame sources and reading raw frame bytes from descriptors."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

from lib.pipeline.datasets.descriptors import (
    ClipDescriptor,
    STORAGE_IMAGE_SEQUENCE,
    STORAGE_TAR_SHARD,
)


def build_frame_source_from_descriptor(descriptor: ClipDescriptor):
    from lib.pipeline.frame_source import ImageFolderFrameSource, ShardVideoFrameSource

    if descriptor.storage_kind == STORAGE_TAR_SHARD:
        if not descriptor.shard_path:
            raise ValueError(f"Descriptor {descriptor.clip_id} missing shard_path")
        return ShardVideoFrameSource(
            descriptor.shard_path,
            descriptor.frame_names,
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
                    f"Short read from shard {descriptor.shard_path} frame {descriptor.frame_names[frame_idx]}"
                )
            return payload

        tar_reader = None if shard_tar_cache is None else shard_tar_cache.get(descriptor.shard_path)
        if tar_reader is None:
            tar_reader = tarfile.open(descriptor.shard_path, "r")
            if shard_tar_cache is not None:
                shard_tar_cache[descriptor.shard_path] = tar_reader
        member_name = descriptor.frame_names[frame_idx]
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
