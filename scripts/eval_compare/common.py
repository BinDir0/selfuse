"""Small shared utilities for the eval_compare scripts (config, paths, video I/O)."""

from __future__ import annotations

import os
import random

import numpy as np
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def sanitize(seq_id: str) -> str:
    """Filesystem-safe key for a sequence id (may contain '/', spaces, parens)."""
    return seq_id.replace("/", "__").replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")


def seq_workdir(work_dir: str, dataset: str, seq_id: str) -> str:
    return os.path.join(work_dir, dataset, sanitize(seq_id))


def sample_sequences(seqs: list[str], n: int, seed: int) -> list[str]:
    if len(seqs) <= n:
        return seqs
    rng = random.Random(seed)
    return sorted(rng.sample(seqs, n))


def write_video_from_frames(frame_paths: list[str], out_path: str, fps: float) -> None:
    """Encode an mp4 from ordered image paths (lossless-ish, yuv420p for player compat)."""
    import cv2

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    first = cv2.imread(frame_paths[0])
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in frame_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        vw.write(img)
    vw.release()


def write_video_from_zarr(zarr_path: str, out_path: str, fps: float, key: str = "images.front_1") -> int:
    """Decode jpeg-encoded ego frames from a zarr episode into an mp4. Returns frame count."""
    import cv2
    import zarr

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    z = zarr.open(zarr_path, mode="r")
    arr = z[key]
    n = arr.shape[0]
    vw = None
    for i in range(n):
        raw = arr[i]
        if isinstance(raw, (bytes, bytearray)):
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        else:
            img = np.asarray(raw)
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if vw is None:
            h, w = img.shape[:2]
            vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        vw.write(img)
    if vw is not None:
        vw.release()
    return n
