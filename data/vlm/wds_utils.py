"""
Common WebDataset writing utilities for VLM data transform scripts.

Provides a unified interface so each transform script can write WebDataset
shards directly, skipping the intermediate HF Arrow step.
"""

import io
import json
import os
import random
import time

import webdataset as wds
from PIL import Image


def encode_image_jpeg(img, quality=95):
    """Encode a PIL image to JPEG bytes."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def write_wds_sample(
    writer,
    key,
    images,
    texts,
    source,
    sample_idx,
    split="train",
    extra_images=None,
    extra_meta=None,
    image_quality=95,
):
    """
    Write one sample to a WebDataset TarWriter.

    Args:
        writer: wds.TarWriter
        key: unique string key for this sample
        images: list of PIL Images (stored as image_000.jpg, image_001.jpg, ...)
        texts: list of {"user": str, "assistant": str} dicts
        source: dataset source name
        sample_idx: integer index within source
        split: "train" or "test"
        extra_images: dict of {filename: PIL Image} for non-standard images (e.g. depth)
        extra_meta: dict of extra fields to merge into meta.json
        image_quality: JPEG quality (default 95)
    """
    image_bytes = {}
    total_bytes = 0
    for i, img in enumerate(images):
        data = encode_image_jpeg(img, quality=image_quality)
        image_bytes[f"image_{i:03d}.jpg"] = data
        total_bytes += len(data)

    if extra_images:
        for fname, img in extra_images.items():
            data = encode_image_jpeg(img, quality=image_quality)
            image_bytes[fname] = data
            total_bytes += len(data)

    n_texts = len(texts) if isinstance(texts, list) else 1
    meta = {
        "dataset_name": source,
        "source": source,
        "split": split,
        "sample_idx": int(sample_idx),
        "n_images": len(images),
        "texts": texts if isinstance(texts, list) else [texts],
        "formatting_ratings": [0] * n_texts,
        "visual_dependency_ratings": [0] * n_texts,
        "relevance_ratings": [0] * n_texts,
    }
    if extra_meta:
        meta.update(extra_meta)

    meta_json = json.dumps(meta, ensure_ascii=False).encode("utf-8")

    wds_sample = {"__key__": key, "meta.json": meta}
    wds_sample.update(image_bytes)
    writer.write(wds_sample)

    return total_bytes + len(meta_json)


def split_train_test(data, val_ratio=0.001, seed=42):
    """Randomly split a list into (train, test) sets."""
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    n_test = max(1, int(len(data) * val_ratio))
    test_indices = set(indices[:n_test])
    train = [data[i] for i in range(len(data)) if i not in test_indices]
    test = [data[i] for i in range(len(data)) if i in test_indices]
    return train, test


class ShardWriter:
    """
    Convenience wrapper around wds.TarWriter that auto-rotates shards.

    Usage:
        sw = ShardWriter(output_dir, split="train", worker_id=0, prefix="shard")
        sw.write(key, images, texts, source, sample_idx)
        sw.close()
    """

    def __init__(
        self,
        output_dir,
        split="train",
        worker_id=0,
        prefix="shard",
        maxcount=20000,
        maxsize=int(1e9),
        image_quality=95,
    ):
        self.output_dir = os.path.join(output_dir, split)
        os.makedirs(self.output_dir, exist_ok=True)

        self.pattern = os.path.join(
            self.output_dir, f"{prefix}-w{worker_id:04d}-%06d.tar"
        )
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.image_quality = image_quality

        self.shard_idx = 0
        self.shard_count = 0
        self.shard_size = 0
        self.total_written = 0
        self.writer = None

        self._t0 = time.time()

    def _ensure_writer(self):
        if self.writer is None:
            self.writer = wds.TarWriter(self.pattern % self.shard_idx)

    def _maybe_rotate(self):
        if self.writer is not None and (
            self.shard_count >= self.maxcount or self.shard_size >= self.maxsize
        ):
            self.writer.close()
            self.shard_idx += 1
            self.shard_count = 0
            self.shard_size = 0
            self.writer = None

    def write(self, key, images, texts, source, sample_idx, **kwargs):
        """Write one sample, auto-rotating shards as needed."""
        self._maybe_rotate()
        self._ensure_writer()
        nbytes = write_wds_sample(
            self.writer,
            key=key,
            images=images,
            texts=texts,
            source=source,
            sample_idx=sample_idx,
            image_quality=self.image_quality,
            **kwargs,
        )
        self.shard_count += 1
        self.shard_size += nbytes
        self.total_written += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        elapsed = time.time() - self._t0
        sps = self.total_written / elapsed if elapsed > 0 else 0
        print(
            f"  ShardWriter done: {self.total_written} samples, "
            f"{self.shard_idx + 1} shards, {elapsed:.1f}s ({sps:.1f} samples/s)"
        )
