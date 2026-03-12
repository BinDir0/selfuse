"""Unit tests for VLMWdsDataset."""

import sys
import traceback
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.dataset.vlm_dataset import VLMWdsDataset


class DummyTokenizer:
    pad_token_id = 0


class DummyPreprocessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()
        self.ignore_index = -100

    def __call__(self, images, text, target, mode):
        n_img = images.shape[0]
        return {
            "input_ids": np.array([1, 2, 3], dtype=np.int64),
            "labels": np.array([4, 5, 6], dtype=np.int64),
            "attention_mask": np.array([1, 1, 1], dtype=np.int64),
            "pixel_values": np.zeros((n_img, 3, 4, 4), dtype=np.float32),
            "answer_start_idx": np.array(1, dtype=np.int64),
        }


def make_sample():
    meta = {
        "dataset_name": "demo",
        "source": "demo_src",
        "sample_idx": 7,
        "texts": [{"user": "q", "assistant": "a"}],
        "formatting_ratings": [1],
        "visual_dependency_ratings": [1],
        "relevance_ratings": [1],
    }
    img0 = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB")
    img1 = Image.fromarray(np.ones((8, 8, 3), dtype=np.uint8), mode="RGB")
    return {
        "meta.json": meta,
        "image_000.jpg": img0,
        "image_001.jpg": img1,
    }


def test_sample_to_data():
    ds = VLMWdsDataset(
        wds_datasets=[{"shard_urls": "/tmp/fake/shard-*.tar"}],
        mode="val",
        return_dataset_info=True,
    )
    ds.set_preprocessor(DummyPreprocessor())

    data = ds.sample_to_data(make_sample())

    assert data["input_ids"].shape[0] == 3
    assert data["labels"].shape[0] == 3
    assert data["pixel_values"].shape[0] == 2
    assert bool(data["is_vla_data"]) is False
    assert data["dataset_name"] == "demo_src"
    assert data["episode_index"].item() == 7


def test_build_pipeline_no_sliding_window():
    ds = VLMWdsDataset(
        wds_datasets=[{"name": "demo", "shard_urls": "/tmp/fake/shard-*.tar", "weight": 1.0}],
        mode="train",
    )
    ds.set_preprocessor(DummyPreprocessor())

    with patch("src.dataset.vlm_dataset.build_blended_dataset") as mock_build:
        mock_build.return_value = [{"__key__": "k0", "x": 1}]
        out = list(ds.build_pipeline())
        _, kwargs = mock_build.call_args
        assert kwargs["use_sliding_window"] is False
        assert "__key__" not in out[0]
        assert out[0]["x"] == 1


def test_get_validation_dataset_uses_val_wds():
    ds = VLMWdsDataset(
        wds_datasets=[{"name": "train", "shard_urls": "/tmp/train/shard-*.tar"}],
        val_wds_datasets=[{"name": "val", "shard_urls": "/tmp/val/shard-*.tar"}],
        mode="train",
    )
    ds.set_preprocessor(DummyPreprocessor())

    val_ds = ds.get_validation_dataset()
    assert val_ds.mode == "val"
    assert val_ds.wds_datasets[0]["name"] == "val"
    assert val_ds.preprocessor is ds.preprocessor


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
