"""Unit tests for VLMWdsDataset."""

import sys
import traceback
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.dataset.vlm_dataset import VLMWdsDataset


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

    data = ds.sample_to_data(make_sample())

    assert data["images"].shape[0] == 2
    assert data["question"] == "q"
    assert data["answer"] == "a"
    assert data["vision_type"] == "image"
    assert bool(data["is_vla_data"]) is False
    assert data["dataset_name"] == "demo_src"
    assert data["episode_index"].item() == 7


def test_build_pipeline_no_sliding_window():
    ds = VLMWdsDataset(
        wds_datasets=[{"name": "demo", "shard_urls": "/tmp/fake/shard-*.tar", "weight": 1.0}],
        mode="train",
    )
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
    sentinel_collator = object()
    ds.set_collator(sentinel_collator)

    val_ds = ds.get_validation_dataset()
    assert val_ds.mode == "val"
    assert val_ds.wds_datasets[0]["name"] == "val"
    assert val_ds.collator is sentinel_collator


def test_sample_to_data_returns_raw_fields_when_collator_is_set():
    ds = VLMWdsDataset(
        wds_datasets=[{"shard_urls": "/tmp/fake/shard-*.tar"}],
        mode="val",
        return_dataset_info=True,
    )
    ds.set_collator(object())

    data = ds.sample_to_data(make_sample())

    assert "input_ids" not in data
    assert data["images"].shape == (2, 8, 8, 3)
    assert data["question"] == "q"
    assert data["answer"] == "a"
    assert data["vision_type"] == "image"
    assert bool(data["is_vla_data"]) is False


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
