"""Tests for shard-based normalizer sampling and streaming correctness."""

import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from src.dataset.collator import ConcatDataCollator
from src.dataset.normalizer_utils import get_normalizer
from src.dataset.vla_dataset import VLALowLevelWdsDataset


def shape_meta_mano():
    return {
        "obs": {
            "rgb": {"shape": [4, 4, 3], "type": "rgb", "horizon": 1, "stride": 1},
            "depth": {"shape": [4, 4], "type": "depth", "horizon": 1, "stride": 1},
            "state": {
                "wrist": {"shape": [18]},
                "hand": {"shape": [30]},
                "shape": [48],
                "type": "mano",
                "horizon": 1,
                "stride": 1,
            },
        },
        "action": {"shape": [48], "type": "mano", "horizon": 1, "stride": 1},
    }


class RandomMotionDataset(torch.utils.data.IterableDataset):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def __iter__(self):
        for sample in self.samples:
            yield sample

    def get_collator(self):
        return ConcatDataCollator()


def add_bytes_to_tar(tar_obj, name, data):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar_obj.addfile(info, io.BytesIO(data))


def encode_npy(array):
    buffer = io.BytesIO()
    np.save(buffer, array.astype(np.float32))
    return buffer.getvalue()


def make_lowdim_vector(rng):
    wrist_state = rng.normal(size=18).astype(np.float32)
    hand_state = rng.normal(size=30).astype(np.float32)
    wrist_action = rng.normal(size=18).astype(np.float32)
    hand_action = rng.normal(size=30).astype(np.float32)
    extrinsic = np.eye(4, dtype=np.float32).reshape(-1)
    intrinsic = rng.normal(size=4).astype(np.float32)
    return np.concatenate(
        [wrist_state, hand_state, wrist_action, hand_action, extrinsic, intrinsic],
        axis=0,
    ).astype(np.float32)


def write_random_shard(
    shard_path: Path,
    dataset_name: str,
    episode_index: int,
    num_frames: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(shard_path, "w") as tar_obj:
        for frame_index in range(num_frames):
            sample_key = f"ep{episode_index:04d}_{frame_index:06d}"
            meta = {
                "dataset_name": dataset_name,
                "episode_index": episode_index,
                "instruction": f"do task {dataset_name}",
                "instruction_num": 0,
                "presence": 3,
            }
            add_bytes_to_tar(
                tar_obj,
                f"{sample_key}.meta.json",
                json.dumps(meta).encode("utf-8"),
            )
            add_bytes_to_tar(
                tar_obj,
                f"{sample_key}.lowdim.npy",
                encode_npy(make_lowdim_vector(rng)),
            )


def build_random_wds_root(tmp_path: Path, shard_counts=(3, 3), num_frames: int = 5):
    dataset_specs = []
    for dataset_idx, (dataset_name, shard_count) in enumerate(
        zip(["dataset_a", "dataset_b"], shard_counts)
    ):
        dataset_dir = tmp_path / dataset_name
        for shard_idx in range(shard_count):
            write_random_shard(
                dataset_dir / f"shard-{shard_idx:06d}.tar",
                dataset_name=dataset_name,
                episode_index=dataset_idx * 100 + shard_idx,
                num_frames=num_frames,
                seed=dataset_idx * 1000 + shard_idx,
            )
        dataset_specs.append(
            {
                "name": dataset_name,
                "shard_urls": str(dataset_dir / "shard-*.tar"),
                "weight": 1.0,
            }
        )
    return dataset_specs


def count_selected_shards_by_dataset(shard_urls):
    return Counter(Path(url).parent.name for url in shard_urls)


def assert_stats_match(normalizer, key, rows):
    stats = normalizer.params_dict[key]["input_stats"]
    rows = rows.to(torch.float64)
    expected = {
        "min": rows.min(dim=0).values,
        "max": rows.max(dim=0).values,
        "mean": rows.mean(dim=0),
        "std": rows.std(dim=0, unbiased=True),
        "q01": torch.quantile(rows, 0.01, dim=0),
        "q99": torch.quantile(rows, 0.99, dim=0),
    }
    for stat_name, expected_value in expected.items():
        actual = stats[stat_name].detach().to(torch.float64)
        torch.testing.assert_close(
            actual,
            expected_value,
            rtol=1e-5,
            atol=1e-5,
        )


def test_get_normalizer_matches_direct_stats_with_concat_collator():
    rng = np.random.default_rng(0)
    samples = []
    expected_rows = []
    for num_rows in [1, 3, 2, 4, 5]:
        motion = torch.tensor(rng.normal(size=(num_rows, 24)).astype(np.float32))
        samples.append({"motions": motion})
        expected_rows.append(motion)

    dataset = RandomMotionDataset(samples)
    normalizer = get_normalizer({"batch_size": 2, "num_workers": 0}, dataset)

    assert set(normalizer.params_dict.keys()) == {"motions"}
    assert_stats_match(normalizer, "motions", torch.cat(expected_rows, dim=0))


def test_get_normalizer_return_metadata_tracks_frames_and_rows():
    rng = np.random.default_rng(1)
    samples = []
    row_counts = [2, 1, 4, 3]
    for num_rows in row_counts:
        motion = torch.tensor(rng.normal(size=(num_rows, 24)).astype(np.float32))
        samples.append({"motions": motion})

    dataset = RandomMotionDataset(samples)
    _, metadata = get_normalizer(
        {"batch_size": 2, "num_workers": 0},
        dataset,
        return_metadata=True,
    )

    assert metadata["normalizer_keys"] == ["motions"]
    assert metadata["current_frames_scanned"] == len(samples)
    assert metadata["effective_rows"]["motions"] == sum(row_counts)


def test_lowlevel_dataset_respects_total_shard_budget(tmp_path):
    wds_datasets = build_random_wds_root(tmp_path, shard_counts=(3, 3))
    dataset = VLALowLevelWdsDataset(
        wds_datasets=wds_datasets,
        shape_meta=shape_meta_mano(),
        mode="val",
        max_total_shards=4,
        min_shards_per_dataset=1,
        seed=7,
    )

    shard_urls = dataset.build_shard_urls()
    counts = count_selected_shards_by_dataset(shard_urls)

    assert len(shard_urls) == 4
    assert counts["dataset_a"] == 2
    assert counts["dataset_b"] == 2


def test_lowlevel_dataset_preserves_minimum_shards_when_budget_allows(tmp_path):
    wds_datasets = build_random_wds_root(tmp_path, shard_counts=(5, 2))
    dataset = VLALowLevelWdsDataset(
        wds_datasets=wds_datasets,
        shape_meta=shape_meta_mano(),
        mode="val",
        max_total_shards=5,
        min_shards_per_dataset=2,
        seed=3,
    )

    counts = count_selected_shards_by_dataset(dataset.build_shard_urls())

    assert counts["dataset_a"] >= 2
    assert counts["dataset_b"] >= 2
    assert counts["dataset_a"] + counts["dataset_b"] == 5


def test_lowlevel_dataset_gives_larger_dataset_more_shards(tmp_path):
    wds_datasets = build_random_wds_root(tmp_path, shard_counts=(5, 2))
    dataset = VLALowLevelWdsDataset(
        wds_datasets=wds_datasets,
        shape_meta=shape_meta_mano(),
        mode="val",
        max_total_shards=4,
        min_shards_per_dataset=1,
        seed=7,
    )

    counts = count_selected_shards_by_dataset(dataset.build_shard_urls())

    assert counts["dataset_a"] == 3
    assert counts["dataset_b"] == 1


def test_lowlevel_dataset_describe_shard_selection_reports_coverage(tmp_path):
    wds_datasets = build_random_wds_root(tmp_path, shard_counts=(5, 2))
    dataset = VLALowLevelWdsDataset(
        wds_datasets=wds_datasets,
        shape_meta=shape_meta_mano(),
        mode="val",
        max_total_shards=4,
        min_shards_per_dataset=1,
        seed=7,
    )

    summary = dataset.describe_shard_selection()
    per_dataset = {item["name"]: item for item in summary["datasets"]}

    assert summary["available_shards_total"] == 7
    assert summary["selected_shards_total"] == 4
    assert not summary["full_dataset_coverage"]
    assert per_dataset["dataset_a"]["selected_shards"] == 3
    assert per_dataset["dataset_b"]["selected_shards"] == 1
    assert not per_dataset["dataset_a"]["full_coverage"]
    assert not per_dataset["dataset_b"]["full_coverage"]


def test_lowlevel_dataset_raises_when_budget_is_smaller_than_floor(tmp_path):
    wds_datasets = build_random_wds_root(tmp_path, shard_counts=(3, 3))
    dataset = VLALowLevelWdsDataset(
        wds_datasets=wds_datasets,
        shape_meta=shape_meta_mano(),
        mode="val",
        max_total_shards=3,
        min_shards_per_dataset=2,
        seed=1,
    )

    with pytest.raises(
        ValueError,
        match="max_total_shards is smaller than the required minimum shard coverage",
    ):
        dataset.build_shard_urls()


def test_lowlevel_dataset_build_shard_urls_raises_value_error_when_empty(tmp_path):
    dataset = VLALowLevelWdsDataset(
        wds_datasets=[
            {
                "name": "empty",
                "shard_urls": str(tmp_path / "missing" / "shard-*.tar"),
                "weight": 1.0,
            }
        ],
        shape_meta=shape_meta_mano(),
        mode="val",
    )

    with pytest.raises(ValueError, match="No shards found across all datasets"):
        dataset.build_shard_urls()


def test_lowlevel_wds_metadata_tracks_frames_per_dataset(tmp_path):
    wds_datasets = build_random_wds_root(tmp_path, shard_counts=(1, 1), num_frames=5)
    dataset = VLALowLevelWdsDataset(
        wds_datasets=wds_datasets,
        shape_meta=shape_meta_mano(),
        mode="val",
        max_total_shards=2,
        min_shards_per_dataset=1,
        seed=0,
    )

    _, metadata = get_normalizer(
        {"batch_size": 2, "num_workers": 0},
        dataset,
        return_metadata=True,
    )

    assert metadata["current_frames_scanned"] == 10
    assert metadata["current_frames_scanned_by_dataset"] == {
        "dataset_a": 5,
        "dataset_b": 5,
    }


def test_lowlevel_wds_normalizer_matches_direct_rows_on_random_wds(tmp_path):
    wds_datasets = build_random_wds_root(tmp_path, shard_counts=(3, 3))
    dataset = VLALowLevelWdsDataset(
        wds_datasets=wds_datasets,
        shape_meta=shape_meta_mano(),
        mode="val",
        max_total_shards=4,
        min_shards_per_dataset=2,
        seed=11,
    )

    direct_rows = torch.cat([sample["motions"] for sample in dataset], dim=0)
    normalizer = get_normalizer({"batch_size": 3, "num_workers": 0}, dataset)

    assert_stats_match(normalizer, "motions", direct_rows)
