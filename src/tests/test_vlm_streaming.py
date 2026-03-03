"""Unit tests for VLMStreamingDataset stream building and distribution."""

import sys
import traceback
import warnings
from unittest.mock import MagicMock, patch, call

from src.dataset.vlm_dataset import VLMStreamingDataset


# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

MOD = "src.dataset.legendvlm_dataset"


def make_mock_stream(name="ds"):
    """Create a MagicMock that behaves like an IterableDataset."""
    ds = MagicMock(name=name)
    ds.info = None
    ds.shuffle = MagicMock(return_value=ds)
    return ds


# ---------------------------------------------------------------------------
# Tests — build_stream (merged file loading)
# ---------------------------------------------------------------------------

@patch(f"{MOD}.load_dataset")
@patch.object(VLMStreamingDataset, "collect_data_files",
              return_value=["/data/a/train/data-00000.arrow", "/data/b/train/data-00000.arrow"])
def test_build_stream_train_shuffles(mock_collect, mock_load):
    """Train mode: load_dataset called once with all files, then shuffle."""
    stream = make_mock_stream("stream")
    mock_load.return_value = stream

    obj = VLMStreamingDataset(
        dataset_paths=["/data/a", "/data/b"],
        split="train", mode="train",
    )
    mock_load.assert_called_once()
    # Verify data_files contains both files
    _, kwargs = mock_load.call_args
    assert len(kwargs["data_files"]) == 2
    stream.shuffle.assert_called_once()
    assert obj.stream is not None


@patch(f"{MOD}.load_dataset")
@patch.object(VLMStreamingDataset, "collect_data_files",
              return_value=["/data/a/test/data-00000.arrow"])
def test_build_stream_val_no_shuffle(mock_collect, mock_load):
    """Val mode: load_dataset called, no shuffle."""
    stream = make_mock_stream("stream")
    mock_load.return_value = stream

    obj = VLMStreamingDataset(
        dataset_paths=["/data/a"],
        split="test", mode="val",
    )
    mock_load.assert_called_once()
    stream.shuffle.assert_not_called()


@patch.object(VLMStreamingDataset, "collect_data_files", return_value=[])
def test_build_stream_no_files_returns_none(mock_collect):
    """No data files found -> stream is None."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = VLMStreamingDataset(
            dataset_paths=["/nonexistent"],
            split="train", mode="train",
        )
    assert obj.stream is None


@patch(f"{MOD}.load_dataset")
@patch.object(VLMStreamingDataset, "collect_data_files",
              return_value=["/data/a/train/part-00000.parquet"])
def test_build_stream_detects_parquet(mock_collect, mock_load):
    """Parquet files detected and loaded with format='parquet'."""
    stream = make_mock_stream("stream")
    mock_load.return_value = stream

    obj = VLMStreamingDataset(
        dataset_paths=["/data/a"],
        split="train", mode="train",
    )
    args, _ = mock_load.call_args
    assert args[0] == "parquet"


# ---------------------------------------------------------------------------
# Tests — collect_data_files
# ---------------------------------------------------------------------------

@patch(f"{MOD}.glob.glob")
def test_collect_finds_arrow_in_split_dir(mock_glob):
    """Finds arrow files under {path}/{split}/ pattern."""
    mock_glob.side_effect = lambda pattern: (
        ["/d/train/data-00000.arrow", "/d/train/data-00001.arrow"]
        if "train/data-*.arrow" in pattern else []
    )
    obj = VLMStreamingDataset.__new__(VLMStreamingDataset)
    obj.dataset_paths = ["/d"]
    obj.split = "train"
    files = obj.collect_data_files()
    assert len(files) == 2


@patch(f"{MOD}.glob.glob")
def test_collect_falls_back_to_parquet(mock_glob):
    """Falls back to parquet when no arrow files found."""
    def side_effect(pattern):
        if pattern.endswith(".parquet"):
            return ["/d/train/part-00000.parquet"]
        return []
    mock_glob.side_effect = side_effect
    obj = VLMStreamingDataset.__new__(VLMStreamingDataset)
    obj.dataset_paths = ["/d"]
    obj.split = "train"
    files = obj.collect_data_files()
    assert len(files) == 1
    assert files[0].endswith(".parquet")


# ---------------------------------------------------------------------------
# Tests — infer_proportional_probs (kept for backward compat)
# ---------------------------------------------------------------------------

def test_infer_probs_with_metadata():
    """When metadata has num_examples, return proportional sizes."""
    ds1 = MagicMock()
    split_info1 = MagicMock()
    split_info1.num_examples = 1000
    ds1.info.splits.values.return_value = [split_info1]

    ds2 = MagicMock()
    split_info2 = MagicMock()
    split_info2.num_examples = 3000
    ds2.info.splits.values.return_value = [split_info2]

    result = VLMStreamingDataset.infer_proportional_probs([ds1, ds2])
    assert result == [1000.0, 3000.0]


def test_infer_probs_no_metadata():
    """When info is None, return None."""
    ds = MagicMock()
    ds.info = None
    result = VLMStreamingDataset.infer_proportional_probs([ds])
    assert result is None


# ---------------------------------------------------------------------------
# Tests — distribute
# ---------------------------------------------------------------------------

@patch(f"{MOD}.load_dataset")
@patch.object(VLMStreamingDataset, "collect_data_files",
              return_value=["/d/train/data-00000.arrow"])
def test_distribute_calls_split_by_node(mock_collect, mock_load):
    """distribute() calls split_dataset_by_node with correct args."""
    stream = make_mock_stream("stream")
    mock_load.return_value = stream

    obj = VLMStreamingDataset(
        dataset_paths=["/d"], split="train", mode="train",
    )

    with patch("datasets.distributed.split_dataset_by_node") as mock_split:
        mock_split.return_value = MagicMock(name="split_stream")
        obj.distribute(rank=1, world_size=4)
        mock_split.assert_called_once()
        _, kwargs = mock_split.call_args
        assert kwargs["rank"] == 1
        assert kwargs["world_size"] == 4


@patch.object(VLMStreamingDataset, "collect_data_files", return_value=[])
def test_distribute_none_stream(mock_collect):
    """distribute with stream=None does not raise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = VLMStreamingDataset(
            dataset_paths=["/fake"], split="train", mode="train",
        )
    assert obj.stream is None
    obj.distribute(rank=0, world_size=2)  # should not raise


# ---------------------------------------------------------------------------
# Tests — return_dataset_info
# ---------------------------------------------------------------------------

@patch(f"{MOD}.load_dataset")
@patch.object(VLMStreamingDataset, "collect_data_files",
              return_value=["/d/train/data-00000.arrow"])
def test_return_dataset_info_maps_episode_index(mock_collect, mock_load):
    """return_dataset_info=True causes .map() on the merged stream."""
    stream = make_mock_stream("stream")
    mapped = make_mock_stream("mapped")
    stream.map = MagicMock(return_value=mapped)
    mock_load.return_value = stream

    obj = VLMStreamingDataset(
        dataset_paths=["/d"], split="train", mode="train",
        return_dataset_info=True,
    )
    stream.map.assert_called_once()
    _, kwargs = stream.map.call_args
    assert kwargs.get("with_indices") is True


@patch(f"{MOD}.load_dataset")
@patch.object(VLMStreamingDataset, "collect_data_files",
              return_value=["/d/train/data-00000.arrow"])
def test_return_dataset_info_false_no_map(mock_collect, mock_load):
    """return_dataset_info=False (default) does not call .map()."""
    stream = make_mock_stream("stream")
    stream.map = MagicMock()
    mock_load.return_value = stream

    obj = VLMStreamingDataset(
        dataset_paths=["/d"], split="train", mode="train",
        return_dataset_info=False,
    )
    stream.map.assert_not_called()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

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
