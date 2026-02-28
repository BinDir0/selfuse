"""Unit tests for LegendVLMStreamingDataset stream building and distribution."""

import sys
import traceback
import warnings
from unittest.mock import MagicMock, patch, call

from src.dataset.legendvlm_dataset import LegendVLMStreamingDataset


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
# Tests
# ---------------------------------------------------------------------------

@patch(f"{MOD}.load_dataset")
@patch(f"{MOD}.interleave_datasets")
def test_build_stream_train_interleaves(mock_interleave, mock_load):
    """Train mode with multiple paths calls interleave_datasets + shuffle."""
    ds1, ds2 = make_mock_stream("ds1"), make_mock_stream("ds2")
    mock_load.side_effect = [ds1, ds2]
    merged = make_mock_stream("merged")
    mock_interleave.return_value = merged

    obj = LegendVLMStreamingDataset(
        dataset_paths=["/fake/a", "/fake/b"],
        split="train", mode="train",
    )
    mock_interleave.assert_called_once()
    merged.shuffle.assert_called_once()
    assert obj.stream is not None


@patch(f"{MOD}.load_dataset")
@patch(f"{MOD}.concatenate_datasets")
def test_build_stream_val_concatenates(mock_concat, mock_load):
    """Val mode with multiple paths calls concatenate_datasets, no shuffle."""
    ds1, ds2 = make_mock_stream("ds1"), make_mock_stream("ds2")
    mock_load.side_effect = [ds1, ds2]
    merged = make_mock_stream("merged")
    mock_concat.return_value = merged

    obj = LegendVLMStreamingDataset(
        dataset_paths=["/fake/a", "/fake/b"],
        split="test", mode="val",
    )
    mock_concat.assert_called_once()
    merged.shuffle.assert_not_called()


@patch(f"{MOD}.load_dataset")
@patch(f"{MOD}.interleave_datasets")
def test_single_path_no_interleave(mock_interleave, mock_load):
    """Single path: no interleave/concatenate, just the raw stream."""
    ds = make_mock_stream("ds")
    mock_load.return_value = ds

    obj = LegendVLMStreamingDataset(
        dataset_paths=["/fake/a"],
        split="train", mode="train",
    )
    mock_interleave.assert_not_called()
    # shuffle is still called for train mode
    ds.shuffle.assert_called_once()


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

    result = LegendVLMStreamingDataset.infer_proportional_probs([ds1, ds2])
    assert result == [1000.0, 3000.0]


def test_infer_probs_no_metadata():
    """When info is None, return None."""
    ds = MagicMock()
    ds.info = None
    result = LegendVLMStreamingDataset.infer_proportional_probs([ds])
    assert result is None


@patch(f"{MOD}.load_dataset")
def test_distribute_calls_split_by_node(mock_load):
    """distribute() calls split_dataset_by_node with correct args."""
    ds = make_mock_stream("ds")
    mock_load.return_value = ds

    obj = LegendVLMStreamingDataset(
        dataset_paths=["/fake/a"],
        split="train", mode="train",
    )

    with patch(f"datasets.distributed.split_dataset_by_node") as mock_split:
        mock_split.return_value = MagicMock(name="split_stream")
        obj.distribute(rank=1, world_size=4)
        mock_split.assert_called_once()
        _, kwargs = mock_split.call_args
        assert kwargs["rank"] == 1
        assert kwargs["world_size"] == 4


@patch(f"{MOD}.load_dataset")
def test_distribute_different_ranks(mock_load):
    """Different ranks get different stream objects after distribute."""
    ds = make_mock_stream("ds")
    mock_load.return_value = ds

    obj1 = LegendVLMStreamingDataset(
        dataset_paths=["/fake/a"], split="train", mode="train",
    )
    obj2 = LegendVLMStreamingDataset(
        dataset_paths=["/fake/a"], split="train", mode="train",
    )

    with patch(f"datasets.distributed.split_dataset_by_node") as mock_split:
        stream_r0 = MagicMock(name="stream_r0")
        stream_r1 = MagicMock(name="stream_r1")
        mock_split.side_effect = [stream_r0, stream_r1]

        obj1.distribute(rank=0, world_size=2)
        obj2.distribute(rank=1, world_size=2)

        assert obj1.stream is stream_r0
        assert obj2.stream is stream_r1


@patch(f"{MOD}.load_dataset")
def test_distribute_none_stream(mock_load):
    """distribute with stream=None does not raise."""
    mock_load.side_effect = Exception("fail")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = LegendVLMStreamingDataset(
            dataset_paths=["/fake/a"], split="train", mode="train",
        )
    assert obj.stream is None
    obj.distribute(rank=0, world_size=2)  # should not raise


@patch(f"{MOD}.load_dataset")
@patch(f"{MOD}.interleave_datasets")
def test_probs_length_mismatch_fallback(mock_interleave, mock_load):
    """dataset_probs length mismatch falls back to proportional-to-size."""
    ds1, ds2 = make_mock_stream("ds1"), make_mock_stream("ds2")
    mock_load.side_effect = [ds1, ds2]
    merged = make_mock_stream("merged")
    mock_interleave.return_value = merged

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = LegendVLMStreamingDataset(
            dataset_paths=["/fake/a", "/fake/b"],
            split="train", mode="train",
            dataset_probs=[0.3],  # length 1 != 2 streams
        )
    # Should still call interleave (with fallback probs)
    mock_interleave.assert_called_once()


@patch(f"{MOD}.load_dataset")
def test_return_dataset_info_maps_episode_index(mock_load):
    """return_dataset_info=True causes .map() to be called on each loaded stream."""
    ds = make_mock_stream("ds")
    mapped_ds = make_mock_stream("mapped_ds")
    ds.map = MagicMock(return_value=mapped_ds)
    mock_load.return_value = ds

    obj = LegendVLMStreamingDataset(
        dataset_paths=["/fake/a"],
        split="train", mode="train",
        return_dataset_info=True,
    )
    ds.map.assert_called_once()
    # Verify map was called with with_indices=True
    _, kwargs = ds.map.call_args
    assert kwargs.get("with_indices") is True


@patch(f"{MOD}.load_dataset")
def test_return_dataset_info_false_no_map(mock_load):
    """return_dataset_info=False (default) does not call .map()."""
    ds = make_mock_stream("ds")
    ds.map = MagicMock()
    mock_load.return_value = ds

    obj = LegendVLMStreamingDataset(
        dataset_paths=["/fake/a"],
        split="train", mode="train",
        return_dataset_info=False,
    )
    ds.map.assert_not_called()


@patch(f"{MOD}.load_dataset")
@patch(f"{MOD}.interleave_datasets")
def test_return_dataset_info_multiple_paths(mock_interleave, mock_load):
    """return_dataset_info=True maps episode_index on every loaded stream."""
    ds1 = make_mock_stream("ds1")
    ds2 = make_mock_stream("ds2")
    mapped1 = make_mock_stream("mapped1")
    mapped2 = make_mock_stream("mapped2")
    ds1.map = MagicMock(return_value=mapped1)
    ds2.map = MagicMock(return_value=mapped2)
    mock_load.side_effect = [ds1, ds2]
    merged = make_mock_stream("merged")
    mock_interleave.return_value = merged

    obj = LegendVLMStreamingDataset(
        dataset_paths=["/fake/a", "/fake/b"],
        split="train", mode="train",
        return_dataset_info=True,
    )
    ds1.map.assert_called_once()
    ds2.map.assert_called_once()
    # interleave should receive the mapped streams, not the originals
    args, _ = mock_interleave.call_args
    assert mapped1 in args[0]
    assert mapped2 in args[0]


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
