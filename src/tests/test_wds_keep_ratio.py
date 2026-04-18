"""Tests for keep_ratio behavior in build_wds_pipeline and build_blended_dataset.

keep_ratio injects a per-sample Bernoulli(keep_ratio) drop stage between
sliding_window_compose and the shuffle buffer. The tests here cover:
  1. assertion guards (invalid values raise)
  2. default (1.0) produces no drop stage
  3. train + keep_ratio<1.0 inserts .select() and statistically keeps ~ratio
  4. val mode ignores keep_ratio even when < 1.0 (single-pass eval must see all)
"""

from unittest.mock import patch

import pytest

from src.dataset.wds_dataset import build_wds_pipeline


class _RecordingPipeline:
    """Records which FluidInterface methods get chained and their predicates."""

    def __init__(self, name: str = "root"):
        self.name = name
        self.calls: list[tuple[str, object]] = []

    def _record(self, method: str, arg):
        self.calls.append((method, arg))
        return self

    def map(self, fn):
        return self._record("map", fn)

    def compose(self, fn):
        return self._record("compose", fn)

    def shuffle(self, *args, **kwargs):
        return self._record("shuffle", (args, kwargs))

    def select(self, predicate):
        return self._record("select", predicate)


def _build_with_dummy_wds(**kwargs):
    """Invoke build_wds_pipeline with wds.WebDataset mocked out.

    Returns the _RecordingPipeline produced inside (not the outer return value,
    which may be the same object). We stop the pipeline from doing anything
    real by passing shard_urls that expand to at least one URL, and by mocking
    wds.WebDataset to return our recorder.
    """
    pipeline = _RecordingPipeline()

    def fake_webdataset(shard_urls, **_kw):
        return pipeline

    with patch("src.dataset.wds_dataset.wds.WebDataset", side_effect=fake_webdataset):
        build_wds_pipeline(["/tmp/fake/shard-000001.tar"], **kwargs)
    return pipeline


def test_keep_ratio_assert_rejects_zero():
    with pytest.raises(AssertionError, match="keep_ratio"):
        build_wds_pipeline(["/tmp/fake/shard-000001.tar"], keep_ratio=0.0)


def test_keep_ratio_assert_rejects_negative():
    with pytest.raises(AssertionError, match="keep_ratio"):
        build_wds_pipeline(["/tmp/fake/shard-000001.tar"], keep_ratio=-0.1)


def test_keep_ratio_assert_rejects_above_one():
    with pytest.raises(AssertionError, match="keep_ratio"):
        build_wds_pipeline(["/tmp/fake/shard-000001.tar"], keep_ratio=1.5)


def test_keep_ratio_default_inserts_no_select_stage():
    pipeline = _build_with_dummy_wds(mode="train", lowdim_only=True)
    select_calls = [c for c in pipeline.calls if c[0] == "select"]
    assert select_calls == [], "default keep_ratio=1.0 must not insert a .select() stage"


def test_keep_ratio_below_one_inserts_select_in_train():
    pipeline = _build_with_dummy_wds(mode="train", lowdim_only=True, keep_ratio=0.1)
    select_calls = [c for c in pipeline.calls if c[0] == "select"]
    assert len(select_calls) == 1, "train mode with keep_ratio<1.0 must insert exactly one .select()"


def test_keep_ratio_below_one_skipped_in_val():
    pipeline = _build_with_dummy_wds(mode="val", lowdim_only=True, keep_ratio=0.1)
    select_calls = [c for c in pipeline.calls if c[0] == "select"]
    assert select_calls == [], "val mode must skip keep_ratio drop so full val set is evaluated"


def test_keep_ratio_predicate_distribution_is_bernoulli():
    """The injected predicate should keep samples with probability ~keep_ratio."""
    pipeline = _build_with_dummy_wds(mode="train", lowdim_only=True, keep_ratio=0.25)
    select_calls = [c for c in pipeline.calls if c[0] == "select"]
    assert len(select_calls) == 1
    predicate = select_calls[0][1]

    import random as _random
    _random.seed(1234)
    n_trials = 20000
    kept = sum(1 for _ in range(n_trials) if predicate({"x": 0}))
    empirical_ratio = kept / n_trials
    # 4σ bound for Binomial(20000, 0.25): σ = sqrt(20000*0.25*0.75) ≈ 61.2,
    # so 4σ ≈ 245 → ±0.0123 on ratio. Use 0.02 to be safe.
    assert abs(empirical_ratio - 0.25) < 0.02, (
        f"keep probability ~0.25 expected, got empirical {empirical_ratio:.4f}"
    )


def test_build_blended_dataset_forwards_keep_ratio():
    """Per-subset pipelines built inside build_blended_dataset should receive keep_ratio."""
    from src.dataset.wds_dataset import build_blended_dataset

    captured_kwargs = []

    def fake_build_wds_pipeline(*args, **kwargs):
        captured_kwargs.append(kwargs)
        # Return a no-op iterable so the outer DataPipeline construction works
        return iter([])

    datasets_config = [
        {"name": "a", "shard_urls": "/tmp/a/shard-*.tar", "weight": 1.0},
        {"name": "b", "shard_urls": "/tmp/b/shard-*.tar", "weight": 2.0},
    ]
    with patch("src.dataset.wds_dataset.build_wds_pipeline", side_effect=fake_build_wds_pipeline):
        # mode='val' avoids the RandomMix/DataPipeline path that would require
        # real iterators. We only need to verify keep_ratio propagation.
        build_blended_dataset(datasets_config, mode="val", keep_ratio=0.3)

    assert len(captured_kwargs) == 2
    for kw in captured_kwargs:
        assert kw["keep_ratio"] == 0.3
