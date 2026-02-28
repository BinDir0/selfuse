"""Unit tests for LegendUnifiedWdsDataset interleaving and padding logic."""

import sys
import traceback

import torch

from src.dataset.legendvla_wds_dataset import LegendUnifiedWdsDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vla_sample(idx):
    """Create a mock VLA sample with typical fields."""
    return {
        "input_ids": torch.tensor([idx, idx + 1, idx + 2]),
        "labels": torch.tensor([idx, idx + 1, idx + 2]),
        "attention_mask": torch.ones(3, dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 4, 4),
        "states": torch.full((16, 48), float(idx)),
        "actions": torch.full((32, 48), float(idx)),
        "actions_valid_mask": torch.ones(32, 48, dtype=torch.bool),
        "n_states": torch.tensor(16, dtype=torch.int32),
        "n_actions": torch.tensor(32, dtype=torch.int32),
        "answer_start_idx": torch.tensor(1),
        "is_vla_data": torch.tensor(True),
        "has_depth_values": torch.tensor(False),
    }


def make_vlm_sample(idx):
    """Create a mock VLM sample (no states/actions)."""
    return {
        "input_ids": torch.tensor([idx + 100]),
        "labels": torch.tensor([idx + 100]),
        "attention_mask": torch.ones(1, dtype=torch.long),
        "pixel_values": torch.zeros(2, 3, 4, 4),
        "answer_start_idx": torch.tensor(0),
        "is_vla_data": torch.tensor(False),
    }


VLA_KEYS = {
    "input_ids", "labels", "attention_mask", "pixel_values",
    "states", "actions", "actions_valid_mask",
    "n_states", "n_actions", "answer_start_idx",
    "is_vla_data",
}

# has_depth_values is VLA-only; VLM samples don't get it via pad
VLA_ONLY_KEYS = VLA_KEYS | {"has_depth_values"}


class MockIterableDataset(torch.utils.data.IterableDataset):
    """Finite iterable dataset backed by a list of samples."""

    def __init__(self, samples):
        self.samples = list(samples)

    def __iter__(self):
        return iter(self.samples)


def build_unified(vla_samples, vlm_samples=None, vla_ratio=5/6,
                  batch_size=6, mode="train"):
    """Shortcut to build a LegendUnifiedWdsDataset from sample lists."""
    vla_ds = MockIterableDataset(vla_samples)
    vlm_ds = MockIterableDataset(vlm_samples) if vlm_samples is not None else None
    return LegendUnifiedWdsDataset(
        vla_dataset=vla_ds,
        vlm_dataset=vlm_ds,
        vla_ratio=vla_ratio,
        batch_size=batch_size,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_train_output_keys():
    """VLA samples keep all keys; padded VLM samples have core VLA keys."""
    ds = build_unified(
        [make_vla_sample(i) for i in range(5)],
        [make_vlm_sample(i) for i in range(2)],
        vla_ratio=5/6, batch_size=6,
    )
    samples = list(ds)
    for s in samples:
        if bool(s["is_vla_data"]):
            assert set(s.keys()) >= VLA_ONLY_KEYS, f"Missing keys: {VLA_ONLY_KEYS - set(s.keys())}"
        else:
            assert set(s.keys()) >= VLA_KEYS, f"Missing keys: {VLA_KEYS - set(s.keys())}"


def test_val_output_keys():
    """Val mode samples also have core VLA keys."""
    ds = build_unified(
        [make_vla_sample(i) for i in range(3)],
        [make_vlm_sample(i) for i in range(2)],
        mode="val",
    )
    samples = list(ds)
    assert len(samples) == 5
    for s in samples:
        if bool(s["is_vla_data"]):
            assert set(s.keys()) >= VLA_ONLY_KEYS
        else:
            assert set(s.keys()) >= VLA_KEYS


def test_train_interleaving_ratio():
    """batch_size=6, vla_ratio=5/6 -> 5 VLA then 1 VLM per batch."""
    n_vla = 10
    ds = build_unified(
        [make_vla_sample(i) for i in range(n_vla)],
        [make_vlm_sample(i) for i in range(5)],
        vla_ratio=5/6, batch_size=6,
    )
    samples = list(ds)
    flags = [bool(s["is_vla_data"]) for s in samples]
    # Pattern: [True]*5 + [False]*1 + [True]*5 + [False]*1
    assert flags == [True]*5 + [False]*1 + [True]*5 + [False]*1


def test_train_interleaving_custom_ratio():
    """batch_size=4, vla_ratio=0.5 -> 2 VLA then 2 VLM per batch."""
    n_vla = 4
    ds = build_unified(
        [make_vla_sample(i) for i in range(n_vla)],
        [make_vlm_sample(i) for i in range(10)],
        vla_ratio=0.5, batch_size=4,
    )
    samples = list(ds)
    flags = [bool(s["is_vla_data"]) for s in samples]
    # Pattern: [True]*2 + [False]*2 + [True]*2 + [False]*2
    assert flags == [True]*2 + [False]*2 + [True]*2 + [False]*2


def test_vlm_auto_restart():
    """VLM has only 1 sample; it auto-restarts when exhausted."""
    ds = build_unified(
        [make_vla_sample(i) for i in range(10)],
        [make_vlm_sample(0)],
        vla_ratio=5/6, batch_size=6,
    )
    samples = list(ds)
    vlm_samples = [s for s in samples if not bool(s["is_vla_data"])]
    # 10 VLA -> 2 batches of 5 -> 2 VLM insertions, each 1 sample
    assert len(vlm_samples) == 2
    # Both VLM samples should have the same input_ids (restarted from same source)
    assert torch.equal(vlm_samples[0]["input_ids"], vlm_samples[1]["input_ids"])


def test_val_sequential_vla_then_vlm():
    """Val mode: all VLA first, then all VLM."""
    n_vla, n_vlm = 4, 3
    ds = build_unified(
        [make_vla_sample(i) for i in range(n_vla)],
        [make_vlm_sample(i) for i in range(n_vlm)],
        mode="val",
    )
    samples = list(ds)
    flags = [bool(s["is_vla_data"]) for s in samples]
    assert flags == [True]*n_vla + [False]*n_vlm


def test_val_single_pass_finite():
    """Val mode iterates exactly once; two passes yield identical results."""
    n_vla, n_vlm = 3, 2
    ds = build_unified(
        [make_vla_sample(i) for i in range(n_vla)],
        [make_vlm_sample(i) for i in range(n_vlm)],
        mode="val",
    )
    first_pass = list(ds)
    second_pass = list(ds)
    assert len(first_pass) == n_vla + n_vlm
    assert len(first_pass) == len(second_pass)


def test_train_vla_only():
    """vlm_dataset=None -> only VLA samples produced."""
    n_vla = 5
    ds = build_unified(
        [make_vla_sample(i) for i in range(n_vla)],
        vlm_samples=None,
    )
    samples = list(ds)
    assert len(samples) == n_vla
    assert all(bool(s["is_vla_data"]) for s in samples)


def test_pad_vlm_sample():
    """pad_vlm_sample adds zero-filled states/actions and sets n_states/n_actions=0."""
    vlm = make_vlm_sample(0)
    shape_meta = {
        "states": torch.Size([16, 48]),
        "actions": torch.Size([32, 48]),
        "n_states": torch.Size([]),
        "n_actions": torch.Size([]),
    }
    LegendUnifiedWdsDataset.pad_vlm_sample(vlm, shape_meta)

    assert vlm["states"].shape == (16, 48)
    assert vlm["actions"].shape == (32, 48)
    assert vlm["actions_valid_mask"].shape == (32, 48)
    assert torch.all(vlm["states"] == 0)
    assert torch.all(vlm["actions"] == 0)
    assert vlm["n_states"].item() == 0
    assert vlm["n_actions"].item() == 0
    # Original keys unchanged
    assert torch.equal(vlm["input_ids"], torch.tensor([100]))
    assert bool(vlm["is_vla_data"]) is False


def test_distribute_delegates_to_vlm():
    """distribute(rank, world_size) is forwarded to vlm_dataset."""
    vla_ds = MockIterableDataset([make_vla_sample(0)])
    vlm_ds = MockIterableDataset([make_vlm_sample(0)])
    vlm_ds.distribute = lambda rank, world_size: None
    called = {}

    def mock_distribute(rank, world_size):
        called["rank"] = rank
        called["world_size"] = world_size

    vlm_ds.distribute = mock_distribute
    unified = LegendUnifiedWdsDataset(
        vla_dataset=vla_ds, vlm_dataset=vlm_ds,
        mode="train",
    )
    unified.distribute(rank=2, world_size=8)
    assert called == {"rank": 2, "world_size": 8}


def test_return_dataset_info_passthrough():
    """VLA samples with dataset_info fields pass through interleaving intact."""
    vla_samples = []
    for i in range(5):
        s = make_vla_sample(i)
        s["dataset_name"] = f"ds_{i}"
        s["episode_index"] = torch.tensor(i * 10)
        vla_samples.append(s)
    vlm_samples = []
    for i in range(2):
        s = make_vlm_sample(i)
        s["dataset_name"] = f"vlm_{i}"
        s["dataset_local_idx"] = torch.tensor(i, dtype=torch.int32)
        vlm_samples.append(s)

    ds = build_unified(vla_samples, vlm_samples, vla_ratio=5/6, batch_size=6)
    samples = list(ds)

    vla_out = [s for s in samples if bool(s["is_vla_data"])]
    vlm_out = [s for s in samples if not bool(s["is_vla_data"])]

    # VLA samples retain dataset_name and episode_index
    for i, s in enumerate(vla_out):
        assert s["dataset_name"] == f"ds_{i}"
        assert s["episode_index"].item() == i * 10

    # VLM samples retain dataset_name and dataset_local_idx after padding
    for i, s in enumerate(vlm_out):
        assert s["dataset_name"] == f"vlm_{i}"
        assert s["dataset_local_idx"].item() == i


def test_return_dataset_info_vlm_padded_keeps_info():
    """pad_vlm_sample does not overwrite dataset_info fields."""
    vlm = make_vlm_sample(0)
    vlm["dataset_name"] = "my_vlm"
    vlm["dataset_local_idx"] = torch.tensor(42, dtype=torch.int32)

    shape_meta = {
        "states": torch.Size([16, 48]),
        "actions": torch.Size([32, 48]),
        "n_states": torch.Size([]),
        "n_actions": torch.Size([]),
    }
    LegendUnifiedWdsDataset.pad_vlm_sample(vlm, shape_meta)

    # Padding adds states/actions but does not touch dataset_info
    assert vlm["dataset_name"] == "my_vlm"
    assert vlm["dataset_local_idx"].item() == 42
    assert vlm["states"].shape == (16, 48)


def test_distribute_no_vlm():
    """distribute with vlm_dataset=None does not raise."""
    unified = build_unified(
        [make_vla_sample(0)],
        vlm_samples=None,
    )
    unified.distribute(rank=0, world_size=4)  # should not raise


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
