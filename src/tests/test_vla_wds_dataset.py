"""Unit tests for VLAWdsDataset-specific WebDataset behavior."""

from unittest.mock import patch

import numpy as np

from src.dataset.vla_dataset import VLAWdsDataset


def _shape_meta():
    return {
        "obs": {
            "rgb": {"shape": [4, 4, 3], "type": "rgb", "horizon": 1, "stride": 1},
            "depth": {"shape": [4, 4], "type": "depth", "horizon": 1, "stride": 1},
            "state": {
                "wrist": {"shape": [18]},
                "hand": {"shape": [30]},
                "shape": [48],
                "type": "fingertips",
                "horizon": 2,
                "stride": 1,
            },
        },
        "action": {"shape": [48], "type": "fingertips", "horizon": 4, "stride": 1},
    }


def test_window_config_preserves_separate_pad_modes():
    dataset = VLAWdsDataset(
        wds_datasets=[{"name": "demo", "shard_urls": "/tmp/unused/shard-*.tar"}],
        val_wds_datasets=[{"name": "demo_val", "shard_urls": "/tmp/unused-val/shard-*.tar"}],
        shape_meta=_shape_meta(),
        mode="train",
        history_pad_mode="repeat",
        future_pad_mode="truncate",
    )

    assert dataset.window_config.history_pad_mode == "repeat"
    assert dataset.window_config.future_pad_mode == "truncate"

    val_dataset = dataset.get_validation_dataset()
    assert val_dataset.window_config.history_pad_mode == "repeat"
    assert val_dataset.window_config.future_pad_mode == "truncate"


def test_sample_to_data_uses_truncated_action_shape_for_prompt_tokens():
    dataset = VLAWdsDataset(
        wds_datasets=[{"name": "demo", "shard_urls": "/tmp/unused/shard-*.tar"}],
        shape_meta=_shape_meta(),
        mode="val",
    )

    captured = {}

    def fake_preprocessor(**kwargs):
        captured["actions_len"] = len(kwargs["actions"])
        return {
            "input_ids": np.array([1, 2, 3], dtype=np.int64),
            "answer_start_idx": np.array(2, dtype=np.int64),
            "attention_mask": np.array([1, 1, 1], dtype=np.int64),
            "pixel_values": np.zeros((1, 3, 4, 4), dtype=np.float32),
            "labels": np.array([10, 11, 12], dtype=np.int64),
        }

    dataset.set_preprocessor(fake_preprocessor)

    sample = {
        "wrist_state": np.zeros((2, 18), dtype=np.float32),
        "hand_state": np.zeros((2, 30), dtype=np.float32),
        "wrist_action": np.zeros((2, 18), dtype=np.float32),
        "hand_action": np.zeros((2, 30), dtype=np.float32),
        "extrinsic": np.eye(4, dtype=np.float32).reshape(-1),
        "intrinsic": np.ones(4, dtype=np.float32),
        "instruction": ["pick up object"],
        "instruction_num": 1,
        "image": np.zeros((1, 4, 4, 3), dtype=np.uint8),
    }

    mocked_state = np.zeros((2, 48), dtype=np.float32)
    mocked_action = np.ones((2, 48), dtype=np.float32)
    mocked_image = np.zeros((1, 4, 4, 3), dtype=np.uint8)

    with patch("src.dataset.vla_dataset.process_state_action", return_value=(mocked_state, mocked_action)):
        with patch("src.dataset.vla_dataset.process_image", return_value=(mocked_image, None)):
            data = dataset.sample_to_data(sample)

    assert captured["actions_len"] == 2
    assert data["actions"].shape == (4, 48)
    assert data["n_actions"] == np.array(2, dtype=np.int32)
    assert data["actions_valid_mask"][:2].all()
    assert not data["actions_valid_mask"][2:].any()
