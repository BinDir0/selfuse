"""Unit tests for VLAWdsDataset-specific WebDataset behavior."""

from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from src.dataset.vla_dataset import VLAWdsDataset
from src.dataset.wds_dataset import build_wds_pipeline


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
        "future_frame": {"horizon": 3, "stride": 1},
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


def test_video_fps_follows_image_stride():
    shape_meta = _shape_meta()
    shape_meta["obs"]["rgb"]["stride"] = 2

    dataset = VLAWdsDataset(
        wds_datasets=[{"name": "demo", "shard_urls": "/tmp/unused/shard-*.tar"}],
        shape_meta=shape_meta,
        mode="val",
        video_base_fps=30.0,
    )

    data = dataset.build_raw_model_inputs(
        instruction="pick up object",
        image=np.zeros((1, 4, 4, 3), dtype=np.uint8),
        intrinsic=np.ones(4, dtype=np.float32),
    )

    assert data["vision_type"] == "video"
    assert data["video_fps"] == np.array(15.0, dtype=np.float32)


def test_sample_to_data_keeps_truncated_action_shape_metadata():
    dataset = VLAWdsDataset(
        wds_datasets=[{"name": "demo", "shard_urls": "/tmp/unused/shard-*.tar"}],
        shape_meta=_shape_meta(),
        mode="val",
    )

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
        with patch("src.dataset.vla_dataset.process_image", return_value=(mocked_image, None, sample["intrinsic"])):
            data = dataset.sample_to_data(sample)

    assert data["images"].shape == (1, 4, 4, 3)
    assert data["actions"].shape == (4, 48)
    assert data["n_actions"] == np.array(2, dtype=np.int32)
    assert data["actions_valid_mask"][:2].all()
    assert not data["actions_valid_mask"][2:].any()
    assert data["vision_type"] == "video"
    assert data["video_fps"] == np.array(30.0, dtype=np.float32)


def test_sample_to_data_returns_raw_fields_when_collator_is_set():
    dataset = VLAWdsDataset(
        wds_datasets=[{"name": "demo", "shard_urls": "/tmp/unused/shard-*.tar"}],
        shape_meta=_shape_meta(),
        mode="val",
    )
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
        with patch("src.dataset.vla_dataset.process_image", return_value=(mocked_image, None, sample["intrinsic"])):
            data = dataset.sample_to_data(sample)

    assert "input_ids" not in data
    assert data["images"].shape == (1, 4, 4, 3)
    assert data["instruction"] == "pick up object"
    assert data["vision_type"] == "video"
    assert data["video_fps"] == np.array(30.0, dtype=np.float32)
    assert data["n_actions"] == np.array(2, dtype=np.int32)


def test_sample_to_data_pads_future_frames_and_tracks_valid_count():
    dataset = VLAWdsDataset(
        wds_datasets=[{"name": "demo", "shard_urls": "/tmp/unused/shard-*.tar"}],
        shape_meta=_shape_meta(),
        mode="val",
    )
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
        "future_frames": np.full((2, 4, 4, 3), 7, dtype=np.uint8),
        "valid_future_frame_len": 2,
    }

    mocked_state = np.zeros((2, 48), dtype=np.float32)
    mocked_action = np.ones((2, 48), dtype=np.float32)
    mocked_image = np.zeros((1, 4, 4, 3), dtype=np.uint8)

    with patch("src.dataset.vla_dataset.process_state_action", return_value=(mocked_state, mocked_action)):
        with patch("src.dataset.vla_dataset.process_image", return_value=(mocked_image, None, sample["intrinsic"])):
            data = dataset.sample_to_data(sample)

    assert data["future_frames"].shape == (3, 4, 4, 3)
    assert data["n_future_frames"] == np.array(2, dtype=np.int32)
    assert np.all(data["future_frames"][:2] == 7)
    assert np.all(data["future_frames"][2] == 0)


def test_sample_to_data_surfaces_breast_fields_when_breast_image_present():
    """Dual-view sample should produce breast_images/breast_intrinsic/breast_future_frames."""
    dataset = VLAWdsDataset(
        wds_datasets=[{"name": "demo", "shard_urls": "/tmp/unused/shard-*.tar"}],
        shape_meta=_shape_meta(),
        mode="val",
        load_breast_camera=True,
    )
    sample = {
        "wrist_state": np.zeros((2, 18), dtype=np.float32),
        "hand_state": np.zeros((2, 30), dtype=np.float32),
        "wrist_action": np.zeros((2, 18), dtype=np.float32),
        "hand_action": np.zeros((2, 30), dtype=np.float32),
        "extrinsic": np.eye(4, dtype=np.float32).reshape(-1),
        "intrinsic": np.ones(4, dtype=np.float32),
        "breast_extrinsic": np.eye(4, dtype=np.float32).reshape(-1),
        "breast_intrinsic": np.full(4, 2.0, dtype=np.float32),
        "instruction": ["pick up object"],
        "instruction_num": 1,
        "image": np.zeros((1, 4, 4, 3), dtype=np.uint8),
        "breast_image": np.full((1, 4, 4, 3), 99, dtype=np.uint8),
        "future_frames": np.full((2, 4, 4, 3), 7, dtype=np.uint8),
        "breast_future_frames": np.full((2, 4, 4, 3), 8, dtype=np.uint8),
        "valid_future_frame_len": 2,
    }

    mocked_state = np.zeros((2, 48), dtype=np.float32)
    mocked_action = np.ones((2, 48), dtype=np.float32)
    mocked_image = np.zeros((1, 4, 4, 3), dtype=np.uint8)

    def fake_process_image(img, _depth, intr, *_args, **_kwargs):
        return mocked_image, None, intr

    with patch("src.dataset.vla_dataset.process_state_action", return_value=(mocked_state, mocked_action)):
        with patch("src.dataset.vla_dataset.process_image", side_effect=fake_process_image):
            data = dataset.sample_to_data(sample)

    assert "breast_images" in data
    assert data["breast_images"].shape == (1, 4, 4, 3)
    assert np.allclose(data["breast_intrinsic"], 2.0)
    assert "breast_future_frames" in data
    assert data["breast_future_frames"].shape == (3, 4, 4, 3)
    assert np.all(data["breast_future_frames"][:2] == 8)
    assert np.all(data["breast_future_frames"][2] == 0)


def test_sample_to_data_omits_breast_keys_when_absent():
    """Head-only sample should not produce any breast_* keys."""
    dataset = VLAWdsDataset(
        wds_datasets=[{"name": "demo", "shard_urls": "/tmp/unused/shard-*.tar"}],
        shape_meta=_shape_meta(),
        mode="val",
    )
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
        with patch("src.dataset.vla_dataset.process_image", return_value=(mocked_image, None, sample["intrinsic"])):
            data = dataset.sample_to_data(sample)

    assert "breast_images" not in data
    assert "breast_intrinsic" not in data
    assert "breast_future_frames" not in data


def test_build_wds_pipeline_expands_listconfig_globs_into_one_subset(tmp_path):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    (left_dir / "shard-000001.tar").touch()
    (left_dir / "shard-000003.tar").touch()
    (right_dir / "shard-000002.tar").touch()

    shard_urls_cfg = OmegaConf.create(
        {
            "shard_urls": [
                str(left_dir / "shard-*.tar"),
                str(right_dir / "shard-*.tar"),
            ]
        }
    )
    captured = {}

    class DummyPipeline:
        def map(self, _fn):
            return self

        def compose(self, _fn):
            return self

        def shuffle(self, _size):
            return self

    def fake_webdataset(shard_urls, **kwargs):
        captured["shard_urls"] = shard_urls
        captured["kwargs"] = kwargs
        return DummyPipeline()

    with patch("src.dataset.wds_dataset.wds.WebDataset", side_effect=fake_webdataset):
        build_wds_pipeline(
            shard_urls_cfg.shard_urls,
            mode="val",
            lowdim_only=True,
        )

    assert captured["shard_urls"] == [
        str(left_dir / "shard-000001.tar"),
        str(left_dir / "shard-000003.tar"),
        str(right_dir / "shard-000002.tar"),
    ]
    assert captured["kwargs"]["resampled"] is False
