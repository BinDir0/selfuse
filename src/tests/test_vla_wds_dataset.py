"""Unit tests for VLAWdsDataset-specific WebDataset behavior."""

from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from src.dataset.data_transforms import compute_relative_motion_padded
from src.dataset.vla_dataset import ViewDropoutConfig, VLAWdsDataset
from src.dataset.wds_dataset import build_wds_pipeline


def _shape_meta(
    *,
    history_pad_mode: str = "repeat",
    action_pad_mode: str = "repeat",
    future_frame_pad_mode: str = "repeat",
):
    return {
        "history_pad_mode": history_pad_mode,
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
        "action": {
            "shape": [48], "type": "fingertips",
            "horizon": 4, "stride": 1,
            "pad_mode": action_pad_mode,
        },
        "future_frame": {"horizon": 3, "stride": 1, "pad_mode": future_frame_pad_mode},
    }


def _dual_view_raw_sample():
    return {
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
    }


def _sample_to_data_with_mocked_transforms(dataset, sample):
    mocked_state = np.zeros((2, 48), dtype=np.float32)
    mocked_action = np.ones((2, 48), dtype=np.float32)

    def fake_process_image(img, _depth, intr, *_args, **_kwargs):
        return img.copy(), None, intr

    with patch("src.dataset.vla_dataset.process_state_action", return_value=(mocked_state, mocked_action)):
        with patch("src.dataset.vla_dataset.process_image", side_effect=fake_process_image):
            return dataset.sample_to_data(sample)


def test_window_config_reads_split_pad_modes_from_shape_meta():
    shape_meta = _shape_meta(
        history_pad_mode="repeat",
        action_pad_mode="truncate",
        future_frame_pad_mode="repeat",
    )
    dataset = VLAWdsDataset(
        wds_datasets=[{"name": "demo", "shard_urls": "/tmp/unused/shard-*.tar"}],
        val_wds_datasets=[{"name": "demo_val", "shard_urls": "/tmp/unused-val/shard-*.tar"}],
        shape_meta=shape_meta,
        mode="train",
    )

    assert dataset.window_config.history_pad_mode == "repeat"
    assert dataset.window_config.action_pad_mode == "truncate"
    assert dataset.window_config.future_frame_pad_mode == "repeat"

    val_dataset = dataset.get_validation_dataset()
    assert val_dataset.window_config.history_pad_mode == "repeat"
    assert val_dataset.window_config.action_pad_mode == "truncate"
    assert val_dataset.window_config.future_frame_pad_mode == "repeat"


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
        active_views=["head"],
    )

    assert data["vision_type"] == "video"
    assert data["video_fps"] == np.array(15.0, dtype=np.float32)
    assert data["active_views"] == ["head"]
    assert data["view_mask"].tolist() == [True, False]


def test_sample_to_data_keeps_truncated_action_shape_metadata():
    dataset = VLAWdsDataset(
        wds_datasets=[{"name": "demo", "shard_urls": "/tmp/unused/shard-*.tar"}],
        shape_meta=_shape_meta(),
        mode="val",
        target_image_size=(4, 4),
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
        target_image_size=(4, 4),
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
        target_image_size=(4, 4),
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
        load_breast=True,
        target_image_size=(4, 4),
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


def test_view_dropout_keep_both_marks_both_views_active():
    dataset = VLAWdsDataset(
        wds_datasets=[{"name": "demo", "shard_urls": "/tmp/unused/shard-*.tar"}],
        shape_meta=_shape_meta(),
        mode="train",
        load_breast=True,
        view_dropout=ViewDropoutConfig(drop_head=0.0, drop_breast=0.0),
        target_image_size=(4, 4),
    )

    data = _sample_to_data_with_mocked_transforms(dataset, _dual_view_raw_sample())

    assert data["active_views"] == ["head", "breast"]
    assert data["view_mask"].tolist() == [True, True]
    assert "breast_images" in data
    assert "breast_future_frames" in data
    assert "future_breast_motion" in data


def test_view_dropout_drop_breast_keeps_only_head_active():
    dataset = VLAWdsDataset(
        wds_datasets=[{"name": "demo", "shard_urls": "/tmp/unused/shard-*.tar"}],
        shape_meta=_shape_meta(),
        mode="train",
        load_breast=True,
        view_dropout=ViewDropoutConfig(drop_head=0.0, drop_breast=1.0),
        target_image_size=(4, 4),
    )

    data = _sample_to_data_with_mocked_transforms(dataset, _dual_view_raw_sample())

    assert data["active_views"] == ["head"]
    assert data["view_mask"].tolist() == [True, False]
    assert "breast_images" in data
    assert "breast_future_frames" in data


def test_view_dropout_drop_head_keeps_only_breast_active():
    dataset = VLAWdsDataset(
        wds_datasets=[{"name": "demo", "shard_urls": "/tmp/unused/shard-*.tar"}],
        shape_meta=_shape_meta(),
        mode="train",
        load_breast=True,
        view_dropout=ViewDropoutConfig(drop_head=1.0, drop_breast=0.0),
        target_image_size=(4, 4),
    )

    data = _sample_to_data_with_mocked_transforms(dataset, _dual_view_raw_sample())

    assert data["active_views"] == ["breast"]
    assert data["view_mask"].tolist() == [False, True]
    assert "breast_images" in data
    assert np.allclose(data["breast_intrinsic"], 2.0)
    assert "breast_future_frames" in data


def test_view_dropout_disabled_in_validation():
    dataset = VLAWdsDataset(
        wds_datasets=[{"name": "demo", "shard_urls": "/tmp/unused/shard-*.tar"}],
        shape_meta=_shape_meta(),
        mode="val",
        load_breast=True,
        view_dropout=ViewDropoutConfig(drop_head=1.0, drop_breast=0.0),
        target_image_size=(4, 4),
    )

    data = _sample_to_data_with_mocked_transforms(dataset, _dual_view_raw_sample())

    assert data["active_views"] == ["head", "breast"]
    assert data["view_mask"].tolist() == [True, True]


def test_sample_to_data_marks_head_only_when_breast_absent():
    """Head-only samples still carry view metadata and zero WM breast targets."""
    dataset = VLAWdsDataset(
        wds_datasets=[{"name": "demo", "shard_urls": "/tmp/unused/shard-*.tar"}],
        shape_meta=_shape_meta(),
        mode="val",
        target_image_size=(4, 4),
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
    assert data["active_views"] == ["head"]
    assert data["view_mask"].tolist() == [True, False]
    assert "breast_future_frames" in data
    assert np.all(data["breast_future_frames"] == 0)
    assert np.all(data["future_breast_motion"] == 0)


def _make_world_to_cam(axis_angle_deg: float, position_world: np.ndarray) -> np.ndarray:
    """Construct T_world→cam for a camera placed at *position_world* in world
    coordinates with an in-plane rotation of *axis_angle_deg* around world z
    (cam axes rotated by that angle relative to world axes).

    Returns a [4, 4] float32 matrix such that ``T @ [p_world; 1] = [p_cam; 1]``.
    """
    theta = np.deg2rad(axis_angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    # R_c2w: rotation of cam axes as seen in world.
    R_c2w = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    # T_c2w: [R_c2w | t ; 0 1]; T_w2c = inverse.
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ position_world.astype(np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_w2c
    T[:3, 3] = t_w2c
    return T.astype(np.float32)


def test_compute_relative_motion_identity_when_current_equals_future():
    """Same extrinsic at both timestamps → rel = I."""
    T = _make_world_to_cam(30.0, np.array([0.5, -0.25, 0.75]))
    T_flat = T.reshape(-1)
    future = np.stack([T_flat, T_flat], axis=0)

    out = compute_relative_motion_padded(
        current_flat16=T_flat, future_flat=future, n_valid=2, K=3,
    )

    assert out.shape == (3, 16)
    np.testing.assert_allclose(out[0].reshape(4, 4), np.eye(4), atol=1e-5)
    np.testing.assert_allclose(out[1].reshape(4, 4), np.eye(4), atol=1e-5)
    np.testing.assert_allclose(out[2], np.zeros(16))


def test_compute_relative_motion_rotation_plus_translation_discriminating():
    """Non-commuting (rotation + translation) case: catches the previous
    inv(T_cur) @ T_fut vs correct T_cur @ inv(T_fut) mix-up.

    Under T_w2c convention, the target semantic is
        rel = T_cam_cur ← cam_fut = T_w2c_cur @ inv(T_w2c_fut).
    Verified by mapping the future-cam origin through rel and comparing
    against an independent world→current-cam transform of the future-cam
    world position.
    """
    T_cur = _make_world_to_cam(90.0, np.array([1.0, 0.0, 0.0]))
    T_fut = _make_world_to_cam(180.0, np.array([2.0, 1.0, 0.0]))

    out = compute_relative_motion_padded(
        current_flat16=T_cur.reshape(-1),
        future_flat=T_fut.reshape(-1)[None, :],
        n_valid=1, K=1,
    )
    rel = out[0].reshape(4, 4)

    # Expected: T_w2c_cur @ inv(T_w2c_fut).
    expected = T_cur @ np.linalg.inv(T_fut)
    np.testing.assert_allclose(rel, expected, atol=1e-5)

    # Wrong formula (inv(T_cur) @ T_fut) must differ — if this assert fails,
    # the test case isn't actually discriminating.
    wrong = np.linalg.inv(T_cur) @ T_fut
    assert not np.allclose(rel, wrong, atol=1e-4), (
        "Discriminating test degenerated: rel equals the wrong formula too."
    )


def test_compute_relative_motion_semantic_transports_future_origin():
    """rel[k] must transport a point given in future-cam coords to current-cam
    coords. Check by picking the future-cam origin (future cam's own position
    expressed in current-cam coords)."""
    T_cur = _make_world_to_cam(45.0, np.array([0.3, -0.4, 0.0]))
    T_fut = _make_world_to_cam(-20.0, np.array([1.5, 0.8, 0.2]))

    out = compute_relative_motion_padded(
        current_flat16=T_cur.reshape(-1),
        future_flat=T_fut.reshape(-1)[None, :],
        n_valid=1, K=1,
    )
    rel = out[0].reshape(4, 4)

    # Expected current-cam coords of the future-cam origin:
    # future-cam origin in world = inverse(T_w2c_fut)'s translation.
    fut_origin_world = np.append(np.linalg.inv(T_fut)[:3, 3], 1.0)
    expected_in_cur_cam = T_cur @ fut_origin_world

    # Via rel: transport future-cam origin through rel.
    fut_origin_local = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    got = rel @ fut_origin_local
    np.testing.assert_allclose(got, expected_in_cur_cam, atol=1e-5)


def test_compute_relative_motion_batched_futures_are_per_step():
    """Multiple future poses must be processed independently; K=4 output keeps
    per-step alignment with the input [n_raw, 16] stack."""
    T_cur = _make_world_to_cam(10.0, np.array([0.1, 0.2, 0.3]))
    futures = [
        _make_world_to_cam(15.0, np.array([0.2, 0.2, 0.3])),
        _make_world_to_cam(25.0, np.array([0.4, 0.1, 0.5])),
        _make_world_to_cam(-5.0, np.array([0.0, 0.3, 0.2])),
    ]
    future_flat = np.stack([T.reshape(-1) for T in futures], axis=0)

    out = compute_relative_motion_padded(
        current_flat16=T_cur.reshape(-1),
        future_flat=future_flat,
        n_valid=3, K=4,
    )

    assert out.shape == (4, 16)
    for k in range(3):
        expected = T_cur @ np.linalg.inv(futures[k])
        np.testing.assert_allclose(out[k].reshape(4, 4), expected, atol=1e-5)
    # Padded slot stays zero.
    np.testing.assert_allclose(out[3], np.zeros(16))


def test_compute_relative_motion_pads_tail_when_n_valid_below_K():
    """When fewer future steps than K are valid, the tail of the output stays
    zero. Downstream mask (frame_valid = arange(K) < n_future_frames) drops
    those positions from the loss."""
    T_cur = _make_world_to_cam(0.0, np.zeros(3))
    T_fut = _make_world_to_cam(30.0, np.array([1.0, 0.0, 0.0]))
    futures = np.stack([T_fut.reshape(-1)] * 5, axis=0)  # 5 raw rows

    out = compute_relative_motion_padded(
        current_flat16=T_cur.reshape(-1),
        future_flat=futures,
        n_valid=2, K=4,
    )

    assert out.shape == (4, 16)
    expected_valid = T_cur @ np.linalg.inv(T_fut)
    for k in range(2):
        np.testing.assert_allclose(out[k].reshape(4, 4), expected_valid, atol=1e-5)
    for k in range(2, 4):
        np.testing.assert_allclose(out[k], np.zeros(16))


def test_compute_relative_motion_clamps_when_n_valid_above_K():
    """If caller accidentally passes n_valid > K, the helper clamps to K and
    never indexes past the K-th row of the output buffer."""
    T_cur = _make_world_to_cam(0.0, np.zeros(3))
    T_fut = _make_world_to_cam(10.0, np.array([0.5, 0.0, 0.0]))
    futures = np.stack([T_fut.reshape(-1)] * 10, axis=0)

    out = compute_relative_motion_padded(
        current_flat16=T_cur.reshape(-1),
        future_flat=futures,
        n_valid=7, K=3,
    )

    assert out.shape == (3, 16)
    # All 3 slots produced (none padded) since min(7, 3, 10) == 3.
    expected = T_cur @ np.linalg.inv(T_fut)
    for k in range(3):
        np.testing.assert_allclose(out[k].reshape(4, 4), expected, atol=1e-5)


def test_compute_relative_motion_returns_zeros_when_source_missing():
    """Defensive path: missing inputs or n_valid<=0 returns a zero buffer."""
    assert np.all(
        compute_relative_motion_padded(None, None, 0, 4) == 0.0
    )
    T = _make_world_to_cam(10.0, np.array([0.1, 0.2, 0.3]))
    assert np.all(
        compute_relative_motion_padded(T.reshape(-1), None, 3, 4) == 0.0
    )
    assert np.all(
        compute_relative_motion_padded(None, T.reshape(-1)[None, :], 1, 4) == 0.0
    )
    assert np.all(
        compute_relative_motion_padded(T.reshape(-1), T.reshape(-1)[None, :], 0, 4) == 0.0
    )


def test_sample_to_data_emits_future_head_motion_matching_helper():
    """End-to-end: sample_to_data routes (extrinsic, future_head_extrinsic,
    n_valid) through compute_relative_motion_padded and emits the output as
    future_head_motion with shape [K, 16]."""
    dataset = VLAWdsDataset(
        wds_datasets=[{"name": "demo", "shard_urls": "/tmp/unused/shard-*.tar"}],
        shape_meta=_shape_meta(),
        mode="val",
        target_image_size=(4, 4),
    )
    T_cur = _make_world_to_cam(30.0, np.array([0.2, 0.1, 0.3]))
    T_fut0 = _make_world_to_cam(40.0, np.array([0.3, 0.1, 0.3]))
    T_fut1 = _make_world_to_cam(50.0, np.array([0.4, 0.2, 0.3]))
    sample = {
        "wrist_state": np.zeros((2, 18), dtype=np.float32),
        "hand_state": np.zeros((2, 30), dtype=np.float32),
        "wrist_action": np.zeros((2, 18), dtype=np.float32),
        "hand_action": np.zeros((2, 30), dtype=np.float32),
        "extrinsic": T_cur.reshape(-1),
        "intrinsic": np.ones(4, dtype=np.float32),
        "instruction": ["pick up object"],
        "instruction_num": 1,
        "image": np.zeros((1, 4, 4, 3), dtype=np.uint8),
        "future_frames": np.zeros((2, 4, 4, 3), dtype=np.uint8),
        "future_head_extrinsic": np.stack([T_fut0.reshape(-1), T_fut1.reshape(-1)], axis=0),
    }

    mocked_state = np.zeros((2, 48), dtype=np.float32)
    mocked_action = np.ones((2, 48), dtype=np.float32)
    mocked_image = np.zeros((1, 4, 4, 3), dtype=np.uint8)

    with patch("src.dataset.vla_dataset.process_state_action", return_value=(mocked_state, mocked_action)):
        with patch("src.dataset.vla_dataset.process_image", return_value=(mocked_image, None, sample["intrinsic"])):
            data = dataset.sample_to_data(sample)

    assert "future_head_motion" in data
    # K=3 in shape_meta; 2 valid, tail padded with zeros.
    assert data["future_head_motion"].shape == (3, 16)
    expected_k0 = T_cur @ np.linalg.inv(T_fut0)
    expected_k1 = T_cur @ np.linalg.inv(T_fut1)
    np.testing.assert_allclose(data["future_head_motion"][0].reshape(4, 4), expected_k0, atol=1e-5)
    np.testing.assert_allclose(data["future_head_motion"][1].reshape(4, 4), expected_k1, atol=1e-5)
    np.testing.assert_allclose(data["future_head_motion"][2], np.zeros(16))


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
            load_image=False,
            load_depth=False,
        )

    assert captured["shard_urls"] == [
        str(left_dir / "shard-000001.tar"),
        str(left_dir / "shard-000003.tar"),
        str(right_dir / "shard-000002.tar"),
    ]
    assert captured["kwargs"]["resampled"] is False
