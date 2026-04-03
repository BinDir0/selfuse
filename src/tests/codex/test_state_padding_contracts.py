from __future__ import annotations

import numpy as np

from src.tests.codex.common import build_wrapper_stub


def test_prepare_history_repeat_left_pads_and_promotes_n_states():
    wrapper = build_wrapper_stub(history_pad_mode="repeat")
    padded, n_valid = wrapper.prepare_history(
        np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32), wrapper.state_horizon,
    )

    expected = np.array(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(padded, expected)
    assert n_valid == wrapper.state_horizon


def test_prepare_history_truncate_returns_as_is_with_raw_length():
    wrapper = build_wrapper_stub(history_pad_mode="truncate")
    padded, n_valid = wrapper.prepare_history(
        np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32), wrapper.state_horizon,
    )

    expected = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(padded, expected)
    assert n_valid == 2


def test_prepare_history_empty_repeat_returns_zero_tensor_and_zero_count():
    wrapper = build_wrapper_stub(history_pad_mode="repeat")
    padded, n_valid = wrapper.prepare_history(
        np.zeros((0, 2), dtype=np.float32), wrapper.state_horizon,
    )

    np.testing.assert_allclose(padded, np.zeros((wrapper.state_horizon, wrapper.state_dim), dtype=np.float32))
    assert n_valid == 0


def test_prepare_history_long_history_keeps_tail_window():
    wrapper = build_wrapper_stub(history_pad_mode="repeat")
    padded, n_valid = wrapper.prepare_history(
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=np.float32),
        wrapper.state_horizon,
    )

    expected = np.array(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(padded, expected)
    assert n_valid == wrapper.state_horizon


def test_prepare_history_repeat_pads_images():
    wrapper = build_wrapper_stub(history_pad_mode="repeat")
    wrapper.image_horizon = 4
    images = np.ones((2, 8, 8, 3), dtype=np.uint8)
    padded, n_valid = wrapper.prepare_history(images, wrapper.image_horizon)

    assert padded.shape == (4, 8, 8, 3)
    # First two frames are repeat-padded copies of earliest frame
    np.testing.assert_array_equal(padded[0], images[0])
    np.testing.assert_array_equal(padded[1], images[0])
    # Last two are original
    np.testing.assert_array_equal(padded[2], images[0])
    np.testing.assert_array_equal(padded[3], images[1])
    assert n_valid == 4
