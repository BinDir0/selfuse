from __future__ import annotations

import numpy as np

import pytest

from src.test.codex.common import build_wrapper_stub


def test_prepare_states_repeat_left_pads_and_promotes_n_states():
    wrapper = build_wrapper_stub(history_pad_mode="repeat")
    padded, n_states = wrapper.prepare_states(np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32))

    expected = np.array(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(padded, expected)
    assert n_states == wrapper.state_horizon


def test_prepare_states_truncate_right_pads_with_zeros_and_keeps_raw_length():
    wrapper = build_wrapper_stub(history_pad_mode="truncate")
    padded, n_states = wrapper.prepare_states(np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32))

    expected = np.array(
        [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(padded, expected)
    assert n_states == 2


def test_prepare_states_empty_repeat_returns_zero_tensor_and_zero_n_states():
    wrapper = build_wrapper_stub(history_pad_mode="repeat")
    padded, n_states = wrapper.prepare_states(np.zeros((0, 2), dtype=np.float32))

    np.testing.assert_allclose(padded, np.zeros((wrapper.state_horizon, wrapper.state_dim), dtype=np.float32))
    assert n_states == 0


def test_prepare_states_long_history_keeps_tail_window():
    wrapper = build_wrapper_stub(history_pad_mode="repeat")
    padded, n_states = wrapper.prepare_states(
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=np.float32)
    )

    expected = np.array(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(padded, expected)
    assert n_states == wrapper.state_horizon


def test_prepare_states_rejects_unknown_history_mode():
    wrapper = build_wrapper_stub(history_pad_mode="invalid")
    with pytest.raises(ValueError, match="Unsupported history_pad_mode"):
        wrapper.prepare_states(np.array([[1.0, 1.0]], dtype=np.float32))

