"""Unit tests for sliding window state ordering and padding behavior."""

import collections
import sys
import traceback

import numpy as np

from src.dataset.wds_dataset import (
    WindowConfig,
    LOWDIM_SLICES,
    build_sample_from_window,
    gather_future_refs,
    materialize_sample_media,
    sliding_window_compose,
)


def make_frame(frame_idx, episode_index=0, dataset_name="test", with_depth=False):
    """Create a mock frame where lowdim encodes the frame index for tracing."""
    lowdim = np.full(116, float(frame_idx), dtype=np.float32)
    img = np.full((4, 4, 3), frame_idx, dtype=np.uint8)
    meta = {
        "dataset_name": dataset_name,
        "episode_index": episode_index,
        "instruction": "pick up",
        "instruction_num": 1,
        "presence": 3,
    }
    frame = {"lowdim.npy": lowdim, "image.jpg": img, "meta.json": meta}
    if with_depth:
        frame["depth.npy"] = np.full((4, 4), frame_idx, dtype=np.uint16)
    return frame


def make_episode(n_frames, episode_index=0):
    return [make_frame(i, episode_index=episode_index) for i in range(n_frames)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_state_includes_current_frame():
    """The last element of wrist_state must be the current frame."""
    config = WindowConfig(
        state_horizon=4, state_stride=1,
        action_horizon=2, image_horizon=1, image_stride=1,
    )
    past = collections.deque(make_episode(10), maxlen=config.past_size)
    buf = collections.deque([make_frame(10), make_frame(11), make_frame(12)])

    sample = build_sample_from_window(buf, past, config, LOWDIM_SLICES)
    ws = sample["wrist_state"]

    assert ws.shape[0] == config.state_horizon
    np.testing.assert_allclose(ws[-1], 10.0)


def test_state_causal_order():
    """State rows must be in chronological order: oldest first."""
    config = WindowConfig(
        state_horizon=4, state_stride=2,
        action_horizon=2, image_horizon=1, image_stride=1,
    )
    past = collections.deque(
        [make_frame(i) for i in range(20)], maxlen=config.past_size)
    buf = collections.deque([make_frame(20), make_frame(21)])

    sample = build_sample_from_window(buf, past, config, LOWDIM_SLICES)
    ws = sample["wrist_state"]

    # state_horizon=4, stride=2 => offsets -6, -4, -2, 0 relative to current
    # current=20 => frames 14, 16, 18, 20
    expected = [14.0, 16.0, 18.0, 20.0]
    actual = [ws[i, 0] for i in range(4)]
    assert actual == expected, f"Expected {expected}, got {actual}"


def test_state_causal_order_stride1():
    """Stride=1 case: consecutive frames."""
    config = WindowConfig(
        state_horizon=3, state_stride=1,
        action_horizon=2, image_horizon=1, image_stride=1,
    )
    past = collections.deque(
        [make_frame(i) for i in range(5)], maxlen=config.past_size)
    buf = collections.deque([make_frame(5), make_frame(6)])

    sample = build_sample_from_window(buf, past, config, LOWDIM_SLICES)
    ws = sample["wrist_state"]

    expected = [3.0, 4.0, 5.0]
    actual = [ws[i, 0] for i in range(3)]
    assert actual == expected


def test_state_repeat_no_past():
    """Episode start: no past at all, all state slots repeat current frame."""
    config = WindowConfig(
        state_horizon=4, state_stride=2, history_pad_mode="repeat",
        action_horizon=4, image_horizon=1, image_stride=1,
    )
    past = collections.deque(maxlen=config.past_size)
    buf = collections.deque(make_episode(4))

    sample = build_sample_from_window(buf, past, config, LOWDIM_SLICES)
    ws = sample["wrist_state"]

    assert ws.shape[0] == config.state_horizon
    for i in range(config.state_horizon):
        np.testing.assert_allclose(ws[i], 0.0)


def test_state_repeat_partial_past():
    """Partial past: some slots repeat earliest available frame."""
    config = WindowConfig(
        state_horizon=4, state_stride=2, history_pad_mode="repeat",
        action_horizon=2, image_horizon=1, image_stride=1,
    )
    # past = [frame0, frame1, frame2], current = frame3
    # offsets needed: -6, -4, -2
    # -6 > len=3 => repeat past[0]=0
    # -4 > len=3 => repeat past[0]=0
    # -2 <= 3 => past[-2]=frame1
    # current = frame3
    past = collections.deque(
        [make_frame(i) for i in range(3)], maxlen=config.past_size)
    buf = collections.deque([make_frame(3), make_frame(4)])

    sample = build_sample_from_window(buf, past, config, LOWDIM_SLICES)
    ws = sample["wrist_state"]

    expected = [0.0, 0.0, 1.0, 3.0]
    actual = [ws[i, 0] for i in range(4)]
    assert actual == expected, f"Expected {expected}, got {actual}"


def test_action_repeat_padding():
    """Action chunk pads with last available frame in repeat mode."""
    config = WindowConfig(
        action_horizon=6, future_pad_mode="repeat",
        state_horizon=1, state_stride=1,
        image_horizon=1, image_stride=1,
    )
    past = collections.deque(maxlen=config.past_size)
    buf = collections.deque([make_frame(10), make_frame(11), make_frame(12)])

    sample = build_sample_from_window(buf, past, config, LOWDIM_SLICES)
    wa = sample["wrist_action"]

    assert wa.shape[0] == 6
    expected = [10.0, 11.0, 12.0, 12.0, 12.0, 12.0]
    actual = [wa[i, 0] for i in range(6)]
    assert actual == expected


def test_state_truncate_no_past():
    """Truncate mode with no past: state has only current frame."""
    config = WindowConfig(
        state_horizon=4, state_stride=2, history_pad_mode="truncate",
        action_horizon=2, image_horizon=1, image_stride=1,
    )
    past = collections.deque(maxlen=config.past_size)
    buf = collections.deque([make_frame(5), make_frame(6)])

    sample = build_sample_from_window(buf, past, config, LOWDIM_SLICES)
    ws = sample["wrist_state"]

    assert ws.shape[0] == 1
    np.testing.assert_allclose(ws[0], 5.0)


def test_action_truncate():
    """Truncate mode: action chunk is shorter than horizon."""
    config = WindowConfig(
        action_horizon=6, future_pad_mode="truncate",
        state_horizon=1, state_stride=1,
        image_horizon=1, image_stride=1,
    )
    past = collections.deque(maxlen=config.past_size)
    buf = collections.deque([make_frame(10), make_frame(11)])

    sample = build_sample_from_window(buf, past, config, LOWDIM_SLICES)
    wa = sample["wrist_action"]
    assert wa.shape[0] == 2


def test_action_sampling_uses_own_horizon_when_future_frames_need_longer_buffer():
    """Action chunk length should stay capped by action_horizon even if future_size is larger."""
    config = WindowConfig(
        action_horizon=4,
        action_stride=1,
        future_pad_mode="truncate",
        future_frame_horizon=4,
        future_frame_stride=16,
        state_horizon=1,
        state_stride=1,
        image_horizon=1,
        image_stride=1,
    )
    past = collections.deque(maxlen=config.past_size)
    buf = collections.deque([make_frame(i) for i in range(66)])

    sample = build_sample_from_window(buf, past, config, LOWDIM_SLICES)

    assert config.future_size == 65
    assert sample["valid_action_len"] == 4
    assert sample["wrist_action"].shape[0] == 4
    assert [sample["wrist_action"][i, 0] for i in range(4)] == [0.0, 1.0, 2.0, 3.0]
    assert sample["valid_future_frame_len"] == 4
    future_indices = [frame["lowdim.npy"][0] for frame in sample["future_frame_refs"]]
    assert future_indices == [16.0, 32.0, 48.0, 64.0]


def test_history_and_future_pad_modes_are_independent():
    """History and future padding policies should be configurable independently."""
    config = WindowConfig(
        action_horizon=4,
        state_horizon=3,
        state_stride=1,
        image_horizon=1,
        image_stride=1,
        history_pad_mode="repeat",
        future_pad_mode="truncate",
    )
    past = collections.deque(maxlen=config.past_size)
    buf = collections.deque([make_frame(5), make_frame(6)])

    sample = build_sample_from_window(buf, past, config, LOWDIM_SLICES)

    assert sample["wrist_state"].shape[0] == 3
    assert sample["wrist_action"].shape[0] == 2
    np.testing.assert_allclose(sample["wrist_state"][0], 5.0)


def test_single_episode_state_progression():
    """Walk through a 10-frame episode and check state at each yield."""
    config = WindowConfig(
        action_horizon=3, state_horizon=3, state_stride=1,
        image_horizon=1, image_stride=1, history_pad_mode="repeat",
    )
    frames = make_episode(10)
    samples = list(sliding_window_compose(iter(frames), config, LOWDIM_SLICES))

    assert len(samples) == 10

    for t, s in enumerate(samples):
        ws = s["wrist_state"]
        assert ws.shape[0] == 3
        np.testing.assert_allclose(ws[-1], float(t))

        for i in range(ws.shape[0] - 1):
            assert ws[i, 0] <= ws[i + 1, 0], (
                f"t={t}: not causal, row {i}={ws[i,0]} > row {i+1}={ws[i+1,0]}")


def test_episode_boundary_resets_past():
    """Past buffer clears at episode boundary."""
    config = WindowConfig(
        action_horizon=2, state_horizon=3, state_stride=1,
        image_horizon=1, image_stride=1, history_pad_mode="repeat",
    )
    ep1 = make_episode(5, episode_index=0)
    ep2 = make_episode(5, episode_index=1)
    frames = ep1 + ep2

    samples = list(sliding_window_compose(iter(frames), config, LOWDIM_SLICES))

    ep2_first = samples[5]
    ws = ep2_first["wrist_state"]
    for i in range(3):
        np.testing.assert_allclose(ws[i], 0.0)


def test_window_media_stays_lazy_until_materialized():
    """Window samples should keep media as frame refs until post-shuffle materialization."""
    config = WindowConfig(
        action_horizon=2, state_horizon=1, state_stride=1,
        image_horizon=2, image_stride=1, history_pad_mode="repeat",
    )
    past = collections.deque([make_frame(4, with_depth=True)], maxlen=config.past_size)
    buf = collections.deque([make_frame(5, with_depth=True), make_frame(6, with_depth=True)])

    sample = build_sample_from_window(buf, past, config, LOWDIM_SLICES)

    assert "image" not in sample
    assert "depth" not in sample
    assert len(sample["image_frame_refs"]) == 2


def test_materialize_sample_media_copies_and_drops_refs():
    """Materialization should copy media arrays and release frame refs."""
    config = WindowConfig(
        action_horizon=2, state_horizon=1, state_stride=1,
        image_horizon=2, image_stride=1, history_pad_mode="repeat",
    )
    past = collections.deque([make_frame(4, with_depth=True)], maxlen=config.past_size)
    current = make_frame(5, with_depth=True)
    future = make_frame(6, with_depth=True)
    buf = collections.deque([current, future])

    sample = build_sample_from_window(buf, past, config, LOWDIM_SLICES)
    materialize_sample_media(sample)

    assert "image_frame_refs" not in sample
    assert sample["image"].shape[0] == 2
    assert sample["depth"].shape[0] == 2

    sample["image"][0, 0, 0, 0] = 255
    sample["depth"][0, 0, 0] = 123

    assert current["image.jpg"][0, 0, 0] == 5
    assert current["depth.npy"][0, 0] == 5


def test_lowdim_slices_no_magic_numbers():
    """Verify output uses lowdim_slices, not hardcoded indices."""
    custom_slices = {
        'wrist_state':  (0, 10),
        'hand_state':   (10, 30),
        'wrist_action': (30, 48),
        'hand_action':  (48, 78),
        'extrinsic':    (78, 94),
        'intrinsic':    (94, 98),
    }
    config = WindowConfig(
        action_horizon=2, state_horizon=1, state_stride=1,
        image_horizon=1, image_stride=1,
    )
    past = collections.deque(maxlen=config.past_size)
    buf = collections.deque([make_frame(7), make_frame(8)])

    sample = build_sample_from_window(buf, past, config, custom_slices)

    assert sample["wrist_state"].shape[-1] == 10
    assert sample["hand_state"].shape[-1] == 20
    assert sample["wrist_action"].shape[-1] == 18
    assert sample["hand_action"].shape[-1] == 30
    assert sample["extrinsic"].shape[-1] == 16
    assert sample["intrinsic"].shape[-1] == 4


def test_depth_uses_history_pad_mode_with_image_history():
    """Depth should share the same history sampling/padding policy as image."""
    config = WindowConfig(
        action_horizon=2, state_horizon=1, state_stride=1,
        image_horizon=3, image_stride=2, history_pad_mode="repeat",
    )
    # Past: frames 0..9, current = frame 10
    past = collections.deque(
        [make_frame(i, with_depth=True) for i in range(10)],
        maxlen=config.past_size)
    buf = collections.deque([
        make_frame(10, with_depth=True),
        make_frame(11, with_depth=True)])

    sample = build_sample_from_window(buf, past, config, LOWDIM_SLICES)
    materialize_sample_media(sample)

    # image_horizon=3, stride=2 => offsets -4, -2, 0 => frames 6, 8, 10
    assert sample["image"].shape[0] == 3
    assert sample["depth"].shape[0] == 3

    # Verify depth matches image frames
    expected_frames = [6.0, 8.0, 10.0]
    for i, expected in enumerate(expected_frames):
        np.testing.assert_allclose(sample["image"][i, 0, 0, 0], expected)
        np.testing.assert_allclose(sample["depth"][i, 0, 0], expected)


def test_depth_none_when_missing():
    """Depth should be None if current frame has no depth."""
    config = WindowConfig(
        action_horizon=2, state_horizon=1, state_stride=1,
        image_horizon=2, image_stride=1, history_pad_mode="repeat",
    )
    past = collections.deque(maxlen=config.past_size)
    buf = collections.deque([make_frame(5), make_frame(6)])  # no depth

    sample = build_sample_from_window(buf, past, config, LOWDIM_SLICES)
    materialize_sample_media(sample)
    assert "depth" not in sample or sample["depth"] is None


def test_gather_future_refs_basic():
    """gather_future_refs collects refs from buf with correct offsets."""
    buf = collections.deque([make_frame(i) for i in range(10)])
    refs, valid_count = gather_future_refs(buf, horizon=3, stride=2, pad_mode="repeat", offset_base=0)
    assert valid_count == 3
    assert len(refs) == 3
    assert refs[0]["lowdim.npy"][0] == 0.0
    assert refs[1]["lowdim.npy"][0] == 2.0
    assert refs[2]["lowdim.npy"][0] == 4.0


def test_gather_future_refs_with_offset_base():
    """offset_base shifts the starting position (used for future frames)."""
    buf = collections.deque([make_frame(i) for i in range(20)])
    refs, valid_count = gather_future_refs(buf, horizon=3, stride=4, pad_mode="repeat", offset_base=4)
    assert valid_count == 3
    assert refs[0]["lowdim.npy"][0] == 4.0
    assert refs[1]["lowdim.npy"][0] == 8.0
    assert refs[2]["lowdim.npy"][0] == 12.0


def test_gather_future_refs_repeat_padding():
    """When buf is shorter than needed, repeat last frame."""
    buf = collections.deque([make_frame(i) for i in range(3)])
    refs, valid_count = gather_future_refs(buf, horizon=4, stride=1, pad_mode="repeat", offset_base=0)
    assert valid_count == 3
    assert len(refs) == 4
    assert refs[3]["lowdim.npy"][0] == 2.0


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
