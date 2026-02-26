"""
WebDataset pipeline utilities for LegendVLA training.

Provides sliding window compose for action chunk assembly and
pipeline builders for single/blended WebDataset sources.
"""

import collections
import os
import glob
import json
import random

import numpy as np
import webdataset as wds


# lowdim.npy layout (116,) float32:
#   [0:18]    state/wrist
#   [18:48]   state/hand
#   [48:66]   action/wrist
#   [66:96]   action/hand
#   [96:112]  extrinsic
#   [112:116] intrinsic
LOWDIM_SLICES = {
    'wrist_state':  (0, 18),
    'hand_state':   (18, 48),
    'wrist_action': (48, 66),
    'hand_action':  (66, 96),
    'extrinsic':    (96, 112),
    'intrinsic':    (112, 116),
}


def decode_meta(sample):
    """Decode meta.json from bytes to dict if needed."""
    meta = sample.get("meta.json")
    if isinstance(meta, bytes):
        meta = json.loads(meta.decode("utf-8"))
        sample["meta.json"] = meta
    return sample


def decode_lowdim(sample):
    """Decode lowdim.npy from bytes to numpy array if needed."""
    ld = sample.get("lowdim.npy")
    if isinstance(ld, bytes):
        import io
        ld = np.load(io.BytesIO(ld))
        sample["lowdim.npy"] = ld
    return sample


def unpack_lowdim(lowdim):
    """Unpack a (116,) float32 lowdim vector into named fields."""
    return {k: lowdim[s:e] for k, (s, e) in LOWDIM_SLICES.items()}


def build_sample_from_window(buf, action_horizon, state_horizon, state_stride,
                             image_horizon, image_stride):
    """Build a training sample from the sliding window buffer.

    Output dict matches SequenceSampler.sample_sequence() format so that
    existing sample_to_data() can be reused without modification.

    Args:
        buf: deque of decoded samples, buf[0] is the current frame
        action_horizon: number of future action steps
        state_horizon: number of past state steps
        state_stride: stride between state steps
        image_horizon: number of past image steps
        image_stride: stride between image steps
    """
    current = buf[0]
    meta = current["meta.json"]
    ld = current["lowdim.npy"]  # (116,)

    # --- Action chunk: gather future action_horizon steps, clamp at boundary ---
    action_wrist_list = []
    action_hand_list = []
    for i in range(action_horizon):
        idx = min(i, len(buf) - 1)
        a = buf[idx]["lowdim.npy"]
        action_wrist_list.append(a[48:66])
        action_hand_list.append(a[66:96])
    wrist_action = np.stack(action_wrist_list, axis=0)   # (action_horizon, 18)
    hand_action = np.stack(action_hand_list, axis=0)      # (action_horizon, 30)

    # --- State: gather past state_horizon steps with stride, clamp at boundary ---
    # buf[0] is the current frame; past frames are not in the buffer
    # (they were already popped). For WebDataset streaming, we only have
    # the current frame's state available, so we replicate it.
    wrist_state = np.tile(ld[0:18], (state_horizon, 1))   # (state_horizon, 18)
    hand_state = np.tile(ld[18:48], (state_horizon, 1))    # (state_horizon, 30)

    # --- Image: current frame only (image_horizon=1 in default config) ---
    image = np.array(current["image.png"])  # (H, W, 3) uint8
    if image.ndim == 2:
        # grayscale edge case
        image = np.stack([image] * 3, axis=-1)
    image = image[np.newaxis, ...]  # (1, H, W, 3)

    # --- Depth: current frame ---
    depth = current.get("depth.npy")  # (H, W) uint16 or None

    # --- Extrinsic / Intrinsic ---
    extrinsic = ld[96:112]    # (16,)
    intrinsic = ld[112:116]   # (4,)

    # --- Instruction ---
    instruction = meta["instruction"]
    instruction_num = meta["instruction_num"]

    # --- Presence: per-frame [left, right] ---
    presence = meta.get("presence", [1, 1])

    result = {
        "wrist_state":    wrist_state.astype(np.float32),
        "hand_state":     hand_state.astype(np.float32),
        "wrist_action":   wrist_action.astype(np.float32),
        "hand_action":    hand_action.astype(np.float32),
        "image":          image,
        "extrinsic":      extrinsic.astype(np.float32),
        "intrinsic":      intrinsic.astype(np.float32),
        "instruction":    instruction,
        "instruction_num": instruction_num,
        "presence":       np.array(presence, dtype=np.int32),
    }
    if depth is not None:
        result["depth"] = depth
    return result


def sliding_window_compose(src, action_horizon=32, state_horizon=16,
                           state_stride=2, image_horizon=1, image_stride=30):
    """Compose filter: sliding window over episode frames.

    Guarantees:
    - Frames within an episode are contiguous and ordered in the shard
    - Yields a sample as soon as action_horizon future frames are available
    - At episode boundary, clamps action indices to the last frame (padding)

    Latency: only need to buffer action_horizon frames before first yield,
    NOT the entire episode. This prevents worker stalls on long episodes.
    """
    buf = collections.deque()
    cur_ep = None

    for sample in src:
        meta = sample["meta.json"]
        ep_key = (meta.get("dataset_name", ""), meta["episode_index"])

        if ep_key != cur_ep:
            # Episode boundary: flush remaining frames with clamped actions
            while buf:
                yield build_sample_from_window(
                    buf, action_horizon, state_horizon, state_stride,
                    image_horizon, image_stride)
                buf.popleft()
            cur_ep = ep_key

        buf.append(sample)

        # Yield as soon as we have enough future context
        if len(buf) > action_horizon:
            yield build_sample_from_window(
                buf, action_horizon, state_horizon, state_stride,
                image_horizon, image_stride)
            buf.popleft()

    # Final flush
    while buf:
        yield build_sample_from_window(
            buf, action_horizon, state_horizon, state_stride,
            image_horizon, image_stride)
        buf.popleft()


def no_split(src):
    """Identity splitter: yield all shards to every worker/node."""
    yield from src


def build_wds_pipeline(shard_urls, action_horizon=32, state_horizon=16,
                       state_stride=2, image_horizon=1, image_stride=30,
                       preprocess_fn=None, shuffle_buffer=8192):
    """Build a WebDataset training pipeline for a single dataset.

    Args:
        shard_urls: list of shard tar paths, or a braceexpand pattern string
        action_horizon: number of future action steps for action chunk
        state_horizon: number of past state steps
        state_stride: stride between state steps
        image_horizon: number of past image steps
        image_stride: stride between image steps
        preprocess_fn: optional callable(sample_dict) -> sample_dict
        shuffle_buffer: sample-level shuffle buffer size
    """
    if isinstance(shard_urls, str):
        shard_urls = sorted(glob.glob(shard_urls))
    assert shard_urls, f"No shards found: {shard_urls}"

    pipeline = (
        wds.WebDataset(
            shard_urls,
            shardshuffle=True,
            nodesplitter=wds.split_by_node,
            resampled=True,
        )
        .decode("pil")
        .map(decode_meta)
        .map(decode_lowdim)
        .compose(lambda src: sliding_window_compose(
            src, action_horizon, state_horizon, state_stride,
            image_horizon, image_stride))
    )

    if preprocess_fn is not None:
        pipeline = pipeline.map(preprocess_fn)

    pipeline = (
        pipeline
        .shuffle(shuffle_buffer, initial=shuffle_buffer)
    )

    return pipeline


def build_blended_dataset(datasets_config, action_horizon=32, state_horizon=16,
                          state_stride=2, image_horizon=1, image_stride=30,
                          preprocess_fn=None, shuffle_buffer=8192):
    """Build a blended dataset from multiple WebDataset sources.

    Args:
        datasets_config: list of dicts with keys:
            - shard_urls: list of shard tar paths or glob pattern
            - weight: sampling weight
        action_horizon: number of future action steps
        state_horizon: number of past state steps
        state_stride: stride between state steps
        image_horizon: number of past image steps
        image_stride: stride between image steps
        preprocess_fn: optional preprocess function
        shuffle_buffer: sample-level shuffle buffer size
    """
    subsets = []
    weights = []
    for c in datasets_config:
        urls = c["shard_urls"]
        if isinstance(urls, str):
            urls = sorted(glob.glob(urls))
        if not urls:
            print(f"Warning: No shards found for {c.get('name', '?')}, skipping.")
            continue
        pipe = build_wds_pipeline(
            urls, action_horizon, state_horizon, state_stride,
            image_horizon, image_stride, preprocess_fn, shuffle_buffer)
        subsets.append(pipe)
        weights.append(c.get("weight", 1.0))

    if len(subsets) == 1:
        return subsets[0]

    return wds.RandomMix(subsets, weights, longest=False)
