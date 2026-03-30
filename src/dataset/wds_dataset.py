"""
WebDataset pipeline utilities for LegendVLA training.

Provides sliding window compose for action chunk assembly and
pipeline builders for single/blended WebDataset sources.
"""

import collections
import glob
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

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


def _is_shard_sequence(shard_patterns):
    """Return True when *shard_patterns* is a non-string sequence of shard entries."""
    return isinstance(shard_patterns, Sequence) and not isinstance(shard_patterns, (str, bytes, os.PathLike))


def _is_glob_pattern(shard_entry):
    """Return True when a shard entry uses shell-style glob wildcards."""
    return any(token in shard_entry for token in "*?[")


def expand_shard_patterns(shard_patterns):
    """Expand shard patterns into explicit shard paths.

    Supports a single string/path-like shard pattern or a sequence of patterns.
    Sequence entries are expanded independently and concatenated so one logical
    dataset subset can keep a single downstream shuffle buffer.

    Glob patterns that match nothing are ignored; literal paths and WebDataset
    brace patterns are preserved as-is.
    """
    if isinstance(shard_patterns, (str, bytes, os.PathLike)):
        shard_entries = [os.fspath(shard_patterns)]
        shard_patterns_metadata = shard_entries[0]
    elif _is_shard_sequence(shard_patterns):
        shard_entries = [os.fspath(entry) for entry in shard_patterns]
        shard_patterns_metadata = list(shard_entries)
    else:
        raise TypeError(
            "shard_patterns must be a string/path-like value or a sequence of shard patterns, "
            f"got {type(shard_patterns)!r}"
        )

    shard_urls = []
    for entry in shard_entries:
        matches = sorted(glob.glob(entry))
        if matches:
            shard_urls.extend(matches)
        elif not _is_glob_pattern(entry):
            shard_urls.append(entry)

    return shard_urls, shard_patterns_metadata


@dataclass
class WindowConfig:
    """Sampling window parameters for sliding window compose."""
    action_horizon: int = 32
    action_stride: int = 1
    state_horizon: int = 16
    state_stride: int = 2
    image_horizon: int = 1
    image_stride: int = 30
    history_pad_mode: str = "repeat"
    future_pad_mode: str = "repeat"
    future_frame_horizon: int = 0
    future_frame_stride: int = 1

    def __post_init__(self):
        valid_modes = {"repeat", "truncate"}
        if self.history_pad_mode not in valid_modes:
            raise ValueError(f"Invalid history pad mode: {self.history_pad_mode}")
        if self.future_pad_mode not in valid_modes:
            raise ValueError(f"Invalid future pad mode: {self.future_pad_mode}")

    @property
    def past_size(self):
        """Max number of past frames needed for state and image."""
        return max(
            (self.state_horizon - 1) * self.state_stride,
            (self.image_horizon - 1) * self.image_stride,
        )
    
    @property
    def future_size(self):
        """Number of future frames (including current) needed for action + future frame prediction."""
        action_max = (self.action_horizon - 1) * self.action_stride
        ff_max = self.future_frame_horizon * self.future_frame_stride
        return max(action_max, ff_max) + 1


def decode_sample_fields(sample, lowdim_only=False):
    """Decode metadata fields from a raw WebDataset sample.

    Always decodes meta.json and lowdim.npy.  When *lowdim_only* is True
    every other key is dropped so the sample is as small as possible
    (used by normalizer-fitting pipelines).  Otherwise image/depth fields
    are kept as raw bytes for deferred decoding after the shuffle buffer.
    """
    import io as _io
    meta = sample.get("meta.json")
    if isinstance(meta, bytes):
        sample["meta.json"] = json.loads(meta.decode("utf-8"))
    ld = sample.get("lowdim.npy")
    if isinstance(ld, bytes):
        sample["lowdim.npy"] = np.load(_io.BytesIO(ld))
    if lowdim_only:
        return {
            "__key__": sample.get("__key__", ""),
            "meta.json": sample["meta.json"],
            "lowdim.npy": sample.get("lowdim.npy"),
        }
    return sample


def decode_media_fields(sample):
    """Decode image/depth bytes into PIL Image / numpy array.

    Called after the shuffle buffer so that downstream consumers (VLA /
    VLM datasets) always receive decoded media, keeping the deferred-
    decode logic internal to the pipeline.
    """
    from PIL import Image
    import io as _io
    for key in list(sample.keys()):
        val = sample[key]
        if not isinstance(val, bytes):
            continue
        if key.endswith(".jpg") or key.endswith(".jpeg") or key.endswith(".png"):
            sample[key] = Image.open(_io.BytesIO(val)).convert("RGB")
        elif key.endswith(".npy"):
            sample[key] = np.load(_io.BytesIO(val))
    return sample


def unpack_lowdim(lowdim):
    """Unpack a (116,) float32 lowdim vector into named fields."""
    return {k: lowdim[s:e] for k, (s, e) in LOWDIM_SLICES.items()}


def gather_history_frames(past, buf, horizon, stride, pad_mode):
    """Gather history frames from past buffer in causal order.

    Args:
        past: deque of past frames (already yielded)
        buf: deque of current + future frames, buf[0] is current
        horizon: number of frames to gather (including current)
        stride: temporal stride between frames
        pad_mode: "repeat" or "truncate" for history features

    Returns:
        List of frames in causal order (oldest first, current last).
        Length is horizon in repeat mode, or <= horizon in truncate mode.
    """
    frames = []
    # Gather past frames: from oldest to newest
    for i in range(horizon - 1, 0, -1):
        offset = i * stride
        if offset <= len(past):
            frames.append(past[-offset])
        elif pad_mode == "repeat":
            # Not enough past: repeat earliest available
            if past:
                frames.append(past[0])
            else:
                frames.append(buf[0])
        # truncate mode: skip missing frames

    # Append current frame at the end
    frames.append(buf[0])
    return frames


def build_sample_from_window(buf, past, config, lowdim_slices, lowdim_only=False):
    """Build a training sample from the sliding window buffer.

    Lowdim fields are materialized eagerly because they are small. RGB/depth
    media stay as frame references so WebDataset shuffle buffers only retain
    lightweight window descriptors; media arrays are copied later, after
    shuffling, by ``materialize_sample_media``.

    Args:
        buf: deque of decoded samples, buf[0] is the current frame
        past: deque of past frames (already yielded), past[-1] is most recent
        config: WindowConfig with sampling parameters
        lowdim_slices: dict mapping field names to (start, end) index pairs
        lowdim_only: if True, only extract lowdim fields (skip image/depth)
    """
    current = buf[0]
    meta = current["meta.json"]

    # --- Action chunk: vectorized gather of future frames ---
    n_avail = min(config.future_size, len(buf))
    lowdims = np.stack([buf[i]["lowdim.npy"] for i in range(0, n_avail, config.action_stride)], axis=0)

    len_lowdims = len(lowdims)
    if config.future_pad_mode == "repeat":
        if len_lowdims < config.action_horizon:
            pad = np.tile(lowdims[-1:], (config.action_horizon - len_lowdims, 1))
            lowdims_full = np.concatenate([lowdims, pad], axis=0)
        else:
            lowdims_full = lowdims
    elif config.future_pad_mode == "truncate":
        lowdims_full = lowdims
    else:
        raise ValueError(f"Invalid future pad mode: {config.future_pad_mode}")

    ws, we = lowdim_slices['wrist_action']
    hs, he = lowdim_slices['hand_action']
    wrist_action = lowdims_full[:, ws:we]  # (H, 18)
    hand_action = lowdims_full[:, hs:he]   # (H, 30)

    # --- State: gather history frames and extract state slices ---
    state_frames = gather_history_frames(
        past, buf, config.state_horizon, config.state_stride, config.history_pad_mode)
    state_lds = np.stack([f["lowdim.npy"] for f in state_frames], axis=0)

    wss, wse = lowdim_slices['wrist_state']
    hss, hse = lowdim_slices['hand_state']
    wrist_state = state_lds[:, wss:wse]  # (state_horizon, 18)
    hand_state = state_lds[:, hss:hse]   # (state_horizon, 30)

    # --- Image/depth: keep frame references for post-shuffle materialization ---
    image_frame_refs = None
    if not lowdim_only:
        image_frames = gather_history_frames(
            past, buf, config.image_horizon, config.image_stride, config.history_pad_mode)
        image_frame_refs = tuple(image_frames)

    # --- Future frames for world model supervision ---
    future_frame_refs = None
    if not lowdim_only and config.future_frame_horizon > 0:
        ff_refs = []
        for i in range(config.future_frame_horizon):
            offset = (i + 1) * config.future_frame_stride
            if offset < len(buf):
                ff_refs.append(buf[offset])
            elif config.future_pad_mode == "repeat":
                ff_refs.append(buf[len(buf) - 1])
        if ff_refs:
            future_frame_refs = tuple(ff_refs)

    # --- Extrinsic / Intrinsic: from current frame ---
    ld = current["lowdim.npy"]
    es, ee = lowdim_slices['extrinsic']
    ins, ine = lowdim_slices['intrinsic']
    extrinsic = ld[es:ee]
    intrinsic = ld[ins:ine]

    # --- Instruction ---
    instruction = meta["instruction"]
    instruction_num = meta["instruction_num"]

    # --- Presence: hand visibility flag (1=left, 2=right, 3=both) ---
    presence = meta.get("presence", 3)

    result = {
        "valid_action_len": len_lowdims,
        "wrist_state": wrist_state.astype(np.float32),
        "hand_state": hand_state.astype(np.float32),
        "wrist_action": wrist_action.astype(np.float32),
        "hand_action": hand_action.astype(np.float32),
        "extrinsic": extrinsic.astype(np.float32),
        "intrinsic": intrinsic.astype(np.float32),
        "instruction": instruction,
        "instruction_num": instruction_num,
        "presence": int(presence),
        "dataset_name": meta.get("dataset_name", ""),
        "episode_index": meta.get("episode_index", 0),
    }
    if image_frame_refs is not None:
        result["image_frame_refs"] = image_frame_refs
    if future_frame_refs is not None:
        result["future_frame_refs"] = future_frame_refs
        result["valid_future_frame_len"] = len(future_frame_refs)
    return result


def decode_image_bytes(raw):
    """Decode a single image from raw bytes or PIL Image to numpy array."""
    if isinstance(raw, bytes):
        from PIL import Image
        import io
        image = np.array(Image.open(io.BytesIO(raw)).convert("RGB"), copy=False)
    else:
        # Already decoded (PIL Image or numpy array)
        image = np.array(raw, copy=True)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return image


def decode_depth_bytes(raw):
    """Decode a single depth map from raw npy bytes or numpy array."""
    if isinstance(raw, bytes):
        import io
        return np.load(io.BytesIO(raw))
    return np.array(raw, copy=True)


def materialize_sample_media(sample):
    """Materialize RGB/depth arrays from frame refs and drop the refs.

    Supports both deferred-decode mode (raw bytes) and legacy mode
    (already-decoded PIL Images / numpy arrays).  This runs after
    shuffle so buffered samples share underlying frame objects.
    """
    image_frame_refs = sample.pop("image_frame_refs", None)
    if image_frame_refs is not None:
        images = [decode_image_bytes(frame["image.jpg"]) for frame in image_frame_refs]
        sample["image"] = np.stack(images, axis=0)

    if image_frame_refs and image_frame_refs[-1].get("depth.npy") is not None:
        depth_list = [
            decode_depth_bytes(frame["depth.npy"])
            for frame in image_frame_refs
            if frame.get("depth.npy") is not None
        ]
        if len(depth_list) != len(images):
            raise ValueError(f"Depth list length {len(depth_list)} does not match image list length {len(images)}")
        if depth_list:
            sample["depth"] = np.stack(depth_list, axis=0)

    future_frame_refs = sample.pop("future_frame_refs", None)
    if future_frame_refs is not None:
        ff_images = [decode_image_bytes(frame["image.jpg"]) for frame in future_frame_refs]
        sample["future_frames"] = np.stack(ff_images, axis=0)

    return sample


def sliding_window_compose(src, config, lowdim_slices, lowdim_only=False):
    """Compose filter: sliding window over episode frames.

    Guarantees:
    - Frames within an episode are contiguous and ordered in the shard
    - Yields a sample as soon as action_horizon future frames are available
    - At episode boundary, clamps action indices to the last frame (padding)
    - Maintains a past deque for correct state/image history

    Latency: only need to buffer action_horizon frames before first yield,
    NOT the entire episode. This prevents worker stalls on long episodes.
    """
    buf = collections.deque()
    past = collections.deque(maxlen=config.past_size)
    cur_ep = None

    for sample in src:
        meta = sample["meta.json"]
        ep_key = (meta.get("dataset_name", ""), meta["episode_index"])

        if ep_key != cur_ep:
            # Episode boundary: flush remaining frames with clamped actions
            while buf:
                yield build_sample_from_window(buf, past, config, lowdim_slices, lowdim_only)
                past.append(buf.popleft())
            past.clear()
            cur_ep = ep_key

        buf.append(sample)

        # Yield as soon as we have enough future context
        if len(buf) > config.future_size:
            yield build_sample_from_window(buf, past, config, lowdim_slices, lowdim_only)
            past.append(buf.popleft())

    # Final flush
    while buf:
        yield build_sample_from_window(buf, past, config, lowdim_slices, lowdim_only)
        past.append(buf.popleft())


def no_split(src):
    """Identity splitter: yield all shards to every worker/node."""
    yield from src


def select_lowdim_files(fname):
    """Keep only lowdim metadata files for lowdim-only pipelines."""
    return fname.endswith("meta.json") or fname.endswith("lowdim.npy")


def build_wds_pipeline(shard_urls, config=None, lowdim_slices=None,
                       preprocess_fn=None, shuffle_buffer=16384, mode='train',
                       use_sliding_window=True, lowdim_only=False,
                       include_post_stages=True):
    """Build a WebDataset pipeline for a single dataset.

    Training: resampled infinite stream with shard-level shuffle.
    Validation: finite single-pass, deterministic order, no shuffle.

    When *include_post_stages* is False the pipeline stops after
    sliding-window compose (or decode), omitting shuffle / media
    materialization / preprocess. This allows build_blended_dataset
    to attach a single shared shuffle buffer after RandomMix.

    Args:
        shard_urls: list of shard tar paths, a braceexpand pattern string,
                    or a list of glob pattern strings (each expanded separately)
        config: WindowConfig with sampling parameters (uses defaults if None)
        lowdim_slices: dict mapping field names to (start, end) pairs
        preprocess_fn: optional callable(sample_dict) -> sample_dict
        shuffle_buffer: sample-level shuffle buffer size (train only)
        mode: 'train' or 'val'
        use_sliding_window: whether to compose sliding windows (VLA=True, VLM=False)
        lowdim_only: if True, only decode lowdim.npy and meta.json (skip image/depth)
        include_post_stages: if False, skip shuffle / materialize / preprocess
    """
    if config is None:
        config = WindowConfig()
    if lowdim_slices is None:
        lowdim_slices = LOWDIM_SLICES

    shard_urls, shard_patterns_metadata = expand_shard_patterns(shard_urls)
    assert shard_urls, f"No shards found: {shard_patterns_metadata}"

    is_train = (mode == 'train')
    select_files = select_lowdim_files if lowdim_only else None

    # resampled=True uses ResampledShards (shardlists.py) whose seed mixes
    # worker_seed/epoch with pid/time_ns/os.urandom, giving each worker/node
    # an independent random shard sequence. Explicit node/worker splitters
    # are therefore redundant and would only discard generated URLs.
    # Ref: webdataset/shardlists.py ResampledShards.__iter__
    pipeline = wds.WebDataset(
        shard_urls,
        shardshuffle=False,
        nodesplitter=no_split if is_train else wds.split_by_node,
        workersplitter=no_split if is_train else wds.shardlists.split_by_worker,
        resampled=is_train,
        empty_check=False,
        select_files=select_files,
    )
    # Deferred decode: parse meta.json and lowdim.npy eagerly;
    # image/depth stay as compressed bytes through the shuffle buffer.
    pipeline = pipeline.map(
        lambda s: decode_sample_fields(s, lowdim_only=lowdim_only)
    )

    if use_sliding_window:
        pipeline = pipeline.compose(
            lambda src: sliding_window_compose(src, config, lowdim_slices, lowdim_only)
        )

    if not include_post_stages:
        return pipeline

    # Shuffle before media materialization so the buffer retains lightweight
    # window descriptors with shared frame refs rather than copied image arrays.
    if is_train and shuffle_buffer and shuffle_buffer > 0:
        pipeline = pipeline.shuffle(shuffle_buffer, initial=shuffle_buffer)

    # Materialize media after shuffle: VLA path decodes from frame refs,
    # non-sliding-window path (VLM) decodes raw bytes in-place.
    if not lowdim_only:
        if use_sliding_window:
            pipeline = pipeline.map(materialize_sample_media)
        else:
            pipeline = pipeline.map(decode_media_fields)

    if preprocess_fn is not None:
        pipeline = pipeline.map(preprocess_fn)

    return pipeline


def build_blended_dataset(datasets_config, config=None, lowdim_slices=None,
                          preprocess_fn=None, shuffle_buffer=16384, mode='train',
                          use_sliding_window=True, lowdim_only=False):
    """Build a blended dataset from multiple WebDataset sources.

    Training: per-subset pipelines (no per-subset shuffle) mixed via
    RandomMix, then a single shared shuffle buffer + media decode +
    preprocess. This keeps memory independent of subset count N.
    Validation: per-subset pipelines concatenated for single-pass evaluation.

    Args:
        datasets_config: list of dicts with keys:
            - shard_urls: list of shard tar paths or glob pattern
            - weight: sampling weight (train only)
        config: WindowConfig with sampling parameters (uses defaults if None)
        lowdim_slices: dict mapping field names to (start, end) pairs
        preprocess_fn: optional preprocess function
        shuffle_buffer: sample-level shuffle buffer size (train only)
        mode: 'train' or 'val'
        use_sliding_window: whether to compose sliding windows (VLA=True, VLM=False)
        lowdim_only: if True, only decode lowdim.npy and meta.json (skip image/depth)
    """
    if config is None:
        config = WindowConfig()
    if lowdim_slices is None:
        lowdim_slices = LOWDIM_SLICES

    is_train = (mode == 'train')

    subsets = []
    weights = []
    for c in datasets_config:
        urls = c["shard_urls"]
        # Train: each subset produces raw samples (no shuffle / materialize /
        # preprocess); these stages are applied once after RandomMix.
        pipe = build_wds_pipeline(
            urls, config, lowdim_slices,
            preprocess_fn=preprocess_fn if not is_train else None,
            shuffle_buffer=shuffle_buffer,
            mode=mode,
            use_sliding_window=use_sliding_window,
            lowdim_only=lowdim_only,
            include_post_stages=not is_train,
        )
        subsets.append(pipe)
        weights.append(c.get("weight", 1.0))

    assert subsets, "No shards found across all datasets."

    if not is_train:
        def chain_pipelines():
            for pipe in subsets:
                yield from pipe
        return chain_pipelines()

    # Train: RandomMix → single shuffle → materialize → preprocess.
    # RandomMix is an IterableDataset (not FluidInterface), so wrap
    # with DataPipeline to append callable stages.
    mixed = subsets[0] if len(subsets) == 1 else wds.RandomMix(subsets, weights, longest=False)

    stages = [mixed]
    if shuffle_buffer and shuffle_buffer > 0:
        stages.append(wds.shuffle(shuffle_buffer, initial=shuffle_buffer))

    if not lowdim_only:
        if use_sliding_window:
            stages.append(wds.map(materialize_sample_media))
        else:
            stages.append(wds.map(decode_media_fields))

    if preprocess_fn is not None:
        stages.append(wds.map(preprocess_fn))

    return wds.DataPipeline(*stages)
