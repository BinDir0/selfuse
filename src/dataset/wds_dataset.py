"""
WebDataset pipeline utilities for LegendVLA training.

Provides sliding window compose for action chunk assembly and
pipeline builders for single/blended WebDataset sources.
"""

import collections
import glob
import json
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
        """Number of future frames(including current) needed for action chunk."""
        return (self.action_horizon - 1) * self.action_stride + 1


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


def decode_lowdim_only(sample):
    """Decode only meta.json and lowdim.npy, drop image/depth bytes.

    Used by lowdim-only pipelines (e.g. normalizer fitting) to avoid
    the cost of JPEG decompression and PIL Image creation.
    """
    import io
    result = {"__key__": sample.get("__key__", "")}
    meta = sample.get("meta.json")
    if isinstance(meta, bytes):
        result["meta.json"] = json.loads(meta.decode("utf-8"))
    elif meta is not None:
        result["meta.json"] = meta
    ld = sample.get("lowdim.npy")
    if isinstance(ld, bytes):
        result["lowdim.npy"] = np.load(io.BytesIO(ld))
    elif ld is not None:
        result["lowdim.npy"] = ld
    return result


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
    return result


def materialize_sample_media(sample):
    """Materialize RGB/depth arrays from frame refs and drop the refs.

    This runs after shuffle so buffered samples share underlying decoded frame
    objects. The returned arrays are copied to keep per-sample media private for
    later augmentation / preprocessing.
    """
    image_frame_refs = sample.pop("image_frame_refs", None)
    if image_frame_refs is not None:
        images = []
        for frame in image_frame_refs:
            image = np.array(frame["image.jpg"], copy=True)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            images.append(image)
        sample["image"] = np.stack(images, axis=0)

    if image_frame_refs and image_frame_refs[-1].get("depth.npy") is not None:
        depth_list = [
            np.array(frame["depth.npy"], copy=True)
            for frame in image_frame_refs
            if frame.get("depth.npy") is not None
        ]
        if depth_list:
            sample["depth"] = np.stack(depth_list, axis=0)

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


def build_wds_pipeline(shard_urls, config=None, lowdim_slices=None,
                       preprocess_fn=None, shuffle_buffer=8192, mode='train',
                       use_sliding_window=True, lowdim_only=False):
    """Build a WebDataset pipeline for a single dataset.

    Training: resampled infinite stream, shard shuffle, buffer shuffle.
    Validation: finite single-pass, deterministic order, no shuffle.

    Args:
        shard_urls: list of shard tar paths, or a braceexpand pattern string
        config: WindowConfig with sampling parameters (uses defaults if None)
        lowdim_slices: dict mapping field names to (start, end) pairs
        preprocess_fn: optional callable(sample_dict) -> sample_dict
        shuffle_buffer: sample-level shuffle buffer size (train only)
        mode: 'train' or 'val'
        use_sliding_window: whether to compose sliding windows (VLA=True, VLM=False)
        lowdim_only: if True, only decode lowdim.npy and meta.json (skip image/depth)
    """
    if config is None:
        config = WindowConfig()
    if lowdim_slices is None:
        lowdim_slices = LOWDIM_SLICES

    if isinstance(shard_urls, str):
        shard_urls = sorted(glob.glob(shard_urls))
    assert shard_urls, f"No shards found: {shard_urls}"

    is_train = (mode == 'train')
    # resampled mode shuffles shards internally, but newer webdataset
    # versions still require an explicit shardshuffle value.
    #
    # NOTE (webdataset==1.0.2):
    # - resampled=True enters ResampledShards in shardlists.py
    # - default deterministic=False, and its seed mixes worker_seed/epoch
    #   with pid/time_ns/os.urandom, so shard sampling is time-dependent
    # - this is not fully controlled by torch/manual seed alone
    pipeline = wds.WebDataset(
        shard_urls,
        shardshuffle=False,
        nodesplitter=wds.split_by_node,
        resampled=is_train,
        empty_check=False,
    )
    if lowdim_only:
        # Only decode lowdim.npy and meta.json, drop everything else
        pipeline = pipeline.map(decode_lowdim_only)
    else:
        pipeline = pipeline.decode("pil").map(decode_meta)

    if use_sliding_window:
        if not lowdim_only:
            pipeline = pipeline.map(decode_lowdim)
        pipeline = pipeline.compose(
            lambda src: sliding_window_compose(src, config, lowdim_slices, lowdim_only)
        )

    # Shuffle before media materialization so the buffer retains lightweight
    # window descriptors with shared frame refs rather than copied image arrays.
    # NOTE (webdataset==1.0.2): shuffle() without seed uses
    # random.Random(int((pid + time) * 1e9)), which is also time-dependent.
    if is_train and shuffle_buffer and shuffle_buffer > 0:
        pipeline = pipeline.shuffle(shuffle_buffer)

    if use_sliding_window and not lowdim_only:
        pipeline = pipeline.map(materialize_sample_media)

    if preprocess_fn is not None:
        pipeline = pipeline.map(preprocess_fn)

    return pipeline


def build_blended_dataset(datasets_config, config=None, lowdim_slices=None,
                          preprocess_fn=None, shuffle_buffer=8192, mode='train',
                          use_sliding_window=True, lowdim_only=False):
    """Build a blended dataset from multiple WebDataset sources.

    Training: weighted random mixing across sources.
    Validation: sequential concatenation for single-pass evaluation.

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
            urls, config, lowdim_slices, preprocess_fn, shuffle_buffer, mode=mode,
            use_sliding_window=use_sliding_window, lowdim_only=lowdim_only)
        subsets.append(pipe)
        weights.append(c.get("weight", 1.0))

    assert subsets, "No shards found across all datasets."

    if len(subsets) == 1:
        return subsets[0]

    if mode == 'train':
        return wds.RandomMix(subsets, weights, longest=False)
    else:
        def chain_pipelines():
            for pipe in subsets:
                yield from pipe
        return chain_pipelines()
