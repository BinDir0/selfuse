"""
WebDataset-based VLA datasets for LegendVLA training and normalizer fitting.
"""

import math
import warnings
import time
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
from torchvision import transforms

from src.model.common.normalizer import LinearNormalizer
from src.utils.pytorch_util import dict_apply
from .data_transforms import process_state_action, process_image
from .sanity_checks import NonFiniteDataError, build_sample_context, ensure_mapping_finite
from .collator import ConcatDataCollator
from .wds_dataset import (
    build_blended_dataset, build_wds_pipeline, WindowConfig, LOWDIM_SLICES,
    expand_shard_patterns,
)


class VLAWdsDataset(torch.utils.data.IterableDataset):
    """WebDataset-backed VLA dataset for LegendVLA training.

    This is an IterableDataset that streams data from WebDataset shards.

    Usage:
        dataset = VLAWdsDataset(
            wds_datasets=[
                {"shard_urls": "/data/wds/taco/shard-*.tar", "weight": 1.0, "name": "taco"},
                ...
            ],
            shape_meta=shape_meta,
        )
        dataset.set_collator(collator)
        dataset.set_normalizer(normalizer)
        dataloader = DataLoader(dataset, batch_size=20, num_workers=20)
    """
    def __init__(
        self,
        wds_datasets: List[Dict],
        shape_meta: Dict,
        objective: Optional[str] = None,
        use_relative_action: bool = False,
        mode: str = "train",
        depth_clip_range=None,
        shuffle_buffer: int = 8192,
        history_pad_mode: str = "repeat",
        future_pad_mode: str = "repeat",
        lowdim_slices: Optional[Dict] = None,
        return_dataset_info: bool = False,
        val_wds_datasets: Optional[List[Dict]] = None,
        video_base_fps: float = 30.0,
        debug_capture_raw_sample: bool = False,
        debug_capture_processed_sample: bool = False,
        debug_profile_timing: bool = False,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.motion_type = shape_meta["obs"]["state"]["type"]
        self.hand_ndim = shape_meta["obs"]["state"]["hand"]["shape"][-1] // 2
        self.action_ndim = shape_meta["action"]["shape"][-1]
        self.depth_image_shape = shape_meta["obs"]["depth"]["shape"]
        self.objective = objective
        self.use_relative_action = use_relative_action
        self.mode = mode
        self.depth_clip_range = depth_clip_range
        self.shuffle_buffer = shuffle_buffer
        self.wds_datasets = wds_datasets
        self.val_wds_datasets = val_wds_datasets
        self.return_dataset_info = return_dataset_info
        self.video_base_fps = float(video_base_fps)
        self.debug_capture_raw_sample = bool(debug_capture_raw_sample)
        self.debug_capture_processed_sample = bool(debug_capture_processed_sample)
        self.debug_profile_timing = bool(debug_profile_timing)

        self.collator = None
        self.normalizer = None

        # Sampling config from shape_meta
        self.action_horizon = shape_meta["action"]["horizon"]
        self.state_horizon = shape_meta["obs"]["state"]["horizon"]
        self.image_horizon = shape_meta["obs"]["rgb"]["horizon"]

        self.window_config = WindowConfig(
            action_horizon=shape_meta["action"]["horizon"],
            action_stride=shape_meta["action"]["stride"],
            state_horizon=shape_meta["obs"]["state"]["horizon"],
            state_stride=shape_meta["obs"]["state"]["stride"],
            image_horizon=shape_meta["obs"]["rgb"]["horizon"],
            image_stride=shape_meta["obs"]["rgb"]["stride"],
            history_pad_mode=history_pad_mode,
            future_pad_mode=future_pad_mode,
        )
        self.lowdim_slices = lowdim_slices or LOWDIM_SLICES

        self.aug_transform = None
        if self.mode == "train":
            self.aug_transform = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.GaussianBlur(
                    kernel_size=(5, 5), sigma=(0.1, 2.0)),
            ])

    def set_collator(self, collator):
        """Set the batch collator used to build model inputs."""
        self.collator = collator

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set the normalizer for state/action."""
        self.normalizer = normalizer

    def build_raw_model_inputs(self, instruction, image, intrinsic):
        return {
            "images": image,
            "instruction": instruction,
            "intrinsic": intrinsic,
            "vision_type": "video",
            "video_fps": np.array(
                self.video_base_fps / self.window_config.image_stride,
                dtype=np.float32,
            ),
            "has_depth_values": np.array(False, dtype=bool),
        }

    def copy_debug_value(self, value):
        """Create a detached debug copy of one sample field."""
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, dict):
            return {key: self.copy_debug_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.copy_debug_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.copy_debug_value(item) for item in value)
        return value

    def build_debug_raw_sample(self, sample):
        """Keep the dataset sample before state/image processing for offline inspection."""
        debug_sample = {
            "valid_action_len": np.array(sample["valid_action_len"], dtype=np.int32),
            "wrist_state": sample["wrist_state"].astype(np.float32),
            "hand_state": sample["hand_state"].astype(np.float32),
            "wrist_action": sample["wrist_action"].astype(np.float32),
            "hand_action": sample["hand_action"].astype(np.float32),
            "extrinsic": sample["extrinsic"].astype(np.float32).reshape(4, 4),
            "intrinsic": sample["intrinsic"].astype(np.float32),
            "instruction": self.copy_debug_value(sample["instruction"]),
            "instruction_num": np.array(sample["instruction_num"], dtype=np.int32),
            "presence": np.array(sample.get("presence", 3), dtype=np.int32),
            "image": sample["image"].copy(),
        }
        if "dataset_name" in sample:
            debug_sample["dataset_name"] = self.copy_debug_value(sample["dataset_name"])
        if "episode_index" in sample:
            debug_sample["episode_index"] = np.array(sample["episode_index"], dtype=np.int32)
        return debug_sample

    def build_debug_processed_sample(self, data):
        """Keep the exact collator input sample for offline inspection."""
        return {
            key: self.copy_debug_value(value)
            for key, value in data.items()
            if not key.startswith("debug_")
        }

    def sample_to_data(self, sample):
        """Convert one WebDataset sample into the raw VLA sample schema used by the project.

        This is the dataset-side producer contract for `UnifiedVLACollator`.
        The returned mapping contains visual history, instruction text, intrinsic parameters, padded state/action
        tensors, action-valid masks, and bookkeeping fields such as `n_states`, `n_actions`, and `is_vla_data`.
        """
        sample_context = build_sample_context(sample)
        state, action = process_state_action(
            wrist_state=sample["wrist_state"].astype(np.float32),
            hand_state=sample["hand_state"].astype(np.float32),
            wrist_action=sample["wrist_action"].astype(np.float32),
            hand_action=sample["hand_action"].astype(np.float32),
            extrinsic=sample["extrinsic"].astype(np.float32).reshape(4, 4),
            normalizer=self.normalizer,
            hand_ndim=self.hand_ndim,
            motion_type=self.motion_type,
            use_relative_action=self.use_relative_action,
        )
        ensure_mapping_finite(
            {"state": state, "action": action},
            stage="vla_after_state_action",
            context=sample_context,
        )

        intrinsic = sample["intrinsic"].astype(np.float32)
        image, depth_images, intrinsic = process_image(
            sample["image"],
            sample.get("depth", None),
            intrinsic,
            self.aug_transform,
            self.depth_clip_range,
        )
        ensure_mapping_finite(
            {"image": image, "depth_images": depth_images},
            stage="vla_after_image_process",
            context=sample_context,
        )
        instruction = sample["instruction"]
        instruction_num = sample["instruction_num"]

        # Sample a random instruction from candidates
        if self.mode == "train":
            idx = np.random.randint(0, instruction_num)
        else:
            idx = 0
        # instruction may be a single string or list
        if isinstance(instruction, list):
            instruction = instruction[idx]

        state_pad = np.zeros(
            (self.state_horizon, *state.shape[1:]), dtype=np.float32)
        state_pad[:state.shape[0]] = state
        action_pad = np.zeros(
            (self.action_horizon, *action.shape[1:]), dtype=np.float32)
        actions_valid_mask = np.zeros(
            (self.action_horizon, *action.shape[1:]), dtype=bool)
        action_pad[:action.shape[0]] = action
        actions_valid_mask[:action.shape[0]] = True

        data = self.build_raw_model_inputs(
            instruction=instruction,
            image=image,
            intrinsic=intrinsic,
        )

        data.update({
            "states": state_pad,
            "n_states": np.array(state.shape[0], dtype=np.int32),
            "actions": action_pad,
            "actions_valid_mask": actions_valid_mask,
            "n_actions": np.array(action.shape[0], dtype=np.int32),
            "is_vla_data": np.array(True, dtype=bool),
        })
        if self.return_dataset_info:
            data["dataset_name"] = sample["dataset_name"]
            data["episode_index"] = sample["episode_index"]
        ensure_mapping_finite(
            data,
            stage="vla_dataset_output",
            context=sample_context,
        )

        if self.debug_capture_raw_sample:
            data["debug_raw_sample"] = self.build_debug_raw_sample(sample)
        if self.debug_capture_processed_sample:
            data["debug_processed_sample"] = self.build_debug_processed_sample(data)
        return data

    def build_pipeline(self):
        """Build the WebDataset pipeline.

        Training: resampled infinite stream with shuffle.
        Validation: finite single-pass, no shuffle.
        """
        datasets_config = []
        for ds in self.wds_datasets:
            datasets_config.append({
                "shard_urls": ds["shard_urls"],
                "weight": ds.get("weight", 1.0),
                "name": ds.get("name", "unknown"),
            })

        def preprocess_fn(sample):
            worker_info = torch.utils.data.get_worker_info()
            worker_id = -1 if worker_info is None else int(worker_info.id)
            preprocess_start = time.perf_counter()
            try:
                sample_to_data_start = time.perf_counter()
                data = self.sample_to_data(sample)
                sample_to_data_s = time.perf_counter() - sample_to_data_start
                torch_data = dict_apply(
                    data,
                    lambda x: torch.from_numpy(x)
                    if isinstance(x, np.ndarray) else x,
                )
                preprocess_total_s = time.perf_counter() - preprocess_start
                if self.debug_profile_timing:
                    torch_data["debug_sample_profile"] = {
                        "worker_id": worker_id,
                        "sample_to_data_s": sample_to_data_s,
                        "preprocess_total_s": preprocess_total_s,
                    }
                return torch_data
            except NonFiniteDataError:
                raise
            except Exception as e:
                warnings.warn(f"Error in preprocess: {e}")
                return None

        def filter_none(src):
            for sample in src:
                if sample is not None:
                    # wds .map() auto-injects __key__; strip it before collation
                    sample.pop("__key__", None)
                    yield sample

        pipeline = build_blended_dataset(
            datasets_config=datasets_config,
            config=self.window_config,
            lowdim_slices=self.lowdim_slices,
            preprocess_fn=preprocess_fn,
            shuffle_buffer=self.shuffle_buffer,
            mode=self.mode,
        )
        return filter_none(pipeline)

    def __iter__(self):
        pipeline = self.build_pipeline()
        return iter(pipeline)

    def get_validation_dataset(self):
        """Create a validation dataset from separate val shard URLs.
        """
        assert self.val_wds_datasets is not None, "val_wds_datasets is not set"
        val_dataset = VLAWdsDataset(
            wds_datasets=self.val_wds_datasets,
            shape_meta=self.shape_meta,
            objective=self.objective,
            use_relative_action=self.use_relative_action,
            mode="val",
            depth_clip_range=self.depth_clip_range,
            shuffle_buffer=0,
            history_pad_mode=self.window_config.history_pad_mode,
            future_pad_mode=self.window_config.future_pad_mode,
            lowdim_slices=self.lowdim_slices,
            return_dataset_info=self.return_dataset_info,
            debug_capture_raw_sample=self.debug_capture_raw_sample,
            debug_capture_processed_sample=self.debug_capture_processed_sample,
            debug_profile_timing=self.debug_profile_timing,
        )
        if self.collator is not None:
            val_dataset.set_collator(self.collator)
        if self.normalizer is not None:
            val_dataset.set_normalizer(self.normalizer)
        return val_dataset

    def get_collator(self):
        """Build a data collator for batching."""
        assert self.collator is not None, "Collator not set"
        return self.collator.for_mode(self.mode)


class UnifiedWdsDataset(torch.utils.data.IterableDataset):
    """Unified dataset combining WebDataset VLA with streaming VLM dataset.

    Training mode: VLM samples interleaved at a fixed ratio, VLM auto-restarts.
    Validation mode: all VLA samples first, then all VLM samples sequentially.
    """

    def __init__(
        self,
        vla_dataset: VLAWdsDataset,
        vlm_dataset=None,
        vla_ratio: float = 5 / 6,
        batch_size: int = 20,
        mode: str = "train",
    ):
        super().__init__()
        self.vla_dataset = vla_dataset
        self.vlm_dataset = vlm_dataset
        self.vla_ratio = vla_ratio
        self.batch_size = batch_size
        self.mode = mode
        assert vla_ratio > 0 and vla_ratio <= 1, "vla_ratio must be in (0, 1]"

        self.build_vla_shape_meta()

    def build_vla_shape_meta(self):
        """Build the shape meta for the VLA dataset."""
        chunk_config = self.vla_dataset.window_config
        action_ndim = self.vla_dataset.action_ndim
        self.shape_meta = {
            "states": (chunk_config.state_horizon, action_ndim),
            "actions": (chunk_config.action_horizon, action_ndim),
            "n_states": 1,
            "n_actions": 1,
            # (T, 3, H, W)
            "depth_values": (chunk_config.image_horizon, 3, *self.vla_dataset.depth_image_shape),
            "has_depth_values": 1,
        }

    def distribute(self, rank: int, world_size: int):
        """Apply distributed shard splitting to sub-datasets.

        VLA uses wds.split_by_node internally, so only VLM needs explicit splitting.
        """
        if self.vlm_dataset is not None and hasattr(self.vlm_dataset, 'distribute'):
            self.vlm_dataset.distribute(rank, world_size)

    def get_collator(self):
        return self.vla_dataset.get_collator()

    def get_validation_dataset(self):
        """Create a unified validation dataset.
        """
        vla_val = self.vla_dataset.get_validation_dataset()

        vlm_val = None
        has_vlm_val = getattr(self.vlm_dataset, "val_wds_datasets", None) is not None
        if (
            self.vlm_dataset is not None
            and has_vlm_val
            and hasattr(self.vlm_dataset, 'get_validation_dataset')
        ):
            vlm_val = self.vlm_dataset.get_validation_dataset()

        return UnifiedWdsDataset(
            vla_dataset=vla_val,
            vlm_dataset=vlm_val,
            mode="val",
        )

    def __iter__(self):
        if self.mode == 'train':
            yield from self.iter_train()
        else:
            yield from self.iter_val()

    def iter_train(self):
        """Interleave VLA and VLM at the configured ratio."""
        vla_iter = iter(self.vla_dataset)

        if self.vlm_dataset is None:
            yield from vla_iter
            return

        vlm_iter = iter(self.vlm_dataset)
        vla_per_batch = math.ceil(self.vla_ratio * self.batch_size)
        vlm_per_batch = self.batch_size - vla_per_batch

        count = 0
        for vla_sample in vla_iter:
            yield vla_sample
            count += 1

            if count % vla_per_batch == 0:
                for _ in range(vlm_per_batch):
                    try:
                        vlm_sample = next(vlm_iter)
                    except StopIteration:
                        vlm_iter = iter(self.vlm_dataset)
                        vlm_sample = next(vlm_iter)
                    self.pad_vlm_sample(vlm_sample)
                    yield vlm_sample

    def iter_val(self):
        """Sequential single-pass: all VLA samples, then all VLM samples."""
        for vla_sample in self.vla_dataset:
            yield vla_sample

        if self.vlm_dataset is None:
            return

        for vlm_sample in self.vlm_dataset:
            self.pad_vlm_sample(vlm_sample)
            yield vlm_sample

    def pad_vlm_sample(self, vlm_sample):
        """Pad missing VLA fields on a VLM sample so the collator sees uniform keys."""
        shape_meta = self.shape_meta
        vlm_sample["states"] = torch.zeros(*shape_meta["states"])
        vlm_sample["actions"] = torch.zeros(*shape_meta["actions"])
        vlm_sample["actions_valid_mask"] = torch.zeros(*shape_meta["actions"], dtype=torch.bool)
        vlm_sample["n_states"] = torch.tensor(0, dtype=torch.int32)
        vlm_sample["n_actions"] = torch.tensor(0, dtype=torch.int32)
        vlm_sample["depth_values"] = torch.zeros(*shape_meta["depth_values"])
        vlm_sample["has_depth_values"] = torch.tensor(False, dtype=torch.bool)
        if self.vla_dataset.debug_capture_raw_sample:
            vlm_sample["debug_raw_sample"] = None
        if self.vla_dataset.debug_capture_processed_sample:
            vlm_sample["debug_processed_sample"] = None
        if self.vla_dataset.debug_profile_timing:
            vlm_sample["debug_sample_profile"] = None


class VLALowLevelWdsDataset(torch.utils.data.IterableDataset):
    """Low-level WebDataset for normalizer fitting (state/action only, no images).

    This dataset is dedicated to normalizer computation. In practice it should
    run with ``mode="val"`` so the scan is a finite single pass over shards,
    without the training-time ``resampled=True`` behavior.

    Sampling notes:
    - ``max_total_shards`` caps the total number of scanned shards. This keeps
      the approximation logic at shard level instead of introducing frame-level
      early stop semantics inside the dataloader.
    - ``min_shards_per_dataset`` is a per-dataset coverage floor. After the
      floor is reserved, the remaining shard pool is sampled without replacement.
      Larger datasets naturally contribute more shards because they occupy more
      entries in that remaining pool.
    - The final shard list preserves dataset-grouped ordering after shard
      selection is finished. Since selected shards are scanned to completion,
      the normalizer statistics depend on coverage rather than on an additional
      mixing order.
    - The collator intentionally concatenates rows instead of stacking windows.
      History/action horizons can be shorter under ``truncate`` mode; concat
      preserves only valid rows and lets the streaming normalizer consume a flat
      matrix directly.
    """

    def __init__(
        self,
        wds_datasets: List[Dict],
        shape_meta: Dict,
        use_relative_action: bool = False,
        mode: str = "val",
        max_total_shards: Optional[int] = None,
        min_shards_per_dataset: int = 8,
        seed: int = 0,
        history_pad_mode: str = "repeat",
        future_pad_mode: str = "repeat",
        lowdim_slices: Optional[Dict] = None,
    ):
        super().__init__()
        self.wds_datasets = wds_datasets
        self.shape_meta = shape_meta
        self.motion_type = shape_meta["obs"]["state"]["type"]
        self.hand_ndim = shape_meta["obs"]["state"]["hand"]["shape"][-1] // 2
        self.use_relative_action = use_relative_action
        self.mode = mode
        self.max_total_shards = max_total_shards
        self.min_shards_per_dataset = min_shards_per_dataset
        self.seed = seed
        self.lowdim_slices = lowdim_slices or LOWDIM_SLICES

        if self.mode != "val":
            warnings.warn(
                "VLALowLevelWdsDataset is intended for normalizer fitting and usually "
                "should run with mode='val' for a finite single-pass scan."
            )
        if self.min_shards_per_dataset < 1:
            raise ValueError("min_shards_per_dataset must be >= 1")
        if self.max_total_shards is not None and self.max_total_shards < 1:
            raise ValueError("max_total_shards must be >= 1 when provided")

        self.window_config = WindowConfig(
            action_horizon=shape_meta["action"]["horizon"],
            action_stride=shape_meta["action"]["stride"],
            state_horizon=shape_meta["obs"]["state"]["horizon"],
            state_stride=shape_meta["obs"]["state"]["stride"],
            image_horizon=shape_meta["obs"]["rgb"]["horizon"],
            image_stride=shape_meta["obs"]["rgb"]["stride"],
            history_pad_mode=history_pad_mode,
            future_pad_mode=future_pad_mode,
        )

    def sample_to_data(self, sample):
        """Extract lowdim fields and compute state/action."""
        state, action = process_state_action(
            wrist_state=sample["wrist_state"].astype(np.float32),
            hand_state=sample["hand_state"].astype(np.float32),
            wrist_action=sample["wrist_action"].astype(np.float32),
            hand_action=sample["hand_action"].astype(np.float32),
            extrinsic=sample["extrinsic"].astype(np.float32).reshape(4, 4),
            normalizer=None,
            hand_ndim=self.hand_ndim,
            motion_type=self.motion_type,
            use_relative_action=self.use_relative_action,
        )

        if not self.use_relative_action:
            return {
                "motions": np.concatenate([state, action], axis=0),
            }
        return {
            "states": state,
            "actions": action,
        }

    def build_shard_groups(self):
        """Expand shard globs and shuffle each dataset independently."""
        shard_groups = []
        for dataset_index, dataset_cfg in enumerate(self.wds_datasets):
            shard_patterns = dataset_cfg["shard_urls"]
            shard_urls, shard_patterns_metadata = expand_shard_patterns(shard_patterns)
            if not shard_urls:
                warnings.warn(
                    f"No shards found for {dataset_cfg.get('name', '?')}: "
                    f"{shard_patterns_metadata}, skipping."
                )
                continue

            rng = np.random.default_rng(self.seed + dataset_index)
            order = rng.permutation(len(shard_urls)).tolist()
            shuffled_urls = [shard_urls[idx] for idx in order]
            shard_groups.append({
                "dataset_index": dataset_index,
                "name": dataset_cfg.get("name", f"dataset_{dataset_index}"),
                "shard_patterns": shard_patterns_metadata,
                "shard_urls": shuffled_urls,
            })

        if not shard_groups:
            raise ValueError("No shards found across all datasets.")
        return shard_groups

    def select_shards(self):
        """Select final shard URLs with a coverage floor and random remainder sampling."""
        shard_groups = self.build_shard_groups()
        selected_shards = []
        remaining_shards = []
        minimum_selected = 0

        for group in shard_groups:
            base_count = min(len(group["shard_urls"]), self.min_shards_per_dataset)
            selected_shards.extend(
                (group["dataset_index"], shard_url)
                for shard_url in group["shard_urls"][:base_count]
            )
            remaining_shards.extend(
                (group["dataset_index"], shard_url)
                for shard_url in group["shard_urls"][base_count:]
            )
            minimum_selected += base_count

        if self.max_total_shards is not None and self.max_total_shards < minimum_selected:
            raise ValueError(
                "max_total_shards is smaller than the required minimum shard coverage"
            )

        if self.max_total_shards is None:
            extra_budget = len(remaining_shards)
        else:
            extra_budget = min(
                len(remaining_shards),
                self.max_total_shards - minimum_selected,
            )

        if extra_budget > 0:
            rng = np.random.default_rng(self.seed)
            chosen_indices = np.sort(rng.permutation(len(remaining_shards))[:extra_budget])
            selected_shards.extend(
                remaining_shards[int(pool_index)] for pool_index in chosen_indices
            )

        return shard_groups, selected_shards

    def build_shard_urls(self):
        """Build the final shard list used for normalizer fitting."""
        _, selected_shards = self.select_shards()
        return [shard_url for _, shard_url in selected_shards]

    def describe_shard_selection(self):
        """Return a JSON-serializable summary of shard coverage."""
        shard_groups, selected_shards = self.select_shards()
        selected_counts = Counter(dataset_index for dataset_index, _ in selected_shards)
        datasets = []
        for group in shard_groups:
            available_count = len(group["shard_urls"])
            selected_count = selected_counts.get(group["dataset_index"], 0)
            datasets.append({
                "name": group["name"],
                "shard_patterns": group["shard_patterns"],
                "available_shards": available_count,
                "selected_shards": selected_count,
                "full_coverage": selected_count == available_count,
                "selected_fraction": (
                    float(selected_count) / float(available_count)
                    if available_count > 0 else 0.0
                ),
            })

        available_total = sum(item["available_shards"] for item in datasets)
        selected_total = len(selected_shards)
        return {
            "mode": self.mode,
            "seed": self.seed,
            "max_total_shards": self.max_total_shards,
            "min_shards_per_dataset": self.min_shards_per_dataset,
            "available_shards_total": available_total,
            "selected_shards_total": selected_total,
            "full_dataset_coverage": selected_total == available_total,
            "datasets": datasets,
        }

    def build_pipeline(self):
        """Build a streaming pipeline for lowdim-only data."""
        shard_urls = self.build_shard_urls()

        def preprocess_fn(sample):
            try:
                data = self.sample_to_data(sample)
                return {
                    key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                    for key, value in data.items()
                }
            except Exception as exc:
                warnings.warn(f"Error in lowlevel preprocess: {exc}")
                return None

        def filter_none(src):
            for sample in src:
                if sample is not None:
                    sample.pop("__key__", None)
                    yield sample

        pipeline = build_wds_pipeline(
            shard_urls=shard_urls,
            config=self.window_config,
            lowdim_slices=self.lowdim_slices,
            preprocess_fn=preprocess_fn,
            shuffle_buffer=0,
            mode=self.mode,
            use_sliding_window=True,
            lowdim_only=True,
        )
        return filter_none(pipeline)

    def __iter__(self):
        pipeline = self.build_pipeline()
        return iter(pipeline)

    def get_collator(self):
        """Return ConcatDataCollator for normalizer fitting.

        Concatenation keeps only valid lowdim rows when truncate padding is used,
        and avoids stacking windows only to flatten them again for streaming stats.
        """
        return ConcatDataCollator()
