"""
WebDataset-based LegendVLA dataset.

Reuses existing preprocess logic (process_state_action,
process_image, PaliGemmaVLAProcessor) without modification.
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
from torchvision import transforms

from src.model.common.normalizer import LinearNormalizer
from src.utils.pytorch_util import dict_apply
from .data_transforms import process_state_action, process_image
from .collator import LegendVLDataCollator
from .wds_dataset import (
    build_blended_dataset, WindowConfig, LOWDIM_SLICES,
)


class LegendVLAWdsDataset(torch.utils.data.IterableDataset):
    """WebDataset-backed VLA dataset for LegendVLA training.

    This is an IterableDataset that streams data from WebDataset shards.

    Usage:
        dataset = LegendVLAWdsDataset(
            wds_datasets=[
                {"shard_urls": "/data/wds/taco/shard-*.tar", "weight": 1.0, "name": "taco"},
                ...
            ],
            shape_meta=shape_meta,
        )
        dataset.set_preprocessor(processor)
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
        lowdim_slices: Optional[Dict] = None,
        return_dataset_info: bool = False,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.motion_type = shape_meta["obs"]["state"]["type"]
        self.hand_ndim = shape_meta["obs"]["state"]["hand"]["shape"][-1] // 2
        self.objective = objective
        self.use_relative_action = use_relative_action
        self.mode = mode
        self.depth_clip_range = depth_clip_range
        self.shuffle_buffer = shuffle_buffer
        self.wds_datasets = wds_datasets
        self.return_dataset_info = return_dataset_info

        self.preprocessor = None
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

    def set_preprocessor(self, preprocessor):
        """Set the tokenizer/vision preprocessor."""
        self.preprocessor = preprocessor

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set the normalizer for state/action."""
        self.normalizer = normalizer

    def sample_to_data(self, sample):
        """Convert a WebDataset sample dict into model-ready tensors.
        """
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
        image, depth_images = process_image(
            sample["image"],
            sample.get("depth", None),
            self.aug_transform,
            self.depth_clip_range,
        )

        intrinsic = sample["intrinsic"].astype(np.float32)
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

        processed_results = self.preprocessor(
            text=instruction,
            images=image,
            states=state,
            actions=action,
            intrinsic=intrinsic,
            objective=self.objective,
            depth_images=depth_images,
            mode=self.mode,
        )

        state_pad = np.zeros(
            (self.state_horizon, *state.shape[1:]), dtype=np.float32)
        state_pad[:state.shape[0]] = state
        action_pad = np.zeros(
            (self.action_horizon, *action.shape[1:]), dtype=np.float32)
        actions_valid_mask = np.zeros(
            (self.action_horizon, *action.shape[1:]), dtype=bool)
        actions_valid_mask[:action.shape[0]] = True
        action_pad[:action.shape[0]] = action

        data = {
            "input_ids": processed_results["input_ids"],
            "answer_start_idx": processed_results["answer_start_idx"],
            "attention_mask": processed_results["attention_mask"],
            "pixel_values": processed_results["pixel_values"],
            "states": state_pad,
            "n_states": np.array(state.shape[0], dtype=np.int32),
            "actions": action_pad,
            "actions_valid_mask": actions_valid_mask,
            "n_actions": np.array(action.shape[0], dtype=np.int32),
            "is_vla_data": np.array(True, dtype=bool),
        }
        if "depth_values" in processed_results:
            data["depth_values"] = processed_results["depth_values"]
            data["has_depth_values"] = np.array(True, dtype=bool)
        else: 
            data["has_depth_values"] = np.array(False, dtype=bool)
        if self.objective != "train_flow":
            data["labels"] = processed_results["labels"]
        if self.return_dataset_info: 
            data["dataset_name"] = sample["dataset_name"]
            data["episode_index"] = sample["episode_index"]
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
            try:
                data = self.sample_to_data(sample)
                torch_data = dict_apply(
                    data,
                    lambda x: torch.from_numpy(x)
                    if isinstance(x, np.ndarray) else x,
                )
                return torch_data
            except Exception as e:
                warnings.warn(f"Error in preprocess: {e}")
                return None

        def filter_none(src):
            for sample in src:
                if sample is not None:
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
        assert self.preprocessor is not None, "Preprocessor not set"
        pipeline = self.build_pipeline()
        return iter(pipeline)

    def get_validation_dataset(self, val_wds_datasets: List[Dict]):
        """Create a validation dataset from separate val shard URLs.

        Args:
            val_wds_datasets: list of dicts with keys:
                - shard_urls: glob pattern or list of val shard tar paths
                - name: (optional) dataset name
        """
        val_dataset = LegendVLAWdsDataset(
            wds_datasets=val_wds_datasets,
            shape_meta=self.shape_meta,
            objective=self.objective,
            use_relative_action=self.use_relative_action,
            mode="val",
            depth_clip_range=self.depth_clip_range,
            shuffle_buffer=0,
            lowdim_slices=self.lowdim_slices,
            return_dataset_info=self.return_dataset_info,
        )
        if self.preprocessor is not None:
            val_dataset.set_preprocessor(self.preprocessor)
        if self.normalizer is not None:
            val_dataset.set_normalizer(self.normalizer)
        return val_dataset

    def get_collator(self):
        """Build a data collator for batching."""
        assert self.preprocessor is not None, "Preprocessor not set"
        padding_side = "left" if self.mode == "infer-ar" else "right"
        return LegendVLDataCollator(
            pad_token_id=self.preprocessor.tokenizer.pad_token_id,
            ignore_index=self.preprocessor.ignore_index,
            padding_side=padding_side,
        )


class LegendUnifiedWdsDataset(torch.utils.data.IterableDataset):
    """Unified dataset combining WebDataset VLA with streaming VLM dataset.

    Training mode: VLM samples interleaved at a fixed ratio, VLM auto-restarts.
    Validation mode: all VLA samples first, then all VLM samples sequentially.
    """

    def __init__(
        self,
        vla_dataset: LegendVLAWdsDataset,
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
        assert vla_ratio >= 0 and vla_ratio <= 1, "vla_ratio must be between 0 and 1"

    def distribute(self, rank: int, world_size: int):
        """Apply distributed shard splitting to sub-datasets.

        VLA uses wds.split_by_node internally, so only VLM needs explicit splitting.
        """
        if self.vlm_dataset is not None and hasattr(self.vlm_dataset, 'distribute'):
            self.vlm_dataset.distribute(rank, world_size)

    def get_collator(self):
        return self.vla_dataset.get_collator()

    def get_validation_dataset(self, val_wds_datasets: List[Dict]):
        """Create a unified validation dataset.

        Args:
            val_wds_datasets: VLA validation shard configs.
        """
        vla_val = self.vla_dataset.get_validation_dataset(val_wds_datasets)

        vlm_val = None
        if self.vlm_dataset is not None and hasattr(self.vlm_dataset, 'get_validation_dataset'):
            vlm_val = self.vlm_dataset.get_validation_dataset()

        return LegendUnifiedWdsDataset(
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
        vla_per_batch = int(self.vla_ratio * self.batch_size)
        vlm_per_batch = self.batch_size - vla_per_batch
        shape_meta = None

        count = 0
        for vla_sample in vla_iter:
            yield vla_sample
            count += 1

            if count % vla_per_batch == 0:
                if shape_meta is None:
                    shape_meta = {}
                    for key in vla_sample:
                        if hasattr(vla_sample[key], "shape"):
                            shape_meta[key] = vla_sample[key].shape

                for _ in range(vlm_per_batch):
                    try:
                        vlm_sample = next(vlm_iter)
                    except StopIteration:
                        vlm_iter = iter(self.vlm_dataset)
                        vlm_sample = next(vlm_iter)
                    self.pad_vlm_sample(vlm_sample, shape_meta)
                    yield vlm_sample

    def iter_val(self):
        """Sequential single-pass: all VLA samples, then all VLM samples."""
        shape_meta = None

        for vla_sample in self.vla_dataset:
            if shape_meta is None:
                shape_meta = {}
                for key in vla_sample:
                    if hasattr(vla_sample[key], "shape"):
                        shape_meta[key] = vla_sample[key].shape
            yield vla_sample

        if self.vlm_dataset is None:
            return

        for vlm_sample in self.vlm_dataset:
            if shape_meta is not None:
                self.pad_vlm_sample(vlm_sample, shape_meta)
            yield vlm_sample

    @staticmethod
    def pad_vlm_sample(vlm_sample, shape_meta):
        """Pad missing VLA fields on a VLM sample so the collator sees uniform keys."""
        if "states" in shape_meta:
            vlm_sample["states"] = torch.zeros(*shape_meta["states"])
        if "actions" in shape_meta:
            vlm_sample["actions"] = torch.zeros(*shape_meta["actions"])
            vlm_sample["actions_valid_mask"] = torch.zeros(*shape_meta["actions"])
        if "n_states" in shape_meta:
            vlm_sample["n_states"] = torch.tensor(0, dtype=torch.int32)
        if "n_actions" in shape_meta:
            vlm_sample["n_actions"] = torch.tensor(0, dtype=torch.int32)
