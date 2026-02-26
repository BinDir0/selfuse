"""
WebDataset-based LegendVLA dataset.

Drop-in replacement for LegendVLADataset that reads from WebDataset shards
instead of Zarr files. Reuses existing preprocess logic (process_state_action,
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
from .wds_dataset import build_wds_pipeline, build_blended_dataset


class LegendVLAWdsDataset(torch.utils.data.IterableDataset):
    """WebDataset-backed VLA dataset for LegendVLA training.

    This is an IterableDataset that streams data from WebDataset shards.
    It reuses the same preprocess logic as LegendVLADataset so that the
    training output dict is identical.

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

        self.preprocessor = None
        self.normalizer = None

        # Sampling config from shape_meta
        self.action_horizon = shape_meta["action"]["horizon"]
        self.action_stride = shape_meta["action"]["stride"]
        self.state_horizon = shape_meta["obs"]["state"]["horizon"]
        self.state_stride = shape_meta["obs"]["state"]["stride"]
        self.image_horizon = shape_meta["obs"]["rgb"]["horizon"]
        self.image_stride = shape_meta["obs"]["rgb"]["stride"]

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

        Reuses the same logic as LegendVLADataset.sample_to_data().
        The input sample dict has the same keys as SequenceSampler output.
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
        if self.objective != "train_flow":
            data["labels"] = processed_results["labels"]
        return data

    def build_pipeline(self):
        """Build the WebDataset pipeline."""
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

        if len(datasets_config) == 1:
            pipeline = build_wds_pipeline(
                shard_urls=datasets_config[0]["shard_urls"],
                action_horizon=self.action_horizon,
                state_horizon=self.state_horizon,
                state_stride=self.state_stride,
                image_horizon=self.image_horizon,
                image_stride=self.image_stride,
                preprocess_fn=preprocess_fn,
                shuffle_buffer=self.shuffle_buffer,
            )
        else:
            pipeline = build_blended_dataset(
                datasets_config=datasets_config,
                action_horizon=self.action_horizon,
                state_horizon=self.state_horizon,
                state_stride=self.state_stride,
                image_horizon=self.image_horizon,
                image_stride=self.image_stride,
                preprocess_fn=preprocess_fn,
                shuffle_buffer=self.shuffle_buffer,
            )

        # Filter out None samples from failed preprocessing
        pipeline = pipeline.compose(filter_none)
        return pipeline

    def __iter__(self):
        assert self.preprocessor is not None, "Preprocessor not set"
        pipeline = self.build_pipeline()
        return iter(pipeline)

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
    """Unified dataset combining WebDataset VLA with map-style VLM dataset.

    For WebDataset VLA + existing VLM dataset integration.
    VLM samples are interleaved at a fixed ratio.
    """

    def __init__(
        self,
        vla_dataset: LegendVLAWdsDataset,
        vlm_dataset=None,
        vla_ratio: float = 5 / 6,
        batch_size: int = 20,
    ):
        super().__init__()
        self.vla_dataset = vla_dataset
        self.vlm_dataset = vlm_dataset
        self.vla_ratio = vla_ratio
        self.batch_size = batch_size

    def get_collator(self):
        return self.vla_dataset.get_collator()

    def __iter__(self):
        """Yield samples, interleaving VLA and VLM at the configured ratio."""
        vla_iter = iter(self.vla_dataset)

        if self.vlm_dataset is None or len(self.vlm_dataset) == 0:
            yield from vla_iter
            return

        vla_per_batch = int(self.vla_ratio * self.batch_size)
        vlm_per_batch = self.batch_size - vla_per_batch
        vlm_len = len(self.vlm_dataset)
        vlm_idx = 0
        shape_meta = None

        count = 0
        for vla_sample in vla_iter:
            yield vla_sample
            count += 1

            # After every vla_per_batch VLA samples, yield vlm_per_batch VLM samples
            if count % vla_per_batch == 0:
                # Lazily capture shape_meta from first VLA sample
                if shape_meta is None:
                    shape_meta = {}
                    for key in vla_sample:
                        if hasattr(vla_sample[key], "shape"):
                            shape_meta[key] = vla_sample[key].shape

                for _ in range(vlm_per_batch):
                    vlm_sample = self.vlm_dataset[vlm_idx % vlm_len]
                    vlm_idx += 1
                    # Pad missing VLA fields
                    if "states" in shape_meta:
                        vlm_sample["states"] = torch.zeros(*shape_meta["states"])
                    if "actions" in shape_meta:
                        vlm_sample["actions"] = torch.zeros(*shape_meta["actions"])
                        vlm_sample["actions_valid_mask"] = torch.zeros(
                            *shape_meta["actions"])
                    if "n_states" in shape_meta:
                        vlm_sample["n_states"] = torch.tensor(0, dtype=torch.int32)
                    if "n_actions" in shape_meta:
                        vlm_sample["n_actions"] = torch.tensor(0, dtype=torch.int32)
                    yield vlm_sample
