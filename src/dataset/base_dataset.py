from typing import Dict, List, Any
import os
import copy
import warnings
import pathlib

import numpy as np
import torch

from src.model.common.normalizer import LinearNormalizer
from src.utils.pytorch_util import dict_apply


class BaseDataCollator:
    """
    A generic data collator that can handle most cases and provide extension points for special cases.

    This collator will iterate over all keys in the first sample and call the `collate_key` method for each key.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collates a list of features (a list of sample dictionaries) into a batch.

        Args:
            features: A list, where each element is a dictionary returned by the dataset's __getitem__ method.

        Returns:
            A dictionary with values that have been batched.
        """
        batch = {}
        for key in features[0].keys():
            batch[key] = torch.stack([item[key] for item in features])

        return batch

class BaseZarrDataset(torch.utils.data.Dataset):
    """
    Base class for zarr-backed datasets. Encapsulates common zarr loading,
    sampling, and validation dataset creation logic.

    Subclasses must implement:
      - build_sampler_cfg()   -> sampler config dict
      - build_key_mapping(zarr_path) -> key mapping for this zarr
      - sample_to_data(sample) -> convert sampled result to model input
      - get_collator()        -> return collator
    """
    def __init__(
        self,
        zarr_paths,
        shape_meta,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        return_dataset_info=False
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.motion_type = shape_meta['obs']['state']['type']
        self.hand_ndim = shape_meta['obs']['state']['hand']['shape'][-1] // 2 # per hand dims
        self.zarr_paths = zarr_paths
        self.return_dataset_info = return_dataset_info

        # Subclass-provided sampler config
        self.sampler_cfg = self.build_sampler_cfg()

        # Common storage
        self.replay_buffers = []
        self.train_masks = []
        self.samplers = []
        self.sampler_lens = []
        self.dataset_names = []
        self._weights_list = []

        from .sampler import SequenceSampler, get_val_mask, downsample_mask
        from .streaming_replay_buffer import StreamingReplayBuffer

        for zarr_path in zarr_paths:
            key_mapping = self.build_key_mapping(zarr_path)
            dataset_path = zarr_path['path']
            if not os.path.exists(dataset_path):
                print(f"Warning: Dataset path {dataset_path} does not exist, skipping.")
                continue

            replay_buffer = StreamingReplayBuffer.copy_from_path(
                dataset_path, key_mapping=key_mapping, lazy_load=self.lazy_load()
            )
            if len(replay_buffer) <= 1:
                print(f"Warning: Dataset has only {len(replay_buffer)} episodes, skipping.")
                continue

            try:
                val_mask = get_val_mask(n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
                train_mask = ~val_mask
                train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)
                sampler = SequenceSampler(
                    replay_buffer=replay_buffer, episode_mask=train_mask, **self.sampler_cfg
                )
            except Exception as e:
                print(f"Error creating sampler: {e}")
                continue

            self.replay_buffers.append(replay_buffer)
            self.train_masks.append(train_mask)
            self.samplers.append(sampler)
            self.sampler_lens.append(len(sampler))

            weight = zarr_path.get('weight', None)
            if weight is not None:
                self._weights_list.append(weight)
            dataset_name = zarr_path.get('name', None)
            if dataset_name is None:
                dataset_name = pathlib.Path(str(zarr_path['path'])).stem
            self.dataset_names.append(dataset_name)

        # Initialize weights
        if self._weights_list:
            weights_sum = sum(self._weights_list)
            self.weights = [w / weights_sum for w in self._weights_list]
            self.dataset_lengths = self.sampler_lens
        else:
            self.weights = None
            self.dataset_lengths = None

    # ---------- Subclass must implement ---------- #
    def build_sampler_cfg(self) -> dict:
        raise NotImplementedError

    def build_key_mapping(self, zarr_path: dict) -> dict:
        raise NotImplementedError

    def sample_to_data(self, sample):
        raise NotImplementedError

    def get_collator(self):
        raise NotImplementedError

    # ---------- Subclass may override ---------- #
    def lazy_load(self) -> bool:
        """Whether to lazy-load zarr data. Default True."""
        return True

    def on_validation_copy(self, val_set):
        """Hook for extra processing in get_validation_dataset. Default no-op."""
        pass

    # ---------- Common implementation ---------- #
    def set_normalizer(self, normalizer: LinearNormalizer):
        """
        Set the normalizer for state/action.

        Args:
            normalizer (LinearNormalizer): Normalizer instance.
        """
        self.normalizer = normalizer
        
    def get_validation_dataset(self):
        from .sampler import SequenceSampler
        val_set = copy.copy(self)
        val_set.samplers = []
        val_set.train_masks = []
        val_set.sampler_lens = []
        self.on_validation_copy(val_set)

        for i, replay_buffer in enumerate(self.replay_buffers):
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                episode_mask=~self.train_masks[i],
                **self.sampler_cfg
            )
            val_set.samplers.append(sampler)
            val_set.train_masks.append(~self.train_masks[i])
            val_set.sampler_lens.append(len(sampler))
        return val_set

    def __getitem__(self, idx):
        try: 
            # Find corresponding sampler
            curr_idx, dataset_idx = idx, 0
            while curr_idx >= self.sampler_lens[dataset_idx]:
                curr_idx -= self.sampler_lens[dataset_idx]
                dataset_idx += 1
            sample = self.samplers[dataset_idx].sample_sequence(curr_idx)
            
            data = self.sample_to_data(sample)
            if self.return_dataset_info:
                data['dataset_name'] = self.dataset_names[dataset_idx]
                data['dataset_local_idx'] = np.array(curr_idx, dtype=np.int32)
            torch_data = dict_apply(
                data, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
            )
            return torch_data
        except Exception as e:
            warnings.warn(f"Error getting item {idx} from dataset {self.dataset_names[dataset_idx]}: {e}")
            # backup solution: return the next item
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self):
        return sum(self.sampler_lens)

