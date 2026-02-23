from typing import Dict, List, Any
import os
import copy
import warnings

import numpy as np
import torch

from src.model.common.normalizer import LinearNormalizer
from src.utils.pytorch_util import dict_apply

class BaseLowdimDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseLowdimDataset':
        # return an empty dataset by default
        return BaseLowdimDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: T, Do
            action: T, Da
        """
        raise NotImplementedError()


class BaseRatioDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        weights: List[float] = None, 
        dataset_lengths: List[int] = None, 
    ):
        self.weights = weights
        if weights is not None:
            weights_sum = sum(weights)
            assert weights_sum > 0, "Weights must be non-zero"
            self.weights = [weight / weights_sum for weight in weights]
        self.dataset_lengths = dataset_lengths

    def get_validation_dataset(self) -> 'BaseRatioDataset':
        # return an empty dataset by default
        return BaseRatioDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key: T, *
            action: T, Da
        """
        raise NotImplementedError()


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

class BaseLegendZarrDataset(torch.utils.data.Dataset):
    """
    Zarr 数据集基类，封装通用的 zarr 加载、采样、验证集创建逻辑。
    子类只需实现：
      - _build_sampler_cfg()  → 返回 sampler 配置 dict
      - _build_key_mapping(zarr_path) → 返回该 zarr 的 key mapping
      - _sample_to_data(sample) → 将采样结果转为模型输入
      - get_collator() → 返回 collator
    """

    def __init__(
        self,
        zarr_paths,
        shape_meta,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.motion_type = shape_meta['obs']['state']['type']
        self.hand_ndim = shape_meta['obs']['state']['hand']['shape'][-1] // 2

        # 子类实现
        self.sampler_cfg = self._build_sampler_cfg()

        # 通用存储
        self.replay_buffers = []
        self.train_masks = []
        self.samplers = []
        self.sampler_lens = []

        from .sampler import SequenceSampler, get_val_mask, downsample_mask
        from .streaming_replay_buffer import StreamingReplayBuffer

        for zarr_path in zarr_paths:
            key_mapping = self._build_key_mapping(zarr_path)
            dataset_path = zarr_path['path']
            if not os.path.exists(dataset_path):
                print(f"Warning: Dataset path {dataset_path} does not exist, skipping.")
                continue

            replay_buffer = StreamingReplayBuffer.copy_from_path(
                dataset_path, key_mapping=key_mapping, lazy_load=self._lazy_load()
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

            # 子类钩子：处理额外的 per-zarr 数据（如 weights, dataset_names）
            self._on_zarr_loaded(zarr_path, replay_buffer)

    # ---------- 子类必须实现 ---------- #
    def _build_sampler_cfg(self) -> dict:
        raise NotImplementedError

    def _build_key_mapping(self, zarr_path: dict) -> dict:
        raise NotImplementedError

    def _sample_to_data(self, sample):
        raise NotImplementedError

    def get_collator(self):
        raise NotImplementedError

    # ---------- 子类可选覆盖 ---------- #
    def _lazy_load(self) -> bool:
        """是否懒加载 zarr 数据。默认 True。"""
        return True

    def _on_zarr_loaded(self, zarr_path: dict, replay_buffer):
        """每个 zarr 加载完成后的钩子。默认空操作。"""
        pass

    def _on_validation_copy(self, val_set):
        """get_validation_dataset 中对 val_set 的额外处理。默认空操作。"""
        pass

    # ---------- 通用实现 ---------- #
    def get_validation_dataset(self):
        from .sampler import SequenceSampler
        val_set = copy.copy(self)
        val_set.samplers = []
        val_set.train_masks = []
        val_set.sampler_lens = []
        self._on_validation_copy(val_set)

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
            curr_idx, dataset_idx = idx, 0
            while curr_idx >= self.sampler_lens[dataset_idx]:
                curr_idx -= self.sampler_lens[dataset_idx]
                dataset_idx += 1
            sample = self.samplers[dataset_idx].sample_sequence(curr_idx)
            data = self._sample_to_data(sample)
            torch_data = dict_apply(
                data, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
            )
            return torch_data
        except Exception as e:
            warnings.warn(f"Error getting item {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self):
        return sum(self.sampler_lens)

