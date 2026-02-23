'''
Propise dataset for LegendVLA
Every action is the delta of the next predicted absolute state and the state at the beginning of the action chunk.
'''

import os
from typing import Dict, Optional
import pathlib
import torch
import numpy as np
import copy
from torchvision import transforms
import warnings
from src.utils.pytorch_util import dict_apply
from src.model.common.normalizer import LinearNormalizer
from .base_dataset import BaseRatioDataset, BaseLegendZarrDataset
from .sampler import SequenceSampler, get_val_mask, downsample_mask
from .streaming_replay_buffer import StreamingReplayBuffer
from .data_transforms import process_state_action, process_image
from .collator import LegendVLDataCollator, ConcatDataCollator
from .normalizer_utils import get_normalizer
# Backward-compatible import: LegendVLMDataset was moved to legendvlm_dataset.py
from .legendvlm_dataset import LegendVLMDataset  # noqa: F401


def build_default_key_mapping(motion_type: str) -> Dict[str, str]:
    """
    Build the default key mapping (renamed_key -> original_key) for LegendVLA datasets.
    """
    return {
        'image': 'image',
        'depth': 'depth',
        'wrist_state': 'state/wrist',
        'hand_state': f'state/{motion_type}',
        'wrist_action': 'action/wrist',
        'hand_action': f'action/{motion_type}',
        'extrinsic': 'extrinsic',
        'intrinsic': 'intrinsic',
        'instruction': 'instruction',
        'instruction_num': 'instruction_num',
    }


def merge_key_mapping(custom_mapping: Optional[Dict[str, str]], motion_type: str) -> Dict[str, str]:
    """
    Merge a custom mapping into the default mapping.
    """
    mapping = build_default_key_mapping(motion_type)
    if custom_mapping is not None:
        mapping.update(custom_mapping)
    return mapping


class LegendVLADataset(BaseLegendZarrDataset):
    """
    Dataset for LegendVLA training/validation with async IO support.

    The sampler returns Future-like objects; data is resolved in _sample_to_data.
    """
    def __init__(
        self,
        zarr_paths,
        shape_meta=None,
        seed=42,
        val_ratio=0.0,
        objective=None,
        normalizer_dataloader_cfg=dict(),
        use_relative_action=False,
        max_train_episodes=None,
        mode = 'train',
        depth_clip_range=None,
        return_dataset_info: bool = False,
    ):
        """
        Args:
            zarr_paths (List[Dict]): List of zarr dataset configs with 'path' and optional 'weight'.
            shape_meta (Dict): Metadata describing observation/action shapes.
            seed (int): Random seed for splitting/downsampling.
            val_ratio (float): Validation split ratio.
            objective (Optional[str]): Training objective name.
            normalizer_dataloader_cfg (Dict): DataLoader config for normalizer fit.
            use_relative_action (bool): Use relative action representation if True.
            max_train_episodes (Optional[int]): Cap on number of training episodes.
            mode (str): One of "train", "val", "infer".
            depth_clip_range (Optional[Tuple[float, float]]): Depth normalization range.
        """
        # VLA-specific fields (must be set before super().__init__ which calls _build_sampler_cfg etc.)
        self.zarr_paths = zarr_paths
        self.preprocessor = None
        self.objective = objective
        self.normalizer_dataloader_cfg = normalizer_dataloader_cfg
        self.use_relative_action = use_relative_action
        self.max_train_episodes = max_train_episodes
        self.normalizer = None
        self.depth_clip_range = depth_clip_range
        self.return_dataset_info = return_dataset_info
        self.dataset_names = []
        self._weights_list = []

        self.mode = mode
        self.aug_transform = None
        if self.mode == 'train':
            self.aug_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
            ])

        # Call base class __init__ (triggers _build_sampler_cfg, _build_key_mapping, _on_zarr_loaded)
        super().__init__(zarr_paths, shape_meta, seed, val_ratio, max_train_episodes)

        # Initialize weights (BaseRatioDataset logic)
        if self._weights_list:
            weights_sum = sum(self._weights_list)
            self.weights = [w / weights_sum for w in self._weights_list]
            self.dataset_lengths = self.sampler_lens
        else:
            self.weights = None
            self.dataset_lengths = None

    def _build_sampler_cfg(self):
        s = self.shape_meta
        return {
            'num_image_steps': s['obs']['rgb']['horizon'],
            'num_image_stride': s['obs']['rgb']['stride'],
            'num_state_steps': s['obs']['state']['horizon'],
            'num_state_stride': s['obs']['state']['stride'],
            'num_action_steps': s['action']['horizon'],
            'num_action_stride': s['action']['stride'],
        }

    def _build_key_mapping(self, zarr_path):
        return merge_key_mapping(zarr_path.get('mapping', None), self.motion_type)

    def _on_zarr_loaded(self, zarr_path, replay_buffer):
        weight = zarr_path.get('weight', None)
        if weight is not None:
            self._weights_list.append(weight)
        dataset_name = zarr_path.get('name', None)
        if dataset_name is None:
            dataset_name = pathlib.Path(str(zarr_path['path'])).stem
        self.dataset_names.append(dataset_name)

    def _on_validation_copy(self, val_set):
        val_set.mode = 'val' if self.mode == 'train' else self.mode
        val_set.aug_transform = None

    def _sample_to_data(self, sample):
        """
        Convert a sampled sequence into model-ready tensors/arrays.

        Args:
            sample (Dict[str, Any]): Mapping of keys to arrays or Future-like objects.

        Returns:
            Dict[str, np.ndarray]: Processed sample fields.
        """
        # Select data keys based on motion_type
        state, action = process_state_action(
            wrist_state = sample['wrist_state'].astype(np.float32), 
            hand_state = sample['hand_state'].astype(np.float32), 
            wrist_action = sample['wrist_action'].astype(np.float32), 
            hand_action = sample['hand_action'].astype(np.float32), 
            extrinsic = sample['extrinsic'].astype(np.float32).reshape(4, 4), # [16] -> [4, 4]
            normalizer = self.normalizer, 
            hand_ndim = self.hand_ndim, 
            motion_type = self.motion_type,
            use_relative_action = self.use_relative_action,
        )
        image, depth_images = process_image(
            sample['image'], 
            sample.get('depth', None), 
            self.aug_transform, 
            self.depth_clip_range,
        )

        intrinsic = sample['intrinsic'].astype(np.float32)
        instruction = sample['instruction']
        instruction_num = sample['instruction_num']
        # sample a random instruction from the candidate instructions
        
        if self.mode == 'train':
            idx = np.random.randint(0, instruction_num)
        else: 
            idx = 0
        instruction = instruction[idx]
        
        # Process all images in batch
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
        # Pad state and action to the same length as the sampler configuration
        state_pad = np.zeros((self.sampler_cfg['num_state_steps'], *state.shape[1:]), dtype=np.float32)
        state_pad[:state.shape[0]] = state
        action_pad = np.zeros((self.sampler_cfg['num_action_steps'], *action.shape[1:]), dtype=np.float32)
        actions_valid_mask = np.zeros((self.sampler_cfg['num_action_steps'],*action.shape[1:]), dtype=bool)
        actions_valid_mask[:action.shape[0]] = True
        action_pad[:action.shape[0]] = action

        data = {
            'input_ids': processed_results['input_ids'],
            'answer_start_idx': processed_results['answer_start_idx'],
            'attention_mask': processed_results['attention_mask'],
            'pixel_values': processed_results['pixel_values'], 
            'states': state_pad,
            'n_states': np.array(state.shape[0], dtype=np.int32),
            'actions': action_pad,
            'actions_valid_mask': actions_valid_mask,
            'n_actions': np.array(action.shape[0], dtype=np.int32),
            'is_vla_data': np.array(True, dtype=bool), 
        }
        # Add depth_values if available
        if 'depth_values' in processed_results:
            data['depth_values'] = processed_results['depth_values']
        if self.objective != "train_flow":
            data['labels'] = processed_results['labels']
        return data

    def set_preprocessor(self, preprocessor):
        """
        Set the tokenizer/vision preprocessor.

        Args:
            preprocessor (Callable): Preprocessor with tokenizer and encode logic.
        """
        self.preprocessor = preprocessor

    def set_normalizer(self, normalizer: LinearNormalizer):
        """
        Set the normalizer for state/action.

        Args:
            normalizer (LinearNormalizer): Normalizer instance.
        """
        self.normalizer = normalizer

    def get_normalizer(self):
        """
        Compute and store a normalizer from the dataset.

        Returns:
            LinearNormalizer: Fitted normalizer.
        """
        # Merge all data
        normalizer_dataset = LegendVLALowLevelDataset(
            zarr_paths=self.zarr_paths,
            shape_meta=self.shape_meta,
            max_train_episodes=self.max_train_episodes, 
            return_numpy=True, 
            use_relative_action=self.use_relative_action,
        )
        normalizer = get_normalizer(self.normalizer_dataloader_cfg, normalizer_dataset)
        self.normalizer = normalizer

        return normalizer

    def get_collator(self):
        """
        Build a data collator for batching.

        Returns:
            LegendVLDataCollator: Collator instance.
        """
        assert self.preprocessor is not None, "Preprocessor is not set"
        padding_side = 'left' if self.mode == 'infer-ar' else 'right'
        return LegendVLDataCollator(
            pad_token_id=self.preprocessor.tokenizer.pad_token_id,
            ignore_index=self.preprocessor.ignore_index,
            padding_side=padding_side,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a processed sample by global index.

        Args:
            idx (int): Global dataset index.

        Returns:
            Dict[str, torch.Tensor]: Sample tensors.
        """
        try: 
            # Find corresponding sampler
            curr_idx, dataset_idx = idx, 0
            while curr_idx >= self.sampler_lens[dataset_idx]:
                curr_idx -= self.sampler_lens[dataset_idx]
                dataset_idx += 1
            sample = self.samplers[dataset_idx].sample_sequence(curr_idx)
            
            data = self._sample_to_data(sample)
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
        """
        Total number of samples across all buffers.

        Returns:
            int: Dataset length.
        """
        return sum(self.sampler_lens)

class LegendUnifiedDataset(torch.utils.data.Dataset):
    """
    Unified dataset that concatenates VLA and optional VLM datasets.
    """
    def __init__(self,
        vla_dataset: LegendVLADataset,
        vlm_dataset: LegendVLMDataset = None,
    ):
        """
        Args:
            vla_dataset (LegendVLADataset): VLA dataset.
            vlm_dataset (Optional[LegendVLMDataset]): VLM dataset.
        """
        super().__init__()
        self.vla_dataset = vla_dataset
        self.vlm_dataset = vlm_dataset
        self.shape_meta = None
        
        print(f"LegendUnifiedDataset initialized with {len(self.vla_dataset)} VLA samples")
        if self.vlm_dataset is not None:
            print(f"LegendUnifiedDataset initialized with {len(self.vlm_dataset)} VLM samples")

    def get_collator(self):
        """
        Build a data collator for batching.

        Returns:
            LegendVLDataCollator: Collator instance.
        """
        return LegendVLDataCollator()

    def get_sampler(
        self, 
        batch_size: int, 
        vla_ratio: float = 1/8, 
        shuffle: bool = True, 
        seed: int = 42, 
        drop_last: bool = False,
    ): 
        """
        Create a batch sampler that mixes VLA/VLM data at a fixed ratio.

        Args:
            batch_size (int): Batch size.
            vla_ratio (float): Fraction of VLA samples per batch.
            shuffle (bool): Shuffle samples if True.
            seed (int): Random seed.
            drop_last (bool): Drop the last incomplete batch if True.

        Returns:
            torch.utils.data.BatchSampler: Sampler instance.
        """
        from torch.utils.data import BatchSampler, SequentialSampler
        from .sampler import UnifiedRatioSampler
        if shuffle: 
            assert hasattr(self.vla_dataset, 'weights') and hasattr(self.vla_dataset, 'dataset_lengths'), \
                "vla_dataset must have 'weights' and 'dataset_lengths' attributes"
            return UnifiedRatioSampler(
                weights=self.vla_dataset.weights,
                dataset_lengths=self.vla_dataset.dataset_lengths,
                vla_size=len(self.vla_dataset),
                vlm_size=len(self.vlm_dataset) if self.vlm_dataset is not None else 0,
                vla_ratio=vla_ratio,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last,
            )
        else: 
            return BatchSampler(
                SequentialSampler(range(len(self))), 
                batch_size=batch_size, 
                drop_last=drop_last,
            )

    def get_validation_dataset(self):
        """
        Build a validation dataset from the held-out splits.

        Returns:
            LegendUnifiedDataset: Validation dataset view.
        """
        return LegendUnifiedDataset(
            vla_dataset=self.vla_dataset.get_validation_dataset(),
            vlm_dataset=self.vlm_dataset.get_validation_dataset() if self.vlm_dataset is not None else None, 
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample by global index, padding missing fields for VLM samples.

        Args:
            idx (int): Global dataset index.

        Returns:
            Dict[str, torch.Tensor]: Sample tensors.
        """
        if self.shape_meta is None:
            self.shape_meta = dict()
            vla_sample = self.vla_dataset[0]
            for key in vla_sample.keys():
                # Only store shape for tensor-like objects, skip strings and other non-tensor types
                if hasattr(vla_sample[key], 'shape'):
                    self.shape_meta[key] = vla_sample[key].shape
        if idx < len(self.vla_dataset):
            return self.vla_dataset[idx]
        elif self.vlm_dataset is not None:
            sample = self.vlm_dataset[idx - len(self.vla_dataset)]
            if 'states' in self.shape_meta: 
                sample['states'] = torch.zeros(*self.shape_meta['states'])
            if 'actions' in self.shape_meta:
                sample['actions'] = torch.zeros(*self.shape_meta['actions'])
                sample['actions_valid_mask'] = torch.zeros(*self.shape_meta['actions'])
            if 'depth_values' in self.shape_meta: 
                sample['depth_values'] = torch.zeros(*sample['pixel_values'].shape)
            if 'n_states' in self.shape_meta: 
                sample['n_states'] = torch.tensor(0, dtype=torch.int32)
            if 'n_actions' in self.shape_meta: 
                sample['n_actions'] = torch.tensor(0, dtype=torch.int32)
            return sample
        else:
            raise ValueError("No dataset to get item from")

    def __len__(self):
        """
        Total number of samples.

        Returns:
            int: Dataset length.
        """
        return len(self.vla_dataset) + len(self.vlm_dataset) if self.vlm_dataset is not None else len(self.vla_dataset)

class LegendVLALowLevelDataset(BaseLegendZarrDataset):
    """
    Low-level dataset for normalizer fitting (state/action only).
    """
    def __init__(
        self,
        zarr_paths,
        shape_meta=None,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        return_numpy=True, # whether to return numpy arrays
        normalizer_dataloader_cfg=None,
        use_relative_action=False,
    ):
        """
        Args:
            zarr_paths (List[Dict]): List of zarr dataset configs.
            shape_meta (Dict): Metadata describing observation/action shapes.
            seed (int): Random seed for splitting/downsampling.
            val_ratio (float): Validation split ratio.
            max_train_episodes (Optional[int]): Cap on number of training episodes.
            return_numpy (bool): Return numpy arrays if True.
            normalizer_dataloader_cfg (Optional[Dict]): DataLoader config for normalizer fit.
            use_relative_action (bool): Use relative action representation if True.
        """
        self.return_numpy = return_numpy
        self.normalizer = None
        self.normalizer_dataloader_cfg = normalizer_dataloader_cfg
        self.use_relative_action = use_relative_action
        super().__init__(zarr_paths, shape_meta, seed, val_ratio, max_train_episodes)

    def _build_sampler_cfg(self):
        s = self.shape_meta
        return {
            'num_state_steps': s['obs']['state']['horizon'],
            'num_state_stride': s['obs']['state']['stride'],
            'num_action_steps': s['action']['horizon'],
            'num_action_stride': s['action']['stride'],
        }

    def _build_key_mapping(self, zarr_path):
        full_mapping = merge_key_mapping(zarr_path.get('mapping', None), self.motion_type)
        low_level_keys = ['wrist_state', 'hand_state', 'wrist_action', 'hand_action', 'extrinsic']
        return {k: full_mapping[k] for k in low_level_keys if k in full_mapping}

    def _lazy_load(self):
        return False  # normalizer 需要全量加载

    def _sample_to_data(self, sample):
        """
        Convert a sampled sequence into state/action arrays.

        Args:
            sample (Dict[str, Any]): Mapping of keys to arrays or Future-like objects.

        Returns:
            Dict[str, np.ndarray]: Processed sample fields.
        """
        # Select data keys based on motion_type
        state, action = process_state_action(
            wrist_state = sample['wrist_state'].astype(np.float32),
            hand_state = sample['hand_state'].astype(np.float32),
            wrist_action = sample['wrist_action'].astype(np.float32),
            hand_action = sample['hand_action'].astype(np.float32),
            extrinsic = sample['extrinsic'].astype(np.float32).reshape(4, 4), # [16] -> [4, 4]
            normalizer = self.normalizer,
            hand_ndim = self.hand_ndim,
            motion_type = self.motion_type,
            use_relative_action = self.use_relative_action,
        )

        if not self.use_relative_action:
            data = {
                'motions': np.concatenate([state, action], axis=0),
            }
        else:
            data = {
                'states': state,
                'actions': action,
            }
        return data

    def get_normalizer(self):
        """
        Compute and store a normalizer from the dataset.

        Returns:
            LinearNormalizer: Fitted normalizer.
        """
        self.normalizer = get_normalizer(self.normalizer_dataloader_cfg, self)
        return self.normalizer

    def set_normalizer(self, normalizer: LinearNormalizer):
        """
        Set the normalizer for state/action.

        Args:
            normalizer (LinearNormalizer): Normalizer instance.
        """
        self.normalizer = normalizer

    def get_collator(self):
        """
        Build a data collator for batching.

        Returns:
            ConcatDataCollator: Collator instance.
        """
        return ConcatDataCollator()

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a processed sample by global index.

        Args:
            idx (int): Global dataset index.

        Returns:
            Dict[str, np.ndarray]: Sample arrays (or tensors if return_numpy is False).
        """
        # Find corresponding sampler
        curr_idx = idx
        for i, length in enumerate(self.sampler_lens):
            if curr_idx < length:
                sample = self.samplers[i].sample_sequence(curr_idx)
                break
            curr_idx -= length

        data = self._sample_to_data(sample)
        if not self.return_numpy:
            data = dict_apply(data, torch.from_numpy)
        return data


