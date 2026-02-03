'''
Propise dataset for LegendVLA
Every action is the delta of the next predicted absolute state and the state at the beginning of the action chunk.
'''

from typing import Dict, Optional
import torch
import numpy as np
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import warnings
from PIL import Image
import torch.nn.utils.rnn as rnn_utils
from datasets import concatenate_datasets, load_from_disk, DatasetDict
from src.utils.pytorch_util import dict_apply
from src.utils.geometry import (
    transform_wrist_to_target_frame, 
    homo_matrix_from_trans_6drot, 
    homo_matrix_to_trans_6drot, 
    transform_hand_points_to_wrist_frame,
    transform_hand_points_to_target_frame,
    transform_hand_points_from_wrist_to_camera_frame,
)
from src.model.common.normalizer import LinearNormalizer
from .base_dataset import BaseRatioDataset, BaseLowdimDataset, BaseDataCollator
from .sampler import SequenceSampler, get_val_mask, downsample_mask
from .streaming_replay_buffer import StreamingReplayBuffer


class LegendVLADataset(BaseRatioDataset):
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
        self.zarr_paths = zarr_paths
        self.preprocessor = None
        self.objective = objective
        self.normalizer_dataloader_cfg = normalizer_dataloader_cfg
        self.use_relative_action = use_relative_action
        self.max_train_episodes = max_train_episodes
        self.normalizer = None
        self.depth_clip_range = depth_clip_range
        self.shape_meta = shape_meta
        self.motion_type = shape_meta['obs']['state']['type']
        self.hand_ndim = shape_meta['obs']['state']['hand']['shape'][-1] // 2 # per hand pca ncomponents
        self.sampler_cfg = {
            'num_image_steps': shape_meta['obs']['rgb']['horizon'],
            'num_image_stride': shape_meta['obs']['rgb']['stride'],
            'num_state_steps': shape_meta['obs']['state']['horizon'],
            'num_state_stride': shape_meta['obs']['state']['stride'],
            'num_action_steps': shape_meta['action']['horizon'],
            'num_action_stride': shape_meta['action']['stride'],
        }

        # Initialize storage lists
        self.replay_buffers = []
        self.train_masks = []
        self.samplers = []
        self.sampler_lens = []

        self.mode = mode
        self.aug_transform = None
        if self.mode == 'train':
            self.aug_transform = transforms.Compose([
                # ColorJitter: random change brightness, contrast, saturation, and hue
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                # GaussianBlur: apply gaussian blur
                # kernel_size must be odd
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
            ])
        
        # Process each zarr file
        for zarr_path in zarr_paths:
            # Create replay buffer
            replay_buffer = StreamingReplayBuffer.copy_from_path(
                zarr_path['path'], 
                keys=['image', 'depth', 'state', 'instruction', 'instruction_num', 'action', 'extrinsic', 'intrinsic'], 
                lazy_load=True
            )
            self.replay_buffers.append(replay_buffer)

            # Create train mask
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes,
                val_ratio=val_ratio,
                seed=seed)
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask,
                max_n=max_train_episodes,
                seed=seed)
            self.train_masks.append(train_mask)
            
            # Create sampler
            sampler = SequenceSampler(
                replay_buffer=replay_buffer, 
                episode_mask=train_mask, 
                **self.sampler_cfg
            )
            self.samplers.append(sampler)
            
            # Record sampler length
            self.sampler_lens.append(len(sampler))

        if zarr_paths is not None and zarr_paths[0].get('weight', None) is not None:
            weights = [path['weight'] for path in zarr_paths]
            super().__init__(weights, self.sampler_lens)
        else:
            super().__init__()

    def get_validation_dataset(self):
        """
        Build a validation dataset from the held-out episodes.

        Returns:
            LegendVLADataset: Validation dataset view.
        """
        val_set = copy.copy(self)
        val_set.samplers = []
        val_set.train_masks = []
        val_set.sampler_lens = []
        val_set.mode = 'val'
        val_set.aug_transform = None

        for i, replay_buffer in enumerate(self.replay_buffers):
            # Create validation set sampler
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                episode_mask=~self.train_masks[i],
                **self.sampler_cfg
            )
            val_set.samplers.append(sampler)
            val_set.train_masks.append(~self.train_masks[i])
            val_set.sampler_lens.append(len(sampler))
            
        return val_set

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
            wrist_state = sample['state/wrist'].astype(np.float32), 
            hand_state = sample[f'state/{self.motion_type}'].astype(np.float32), 
            wrist_action = sample['action/wrist'].astype(np.float32), 
            hand_action = sample[f'action/{self.motion_type}'].astype(np.float32), 
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
        padding_side = 'left' if self.mode == 'infer' else 'right'
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
        # Find corresponding sampler
        curr_idx = idx
        for i, length in enumerate(self.sampler_lens):
            if curr_idx < length:
                sample = self.samplers[i].sample_sequence(curr_idx)
                break
            curr_idx -= length
        
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def __len__(self):
        """
        Total number of samples across all buffers.

        Returns:
            int: Dataset length.
        """
        return sum(self.sampler_lens)

class LegendVLMDataset(torch.utils.data.Dataset):
    """
    Dataset for VLM-only data stored in HuggingFace datasets format.
    """
    def __init__(
        self,
        dataset_paths,
        split='train',
        cache_dir=None,
        weights=[0.5, 0.5, 0.5],
        seed=42,
        mode='train',
    ):
        """
        Args:
            dataset_paths (Union[str, List[str]]): Dataset disk paths.
            split (str): Split name to load (train/val/test).
            cache_dir (Optional[str]): HF datasets cache directory.
            weights (List[float]): Weights for rating-based text selection.
            seed (int): Random seed.
            mode (str): One of "train" or "val".
        """
        super().__init__()
        self.dataset_paths = [dataset_paths] if isinstance(dataset_paths, str) else dataset_paths
        self.split = split
        self.weights = weights
        self.cache_dir = cache_dir
        self.mode = mode
        self.seed = seed
        self.preprocessor = None
        
        if self.mode == 'train':
            self.aug_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
            ])
        else:
            self.aug_transform = None

        # --- directly load the dataset according to the split ---
        loaded_datasets = []
        for path in self.dataset_paths:
            try:
                ds = load_from_disk(path)
                
                # case A: if the loaded dataset is a DatasetDict (contains train/test/val)
                if isinstance(ds, DatasetDict):
                    if self.split in ds:
                        ds_to_add = ds[self.split]
                    else:
                        # if the sub dataset is too small to have this split (e.g. no test), skip it
                        print(f"Warning: Split '{self.split}' not found in {path}, skipping this sub-dataset.")
                        continue
                # case B: if the loaded dataset is a Dataset object
                else:
                    ds_to_add = ds

                if len(ds_to_add) == 0:
                    print(f"Warning: Dataset at {path} is empty, skipping.")
                    continue

                loaded_datasets.append(ds_to_add)
                
            except Exception as e:
                warnings.warn(f"Error loading dataset from {path}: {e}")
                continue

        if not loaded_datasets:
            # if the validation set loading fails (maybe all sub datasets don't have test), we can throw an exception or return None
            print(f"Notice: No datasets found for split '{self.split}'.")
            self.main_dataset = None
        else:
            # merge all sub datasets that meet the criteria
            self.main_dataset = concatenate_datasets(loaded_datasets)
            print(f"Successfully loaded split '{self.split}' with {len(self.main_dataset)} samples.")

    def get_validation_dataset(self, val_split='test'):
        """
        directly load the dataset with split='test'
        """
        # instantiate a new object, set split to val_split
        val_dataset = LegendVLMDataset(
            dataset_paths=self.dataset_paths,
            split=val_split,
            cache_dir=self.cache_dir,
            weights=self.weights,
            seed=self.seed,
            mode='val'
        )
        
        # inherit the current preprocessor
        if self.preprocessor is not None:
            val_dataset.set_preprocessor(self.preprocessor)
        
        # if the corresponding split has no data, return None
        if val_dataset.main_dataset is None:
            return None
            
        return val_dataset
    
    def _sample_to_data(self, sample, idx):
        """
        Convert a raw dataset row into model-ready fields.

        Args:
            sample (Dict[str, Any]): Dataset row.
            idx (int): Row index (unused).

        Returns:
            Dict[str, np.ndarray]: Processed sample fields.
        """
        images = sample['images'] # List[PIL.JpegImagePlugin.JpegImageFile]
        text = sample['texts'] 
        weights = self.weights
        # There are some None values in the ratings, we replace them with 0
        formatting_ratings = np.array([rating if rating is not None else 0 for rating in sample['formatting_ratings']])
        visual_dependency_ratings = np.array([rating if rating is not None else 0 for rating in sample['visual_dependency_ratings']])
        relevance_ratings = np.array([rating if rating is not None else 0 for rating in sample['relevance_ratings']])

        if len(text) > 1:
            scores = formatting_ratings * weights[0] + \
                    visual_dependency_ratings * weights[1] + \
                    relevance_ratings * weights[2]
            text = text[np.argmax(scores)]
        else:
            text = text[0]
        question = str(text['user'])
        answer = str(text['assistant'])

        for idx in range(len(images)):
            if images[idx].mode != 'RGB':
                images[idx] = images[idx].convert('RGB')

        augmented_images = []
        for img_pil in images:
            if self.mode == 'train' and self.aug_transform is not None:
                augmented_pil = self.aug_transform(img_pil)
            else:
                augmented_pil = img_pil
            augmented_np = np.array(augmented_pil, dtype=np.uint8)
            augmented_images.append(augmented_np)
        images_to_process = np.stack(augmented_images, dtype=np.uint8)
        # Process all images in batch
        processed_results = self.preprocessor(
            images=images_to_process, 
            text=question, 
            target=answer, 
            mode=self.mode
        )

        data = {
            'input_ids': processed_results['input_ids'],
            'labels': processed_results['labels'],
            'attention_mask': processed_results['attention_mask'],
            'pixel_values': processed_results['pixel_values'], 
            'answer_start_idx': processed_results['answer_start_idx'],
            'is_vla_data': np.array(False, dtype=bool), 
        }
        return data

    def get_collator(self):
        """
        Build a data collator for batching.

        Returns:
            LegendVLDataCollator: Collator instance.
        """
        assert self.preprocessor is not None, "Preprocessor is not set"
        return LegendVLDataCollator(
            pad_token_id=self.preprocessor.tokenizer.pad_token_id,
            ignore_index=self.preprocessor.ignore_index,
        )

    def set_preprocessor(self, preprocessor):
        """
        Set the tokenizer/vision preprocessor.

        Args:
            preprocessor (Callable): Preprocessor with tokenizer and encode logic.
        """
        self.preprocessor = preprocessor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a processed sample by index.

        Args:
            idx (int): Dataset index.

        Returns:
            Dict[str, torch.Tensor]: Sample tensors.
        """
        # Find corresponding sampler
        sample = self.main_dataset[idx]
        data = self._sample_to_data(sample, idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def __len__(self):
        """
        Dataset length.

        Returns:
            int: Number of samples.
        """
        return len(self.main_dataset)

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

class LegendVLALowLevelDataset(BaseLowdimDataset):
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
        super().__init__()
        self.shape_meta = shape_meta
        self.motion_type = shape_meta['obs']['state']['type']
        self.hand_ndim = shape_meta['obs']['state']['hand']['shape'][-1] // 2 # per hand pca ncomponents
        self.return_numpy = return_numpy
        self.normalizer = None
        self.normalizer_dataloader_cfg = normalizer_dataloader_cfg
        self.use_relative_action = use_relative_action
        self.sampler_cfg = {
            'num_state_steps': shape_meta['obs']['state']['horizon'],
            'num_state_stride': shape_meta['obs']['state']['stride'],
            'num_action_steps': shape_meta['action']['horizon'],
            'num_action_stride': shape_meta['action']['stride'],
        }

        # Initialize storage lists
        self.replay_buffers = []
        self.train_masks = []
        self.samplers = []
        self.sampler_lens = []
        
        # Process each zarr file
        for zarr_path in zarr_paths:
            # Create replay buffer
            replay_buffer = StreamingReplayBuffer.copy_from_path(
                zarr_path['path'], keys=['state', 'action', 'extrinsic'], lazy_load=False)
            self.replay_buffers.append(replay_buffer)

            # Create train mask
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes,
                val_ratio=val_ratio,
                seed=seed
            )
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask,
                max_n=max_train_episodes
            )
            self.train_masks.append(train_mask)
            
            # Create sampler
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                episode_mask=train_mask,
                **self.sampler_cfg
            )
            self.samplers.append(sampler)
            
            # Record sampler length
            self.sampler_lens.append(len(sampler))

    def get_validation_dataset(self):
        """
        Build a validation dataset from the held-out episodes.

        Returns:
            LegendVLALowLevelDataset: Validation dataset view.
        """
        val_set = copy.copy(self)
        val_set.samplers = []
        val_set.train_masks = []
        val_set.sampler_lens = []

        for i, replay_buffer in enumerate(self.replay_buffers):
            # Create validation set sampler
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                episode_mask=~self.train_masks[i],
                **self.sampler_cfg
            )
            val_set.samplers.append(sampler)
            val_set.train_masks.append(~self.train_masks[i])
            val_set.sampler_lens.append(len(sampler))
            
        return val_set

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
            wrist_state = sample['state/wrist'].astype(np.float32), 
            hand_state = sample[f'state/{self.motion_type}'].astype(np.float32), 
            wrist_action = sample['action/wrist'].astype(np.float32), 
            hand_action = sample[f'action/{self.motion_type}'].astype(np.float32), 
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

    def __len__(self):
        """
        Total number of samples across all buffers.

        Returns:
            int: Dataset length.
        """
        return sum(self.sampler_lens)


class LegendVLDataCollator(BaseDataCollator):
    def __init__(self, pad_token_id: int = 0, ignore_index: int = -100, padding_side: str = 'right'):
        """
        Args:
            pad_token_id (int): Token ID for padding input_ids.
            ignore_index (int): Ignore index for labels padding.
            padding_side (str): "left" or "right" padding.
        """
        super().__init__()
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.padding_side = padding_side

    def __call__(self, data_list):
        """
        DataLoader will pass a list of samples from the Dataset to this function.
        Args:
            data_list: a list, where each element is the return value of the Dataset's __getitem__ method.
               e.g., [{'image': tensor, 'input_ids': tensor}, {'image': tensor, 'input_ids': tensor}, ...]
        Returns:
            A dictionary with the keys the same as the return value of the Dataset's __getitem__ method.
        """
        batch = {}
        input_ids_batch = [item['input_ids'] for item in data_list]
        labels_batch = [item['labels'] for item in data_list]
        batch["input_ids"] = rnn_utils.pad_sequence(
            input_ids_batch,
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side=self.padding_side
        )
        batch["labels"] = rnn_utils.pad_sequence(
            labels_batch,
            batch_first=True,
            padding_value=self.ignore_index,
            padding_side=self.padding_side
        )
        batch["attention_mask"] = (batch["input_ids"] != self.pad_token_id).long()
        has_depth_values = [(item['is_vla_data'] == True) for item in data_list]
        batch['has_depth_values'] = torch.tensor(has_depth_values, dtype=torch.bool)
        for key in data_list[0].keys():
            if key in ['input_ids', 'attention_mask', 'labels']: 
                continue 
            if key in ['pixel_values', 'depth_values']: 
                batch[key] = rnn_utils.pad_sequence(
                    [item[key] for item in data_list],
                    batch_first=True,
                    padding_value=0,
                    padding_side='right'
                )
            else:
                batch[key] = torch.stack([item[key] for item in data_list])

        return batch

class ConcatDataCollator(BaseDataCollator):
    def __init__(self):
        super().__init__()

    def __call__(self, data_list):
        """
        A Collator that concatenates a list of samples into a single batch, at the first dimension.
        Args:
            data_list: a list, where each element is the return value of the Dataset's __getitem__ method.
               e.g., [{'state/hand': np.ndarray, 'action/hand': np.ndarray}, {'state/hand': np.ndarray, 'action/hand': np.ndarray}, ...]
        Returns:
            A dictionary with the keys the same as the return value of the Dataset's __getitem__ method.
        """
        batch = {}
        for key in data_list[0].keys():
            if isinstance(data_list[0][key], torch.Tensor): # tensor
                batch[key] = torch.cat([item[key] for item in data_list], dim=0)
            elif isinstance(data_list[0][key], np.ndarray): # numpy
                batch[key] = np.concatenate([item[key] for item in data_list], axis=0)
            else: # list
                batch[key] = [item[key] for item in data_list]
        return batch


def get_relative_action(state, action):
    '''
    Args:
        state: np.ndarray, shape: [wrist_dim + hand_dim]
        action: np.ndarray, shape: [H, wrist_dim + hand_dim]
    Returns:
        action: np.ndarray, shape: [H, wrist_dim + hand_dim]
    '''
    action = action.copy() # avoid modifying the original action
    for idx in range(2): 
        wrist_action_homo_mat = homo_matrix_from_trans_6drot(action[..., idx*3 : idx*3+3], action[..., 6+idx*6 : 6+idx*6+6])
        wrist_state_homo_mat = homo_matrix_from_trans_6drot(state[idx*3 : idx*3+3], state[6+idx*6 : 6+idx*6+6])
        wrist_action_homo_mat = np.linalg.pinv(wrist_state_homo_mat) @ wrist_action_homo_mat
        trans, rot_6d = homo_matrix_to_trans_6drot(wrist_action_homo_mat)
        action[..., idx*3 : idx*3+3] = trans
        action[..., 6+idx*6 : 6+idx*6+6] = rot_6d
    
    action[..., 18:] = action[..., 18:] - state[18:]
    return action

def get_absolute_action(state, relative_action):
    '''
    Convert relative action back to absolute action.
    This is the inverse operation of get_relative_action.
    
    Args:
        state: torch.Tensor or np.ndarray, shape: [wrist_dim + hand_dim] - current state
            state is in the first frame's camera coordinate system, where wrist is in cam frame and hand is in wrist frame
        relative_action: torch.Tensor or np.ndarray, shape: [H, wrist_dim + hand_dim] - relative action

    Returns:
        absolute_action: torch.Tensor or np.ndarray, shape: [H, wrist_dim + hand_dim] - absolute action
    '''
    # Create a copy of relative_action to store absolute_action
    if isinstance(relative_action, torch.Tensor):
        absolute_action = relative_action.clone()
    else:
        absolute_action = relative_action.copy()
    
    # For wrist parameters, absolute action = state @ relative action
    # This is the inverse of: relative = pinv(state) @ action
    for idx in range(2): 
        wrist_relative_action_homo_mat = homo_matrix_from_trans_6drot(relative_action[..., idx*3 : idx*3+3], relative_action[..., 6+idx*6 : 6+idx*6+6])
        wrist_state_homo_mat = homo_matrix_from_trans_6drot(state[idx*3 : idx*3+3], state[6+idx*6 : 6+idx*6+6])
        wrist_action_homo_mat = wrist_state_homo_mat @ wrist_relative_action_homo_mat
        trans, rot_6d = homo_matrix_to_trans_6drot(wrist_action_homo_mat)
        absolute_action[..., idx*3 : idx*3+3] = trans
        absolute_action[..., 6+idx*6 : 6+idx*6+6] = rot_6d
    
    # For hand parameters, absolute action = relative action + state
    # This is the inverse of: relative = action - state
    absolute_action[..., 18:] = relative_action[..., 18:] + state[18:]
    
    return absolute_action

def transform_hand_from_wrist_to_camera(absolute_action, extrinsic):
    '''
    Transform hand points from wrist frame to camera coordinate system.
    This function transforms hand points from wrist frame to first frame's camera coordinate,
    then to target frames' camera coordinates.
    
    Args:
        absolute_action: torch.Tensor or np.ndarray, shape: [H, wrist_dim + hand_dim] or [wrist_dim + hand_dim]
            absolute action where hand is in wrist frame
        extrinsic: torch.Tensor or np.ndarray, shape: [H, 4, 4] or [4, 4] - camera extrinsic (world2cam) for each frame
    Returns:
        absolute_action: torch.Tensor or np.ndarray, shape: [H, wrist_dim + hand_dim] or [wrist_dim + hand_dim]
            absolute action where hand is in camera frame
    '''
    # Create a copy to avoid modifying the input
    if isinstance(absolute_action, torch.Tensor):
        absolute_action = absolute_action.clone()
    else:
        absolute_action = absolute_action.copy()
    
    hand_points_wrist = absolute_action[..., 18:]  # (H, 30) or (30,) - hand in wrist frame
    wrist_action_initial = absolute_action[..., :18]  # (H, 18) or (18,) - wrist in first frame's cam coordinate
    
    # Transform hand points from wrist frame to first frame's cam coordinate
    if absolute_action.ndim > 1:
        # Multiple frames
        hand_points_cam_initial = transform_hand_points_from_wrist_to_camera_frame(hand_points_wrist, wrist_action_initial)
    else:
        # Single frame
        hand_points_cam_initial = transform_hand_points_from_wrist_to_camera_frame(hand_points_wrist.reshape(1, -1), wrist_action_initial.reshape(1, -1))
        hand_points_cam_initial = hand_points_cam_initial.reshape(-1)
    
    # Now both wrist and hand are in first frame's cam coordinate
    absolute_action[..., 18:] = hand_points_cam_initial
    initial_extrinsic_inv = np.linalg.inv(extrinsic[0])
    # Transform wrist from first frame's cam coordinate to world, then to target frames' cam coordinates
    wrist_action_world = transform_wrist_to_target_frame(wrist_action_initial, initial_extrinsic_inv)  # Transform to world
    wrist_action_target = transform_wrist_to_target_frame(wrist_action_world, extrinsic)  # Transform to target frames
    absolute_action[..., :18] = wrist_action_target
    
    # Transform hand points from first frame's cam coordinate to world, then to target frames' cam coordinates
    hand_points_world = transform_hand_points_to_target_frame(hand_points_cam_initial, initial_extrinsic_inv)  # Transform to world
    hand_points_target = transform_hand_points_to_target_frame(hand_points_world, extrinsic)  # Transform to target frames
    absolute_action[..., 18:] = hand_points_target
    
    return absolute_action

def process_state_action(
    wrist_state, 
    hand_state, 
    wrist_action, 
    hand_action, 
    extrinsic, 
    hand_ndim, 
    normalizer : Optional[LinearNormalizer] = None, 
    motion_type = 'mano',
    use_relative_action = False,
):
    '''
    Args:
        wrist_state: np.ndarray, shape: [N_state, wrist_dim]
        hand_state: np.ndarray, shape: [N_state, all_hand_dim]
        wrist_action: np.ndarray, shape: [N_action, wrist_dim]
        hand_action: np.ndarray, shape: [N_action, all_hand_dim]
        extrinsic: np.ndarray, shape: [4, 4]
        hand_ndim: int
        normalizer: Optional[LinearNormalizer]
        motion_type: str, 'mano' or 'keypoint'
        use_relative_action: bool
    Returns:
        state: np.ndarray, shape: [N_state, wrist_dim + hand_dim]
        action: np.ndarray, shape: [N_action, wrist_dim + hand_dim]
    '''
    # use first self.hand_ndim components of hand state and action
    all_hand_ndim = hand_state.shape[-1] // 2 # per hand dims, i.e. 45 in MANO hand params
    hand_state = np.concatenate([
        hand_state[:, :hand_ndim], 
        hand_state[:, all_hand_ndim:all_hand_ndim + hand_ndim]
    ], axis=-1)
    hand_action = np.concatenate([
        hand_action[:, :hand_ndim], 
        hand_action[:, all_hand_ndim:all_hand_ndim + hand_ndim]
    ], axis=-1)

    if motion_type == 'fingertips': 
        # TODO: We can try transform the fingertips to the camera coordinate system or wrist frame coordinate system
        processed_hand_state = transform_hand_points_to_wrist_frame(hand_state, wrist_state)
        processed_hand_state = processed_hand_state.reshape(hand_state.shape)
        processed_hand_action = transform_hand_points_to_wrist_frame(hand_action, wrist_action)
        processed_hand_action = processed_hand_action.reshape(hand_action.shape)
    elif motion_type == 'mano':
        processed_hand_state = hand_state
        processed_hand_action = hand_action
    else:
        raise ValueError(f"Unsupported motion type: {motion_type}")

    # transform the wrist state and action to the camera coordinate system
    processed_wrist_state = transform_wrist_to_target_frame(wrist_state, extrinsic)
    processed_wrist_action = transform_wrist_to_target_frame(wrist_action, extrinsic)

    # use delta of wrist translation and hand mano params as action
    processed_state = np.concatenate([processed_wrist_state, processed_hand_state], axis=-1)
    processed_action = np.concatenate([processed_wrist_action, processed_hand_action], axis=-1)
    if use_relative_action:
        processed_action = get_relative_action(processed_state[-1], processed_action)

    if normalizer is not None:
        if not use_relative_action: # Use unified normalizer for both state and action
            state = normalizer['motions'](processed_state)
            action = normalizer['motions'](processed_action)
        else: # Use separate normalizers for state and action
            state = normalizer['states'](processed_state)
            action = normalizer['actions'](processed_action)
    else: # No normalizer
        state = processed_state
        action = processed_action
    return state, action

# TODO: maybe we need to use the same augmentation for all images in the action chunk
# TODO: we can try more advanced augmentation techniques, notably, we should care about the depth image augmentation
def process_image(image, depth_image = None, aug_transform = None, depth_clip_range = None):
    '''
    Args:
        image: np.ndarray, shape: [N, H, W, 3]
        depth_image: np.ndarray, shape: [N, H, W]
        aug_transform: Optional[Callable]
    Returns:
        image: np.ndarray, shape: [N, H, W, 3]
        depth_image: np.ndarray, shape: [N, H, W]
    '''
    images_to_process = image
    depth_images_to_process = None
    if depth_image is not None:
        # Convert to float32 first to avoid dtype leak to float64
        depth_images_to_process = depth_image / np.float32(1000.0) # convert mm to m
        # normalize the depth images to [0, 1]
        depth_images_to_process = np.clip(
            depth_images_to_process, 
            np.float32(depth_clip_range[0]), 
            np.float32(depth_clip_range[1])
        ) / np.float32(depth_clip_range[1] - depth_clip_range[0] + 1e-6)
    if aug_transform is not None:
        augmented_images = []
        for img_np in images_to_process:
            # convert NumPy array (H, W, C) to PIL Image
            img_pil = Image.fromarray(img_np)
            augmented_pil = aug_transform(img_pil)
            augmented_np = np.array(augmented_pil, dtype=np.uint8)
            augmented_images.append(augmented_np)
        images_to_process = np.stack(augmented_images, dtype=np.uint8)

    return images_to_process, depth_images_to_process


def get_normalizer(dataloader_cfg, normalizer_dataset = None, **kwargs):
    # Merge all data
    if normalizer_dataset is None:
        normalizer_dataset = LegendVLALowLevelDataset(**kwargs)
    dataloader = DataLoader(normalizer_dataset, collate_fn=normalizer_dataset.get_collator(), **dataloader_cfg)
    assert len(dataloader) > 0, "No data to calculate normalizer"
    normalizer = LinearNormalizer()
    normalizer_keys = next(iter(dataloader)).keys()
    normalizer.start_streaming_fit(keys=normalizer_keys)
    for batch in tqdm(dataloader, desc="Calculating normalizer"):
        input_data = {
            k: v.reshape(-1, v.shape[-1]) for k, v in batch.items() \
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray)
        }
        normalizer.update_streaming_fit(input_data)
    normalizer.finish_streaming_fit()
    # ignore the wrist rotation
    for key in normalizer_keys:
        if key in ['states', 'actions', 'motions']:
            normalizer.ignore_dim(key=key, dim=slice(6, 18))
        else: 
            raise ValueError(f"Unsupported key: {key}")

    def print_dict(d):
        for k, v in d.items():
            print(f"{k}: {v}")
    
    for key in normalizer.params_dict.keys():
        print(f"{key}: ")
        print_dict(normalizer.params_dict[key]['input_stats'])
        print(f"scale: {normalizer.params_dict[key]['scale']}")
        print(f"offset: {normalizer.params_dict[key]['offset']}")

    return normalizer


def test_dataset_loading():
    """
    Test function to load and test the UnifiedDataset DataLoader using the config file.
    """
    from omegaconf import OmegaConf
    import hydra
    from torch.utils.data import DataLoader
    import pathlib
    from datetime import datetime
    
    # Register eval resolver for config
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    
    # Register now resolver for datetime formatting (used by Hydra)
    def now_resolver(format_str: str) -> str:
        """Resolver for ${now:format} interpolation."""
        return datetime.now().strftime(format_str)
    OmegaConf.register_new_resolver("now", now_resolver, replace=True)
    
    # Register hydra resolver (returns empty string for non-hydra contexts)
    def hydra_resolver(key: str) -> str:
        """Resolver for ${hydra:key} interpolation. Returns empty string in test context."""
        return ""
    OmegaConf.register_new_resolver("hydra", hydra_resolver, replace=True)
    
    # Load config file
    config_path = pathlib.Path(__file__).parent.parent.parent / "src" / "config" / "experiment" / "pretrain_legendvla_deepspeed.yaml"
    print(f"Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # Resolve config to evaluate all ${eval:}, ${now:}, and ${hydra:} expressions
    try:
        OmegaConf.resolve(cfg)
    except Exception as e:
        print(f"   Warning: Some config values could not be resolved: {e}")
        print("   Continuing with unresolved config (this is OK for testing)...")
    
    print("\n" + "="*80)
    print("Testing UnifiedDataset Dataloader Only")
    print("="*80)

    try:
        vla_dataset = hydra.utils.instantiate(cfg.dataset.vla_dataset)
        vla_processor = hydra.utils.instantiate(cfg.vla_processor)
        vla_dataset.set_preprocessor(vla_processor)

        try:
            vlm_dataset = hydra.utils.instantiate(cfg.dataset.vlm_dataset)
            vlm_processor = hydra.utils.instantiate(cfg.vlm_processor)
            vlm_dataset.set_preprocessor(vlm_processor)
        except Exception:
            vlm_dataset = None

        unified_dataset = LegendUnifiedDataset(
            vla_dataset=vla_dataset,
            vlm_dataset=vlm_dataset,
        )

        batch_sampler = unified_dataset.get_sampler(
            batch_size=cfg.dataloader.batch_sampler.batch_size,
            vla_ratio=cfg.dataloader.batch_sampler.vla_ratio,
            shuffle=cfg.dataloader.batch_sampler.shuffle,
            seed=cfg.dataloader.batch_sampler.seed,
            drop_last=cfg.dataloader.batch_sampler.drop_last,
        )

        dataloader = DataLoader(
            dataset=unified_dataset,
            batch_sampler=batch_sampler,
            collate_fn=unified_dataset.get_collator(),
            **cfg.dataloader.loader
        )
        print(f"   ✓ UnifiedDataset Dataloader created successfully")
        print(f"   - Dataloader length: {len(dataloader)}")
        print(f"   - Dataloader batch size: {cfg.dataloader.batch_sampler.batch_size}")
        for idx, batch in enumerate(dataloader):
            if idx > 100:
                break
            print(f"batch {idx} pixel values range: {batch['pixel_values'].min()}, {batch['pixel_values'].max()}")
        
        # Get first batch and output complete contents
        print(f"\n   Getting first batch and outputting complete contents...")
        first_batch = next(iter(dataloader))
        
        # Set torch and numpy print options to show all values without truncation
        import sys
        
        # Set torch to show all values (PyTorch doesn't have get_printoptions, so we just set it)
        torch.set_printoptions(
            threshold=sys.maxsize,  # Show all elements
            edgeitems=sys.maxsize,  # Show all edge items
            linewidth=sys.maxsize,  # No line wrapping limit
            precision=6,  # Keep reasonable precision
        )
        
        np.set_printoptions(
            threshold=sys.maxsize,  # Show all elements
            edgeitems=sys.maxsize,  # Show all edge items
            linewidth=sys.maxsize,  # No line wrapping limit
            precision=6,  # Keep reasonable precision
        )
        
        print(f"\n   {'='*80}")
        print(f"   First Batch Complete Contents:")
        print(f"   {'='*80}")
        print(f"   Batch keys: {list(first_batch.keys())}")
        print(f"   Batch size: {len(first_batch[list(first_batch.keys())[0]]) if isinstance(first_batch[list(first_batch.keys())[0]], (torch.Tensor, np.ndarray, list)) else 'N/A'}")
        print(f"\n")
        
        for key, value in first_batch.items():
            print(f"   {'-'*80}")
            print(f"   Key: {key}")
            print(f"   Type: {type(value)}")
            
            if isinstance(value, torch.Tensor):
                print(f"   Shape: {value.shape}")
                print(f"   Dtype: {value.dtype}")
                print(f"   Device: {value.device}")
                print(f"   Requires grad: {value.requires_grad}")
                
                # Output statistics
                if value.numel() > 0:
                    print(f"   Min: {value.min().item()}")
                    print(f"   Max: {value.max().item()}")
                    print(f"   Mean: {value.float().mean().item()}")
                    print(f"   Std: {value.float().std().item()}")
                
                # Output complete values - no size limit
                print(f"   Complete Values:\n{value}")
            
            elif isinstance(value, np.ndarray):
                print(f"   Shape: {value.shape}")
                print(f"   Dtype: {value.dtype}")
                
                # Output statistics
                if value.size > 0:
                    print(f"   Min: {value.min()}")
                    print(f"   Max: {value.max()}")
                    print(f"   Mean: {value.mean()}")
                    print(f"   Std: {value.std()}")
                
                # Output complete values - no size limit
                print(f"   Complete Values:\n{value}")
            
            elif isinstance(value, (list, tuple)):
                print(f"   Length: {len(value)}")
                if len(value) > 0:
                    print(f"   Element type: {type(value[0])}")
                # Output complete values - no size limit
                print(f"   Complete Values:\n{value}")
            
            elif isinstance(value, (int, float, bool, str)):
                print(f"   Value: {value}")
            
            else:
                print(f"   Value: {value}")
                if hasattr(value, '__dict__'):
                    print(f"   Attributes: {list(value.__dict__.keys())}")
            
            print(f"")
        
        # Validation checks
        print(f"\n   {'='*80}")
        print(f"   Validation Checks:")
        print(f"   {'='*80}")
        
        # Check 1: input_ids 中为 0 的部分和 attention_mask 重合
        print(f"\n   1. Checking input_ids padding matches attention_mask...")
        if 'input_ids' in first_batch and 'attention_mask' in first_batch:
            input_ids = first_batch['input_ids']
            attention_mask = first_batch['attention_mask']
            padding_mask = (input_ids == 0)
            attention_zero = (attention_mask == 0)
            
            if torch.equal(padding_mask, attention_zero):
                print(f"      ✓ PASS: input_ids padding matches attention_mask")
            else:
                mismatches = (padding_mask != attention_zero).sum().item()
                print(f"      ✗ FAIL: {mismatches} mismatches found between input_ids padding and attention_mask")
        else:
            print(f"      ⚠ SKIP: Missing input_ids or attention_mask")
        
        # Check 2 & 3: state_token_id 和 action_token_id 个数检查
        print(f"\n   2. Checking state_token_id counts match non-zero timesteps in states...")
        print(f"   3. Checking action_token_id counts match non-zero timesteps in actions...")
        if 'input_ids' in first_batch and 'states' in first_batch and 'actions' in first_batch:
            input_ids = first_batch['input_ids']
            states = first_batch['states']
            actions = first_batch['actions']
            
            # Get token IDs from processor
            if vla_dataset.preprocessor is not None:
                state_token_id = vla_dataset.preprocessor.state_token_id
                action_token_id = vla_dataset.preprocessor.action_token_id
                
                batch_size = input_ids.shape[0]
                state_check_passed = True
                action_check_passed = True
                
                for i in range(batch_size):
                    # Count state_token_id in input_ids
                    state_token_count = (input_ids[i] == state_token_id).sum().item()
                    
                    # Count non-zero timesteps in states
                    # states shape: [batch_size, n_obs_state_steps, state_dim]
                    state_non_zero_timesteps = 0
                    for t in range(states.shape[1]):
                        if not torch.all(states[i, t] == 0):
                            state_non_zero_timesteps += 1
                    
                    if state_token_count != state_non_zero_timesteps:
                        print(f"      ✗ FAIL: Sample {i}: state_token_id count ({state_token_count}) != non-zero state timesteps ({state_non_zero_timesteps})")
                        state_check_passed = False
                    
                    # Count action_token_id in input_ids
                    action_token_count = (input_ids[i] == action_token_id).sum().item()
                    
                    # Count non-zero timesteps in actions
                    # actions shape: [batch_size, horizon, action_dim]
                    action_non_zero_timesteps = 0
                    if 'actions_valid_mask' in first_batch:
                        # Use actions_valid_mask if available
                        actions_valid_mask = first_batch['actions_valid_mask']
                        action_non_zero_timesteps = actions_valid_mask[i].any(dim=1).sum().item()
                    else:
                        # Fallback: check if action is non-zero
                        for t in range(actions.shape[1]):
                            if not torch.all(actions[i, t] == 0):
                                action_non_zero_timesteps += 1
                    
                    if action_token_count != action_non_zero_timesteps:
                        print(f"      ✗ FAIL: Sample {i}: action_token_id count ({action_token_count}) != non-zero action timesteps ({action_non_zero_timesteps})")
                        action_check_passed = False
                
                if state_check_passed:
                    print(f"      ✓ PASS: All samples have matching state_token_id counts")
                if action_check_passed:
                    print(f"      ✓ PASS: All samples have matching action_token_id counts")
            else:
                print(f"      ⚠ SKIP: Preprocessor not set, cannot get token IDs")
        else:
            print(f"      ⚠ SKIP: Missing input_ids, states, or actions")
        
        # Check 4: labels 中每一条不为 -1 的第一个位置是否等于 answer_start_idx
        print(f"\n   4. Checking labels first non-ignore position matches answer_start_idx...")
        if 'labels' in first_batch and 'answer_start_idx' in first_batch:
            labels = first_batch['labels']
            answer_start_idx = first_batch['answer_start_idx']
            ignore_index = vla_dataset.preprocessor.ignore_index if vla_dataset.preprocessor is not None else -100
            
            batch_size = labels.shape[0]
            check_passed = True
            
            for i in range(batch_size):
                # Find first position where labels[i] != ignore_index
                non_ignore_mask = (labels[i] != ignore_index)
                if non_ignore_mask.any():
                    first_non_ignore_pos = non_ignore_mask.nonzero(as_tuple=False)[0, 0].item()
                    expected_start = answer_start_idx[i].item() if isinstance(answer_start_idx, torch.Tensor) else answer_start_idx[i]
                    
                    if first_non_ignore_pos != expected_start:
                        print(f"      ✗ FAIL: Sample {i}: first non-ignore position ({first_non_ignore_pos}) != answer_start_idx ({expected_start})")
                        check_passed = False
                else:
                    print(f"      ⚠ WARN: Sample {i}: All labels are ignore_index")
            
            if check_passed:
                print(f"      ✓ PASS: All samples have matching answer_start_idx")
        else:
            print(f"      ⚠ SKIP: Missing labels or answer_start_idx")
        
        print(f"\n   {'='*80}")
        print(f"   First Batch Output Complete")
        print(f"   {'='*80}\n")

        # Simple speed test: load 500 batches
        import time
        max_batches = 500
        print(f"\n   {'='*80}")
        print(f"   Speed Test: Loading {max_batches} Batches")
        print(f"   {'='*80}")
        start_time = time.perf_counter()
        loaded_batches = 0
        for batch_idx, _ in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            loaded_batches += 1
        elapsed = time.perf_counter() - start_time
        if loaded_batches > 0:
            print(f"   Loaded batches: {loaded_batches}")
            print(f"   Total time: {elapsed:.4f}s")
            print(f"   Batches/sec: {loaded_batches / elapsed:.2f}")
            print(f"   Sec/batch: {elapsed / loaded_batches:.6f}")
        else:
            print(f"   ⚠ No batches loaded (dataloader may be empty)")
        
    except Exception as e:
        print(f"   ✗ Error creating unified dataset dataloader: {e}")
        import traceback
        traceback.print_exc()

def visualize_state_action(
    dataset: LegendUnifiedDataset,
    dataset_idx: int,
    output_dir: str = "./outputs/visualization",
    skeleton_frame_interval: int = 5,
    depth_overlay_alpha: float = 0.3,
):
    """
    最终版可视化测试函数：可视化渲染 state 和 action，并将 depth 叠加到 image 上。
    
    Args:
        dataset: LegendUnifiedDataset 实例
        dataset_idx: Unified dataset 索引，用于随机抽查
        output_dir: 输出目录
        skeleton_frame_interval: 每隔多少帧渲染一个骨架
        depth_overlay_alpha: depth 叠加的透明度
    """
    import os
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查索引范围
    if dataset_idx >= len(dataset):
        raise ValueError(f"Dataset index {dataset_idx} out of range (max: {len(dataset)-1})")
    
    # 直接从 unified dataset 的 __getitem__ 获取已处理的数据
    processed_sample = dataset[dataset_idx]
    
    # 检查是否是 VLA 数据
    is_vla_data = processed_sample.get('is_vla_data', None)
    if is_vla_data is None:
        # 如果没有 is_vla_data 字段，根据 index 判断
        is_vla_data = dataset_idx < len(dataset.vla_dataset)
    else:
        # 转换为 bool（可能是 tensor）
        if hasattr(is_vla_data, 'item'):
            is_vla_data = is_vla_data.item()
        else:
            is_vla_data = bool(is_vla_data)
    
    if not is_vla_data:
        print(f"Dataset index {dataset_idx} is not VLA data (is VLM data), skipping visualization.")
        return None
    
    # 获取已处理的 states 和 actions（tensor 格式）
    states = processed_sample['states']  # [N_state, state_dim]
    actions = processed_sample['actions']  # [N_action, action_dim]
    
    # 转换为 numpy
    if isinstance(states, torch.Tensor):
        states = states.cpu().numpy()
    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()
    
    # 获取处理后的图像（pixel_values）
    pixel_values = processed_sample.get('pixel_values', None)  # [N_image, C, H, W]
    if pixel_values is None:
        raise ValueError("processed_sample does not contain 'pixel_values'")
    
    # 转换为 numpy
    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values.cpu().numpy()
    
    # 获取处理后的深度图（如果有）
    depth_values = processed_sample.get('depth_values', None)  # [N_image, 1, H, W] or None
    if depth_values is not None and isinstance(depth_values, torch.Tensor):
        depth_values = depth_values.cpu().numpy()
    
    # 获取原始样本以获取原始内参和 instruction
    vla_dataset = dataset.vla_dataset
    vla_idx = dataset_idx  # unified dataset 的前 len(vla_dataset) 个是 VLA 数据
    
    # 找到对应的 sampler 和原始索引
    curr_idx = vla_idx
    sampler_idx = None
    for i, length in enumerate(vla_dataset.sampler_lens):
        if curr_idx < length:
            sampler_idx = i
            break
        curr_idx -= length
    
    if sampler_idx is None:
        raise ValueError(f"VLA dataset index {vla_idx} out of range (max: {sum(vla_dataset.sampler_lens)-1})")
    
    # 获取原始样本（只用于获取原始内参和 instruction）
    raw_sample = vla_dataset.samplers[sampler_idx].sample_sequence(curr_idx)
    
    # 获取原始内参和图像尺寸
    raw_intrinsic = raw_sample['intrinsic'].astype(np.float32)  # [4] or [3, 3]
    raw_images = raw_sample['image'] # [N_image, H, W, 3]
    original_height, original_width = raw_images.shape[1], raw_images.shape[2]
    
    # 获取处理后的图像尺寸（从 preprocessor）
    if vla_dataset.preprocessor is not None:
        processed_image_size = vla_dataset.preprocessor.image_size
    else:
        # 如果没有 preprocessor，从 pixel_values 推断
        processed_image_size = pixel_values.shape[2]  # H or W (assuming square)
    
    # 调整内参以适应 resize 后的图像
    from src.dataset.paligemma_processing import get_resized_intrinsic
    if raw_intrinsic.shape == (4,):
        # [fx, fy, cx, cy] 格式
        intrinsic = get_resized_intrinsic(raw_intrinsic, original_width, original_height, processed_image_size)
    elif raw_intrinsic.shape == (3, 3):
        # 转换为 [fx, fy, cx, cy] 格式
        fx, fy = raw_intrinsic[0, 0], raw_intrinsic[1, 1]
        cx, cy = raw_intrinsic[0, 2], raw_intrinsic[1, 2]
        intrinsic_4d = np.array([fx, fy, cx, cy], dtype=np.float32)
        intrinsic_4d = get_resized_intrinsic(intrinsic_4d, original_width, original_height, processed_image_size)
        # 转换回 [3, 3] 格式
        intrinsic = np.array([
            [intrinsic_4d[0], 0, intrinsic_4d[2]],
            [0, intrinsic_4d[1], intrinsic_4d[3]],
            [0, 0, 1]
        ], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported intrinsic shape: {raw_intrinsic.shape}")
    
    # 反归一化图像：pixel_values * std + mean，然后转换回 [0, 255]
    from src.dataset.paligemma_processing import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
    # pixel_values 格式: [N_image, C, H, W]
    # 转换为 [N_image, H, W, C] 格式
    pixel_values_transposed = np.transpose(pixel_values, (0, 2, 3, 1))  # [N_image, H, W, C]
    
    # 反归一化
    images = pixel_values_transposed * IMAGENET_STANDARD_STD + IMAGENET_STANDARD_MEAN
    # 转换回 [0, 255] 范围
    images = np.clip(images * 255.0, 0, 255).astype(np.uint8)
    
    # 处理深度图（如果有）
    depth_images = None
    if depth_values is not None:
        # depth_values 格式: [N_image, 1, H, W]
        # 转换为 [N_image, H, W] 格式
        depth_images = depth_values[:, 0, :, :]  # [N_image, H, W]
        # depth_values 已经是归一化的，需要反归一化
        # 但这里我们直接使用原始深度图，因为 depth_values 的处理方式可能不同
        # 如果需要，可以从 raw_sample 获取原始深度图并 resize
        raw_depth_images = raw_sample.get('depth', None)
        if raw_depth_images is not None:
            # Resize 原始深度图到处理后的尺寸
            import cv2
            depth_images = []
            for i in range(raw_depth_images.shape[0]):
                depth = raw_depth_images[i]
                depth_resized = cv2.resize(depth, (processed_image_size, processed_image_size), interpolation=cv2.INTER_LINEAR)
                depth_images.append(depth_resized)
            depth_images = np.stack(depth_images)
    
    # 获取 instruction（如果有）
    instruction = None
    if 'instruction' in raw_sample:
        instruction_data = raw_sample['instruction']
        if isinstance(instruction_data, np.ndarray) and len(instruction_data) > 0:
            # 如果是数组，取第一个
            instruction = str(instruction_data[0]) if len(instruction_data.shape) > 0 else str(instruction_data)
        elif isinstance(instruction_data, (list, tuple)) and len(instruction_data) > 0:
            instruction = str(instruction_data[0])
        elif isinstance(instruction_data, str):
            instruction = instruction_data
    
    # Unnormalize states 和 actions（如果有 normalizer）
    if vla_dataset.normalizer is not None:
        if not vla_dataset.use_relative_action:
            states = vla_dataset.normalizer['motions'].unnormalize(states)
            actions = vla_dataset.normalizer['motions'].unnormalize(actions)
        else:
            states = vla_dataset.normalizer['states'].unnormalize(states)
            actions = vla_dataset.normalizer['actions'].unnormalize(actions)
    
    # 如果是 relative action，需要转换为 absolute action
    if vla_dataset.use_relative_action:
        # 将 relative action 转换为 absolute action
        # action 是相对于 state[-1] 的
        actions = get_absolute_action(states[-1], actions)
    
    # 现在 states 和 actions 已经是 unnormalized 的，且 hand 在 wrist 坐标系中
    # 只需要将 hand 从 wrist 坐标系转换到相机坐标系
    state = states
    action = actions
    
    # 合并 state 和 action 用于可视化
    # image 的最后一帧代表当前帧，所以我们需要将 state 和 action 都转换到对应的相机坐标系
    n_state = state.shape[0]
    n_action = action.shape[0]
    n_image = images.shape[0]
    
    # 提取 wrist 和 hand
    wrist_all = np.concatenate([state[:, :18], action[:, :18]], axis=0)  # [n_state + n_action, 18]
    hand_all = np.concatenate([state[:, 18:], action[:, 18:]], axis=0)  # [n_state + n_action, hand_dim]
    
    # hand_all 现在在 wrist 坐标系中（因为 process_state_action 已经转换了）
    # 需要将每一帧的 hand 转换到相机坐标系
    # 注意：extrinsic 是第一帧的相机外参，我们需要为每一帧获取对应的 extrinsic
    # 但根据代码，state 和 action 中的 wrist 已经在相机坐标系中了
    # 所以我们需要将 hand 从 wrist 坐标系转换到相机坐标系
    
    # 将 hand 从 wrist 坐标系转换到相机坐标系
    hand_all_camera = transform_hand_points_from_wrist_to_camera_frame(
        hand_all, wrist_all
    )
    
    # 解析相机内参
    if intrinsic.shape == (4,):
        # [fx, fy, cx, cy] 格式
        fx, fy, cx, cy = intrinsic
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
    elif intrinsic.shape == (3, 3):
        K = intrinsic
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
    else:
        raise ValueError(f"Unsupported intrinsic shape: {intrinsic.shape}")
    
    # 准备可视化
    # image 的最后一帧代表当前帧
    # 我们需要在每一帧图像上渲染对应的 state/action
    
    # 计算全局深度范围（用于 colorbar）
    # 深度单位从 mm 转换为米
    depth_min_global = None
    depth_max_global = None
    if depth_images is not None:
        depth_min_global = float(np.min(depth_images)) / 1000.0  # mm -> m
        depth_max_global = float(np.max(depth_images)) / 1000.0  # mm -> m
    
    # 保存所有输出路径
    output_paths = []
    
    # 投影函数
    def project_points(points_3d, K):
        """投影3D点到2D图像平面"""
        points_2d_homo = (K @ points_3d.T).T  # [N, 3]
        points_2d = points_2d_homo[:, :2] / (points_2d_homo[:, 2:3] + 1e-6)
        return points_2d
    
    # 检查点是否在图像范围内
    def is_valid_point(pt_2d, W, H):
        return 0 <= pt_2d[0] < W and 0 <= pt_2d[1] < H
    
    # 处理每个 state
    for state_idx in range(n_state):
        # 检查是否符合间隔要求
        if state_idx % skeleton_frame_interval != 0:
            continue
        
        # 找到对应的图像索引（image 的最后一帧代表当前帧）
        # image[0] 对应 state[0]，image[-1] 对应 state[-1]
        img_idx = min(state_idx, n_image - 1)
        
        # 获取当前帧图像
        img = images[img_idx].copy()  # [H, W, 3]
        H, W = img.shape[:2]
        
        # 获取对应的深度图
        depth = None
        depth_m = None  # 转换为米后的深度
        if depth_images is not None:
            depth = depth_images[img_idx]  # [H_depth, W_depth]
            # 确保 depth 和 img 的尺寸一致
            if depth.shape[:2] != img.shape[:2]:
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
            # 将深度从 mm 转换为米
            depth_m = depth / 1000.0
        
        # 获取当前帧的 wrist 和 hand
        wrist_pose = wrist_all[state_idx]  # [18]
        hand_points = hand_all_camera[state_idx]  # [hand_dim]
        
        # 提取左右手的 wrist 位置
        left_wrist_trans = wrist_pose[:3]
        right_wrist_trans = wrist_pose[3:6]
        
        # 提取左右手的 hand points（fingertips）
        hand_dim = hand_points.shape[0]
        
        # 尝试 reshape 为左右手各 [n_points_per_hand, 3]
        if hand_dim % 6 == 0:
            n_points_per_hand = hand_dim // 2 // 3
            left_hand_points = hand_points[:hand_dim//2].reshape(n_points_per_hand, 3)
            right_hand_points = hand_points[hand_dim//2:].reshape(n_points_per_hand, 3)
        else:
            hand_ndim = vla_dataset.hand_ndim
            if hand_dim == hand_ndim * 2:
                dim_per_point = hand_ndim // 5 if hand_ndim >= 5 else 3
                n_points_per_hand = hand_ndim // dim_per_point if dim_per_point > 0 else 5
                left_hand_points = hand_points[:hand_ndim].reshape(n_points_per_hand, dim_per_point)[:, :3]
                right_hand_points = hand_points[hand_ndim:].reshape(n_points_per_hand, dim_per_point)[:, :3]
            else:
                print(f"Warning: Cannot reshape hand_points with shape {hand_points.shape}, hand_ndim={vla_dataset.hand_ndim}, hand_dim={hand_dim}")
                continue
        
        # 投影到图像平面
        left_wrist_2d = project_points(left_wrist_trans.reshape(1, 3), K)[0]
        right_wrist_2d = project_points(right_wrist_trans.reshape(1, 3), K)[0]
        left_hand_2d = project_points(left_hand_points, K)
        right_hand_2d = project_points(right_hand_points, K)
        
        # 创建 matplotlib 图像用于添加 colorbar
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # 叠加 depth（如果有）
        if depth_m is not None:
            # 将 depth 转换为彩色叠加（使用米为单位）
            if depth_max_global > depth_min_global:
                depth_normalized = (depth_m - depth_min_global) / (depth_max_global - depth_min_global + 1e-6)
            else:
                depth_normalized = np.zeros_like(depth_m)
            
            # 使用 colormap 转换为彩色（反转 colormap 使近处为红色，远处为蓝色）
            depth_colormap = cm.jet_r(depth_normalized)[:, :, :3]  # [H, W, 3]
            depth_colormap = (depth_colormap * 255).astype(np.uint8)
            
            # 确保 depth_colormap 和 img 的尺寸一致
            if depth_colormap.shape[:2] != img.shape[:2]:
                depth_colormap = cv2.resize(depth_colormap, (W, H), interpolation=cv2.INTER_LINEAR)
            
            # 叠加 depth
            img = cv2.addWeighted(img, 1 - depth_overlay_alpha, depth_colormap, depth_overlay_alpha, 0)
        
        # 绘制左手骨架（减小粗细）- 改进版本，即使手腕不在范围内也绘制可见的指尖
        left_color = (0, 255, 0)  # 绿色 (BGR)
        left_wrist_valid = is_valid_point(left_wrist_2d, W, H)
        if left_wrist_valid:
            cv2.circle(img, tuple(left_wrist_2d.astype(int)), 3, left_color, -1)
            cv2.circle(img, tuple(left_wrist_2d.astype(int)), 3, (0, 0, 0), 1)
        
        for i in range(left_hand_2d.shape[0]):
            tip_2d = left_hand_2d[i]
            tip_valid = is_valid_point(tip_2d, W, H)
            if left_wrist_valid and tip_valid:
                # 绘制从手腕到指尖的线
                cv2.line(img, 
                        tuple(left_wrist_2d.astype(int)), 
                        tuple(tip_2d.astype(int)), 
                        left_color, 1)
            if tip_valid:
                # 绘制指尖
                cv2.circle(img, tuple(tip_2d.astype(int)), 2, left_color, -1)
                cv2.circle(img, tuple(tip_2d.astype(int)), 2, (0, 0, 0), 1)
        
        # 绘制右手骨架（减小粗细）- 改进版本，即使手腕不在范围内也绘制可见的指尖
        right_color = (255, 0, 0)  # 蓝色 (BGR)
        right_wrist_valid = is_valid_point(right_wrist_2d, W, H)
        if right_wrist_valid:
            cv2.circle(img, tuple(right_wrist_2d.astype(int)), 3, right_color, -1)
            cv2.circle(img, tuple(right_wrist_2d.astype(int)), 3, (0, 0, 0), 1)
        
        for i in range(right_hand_2d.shape[0]):
            tip_2d = right_hand_2d[i]
            tip_valid = is_valid_point(tip_2d, W, H)
            if right_wrist_valid and tip_valid:
                # 绘制从手腕到指尖的线
                cv2.line(img, 
                        tuple(right_wrist_2d.astype(int)), 
                        tuple(tip_2d.astype(int)), 
                        right_color, 1)
            if tip_valid:
                # 绘制指尖
                cv2.circle(img, tuple(tip_2d.astype(int)), 2, right_color, -1)
                cv2.circle(img, tuple(tip_2d.astype(int)), 2, (0, 0, 0), 1)
        
        # 添加文本标注
        label = f"Frame {img_idx} (State {state_idx})"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # 显示图像
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(f"State {state_idx}", fontsize=14, fontweight='bold')
        
        # 添加深度 colorbar（如果有深度图，使用米为单位）
        if depth_m is not None and depth_max_global > depth_min_global:
            # 使用 ScalarMappable 创建 colorbar，确保与深度叠加使用相同的 colormap
            # 深度叠加使用 cm.jet_r（近处红色，远处蓝色），所以 colorbar 也要使用 jet_r
            from matplotlib.cm import ScalarMappable
            # 使用 cm.jet_r 而不是字符串 'jet_r'，确保完全一致
            sm = ScalarMappable(cmap=cm.jet_r, norm=plt.Normalize(vmin=depth_min_global, vmax=depth_max_global))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Depth (m)', rotation=270, labelpad=15, fontsize=12)
        
        # 添加 instruction 文本（如果有）
        if instruction is not None:
            # 在图像顶部添加 instruction
            instruction_text = f"Instruction: {instruction}"
            # 限制文本长度，避免太长，手动换行
            max_chars_per_line = 80
            words = instruction_text.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + " " + word) <= max_chars_per_line:
                    current_line += (" " + word if current_line else word)
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            instruction_text = "\n".join(lines)
            
            ax.text(0.02, 0.98, instruction_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1))
        
        # 保存图像
        output_path = os.path.join(output_dir, f"visualization_dataset_idx_{dataset_idx:05d}_state_{state_idx:03d}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        output_paths.append(output_path)
    
    # 处理每个 action
    for action_idx in range(n_action):
        # 检查是否符合间隔要求
        if action_idx % skeleton_frame_interval != 0:
            continue
        
        # 找到对应的图像索引（通常使用最后一帧）
        img_idx = min(n_state - 1 + action_idx, n_image - 1)
        
        # 获取当前帧图像
        img = images[img_idx].copy()  # [H, W, 3]
        H, W = img.shape[:2]
        
        # 获取对应的深度图
        depth = None
        depth_m = None  # 转换为米后的深度
        if depth_images is not None:
            depth = depth_images[img_idx]
            if depth.shape[:2] != img.shape[:2]:
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
            # 将深度从 mm 转换为米
            depth_m = depth / 1000.0
        
        # 获取当前帧的 wrist 和 hand
        wrist_pose = wrist_all[n_state + action_idx]  # [18]
        hand_points = hand_all_camera[n_state + action_idx]  # [hand_dim]
        
        # 提取左右手的 wrist 位置
        left_wrist_trans = wrist_pose[:3]
        right_wrist_trans = wrist_pose[3:6]
        
        # 提取左右手的 hand points
        hand_dim = hand_points.shape[0]
        
        if hand_dim % 6 == 0:
            n_points_per_hand = hand_dim // 2 // 3
            left_hand_points = hand_points[:hand_dim//2].reshape(n_points_per_hand, 3)
            right_hand_points = hand_points[hand_dim//2:].reshape(n_points_per_hand, 3)
        else:
            hand_ndim = vla_dataset.hand_ndim
            if hand_dim == hand_ndim * 2:
                dim_per_point = hand_ndim // 5 if hand_ndim >= 5 else 3
                n_points_per_hand = hand_ndim // dim_per_point if dim_per_point > 0 else 5
                left_hand_points = hand_points[:hand_ndim].reshape(n_points_per_hand, dim_per_point)[:, :3]
                right_hand_points = hand_points[hand_ndim:].reshape(n_points_per_hand, dim_per_point)[:, :3]
            else:
                print(f"Warning: Cannot reshape hand_points with shape {hand_points.shape}, hand_ndim={vla_dataset.hand_ndim}, hand_dim={hand_dim}")
                continue
        
        # 投影到图像平面
        left_wrist_2d = project_points(left_wrist_trans.reshape(1, 3), K)[0]
        right_wrist_2d = project_points(right_wrist_trans.reshape(1, 3), K)[0]
        left_hand_2d = project_points(left_hand_points, K)
        right_hand_2d = project_points(right_hand_points, K)
        
        # 创建 matplotlib 图像用于添加 colorbar
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # 叠加 depth（如果有）
        if depth_m is not None:
            if depth_max_global > depth_min_global:
                depth_normalized = (depth_m - depth_min_global) / (depth_max_global - depth_min_global + 1e-6)
            else:
                depth_normalized = np.zeros_like(depth_m)
            
            depth_colormap = cm.jet(depth_normalized)[:, :, :3]
            depth_colormap = (depth_colormap * 255).astype(np.uint8)
            
            if depth_colormap.shape[:2] != img.shape[:2]:
                depth_colormap = cv2.resize(depth_colormap, (W, H), interpolation=cv2.INTER_LINEAR)
            
            img = cv2.addWeighted(img, 1 - depth_overlay_alpha, depth_colormap, depth_overlay_alpha, 0)
        
        # 绘制左手骨架（减小粗细）- 改进版本，即使手腕不在范围内也绘制可见的指尖
        left_color = (0, 255, 0)  # 绿色 (BGR)
        left_wrist_valid = is_valid_point(left_wrist_2d, W, H)
        if left_wrist_valid:
            cv2.circle(img, tuple(left_wrist_2d.astype(int)), 3, left_color, -1)
            cv2.circle(img, tuple(left_wrist_2d.astype(int)), 3, (0, 0, 0), 1)
        
        for i in range(left_hand_2d.shape[0]):
            tip_2d = left_hand_2d[i]
            tip_valid = is_valid_point(tip_2d, W, H)
            if left_wrist_valid and tip_valid:
                # 绘制从手腕到指尖的线
                cv2.line(img, 
                        tuple(left_wrist_2d.astype(int)), 
                        tuple(tip_2d.astype(int)), 
                        left_color, 1)
            if tip_valid:
                # 绘制指尖
                cv2.circle(img, tuple(tip_2d.astype(int)), 2, left_color, -1)
                cv2.circle(img, tuple(tip_2d.astype(int)), 2, (0, 0, 0), 1)
        
        # 绘制右手骨架（减小粗细）- 改进版本，即使手腕不在范围内也绘制可见的指尖
        right_color = (255, 0, 0)  # 蓝色 (BGR)
        right_wrist_valid = is_valid_point(right_wrist_2d, W, H)
        if right_wrist_valid:
            cv2.circle(img, tuple(right_wrist_2d.astype(int)), 3, right_color, -1)
            cv2.circle(img, tuple(right_wrist_2d.astype(int)), 3, (0, 0, 0), 1)
        
        for i in range(right_hand_2d.shape[0]):
            tip_2d = right_hand_2d[i]
            tip_valid = is_valid_point(tip_2d, W, H)
            if right_wrist_valid and tip_valid:
                # 绘制从手腕到指尖的线
                cv2.line(img, 
                        tuple(right_wrist_2d.astype(int)), 
                        tuple(tip_2d.astype(int)), 
                        right_color, 1)
            if tip_valid:
                # 绘制指尖
                cv2.circle(img, tuple(tip_2d.astype(int)), 2, right_color, -1)
                cv2.circle(img, tuple(tip_2d.astype(int)), 2, (0, 0, 0), 1)
        
        # 添加文本标注
        label = f"Frame {img_idx} (Action {action_idx})"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # 显示图像
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(f"Action {action_idx}", fontsize=14, fontweight='bold')
        
        # 添加深度 colorbar（如果有深度图，使用米为单位）
        if depth_m is not None and depth_max_global > depth_min_global:
            # 使用 ScalarMappable 创建 colorbar，而不是用 alpha=0 的 imshow
            # 反转 colormap 使近处（小值）为红色，远处（大值）为蓝色
            from matplotlib.cm import ScalarMappable
            sm = ScalarMappable(cmap='jet_r', norm=plt.Normalize(vmin=depth_min_global, vmax=depth_max_global))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Depth (m)', rotation=270, labelpad=15, fontsize=12)
        
        # 添加 instruction 文本（如果有）
        if instruction is not None:
            # 在图像顶部添加 instruction
            instruction_text = f"Instruction: {instruction}"
            # 限制文本长度，避免太长，手动换行
            max_chars_per_line = 80
            words = instruction_text.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + " " + word) <= max_chars_per_line:
                    current_line += (" " + word if current_line else word)
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            instruction_text = "\n".join(lines)
            
            ax.text(0.02, 0.98, instruction_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1))
        
        # 保存图像
        output_path = os.path.join(output_dir, f"visualization_dataset_idx_{dataset_idx:05d}_action_{action_idx:03d}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        output_paths.append(output_path)
    
    # 打印保存信息
    print(f"可视化结果已保存到: {output_dir}")
    print(f"  - Unified dataset 索引: {dataset_idx}")
    print(f"  - VLA dataset 索引: {vla_idx}")
    print(f"  - State 帧数: {n_state}")
    print(f"  - Action 帧数: {n_action}")
    print(f"  - Image 帧数: {n_image}")
    print(f"  - 骨架渲染间隔: {skeleton_frame_interval} 帧")
    print(f"  - 共保存 {len(output_paths)} 张图像")
    if depth_images is not None:
        print(f"  - 深度范围: [{depth_min_global:.3f}, {depth_max_global:.3f}] m")
    
    return output_paths[0] if output_paths else None


def test_visualize_state_action(dataset_idx: int = 0):
    """
    测试可视化函数：可视化渲染 state 和 action。
    
    Args:
        dataset_idx: Unified dataset 索引，用于随机抽查
    """
    from omegaconf import OmegaConf
    import hydra
    import pathlib
    import os
    import pickle
    from datetime import datetime
    
    # Register eval resolver for config
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    
    # Register now resolver for datetime formatting (used by Hydra)
    def now_resolver(format_str: str) -> str:
        """Resolver for ${now:format} interpolation."""
        return datetime.now().strftime(format_str)
    OmegaConf.register_new_resolver("now", now_resolver, replace=True)
    
    # Register hydra resolver (returns empty string for non-hydra contexts)
    def hydra_resolver(key: str) -> str:
        """Resolver for ${hydra:key} interpolation. Returns empty string in test context."""
        return ""
    OmegaConf.register_new_resolver("hydra", hydra_resolver, replace=True)
    
    # Load config file
    config_path = pathlib.Path(__file__).parent.parent.parent / "src" / "config" / "experiment" / "pretrain_legendvla_deepspeed.yaml"
    print(f"Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # Resolve config to evaluate all ${eval:}, ${now:}, and ${hydra:} expressions
    try:
        OmegaConf.resolve(cfg)
    except Exception as e:
        print(f"   Warning: Some config values could not be resolved: {e}")
        print("   Continuing with unresolved config (this is OK for testing)...")
    
    print("\n" + "="*80)
    print("Testing State and Action Visualization")
    print("="*80)
    
    # Create Unified Dataset
    print("\n1. Creating LegendUnifiedDataset...")
    try:
        vla_dataset = hydra.utils.instantiate(cfg.dataset.vla_dataset)
        print(f"   ✓ VLA Dataset created successfully")
        print(f"   - VLA Dataset length: {len(vla_dataset)}")
        
        # Set preprocessor for VLA dataset
        print(f"\n   Setting preprocessor for VLA dataset...")
        try:
            vla_processor = hydra.utils.instantiate(cfg.vla_processor)
            vla_dataset.set_preprocessor(vla_processor)
            print(f"   ✓ Preprocessor set successfully")
        except Exception as e:
            print(f"   ⚠ Warning: Could not set preprocessor: {e}")
        
        # Load normalizer from config
        print(f"\n   Loading normalizer from config...")
        try:
            normalizer_path = cfg.training.normalizer_path
            if normalizer_path is not None and os.path.exists(normalizer_path):
                normalizer = pickle.load(open(normalizer_path, 'rb'))
                vla_dataset.set_normalizer(normalizer)
                print(f"   ✓ Normalizer loaded from {normalizer_path}")
            else:
                print(f"   ⚠ Warning: Normalizer path not found: {normalizer_path}")
                print(f"   Computing normalizer from dataset...")
                normalizer = vla_dataset.get_normalizer()
                print(f"   ✓ Normalizer computed successfully")
        except Exception as e:
            print(f"   ⚠ Warning: Could not load normalizer: {e}")
            import traceback
            traceback.print_exc()
        
        # Create VLM Dataset (if available)
        vlm_dataset = None
        try:
            vlm_dataset = hydra.utils.instantiate(cfg.dataset.vlm_dataset)
            print(f"   ✓ VLM Dataset created successfully")
            print(f"   - VLM Dataset length: {len(vlm_dataset)}")
            
            # Set preprocessor for VLM dataset
            print(f"\n   Setting preprocessor for VLM dataset...")
            try:
                vlm_processor = hydra.utils.instantiate(cfg.vlm_processor)
                vlm_dataset.set_preprocessor(vlm_processor)
                print(f"   ✓ Preprocessor set successfully")
            except Exception as e:
                print(f"   ⚠ Warning: Could not set preprocessor: {e}")
        except Exception as e:
            print(f"   ⚠ Warning: Could not create VLM dataset: {e}")
        
        # Create Unified Dataset
        unified_dataset = LegendUnifiedDataset(
            vla_dataset=vla_dataset,
            vlm_dataset=vlm_dataset,
        )
        print(f"   ✓ Unified Dataset created successfully")
        print(f"   - Unified Dataset length: {len(unified_dataset)}")
        
        # Test visualization
        print(f"\n2. Visualizing unified dataset index {dataset_idx}...")
        try:
            output_path = visualize_state_action(
                dataset=unified_dataset,
                dataset_idx=dataset_idx,
                output_dir="./outputs/visualization",
                skeleton_frame_interval=5,
                depth_overlay_alpha=0.3,
            )
            if output_path is not None:
                print(f"   ✓ Visualization completed successfully")
                print(f"   - Output path: {output_path}")
            else:
                print(f"   ⚠ Visualization skipped (not VLA data)")
        except Exception as e:
            print(f"   ✗ Error during visualization: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"   ✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LegendVLA dataset")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "visualize"],
        default="test",
        help="Mode: 'test' for dataset loading test, 'visualize' for visualization test"
    )
    parser.add_argument(
        "--dataset_idx",
        type=int,
        default=0,
        help="Dataset index for visualization (only used when mode='visualize')"
    )
    
    args = parser.parse_args()
    
    if args.mode == "visualize":
        test_visualize_state_action(dataset_idx=args.dataset_idx)
    else:
        test_dataset_loading()
