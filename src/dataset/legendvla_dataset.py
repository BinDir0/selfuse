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
from PIL import Image
import torch.nn.utils.rnn as rnn_utils
from datasets import load_dataset
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
    def __init__(self,
            zarr_paths,
            horizon=1,
            pad_before=0,
            pad_after=0,
            shape_meta=None,
            seed=42,
            val_ratio=0.0,
            history=30,
            objective=None,
            normalizer_dataloader_cfg=dict(),
            use_relative_action=False,
            max_train_episodes=None,
            mode = 'train',
            depth_clip_range=None,
        ):
        self.zarr_paths = zarr_paths
        self.preprocessor = None
        self.history = history
        self.objective = objective
        self.normalizer_dataloader_cfg = normalizer_dataloader_cfg
        self.use_relative_action = use_relative_action
        self.max_train_episodes = max_train_episodes
        self.normalizer = None
        self.depth_clip_range = depth_clip_range
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.shape_meta = shape_meta
        self.motion_type = shape_meta['obs']['state']['type']
        self.hand_ndim = shape_meta['obs']['state']['hand']['shape'][-1] // 2 # per hand pca ncomponents
        self.n_obs_image_steps = shape_meta['obs']['rgb']['horizon']
        self.n_obs_state_steps = shape_meta['obs']['state']['horizon']

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
                keys=['image', 'depth', 'state', 'instruction', 'instruction_num', 'action', 'extrinsic', 'intrinsic', 'presence'], 
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
            self.image_history = history + 1 if self.n_obs_image_steps > 1 else 1
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=train_mask,
                key_first_k=dict(image=self.image_history, depth=self.image_history))
            self.samplers.append(sampler)
            
            # Record sampler length
            self.sampler_lens.append(len(sampler))

        if zarr_paths is not None and zarr_paths[0].get('weight', None) is not None:
            weights = [path['weight'] for path in zarr_paths]
            super().__init__(weights, self.sampler_lens)
        else:
            super().__init__()

    def get_validation_dataset(self):
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
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=~self.train_masks[i],
                key_first_k=dict(image=self.image_history, depth=self.image_history))
            val_set.samplers.append(sampler)
            val_set.train_masks.append(~self.train_masks[i])
            val_set.sampler_lens.append(len(sampler))
            
        return val_set

    def _sample_to_data(self, sample):
        # Select data keys based on motion_type
        state, action, action_valid_mask, state_presence, action_presence = process_state_action(
            wrist_state = sample['state/wrist'].astype(np.float32), 
            hand_state = sample[f'state/{self.motion_type}'].astype(np.float32), 
            wrist_action = sample['action/wrist'].astype(np.float32), 
            hand_action = sample[f'action/{self.motion_type}'].astype(np.float32), 
            extrinsic = sample['extrinsic'].astype(np.float32).reshape(-1, 4, 4), # [Horizon, 16] -> [Horizon, 4, 4]
            presence = sample['presence'], 
            normalizer = self.normalizer, 
            hand_ndim = self.hand_ndim, 
            history = self.history, 
            n_obs_state_steps = self.n_obs_state_steps, 
            motion_type = self.motion_type,
            use_relative_action = self.use_relative_action,
        )
        image, depth_images = process_image(
            sample['image'], 
            self.history, 
            self.n_obs_image_steps,
            sample.get('depth', None), 
            self.aug_transform, 
            self.depth_clip_range,
        )

        intrinsic = sample['intrinsic'][self.history].astype(np.float32)
        instruction = sample['instruction'][self.history]
        instruction_num = sample['instruction_num'][self.history]
        # sample a random instruction from the candidate instructions
        idx = np.random.randint(0, instruction_num)
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

        data = {
            'input_ids': processed_results['input_ids'],
            'answer_start_idx': processed_results['answer_start_idx'],
            'attention_mask': processed_results['attention_mask'],
            'pixel_values': processed_results['pixel_values'], 
            'states': state,
        }
        # Add depth_values if available
        if 'depth_values' in processed_results:
            data['depth_values'] = processed_results['depth_values']
        if self.objective != "train_ar":
            data['actions'] = action
            data['actions_valid_mask'] = action_valid_mask
        if self.objective != "train_flow":
            data['labels'] = processed_results['labels']
        return data

    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer

    def get_normalizer(self):
        # Merge all data
        normalizer_dataset = LegendVLALowLevelDataset(
            zarr_paths=self.zarr_paths,
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            shape_meta=self.shape_meta,
            history=self.history,
            max_train_episodes=self.max_train_episodes, 
            return_numpy=True, 
            use_relative_action=self.use_relative_action,
        )
        normalizer = get_normalizer(self.normalizer_dataloader_cfg, normalizer_dataset)
        self.normalizer = normalizer

        return normalizer

    def get_collator(self):
        assert self.preprocessor is not None, "Preprocessor is not set"
        return LegendVLDataCollator(
            pad_token_id=self.preprocessor.tokenizer.pad_token_id,
            ignore_index=self.preprocessor.ignore_index,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find corresponding sampler
        # For validation, we need to directly sample from the validation set
        curr_idx = idx
        dataset_idx = 0
        for i, length in enumerate(self.sampler_lens):
            if curr_idx < length:
                sample = self.samplers[i].sample_sequence(curr_idx)
                dataset_idx = i
                break
            curr_idx -= length
        
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def __len__(self):
        return sum(self.sampler_lens)

class LegendVLMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_paths,
        split='train',
        cache_dir=None,
        weights=[0.5, 0.5, 0.5],
        seed=42,
        val_ratio=0.0,
        mode='train',
    ):
        super().__init__()
        self.dataset_paths = dataset_paths
        self.split = split
        self.weights = weights
        self.cache_dir = cache_dir
        self.datasets = None
        self.preprocessor = None
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
        if isinstance(dataset_paths, str):
            data_files = f"{dataset_paths}/*.parquet"
        else:
            data_files = [f"{p}/*.parquet" for p in dataset_paths]

        self.datasets = load_dataset(
            "parquet",
            data_files=data_files,
            split=self.split,
            cache_dir=self.cache_dir,
        )
        if val_ratio > 0:
            datasets_splits = self.datasets.train_test_split(test_size=val_ratio, seed=seed)
            self.train_datasets = datasets_splits['train']
            self.val_datasets = datasets_splits['test']
        else:
            self.train_datasets = self.datasets
            self.val_datasets = None

    def get_validation_dataset(self):
        if self.val_datasets is None:
            return None
        val_copy = copy.copy(self)
        val_copy.mode = 'val'
        val_copy.aug_transform = None
        val_copy.train_datasets = self.val_datasets
        return val_copy
    
    def _sample_to_data(self, sample, idx):
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
            augmented_np = np.array(augmented_pil)
            augmented_images.append(augmented_np)
        images_to_process = np.stack(augmented_images)
        # Process all images in batch
        processed_results = self.preprocessor(
            images=images_to_process, 
            text=question, 
            target=answer, 
            mode=self.mode
        )

        data = {
            'input_ids': processed_results['input_ids'],
            'labels': processed_results['labels'] ,
            'attention_mask': processed_results['attention_mask'] ,
            'pixel_values': processed_results['pixel_values'], 
            'answer_start_idx': processed_results['answer_start_idx'],
        }
        return data

    def get_collator(self):
        assert self.preprocessor is not None, "Preprocessor is not set"
        return LegendVLDataCollator(
            pad_token_id=self.preprocessor.tokenizer.pad_token_id,
            ignore_index=self.preprocessor.ignore_index,
        )

    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find corresponding sampler
        sample = self.train_datasets[idx]
        data = self._sample_to_data(sample, idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def __len__(self):
        return len(self.train_datasets)

class LegendUnifiedDataset(torch.utils.data.Dataset):
    def __init__(self,
        vla_dataset: LegendVLADataset,
        vlm_dataset: LegendVLMDataset = None,
    ):
        super().__init__()
        self.vla_dataset = vla_dataset
        self.vlm_dataset = vlm_dataset
        self.shape_meta = None
        
        print(f"LegendUnifiedDataset initialized with {len(self.vla_dataset)} VLA samples")
        if self.vlm_dataset is not None:
            print(f"LegendUnifiedDataset initialized with {len(self.vlm_dataset)} VLM samples")

    def get_collator(self):
        return LegendVLDataCollator()

    def get_sampler(
        self, 
        batch_size: int, 
        vla_ratio: float = 1/8, 
        shuffle: bool = True, 
        seed: int = 42, 
        drop_last: bool = False,
    ): 
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
        return LegendUnifiedDataset(
            vla_dataset=self.vla_dataset.get_validation_dataset(),
            vlm_dataset=self.vlm_dataset.get_validation_dataset() if self.vlm_dataset is not None else None, 
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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
            sample['states'] = torch.zeros(self.shape_meta['states'])
            sample['actions'] = torch.zeros(self.shape_meta['actions'])
            sample['actions_valid_mask'] = torch.zeros(self.shape_meta['actions'], dtype=torch.bool)
            return sample
        else:
            raise ValueError("No dataset to get item from")

    def __len__(self):
        return len(self.vla_dataset) + len(self.vlm_dataset) if self.vlm_dataset is not None else len(self.vla_dataset)

class LegendVLALowLevelDataset(BaseLowdimDataset):
    def __init__(
        self,
        zarr_paths,
        horizon=1,
        pad_before=0,
        pad_after=0,
        shape_meta=None,
        history=30,
        seed=42,
        val_ratio=0.0,
        dims=None, 
        max_train_episodes=None,
        return_numpy=True, # whether to return numpy arrays
        normalizer_dataloader_cfg=None,
        use_relative_action=False,
        debug=False, 
    ):
        super().__init__()
        self.history = history
        # Initialize storage lists
        self.replay_buffers = []
        self.train_masks = []
        self.samplers = []
        self.sampler_lens = []
        
        # Process each zarr file
        for zarr_path in zarr_paths:
            # Create replay buffer
            replay_buffer = StreamingReplayBuffer.copy_from_path(
                zarr_path['path'], keys=['state', 'action', 'extrinsic', 'presence'], lazy_load=False)
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
                sequence_length=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=train_mask,
                key_first_k=dict())
            self.samplers.append(sampler)
            
            # Record sampler length
            self.sampler_lens.append(len(sampler))

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.shape_meta = shape_meta
        self.motion_type = shape_meta['obs']['state']['type']
        self.hand_ndim = shape_meta['obs']['state']['hand']['shape'][-1] // 2 # per hand pca ncomponents
        self.n_obs_state_steps = shape_meta['obs']['state']['horizon']
        self.dims = dims
        self.return_numpy = return_numpy
        self.normalizer = None
        self.normalizer_dataloader_cfg = normalizer_dataloader_cfg
        self.debug = debug
        self.use_relative_action = use_relative_action

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.samplers = []
        val_set.train_masks = []
        val_set.sampler_lens = []

        for i, replay_buffer in enumerate(self.replay_buffers):
            # Create validation set sampler
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=~self.train_masks[i],
                key_first_k=dict())
            val_set.samplers.append(sampler)
            val_set.train_masks.append(~self.train_masks[i])
            val_set.sampler_lens.append(len(sampler))
            
        return val_set

    def _sample_to_data(self, sample):
        # Select data keys based on motion_type
        state, action, _, _, _ = process_state_action(
            wrist_state = sample['state/wrist'].astype(np.float32), 
            hand_state = sample[f'state/{self.motion_type}'].astype(np.float32), 
            wrist_action = sample['action/wrist'].astype(np.float32), 
            hand_action = sample[f'action/{self.motion_type}'].astype(np.float32), 
            extrinsic = sample['extrinsic'].astype(np.float32).reshape(-1, 4, 4), # [Horizon, 16] -> [Horizon, 4, 4]
            presence = sample['presence'], 
            normalizer = self.normalizer, 
            hand_ndim = self.hand_ndim, 
            history = self.history, 
            n_obs_state_steps = self.n_obs_state_steps,
            motion_type = self.motion_type,
            use_relative_action = self.use_relative_action,
        )
        if self.dims is not None:
            dim_slice = slice(self.dims[0], self.dims[1])
            state = state[:, dim_slice]
            action = action[:, dim_slice]

        data = {
            'states': state,
            'actions': action,
        }
        return data

    def get_normalizer(self):
        self.normalizer = get_normalizer(self.normalizer_dataloader_cfg, self)
        return self.normalizer

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer

    def get_collator(self):
        return BaseDataCollator()

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
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
        if self.debug: 
            data.update({'idx': idx})
        return data

    def __len__(self):
        return sum(self.sampler_lens)


class LegendVLDataCollator(BaseDataCollator):
    def __init__(self, pad_token_id: int = 0, ignore_index: int = -100):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

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
            padding_value=self.pad_token_id
        )
        batch["labels"] = rnn_utils.pad_sequence(
            labels_batch,
            batch_first=True,
            padding_value=self.ignore_index
        )
        batch["attention_mask"] = (batch["input_ids"] != self.pad_token_id).long()
        depth_values = []
        depth_ids = []
        for idx, item in enumerate(data_list):
            if 'depth_values' in item:
                depth_values.append(item['depth_values'])
                depth_ids.append(idx)
        if len(depth_values) > 0:
            batch['depth_values'] = torch.stack(depth_values, dim=0)
            depth_ids_tensor = torch.ones(len(data_list), dtype=torch.int32) * self.ignore_index
            for i, idx in enumerate(depth_ids): 
                depth_ids_tensor[idx] = i
            batch['depth_ids'] = depth_ids_tensor
        for key in data_list[0].keys():
            if key not in ['input_ids', 'attention_mask', 'labels', 'depth_values']:
                batch[key] = torch.stack([item[key] for item in data_list])

        return batch

class BaseDataCollator(BaseDataCollator):
    def __init__(self):
        super().__init__()

    def __call__(self, data_list):
        """
        DataLoader will pass a list of samples from the Dataset to this function.
        Args:
            data_list: a list, where each element is the return value of the Dataset's __getitem__ method.
               e.g., [{'state/hand': np.ndarray, 'action/hand': np.ndarray}, {'state/hand': np.ndarray, 'action/hand': np.ndarray}, ...]
        Returns:
            A dictionary with the keys the same as the return value of the Dataset's __getitem__ method.
        """
        batch = {}
        for key in data_list[0].keys():
            if isinstance(data_list[0][key], torch.Tensor): # tensor
                batch[key] = torch.stack([item[key] for item in data_list], axis=0)
            elif isinstance(data_list[0][key], np.ndarray): # numpy
                batch[key] = np.stack([item[key] for item in data_list], axis=0)
            else: # list
                batch[key] = [item[key] for item in data_list]
        return batch


def get_presence_value(state_presence, action_presence, action, state, hand_ndim):
    action_valid_mask = np.zeros_like(action, dtype=bool)

    state_presence_left = (state_presence & 1) == True
    action_presence_left = (action_presence & 1) == True
    action_valid_mask[action_presence_left, :3] = True
    action_valid_mask[action_presence_left, 6:12] = True
    action_valid_mask[action_presence_left, 18:18+hand_ndim] = True
    action[~action_presence_left, :3] = state[~state_presence_left, :3] = 0
    action[~action_presence_left, 6:12] = state[~state_presence_left, 6:12] = np.array([1, 0, 0, 0, 1, 0])
    action[~action_presence_left, 18:18+hand_ndim] = state[~state_presence_left, 18:18+hand_ndim] = 0

    state_presence_right = (state_presence >> 1) == True
    action_presence_right = (action_presence >> 1) == True
    action_valid_mask[action_presence_right, 3:6] = True
    action_valid_mask[action_presence_right, 12:18] = True
    action_valid_mask[action_presence_right, 18+hand_ndim:18+hand_ndim*2] = True
    action[~action_presence_right, 3:6] = state[~state_presence_right, 3:6] = 0
    action[~action_presence_right, 12:18] = state[~state_presence_right, 12:18] = np.array([1, 0, 0, 0, 1, 0])
    action[~action_presence_right, 18+hand_ndim:18+hand_ndim*2] = state[~state_presence_right, 18+hand_ndim:18+hand_ndim*2] = 0
    
    return state, action, action_valid_mask

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
    presence, 
    hand_ndim, 
    history, 
    n_obs_state_steps, 
    normalizer : Optional[LinearNormalizer] = None, 
    motion_type = 'mano',
    use_relative_action = False,
):
    '''
    Args:
        wrist_state: np.ndarray, shape: [N, wrist_dim]
        hand_state: np.ndarray, shape: [N, all_hand_dim]
        wrist_action: np.ndarray, shape: [N, wrist_dim]
        hand_action: np.ndarray, shape: [N, all_hand_dim]
        extrinsic: np.ndarray, shape: [N, 4, 4]
        presence: np.ndarray, shape: [N]
        hand_ndim: int
        history: int
        n_obs_state_steps: int
        normalizer: Optional[LinearNormalizer]
        motion_type: str, 'mano' or 'keypoint'
    Returns:
        state: np.ndarray, shape: [T, wrist_dim + hand_dim]
        action: np.ndarray, shape: [H, wrist_dim + hand_dim]
        action_valid_mask: np.ndarray, shape: [H, wrist_dim + hand_dim]
        state_presence: np.ndarray, shape: [H]
        action_presence: np.ndarray, shape: [H]
    '''

    step = history // n_obs_state_steps
    state_slice = [history - i * step for i in range(0, n_obs_state_steps)]
    state_slice = state_slice[::-1]
    # use first self.hand_ndim components of hand state and action
    all_hand_ndim = hand_state.shape[-1] // 2 # per hand dims, i.e. 45 in MANO hand params
    hand_state = np.concatenate([
        hand_state[state_slice, :hand_ndim], 
        hand_state[state_slice, all_hand_ndim:all_hand_ndim + hand_ndim]
    ], axis=-1)
    hand_action = np.concatenate([
        hand_action[history:, :hand_ndim], 
        hand_action[history:, all_hand_ndim:all_hand_ndim + hand_ndim]
    ], axis=-1)
    processed_wrist_state = wrist_state[state_slice]
    processed_wrist_action = wrist_action[history:]

    if motion_type == 'fingertips': 
        # TODO: We can try transform the fingertips to the camera coordinate system or wrist frame coordinate system
        processed_hand_state = transform_hand_points_to_wrist_frame(hand_state, processed_wrist_state)
        processed_hand_state = processed_hand_state.reshape(hand_state.shape)
        processed_hand_action = transform_hand_points_to_wrist_frame(hand_action, processed_wrist_action)
        processed_hand_action = processed_hand_action.reshape(hand_action.shape)
    elif motion_type == 'mano':
        processed_hand_state = hand_state
        processed_hand_action = hand_action
    else:
        raise ValueError(f"Unsupported motion type: {motion_type}")

    # transform the wrist state and action to the camera coordinate system
    processed_wrist_state = transform_wrist_to_target_frame(processed_wrist_state, extrinsic[history])
    processed_wrist_action = transform_wrist_to_target_frame(processed_wrist_action, extrinsic[history])

    # use delta of wrist translation and hand mano params as action
    state_presence = presence[state_slice]
    action_presence = presence[history:]
    processed_state = np.concatenate([processed_wrist_state, processed_hand_state], axis=-1)
    processed_action = np.concatenate([processed_wrist_action, processed_hand_action], axis=-1)
    processed_state, processed_action, action_valid_mask = get_presence_value(
        state_presence, action_presence, processed_action, processed_state, hand_ndim
    )
    if use_relative_action:
        processed_action = get_relative_action(processed_state[-1], processed_action)

    if normalizer is not None:
        state = normalizer['states'](processed_state)
        action = normalizer['actions'](processed_action)
    else:
        state = processed_state
        action = processed_action

    return state, action, action_valid_mask, state_presence, action_presence

# TODO: maybe we need to use the same augmentation for all images in the action chunk
# TODO: we can try more advanced augmentation techniques, notably, we should care about the depth image augmentation
def process_image(image, history, n_obs_image_steps, depth_image = None, aug_transform = None, depth_clip_range = None):
    '''
    Args:
        image: np.ndarray, shape: [N, H, W, 3]
        history: int
        n_obs_image_steps: int
        depth_image: np.ndarray, shape: [N, H, W]
        aug_transform: Optional[Callable]
    Returns:
        image: np.ndarray, shape: [N, H, W, 3]
        depth_image: np.ndarray, shape: [N, H, W]
    '''
    if n_obs_image_steps > 1:
        image_slice = [i for i in range(0, history + 1, history // (n_obs_image_steps - 1))]
    else:
        image_slice = [-1]

    images_to_process = image[image_slice]
    depth_images_to_process = None
    if depth_image is not None:
        depth_images_to_process = depth_image[image_slice]
        # normalize the depth images to [0, 1]
        depth_images_to_process = np.clip(
            depth_images_to_process, 
            depth_clip_range[0], 
            depth_clip_range[1]
        ) / (depth_clip_range[1] + 1e-6)
    if aug_transform is not None:
        augmented_images = []
        for img_np in images_to_process:
            # convert NumPy array (H, W, C) to PIL Image
            img_pil = Image.fromarray(img_np)
            augmented_pil = aug_transform(img_pil)
            augmented_np = np.array(augmented_pil)
            augmented_images.append(augmented_np)
        images_to_process = np.stack(augmented_images)

    return images_to_process, depth_images_to_process


# TODO: consider action valid mask when calculating normalizer
def get_normalizer(dataloader_cfg, normalizer_dataset = None, **kwargs):
    # Merge all data
    if normalizer_dataset is None:
        normalizer_dataset = LegendVLALowLevelDataset(**kwargs)
    dataloader = DataLoader(normalizer_dataset, collate_fn=normalizer_dataset.get_collator(), **dataloader_cfg)
    assert len(dataloader) > 0, "No data to calculate normalizer"
    normalizer = LinearNormalizer()
    normalizer.start_streaming_fit(keys=next(iter(dataloader)).keys())
    for batch in tqdm(dataloader, desc="Calculating normalizer"):
        input_data = {
            k: v.reshape(-1, v.shape[-1]) for k, v in batch.items() \
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray)
        }
        normalizer.update_streaming_fit(input_data)
    normalizer.finish_streaming_fit()
    # ignore the wrist rotation
    normalizer.ignore_dim(key='states', dim=slice(6, 18))
    normalizer.ignore_dim(key='actions', dim=slice(6, 18))

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
    Test function to load and test the dataset using the config file.
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
    print("Testing Dataset Loading")
    print("="*80)
    
    # Test VLA Dataset
    print("\n1. Testing LegendVLADataset...")
    try:
        vla_dataset = hydra.utils.instantiate(cfg.dataset.vla_dataset)
        print(f"   ✓ VLA Dataset created successfully")
        print(f"   - Dataset length: {len(vla_dataset)}")
        print(f"   - Number of replay buffers: {len(vla_dataset.replay_buffers)}")
        print(f"   - Sampler lengths: {vla_dataset.sampler_lens}")
        
        # Set preprocessor for VLA dataset
        print(f"\n   Setting preprocessor for VLA dataset...")
        try:
            vla_processor = hydra.utils.instantiate(cfg.vla_processor)
            vla_dataset.set_preprocessor(vla_processor)
            print(f"   ✓ Preprocessor set successfully")
        except Exception as e:
            print(f"   ✗ Error setting preprocessor: {e}")
            print(f"   Skipping sample access test (preprocessor required)")
            import traceback
            traceback.print_exc()
        
        # Test getting a sample (with preprocessor)
        if len(vla_dataset) > 0 and vla_dataset.preprocessor is not None:
            print(f"\n   Testing sample access (with preprocessor)...")
            try:
                sample = vla_dataset[0]
                print(f"   ✓ Sample accessed successfully")
                print(f"   - Sample keys: {list(sample.keys())}")
                for key, value in sample.items():
                    if hasattr(value, 'shape'):
                        print(f"   - {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"   - {key}: type={type(value)}")
            except Exception as e:
                print(f"   ✗ Error accessing sample: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"   ✗ Error creating VLA dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test VLM Dataset
    print("\n2. Testing LegendVLMDataset...")
    try:
        vlm_dataset = hydra.utils.instantiate(cfg.dataset.vlm_dataset)
        print(f"   ✓ VLM Dataset created successfully")
        print(f"   - Dataset length: {len(vlm_dataset)}")
        
        # Set preprocessor for VLM dataset
        print(f"\n   Setting preprocessor for VLM dataset...")
        try:
            vlm_processor = hydra.utils.instantiate(cfg.vlm_processor)
            vlm_dataset.set_preprocessor(vlm_processor)
            print(f"   ✓ Preprocessor set successfully")
        except Exception as e:
            print(f"   ✗ Error setting preprocessor: {e}")
            print(f"   Skipping sample access test (preprocessor required)")
            import traceback
            traceback.print_exc()
        
        # Test getting a sample (with preprocessor)
        if len(vlm_dataset) > 0 and vlm_dataset.preprocessor is not None:
            print(f"\n   Testing sample access (with preprocessor)...")
            try:
                sample = vlm_dataset[0]
                print(f"   ✓ Sample accessed successfully")
                print(f"   - Sample keys: {list(sample.keys())}")
                for key, value in sample.items():
                    if hasattr(value, 'shape'):
                        print(f"   - {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"   - {key}: type={type(value)}")
            except Exception as e:
                print(f"   ✗ Error accessing sample: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"   ✗ Error creating VLM dataset: {e}")
        import traceback
        traceback.print_exc()
        vlm_dataset = None
    
    # Test Unified Dataset
    print("\n3. Testing LegendUnifiedDataset...")
    try:
        unified_dataset = LegendUnifiedDataset(
            vla_dataset=vla_dataset,
            vlm_dataset=vlm_dataset,
        )
        print(f"   ✓ Unified Dataset created successfully")
        print(f"   - Total dataset length: {len(unified_dataset)}")
        print(f"   - VLA samples: {len(vla_dataset)}")
        if vlm_dataset is not None:
            print(f"   - VLM samples: {len(vlm_dataset)}")
        
        # Test getting samples
        if len(unified_dataset) > 0:
            print(f"\n   Testing sample access from unified dataset...")
            # Test VLA sample
            if len(vla_dataset) > 0 and vla_dataset.preprocessor is not None:
                try:
                    sample = unified_dataset[0]
                    print(f"   ✓ VLA sample accessed successfully")
                    print(f"   - Sample keys: {list(sample.keys())}")
                except Exception as e:
                    print(f"   ✗ Error accessing VLA sample: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"   ⚠ Skipping VLA sample access (preprocessor not set)")
            
            # Test VLM sample
            if vlm_dataset is not None and len(vlm_dataset) > 0 and vlm_dataset.preprocessor is not None:
                try:
                    vlm_idx = len(vla_dataset)
                    sample = unified_dataset[vlm_idx]
                    print(f"   ✓ VLM sample accessed successfully")
                    print(f"   - Sample keys: {list(sample.keys())}")
                except Exception as e:
                    print(f"   ✗ Error accessing VLM sample: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"   ⚠ Skipping VLM sample access (preprocessor not set)")
    except Exception as e:
        print(f"   ✗ Error creating unified dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test validation dataset
    print("\n4. Testing Validation Dataset...")
    try:
        val_dataset = unified_dataset.get_validation_dataset()
        print(f"   ✓ Validation dataset created successfully")
        print(f"   - Validation dataset length: {len(val_dataset)}")
        if len(val_dataset) > 0:
            print(f"   - Testing validation sample access...")
            # Validation dataset inherits preprocessors from parent datasets
            if val_dataset.vla_dataset.preprocessor is not None:
                try:
                    sample = val_dataset[0]
                    print(f"   ✓ Validation sample accessed successfully")
                except Exception as e:
                    print(f"   ✗ Error accessing validation sample: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"   ⚠ Skipping validation sample access (preprocessor not set)")
    except Exception as e:
        print(f"   ✗ Error creating validation dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # Test sampler
    print("\n5. Testing Batch Sampler...")
    try:
        batch_sampler = unified_dataset.get_sampler(
            batch_size=cfg.dataloader.batch_sampler.batch_size,
            vla_ratio=cfg.dataloader.batch_sampler.vla_ratio,
            shuffle=cfg.dataloader.batch_sampler.shuffle,
            seed=cfg.dataloader.batch_sampler.seed,
            drop_last=cfg.dataloader.batch_sampler.drop_last,
        )
        print(f"   ✓ Batch sampler created successfully")
        print(f"   - Sampler length: {len(batch_sampler)}")
        
        # Test getting a batch
        if len(batch_sampler) > 0:
            print(f"\n   Testing batch sampling...")
            try:
                batch_indices = next(iter(batch_sampler))
                print(f"   ✓ Batch sampled successfully")
                print(f"   - Batch size: {len(batch_indices)}")
                print(f"   - Batch indices (first 5): {batch_indices[:5]}")
            except Exception as e:
                print(f"   ✗ Error sampling batch: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"   ✗ Error creating batch sampler: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Dataset Loading Test Completed!")
    print("="*80)


if __name__ == "__main__":
    test_dataset_loading()
