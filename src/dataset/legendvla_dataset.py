'''
Propise dataset for LegendVLA
Every action is the delta of the next predicted absolute state and the state at the beginning of the action chunk.
'''

import os
from typing import Dict, Optional
import torch
import numpy as np
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import random
from datasets import load_dataset
from src.utils.pytorch_util import dict_apply
from src.utils.streaming_replay_buffer import StreamingReplayBuffer
from src.utils.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from src.utils.geometry import transform_wrist_to_target_frame, rot_matrix_from_6drot, rot_matrix_to_6drot
from src.model.common.normalizer import LinearNormalizer
from .base_dataset import BaseImageDataset, BaseDataCollator
from .base_vl_preprocessor import BaseVLPreprocessor


class LegendVLADataset(BaseImageDataset):
    def __init__(self,
            zarr_paths,
            horizon=1,
            pad_before=0,
            pad_after=0,
            shape_meta=None,
            seed=42,
            val_ratio=0.0,
            history=30,
            normalizer_dataloader_cfg=dict(),
            max_train_episodes=None,
            train_mode=True,
        ):
        
        super().__init__()
        self.zarr_paths = zarr_paths
        self.preprocessor = None
        self.history = history
        self.normalizer_dataloader_cfg = normalizer_dataloader_cfg
        self.max_train_episodes = max_train_episodes
        self.normalizer = None

        # Initialize storage lists
        self.replay_buffers = []
        self.train_masks = []
        self.samplers = []
        self.sampler_lens = []

        self.train_mode = train_mode
        self.aug_transform = None
        if self.train_mode:
            self.aug_transform = transforms.Compose([
                # ColorJitter: random change brightness, contrast, saturation, and hue
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                # GaussianBlur: apply gaussian blur
                # kernel_size must be odd
                transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))
            ])
        
        # Process each zarr file
        for zarr_path in zarr_paths:
            # Create replay buffer
            replay_buffer = StreamingReplayBuffer.copy_from_path(
                zarr_path, keys=['image', 'state', 'instruction', 'instruction_num', 'action', 'extrinsic', 'presence'])
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
                sequence_length=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=train_mask,
                key_first_k=dict(image=history+1))
            self.samplers.append(sampler)
            
            # Record sampler length
            self.sampler_lens.append(len(sampler))

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.shape_meta = shape_meta
        self.hand_ndim = shape_meta['obs']['state']['hand']['shape'][-1] // 2 # per hand pca ncomponents
        self.n_obs_image_steps = shape_meta['obs']['rgb']['horizon']
        self.n_obs_state_steps = shape_meta['obs']['state']['horizon']

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.samplers = []
        val_set.train_masks = []
        val_set.sampler_lens = []
        val_set.train_mode = False
        val_set.aug_transform = None

        for i, replay_buffer in enumerate(self.replay_buffers):
            # Create validation set sampler
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=~self.train_masks[i],
                key_first_k=dict(image=self.history+1))
            val_set.samplers.append(sampler)
            val_set.train_masks.append(~self.train_masks[i])
            val_set.sampler_lens.append(len(sampler))
            
        return val_set
    
    def _sample_to_data(self, sample):
        state, action, action_valid_mask = process_state_action(
            wrist_state = sample['state/wrist'].astype(np.float32), 
            hand_state = sample['state/hand'].astype(np.float32), 
            wrist_action = sample['action/wrist'].astype(np.float32), 
            hand_action = sample['action/hand'].astype(np.float32), 
            extrinsic = sample['extrinsic'].astype(np.float32).reshape(-1, 4, 4), # [Horizon, 16] -> [Horizon, 4, 4]
            presence = sample['presence'], 
            normalizer = self.normalizer, 
            hand_ndim = self.hand_ndim, 
            history = self.history, 
            n_obs_state_steps = self.n_obs_state_steps
        )
        image = process_image(sample['image'], self.history, self.n_obs_image_steps, self.aug_transform)

        instruction = sample['instruction'][self.history]
        instruction_num = sample['instruction_num'][self.history]
        # sample a random instruction from the candidate instructions
        idx = np.random.randint(0, instruction_num)
        instruction = instruction[idx]

        # Process all images in batch
        processed_results = self.preprocessor(images=image, text=instruction, states=state, human_actions=action)

        data = {
            'input_ids': processed_results['input_ids'],
            'labels': processed_results['labels'],
            'answer_start_idx': processed_results['answer_start_idx'],
            'attention_mask': processed_results['attention_mask'],
            'pixel_values': processed_results['pixel_values'], 
            # we assume the history of the data is 30 Hz, the image should cover the past 1 second
            'human_actions': action,
            'human_actions_valid_mask': action_valid_mask,
        }
        return data

    def set_preprocessor(self, preprocessor: BaseVLPreprocessor):
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
            max_train_episodes=self.max_train_episodes
        )
        normalizer = get_normalizer(self.normalizer_dataloader_cfg, normalizer_dataset)
        self.normalizer = normalizer

        return normalizer

    def get_collator(self):
        return LegendVLDataCollator()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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
        return sum(self.sampler_lens)
        

class LegendVLMDataset(BaseImageDataset):
    def __init__(self,
            dataset_paths,
            split='train',
            cache_dir=None,
            weights=[0.5, 0.5, 0.5],
            seed=42,
            val_ratio=0.0,
            train_mode=True,
        ):
        
        super().__init__()
        self.dataset_paths = dataset_paths
        self.split = split
        self.weights = weights
        self.cache_dir = cache_dir
        self.datasets = None
        self.preprocessor = None
        self.train_mode = train_mode
        self.aug_transform = None
        if self.train_mode:
            self.aug_transform = transforms.Compose([
                # ColorJitter: random change brightness, contrast, saturation, and hue
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                # GaussianBlur: apply gaussian blur
                # kernel_size must be odd
                transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))
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
        val_copy.train_mode = False
        val_copy.aug_transform = None
        val_copy.train_datasets = self.val_datasets
        return val_copy
    
    def _sample_to_data(self, sample):
        images = sample['images'] # List[PIL.JpegImagePlugin.JpegImageFile]
        text = sample['texts'] 
        weights = self.weights
        formatting_ratings = np.array(sample['formatting_ratings'])
        visual_dependency_ratings = np.array(sample['visual_dependency_ratings'])
        relevance_ratings = np.array(sample['relevance_ratings'])

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
            if self.train_mode and self.aug_transform is not None:
                augmented_pil = self.aug_transform(img_pil)
            else:
                augmented_pil = img_pil
            augmented_np = np.array(augmented_pil)
            augmented_images.append(augmented_np)
        images_to_process = np.stack(augmented_images)
        # Process all images in batch
        processed_results = self.preprocessor(images=images_to_process, text=question, target=answer)

        data = {
            'input_ids': processed_results['input_ids'],
            'labels': processed_results['labels'] ,
            'attention_mask': processed_results['attention_mask'] ,
            'pixel_values': processed_results['pixel_values'], 
            'answer_start_idx': processed_results['answer_start_idx'],
        }
        return data

    def get_collator(self):
        return LegendVLDataCollator()

    def set_preprocessor(self, preprocessor: BaseVLPreprocessor):
        self.preprocessor = preprocessor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find corresponding sampler
        sample = self.train_datasets[idx]
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def __len__(self):
        return len(self.train_datasets)


class LegendUnifiedDataset(BaseImageDataset):
    def __init__(self,
        vla_dataset: LegendVLADataset,
        vlm_dataset: LegendVLMDataset = None,
    ):
        super().__init__()
        self.vla_dataset = vla_dataset
        self.vlm_dataset = vlm_dataset
        self.shape_meta = None

    def get_collator(self):
        return LegendUnifiedDataCollator()

    def get_validation_dataset(self):
        return LegendUnifiedDataset(
            vla_dataset=self.vla_dataset.get_validation_dataset(),
            vlm_dataset=self.vlm_dataset.get_validation_dataset() if self.vlm_dataset is not None else None
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.shape_meta is None:
            self.shape_meta = dict()
            vla_sample = self.vla_dataset[0]
            for key in vla_sample.keys():
                self.shape_meta[key] = vla_sample[key].shape
        if idx < len(self.vla_dataset):
            return self.vla_dataset[idx]
        elif self.vlm_dataset is not None:
            sample = self.vlm_dataset[idx - len(self.vla_dataset)]
            for key in self.shape_meta.keys():
                if key not in sample and "valid_mask" not in key:
                    sample[key] = torch.zeros(self.shape_meta[key])
                    sample[f"{key}_valid_mask"] = torch.zeros(self.shape_meta[key], dtype=torch.bool)
            return sample
        else:
            raise ValueError("No dataset to get item from")

    def __len__(self):
        return len(self.vla_dataset) + len(self.vlm_dataset) if self.vlm_dataset is not None else len(self.vla_dataset)


class LegendVLALowLevelDataset(BaseImageDataset):
    def __init__(
        self,
        zarr_paths,
        horizon=1,
        pad_before=0,
        pad_after=0,
        shape_meta=None,
        history=30,
        max_train_episodes=None,
        normalizer_dataloader_cfg=None,
    ):
        
        super().__init__()
        self.history = history

        # Initialize storage lists
        self.replay_buffers = []
        self.samplers = []
        self.sampler_lens = []
        
        # Process each zarr file
        for zarr_path in zarr_paths:
            # Create replay buffer
            replay_buffer = StreamingReplayBuffer.copy_from_path(
                zarr_path, keys=['state', 'action', 'extrinsic', 'presence'])
            self.replay_buffers.append(replay_buffer)

            # Create train mask
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes,
                val_ratio=0,
            )
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask,
                max_n=max_train_episodes
            )
            
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
        self.hand_ndim = shape_meta['obs']['state']['hand']['shape'][-1] // 2 # per hand pca ncomponents
        self.n_obs_state_steps = shape_meta['obs']['state']['horizon']
        self.normalizer = None
        self.normalizer_dataloader_cfg = normalizer_dataloader_cfg

    def _sample_to_data(self, sample):
        state, action, _ = process_state_action(
            wrist_state = sample['state/wrist'].astype(np.float32), 
            hand_state = sample['state/hand'].astype(np.float32), 
            wrist_action = sample['action/wrist'].astype(np.float32), 
            hand_action = sample['action/hand'].astype(np.float32), 
            extrinsic = sample['extrinsic'].astype(np.float32).reshape(-1, 4, 4), # [Horizon, 16] -> [Horizon, 4, 4]
            presence = sample['presence'], 
            normalizer = self.normalizer, 
            hand_ndim = self.hand_ndim, 
            history = self.history, 
            n_obs_state_steps = self.n_obs_state_steps
        )

        data = {
            'states': state,
            'human_actions': action,
        }
        return data

    def get_normalizer(self):
        self.normalizer = get_normalizer(self.normalizer_dataloader_cfg, self)
        return self.normalizer

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer

    def get_collator(self):
        return BaseDataCollator4numpy()

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # Find corresponding sampler
        curr_idx = idx
        for i, length in enumerate(self.sampler_lens):
            if curr_idx < length:
                sample = self.samplers[i].sample_sequence(curr_idx)
                break
            curr_idx -= length
            
        data = self._sample_to_data(sample)
        return data

    def __len__(self):
        return sum(self.sampler_lens)
                

class LegendVLDataCollator(BaseDataCollator):
    def __init__(self, pad_token_id: int = None):
        super().__init__()
        self.pad_token_id = pad_token_id

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
        for key in data_list[0].keys():
            '''
            if key != 'instruction':
                batch[key] = torch.stack([item[key] for item in data_list])
            else:
                input_ids_batch = [item[key] for item in data_list]
                batch["input_ids"] = rnn_utils.pad_sequence(
                    input_ids_batch,
                    batch_first=True,
                    padding_value=self.pad_token_id
                )
                attention_mask_batch = (batch["input_ids"] != self.pad_token_id).long()
                batch["attention_mask"] = attention_mask_batch
            '''
            # We assume the length of tokenized instruction is the same for all samples
            batch[key] = torch.stack([item[key] for item in data_list])

        return batch


class LegendUnifiedDataCollator(LegendVLDataCollator):
    def __init__(self):
        super().__init__()

    def __call__(self, data_list):
        return super().__call__(data_list)


class BaseDataCollator4numpy(BaseDataCollator):
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
            batch[key] = np.stack([item[key] for item in data_list], axis=0)

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
    action[..., :6] = action[..., :6] - state[:6]
    wrist_action_rot_mat = [
        rot_matrix_from_6drot(action[..., 6:12]),
        rot_matrix_from_6drot(action[..., 12:18])
    ]
    wrist_state_rot_mat = [
        rot_matrix_from_6drot(state[6:12]),
        rot_matrix_from_6drot(state[12:18])
    ]
    for idx in range(2): 
        wrist_action_rot_mat[idx] = wrist_action_rot_mat[idx] @ np.linalg.pinv(wrist_state_rot_mat[idx])

    action[..., 6:12] = rot_matrix_to_6drot(wrist_action_rot_mat[0])
    action[..., 12:18] = rot_matrix_to_6drot(wrist_action_rot_mat[1])
    action[..., 18:] = action[..., 18:] - state[18:]
    return action


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
    Returns:
        state: np.ndarray, shape: [T, wrist_dim + all_hand_dim]
        action: np.ndarray, shape: [H, wrist_dim + all_hand_dim]
        action_valid_mask: np.ndarray, shape: [H, wrist_dim + all_hand_dim]
    '''
    if n_obs_state_steps > 1:
        state_slice = [i for i in range(0, history + 1, history // (n_obs_state_steps - 1))]
    else:
        state_slice = [history]
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

    processed_wrist_state = transform_wrist_to_target_frame(wrist_state[state_slice], extrinsic[history])
    processed_wrist_action = transform_wrist_to_target_frame(wrist_action[history:], extrinsic[history])

    # use delta of wrist translation and hand mano params as action
    state_presence = presence[state_slice]
    action_presence = presence[history:]
    processed_state = np.concatenate([processed_wrist_state, hand_state], axis=-1)
    processed_action = np.concatenate([processed_wrist_action, hand_action], axis=-1)
    processed_state, processed_action, action_valid_mask = get_presence_value(
        state_presence, action_presence, processed_action, processed_state, hand_ndim
    )

    processed_action = get_relative_action(processed_state[-1], processed_action)

    if normalizer is not None:
        state = normalizer['states'](processed_state)
        action = normalizer['human_actions'](processed_action)
    else:
        state = processed_state
        action = processed_action

    return state, action, action_valid_mask


def process_image(image, history, n_obs_image_steps, aug_transform = None):
    '''
    Args:
        image: np.ndarray, shape: [N, H, W, 3]
        history: int
        n_obs_image_steps: int
        aug_transform: Optional[Callable]
    Returns:
        image: np.ndarray, shape: [T, H, W, 3]
    '''
    if n_obs_image_steps > 1:
        image_slice = [i for i in range(0, history + 1, history // (n_obs_image_steps - 1))]
    else:
        image_slice = [history]

    images_to_process = image[image_slice]
    if aug_transform is not None:
        augmented_images = []
        for img_np in images_to_process:
            # convert NumPy array (H, W, C) to PIL Image
            img_pil = Image.fromarray(img_np)
            augmented_pil = aug_transform(img_pil)
            augmented_np = np.array(augmented_pil)
            augmented_images.append(augmented_np)
        images_to_process = np.stack(augmented_images)

    return images_to_process


def get_normalizer(dataloader_cfg, normalizer_dataset = None, **kwargs):
    # Merge all data
    if normalizer_dataset is None:
        normalizer_dataset = LegendVLALowLevelDataset(**kwargs)
    dataloader = DataLoader(normalizer_dataset, collate_fn=normalizer_dataset.get_collator(), **dataloader_cfg)
    assert len(dataloader) > 0, "No data to calculate normalizer"
    normalizer = LinearNormalizer()
    normalizer.start_streaming_fit(keys=next(iter(dataloader)).keys())
    for batch in tqdm(dataloader, desc="Calculating normalizer"):
        input_data = {k: v.reshape(-1, v.shape[-1]) for k, v in batch.items()}
        normalizer.update_streaming_fit(input_data)
    normalizer.finish_streaming_fit()
    # ignore the wrist rotation
    normalizer.ignore_dim(key='states', dim=slice(6, 18))
    normalizer.ignore_dim(key='human_actions', dim=slice(6, 18))

    def print_dict(d):
        for k, v in d.items():
            print(f"{k}: {v}")
    
    for key in normalizer.params_dict.keys():
        print(f"{key}: ")
        print_dict(normalizer.params_dict[key]['input_stats'])
        print(f"scale: {normalizer.params_dict[key]['scale']}")
        print(f"offset: {normalizer.params_dict[key]['offset']}")

    return normalizer

