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
from src.utils.streaming_replay_buffer import StreamingReplayBuffer
from src.utils.sampler import (
    SequenceSampler, VariableLengthSequenceSampler, get_val_mask, downsample_mask)
from src.utils.geometry import (
    transform_wrist_to_target_frame, 
    homo_matrix_from_trans_6drot, 
    homo_matrix_to_trans_6drot, 
    transform_hand_points_to_wrist_frame,
)
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
            objective=None,
            normalizer_dataloader_cfg=dict(),
            max_train_episodes=None,
            train_mode=True,
            token_len_buckets=None, 
            return_raw_sample=False,
        ):
        
        super().__init__()
        self.zarr_paths = zarr_paths
        self.preprocessor = None
        self.history = history
        self.objective = objective
        self.normalizer_dataloader_cfg = normalizer_dataloader_cfg
        self.max_train_episodes = max_train_episodes
        self.normalizer = None
        self.return_raw_sample = return_raw_sample

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
                zarr_path, 
                keys=['image', 'state', 'instruction', 'instruction_num', 'action', 'extrinsic', 'intrinsic', 'presence'], 
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
        self.motion_type = shape_meta['obs']['state']['type']
        self.hand_ndim = shape_meta['obs']['state']['hand']['shape'][-1] // 2 # per hand pca ncomponents
        self.n_obs_image_steps = shape_meta['obs']['rgb']['horizon']
        self.n_obs_state_steps = shape_meta['obs']['state']['horizon']
        self.token_len_buckets = token_len_buckets

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.samplers = []
        val_set.train_masks = []
        val_set.sampler_lens = []
        val_set.train_mode = False
        val_set.aug_transform = None
        # Preserve the return_raw_sample setting

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

    def sample_for_inference(self, sample):
        wrist_state = sample['state/wrist'].astype(np.float32)
        hand_state = sample[f'state/{self.motion_type}'].astype(np.float32)
        wrist_action = sample['action/wrist'].astype(np.float32)
        hand_action = sample[f'action/{self.motion_type}'].astype(np.float32)        
        presence = sample['presence']
        extrinsic = sample['extrinsic'].astype(np.float32).reshape(-1, 4, 4)
        step = self.history // self.n_obs_state_steps
        state_slice = [self.history - i * step for i in range(0, self.n_obs_state_steps)]
        state_slice = state_slice[::-1]
        # use first self.hand_ndim components of hand state and action
        all_hand_ndim = hand_state.shape[-1] // 2 # per hand dims, e.g. 45 in MANO hand params
        # processed_wrist_state = transform_wrist_to_target_frame(wrist_state[state_slice], extrinsic[self.history])
        # processed_wrist_action = transform_wrist_to_target_frame(wrist_action[self.history:], extrinsic[self.history])

        hand_state = np.concatenate([
            hand_state[state_slice, :self.hand_ndim], 
            hand_state[state_slice, all_hand_ndim:all_hand_ndim + self.hand_ndim]
        ], axis=-1)
        hand_action = np.concatenate([
            hand_action[self.history:, :self.hand_ndim], 
            hand_action[self.history:, all_hand_ndim:all_hand_ndim + self.hand_ndim]
        ], axis=-1)

        # use delta of wrist translation and hand mano params as action
        state_presence = presence[state_slice]
        action_presence = presence[self.history:]
        processed_state = np.concatenate([wrist_state[state_slice], hand_state], axis=-1)
        processed_action = np.concatenate([wrist_action[self.history:], hand_action], axis=-1)
        processed_state, processed_action, action_valid_mask = get_presence_value(
            state_presence, action_presence, processed_action, processed_state, self.hand_ndim
        )

        # image = process_image(sample['image'], self.history, self.n_obs_image_steps, self.aug_transform)
        image = sample['image'][self.history:]
        instruction = sample['instruction'][self.history]
        instruction_num = sample['instruction_num'][self.history]
        # sample a random instruction from the candidate instructions
        idx = np.random.randint(0, instruction_num)
        instruction = instruction[idx]
        # Follow the same slicing pattern as process_state_action
        # For state-related data, take the last time step (history)
        # For action-related data, take from history onwards (history:)
        extrinsic = sample['extrinsic'][self.history:].astype(np.float32).reshape(-1, 4, 4)
        intrinsic = sample['intrinsic'][self.history:].astype(np.float32)
        action_shape = sample['action/shape'][self.history:].astype(np.float32)
        state_shape = sample['state/shape'][self.history:].astype(np.float32)
        presence = sample['presence'][self.history:]

        data = {
            'state': processed_state,
            'action': processed_action,
            'action_valid_mask': action_valid_mask,
            'image': image,
            'extrinsic': extrinsic,
            'intrinsic': intrinsic,
            'action_shape': action_shape,
            'state_shape': state_shape,
            'presence': presence,
        }
        return data

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
        image = process_image(sample['image'], self.history, self.n_obs_image_steps, self.aug_transform)

        intrinsic = sample['intrinsic'][self.history].astype(np.float32)
        instruction = sample['instruction'][self.history]
        instruction_num = sample['instruction_num'][self.history]
        # sample a random instruction from the candidate instructions
        idx = np.random.randint(0, instruction_num)
        instruction = instruction[idx]

        # Process all images in batch
        processed_results = self.preprocessor(
            images=image, 
            text=instruction, 
            states=state, 
            actions=action, 
            intrinsic=intrinsic, 
            objective=self.objective,
        )

        data = {
            'input_ids': processed_results['input_ids'],
            'answer_start_idx': processed_results['answer_start_idx'],
            'attention_mask': processed_results['attention_mask'],
            'pixel_values': processed_results['pixel_values'], 
        }
        if self.objective != "train_ar":
            data['actions'] = action
            data['actions_valid_mask'] = action_valid_mask
        if self.objective != "train_flow":
            data['labels'] = processed_results['labels']
        return data

    def set_preprocessor(self, preprocessor: BaseVLPreprocessor):
        self.preprocessor = preprocessor
        if self.token_len_buckets is None:
            self.token_len_buckets = [preprocessor.max_seq_len]

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer

    def set_return_raw_sample(self, return_raw_sample: bool):
        """Set whether to return raw sample data (for testing/debugging purposes)
        
        When enabling raw sample mode, recreates samplers to load more image frames
        for visualization purposes (history + horizon frames instead of just history + 1).
        """
        if self.return_raw_sample == return_raw_sample:
            # No change needed
            return
            
        self.return_raw_sample = return_raw_sample
        
        # Recreate samplers with appropriate image frame loading
        self.samplers = []
        self.sampler_lens = []
        
        for i, replay_buffer in enumerate(self.replay_buffers):
            # Determine how many image frames to load
            # For inference/visualization, we need history + horizon frames
            # For training, we only need history + 1 frames
            image_frames_to_load = self.history + 30 if return_raw_sample else self.history + 1
            
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=self.train_masks[i],
                key_first_k=dict(image=image_frames_to_load))
            self.samplers.append(sampler)
            self.sampler_lens.append(len(sampler))
        
        print(f"Samplers recreated: {'raw mode' if return_raw_sample else 'training mode'} - loading {image_frames_to_load} image frames")


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
            return_numpy=True
        )
        normalizer = get_normalizer(self.normalizer_dataloader_cfg, normalizer_dataset)
        self.normalizer = normalizer

        return normalizer

    def get_collator(self):
        assert self.preprocessor is not None, "Preprocessor is not set"
        return LegendVLDataCollator(
            pad_token_id=self.preprocessor.tokenizer.pad_token_id,
            ignore_index=self.preprocessor.ignore_index,
            token_len_buckets=self.token_len_buckets,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find corresponding sampler
        curr_idx = idx
        dataset_idx = 0
        for i, length in enumerate(self.sampler_lens):
            if curr_idx < length:
                sample = self.samplers[i].sample_sequence(curr_idx)
                dataset_idx = i
                break
            curr_idx -= length
        
        # Return raw sample if requested (for testing/debugging purposes)
        if self.return_raw_sample:
            # Convert numpy arrays to torch tensors for consistency
            data = self.sample_for_inference(sample)
            torch_data = dict_apply(data, torch.from_numpy)
            # Add dataset source information
            torch_data['dataset_source'] = self.zarr_paths[dataset_idx]
            torch_data['dataset_idx'] = dataset_idx
            return torch_data
            
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
            token_len_buckets=None,
        ):
        
        super().__init__()
        self.dataset_paths = dataset_paths
        self.split = split
        self.weights = weights
        self.cache_dir = cache_dir
        self.datasets = None
        self.preprocessor = None
        self.train_mode = train_mode
        self.token_len_buckets = token_len_buckets
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
        assert self.preprocessor is not None, "Preprocessor is not set"
        return LegendVLDataCollator(
            pad_token_id=self.preprocessor.tokenizer.pad_token_id,
            ignore_index=self.preprocessor.ignore_index,
            token_len_buckets=self.token_len_buckets,
        )

    def set_preprocessor(self, preprocessor: BaseVLPreprocessor):
        self.preprocessor = preprocessor
        if self.token_len_buckets is None:
            self.token_len_buckets = [preprocessor.max_seq_len]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find corresponding sampler
        sample = self.train_datasets[idx]
        data = self._sample_to_data(sample, idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def __len__(self):
        return len(self.train_datasets)


class LegendUnifiedDataset(BaseImageDataset):
    def __init__(self,
        vla_dataset: LegendVLADataset,
        vlm_dataset: LegendVLMDataset = None,
        token_len_buckets: list = None,
    ):
        super().__init__()
        self.vla_dataset = vla_dataset
        self.vlm_dataset = vlm_dataset
        self.shape_meta = None
        self.token_len_buckets = token_len_buckets
        
        print(f"LegendUnifiedDataset initialized with {len(self.vla_dataset)} VLA samples")
        if self.vlm_dataset is not None:
            print(f"LegendUnifiedDataset initialized with {len(self.vlm_dataset)} VLM samples")

    def get_collator(self):
        return LegendUnifiedDataCollator(token_len_buckets=self.token_len_buckets)

    def set_return_raw_sample(self, return_raw_sample: bool):
        """Set whether to return raw sample data for VLA dataset (for testing/debugging purposes)"""
        self.vla_dataset.set_return_raw_sample(return_raw_sample)

    def get_validation_dataset(self):
        return LegendUnifiedDataset(
            vla_dataset=self.vla_dataset.get_validation_dataset(),
            vlm_dataset=self.vlm_dataset.get_validation_dataset() if self.vlm_dataset is not None else None, 
            token_len_buckets=self.token_len_buckets
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
        seed=42,
        val_ratio=0.0,
        dims=None, 
        max_train_episodes=None,
        return_numpy=True, # whether to return numpy arrays
        normalizer_dataloader_cfg=None,
        debug=False,
        use_relative_action=False,
        action_chunk_lengths=None,  # List of action chunk lengths, e.g., [4, 8, 16, 32]
    ):
        
        super().__init__()
        self.history = history
        self.action_chunk_lengths = action_chunk_lengths if action_chunk_lengths is not None else [horizon]

        # Initialize storage lists
        self.replay_buffers = []
        self.train_masks = []
        # Structure: samplers[zarr_idx] = sampler (VariableLengthSequenceSampler)
        self.samplers = []
        # Structure: sampler_lens[zarr_idx] = length
        self.sampler_lens = []
        # Cumulative lengths for indexing: zarr_idx -> cumulative_length
        self.cumulative_lengths = []
        
        # Process each zarr file
        for zarr_idx, zarr_path in enumerate(zarr_paths):
            # Create replay buffer
            replay_buffer = StreamingReplayBuffer.copy_from_path(
                zarr_path, keys=['state', 'action', 'extrinsic', 'presence'], lazy_load=False)
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
            
            # Create a single sampler that selects appropriate chunk length for each position
            # For each starting position, select the largest chunk_length that fits
            # This ensures each position is sampled only once with the appropriate length
            sampler = VariableLengthSequenceSampler(
                replay_buffer=replay_buffer,
                history=history,
                action_chunk_lengths=self.action_chunk_lengths,
                pad_before=pad_before,
                episode_mask=train_mask,
                key_first_k=dict())
            self.samplers.append(sampler)
            self.sampler_lens.append(len(sampler))
            self.cumulative_lengths.append(len(sampler))

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
        val_set.cumulative_lengths = []

        for i, replay_buffer in enumerate(self.replay_buffers):
            # Create validation set sampler that selects appropriate chunk length for each position
            sampler = VariableLengthSequenceSampler(
                replay_buffer=replay_buffer,
                history=self.history,
                action_chunk_lengths=self.action_chunk_lengths,
                pad_before=self.pad_before,
                episode_mask=~self.train_masks[i],
                key_first_k=dict())
            
            val_set.samplers.append(sampler)
            val_set.sampler_lens.append(len(sampler))
            val_set.train_masks.append(~self.train_masks[i])
            val_set.cumulative_lengths.append(len(sampler))
            
        return val_set

    def _sample_to_data(self, sample, action_chunk_length=None):
        # Select data keys based on motion_type
        # action_chunk_length is used to determine how many action steps to extract
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
        
        # Crop action to the specified chunk length (no padding)
        if action_chunk_length is not None and action.shape[0] > action_chunk_length:
            action = action[:action_chunk_length]
            action_valid_mask = action_valid_mask[:action_chunk_length]
        
        if self.dims is not None:
            dim_slice = slice(self.dims[0], self.dims[1])
            state = state[:, dim_slice]
            action = action[:, dim_slice]
            action_valid_mask = action_valid_mask[:, dim_slice]

        data = {
            'states': state,
            'actions': action,
            'action_chunk_length': action_chunk_length if action_chunk_length is not None else action.shape[0],
        }
        return data

    def get_normalizer(self):
        self.normalizer = get_normalizer(self.normalizer_dataloader_cfg, self)
        return self.normalizer

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer

    def get_collator(self):
        return VariableLengthActionCollator()

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # Find corresponding sampler
        curr_idx = idx
        selected_zarr_idx = None
        selected_sampler_idx = None
        
        # Find which zarr file this index belongs to
        for zarr_idx, cumulative_length in enumerate(self.cumulative_lengths):
            if curr_idx < cumulative_length:
                selected_zarr_idx = zarr_idx
                selected_sampler_idx = curr_idx
                break
            curr_idx -= cumulative_length
        
        if selected_zarr_idx is None:
            raise IndexError(f"Index {idx} out of range")
        
        # Sample from the corresponding sampler (which already selected appropriate chunk length)
        sample = self.samplers[selected_zarr_idx].sample_sequence(selected_sampler_idx)
        
        # Get chunk length from sample (stored by VariableLengthSequenceSampler)
        action_chunk_length = sample.get('_chunk_length')
        # Remove internal metadata
        if '_chunk_length' in sample:
            del sample['_chunk_length']
        
        # Process data with the specific chunk length
        data = self._sample_to_data(sample, action_chunk_length=action_chunk_length)
        if not self.return_numpy:
            data = dict_apply(data, torch.from_numpy) 
        if self.debug: 
            data.update({'idx': idx})
        return data

    def __len__(self):
        return sum(self.cumulative_lengths)


class LegendVLDataCollator(BaseDataCollator):
    def __init__(self, pad_token_id: int = 0, ignore_index: int = -100, token_len_buckets: list = None):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.token_len_buckets = token_len_buckets

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
        max_token_len = max([item.shape[-1] for item in input_ids_batch])
        bucket = None
        for len in self.token_len_buckets:
            if max_token_len <= len:
                bucket = len
                break
        assert bucket is not None, "No bucket found for the max token length"
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
        for key in data_list[0].keys():
            if key != 'input_ids' and key != 'attention_mask' and key != 'labels':
                batch[key] = torch.stack([item[key] for item in data_list])

        return batch

class LegendUnifiedDataCollator(LegendVLDataCollator):
    def __init__(self, pad_token_id: int = 0, ignore_index: int = -100, token_len_buckets: list = None):
        super().__init__(pad_token_id=pad_token_id, ignore_index=ignore_index, token_len_buckets=token_len_buckets)

    def __call__(self, data_list):
        return super().__call__(data_list)

class VariableLengthActionCollator(BaseDataCollator):
    """
    Collator for variable length action chunks.
    Groups samples by action_chunk_length to ensure same-length batches.
    """
    def __init__(self):
        super().__init__()
    
    def __call__(self, data_list):
        """
        Group samples by action_chunk_length and collate each group separately.
        Returns a dictionary with batches for each chunk length.
        """
        # Group samples by action_chunk_length
        grouped_data = {}
        for item in data_list:
            chunk_length = item.get('action_chunk_length', item['actions'].shape[0])
            if chunk_length not in grouped_data:
                grouped_data[chunk_length] = []
            grouped_data[chunk_length].append(item)
        
        # Collate each group
        batches = {}
        for chunk_length, group_data in grouped_data.items():
            batch = {}
            for key in group_data[0].keys():
                if key == 'action_chunk_length':
                    # Skip action_chunk_length, it's the same for all in this group
                    continue
                values = [item[key] for item in group_data]
                if isinstance(values[0], torch.Tensor):
                    batch[key] = torch.stack(values, dim=0)
                elif isinstance(values[0], np.ndarray):
                    batch[key] = torch.from_numpy(np.stack(values, axis=0))
                else:
                    batch[key] = values
            batches[chunk_length] = batch
        
        # If all samples have the same length, return a single batch
        if len(batches) == 1:
            return list(batches.values())[0]
        
        # Otherwise return batches grouped by length
        return batches

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

