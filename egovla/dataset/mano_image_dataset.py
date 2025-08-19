from typing import Dict
import torch
import numpy as np
import copy
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from egovla.utils.pytorch_util import dict_apply
from egovla.utils.streaming_replay_buffer import StreamingReplayBuffer
from egovla.utils.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from egovla.utils.transformation import transform_wrist_to_target_frame
from egovla.model.common.normalizer import LinearNormalizer
from egovla.dataset.base_dataset import BaseImageDataset
from egovla.dataset.base_collator import BaseDataCollator
from egovla.dataset.base_vl_preprocessor import BaseVLPreprocessor

class MANOImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_paths,
            horizon=1,
            n_obs_steps=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            frequency=30,
            max_train_episodes=None,
            image_size=(384, 384)
            ):
        
        super().__init__()
        self.preprocessor = None
        self.image_size = image_size
        self.frequency = frequency

        # Initialize storage lists
        self.replay_buffers = []
        self.train_masks = []
        self.samplers = []
        self.sampler_lens = []
        
        # Process each zarr file
        for zarr_path in zarr_paths:
            # Create replay buffer
            replay_buffer = StreamingReplayBuffer.copy_from_path(
                zarr_path, keys=['image', 'state', 'instruction', 'action', 'extrinsic'])
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
                key_first_k=dict(image=frequency+1))
            self.samplers.append(sampler)
            
            # Record sampler length
            self.sampler_lens.append(len(sampler))

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

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
                key_first_k=dict(image=self.frequency+1))
            val_set.samplers.append(sampler)
            val_set.train_masks.append(~self.train_masks[i])
            val_set.sampler_lens.append(len(sampler))
            
        return val_set

    def _process_image_batch(self, images):
        """Process images in batch"""
        rgb = torch.from_numpy(images[..., :3]).float()  # [T, H, W, 3]
        
        # Process RGB
        rgb = rgb.permute(0, 3, 1, 2)  # [T, 3, H, W]

        rgb = F.interpolate(
            rgb / 255.0,
            size=self.image_size,
            mode='bilinear',
            align_corners=False
        )

        return rgb.numpy()
    
    def _sample_to_data(self, sample):
        hand_state = sample['state/hand'].astype(np.float32)
        wrist_state = sample['state/wrist'].astype(np.float32)
        instruction = str(sample['instruction'][self.frequency]) 
        wrist_action = sample['action/wrist'].astype(np.float32)
        hand_action = sample['action/hand'].astype(np.float32)
        # [Horizon, 16] -> [Horizon, 4, 4]
        extrinsic = sample['extrinsic'].astype(np.float32).reshape(-1, 4, 4)

        T_slice = [i for i in range(0, self.frequency + 1, self.frequency // (self.n_obs_steps - 1))]

        # Process all images in batch
        # processed_frames = self._process_image_batch(sample['image'][T_slice])
        processed_results = self.preprocessor(image=sample['image'][T_slice], instruction=instruction)
        processed_frames = processed_results['image'] # [T, H, W, 3]
        tokenized_instruction = processed_results['input_ids']

        processed_wrist_state = transform_wrist_to_target_frame(wrist_state[T_slice], extrinsic[self.frequency])

        processed_wrist_action = wrist_action[self.frequency:]
        processed_wrist_action = transform_wrist_to_target_frame(processed_wrist_action, extrinsic[self.frequency])

        data = {
            'image': processed_frames, 
            # we assume the frequency of the data is 30 Hz, the image should cover the past 1 second
            'state/wrist': processed_wrist_state,
            'state/hand': hand_state[T_slice], 
            'instruction': tokenized_instruction,
            'action/wrist': processed_wrist_action,
            'action/hand': hand_action[self.frequency:]
        }
        return data

    def set_preprocessor(self, preprocessor: BaseVLPreprocessor):
        self.preprocessor = preprocessor

    def get_normalizer(self, mode='limits', **kwargs):
        # Merge all data
        # TODO: Use StreamingReplayBuffer to calculate the normalizer
        wrist_actions = []
        hand_actions = []
        wrist_states = []
        hand_states = []
        for rb in self.replay_buffers:
            wrist_actions.append(rb['action/wrist'][..., :6])
            hand_actions.append(rb['action/hand'])
            wrist_states.append(rb['state/wrist'][..., :6])
            hand_states.append(rb['state/hand'])
            
        data = {
            'action/wrist_trans': np.concatenate(wrist_actions, axis=0),
            'action/hand': np.concatenate(hand_actions, axis=0),
            'state/wrist_trans': np.concatenate(wrist_states, axis=0),
            'state/hand': np.concatenate(hand_states, axis=0)
        }
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer
    
    def get_collator(self):
        return VLDataCollator(pad_token_id=self.preprocessor.tokenizer.pad_token_id)

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
        

class VLDataCollator(BaseDataCollator):
    def __init__(self, pad_token_id: int):
        super().__init__()
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        """
        DataLoader will pass a list of samples from the Dataset to this function.
        Args:
            features: a list, where each element is the return value of the Dataset's __getitem__ method.
               e.g., [{'image': tensor, 'input_ids': tensor}, {'image': tensor, 'input_ids': tensor}, ...]
        Returns:
            A dictionary with the keys the same as the return value of the Dataset's __getitem__ method.
        """
        batch = {}
        for key in features[0].keys():
            if key != 'instruction':
                batch[key] = torch.stack([item[key] for item in features])
            else:
                input_ids_batch = [item[key] for item in features]
                batch["input_ids"] = rnn_utils.pad_sequence(
                    input_ids_batch,
                    batch_first=True,
                    padding_value=self.pad_token_id
                )
                attention_mask_batch = (batch["input_ids"] != self.pad_token_id).long()
                batch["attention_mask"] = attention_mask_batch

        return batch
