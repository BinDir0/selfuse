'''
Propise dataset for LegendVLA
Every action is the delta of the next predicted absolute state and the state at the beginning of the action chunk.
'''

from typing import Dict
import torch
import numpy as np
import copy
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from src.utils.pytorch_util import dict_apply
from src.utils.streaming_replay_buffer import StreamingReplayBuffer
from src.utils.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from src.utils.geometry import transform_wrist_to_target_frame
from src.model.common.normalizer import LinearNormalizer
from .base_dataset import BaseImageDataset, BaseDataCollator
from .base_vl_preprocessor import BaseVLPreprocessor

class EgoVLADataset(BaseImageDataset):
    def __init__(self,
            zarr_paths,
            horizon=1,
            pad_before=0,
            pad_after=0,
            shape_meta=None,
            seed=42,
            val_ratio=0.0,
            history=30,
            max_train_episodes=None,
            image_size=(384, 384)
            ):
        
        super().__init__()
        self.preprocessor = None
        self.image_size = image_size
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
                key_first_k=dict(image=history+1))
            self.samplers.append(sampler)
            
            # Record sampler length
            self.sampler_lens.append(len(sampler))

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.shape_meta = shape_meta
        self.n_obs_image_steps = shape_meta['obs']['rgb']['horizon']
        self.n_obs_state_steps = shape_meta['obs']['state']['horizon']

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
                key_first_k=dict(image=self.history+1))
            val_set.samplers.append(sampler)
            val_set.train_masks.append(~self.train_masks[i])
            val_set.sampler_lens.append(len(sampler))
            
        return val_set
    
    def _sample_to_data(self, sample):
        hand_state = sample['state/hand'].astype(np.float32)
        wrist_state = sample['state/wrist'].astype(np.float32)
        instruction = str(sample['instruction'][self.history]) 
        wrist_action = sample['action/wrist'].astype(np.float32)
        hand_action = sample['action/hand'].astype(np.float32)
        # [Horizon, 16] -> [Horizon, 4, 4]
        extrinsic = sample['extrinsic'].astype(np.float32).reshape(-1, 4, 4)

        image_slice = [i for i in range(0, self.history + 1, self.history // (self.n_obs_image_steps - 1))]
        state_slice = [i for i in range(0, self.history + 1, self.history // (self.n_obs_state_steps - 1))]

        # Process all images in batch
        # processed_frames = self._process_image_batch(sample['image'][T_slice])
        processed_results = self.preprocessor(image=sample['image'][image_slice], instruction=instruction)
        processed_frames = processed_results['pixel_values'] # [T, C, H, W]
        tokenized_instruction = processed_results['input_ids'] # [L]
        attention_mask = processed_results['attention_mask'] # [L]

        processed_wrist_state = transform_wrist_to_target_frame(wrist_state[state_slice], extrinsic[self.history])

        processed_wrist_action = wrist_action[self.history:]
        processed_wrist_action = transform_wrist_to_target_frame(processed_wrist_action, extrinsic[self.history])

        # use delta of wrist translation and hand mano params as action
        processed_wrist_action[..., :6] = processed_wrist_action[..., :6] - processed_wrist_state[-1, :6]
        processed_hand_state = hand_state[state_slice]
        processed_hand_action = hand_action[self.history:, :] - processed_hand_state[-1, :]

        data = {
            'input_id': tokenized_instruction,
            'attention_mask': attention_mask,
            'pixel_value': processed_frames, 
            # we assume the history of the data is 30 Hz, the image should cover the past 1 second
            'proprio': np.concatenate([processed_wrist_state, processed_hand_state], axis=-1),
            'human_action': np.concatenate([processed_wrist_action, processed_hand_action], axis=-1),
        }
        return data

    def set_preprocessor(self, preprocessor: BaseVLPreprocessor):
        self.preprocessor = preprocessor

    def get_normalizer(self, mode='limits', **kwargs):
        # Merge all data
        # TODO: Use StreamingReplayBuffer to calculate the normalizer
        hand_actions = []
        hand_states = []
        # TODO: add delta of action 
        for rb in self.replay_buffers:
            hand_states.append(rb['state/hand'])

        for idx in range(len(self)): 
            sample = self[idx]
            wrist_dim = self.shape_meta['obs']['state']['wrist']['shape'][0]
            hand_action = sample['human_action'][:, wrist_dim:]
            hand_actions.append(hand_action.cpu().numpy())
            
        data = {
            'action/hand': np.concatenate(hand_actions, axis=0),
            'state/hand': np.concatenate(hand_states, axis=0)
        }
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer
    
    def get_collator(self):
        return LegendVLADataCollator()

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
        

class LegendVLADataCollator(BaseDataCollator):
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
                # TODO: change to max length padding
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
