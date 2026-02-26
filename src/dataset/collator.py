'''
Data collators for LegendVLA datasets.

Extracted from legendvla_dataset.py for modularity.
'''

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils

from .base_dataset import BaseDataCollator


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
                values = [item[key] for item in data_list]
                if isinstance(values[0], str):
                    batch[key] = values
                elif isinstance(values[0], (int, float, bool, np.generic)):
                    batch[key] = torch.tensor(values)
                else:
                    batch[key] = torch.stack(values)

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
