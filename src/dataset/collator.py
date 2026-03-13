'''
Data collators for LegendVLA datasets.
'''

from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils


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
        for key in data_list[0].keys():
            if key in ['input_ids', 'attention_mask', 'labels']:
                continue
            if key in ['pixel_values', 'depth_values']:
                tensors = [item[key] for item in data_list]
                max_len = max(t.shape[0] for t in tensors)
                padded_tensors = []
                for t in tensors:
                    pad_len = max_len - t.shape[0]
                    if pad_len > 0:
                        # left padding with the first frame
                        padding = t[0:1].expand(pad_len, *t.shape[1:])
                        padded_tensors.append(torch.cat([padding, t], dim=0))
                    else:
                        padded_tensors.append(t)
                batch[key] = torch.stack(padded_tensors)
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
