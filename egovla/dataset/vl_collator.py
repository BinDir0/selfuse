import torch
from typing import List, Dict, Any
import torch.nn.utils.rnn as rnn_utils

from egovla.dataset.base_collator import BaseDataCollator

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
