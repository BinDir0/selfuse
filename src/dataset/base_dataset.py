from typing import Dict, List, Any

import torch
import torch.nn
from src.model.common.normalizer import LinearNormalizer

class BaseLowdimDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseLowdimDataset':
        # return an empty dataset by default
        return BaseLowdimDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: T, Do
            action: T, Da
        """
        raise NotImplementedError()


class BaseImageDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseLowdimDataset':
        # return an empty dataset by default
        return BaseImageDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key: T, *
            action: T, Da
        """
        raise NotImplementedError()


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

