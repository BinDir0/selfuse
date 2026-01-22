from typing import Dict, List, Any

import numpy as np
import torch

from src.model.common.normalizer import LinearNormalizer

class BaseLowdimDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseLowdimDataset':
        # return an empty dataset by default
        return BaseLowdimDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
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


class BaseRatioDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        weights: List[float], 
        dataset_lengths: List[int], 
        regenerate_mappings_every: int = 10000,
        seed: int = 42,
    ):
        self.weights = weights
        weights_sum = sum(weights)
        assert weights_sum > 0, "Weights must be non-zero"
        self.weights = [weight / weights_sum for weight in weights]
        self.dataset_lengths = dataset_lengths
        self.mappings = self.get_mappings(dataset_lengths, seed)
        self.regenerate_mappings_every = regenerate_mappings_every
        self.iteration = 0
        self.seed = seed

    def get_mappings(self, dataset_length: List[int], seed: int = 42):
        """
        Calculate the mapping of each slot to the corresponding dataset and sample index.
        """
        assert len(dataset_length) == len(self.weights), "Number of datasets and weights must match"
        rng = np.random.default_rng(seed=seed)
        mapping = []
        total_length = sum(dataset_length)
        counts = [int(p * total_length) for p in self.weights]
        
        # Fill the error (since the integer may be less, add all to the first)
        diff = total_length - sum(counts)
        if diff > 0:
            counts[0] += diff
            
        # Generate random indices for each dataset
        for dataset_idx, count in enumerate(counts):
            dataset_len = dataset_length[dataset_idx]
            
            # Random sampling (Replacement=True allows duplicates, to implement oversampling)
            # For simplicity, just use randint (replace=True) is the most robust
            if count > 0:
                sample_indices = rng.choice(dataset_len, size=count, replace=True)
                
                # Store in mapping
                for sample_idx in sample_indices:
                    mapping.append((dataset_idx, sample_idx))
        
        # Shuffle the mapping
        # This way the DataLoader reads in a random order
        rng.shuffle(mapping)
        
        return mapping

    def maybe_update_mappings(self): 
        if self.iteration % self.regenerate_mappings_every == 0:
            self.mappings = self.get_mappings(self.dataset_lengths, self.seed+self.iteration)
        self.iteration += 1

    def get_validation_dataset(self) -> 'BaseRatioDataset':
        # return an empty dataset by default
        return BaseRatioDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
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

