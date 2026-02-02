from typing import Optional, Iterator, List, Literal
import numpy as np
import numba
import torch


@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray, 
    episode_mask: np.ndarray,
) -> np.ndarray:
    """
    Build per-step index records for masked episodes.

    Args:
        episode_ends (np.ndarray): 1D array of episode end indices.
        episode_mask (np.ndarray): 1D bool mask indicating valid episodes.

    Returns:
        np.ndarray: Array of shape (N, 3) with rows
            [buffer_idx, episode_start_idx, episode_end_idx].
    """
    indices = list()
    
    # Loop over every episode
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # Skip episodes marked as invalid in the mask
            continue
            
        # Determine the absolute start and end indices of the current episode in the buffer
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        
        for t in range(start_idx, end_idx):
            indices.append((t, start_idx, end_idx))
                
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed = 0):
    """
    Create a boolean mask for validation episodes.

    Args:
        n_episodes (int): Total number of episodes.
        val_ratio (float): Fraction of episodes to use for validation.
        seed (int): Random seed.

    Returns:
        np.ndarray: Boolean mask with True for validation episodes.
    """
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    assert n_val > 0 and n_val < n_episodes, f"n_val must be in [1, n_episodes-1], got {n_val}"
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed = 0):
    """
    Downsample a boolean mask to at most max_n True values.

    Args:
        mask (np.ndarray): Boolean mask to downsample.
        max_n (Optional[int]): Maximum number of True values to keep.
        seed (int): Random seed.

    Returns:
        np.ndarray: Downsampled boolean mask.
    """
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask


class SequenceSampler:
    """
    Sample fixed-length sequences around a time index from a replay buffer.

    The sampler returns Future-like objects for async IO. Call .result()
    where actual data is needed.
    """
    def __init__(
        self, 
        replay_buffer, 
        num_image_steps: int = 1, 
        num_image_stride: int = 30, 
        num_state_steps: int = 16, 
        num_state_stride: int = 2, 
        num_action_steps: int = 32,
        num_action_stride: int = 1,
        keys = None,
        episode_mask: Optional[np.ndarray] = None,
    ):
        """
        Args:
            replay_buffer (StreamingReplayBuffer): Data source.
            num_image_steps (int): Number of image timesteps to sample.
            num_image_stride (int): Stride between image timesteps.
            num_state_steps (int): Number of state timesteps to sample.
            num_state_stride (int): Stride between state timesteps.
            num_action_steps (int): Number of action timesteps to sample.
            num_action_stride (int): Stride between action timesteps.
            keys (Optional[List[str]]): Keys to include; defaults to all.
            episode_mask (Optional[np.ndarray]): Boolean mask over episodes.
        """
        self.replay_buffer = replay_buffer
        self.cfg = {
            'image':  (num_image_steps, num_image_stride, 'before'),
            'state':  (num_state_steps, num_state_stride, 'before'),
            'action': (num_action_steps, num_action_stride, 'after'),
        }
        
        if keys is None:
            keys = list(replay_buffer.keys())
        self.keys = list(keys)

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)
            
        if np.any(episode_mask):
            self.indices = create_indices(episode_ends, episode_mask)
        else:
            self.indices = np.zeros((0, 3), dtype=np.int64)

    def __len__(self):
        """
        Returns number of available anchors (steps).

        Returns:
            int: Number of sampleable anchors.
        """
        return len(self.indices)

    def _get_query_indices(self, anchor, start, end, steps, stride, side):
        """
        Calculate indices to sample for a given anchor.

        Args:
            anchor (int): Current step index.
            start (int): Episode start index (inclusive).
            end (int): Episode end index (exclusive).
            steps (int): Number of steps to sample.
            stride (int): Stride between steps.
            side (Literal["before","after"]): Sampling direction.

        Returns:
            np.ndarray: 1D array of indices to read.
        """
        if side == 'before':
            offsets = np.arange((1 - steps) * stride, 1, stride)
            idxs = anchor + offsets
            idxs = idxs[idxs >= start]
        else:
            offsets = np.arange(0, steps * stride, stride)
            idxs = anchor + offsets
            idxs = idxs[idxs < end]
            
        return idxs

    def sample_sequence(self, idx):
        """
        Sample a sequence around a selected anchor.

        Args:
            idx (int): Index into sampler indices.

        Returns:
            Dict[str, Future]: Mapping of keys to Future-like objects.
        """
        t_now, ep_start, ep_end = self.indices[idx]
        result = dict()

        for key in self.keys:
            type_cfg = None
            if any(k in key for k in ['image', 'depth']):
                type_cfg = self.cfg['image'] 
            elif 'state' in key:
                type_cfg = self.cfg['state']
            elif 'action' in key:
                type_cfg = self.cfg['action']
            
            if type_cfg:
                steps, stride, side = type_cfg
                idxs = self._get_query_indices(
                    t_now, ep_start, ep_end, steps, stride, side
                )
                # Return Future-like objects for async IO; caller should use .result().
                result[key] = self.replay_buffer.read_async(key, idxs)
            
            else:
                # Return Future-like objects for async IO; caller should use .result().
                result[key] = self.replay_buffer.read_async(key, t_now)
                
        return result


class VLASampler(torch.utils.data.BatchSampler):
    """
    Sampler for VLA datasets that supports weighted sampling from multiple sub-datasets.
    
    This sampler generates a mapping that combines samples from multiple sub-datasets according
    to specified weights. The mapping is regenerated at the start of each epoch (via set_epoch)
    to ensure different sample combinations across epochs.
    
    Sampling strategy:
    1. Normalizes the provided weights to sum to 1.0
    2. Calculates how many samples to draw from each sub-dataset based on weights and total_length
    3. Samples indices from each sub-dataset with replacement (allows oversampling)
    4. Shuffles the combined mapping to randomize sample order
    5. Returns batches of indices that can be used to index into the unified dataset
    
    Example:
        >>> sampler = VLASampler(
        ...     weights=[0.5, 0.5],  # Equal weight for 2 sub-datasets
        ...     dataset_lengths=[1000, 2000],  # Lengths of each sub-dataset
        ...     total_length=3000,  # Total length of the mapping
        ...     batch_size=32,
        ...     shuffle=True,
        ...     seed=42
        ... )
        >>> sampler.set_epoch(0)  # Regenerate mappings for epoch 0
        >>> for batch_indices in sampler:
        ...     # batch_indices is a list of integer indices [0, 1, 2, ...]
        ...     # These indices correspond to positions in the unified mapping
        ...     pass
    """
    
    def __init__(
        self,
        weights: List[float],
        dataset_lengths: List[int],
        total_length: int,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        """
        Initialize VLASampler.
        
        Args:
            weights: List of weights for each sub-dataset (will be normalized)
            dataset_lengths: List of lengths for each sub-dataset
            total_length: Total length of the mapping to generate
            batch_size: Batch size
            shuffle: Whether to shuffle samples
            seed: Random seed for reproducibility
            drop_last: Whether to drop the last incomplete batch
        """
        self.weights = weights
        weights_sum = sum(weights)
        assert weights_sum > 0, "Weights must be non-zero"
        self.weights = [weight / weights_sum for weight in weights]
        self.dataset_lengths = dataset_lengths
        self.total_length = total_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        assert len(dataset_lengths) == len(self.weights), \
            "Number of datasets and weights must match"
        assert total_length > 0, "total_length must be positive"
        
        # Generate initial mappings
        self.mappings = self._get_mappings(self.dataset_lengths, self.total_length, self.seed)
        
        # Calculate number of batches
        if drop_last:
            self.num_batches = total_length // batch_size
        else:
            self.num_batches = (total_length + batch_size - 1) // batch_size
    
    def _get_mappings(self, dataset_lengths: List[int], total_length: int, seed: int) -> List[tuple]:
        """
        Generate a mapping that combines samples from multiple sub-datasets.
        
        The mapping is a list of sample indices, where:
        - sample indices: Index of the sample within the unified dataset
        
        The number of samples from each sub-dataset is proportional to its weight.
        Samples are drawn with replacement to allow oversampling smaller datasets.
        
        Args:
            dataset_lengths: List of lengths for each sub-dataset
            total_length: Total length of the mapping to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of sample indices representing the unified mapping
        """
        rng = np.random.default_rng(seed=seed)
        mapping = []
        counts = [int(p * total_length) for p in self.weights]
        
        # Fill the error (since the integer may be less, add all to the first)
        diff = total_length - sum(counts)
        if diff > 0:
            counts[0] += diff
            
        # Generate random indices for each dataset
        dataset_total_length = 0
        for dataset_idx, count in enumerate(counts):
            dataset_len = dataset_lengths[dataset_idx]
            
            # Random sampling (Replacement=True allows duplicates, to implement oversampling)
            if count > 0:
                sample_indices = rng.choice(dataset_len, size=count, replace=True) + dataset_total_length
                mapping.extend(sample_indices.tolist())
            dataset_total_length += dataset_len
        
        # Shuffle the mapping
        # This way the DataLoader reads in a random order
        rng.shuffle(mapping)
        
        return mapping
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Generate batches of integer indices in a streaming fashion.
        
        Returns batches of indices that can be used to index into the dataset.
        Each index corresponds to a position in the mappings list.
        """
        # Generate batches in streaming fashion
        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.mappings))
            batch_indices = self.mappings[start_idx:end_idx]
            
            # Skip incomplete batch if drop_last is True
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            
            # If batch is incomplete (last batch), fill from the beginning
            if len(batch_indices) < self.batch_size:
                remaining = self.batch_size - len(batch_indices)
                batch_indices.extend(self.mappings[:remaining])
            
            yield batch_indices
    
    def __len__(self) -> int:
        """
        Return number of batches.

        Returns:
            int: Number of batches.
        """
        return self.num_batches
    
    def set_epoch(self, epoch: int):
        """
        Set the epoch and regenerate mappings.

        Args:
            epoch (int): Epoch number.

        Returns:
            None
        """
        self.epoch = epoch
        # Regenerate mappings for this epoch
        self.mappings = self._get_mappings(self.dataset_lengths, self.total_length, self.seed + self.epoch)


class UnifiedRatioSampler(torch.utils.data.BatchSampler):
    """
    Sampler for LegendUnifiedDataset that maintains a specified ratio between VLA and VLM samples.
    
    This sampler uses VLASampler for VLA data sampling and maintains the ratio within each batch.
    
    Example:
        >>> sampler = UnifiedRatioSampler(
        ...     weights=[0.5, 0.5],
        ...     dataset_lengths=[1000, 2000],
        ...     vla_size=3000,
        ...     vlm_size=500,
        ...     vla_ratio=0.7,  # 70% VLA, 30% VLM
        ...     batch_size=32,
        ...     shuffle=True,
        ...     seed=42
        ... )
        >>> dataloader = DataLoader(dataset, batch_sampler=sampler)
    """
    
    def __init__(
        self,
        weights: List[float],
        dataset_lengths: List[int],
        vla_size: int,
        vlm_size: int = 0,
        vla_ratio: float = 1.0,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        """
        Initialize UnifiedRatioSampler.
        
        Args:
            weights: List of weights for each VLA sub-dataset (will be normalized)
            dataset_lengths: List of lengths for each VLA sub-dataset
            vla_size: Total number of samples in VLA dataset
            vlm_size: Number of samples in VLM dataset (0 means VLA only)
            vla_ratio: Ratio of VLA samples (0.0-1.0). E.g., 0.7 means 70% VLA, 30% VLM
            batch_size: Batch size
            shuffle: Whether to shuffle samples
            seed: Random seed for reproducibility
            drop_last: Whether to drop the last incomplete batch
        """
        self.vlm_size = vlm_size
        self.vla_ratio = vla_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        # Store VLA size
        self.vla_size = vla_size
        
        # Validate inputs
        if vlm_size == 0:
            self.vla_ratio = 1.0
        assert 0.0 <= self.vla_ratio <= 1.0, f"vla_ratio must be in [0, 1], got {self.vla_ratio}"
        
        # vla_ratio * batch_size must be an integer to maintain ratio within each batch
        vla_per_batch = int(self.vla_ratio * batch_size)
        assert vla_per_batch > 0, f"vla_ratio * batch_size must be positive, got {vla_per_batch}"
        
        # Calculate samples per epoch based on ratio
        # Each epoch length = vla_size + vlm_size, sample with replacement if needed
        if vlm_size == 0:
            # VLA only
            self.total_samples = self.vla_size
        else:
            # Total samples per epoch = sum of both dataset sizes
            self.total_samples = self.vla_size + vlm_size
        if drop_last: 
            self.num_batches = self.total_samples // batch_size
        else:
            self.num_batches = (self.total_samples + batch_size - 1) // batch_size
        # Make sure the total number of samples is a multiple of batch_size
        self.total_samples = self.num_batches * batch_size
        # Calculate how many samples from each dataset based on ratio
        self.vla_samples_per_batch = int(batch_size * self.vla_ratio)
        self.vlm_samples_per_batch = batch_size - self.vla_samples_per_batch
        self.vla_samples_per_epoch = self.vla_samples_per_batch * self.num_batches
        self.vlm_samples_per_epoch = self.vlm_samples_per_batch * self.num_batches
        
        # Create VLASampler with total_length = vla_samples_per_epoch
        self.vla_sampler = VLASampler(
            weights=weights,
            dataset_lengths=dataset_lengths,
            total_length=self.vla_samples_per_epoch,
            batch_size=self.vla_samples_per_batch,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,  # We handle drop_last at UnifiedRatioSampler level
        )
    
    def __iter__(self) -> Iterator[list]:
        """
        Generate batches of indices, maintaining VLA/VLM ratio.

        Returns:
            Iterator[List[int]]: Batch indices for dataset indexing.
        """
        # Set epoch for VLA sampler
        self.vla_sampler.set_epoch(self.epoch)
        
        # Set random seed for this epoch
        rng = np.random.default_rng(seed=self.seed + self.epoch)
        
        if self.vlm_size == 0:
            # VLA only mode - use VLASampler directly
            for batch_indices in self.vla_sampler:
                yield batch_indices
        
        else:
            # Create iterators for VLA and VLM samples
            vla_iter = iter(self.vla_sampler)
            vlm_indices = rng.choice(self.vlm_size, size=self.vlm_samples_per_epoch, replace=True) + self.vla_size
            if self.shuffle:
                rng.shuffle(vlm_indices)
            vlm_indices = vlm_indices.tolist()
            
            batch_count = 0
            while batch_count < self.num_batches:
                batch_indices = []
                # Get VLA samples
                batch_indices.extend(next(vla_iter))
                # Get VLM samples
                vlm_start_idx = batch_count * self.vlm_samples_per_batch
                vlm_end_idx = vlm_start_idx + self.vlm_samples_per_batch
                batch_indices.extend(vlm_indices[vlm_start_idx:vlm_end_idx])
                
                # Since we always fill the total number of samples to be a multiple of batch_size, 
                # the length of batch_indices should always be equal to the batch_size
                assert len(batch_indices) == self.batch_size, f"Batch size mismatch: {len(batch_indices)} != {self.batch_size}"
                
                # Shuffle within batch to mix VLA and VLM samples
                if self.shuffle:
                    rng.shuffle(batch_indices)
                
                yield batch_indices
                batch_count += 1
    
    def __len__(self) -> int:
        """
        Return number of batches.

        Returns:
            int: Number of batches.
        """
        return self.num_batches
    
    def set_epoch(self, epoch: int):
        """
        Set the epoch for this sampler.

        Args:
            epoch (int): Epoch number.

        Returns:
            None
        """
        self.epoch = epoch
        # Also update epoch for VLA sampler
        if hasattr(self, 'vla_sampler'):
            self.vla_sampler.set_epoch(epoch)
