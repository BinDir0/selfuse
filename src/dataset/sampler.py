from typing import Optional, Iterator, List
import numpy as np
import numba
import torch
from .replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray, 
    sequence_length: int, 
    episode_mask: np.ndarray,
    pad_before: int = 0, 
    pad_after: int = 0,
    debug: bool = True,
) -> np.ndarray:
    """
    Pre-calculates valid sampling indices for all episodes using Numba for performance.
    
    Returns an array of shape (N, 4) where each row contains:
    [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
    """
    # Ensure mask matches ends shape
    # Note: In nopython mode, assert works, but logic like shape check usually done before.
    # Here it's effectively a no-op check syntax-wise in some Numba versions unless asserted.
    
    # Clamp padding values to be reasonable (cannot exceed sequence length)
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

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
        episode_length = end_idx - start_idx
        
        # Calculate the range of valid starting positions for the sliding window relative to the episode start.
        # min_start: allows the window to start before the episode actually begins (for padding).
        # max_start: allows the window to end after the episode actually ends.
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # Iterate through the sliding window positions
        for idx in range(min_start, max_start+1):
            # Calculate actual data range in the buffer (clamped to episode boundaries)
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            
            # Calculate how much padding is needed at the beginning and end
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            
            # map the buffer data to the output sequence array indices
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            
            if debug:
                assert start_offset >= 0, f"start_offset must be >= 0, got {start_offset}"
                assert end_offset >= 0, f"end_offset must be >= 0, got {end_offset}"
                # Verify length consistency: Valid data length must match destination length
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx), \
                    f"Sample length mismatch: {sample_end_idx - sample_start_idx} != {buffer_end_idx - buffer_start_idx}"
            
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
                
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed = 0):
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
    def __init__(
        self, 
        replay_buffer, # Type: ReplayBuffer 
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys = None,
        key_first_k = dict(),
        episode_mask: Optional[np.ndarray] = None,
    ):
        """
        Args:
            replay_buffer: The dataset buffer containing episodes.
            sequence_length: The length of time steps to sample per sample.
            pad_before: Number of steps to pad at the beginning if sampling starts before episode.
            pad_after: Number of steps to pad at the end if sampling goes beyond episode.
            keys: List of keys (e.g., 'obs', 'action') to extract. If None, uses all.
            key_first_k: Dict mapping key names to integers. For these keys, only the first k steps
                         of the sequence are loaded efficiently (useful for Obs that don't need history).
            episode_mask: Boolean array to include/exclude specific episodes.
        """

        super().__init__()
        assert sequence_length >= 1, f"sequence_length must be >= 1, got {sequence_length}"
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        # Pre-calculate all valid indices mapping (buffer_idx -> sample_idx)
        if np.any(episode_mask):
            indices = create_indices(
                episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
            )
        else:
            indices = np.zeros((0,4), dtype=np.int64)

        # Store indices: (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices 
        self.keys = list(keys) # Convert to list to avoid OmegaConf performance issues if passed as config
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
    
    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
        """
        Extracts a single sequence based on the pre-calculated index `idx`.
        Handles reading from buffer and padding edges.
        """
        # Unpack the pre-calculated indices
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
            
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            
            # --- Performance Optimization ---
            # If the key is not in key_first_k, read the whole slice normally.
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # If key is in key_first_k, we only need the first few steps (e.g., current obs).
                # This avoids loading huge chunks of unused data (like images for future steps).
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                
                # Initialize with zeros (or custom fill) to ensure shape consistency
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=0, dtype=input_arr.dtype)
                try:
                    # Only load the necessary k data
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                except Exception as e:
                    import pdb; pdb.set_trace()
            
            data = sample
            
            # --- Padding Logic ---
            # If the valid data read from buffer is shorter than sequence_length
            # (due to padding at start or end), we need to fill the rest.
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                # Allocate full sequence array
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                
                # Fill edge padding: repeat first element for the beginning padding
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                
                # Fill edge padding: repeat last element for the ending padding
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                
                # Place the actual valid data in the middle
                data[sample_start_idx:sample_end_idx] = sample
                
            result[key] = data
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
        """Return number of batches."""
        return self.num_batches
    
    def set_epoch(self, epoch: int):
        """
        Set the epoch for this sampler and regenerate mappings.
        
        This ensures different shuffling and mapping in each epoch when using DistributedSampler.
        
        Args:
            epoch: Epoch number
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
        vla_per_batch = self.vla_ratio * batch_size
        assert vla_per_batch == int(vla_per_batch), \
            f"vla_ratio * batch_size must be an integer. " \
            f"Got vla_ratio={self.vla_ratio}, batch_size={batch_size}, " \
            f"vla_ratio * batch_size={vla_per_batch}"
        
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
        self.vla_samples_per_epoch = int(self.total_samples * vla_ratio)
        self.vlm_samples_per_epoch = self.total_samples - self.vla_samples_per_epoch
        self.vla_samples_per_batch = int(batch_size * vla_ratio)
        self.vlm_samples_per_batch = batch_size - self.vla_samples_per_batch
        assert self.vla_samples_per_epoch % self.vla_samples_per_batch == 0, \
            f"vla_samples_per_epoch {self.vla_samples_per_epoch} must be a multiple of vla_samples_per_batch {self.vla_samples_per_batch}"
        assert self.vlm_samples_per_epoch % self.vlm_samples_per_batch == 0, \
            f"vlm_samples_per_epoch {self.vlm_samples_per_epoch} must be a multiple of vlm_samples_per_batch {self.vlm_samples_per_batch}"
        
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
        """Generate batches of indices in a streaming fashion, maintaining ratio within each batch."""
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
        """Return number of batches."""
        return self.num_batches
    
    def set_epoch(self, epoch: int):
        """
        Set the epoch for this sampler. 
        
        This ensures different shuffling in each epoch when using DistributedSampler.
        
        Args:
            epoch: Epoch number
        """
        self.epoch = epoch
        # Also update epoch for VLA sampler
        if hasattr(self, 'vla_sampler'):
            self.vla_sampler.set_epoch(epoch)
