from typing import Optional, Iterator
import numpy as np
import numba
import torch
from .replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
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
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length:int,
        pad_before:int=0,
        pad_after:int=0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray]=None,
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert(sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
            )
        else:
            indices = np.zeros((0,4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices 
        self.keys = list(keys) # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
    
    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=0, dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                except Exception as e:
                    import pdb; pdb.set_trace()
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result


class RatioSampler(torch.utils.data.BatchSampler):
    """
    Sampler for LegendUnifiedDataset that maintains a specified ratio between VLA and VLM samples.
    
    This sampler ensures that each epoch contains samples from both datasets according to the
    specified ratio, and optionally maintains this ratio within each batch.
    
    Example:
        >>> sampler = RatioSampler(
        ...     vla_size=1000,
        ...     vlm_size=500,
        ...     vla_ratio=0.7,  # 70% VLA, 30% VLM
        ...     batch_size=32,
        ...     per_batch_ratio=True,  # Maintain ratio within each batch
        ...     shuffle=True,
        ...     seed=42
        ... )
        >>> dataloader = DataLoader(dataset, batch_sampler=sampler)
    """
    
    def __init__(
        self,
        vla_size: int,
        vlm_size: int = 0,
        vla_ratio: float = 1.0,
        batch_size: int = 32,
        per_batch_ratio: bool = True,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        """
        Initialize RatioSampler.
        
        Args:
            vla_size: Number of samples in VLA dataset
            vlm_size: Number of samples in VLM dataset (0 means VLA only)
            vla_ratio: Ratio of VLA samples (0.0-1.0). E.g., 0.7 means 70% VLA, 30% VLM
            batch_size: Batch size
            per_batch_ratio: If True, maintain ratio within each batch. If False, only maintain
                           epoch-level ratio.
            shuffle: Whether to shuffle samples
            seed: Random seed for reproducibility
            drop_last: Whether to drop the last incomplete batch
        """
        self.vla_size = vla_size
        self.vlm_size = vlm_size
        self.vla_ratio = vla_ratio
        self.batch_size = batch_size
        self.per_batch_ratio = per_batch_ratio
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        # Validate inputs
        if vlm_size == 0:
            self.vla_ratio = 1.0
        assert 0.0 <= self.vla_ratio <= 1.0, f"vla_ratio must be in [0, 1], got {self.vla_ratio}"
        
        # If per_batch_ratio is True, vla_ratio * batch_size must be an integer
        if per_batch_ratio:
            vla_per_batch = self.vla_ratio * batch_size
            assert vla_per_batch == int(vla_per_batch), \
                f"When per_batch_ratio=True, vla_ratio * batch_size must be an integer. " \
                f"Got vla_ratio={self.vla_ratio}, batch_size={batch_size}, " \
                f"vla_ratio * batch_size={vla_per_batch}"
        
        # Calculate samples per epoch based on ratio
        # Each epoch length = vla_size + vlm_size, sample with replacement if needed
        if vlm_size == 0:
            # VLA only
            self.total_samples = vla_size
            self.vla_samples_per_epoch = vla_size
            self.vlm_samples_per_epoch = 0
        else:
            # Total samples per epoch = sum of both dataset sizes
            self.total_samples = vla_size + vlm_size
            # Calculate how many samples from each dataset based on ratio
            self.vla_samples_per_epoch = int(self.total_samples * vla_ratio)
            self.vlm_samples_per_epoch = self.total_samples - self.vla_samples_per_epoch
        
        # Calculate number of batches
        if drop_last:
            self.num_batches = self.total_samples // batch_size
        else:
            self.num_batches = (self.total_samples + batch_size - 1) // batch_size
    
    def __iter__(self) -> Iterator[list]:
        """Generate batches of indices."""
        # Set random seed for this epoch
        rng = np.random.default_rng(seed=self.seed + self.epoch)
        
        if self.vlm_size == 0:
            # VLA only mode
            vla_indices = np.arange(self.vla_samples_per_epoch)
            if self.shuffle:
                rng.shuffle(vla_indices)
            
            # Create batches
            for i in range(self.num_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, self.vla_samples_per_epoch)
                batch_indices = vla_indices[start_idx:end_idx].tolist()
                
                # If batch is incomplete, fill from the beginning
                if len(batch_indices) < self.batch_size:
                    remaining = self.batch_size - len(batch_indices)
                    batch_indices.extend(vla_indices[:remaining].tolist())
                
                yield batch_indices
        
        else:
            # Generate indices for VLA and VLM datasets with replacement if needed
            # Sample vla_samples_per_epoch samples from [0, vla_size) with replacement
            vla_indices = rng.choice(self.vla_size, size=self.vla_samples_per_epoch, replace=True)
            # Sample vlm_samples_per_epoch samples from [vla_size, vla_size + vlm_size) with replacement
            vlm_indices = rng.choice(self.vlm_size, size=self.vlm_samples_per_epoch, replace=True) + self.vla_size
            
            # Shuffle the sampled indices if needed (already randomized by choice, but shuffle order)
            if self.shuffle:
                rng.shuffle(vla_indices)
                rng.shuffle(vlm_indices)
            
            if self.per_batch_ratio:
                # Maintain ratio within each batch
                vla_per_batch = int(self.batch_size * self.vla_ratio)
                vlm_per_batch = self.batch_size - vla_per_batch
                
                vla_idx = 0
                vlm_idx = 0
                
                for i in range(self.num_batches):
                    batch_indices = []
                    
                    # Sample VLA
                    vla_end = min(vla_idx + vla_per_batch, self.vla_samples_per_epoch)
                    if vla_end > vla_idx:
                        batch_indices.extend(vla_indices[vla_idx:vla_end])
                    vla_idx = vla_end
                    
                    # Sample VLM
                    vlm_end = min(vlm_idx + vlm_per_batch, self.vlm_samples_per_epoch)
                    if vlm_end > vlm_idx:
                        batch_indices.extend(vlm_indices[vlm_idx:vlm_end])
                    vlm_idx = vlm_end
                    
                    # Skip incomplete batch if drop_last is True
                    if self.drop_last and len(batch_indices) < self.batch_size:
                        continue
                    
                    # If batch is incomplete, fill from the beginning maintaining ratio
                    if len(batch_indices) < self.batch_size:
                        remaining = self.batch_size - len(batch_indices)
                        vla_needed = int(remaining * self.vla_ratio)
                        vlm_needed = remaining - vla_needed
                        
                        # Fill VLA from beginning
                        if vla_needed > 0:
                            batch_indices.extend(vla_indices[:vla_needed].tolist())
                        
                        # Fill VLM from beginning
                        if vlm_needed > 0:
                            batch_indices.extend(vlm_indices[:vlm_needed].tolist())
                    
                    # Shuffle within batch to mix VLA and VLM samples
                    if self.shuffle:
                        rng.shuffle(batch_indices)
                    
                    yield batch_indices.tolist()
            
            else:
                # Epoch-level ratio only, mix all samples and batch sequentially
                all_indices = np.concatenate([vla_indices, vlm_indices])
                if self.shuffle:
                    rng.shuffle(all_indices)
                
                for i in range(self.num_batches):
                    start_idx = i * self.batch_size
                    end_idx = min(start_idx + self.batch_size, self.total_samples)
                    batch_indices = all_indices[start_idx:end_idx].tolist()
                    
                    # Skip incomplete batch if drop_last is True
                    if self.drop_last and len(batch_indices) < self.batch_size:
                        continue
                    
                    # If batch is incomplete, fill from the beginning
                    if len(batch_indices) < self.batch_size:
                        remaining = self.batch_size - len(batch_indices)
                        batch_indices.extend(all_indices[:remaining].tolist())
                    
                    yield batch_indices
    
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
