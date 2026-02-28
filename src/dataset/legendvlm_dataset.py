'''
LegendVLM Dataset for VLM-only data stored in HuggingFace datasets format.

Extracted from legendvla_dataset.py for modularity.
'''

from typing import Dict, List, Optional, Union
import glob
import pathlib
import torch
import numpy as np
import warnings
from torchvision import transforms
from datasets import concatenate_datasets, interleave_datasets, load_from_disk, load_dataset, DatasetDict
from src.utils.pytorch_util import dict_apply
from src.dataset.collator import LegendVLDataCollator


class LegendVLMDataset(torch.utils.data.Dataset):
    """
    Dataset for VLM-only data stored in HuggingFace datasets format.
    """
    def __init__(
        self,
        dataset_paths,
        split='train',
        cache_dir=None,
        weights=[0.5, 0.5, 0.5],
        seed=42,
        mode='train',
        return_dataset_info: bool = False,
    ):
        """
        Args:
            dataset_paths (Union[str, List[str]]): Dataset disk paths.
            split (str): Split name to load (train/val/test).
            cache_dir (Optional[str]): HF datasets cache directory.
            weights (List[float]): Weights for rating-based text selection.
            seed (int): Random seed.
            mode (str): One of "train" or "val" of "infer-ar" or "infer".
        """
        super().__init__()
        self.dataset_paths = [dataset_paths] if isinstance(dataset_paths, str) else dataset_paths
        self.split = split
        self.weights = weights
        self.cache_dir = cache_dir
        self.mode = mode
        self.seed = seed
        self.preprocessor = None
        self.return_dataset_info = return_dataset_info
        self.dataset_names = []
        self.dataset_lengths = []
        self.dataset_offsets = []

        if self.mode == 'train':
            self.aug_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
            ])
        else:
            self.aug_transform = None

        # --- directly load the dataset according to the split ---
        loaded_datasets = []
        for path in self.dataset_paths:
            try:
                ds = load_from_disk(path)

                # case A: if the loaded dataset is a DatasetDict (contains train/test/val)
                if isinstance(ds, DatasetDict):
                    if self.split in ds:
                        ds_to_add = ds[self.split]
                    else:
                        # if the sub dataset is too small to have this split (e.g. no test), skip it
                        print(f"Warning: Split '{self.split}' not found in {path}, skipping this sub-dataset.")
                        continue
                # case B: if the loaded dataset is a Dataset object
                else:
                    ds_to_add = ds

                if len(ds_to_add) == 0:
                    print(f"Warning: Dataset at {path} is empty, skipping.")
                    continue

                loaded_datasets.append(ds_to_add)
                self.dataset_names.append(pathlib.Path(str(path)).stem)
                self.dataset_lengths.append(len(ds_to_add))

            except Exception as e:
                warnings.warn(f"Error loading dataset from {path}: {e}")
                continue

        if not loaded_datasets:
            # if the validation set loading fails (maybe all sub datasets don't have test), we can throw an exception or return None
            print(f"Notice: No datasets found for split '{self.split}'.")
            self.main_dataset = None
        else:
            # merge all sub datasets that meet the criteria
            self.main_dataset = concatenate_datasets(loaded_datasets)
            print(f"Successfully loaded split '{self.split}' with {len(self.main_dataset)} samples.")
            offset = 0
            for length in self.dataset_lengths:
                self.dataset_offsets.append(offset)
                offset += length

    def get_validation_dataset(self, val_split='test'):
        """
        directly load the dataset with split='test'
        """
        # instantiate a new object, set split to val_split
        val_dataset = LegendVLMDataset(
            dataset_paths=self.dataset_paths,
            split=val_split,
            cache_dir=self.cache_dir,
            weights=self.weights,
            seed=self.seed,
            mode='val' if self.mode == 'train' else self.mode,
            return_dataset_info=self.return_dataset_info,
        )

        # inherit the current preprocessor
        if self.preprocessor is not None:
            val_dataset.set_preprocessor(self.preprocessor)

        # if the corresponding split has no data, return None
        if val_dataset.main_dataset is None:
            return None

        return val_dataset

    def sample_to_data(self, sample, idx):
        """
        Convert a raw dataset row into model-ready fields.

        Args:
            sample (Dict[str, Any]): Dataset row.
            idx (int): Row index (unused).

        Returns:
            Dict[str, np.ndarray]: Processed sample fields.
        """
        images = sample['images'] # List[PIL.JpegImagePlugin.JpegImageFile]
        text = sample['texts']
        weights = self.weights
        # There are some None values in the ratings, we replace them with 0
        formatting_ratings = np.array([rating if rating is not None else 0 for rating in sample['formatting_ratings']])
        visual_dependency_ratings = np.array([rating if rating is not None else 0 for rating in sample['visual_dependency_ratings']])
        relevance_ratings = np.array([rating if rating is not None else 0 for rating in sample['relevance_ratings']])

        if len(text) > 1:
            scores = formatting_ratings * weights[0] + \
                    visual_dependency_ratings * weights[1] + \
                    relevance_ratings * weights[2]
            text = text[np.argmax(scores)]
        else:
            text = text[0]
        question = str(text['user'])
        answer = str(text['assistant'])

        for idx in range(len(images)):
            if images[idx].mode != 'RGB':
                images[idx] = images[idx].convert('RGB')

        augmented_images = []
        for img_pil in images:
            if self.mode == 'train' and self.aug_transform is not None:
                augmented_pil = self.aug_transform(img_pil)
            else:
                augmented_pil = img_pil
            augmented_np = np.array(augmented_pil, dtype=np.uint8)
            augmented_images.append(augmented_np)
        images_to_process = np.stack(augmented_images, dtype=np.uint8)
        # Process all images in batch
        processed_results = self.preprocessor(
            images=images_to_process,
            text=question,
            target=answer,
            mode=self.mode
        )

        data = {
            'input_ids': processed_results['input_ids'],
            'labels': processed_results['labels'],
            'attention_mask': processed_results['attention_mask'],
            'pixel_values': processed_results['pixel_values'],
            'answer_start_idx': processed_results['answer_start_idx'],
            'is_vla_data': np.array(False, dtype=bool),
        }
        return data

    def get_collator(self):
        """
        Build a data collator for batching.

        Returns:
            LegendVLDataCollator: Collator instance.
        """
        assert self.preprocessor is not None, "Preprocessor is not set"
        padding_side = 'left' if self.mode == 'infer-ar' else 'right'
        return LegendVLDataCollator(
            pad_token_id=self.preprocessor.tokenizer.pad_token_id,
            ignore_index=self.preprocessor.ignore_index,
            padding_side=padding_side,
        )

    def set_preprocessor(self, preprocessor):
        """
        Set the tokenizer/vision preprocessor.

        Args:
            preprocessor (Callable): Preprocessor with tokenizer and encode logic.
        """
        self.preprocessor = preprocessor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a processed sample by index.

        Args:
            idx (int): Dataset index.

        Returns:
            Dict[str, torch.Tensor]: Sample tensors.
        """
        # Find corresponding sampler
        try:
            dataset_name = None
            dataset_local_idx = None
            for i, offset in enumerate(self.dataset_offsets):
                length = self.dataset_lengths[i]
                if idx < offset + length:
                    dataset_name = self.dataset_names[i]
                    dataset_local_idx = idx - offset
                    break
            sample = self.main_dataset[idx]
            data = self.sample_to_data(sample, idx)
            if self.return_dataset_info:
                data['dataset_name'] = dataset_name
                data['dataset_local_idx'] = np.array(dataset_local_idx, dtype=np.int32)
            torch_data = dict_apply(
                data, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
            )
            return torch_data
        except Exception as e:
            warnings.warn(f"Error getting item {dataset_local_idx} from dataset {dataset_name}: {e}")
            # backup solution: return the next item
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self):
        """
        Dataset length.

        Returns:
            int: Number of samples.
        """
        return len(self.main_dataset)


class LegendVLMStreamingDataset(torch.utils.data.IterableDataset):
    """
    Streaming VLM dataset using HuggingFace IterableDataset.

    Loads data lazily via load_dataset(streaming=True), reducing memory usage
    compared to load_from_disk() which loads everything into memory.
    """

    def __init__(
        self,
        dataset_paths: Union[str, List[str]],
        split: str = 'train',
        weights: List[float] = [0.5, 0.5, 0.5],
        seed: int = 42,
        mode: str = 'train',
        shuffle_buffer: int = 10000,
        dataset_probs: Optional[List[float]] = None,
        return_dataset_info: bool = False,
    ):
        """
        Args:
            dataset_paths: Dataset paths on disk (arrow-format HF datasets).
            split: Split name to load (train/val/test).
            weights: Weights for rating-based text selection.
            seed: Random seed for shuffling.
            mode: One of "train", "val", "infer-ar", "infer".
            shuffle_buffer: Buffer size for streaming shuffle.
            dataset_probs: Sampling probabilities for interleaving multiple datasets.
                If None, automatically computes proportional-to-size probabilities
                from dataset metadata. Falls back to uniform if metadata unavailable.
            return_dataset_info: If True, return dataset_name and episode_index.
        """
        super().__init__()
        self.dataset_paths = [dataset_paths] if isinstance(dataset_paths, str) else list(dataset_paths)
        self.split = split
        self.weights = weights
        self.seed = seed
        self.mode = mode
        self.shuffle_buffer = shuffle_buffer
        self.dataset_probs = dataset_probs
        self.preprocessor = None
        self.return_dataset_info = return_dataset_info

        if self.mode == 'train':
            self.aug_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
            ])
        else:
            self.aug_transform = None

        # Load each path as a HF IterableDataset
        self.stream = self.build_stream()

    def build_stream(self):
        """Build the streaming dataset from all paths.

        Collects all data files (arrow/parquet) across sub-dataset paths and
        loads them as a single HF IterableDataset. This avoids the
        interleave_datasets min(num_shards) bottleneck that kills workers
        when any sub-dataset has fewer shards than num_workers.

        Training mode: single merged stream with buffer shuffle.
        Eval mode: single merged stream, no shuffle.
        """
        all_files = self.collect_data_files()

        if not all_files:
            warnings.warn(f"No data files found for split '{self.split}'.")
            return None

        # Detect format from file extension
        ext = pathlib.Path(all_files[0]).suffix.lstrip(".")
        if ext not in ("arrow", "parquet"):
            warnings.warn(f"Unsupported file format '{ext}', trying arrow.")
            ext = "arrow"

        try:
            stream = load_dataset(
                ext, data_files=all_files, split="train", streaming=True
            )
        except Exception as e:
            warnings.warn(f"Error loading merged stream: {e}")
            return None

        if self.return_dataset_info:
            stream = stream.map(
                lambda x, idx: {**x, 'episode_index': idx},
                with_indices=True
            )

        if self.mode == 'train':
            stream = stream.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)

        return stream

    def collect_data_files(self):
        """Discover all arrow/parquet data files under each dataset path.

        Searches for files matching the HF datasets layout:
            {path}/{split}/data-*.arrow  (or .parquet)
            {path}/data/data-*.arrow
            {path}/data-*.arrow
            {path}/*.parquet
        """
        all_files = []
        for path in self.dataset_paths:
            found = []
            # HF arrow layout: {path}/{split}/data-*.arrow
            found.extend(sorted(glob.glob(f"{path}/{self.split}/*.arrow")))
            # Parquet fallback
            if not found:
                found.extend(sorted(glob.glob(f"{path}/{self.split}/*.parquet")))
            if not found:
                warnings.warn(f"No data files found in {path}, skipping.")
            all_files.extend(found)
        return all_files

    @staticmethod
    def infer_proportional_probs(streams):
        """Try to read num_examples from dataset metadata for proportional sampling.

        Returns None if metadata is unavailable for any stream.
        """
        sizes = []
        for ds in streams:
            try:
                info = ds.info
                if info is not None and info.splits is not None:
                    # IterableDataset loaded with split= stores split info
                    for split_info in info.splits.values():
                        sizes.append(split_info.num_examples)
                        break
                    else:
                        return None
                else:
                    return None
            except Exception:
                return None
        if not sizes or any(s is None or s == 0 for s in sizes):
            return None
        return [float(s) for s in sizes]

    def distribute(self, rank: int, world_size: int):
        """Apply node-level shard splitting for distributed training.

        Must be called before creating the DataLoader.
        Uses shard-level splitting when #shards >= world_size,
        otherwise falls back to example-level splitting.
        """
        if self.stream is None:
            return
        from datasets.distributed import split_dataset_by_node
        self.stream = split_dataset_by_node(
            self.stream, rank=rank, world_size=world_size
        )

    def set_preprocessor(self, preprocessor):
        """Set the tokenizer/vision preprocessor."""
        self.preprocessor = preprocessor

    def get_collator(self):
        """Build a data collator for batching."""
        assert self.preprocessor is not None, "Preprocessor is not set"
        padding_side = 'left' if self.mode == 'infer-ar' else 'right'
        return LegendVLDataCollator(
            pad_token_id=self.preprocessor.tokenizer.pad_token_id,
            ignore_index=self.preprocessor.ignore_index,
            padding_side=padding_side,
        )

    def get_validation_dataset(self, val_split='test'):
        """Create a new streaming dataset instance for validation."""
        val_dataset = LegendVLMStreamingDataset(
            dataset_paths=self.dataset_paths,
            split=val_split,
            weights=self.weights,
            seed=self.seed,
            mode='val' if self.mode == 'train' else self.mode,
            shuffle_buffer=self.shuffle_buffer,
            dataset_probs=self.dataset_probs,
            return_dataset_info=self.return_dataset_info,
        )
        if self.preprocessor is not None:
            val_dataset.set_preprocessor(self.preprocessor)
        if val_dataset.stream is None:
            return None
        return val_dataset

    def sample_to_data(self, sample):
        """
        Convert a raw streaming dataset row into model-ready fields.

        Reuses the same logic as LegendVLMDataset.sample_to_data.
        """
        images = sample['images']
        text = sample['texts']
        weights = self.weights

        formatting_ratings = np.array(
            [r if r is not None else 0 for r in sample['formatting_ratings']]
        )
        visual_dependency_ratings = np.array(
            [r if r is not None else 0 for r in sample['visual_dependency_ratings']]
        )
        relevance_ratings = np.array(
            [r if r is not None else 0 for r in sample['relevance_ratings']]
        )

        if len(text) > 1:
            scores = (formatting_ratings * weights[0]
                      + visual_dependency_ratings * weights[1]
                      + relevance_ratings * weights[2])
            text = text[np.argmax(scores)]
        else:
            text = text[0]
        question = str(text['user'])
        answer = str(text['assistant'])

        for idx in range(len(images)):
            if images[idx].mode != 'RGB':
                images[idx] = images[idx].convert('RGB')

        augmented_images = []
        for img_pil in images:
            if self.mode == 'train' and self.aug_transform is not None:
                augmented_pil = self.aug_transform(img_pil)
            else:
                augmented_pil = img_pil
            augmented_np = np.array(augmented_pil, dtype=np.uint8)
            augmented_images.append(augmented_np)
        images_to_process = np.stack(augmented_images, dtype=np.uint8)

        processed_results = self.preprocessor(
            images=images_to_process,
            text=question,
            target=answer,
            mode=self.mode,
        )

        data = {
            'input_ids': processed_results['input_ids'],
            'labels': processed_results['labels'],
            'attention_mask': processed_results['attention_mask'],
            'pixel_values': processed_results['pixel_values'],
            'answer_start_idx': processed_results['answer_start_idx'],
            'is_vla_data': np.array(False, dtype=bool),
        }

        # Add dataset info if requested
        if self.return_dataset_info:
            data['dataset_name'] = sample.get('source', 'unknown')
            data['dataset_local_idx'] = np.array(
                sample.get('episode_index', -1), dtype=np.int32
            )

        return data

    def __iter__(self):
        assert self.preprocessor is not None, "Preprocessor is not set"
        if self.stream is None:
            return

        for sample in self.stream:
            try:
                data = self.sample_to_data(sample)
                torch_data = dict_apply(
                    data, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
                )
                yield torch_data
            except Exception as e:
                warnings.warn(f"Error processing streaming sample: {e}")
                continue
