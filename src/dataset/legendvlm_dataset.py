'''
LegendVLM Dataset for VLM-only data stored in HuggingFace datasets format.

Extracted from legendvla_dataset.py for modularity.
'''

from typing import Dict
import pathlib
import torch
import numpy as np
import warnings
from torchvision import transforms
from datasets import concatenate_datasets, load_from_disk, DatasetDict
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
