'''
WebDataset-based VLM dataset for LegendVLA training.
'''

from typing import Dict, List, Optional
import torch
import numpy as np
import warnings
from torchvision import transforms
from src.utils.pytorch_util import dict_apply
from src.dataset.collator import LegendVLDataCollator
from src.dataset.wds_dataset import build_blended_dataset


class VLMWdsDataset(torch.utils.data.IterableDataset):
    """
    WebDataset VLM dataset.
    """
    def __init__(
        self,
        wds_datasets: List[Dict],
        weights: List[float] = [0.5, 0.5, 0.5],
        seed: int = 42,
        mode: str = 'train',
        shuffle_buffer: int = 4096,
        return_dataset_info: bool = False,
        val_wds_datasets: Optional[List[Dict]] = None,
    ):
        super().__init__()
        self.wds_datasets = wds_datasets
        self.weights = weights
        self.seed = seed
        self.mode = mode
        self.shuffle_buffer = shuffle_buffer
        self.return_dataset_info = return_dataset_info
        self.val_wds_datasets = val_wds_datasets
        self.preprocessor = None

        if self.mode == 'train':
            self.aug_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
            ])
        else:
            self.aug_transform = None

    def distribute(self, rank: int, world_size: int):
        """WebDataset splitting is handled in build_wds_pipeline."""
        return

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

    def get_validation_dataset(self):
        """Create a new WebDataset instance for validation."""
        assert self.val_wds_datasets is not None, "val_wds_datasets is not set"
        val_dataset = VLMWdsDataset(
            wds_datasets=self.val_wds_datasets,
            weights=self.weights,
            seed=self.seed,
            mode='val' if self.mode == 'train' else self.mode,
            shuffle_buffer=0,
            return_dataset_info=self.return_dataset_info,
            val_wds_datasets=self.val_wds_datasets,
        )
        if self.preprocessor is not None:
            val_dataset.set_preprocessor(self.preprocessor)
        return val_dataset

    def sample_to_data(self, sample):
        """Convert one WDS sample to model-ready fields."""
        meta = sample['meta.json']

        image_keys = sorted([
            k for k in sample.keys()
            if k.startswith("image_") and k.endswith(".jpg")
        ])
        images = [sample[k] for k in image_keys]
        text = meta['texts']
        weights = self.weights

        formatting_ratings = np.array(
            [r if r is not None else 0 for r in meta['formatting_ratings']]
        )
        visual_dependency_ratings = np.array(
            [r if r is not None else 0 for r in meta['visual_dependency_ratings']]
        )
        relevance_ratings = np.array(
            [r if r is not None else 0 for r in meta['relevance_ratings']]
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

        if self.return_dataset_info:
            data['dataset_name'] = meta.get('source', meta.get('dataset_name', 'unknown'))
            data['dataset_local_idx'] = np.array(
                meta.get('sample_idx', -1), dtype=np.int32
            )
        return data

    def build_pipeline(self):
        datasets_config = []
        for ds in self.wds_datasets:
            datasets_config.append({
                "shard_urls": ds["shard_urls"],
                "weight": ds.get("weight", 1.0),
                "name": ds.get("name", "unknown"),
            })

        def preprocess_fn(sample):
            data = self.sample_to_data(sample)
            torch_data = dict_apply(
                data, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
            )
            return torch_data

        def strip_key(src):
            for sample in src:
                sample.pop("__key__", None)
                yield sample

        pipeline = build_blended_dataset(
            datasets_config=datasets_config,
            preprocess_fn=preprocess_fn,
            shuffle_buffer=self.shuffle_buffer,
            mode=self.mode,
            use_sliding_window=False,
        )
        return strip_key(pipeline)

    def __iter__(self):
        assert self.preprocessor is not None, "Preprocessor is not set"
        pipeline = self.build_pipeline()
        return iter(pipeline)
