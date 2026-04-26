'''
WebDataset-based VLM dataset for LegendVLA training.
'''

import time
import warnings
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from src.utils.pytorch_util import dict_apply
from src.dataset.data_transforms import process_image
from src.dataset.sanity_checks import DataChecker, MissingOrInvalidFilesError
from src.dataset.wds_dataset import build_blended_dataset


class VLMWdsDataset(torch.utils.data.IterableDataset):
    """
    WebDataset VLM dataset.

    When `mem_enabled` is True, VLM samples are resized to `target_image_size`
    and padded to `n_obs_image_steps` repeated frames so they share the same
    (T, N) shape as VLA video samples. This lets MEM temporal attention run
    pure view + SDPA without per-sample mask gymnastics. The padding is
    semantically a no-op (all T frames are identical copies of the original
    image), so the LM still receives the same single-image content after the
    backbone slices to the last frame.

    When `mem_enabled` is False, VLM samples keep the original image path
    (vision_type='image', no resize, no padding) and the collator/backbone
    treat them as standard single-image inputs.
    """
    def __init__(
        self,
        wds_datasets: List[Dict],
        weights: List[float] = [0.5, 0.5, 0.5],
        seed: int = 42,
        mode: str = 'train',
        shuffle_buffer: int = 16384,
        shuffle_initial: Optional[int] = None,
        return_dataset_info: bool = False,
        val_wds_datasets: Optional[List[Dict]] = None,
        mem_enabled: bool = False,
        n_obs_image_steps: int = 1,
        target_image_size: Optional[Tuple[int, int]] = None,
        keep_ratio: float = 1.0,
        sanity_checks: Optional[Dict] = None,
    ):
        super().__init__()
        self.wds_datasets = wds_datasets
        self.weights = weights
        self.seed = seed
        self.mode = mode
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_initial = shuffle_initial
        self.return_dataset_info = return_dataset_info
        self.val_wds_datasets = val_wds_datasets
        self.collator = None
        self.sanity_checks = dict(sanity_checks or {})
        self.checker = DataChecker(sanity_cfg=self.sanity_checks)
        self.mem_enabled = mem_enabled
        self.n_obs_image_steps = n_obs_image_steps
        self.target_image_size = (
            tuple(target_image_size) if target_image_size is not None else None
        )
        assert 0.0 < keep_ratio <= 1.0, f"keep_ratio must be in (0, 1], got {keep_ratio}"
        self.keep_ratio = float(keep_ratio)
        if self.mem_enabled:
            assert self.n_obs_image_steps >= 1, "n_obs_image_steps must be >= 1 when mem_enabled"
            assert self.target_image_size is not None, "target_image_size required when mem_enabled"

        # Image augmentation is handled by data_transforms.process_image
        # (shared with VLA). No separate transform object is needed here.

    def distribute(self, rank: int, world_size: int):
        """WebDataset splitting is handled in build_wds_pipeline."""
        return

    def set_collator(self, collator):
        """Set the batch collator used to build model inputs."""
        self.collator = collator

    def get_collator(self):
        """Build a collator copy configured for this dataset's mode."""
        assert self.collator is not None, "Collator is not set"
        from copy import deepcopy
        from src.dataset.unified_vla_collator import UnifiedVLACollator
        return UnifiedVLACollator(
            formatter=self.collator.formatter,
            batch_processor=deepcopy(self.collator.batch_processor),
            mode=self.mode,
            debug_capture_texts=self.collator.debug_capture_texts,
            debug_profile_timing=self.collator.debug_profile_timing,
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
            mem_enabled=self.mem_enabled,
            n_obs_image_steps=self.n_obs_image_steps,
            target_image_size=self.target_image_size,
            keep_ratio=1.0,
            sanity_checks=self.sanity_checks,
        )
        if self.collator is not None:
            val_dataset.set_collator(self.collator)
        return val_dataset

    def sample_to_data(self, sample):
        """Convert one WDS sample to model-ready fields."""
        self.checker.check(sample_schema=(sample, {
            "required_keys": ("meta.json",),
            "required_meta_keys": (
                "texts",
                "formatting_ratings",
                "visual_dependency_ratings",
                "relevance_ratings",
            ),
        }))

        meta = sample['meta.json']

        image_keys = sorted([
            k for k in sample.keys()
            if k.startswith("image_") and k.endswith(".jpg")
        ])
        if not image_keys:
            raise MissingOrInvalidFilesError("missing required image_*.jpg fields")
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
        self.checker.check(instruction=(question, 1))

        raw_images = []
        for img_pil in images:
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            raw_images.append(np.array(img_pil, dtype=np.uint8))
        images_arr = np.stack(raw_images, dtype=np.uint8)

        # Always resize: dynamic aspect ratios cause vision-tower recompiles.
        images_processed, _, _ = process_image(
            images_arr,
            aug_transform=(self.mode == 'train'),
            target_size=self.target_image_size,
        )
        self.checker.check(
            image=images_processed,
            finite={'images_processed': images_processed},
        )

        if self.mem_enabled:
            # Repeat along the temporal axis to match VLA T frames.
            # Repetition is semantically a no-op: backbone slices to the last
            # frame, which is identical to the original single image.
            T = self.n_obs_image_steps
            K = images_processed.shape[0]
            if K < T:
                pad = np.repeat(images_processed[-1:], T - K, axis=0)
                images_padded = np.concatenate([images_processed, pad], axis=0)
            elif K > T:
                images_padded = images_processed[-T:]
            else:
                images_padded = images_processed
            data = {
                'images': images_padded,                    # (T, tH, tW, C)
                'question': question,
                'answer': answer,
                'vision_type': 'video',
                'video_fps': np.float32(1.0),
                'is_vla_data': np.array(False, dtype=bool),
            }
        else:
            data = {
                'images': images_processed,
                'question': question,
                'answer': answer,
                'vision_type': 'image',
                'is_vla_data': np.array(False, dtype=bool),
            }

        if self.return_dataset_info:
            data['dataset_name'] = meta.get('source', meta.get('dataset_name', 'unknown'))
            data['episode_index'] = np.array(
                meta.get('sample_idx', -1), dtype=np.int32
            )
        self.checker.check(finite=data)
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
            # DataSkipError -> attach_sample_ctx logs and drops the sample;
            # any other failure -> attach_sample_ctx re-raises with locator.
            self.checker.note_sample_seen()
            data = self.sample_to_data(sample)
            return dict_apply(
                data, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
            )

        def strip_key(src):
            for sample in src:
                sample.pop("__key__", None)
                yield sample

        pipeline = build_blended_dataset(
            datasets_config=datasets_config,
            preprocess_fn=preprocess_fn,
            shuffle_buffer=self.shuffle_buffer,
            shuffle_initial=self.shuffle_initial,
            mode=self.mode,
            use_sliding_window=False,
            keep_ratio=self.keep_ratio,
            checker=self.checker,
        )
        return strip_key(pipeline)

    def __iter__(self):
        pipeline = self.build_pipeline()
        return iter(pipeline)
