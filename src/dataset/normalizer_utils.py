'''
Normalizer utility functions for LegendVLA datasets.
'''

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.model.common.normalizer import LinearNormalizer


def get_normalizer(dataloader_cfg, normalizer_dataset):
    assert normalizer_dataset is not None, "normalizer_dataset must be provided"
    # Merge all data
    dataloader = DataLoader(normalizer_dataset, collate_fn=normalizer_dataset.get_collator(), **dataloader_cfg)
    assert len(dataloader) > 0, "No data to calculate normalizer"
    normalizer = LinearNormalizer()
    normalizer_keys = next(iter(dataloader)).keys()
    normalizer.start_streaming_fit(keys=normalizer_keys)
    for batch in tqdm(dataloader, desc="Calculating normalizer"):
        input_data = {
            k: v.reshape(-1, v.shape[-1]) for k, v in batch.items() \
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray)
        }
        normalizer.update_streaming_fit(input_data)
    normalizer.finish_streaming_fit()
    # ignore the wrist rotation
    for key in normalizer_keys:
        if key in ['states', 'actions', 'motions']:
            normalizer.ignore_dim(key=key, dim=slice(6, 18))
        else:
            raise ValueError(f"Unsupported key: {key}")

    def print_dict(d):
        for k, v in d.items():
            print(f"{k}: {v}")

    for key in normalizer.params_dict.keys():
        print(f"{key}: ")
        print_dict(normalizer.params_dict[key]['input_stats'])
        print(f"scale: {normalizer.params_dict[key]['scale']}")
        print(f"offset: {normalizer.params_dict[key]['offset']}")

    return normalizer
