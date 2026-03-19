from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils


class UnifiedVLACollator:
    def __init__(self, pad_token_id: int = 0, ignore_index: int = -100, padding_side: str = "right"):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.padding_side = padding_side

    def pad_token_sequence(self, values: list[torch.Tensor], padding_value: int) -> torch.Tensor:
        return rnn_utils.pad_sequence(
            values,
            batch_first=True,
            padding_value=padding_value,
            padding_side=self.padding_side,
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {}

        input_ids = [item["input_ids"] for item in features]
        attention_mask = [item["attention_mask"] for item in features]
        labels = [item["labels"] for item in features]
        mm_token_type_ids = [item["mm_token_type_ids"] for item in features]

        batch["input_ids"] = self.pad_token_sequence(input_ids, self.pad_token_id)
        batch["attention_mask"] = self.pad_token_sequence(attention_mask, 0)
        batch["labels"] = self.pad_token_sequence(labels, self.ignore_index)
        batch["mm_token_type_ids"] = self.pad_token_sequence(mm_token_type_ids, 0)

        pixel_values = [item["pixel_values"] for item in features]
        batch["pixel_values"] = torch.cat(pixel_values, dim=0)

        image_grid_thw = [item["image_grid_thw"] for item in features]
        if image_grid_thw[0].ndim == 1:
            batch["image_grid_thw"] = torch.stack(image_grid_thw)
        else:
            batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)

        for key in features[0].keys():
            if key in {
                "input_ids",
                "attention_mask",
                "labels",
                "pixel_values",
                "image_grid_thw",
                "mm_token_type_ids",
            }:
                continue

            values = [item[key] for item in features]
            if isinstance(values[0], str):
                batch[key] = values
            elif isinstance(values[0], (int, float, bool, np.generic)):
                batch[key] = torch.tensor(values)
            else:
                batch[key] = torch.stack(values)

        return batch
