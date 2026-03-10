"""Unit tests for streaming VLA+VLM unified dataset iteration logic.

Tests cover:
- UnifiedWdsDataset: train interleaving ratio, VLM auto-restart, val sequential
- UnifiedWdsDataset.pad_vlm_sample: zero-padding of missing VLA fields
"""

import sys
import traceback
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.dataset.vla_dataset import UnifiedWdsDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vla_sample(idx):
    """Create a mock VLA sample with typical fields."""
    return {
        "input_ids": torch.tensor([idx, idx + 1, idx + 2]),
        "labels": torch.tensor([idx, idx + 1, idx + 2]),
        "attention_mask": torch.ones(3, dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 4, 4),
        "states": torch.full((16, 48), float(idx)),
        "actions": torch.full((32, 48), float(idx)),
        "actions_valid_mask": torch.ones(32, 48, dtype=torch.bool),
        "n_states": torch.tensor(16, dtype=torch.int32),
        "n_actions": torch.tensor(32, dtype=torch.int32),
        "answer_start_idx": torch.tensor(1),
        "is_vla_data": torch.tensor(True),
        "has_depth_values": torch.tensor(False),
    }


def make_vlm_sample(idx):
    """Create a mock VLM sample (no states/actions)."""
    return {
        "input_ids": torch.tensor([idx + 100]),
        "labels": torch.tensor([idx + 100]),
        "attention_mask": torch.ones(1, dtype=torch.long),
        "pixel_values": torch.zeros(2, 3, 4, 4),
        "answer_start_idx": torch.tensor(0),
        "is_vla_data": torch.tensor(False),
    }
