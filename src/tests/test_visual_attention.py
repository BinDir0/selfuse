"""Unit tests for src.utils.visual_attention.

Pure-numpy/torch synthetic data — no model, no checkpoint, no data shards.
Each test runs in well under a second.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.utils.visual_attention import (
    DEFAULT_MIDDLE_LAYER_RANGE_14,
    aggregate_visual_attention,
    compute_middle_layer_range,
    overlay_attention,
    renormalize_visual_subset,
    reshape_visual_to_grid,
    spatial_entropy,
    stack_visual_attention,
)


# ─────────────────────────────────────────────────────────────────────────────
# compute_middle_layer_range
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_middle_layer_range_scaling():
    # 14-layer default case.
    assert compute_middle_layer_range(14) == DEFAULT_MIDDLE_LAYER_RANGE_14
    assert compute_middle_layer_range(14) == (3, 11)
    # Double the depth -> range scales linearly.
    assert compute_middle_layer_range(28) == (6, 22)
    # Quarter depth -> range shrinks.
    assert compute_middle_layer_range(7) == (2, 6)  # round(7*3/14)=2, round(7*11/14)=6


# ─────────────────────────────────────────────────────────────────────────────
# stack_visual_attention
# ─────────────────────────────────────────────────────────────────────────────

def test_stack_visual_attention_nan_fill():
    torch.manual_seed(0)
    # [n_steps=2, n_layers=3] with the middle layer missing.
    n_heads, action_len, kv_len = 4, 8, 16
    visual_indices = np.array([2, 5, 7, 11], dtype=np.int64)  # 4 visual positions

    def make_layer():
        return torch.randn(1, n_heads, action_len, kv_len)  # B=1

    expert_attn = [
        [make_layer(), None, make_layer()],
        [make_layer(), None, make_layer()],
    ]

    out = stack_visual_attention(
        expert_attn, prefix_len=kv_len - action_len, visual_indices=visual_indices,
    )
    # Shape: [S=2, L=3, H=4, A=8, n_visual=4]
    assert out.shape == (2, 3, n_heads, action_len, len(visual_indices))
    assert out.dtype == torch.float32

    # Middle layer should be all NaN.
    assert torch.isnan(out[:, 1]).all()
    # Other layers should be finite and match input slicing.
    for s in range(2):
        for l in (0, 2):
            expected = expert_attn[s][l][0, :, :, visual_indices].float()
            assert torch.allclose(out[s, l], expected)


def test_stack_visual_attention_all_none_raises():
    # Every layer is None -> no shape to infer.
    expert_attn = [[None, None]]
    with pytest.raises(RuntimeError, match="No expert attention"):
        stack_visual_attention(
            expert_attn, prefix_len=10, visual_indices=np.array([0, 1]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# reshape_visual_to_grid
# ─────────────────────────────────────────────────────────────────────────────

def test_reshape_visual_to_grid_shape_and_order():
    # Qwen3-VL default: 384x384 with patch_size=16, merge_size=2 -> 12x12 token grid.
    T_g, H_g, W_g, m = 3, 24, 24, 2
    n_visual = T_g * (H_g // m) * (W_g // m)  # 432
    assert n_visual == 432

    flat = torch.arange(n_visual, dtype=torch.float32)
    grid, token_H, token_W = reshape_visual_to_grid(flat, T_g, H_g, W_g, merge_size=m)

    assert grid.shape == (T_g, 12, 12)
    assert (token_H, token_W) == (12, 12)
    # Flat index layout: t * 144 + row * 12 + col
    for t in range(T_g):
        for row in range(12):
            for col in range(12):
                expected = t * 144 + row * 12 + col
                assert grid[t, row, col].item() == expected


def test_reshape_visual_to_grid_accepts_numpy():
    flat_np = np.arange(3 * 12 * 12, dtype=np.float32)
    grid, token_H, token_W = reshape_visual_to_grid(flat_np, 3, 24, 24)
    assert isinstance(grid, torch.Tensor)  # torch-native output
    assert grid.shape == (3, 12, 12)


def test_reshape_visual_to_grid_with_leading_dims():
    # Batch dims should be preserved.
    leading_shape = (5, 2)
    flat = torch.zeros(*leading_shape, 3 * 12 * 12)
    grid, _, _ = reshape_visual_to_grid(flat, 3, 24, 24)
    assert grid.shape == (*leading_shape, 3, 12, 12)


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_visual_attention
# ─────────────────────────────────────────────────────────────────────────────

def _make_grid(S=2, L=4, H=3, A=2, T_g=1, tH=4, tW=4, seed=0):
    torch.manual_seed(seed)
    return torch.rand(S, L, H, A, T_g, tH, tW)


def test_aggregate_middle_requires_range():
    grid = _make_grid()
    # No layer_range -> ValueError
    with pytest.raises(ValueError, match="requires an explicit layer_range"):
        aggregate_visual_attention(
            grid, strategy="middle_layers_mean_heads", ode_step=0, layer_range=None,
        )
    # Empty layer_range -> ValueError
    with pytest.raises(ValueError, match="empty layer_range"):
        aggregate_visual_attention(
            grid, strategy="middle_layers_mean_heads", ode_step=0, layer_range=(2, 2),
        )


def test_aggregate_nan_layers_skipped():
    grid = _make_grid(L=5)
    # Mark layer 2 entirely NaN.
    grid[:, 2] = float("nan")
    # mean_layers_heads should reduce to plain mean over layers [0, 1, 3, 4].
    result = aggregate_visual_attention(
        grid, strategy="mean_layers_heads", ode_step=0,
    )
    # Hand-compute: mean over A, then mean over H, then mean over valid layers
    # at ODE step 0.
    step0 = grid[0]  # [L, H, A, T_g, tH, tW]
    valid = torch.stack([step0[0], step0[1], step0[3], step0[4]])  # [4, H, A, ...]
    expected = valid.mean(dim=2).mean(dim=1).mean(dim=0)
    assert torch.allclose(result, expected, atol=1e-6)
    # No NaNs propagated.
    assert not torch.isnan(result).any()


def test_aggregate_max_heads_strategy():
    grid = _make_grid(L=3, H=4)
    result = aggregate_visual_attention(
        grid, strategy="max_heads", ode_step=1,
    )
    # Hand-compute: mean A, max H, mean L at step 1.
    step1 = grid[1]
    expected = step1.mean(dim=2).amax(dim=1).mean(dim=0)
    assert torch.allclose(result, expected, atol=1e-6)


def test_aggregate_last_layer_strategy():
    grid = _make_grid(L=4)
    # Mark layer 3 as NaN so "last valid" is layer 2.
    grid[:, 3] = float("nan")
    result = aggregate_visual_attention(
        grid, strategy="last_layer_mean_heads", ode_step=0,
    )
    last_valid = grid[0, 2]  # [H, A, T_g, tH, tW]
    expected = last_valid.mean(dim=1).mean(dim=0)
    assert torch.allclose(result, expected, atol=1e-6)


def test_aggregate_middle_layers_computes_mean():
    grid = _make_grid(L=6)
    result = aggregate_visual_attention(
        grid, strategy="middle_layers_mean_heads", ode_step=0, layer_range=(2, 5),
    )
    # Subset is layers [2, 3, 4].
    subset = grid[0, 2:5]  # [3, H, A, T_g, tH, tW]
    expected = subset.mean(dim=2).mean(dim=1).mean(dim=0)
    assert torch.allclose(result, expected, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# renormalize_visual_subset
# ─────────────────────────────────────────────────────────────────────────────

def test_renormalize_subset_sums_to_one():
    torch.manual_seed(0)
    grid = torch.rand(3, 12, 12)
    out = renormalize_visual_subset(grid)
    assert torch.allclose(out.sum(), torch.tensor(1.0), atol=1e-6)


def test_renormalize_zero_grid_unchanged():
    grid = torch.zeros(3, 12, 12)
    out = renormalize_visual_subset(grid)
    assert torch.equal(out, grid)


# ─────────────────────────────────────────────────────────────────────────────
# spatial_entropy
# ─────────────────────────────────────────────────────────────────────────────

def test_spatial_entropy_uniform_is_one():
    grid = torch.ones(12, 12)
    assert abs(spatial_entropy(grid) - 1.0) < 1e-6


def test_spatial_entropy_onehot_is_zero():
    grid = torch.zeros(12, 12)
    grid[5, 7] = 1.0
    assert abs(spatial_entropy(grid) - 0.0) < 1e-6


def test_spatial_entropy_accepts_numpy():
    grid = np.ones((8, 8), dtype=np.float32)
    assert abs(spatial_entropy(grid) - 1.0) < 1e-6


def test_spatial_entropy_intermediate_value():
    # Two equal-mass hotspots out of 16 cells -> entropy = log(2) / log(16) = 0.25.
    grid = torch.zeros(4, 4)
    grid[0, 0] = 0.5
    grid[3, 3] = 0.5
    expected = math.log(2) / math.log(16)
    assert abs(spatial_entropy(grid) - expected) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# overlay_attention
# ─────────────────────────────────────────────────────────────────────────────

def test_overlay_attention_shape_dtype():
    pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    frame = np.full((384, 384, 3), 128, dtype=np.uint8)
    attn = torch.rand(12, 12)
    out = overlay_attention(frame, attn)
    assert out.shape == (384, 384, 3)
    assert out.dtype == np.uint8


def test_overlay_attention_accepts_numpy():
    pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    attn = np.random.RandomState(0).rand(8, 8).astype(np.float32)
    out = overlay_attention(frame, attn, upsample="nearest")
    assert out.shape == (256, 256, 3)
    assert out.dtype == np.uint8
