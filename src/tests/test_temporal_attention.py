"""Numerical tests for TemporalCausalAttentionPass.

Compares the vectorized (flash_attn_varlen_func) implementation against a
naive eager for-loop reference to verify correctness of gather, transpose,
cu_seqlens construction, and scatter operations.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from flash_attn import flash_attn_func


# ---------------------------------------------------------------------------
# Minimal mock that exposes the same interface as Qwen3VLVisionBlock
# ---------------------------------------------------------------------------

class MockSpatialBlock(nn.Module):
    """Minimal spatial block that provides norm1, attn.qkv, attn.proj, attn.num_heads."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.attn.proj = nn.Linear(hidden_size, hidden_size)
        self.attn.num_heads = num_heads


# ---------------------------------------------------------------------------
# Eager reference: for-loop per entry, flash_attn_func per entry
# ---------------------------------------------------------------------------

def eager_temporal_attention(
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    norm: nn.LayerNorm,
    qkv_proj: nn.Linear,
    out_proj: nn.Linear,
    num_heads: int,
    temporal_pe: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation with explicit per-entry for loop."""
    entry_lengths = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    entry_offsets = torch.zeros(
        len(entry_lengths) + 1, dtype=torch.long, device=hidden_states.device,
    )
    entry_offsets[1:] = entry_lengths.cumsum(0)

    residual = torch.zeros_like(hidden_states)

    for i in range(len(grid_thw)):
        T = grid_thw[i, 0].item()
        H = grid_thw[i, 1].item()
        W = grid_thw[i, 2].item()
        N = H * W
        start = entry_offsets[i].item()
        end = entry_offsets[i + 1].item()

        if T <= 1:
            continue

        entry_hidden = hidden_states[start:end]

        # (T*N, D) -> (T, N, D) -> (N, T, D)
        x = entry_hidden.reshape(T, N, -1).permute(1, 0, 2).contiguous()
        x = norm(x)
        x = x + temporal_pe[:T].to(dtype=x.dtype, device=x.device)

        # QKV projection
        head_dim = qkv_proj.out_features // (3 * num_heads)
        qkv = qkv_proj(x)  # (N, T, 3*D)
        q, k, v = (
            qkv.reshape(N, T, 3, num_heads, head_dim)
            .permute(2, 0, 1, 3, 4)
            .unbind(0)
        )  # each: (N, T, heads, head_dim)

        # flash_attn_func: (batch, seqlen, nheads, headdim)
        attn_out = flash_attn_func(q, k, v, causal=True)
        attn_out = attn_out.reshape(N, T, -1)
        attn_out = out_proj(attn_out)  # (N, T, D)

        # (N, T, D) -> (T, N, D) -> (T*N, D)
        residual[start:end] = attn_out.permute(1, 0, 2).reshape(T * N, -1)

    return residual


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for flash attention")
    return torch.device("cuda")


HIDDEN = 64
NUM_HEADS = 4
MAX_TEMPORAL_LEN = 32


def make_module(device):
    """Create spatial block + TemporalCausalAttentionPass on device."""
    from src.model.vision.temporal_attention import TemporalCausalAttentionPass

    spatial_block = MockSpatialBlock(HIDDEN, NUM_HEADS).to(device)
    ta = TemporalCausalAttentionPass(
        spatial_block=spatial_block,
        hidden_size=HIDDEN,
        max_temporal_len=MAX_TEMPORAL_LEN,
    ).to(device)
    return spatial_block, ta


def test_vectorized_matches_eager_single_entry(device):
    """One multi-frame entry: vectorized == eager."""
    spatial_block, ta = make_module(device)
    T, H, W = 4, 3, 3
    N = H * W
    grid_thw = torch.tensor([[T, H, W]], dtype=torch.long, device=device)
    hidden = torch.randn(T * N, HIDDEN, device=device, dtype=torch.float16)

    ta.current_grid_thw = grid_thw
    with torch.no_grad():
        vectorized = ta(hidden)
        eager = eager_temporal_attention(
            hidden, grid_thw,
            norm=ta.norm, qkv_proj=ta.qkv_proj, out_proj=ta.out_proj,
            num_heads=ta.num_heads, temporal_pe=ta.temporal_pe,
        )

    torch.testing.assert_close(vectorized, eager, atol=1e-3, rtol=1e-3)


def test_vectorized_matches_eager_multiple_entries(device):
    """Multiple multi-frame entries with same T, N."""
    spatial_block, ta = make_module(device)
    T, H, W = 6, 2, 2
    M = 3
    N = H * W
    grid_thw = torch.tensor([[T, H, W]] * M, dtype=torch.long, device=device)
    hidden = torch.randn(M * T * N, HIDDEN, device=device, dtype=torch.float16)

    ta.current_grid_thw = grid_thw
    with torch.no_grad():
        vectorized = ta(hidden)
        eager = eager_temporal_attention(
            hidden, grid_thw,
            norm=ta.norm, qkv_proj=ta.qkv_proj, out_proj=ta.out_proj,
            num_heads=ta.num_heads, temporal_pe=ta.temporal_pe,
        )

    torch.testing.assert_close(vectorized, eager, atol=1e-3, rtol=1e-3)


def test_vectorized_matches_eager_mixed_with_single_frame(device):
    """Mix of T=1 (image) and T>1 (video) entries in one batch."""
    spatial_block, ta = make_module(device)
    T_video, H, W = 4, 2, 3
    N = H * W
    # Entry layout: [video, image, video, image]
    grid_thw = torch.tensor([
        [T_video, H, W],
        [1, H, W],
        [T_video, H, W],
        [1, H, W],
    ], dtype=torch.long, device=device)

    total_tokens = 2 * T_video * N + 2 * 1 * N
    hidden = torch.randn(total_tokens, HIDDEN, device=device, dtype=torch.float16)

    ta.current_grid_thw = grid_thw
    with torch.no_grad():
        vectorized = ta(hidden)
        eager = eager_temporal_attention(
            hidden, grid_thw,
            norm=ta.norm, qkv_proj=ta.qkv_proj, out_proj=ta.out_proj,
            num_heads=ta.num_heads, temporal_pe=ta.temporal_pe,
        )

    torch.testing.assert_close(vectorized, eager, atol=1e-3, rtol=1e-3)

    # T=1 entries should have zero residual
    entry_lengths = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
    offsets = [0]
    for l in entry_lengths:
        offsets.append(offsets[-1] + l)
    # Entry 1 (image)
    assert (vectorized[offsets[1]:offsets[2]] == 0).all()
    # Entry 3 (image)
    assert (vectorized[offsets[3]:offsets[4]] == 0).all()


def test_all_single_frame_returns_zero(device):
    """All T=1 entries should produce zero residual."""
    spatial_block, ta = make_module(device)
    H, W = 3, 3
    N = H * W
    grid_thw = torch.tensor([[1, H, W], [1, H, W]], dtype=torch.long, device=device)
    hidden = torch.randn(2 * N, HIDDEN, device=device, dtype=torch.float16)

    ta.current_grid_thw = grid_thw
    with torch.no_grad():
        result = ta(hidden)

    assert (result == 0).all()


def test_none_grid_returns_zero(device):
    """grid_thw=None should produce zero residual."""
    spatial_block, ta = make_module(device)
    hidden = torch.randn(20, HIDDEN, device=device, dtype=torch.float16)

    ta.current_grid_thw = None
    with torch.no_grad():
        result = ta(hidden)

    assert (result == 0).all()


def test_gradient_flows_through_temporal_attention(device):
    """Verify gradients propagate through the vectorized path."""
    spatial_block, ta = make_module(device)
    T, H, W = 4, 2, 2
    N = H * W
    grid_thw = torch.tensor([[T, H, W]], dtype=torch.long, device=device)
    hidden = torch.randn(
        T * N, HIDDEN, device=device, dtype=torch.float16, requires_grad=True,
    )

    ta.current_grid_thw = grid_thw
    # float32 for stable gradient computation
    ta = ta.float()
    hidden_f32 = hidden.float().detach().requires_grad_(True)
    ta.current_grid_thw = grid_thw
    residual = ta(hidden_f32)
    loss = residual.sum()
    loss.backward()

    assert hidden_f32.grad is not None
    assert hidden_f32.grad.shape == hidden_f32.shape
    # Multi-frame tokens should have non-zero gradients
    assert (hidden_f32.grad != 0).any()
