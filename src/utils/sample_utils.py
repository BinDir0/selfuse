"""Centralized sampling primitives for training and inference.

All stochastic sampling operations (flow time, geometric spans, biased
positions, RTC delays) are collected here so that each behaviour has a
single authoritative implementation.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Flow-matching time sampling
# ---------------------------------------------------------------------------

def sample_flow_time(
    batch_size: int,
    num_samples: int = 1,
    *,
    sampling: str = "beta",
    alpha: float = 1.5,
    beta: float = 1.0,
    sig_min: float = 0.001,
) -> torch.FloatTensor:
    """Sample flow-matching time steps.

    Supports two strategies:
      - "uniform": stratified sampling — batch elements evenly spaced across
        [0, 1), each column shifted by a shared random offset.
      - "beta": Beta(alpha, beta) distribution, mapped to t via
        t = (1 - sig_min) * (1 - z).

    Args:
        batch_size: Number of samples (B).
        num_samples: Number of independent time draws per sample (T).
        sampling: "uniform" or "beta".
        alpha, beta: Shape parameters for Beta distribution.
        sig_min: Minimum noise level (only used in "beta" mode).

    Returns:
        [B] when num_samples == 1, else [B, T].
    """
    if sampling == "uniform":
        eps = 1e-5
        ranks = torch.arange(batch_size, dtype=torch.float32) / batch_size
        offsets = torch.rand(num_samples, dtype=torch.float32)
        t = (ranks.unsqueeze(1) + offsets.unsqueeze(0)) % (1 - eps)  # [B, T]
        return t.squeeze(1) if num_samples == 1 else t

    if sampling == "beta":
        beta_dist = torch.distributions.Beta(alpha, beta)
        z = beta_dist.sample((batch_size, num_samples))
        t = (1 - sig_min) * (1 - z)
        return t.squeeze(1) if num_samples == 1 else t

    raise ValueError(f"Unsupported flow time sampling strategy: {sampling}")


# ---------------------------------------------------------------------------
# Geometric distribution (inverse-CDF)
# ---------------------------------------------------------------------------

def sample_geometric(
    shape: tuple[int, ...],
    mean: float,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """Sample from a geometric distribution via inverse CDF.

    P(k) = (1-p)^{k-1} * p  with  p = 1/mean,  so E[k] = mean.
    Implemented as k = ceil(log(U) / log(1-p)) for U ~ Uniform(0, 1).

    Args:
        shape: Output tensor shape.
        mean: Mean of the geometric distribution (must be >= 1).
        device: Target device.

    Returns:
        LongTensor of the given shape, values >= 1.
    """
    assert mean >= 1.0, f"sample_geometric requires mean >= 1.0, got {mean}"
    p = 1.0 / mean
    u = torch.rand(shape, device=device).clamp_(1e-7, 1 - 1e-7)
    # Matches PyTorch Geometric.sample(); log1p(-p) is more stable than log(1-p)
    return torch.ceil(u.log() / torch.tensor(-p, device=device).log1p()).long()


# ---------------------------------------------------------------------------
# Beta-biased discrete position sampling
# ---------------------------------------------------------------------------

def sample_beta_positions(
    batch_size: int,
    num_samples: int,
    seq_len: int,
    alpha: float,
    prefer_early: bool = True,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """Sample discrete positions from a Beta-biased distribution.

    - prefer_early=True  → Beta(1, alpha), density concentrated near 0.
    - prefer_early=False → Beta(alpha, 1), density concentrated near seq_len-1.

    The continuous Beta sample is scaled to [0, seq_len) then truncated to long.

    Args:
        batch_size: B.
        num_samples: Number of positions per sample.
        seq_len: Length of the discrete sequence to index into.
        alpha: Shape parameter controlling bias strength.
        prefer_early: Direction of the bias.
        device: Target device.

    Returns:
        [B, num_samples] LongTensor in [0, seq_len).
    """
    if prefer_early:
        beta_dist = torch.distributions.Beta(1.0, alpha)
    else:
        beta_dist = torch.distributions.Beta(alpha, 1.0)
    positions = (beta_dist.sample((batch_size, num_samples)).to(device) * seq_len).long()
    positions.clamp_(max=seq_len - 1)
    return positions


# ---------------------------------------------------------------------------
# RTC prefix delay sampling
# ---------------------------------------------------------------------------

def sample_rtc_delay(
    valid_action_len: torch.LongTensor,
    strategy: str = "uniform",
    max_delay: int | None = None,
    forced_delay: torch.LongTensor | None = None,
) -> torch.LongTensor:
    """Sample one RTC prefix delay per batch element.

    The sampled delay determines how many leading action tokens are treated as
    known action-prefix conditions during RTC flow training.

    Args:
        valid_action_len:
            [B] Number of valid action steps for each sample. Values are expected
            to be in the range [0, horizon_steps].
        strategy:
            Delay sampling strategy. ``uniform`` samples all valid delays with
            equal probability. ``exp`` biases toward smaller delays via
            ``exp(arange(upper)[::-1])``.
        max_delay:
            Optional upper bound for the sampled delay. When provided, the actual
            sampling upper bound becomes min(valid_action_len, max_delay) for each
            sample.
        forced_delay:
            Optional [B] tensor used to bypass random sampling. This is intended
            for deterministic tests and debugging.

    Returns:
        torch.LongTensor:
            [B] Sampled prefix delays. Each entry is in the range
            [0, min(valid_action_len_i, max_delay)) when the upper bound is
            positive, or 0 when the sample has no valid action tokens.
    """
    if forced_delay is not None:
        return forced_delay.to(device=valid_action_len.device, dtype=torch.long)

    delay_upper = valid_action_len.clamp(min=0)
    if max_delay is not None:
        delay_upper = torch.minimum(
            delay_upper,
            torch.full_like(delay_upper, max_delay),
        )
    delay_upper = delay_upper.clamp(min=0)
    strategy = str(strategy).lower()

    if strategy == "uniform":
        random_delay = torch.rand(delay_upper.shape, device=valid_action_len.device, dtype=torch.float32)
        return torch.floor(random_delay * delay_upper.to(torch.float32)).to(dtype=torch.long)

    if strategy == "exp":
        max_upper = delay_upper.max().item()
        if max_upper <= 0:
            return torch.zeros_like(delay_upper, device=valid_action_len.device, dtype=torch.long)
        w = torch.exp(torch.arange(max_upper - 1, -1, -1, device=valid_action_len.device, dtype=torch.float32))
        w = w / w.sum()
        sampled = torch.multinomial(w.unsqueeze(0).expand(delay_upper.shape[0], -1), num_samples=1).squeeze(1)
        sampled = sampled % delay_upper.clamp(min=1)
        return sampled.to(dtype=torch.long)

    raise ValueError(f"Unsupported RTC delay strategy: {strategy}")

