from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


class NonFiniteDataError(ValueError):
    """Raised when dataset preprocessing produces a non-finite numeric value."""


def build_sample_context(sample: Mapping[str, Any]) -> str:
    """Build a compact, human-readable sample identifier string."""
    fields: list[str] = []

    dataset_name = sample.get("dataset_name", sample.get("source", "unknown"))
    dataset_name = _to_python_scalar(dataset_name)
    if dataset_name is not None:
        fields.append(f"dataset={dataset_name}")

    episode_index = sample.get("episode_index", sample.get("sample_idx"))
    episode_index = _to_python_scalar(episode_index)
    if episode_index is not None:
        fields.append(f"episode={episode_index}")

    sample_key = sample.get("__key__")
    sample_key = _to_python_scalar(sample_key)
    if sample_key is not None:
        fields.append(f"key={sample_key}")

    return ", ".join(fields) if fields else "dataset=unknown"


def ensure_mapping_finite(
    values: Mapping[str, Any],
    stage: str,
    context: str,
) -> None:
    """Validate that all numeric arrays or tensors in a mapping are finite."""
    for name, value in values.items():
        ensure_value_finite(name=name, value=value, stage=stage, context=context)


def ensure_value_finite(name: str, value: Any, stage: str, context: str) -> None:
    """Validate that one numeric array or tensor is finite."""
    if value is None:
        return

    summary = summarize_non_finite(name=name, value=value)
    if summary is None:
        return

    raise NonFiniteDataError(
        f"Non-finite value detected at stage={stage}, {context}, {summary}"
    )


def summarize_non_finite(name: str, value: Any) -> str | None:
    """Return a short diagnostic string when a numeric value is non-finite."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0 or value.dtype == torch.bool:
            return None
        if not torch.is_floating_point(value) and not torch.is_complex(value):
            return None

        finite_mask = torch.isfinite(value)
        if bool(finite_mask.all()):
            return None

        nan_count = int(torch.isnan(value).sum().item())
        inf_count = int(torch.isinf(value).sum().item())
        finite_values = value[finite_mask]
        return _format_summary(
            name=name,
            shape=tuple(value.shape),
            dtype=str(value.dtype),
            nan_count=nan_count,
            inf_count=inf_count,
            finite_values=finite_values,
        )

    if isinstance(value, np.ndarray):
        if value.size == 0 or value.dtype == np.bool_:
            return None
        if not np.issubdtype(value.dtype, np.floating) and not np.issubdtype(value.dtype, np.complexfloating):
            return None

        finite_mask = np.isfinite(value)
        if bool(finite_mask.all()):
            return None

        nan_count = int(np.isnan(value).sum())
        inf_count = int(np.isinf(value).sum())
        finite_values = value[finite_mask]
        return _format_summary(
            name=name,
            shape=value.shape,
            dtype=str(value.dtype),
            nan_count=nan_count,
            inf_count=inf_count,
            finite_values=finite_values,
        )

    return None


def _format_summary(
    name: str,
    shape: tuple[int, ...],
    dtype: str,
    nan_count: int,
    inf_count: int,
    finite_values: Any,
) -> str:
    finite_count = _count_values(finite_values)
    if finite_count == 0:
        return (
            f"field={name}, shape={shape}, dtype={dtype}, "
            f"nan_count={nan_count}, inf_count={inf_count}, finite_stats=none"
        )

    if isinstance(finite_values, torch.Tensor):
        finite_values = finite_values.detach().float()
        finite_min = float(finite_values.min().item())
        finite_max = float(finite_values.max().item())
        finite_mean = float(finite_values.mean().item())
    else:
        finite_values = finite_values.astype(np.float32, copy=False)
        finite_min = float(finite_values.min())
        finite_max = float(finite_values.max())
        finite_mean = float(finite_values.mean())

    return (
        f"field={name}, shape={shape}, dtype={dtype}, "
        f"nan_count={nan_count}, inf_count={inf_count}, "
        f"finite_min={finite_min:.6e}, finite_max={finite_max:.6e}, finite_mean={finite_mean:.6e}"
    )


def _to_python_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        return value.reshape(-1)[0].item()
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return value.reshape(-1)[0].item()
    return value


def _count_values(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    if isinstance(value, np.ndarray):
        return int(value.size)
    return 0
