"""
Shared utilities for full-chain verification tests.

Reuses the PhaseReport / CheckResult infrastructure from pretrain_verification
with output paths scoped to outputs/full_chain_verification/.
"""

from __future__ import annotations

from pathlib import Path

from src.tests.pretrain_verification.utils import (  # noqa: F401
    CheckResult,
    PhaseReport,
    assert_check,
    plot_bar_chart,
    plot_histogram_panel,
    plot_loss_curves,
    tensor_stats,
    safe_import_plt,
)


def get_output_dir(part: str) -> Path:
    """Return outputs/full_chain_verification/{part}/ under the project root."""
    project_root = Path(__file__).resolve().parents[3]
    out = project_root / "outputs" / "full_chain_verification" / part
    out.mkdir(parents=True, exist_ok=True)
    return out
