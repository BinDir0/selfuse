#!/usr/bin/env python3
"""Compatibility wrapper for tools/ops/monitor_shared_runs.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    target = project_root / "tools" / "ops" / "monitor_shared_runs.py"
    print(
        "Warning: scripts/monitor_shared_runs.py is deprecated; use tools/ops/monitor_shared_runs.py instead.",
        file=sys.stderr,
    )
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
