#!/usr/bin/env python3
"""Compatibility wrapper for tools/ops/serve_shared_run_monitor.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    target = project_root / "tools" / "ops" / "serve_shared_run_monitor.py"
    print(
        "Warning: scripts/serve_shared_run_monitor.py is deprecated; use tools/ops/serve_shared_run_monitor.py instead.",
        file=sys.stderr,
    )
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
