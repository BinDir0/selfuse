#!/usr/bin/env python3
"""

"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent / "analyze_episodes_full.py"


def main() -> None:
    if not _SCRIPT.is_file():
        print(f"Missing {_SCRIPT}", file=sys.stderr)
        sys.exit(1)
    cmd = [sys.executable, str(_SCRIPT), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=str(_SCRIPT.parent)))


if __name__ == "__main__":
    main()
