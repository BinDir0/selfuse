#!/usr/bin/env python3
"""Legacy CLI wrapper for the old BuildAI-oriented WebDataset exporter.

DEPRECATED: Use ``scripts/run_dataset_pipeline.py`` or
``scripts/build_vla_from_manifest.py`` instead.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.webdataset import main


if __name__ == "__main__":
    main()
