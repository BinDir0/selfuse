#!/usr/bin/env python3
"""Rewrite legacy interpolated BuildAI WDS shards with current 30fps export semantics."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import rewrite_webdataset_lowdim as base


def build_parser():
    parser = base.build_parser()
    parser.description = "Rewrite legacy interpolated BuildAI WDS shards with current 30fps export semantics"
    parser.set_defaults(source_fps=5.0, target_fps=30.0, interpolate_labels=True)
    return parser


def main():
    args = build_parser().parse_args()
    base.run_from_args(args)


if __name__ == "__main__":
    main()
