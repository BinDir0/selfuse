"""CLI entrypoints for the dataset pipeline orchestrator."""

from __future__ import annotations

import argparse

from .constants import OFFICIAL_STAGE_ORDER


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the official dataset pipeline. Preferred stages: prepare, annotate, infer, filter, build, validate"
    )
    parser.add_argument("--config", type=str, required=True, help="YAML pipeline config")
    parser.add_argument(
        "--stages",
        type=str,
        default=",".join(OFFICIAL_STAGE_ORDER),
        help="Comma-separated stage list. Preferred: prepare,annotate,infer,filter,build,validate",
    )
    parser.add_argument("--run_tag", type=str, default=None, help="Optional run tag override")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume compatible stages. Currently forwarded to the build stage to skip existing non-empty shards.",
    )
    parser.add_argument(
        "--descriptor_manifest",
        type=str,
        default=None,
        help="Optional existing descriptor manifest JSONL to use when starting from infer/filter/build/validate without rerunning prepare/manifest.",
    )
    return parser
