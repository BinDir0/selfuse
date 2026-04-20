#!/usr/bin/env python3
"""Standalone sanity checker for built WebDataset shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.wds_sanity import analyze_webdataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sanity checks on built WebDataset shards")
    parser.add_argument("--source_shard_dir", required=True, help="Directory containing shard tar files")
    parser.add_argument("--report_out", default=None, help="Optional JSON report output path")
    parser.add_argument(
        "--start_shard",
        type=int,
        default=0,
        help="Start shard index in sorted shard order (inclusive)",
    )
    parser.add_argument(
        "--end_shard",
        type=int,
        default=None,
        help="End shard index in sorted shard order (exclusive)",
    )
    parser.add_argument("--sample_limit", type=int, default=None, help="Optional max number of samples to scan")
    parser.add_argument("--episode_limit", type=int, default=None, help="Optional max number of episodes to scan")
    parser.add_argument("--render_dir", default=None, help="Optional directory for sampled episode videos")
    parser.add_argument("--render_episodes", type=int, default=2, help="How many episodes to render when --render_dir is set")
    parser.add_argument("--render_max_frames", type=int, default=180, help="Max frames per rendered episode")
    parser.add_argument("--render_fps", type=int, default=15, help="FPS for rendered episode videos")
    parser.add_argument(
        "--decode_images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode JPEG frames to catch image corruption",
    )
    parser.add_argument("--temp_dir", default=None, help="Optional temp dir for lowdim stats backing file")
    parser.add_argument("--max_issue_examples", type=int, default=32, help="How many issue examples to keep in the report")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = analyze_webdataset(
        source_shard_dir=args.source_shard_dir,
        start_shard=args.start_shard,
        end_shard=args.end_shard,
        sample_limit=args.sample_limit,
        episode_limit=args.episode_limit,
        render_dir=args.render_dir,
        render_episodes=args.render_episodes if args.render_dir else 0,
        render_max_frames=args.render_max_frames,
        render_fps=args.render_fps,
        decode_images=bool(args.decode_images),
        temp_dir=args.temp_dir,
        max_issue_examples=args.max_issue_examples,
    )

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
