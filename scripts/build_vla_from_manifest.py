#!/usr/bin/env python3
"""Build final VLA WebDataset from a frozen clip manifest snapshot."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_parser():
    parser = argparse.ArgumentParser(description="Build VLA dataset from clip manifest + HaWoR outputs")
    parser.add_argument("--descriptor_manifest", type=str, required=True, help="Frozen clip manifest JSONL path")
    parser.add_argument("--output_dir", type=str, required=True, help="Final WebDataset output directory")
    parser.add_argument("--annotation_root", type=str, default=None, help="Clip annotation sidecar directory")
    parser.add_argument("--require_annotation", action="store_true", help="Drop clips with missing or invalid annotations")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes for testing")
    parser.add_argument("--repeat_episodes", type=int, default=1, help="Repeat the manifest entries this many times")
    parser.add_argument("--preprocess_workers", type=int, default=8, help="Workers for manifest preparation")
    parser.add_argument("--writer_workers", type=int, default=4, help="Workers for shard writing")
    parser.add_argument("--frames_per_shard", type=int, default=10000, help="Approximate frame budget per shard")
    parser.add_argument("--mano_device", type=str, default="cuda:0", help="Device for MANO forward pass")
    parser.add_argument("--mano_gpus", type=str, default=None, help="Optional comma-separated GPU list for MANO workers")
    parser.add_argument("--mano_dir", type=str, default=None, help="Optional MANO model directory")
    return parser


def main():
    args = get_parser().parse_args()
    from lib.pipeline.exporters.manifest_vla import run_manifest_build

    result = run_manifest_build(
        manifest_path=args.descriptor_manifest,
        output_dir=args.output_dir,
        annotation_root=args.annotation_root,
        require_annotation=args.require_annotation,
        max_episodes=args.max_episodes,
        repeat_episodes=args.repeat_episodes,
        preprocess_workers=args.preprocess_workers,
        writer_workers=args.writer_workers,
        frames_per_shard=args.frames_per_shard,
        mano_device=args.mano_device,
        mano_gpus=args.mano_gpus,
        mano_dir=args.mano_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
