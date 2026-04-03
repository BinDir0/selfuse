#!/usr/bin/env python3
"""Advanced helper for building a frozen clip manifest snapshot."""

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.pipeline_config import normalize_pipeline_config


def get_parser():
    parser = argparse.ArgumentParser(
        description="Advanced helper: build clip manifest from dataset adapter or shard directories"
    )
    parser.add_argument("--config", type=str, default=None, help="Optional pipeline YAML config with dataset adapter settings")
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--shard_root", type=str, help="Root directory containing shard-group directories")
    source_group.add_argument("--shard_dir_list", type=str, help="Text file with one shard directory per line")
    parser.add_argument("--adapter", type=str, default=None, help="Dataset adapter override when using --config")
    parser.add_argument("--source_id", type=str, required=False, help="Logical source identifier, e.g. buildai")
    parser.add_argument("--split", type=str, default="train", help="Dataset split label")
    parser.add_argument("--include_dirs", type=str, nargs="*", default=None, help="Optional shard-group directory names to include")
    parser.add_argument("--manifest_out", type=str, required=True, help="Output JSONL manifest path")
    parser.add_argument("--shard_dirs_out", type=str, default=None, help="Optional output text file listing resolved shard dirs")
    parser.add_argument(
        "--seq_folder_root",
        type=str,
        default=None,
        help="Optional root containing <factory>/outputs/<clip_id> stage outputs to override descriptor.seq_folder",
    )
    return parser


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return normalize_pipeline_config(yaml.safe_load(handle) or {})


def main():
    from lib.pipeline.clip_manifest import (
        build_clip_manifest_records,
        build_manifest_records_from_descriptors,
        discover_shard_dirs,
        remap_descriptor_seq_folders,
        write_clip_manifest,
        write_shard_dir_list,
    )
    from lib.pipeline.datasets import get_dataset_adapter
    from lib.pipeline.frame_sources import classify_descriptor_storage

    args = get_parser().parse_args()
    shard_dirs = []

    if args.config:
        config = load_yaml(args.config)
        dataset_cfg = config.get("dataset", {})
        adapter_cfg = config.get("adapter_config", {})
        paths_cfg = config.get("paths", {})
        adapter_name = args.adapter or dataset_cfg.get("adapter") or dataset_cfg.get("source_type", "buildai")
        source_id = args.source_id or dataset_cfg.get("source_id", adapter_name)
        split = dataset_cfg.get("split", args.split)
        adapter = get_dataset_adapter(adapter_name)
        descriptors = list(
            adapter.build_descriptors(
                dataset_cfg=dataset_cfg,
                adapter_cfg=adapter_cfg,
                paths_cfg=paths_cfg,
                context=None,
                prepared=None,
            )
        )
        if args.seq_folder_root:
            remap_descriptor_seq_folders(descriptors, args.seq_folder_root)
        records = build_manifest_records_from_descriptors(
            descriptors,
            source_id=source_id,
            split=split,
        )
        if args.shard_dirs_out and paths_cfg.get("shard_root"):
            shard_dirs = discover_shard_dirs(
                paths_cfg["shard_root"],
                include_dirs=adapter_cfg.get("include_dirs") or dataset_cfg.get("include_dirs"),
            )
    else:
        if not args.source_id:
            raise RuntimeError("--source_id is required when --config is not provided")
        if not args.shard_root and not args.shard_dir_list:
            raise RuntimeError("One of --shard_root or --shard_dir_list is required when --config is not provided")

        if args.shard_dir_list:
            with open(args.shard_dir_list, "r", encoding="utf-8") as handle:
                shard_dirs = [line.strip() for line in handle if line.strip()]
        else:
            shard_dirs = discover_shard_dirs(args.shard_root, include_dirs=args.include_dirs)

        if not shard_dirs:
            raise RuntimeError("No shard directories found")

        records = build_clip_manifest_records(
            shard_dirs,
            source_id=args.source_id,
            split=args.split,
        )
        if args.seq_folder_root:
            remap_descriptor_seq_folders([record.descriptor for record in records], args.seq_folder_root)
    if not records:
        raise RuntimeError("No clips found while building manifest")

    write_clip_manifest(records, args.manifest_out)
    if args.shard_dirs_out and shard_dirs:
        write_shard_dir_list(shard_dirs, args.shard_dirs_out)

    summary = {
        "source_id": records[0].source_id,
        "split": records[0].split,
        "shard_dir_count": len(shard_dirs),
        "clip_count": len(records),
        "descriptor_paths": {},
        "manifest_out": str(Path(args.manifest_out).resolve()),
    }
    for record in records:
        kind = classify_descriptor_storage(record.descriptor)
        summary["descriptor_paths"][kind] = summary["descriptor_paths"].get(kind, 0) + 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
