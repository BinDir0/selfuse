#!/usr/bin/env python3
"""Lightweight checker for instruction fields in built WebDataset shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.webdataset_rewriter import iter_shard_paths, iter_shard_samples  # noqa: E402
from lib.pipeline.wds_sanity import clip_id_from_sample, parse_instruction_entries  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check WDS instruction metadata only")
    parser.add_argument("--source_shard_dir", required=True, help="Directory containing shard tar files")
    parser.add_argument("--report_out", default=None, help="Optional JSON report output path")
    parser.add_argument("--start_shard", type=int, default=0, help="Start shard index in sorted order (inclusive)")
    parser.add_argument("--end_shard", type=int, default=None, help="End shard index in sorted order (exclusive)")
    parser.add_argument("--sample_limit", type=int, default=None, help="Optional max number of samples to scan")
    parser.add_argument("--episode_limit", type=int, default=None, help="Optional max number of clips to scan")
    parser.add_argument("--max_examples", type=int, default=64, help="Max issue examples to keep")
    return parser


def select_shards(source_shard_dir: str, start_shard: int, end_shard: int | None) -> tuple[list[str], int, int]:
    shard_paths = list(iter_shard_paths(source_shard_dir))
    if not shard_paths:
        raise RuntimeError(f"No shard tar files found in {source_shard_dir}")
    total_shards = len(shard_paths)
    if start_shard < 0 or start_shard >= total_shards:
        raise ValueError(f"--start_shard {start_shard} is out of range [0, {total_shards})")
    resolved_end = total_shards if end_shard is None else int(end_shard)
    if resolved_end < start_shard or resolved_end > total_shards:
        raise ValueError(f"--end_shard {resolved_end} is out of range [{start_shard}, {total_shards}]")
    selected = shard_paths[start_shard:resolved_end]
    if not selected:
        raise RuntimeError(f"No shards selected in range [{start_shard}, {resolved_end}) from {source_shard_dir}")
    return selected, total_shards, resolved_end


def append_example(report: dict, reason: str, *, clip_id: str, sample_key: str, shard_name: str, detail=None) -> None:
    if len(report["issue_examples"]) >= int(report["config"]["max_examples"]):
        return
    record = {
        "reason": reason,
        "clip_id": clip_id,
        "sample_key": sample_key,
        "shard_name": shard_name,
    }
    if detail is not None:
        record["detail"] = detail
    report["issue_examples"].append(record)


def update_clip_issue(report: dict, clip_id: str, reason: str) -> None:
    clip_stats = report["problem_clips"].setdefault(
        clip_id,
        {
            "clip_id": clip_id,
            "missing_meta_frames": 0,
            "invalid_meta_frames": 0,
            "missing_instruction_frames": 0,
            "empty_instruction_frames": 0,
            "instruction_num_mismatch_frames": 0,
        },
    )
    clip_stats[f"{reason}_frames"] += 1


def main() -> None:
    args = build_parser().parse_args()
    selected_shards, total_shards, resolved_end = select_shards(
        args.source_shard_dir,
        args.start_shard,
        args.end_shard,
    )

    report = {
        "source_shard_dir": str(Path(args.source_shard_dir).resolve()),
        "config": {
            "start_shard": int(args.start_shard),
            "end_shard": int(resolved_end),
            "selected_shards": len(selected_shards),
            "total_shards_available": int(total_shards),
            "sample_limit": None if args.sample_limit is None else int(args.sample_limit),
            "episode_limit": None if args.episode_limit is None else int(args.episode_limit),
            "max_examples": int(args.max_examples),
            "shards": [Path(path).name for path in selected_shards],
        },
        "summary": {
            "samples_total": 0,
            "clips_total": 0,
            "problem_clips": 0,
        },
        "checks": {
            "missing_meta_frames": 0,
            "invalid_meta_frames": 0,
            "missing_instruction_frames": 0,
            "empty_instruction_frames": 0,
            "instruction_num_mismatch_frames": 0,
        },
        "issue_examples": [],
        "problem_clips": {},
    }

    seen_clips: set[str] = set()
    current_clip_id: str | None = None

    for shard_path in selected_shards:
        shard_name = Path(shard_path).name
        for sample in iter_shard_samples(shard_path):
            if args.sample_limit is not None and report["summary"]["samples_total"] >= int(args.sample_limit):
                break
            report["summary"]["samples_total"] += 1

            meta = None
            sample_key = sample["key"]
            if sample.get("meta_bytes") is not None:
                try:
                    meta = json.loads(sample["meta_bytes"].decode("utf-8"))
                except Exception as error:
                    clip_id = clip_id_from_sample(sample, None)
                    report["checks"]["invalid_meta_frames"] += 1
                    update_clip_issue(report, clip_id, "invalid_meta")
                    append_example(
                        report,
                        "invalid_meta",
                        clip_id=clip_id,
                        sample_key=sample_key,
                        shard_name=shard_name,
                        detail=str(error),
                    )
                    meta = None

            clip_id = clip_id_from_sample(sample, meta)
            if clip_id != current_clip_id:
                current_clip_id = clip_id
                if clip_id not in seen_clips:
                    seen_clips.add(clip_id)
                    report["summary"]["clips_total"] += 1
                    if args.episode_limit is not None and report["summary"]["clips_total"] > int(args.episode_limit):
                        break

            if sample.get("meta_bytes") is None:
                report["checks"]["missing_meta_frames"] += 1
                update_clip_issue(report, clip_id, "missing_meta")
                append_example(
                    report,
                    "missing_meta",
                    clip_id=clip_id,
                    sample_key=sample_key,
                    shard_name=shard_name,
                )
                continue

            if not isinstance(meta, dict):
                continue

            raw_instruction = meta.get("instruction", [])
            instruction_num, instructions = parse_instruction_entries(meta)
            raw_slots = [raw_instruction] if isinstance(raw_instruction, str) else list(raw_instruction) if isinstance(raw_instruction, (list, tuple)) else []
            effective_slots = raw_slots[:instruction_num]

            if instruction_num <= 0:
                report["checks"]["missing_instruction_frames"] += 1
                update_clip_issue(report, clip_id, "missing_instruction")
                append_example(
                    report,
                    "missing_instruction",
                    clip_id=clip_id,
                    sample_key=sample_key,
                    shard_name=shard_name,
                )
            elif not instructions:
                report["checks"]["empty_instruction_frames"] += 1
                update_clip_issue(report, clip_id, "empty_instruction")
                append_example(
                    report,
                    "empty_instruction",
                    clip_id=clip_id,
                    sample_key=sample_key,
                    shard_name=shard_name,
                    detail={"instruction_num": instruction_num, "instruction": effective_slots},
                )
            elif len(instructions) != min(instruction_num, len(effective_slots)):
                report["checks"]["instruction_num_mismatch_frames"] += 1
                update_clip_issue(report, clip_id, "instruction_num_mismatch")
                append_example(
                    report,
                    "instruction_num_mismatch",
                    clip_id=clip_id,
                    sample_key=sample_key,
                    shard_name=shard_name,
                    detail={
                        "instruction_num": instruction_num,
                        "non_empty_slots": len(instructions),
                        "instruction": effective_slots,
                    },
                )

        if args.sample_limit is not None and report["summary"]["samples_total"] >= int(args.sample_limit):
            break
        if args.episode_limit is not None and report["summary"]["clips_total"] >= int(args.episode_limit):
            break

    report["problem_clips"] = sorted(
        report["problem_clips"].values(),
        key=lambda item: (
            -(
                item["missing_meta_frames"]
                + item["invalid_meta_frames"]
                + item["missing_instruction_frames"]
                + item["empty_instruction_frames"]
                + item["instruction_num_mismatch_frames"]
            ),
            item["clip_id"],
        ),
    )
    report["summary"]["problem_clips"] = len(report["problem_clips"])

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
