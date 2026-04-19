#!/usr/bin/env python3
"""Fast checker for instruction fields in built WebDataset shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from multiprocessing import get_context
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.webdataset_rewriter import iter_shard_paths, split_sample_member_name  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check WDS instruction metadata only")
    parser.add_argument("--source_shard_dir", required=True, help="Directory containing shard tar files")
    parser.add_argument("--report_out", default=None, help="Optional JSON report output path")
    parser.add_argument("--start_shard", type=int, default=0, help="Start shard index in sorted order (inclusive)")
    parser.add_argument("--end_shard", type=int, default=None, help="End shard index in sorted order (exclusive)")
    parser.add_argument("--sample_limit", type=int, default=None, help="Optional max number of samples to scan")
    parser.add_argument("--episode_limit", type=int, default=None, help="Optional max number of clips to scan")
    parser.add_argument("--max_examples", type=int, default=64, help="Max issue examples to keep")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Shard-level parallel workers; falls back to 1 when sample/episode limits are used",
    )
    return parser


def parse_instruction_entries(meta: dict | None) -> tuple[int, list[str]]:
    if not isinstance(meta, dict):
        return 0, []

    raw_instruction_num = meta.get("instruction_num", 0)
    try:
        instruction_num = max(0, int(raw_instruction_num))
    except Exception:
        instruction_num = 0

    raw_instruction = meta.get("instruction", [])
    if isinstance(raw_instruction, str):
        slots = [raw_instruction]
    elif isinstance(raw_instruction, (list, tuple)):
        slots = list(raw_instruction)
    else:
        slots = []

    slots = slots[:instruction_num]
    cleaned = [str(item).strip() for item in slots if str(item).strip()]
    return instruction_num, cleaned


def clip_id_from_sample_key(sample_key: str, meta: dict | None) -> str:
    if isinstance(meta, dict):
        clip_id = meta.get("clip_id")
        if clip_id:
            return str(clip_id)
    return str(sample_key).rsplit("_f", 1)[0]


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


def init_shard_report(shard_name: str, max_examples: int) -> dict:
    return {
        "shard_name": shard_name,
        "samples_total": 0,
        "clips_total": 0,
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


def append_example(container: dict, max_examples: int, reason: str, *, clip_id: str, sample_key: str, shard_name: str, detail=None) -> None:
    if len(container["issue_examples"]) >= max_examples:
        return
    record = {
        "reason": reason,
        "clip_id": clip_id,
        "sample_key": sample_key,
        "shard_name": shard_name,
    }
    if detail is not None:
        record["detail"] = detail
    container["issue_examples"].append(record)


def update_clip_issue(container: dict, clip_id: str, reason: str) -> None:
    clip_stats = container["problem_clips"].setdefault(
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


def iter_shard_meta_entries(shard_path: str):
    with tarfile.open(shard_path, "r|") as tar_reader:
        for member in tar_reader:
            if not member.isfile():
                continue
            sample_key, _, field_name = split_sample_member_name(member.name)
            if sample_key is None:
                continue
            if field_name != "meta_bytes":
                continue
            member_file = tar_reader.extractfile(member)
            if member_file is None:
                yield sample_key, None
                continue
            yield sample_key, member_file.read()


def analyze_one_shard(task: tuple[str, int]) -> dict:
    shard_path, max_examples = task
    shard_name = Path(shard_path).name
    report = init_shard_report(shard_name, max_examples)
    seen_clips: set[str] = set()

    for sample_key, meta_bytes in iter_shard_meta_entries(shard_path):
        report["samples_total"] += 1
        meta = None

        if meta_bytes is None:
            clip_id = clip_id_from_sample_key(sample_key, None)
            report["checks"]["missing_meta_frames"] += 1
            update_clip_issue(report, clip_id, "missing_meta")
            append_example(report, max_examples, "missing_meta", clip_id=clip_id, sample_key=sample_key, shard_name=shard_name)
            if clip_id not in seen_clips:
                seen_clips.add(clip_id)
                report["clips_total"] += 1
            continue

        try:
            meta = json.loads(meta_bytes.decode("utf-8"))
        except Exception as error:
            clip_id = clip_id_from_sample_key(sample_key, None)
            report["checks"]["invalid_meta_frames"] += 1
            update_clip_issue(report, clip_id, "invalid_meta")
            append_example(
                report,
                max_examples,
                "invalid_meta",
                clip_id=clip_id,
                sample_key=sample_key,
                shard_name=shard_name,
                detail=str(error),
            )
            if clip_id not in seen_clips:
                seen_clips.add(clip_id)
                report["clips_total"] += 1
            continue

        clip_id = clip_id_from_sample_key(sample_key, meta)
        if clip_id not in seen_clips:
            seen_clips.add(clip_id)
            report["clips_total"] += 1

        raw_instruction = meta.get("instruction", [])
        instruction_num, instructions = parse_instruction_entries(meta)
        raw_slots = [raw_instruction] if isinstance(raw_instruction, str) else list(raw_instruction) if isinstance(raw_instruction, (list, tuple)) else []
        effective_slots = raw_slots[:instruction_num]

        if instruction_num <= 0:
            report["checks"]["missing_instruction_frames"] += 1
            update_clip_issue(report, clip_id, "missing_instruction")
            append_example(report, max_examples, "missing_instruction", clip_id=clip_id, sample_key=sample_key, shard_name=shard_name)
        elif not instructions:
            report["checks"]["empty_instruction_frames"] += 1
            update_clip_issue(report, clip_id, "empty_instruction")
            append_example(
                report,
                max_examples,
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
                max_examples,
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

    return report


def merge_worker_report(report: dict, worker_report: dict) -> None:
    report["summary"]["samples_total"] += int(worker_report["samples_total"])
    report["summary"]["clips_total"] += int(worker_report["clips_total"])

    for key, value in worker_report["checks"].items():
        report["checks"][key] += int(value)

    max_examples = int(report["config"]["max_examples"])
    remaining = max(0, max_examples - len(report["issue_examples"]))
    if remaining > 0:
        report["issue_examples"].extend(worker_report["issue_examples"][:remaining])

    for clip_id, clip_stats in worker_report["problem_clips"].items():
        merged = report["problem_clips"].setdefault(
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
        for key in (
            "missing_meta_frames",
            "invalid_meta_frames",
            "missing_instruction_frames",
            "empty_instruction_frames",
            "instruction_num_mismatch_frames",
        ):
            merged[key] += int(clip_stats.get(key, 0))


def build_final_report(args, selected_shards: list[str], total_shards: int, resolved_end: int) -> dict:
    shard_preview_limit = 64
    return {
        "source_shard_dir": str(Path(args.source_shard_dir).resolve()),
        "config": {
            "start_shard": int(args.start_shard),
            "end_shard": int(resolved_end),
            "selected_shards": len(selected_shards),
            "total_shards_available": int(total_shards),
            "sample_limit": None if args.sample_limit is None else int(args.sample_limit),
            "episode_limit": None if args.episode_limit is None else int(args.episode_limit),
            "max_examples": int(args.max_examples),
            "workers": int(args.workers),
            "shards_preview": [Path(path).name for path in selected_shards[:shard_preview_limit]],
            "shards_preview_truncated": max(0, len(selected_shards) - shard_preview_limit),
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


def main() -> None:
    args = build_parser().parse_args()
    selected_shards, total_shards, resolved_end = select_shards(
        args.source_shard_dir,
        args.start_shard,
        args.end_shard,
    )
    report = build_final_report(args, selected_shards, total_shards, resolved_end)

    workers = max(1, int(args.workers))
    if args.sample_limit is not None or args.episode_limit is not None:
        workers = 1

    if workers <= 1:
        for shard_path in selected_shards:
            worker_report = analyze_one_shard((shard_path, int(args.max_examples)))
            merge_worker_report(report, worker_report)
            if args.sample_limit is not None and report["summary"]["samples_total"] >= int(args.sample_limit):
                break
            if args.episode_limit is not None and report["summary"]["clips_total"] >= int(args.episode_limit):
                break
    else:
        tasks = [(shard_path, int(args.max_examples)) for shard_path in selected_shards]
        mp_context = get_context()
        with mp_context.Pool(workers) as pool:
            for worker_report in pool.imap_unordered(analyze_one_shard, tasks, chunksize=1):
                merge_worker_report(report, worker_report)

    problem_clips = sorted(
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
    report["problem_clips"] = problem_clips
    report["summary"]["problem_clips"] = len(problem_clips)

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
