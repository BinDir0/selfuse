#!/usr/bin/env python3
"""Extract clip-level annotation sidecars from HOT3D WebDataset meta.json payloads."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.annotation_protocol import ANNOTATION_SUFFIX  # noqa: E402
from lib.pipeline.clip_manifest import load_clip_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract HOT3D clip-level annotations from WDS meta.json files")
    parser.add_argument("--descriptor_manifest", required=True, help="Pipeline clip manifest JSONL")
    parser.add_argument("--annotation_root", required=True, help="Output sidecar annotation directory")
    parser.add_argument("--annotation_suffix", default=ANNOTATION_SUFFIX, help="Annotation filename suffix")
    parser.add_argument("--report_out", default=None, help="Optional JSON report path")
    parser.add_argument(
        "--selection_strategy",
        choices=("longest", "first"),
        default="longest",
        help="How to choose one instruction list when frames disagree",
    )
    parser.add_argument("--max_examples", type=int, default=32, help="Max issue examples to keep")
    return parser


def _meta_member_for_image(frame_name: str) -> str:
    name = str(frame_name)
    if not name.endswith(".image.jpg"):
        raise ValueError(f"Expected HOT3D image member ending in .image.jpg, got: {name}")
    return f"{name[:-len('.image.jpg')]}.meta.json"


def _normalize_instruction(meta: dict | None) -> list[str]:
    if not isinstance(meta, dict):
        return []
    raw_instruction_num = meta.get("instruction_num")
    raw_instruction = meta.get("instruction", [])
    if isinstance(raw_instruction, str):
        slots = [raw_instruction]
    elif isinstance(raw_instruction, (list, tuple)):
        slots = list(raw_instruction)
    else:
        slots = []
    try:
        instruction_num = max(0, int(raw_instruction_num))
    except Exception:
        instruction_num = len(slots)
    slots = slots[:instruction_num]
    return [str(item).strip() for item in slots if str(item).strip()]


def _better_instruction(candidate: list[str], current: list[str], strategy: str) -> bool:
    if not candidate:
        return False
    if not current:
        return True
    if strategy == "first":
        return False
    return (len(candidate), sum(len(item) for item in candidate)) > (
        len(current),
        sum(len(item) for item in current),
    )


def _annotation_payload(record, instruction: list[str], source_meta: dict | None, frames_scanned: int) -> dict:
    hierarchy = {f"level{idx + 1}": value for idx, value in enumerate(instruction[:5])}
    extra = record.descriptor.extra or {}
    payload = {
        "clip_id": record.clip_id,
        "status": "Valid",
        "instruction": list(instruction),
        "instruction_num": len(instruction),
        "language": instruction[0] if instruction else None,
        "hierarchy": hierarchy,
        "source": {
            "dataset_name": "hot3d",
            "source_clip_id": record.clip_id,
            "original_episode_id": extra.get("original_episode_id"),
            "episode_index": extra.get("episode_index"),
            "part_index": extra.get("part_index"),
            "part_count": extra.get("part_count"),
            "source_shard": extra.get("source_shard"),
            "frame_start_idx": extra.get("frame_start_idx"),
            "frame_end_idx": extra.get("frame_end_idx"),
            "frames_scanned": frames_scanned,
        },
    }
    if isinstance(source_meta, dict):
        for key in (
            "dataset_name",
            "episode_index",
            "presence",
            "left_visibility_ratio",
            "right_visibility_ratio",
            "is_good_quality",
        ):
            if key in source_meta:
                payload["source"][f"meta_{key}"] = source_meta[key]
    return payload


def _append_issue(report: dict, max_examples: int, issue: dict) -> None:
    report["summary"]["issue_count"] += 1
    reason = str(issue.get("reason") or "other")
    report["issue_summary"][reason] = int(report["issue_summary"].get(reason, 0)) + 1
    if len(report["issue_examples"]) < max_examples:
        report["issue_examples"].append(issue)


def main() -> None:
    args = build_parser().parse_args()
    records = load_clip_manifest(args.descriptor_manifest)
    annotation_root = Path(args.annotation_root)
    annotation_root.mkdir(parents=True, exist_ok=True)

    by_shard: dict[str, list] = {}
    state = {}
    for record in records:
        shard_path = record.descriptor.shard_path
        if not shard_path:
            continue
        by_shard.setdefault(str(shard_path), []).append(record)
        state[record.clip_id] = {
            "instruction": [],
            "source_meta": None,
            "frames_scanned": 0,
            "missing_meta": 0,
            "invalid_meta": 0,
        }

    report = {
        "descriptor_manifest": str(Path(args.descriptor_manifest).resolve()),
        "annotation_root": str(annotation_root.resolve()),
        "summary": {
            "records_total": len(records),
            "annotations_written": 0,
            "valid_instruction": 0,
            "empty_instruction": 0,
            "issue_count": 0,
        },
        "issue_summary": {},
        "issue_examples": [],
    }

    max_examples = int(args.max_examples)
    for shard_path, shard_records in sorted(by_shard.items()):
        wanted: dict[str, list] = {}
        for record in shard_records:
            for frame_name in record.descriptor.frame_names:
                try:
                    meta_member = _meta_member_for_image(frame_name)
                except Exception as error:
                    _append_issue(
                        report,
                        max_examples,
                        {"reason": "invalid_frame_name", "clip_id": record.clip_id, "frame_name": frame_name, "error": str(error)},
                    )
                    continue
                wanted.setdefault(meta_member, []).append(record)

        seen = set()
        try:
            with tarfile.open(shard_path, "r|") as tar_reader:
                for member in tar_reader:
                    if not member.isfile() or member.name not in wanted:
                        continue
                    member_file = tar_reader.extractfile(member)
                    meta_bytes = None if member_file is None else member_file.read()
                    for record in wanted[member.name]:
                        clip_state = state[record.clip_id]
                        clip_state["frames_scanned"] += 1
                        seen.add((record.clip_id, member.name))
                        if meta_bytes is None:
                            clip_state["missing_meta"] += 1
                            continue
                        try:
                            meta = json.loads(meta_bytes.decode("utf-8"))
                        except Exception:
                            clip_state["invalid_meta"] += 1
                            continue
                        instruction = _normalize_instruction(meta)
                        if _better_instruction(instruction, clip_state["instruction"], args.selection_strategy):
                            clip_state["instruction"] = instruction
                            clip_state["source_meta"] = meta
        except Exception as error:
            for record in shard_records:
                _append_issue(
                    report,
                    max_examples,
                    {"reason": "shard_read_error", "clip_id": record.clip_id, "shard_path": shard_path, "error": str(error)},
                )
            continue

        for meta_member, member_records in wanted.items():
            for record in member_records:
                if (record.clip_id, meta_member) not in seen:
                    state[record.clip_id]["missing_meta"] += 1

    for record in records:
        clip_state = state.get(record.clip_id)
        if clip_state is None:
            _append_issue(report, max_examples, {"reason": "missing_shard_path", "clip_id": record.clip_id})
            continue

        instruction = clip_state["instruction"]
        output_path = annotation_root / f"{record.clip_id}{args.annotation_suffix}"
        output_path.write_text(
            json.dumps(
                _annotation_payload(record, instruction, clip_state["source_meta"], int(clip_state["frames_scanned"])),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        report["summary"]["annotations_written"] += 1
        if instruction:
            report["summary"]["valid_instruction"] += 1
        else:
            report["summary"]["empty_instruction"] += 1
            _append_issue(
                report,
                max_examples,
                {
                    "reason": "empty_instruction",
                    "clip_id": record.clip_id,
                    "frames_scanned": int(clip_state["frames_scanned"]),
                    "missing_meta": int(clip_state["missing_meta"]),
                    "invalid_meta": int(clip_state["invalid_meta"]),
                },
            )

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
