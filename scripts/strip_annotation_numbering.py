#!/usr/bin/env python3
"""Strip leading list numbering from annotation JSON or WDS shard meta."""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.annotation_protocol import strip_leading_instruction_numbering  # noqa: E402


ANNOTATION_PATTERNS = ("*.annotation.json", "*_qwen-annotation.json")
SHARD_PATTERN = "*.tar"
HIERARCHY_KEYS = ("level1", "level2", "level3", "level4", "level5")


def strip_numbering(text: str) -> str:
    return strip_leading_instruction_numbering(text)


def normalize_instruction(value) -> tuple[object, bool]:
    if isinstance(value, str):
        updated = strip_numbering(value)
        return updated, updated != value
    if isinstance(value, list):
        changed = False
        normalized = []
        for item in value:
            if not isinstance(item, str):
                normalized.append(item)
                continue
            updated = strip_numbering(item)
            normalized.append(updated)
            changed = changed or updated != item
        return normalized, changed
    return value, False


def normalize_hierarchy(container) -> tuple[object, bool]:
    if not isinstance(container, dict):
        return container, False

    changed = False
    updated = dict(container)
    for key in HIERARCHY_KEYS:
        value = updated.get(key)
        if not isinstance(value, str):
            continue
        normalized = strip_numbering(value)
        if normalized != value:
            updated[key] = normalized
            changed = True
    return updated, changed


def iter_payload_updates(payload: dict):
    instruction = payload.get("instruction")
    if isinstance(instruction, str):
        updated = strip_numbering(instruction)
        if updated != instruction:
            yield "instruction", instruction, updated
    elif isinstance(instruction, list):
        for idx, item in enumerate(instruction):
            if not isinstance(item, str):
                continue
            updated = strip_numbering(item)
            if updated != item:
                yield f"instruction[{idx}]", item, updated

    for container_key in ("hierarchy", "global_analysis"):
        container = payload.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in HIERARCHY_KEYS:
            value = container.get(key)
            if not isinstance(value, str):
                continue
            updated = strip_numbering(value)
            if updated != value:
                yield f"{container_key}.{key}", value, updated


def normalize_payload(payload: dict) -> tuple[dict, bool]:
    changed = False
    updated = dict(payload)

    instruction, instruction_changed = normalize_instruction(updated.get("instruction"))
    if instruction_changed:
        updated["instruction"] = instruction
        changed = True

    hierarchy, hierarchy_changed = normalize_hierarchy(updated.get("hierarchy"))
    if hierarchy_changed:
        updated["hierarchy"] = hierarchy
        changed = True

    global_analysis, global_analysis_changed = normalize_hierarchy(updated.get("global_analysis"))
    if global_analysis_changed:
        updated["global_analysis"] = global_analysis
        changed = True

    return updated, changed


def dataset_name_from_meta(meta: dict, fallback: str) -> str:
    for key in ("dataset_name", "source_id", "dataset", "source"):
        value = meta.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return fallback


def new_summary() -> dict:
    return {
        "files": set(),
        "entries": 0,
        "fields": 0,
        "clips": set(),
        "examples": [],
    }


def add_summary(
    summary: dict,
    dataset_name: str,
    path: Path,
    payload: dict,
    updates: list[tuple[str, str, str]],
    max_examples: int,
) -> None:
    dataset_summary = summary.setdefault(dataset_name, new_summary())
    dataset_summary["files"].add(str(path))
    dataset_summary["entries"] += 1
    dataset_summary["fields"] += len(updates)
    clip_id = payload.get("clip_id") or payload.get("episode_id")
    if clip_id is not None and str(clip_id).strip():
        dataset_summary["clips"].add(str(clip_id))
    if len(dataset_summary["examples"]) < max_examples:
        dataset_summary["examples"].append(
            {
                "path": str(path),
                "clip_id": None if clip_id is None else str(clip_id),
                "updates": [
                    {
                        "field": field,
                        "before": before,
                        "after": after,
                    }
                    for field, before, after in updates[:5]
                ],
            }
        )


def finalize_summary(summary: dict) -> dict:
    finalized = {}
    for dataset_name, item in sorted(summary.items()):
        finalized[dataset_name] = {
            "files": len(item["files"]),
            "entries": int(item["entries"]),
            "fields": int(item["fields"]),
            "clips": len(item["clips"]),
            "examples": list(item["examples"]),
        }
    return finalized


def clean_annotation_file(
    path: Path,
    *,
    output_path: Path | None,
    dry_run: bool,
    summary: dict,
    max_examples: int,
) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    updates = list(iter_payload_updates(payload))
    if updates:
        add_summary(summary, "annotation_json", path, payload, updates, max_examples)
    updated, changed = normalize_payload(payload)
    if not dry_run:
        dest = output_path if output_path is not None else path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def rewrite_shard_file(
    path: Path,
    *,
    output_path: Path | None,
    dry_run: bool,
    summary: dict,
    max_examples: int,
) -> tuple[bool, int]:
    dest = output_path if output_path is not None else path
    tmp_path = dest.with_name(dest.name + ".tmp")
    changed = False
    updated_meta_members = 0
    tar_writer = None if dry_run else tarfile.open(tmp_path, "w")

    try:
        with tarfile.open(path, "r|") as tar_reader:
            for member in tar_reader:
                member_copy = copy.copy(member)
                extracted = tar_reader.extractfile(member) if member.isfile() else None

                if extracted is None:
                    if tar_writer is not None:
                        tar_writer.addfile(member_copy)
                    continue

                payload = extracted.read()
                if member.name.endswith(".meta.json"):
                    meta = json.loads(payload.decode("utf-8"))
                    updates = list(iter_payload_updates(meta))
                    updated_meta, meta_changed = normalize_payload(meta)
                    if meta_changed:
                        dataset_name = dataset_name_from_meta(meta, path.parent.name or "unknown")
                        add_summary(summary, dataset_name, path, meta, updates, max_examples)
                        changed = True
                        updated_meta_members += 1
                        if tar_writer is not None:
                            payload = json.dumps(updated_meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                            member_copy.size = len(payload)

                if tar_writer is not None:
                    tar_writer.addfile(member_copy, io.BytesIO(payload))
    finally:
        if tar_writer is not None:
            tar_writer.close()

    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp_path, dest)
    return changed, updated_meta_members


def iter_annotation_files(target: Path):
    if target.is_file():
        yield target
        return

    seen = set()
    for pattern in ANNOTATION_PATTERNS:
        for path in sorted(target.rglob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            yield path


def iter_shard_files(target: Path):
    if target.is_file():
        yield target
        return

    for path in sorted(target.rglob(SHARD_PATTERN)):
        yield path


def process_target(
    path: Path,
    *,
    output_path: Path | None,
    dry_run: bool,
    summary: dict,
    max_examples: int,
) -> tuple[str, bool, int]:
    suffixes = set(path.suffixes)
    if path.name.endswith(".tar"):
        changed, updated_members = rewrite_shard_file(
            path, output_path=output_path, dry_run=dry_run, summary=summary, max_examples=max_examples,
        )
        return "shard", changed, updated_members
    if ".json" in suffixes:
        changed = clean_annotation_file(
            path, output_path=output_path, dry_run=dry_run, summary=summary, max_examples=max_examples,
        )
        return "annotation", changed, 1 if changed else 0
    raise ValueError(f"Unsupported file type: {path}")


def _resolve_output_path(path: Path, target_roots: list[Path], output_dir: Path) -> Path | None:
    """Compute output path by finding which target root the path falls under."""
    for root in target_roots:
        try:
            rel = path.resolve().relative_to(root.resolve())
            return output_dir / root.name / rel
        except ValueError:
            continue
    # single file target or no match — put directly under output_dir
    return output_dir / path.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip leading '1. ' style numbering from annotation JSON or WDS shard meta."
    )
    parser.add_argument("target", nargs="+", help="Annotation JSON file/dir or WDS shard file/dir")
    parser.add_argument("--output-dir", default=None, help="Write all output to this directory instead of in-place")
    parser.add_argument("--dry-run", action="store_true", help="Only scan and report matches; do not edit files")
    parser.add_argument("--report-out", default=None, help="Optional JSON report path")
    parser.add_argument("--max-examples", type=int, default=20, help="Maximum examples to keep per dataset in the report")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    target_roots: list[Path] = []
    targets = []
    seen = set()
    for raw_target in args.target:
        target = Path(raw_target)
        if not target.exists():
            raise SystemExit(f"Target not found: {target}")

        target_roots.append(target)
        if target.is_file():
            selected = [target]
        else:
            annotation_paths = list(iter_annotation_files(target))
            shard_paths = list(iter_shard_files(target))
            selected = annotation_paths + [path for path in shard_paths if path not in annotation_paths]

        for path in selected:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            targets.append(path)

    if not targets:
        raise SystemExit("No supported annotation JSON or shard tar files found")

    total = 0
    changed = 0
    updated_entries = 0
    summary = {}
    for path in targets:
        total += 1
        out_path = _resolve_output_path(path, target_roots, output_dir) if output_dir else None
        file_kind, file_changed, file_updates = process_target(
            path,
            output_path=out_path,
            dry_run=args.dry_run,
            summary=summary,
            max_examples=max(0, args.max_examples),
        )
        updated_entries += int(file_updates)
        if file_changed:
            changed += 1
            verb = "would update" if args.dry_run else "updated"
            print(f"{verb} {file_kind} {path} entries={file_updates}")

    dataset_summary = finalize_summary(summary)
    for dataset_name, item in dataset_summary.items():
        print(
            f"dataset={dataset_name} files={item['files']} "
            f"entries={item['entries']} fields={item['fields']} clips={item['clips']}"
        )

    report = {
        "dry_run": bool(args.dry_run),
        "total_files": total,
        "changed_files": changed,
        "updated_entries": updated_entries,
        "datasets": dataset_summary,
    }
    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"report={report_path}")

    mode = "scan" if args.dry_run else "rewrite"
    print(f"done mode={mode} total={total} changed={changed} updated_entries={updated_entries}")


if __name__ == "__main__":
    main()
