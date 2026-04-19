#!/usr/bin/env python3
"""Strip leading list numbering from annotation JSON or WDS shard meta in place."""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import re
import tarfile
from pathlib import Path


ANNOTATION_PATTERNS = ("*.annotation.json", "*_qwen-annotation.json")
SHARD_PATTERN = "*.tar"
HIERARCHY_KEYS = ("level1", "level2", "level3", "level4", "level5")
NUMBERING_PREFIX_RE = re.compile(r"^\s*\d+\.\s+")


def strip_numbering(text: str) -> str:
    return NUMBERING_PREFIX_RE.sub("", text).strip()


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


def clean_annotation_file(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    updated, changed = normalize_payload(payload)
    if changed:
        path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def rewrite_shard_file(path: Path) -> tuple[bool, int]:
    tmp_path = path.with_name(path.name + ".tmp")
    changed = False
    updated_meta_members = 0

    with tarfile.open(path, "r|") as tar_reader, tarfile.open(tmp_path, "w") as tar_writer:
        for member in tar_reader:
            member_copy = copy.copy(member)
            extracted = tar_reader.extractfile(member) if member.isfile() else None

            if extracted is None:
                tar_writer.addfile(member_copy)
                continue

            payload = extracted.read()
            if member.name.endswith(".meta.json"):
                meta = json.loads(payload.decode("utf-8"))
                updated_meta, meta_changed = normalize_payload(meta)
                if meta_changed:
                    payload = json.dumps(updated_meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    member_copy.size = len(payload)
                    changed = True
                    updated_meta_members += 1

            tar_writer.addfile(member_copy, io.BytesIO(payload))

    if changed:
        os.replace(tmp_path, path)
    else:
        tmp_path.unlink(missing_ok=True)
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


def process_target(path: Path) -> tuple[str, bool, int]:
    suffixes = set(path.suffixes)
    if path.name.endswith(".tar"):
        changed, updated_members = rewrite_shard_file(path)
        return "shard", changed, updated_members
    if ".json" in suffixes:
        changed = clean_annotation_file(path)
        return "annotation", changed, 1 if changed else 0
    raise ValueError(f"Unsupported file type: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip leading '1. ' style numbering from annotation JSON or WDS shard meta in place."
    )
    parser.add_argument("target", help="Annotation JSON file/dir or WDS shard file/dir to edit in place")
    args = parser.parse_args()

    target = Path(args.target)
    if not target.exists():
        raise SystemExit(f"Target not found: {target}")

    if target.is_file():
        targets = [target]
    else:
        annotation_paths = list(iter_annotation_files(target))
        shard_paths = list(iter_shard_files(target))
        targets = annotation_paths + [path for path in shard_paths if path not in annotation_paths]

    if not targets:
        raise SystemExit(f"No supported annotation JSON or shard tar files found under: {target}")

    total = 0
    changed = 0
    updated_entries = 0
    for path in targets:
        total += 1
        file_kind, file_changed, file_updates = process_target(path)
        updated_entries += int(file_updates)
        if file_changed:
            changed += 1
            print(f"updated {file_kind} {path}")

    print(f"done total={total} changed={changed} updated_entries={updated_entries}")


if __name__ == "__main__":
    main()
