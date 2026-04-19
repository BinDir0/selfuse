#!/usr/bin/env python3
"""Strip leading list numbering from annotation sidecar language fields in place."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ANNOTATION_PATTERNS = ("*.annotation.json", "*_qwen-annotation.json")
HIERARCHY_KEYS = ("level1", "level2", "level3", "level4", "level5")
NUMBERING_PREFIX_RE = re.compile(r"^\s*\d+\.\s+")


def strip_numbering(text: str) -> str:
    return NUMBERING_PREFIX_RE.sub("", text).strip()


def normalize_string_list(values) -> tuple[list[str], bool]:
    if not isinstance(values, list):
        return values, False

    changed = False
    normalized = []
    for value in values:
        if not isinstance(value, str):
            normalized.append(value)
            continue
        updated = strip_numbering(value)
        normalized.append(updated)
        changed = changed or updated != value
    return normalized, changed


def normalize_hierarchy(container) -> tuple[dict, bool]:
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


def clean_annotation_file(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    instruction, instruction_changed = normalize_string_list(payload.get("instruction"))
    if instruction_changed:
        payload["instruction"] = instruction
        changed = True

    hierarchy, hierarchy_changed = normalize_hierarchy(payload.get("hierarchy"))
    if hierarchy_changed:
        payload["hierarchy"] = hierarchy
        changed = True

    global_analysis, global_analysis_changed = normalize_hierarchy(payload.get("global_analysis"))
    if global_analysis_changed:
        payload["global_analysis"] = global_analysis
        changed = True

    if changed:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip leading '1. ' style numbering from annotation language fields.")
    parser.add_argument("target", help="Annotation file or directory to edit in place")
    args = parser.parse_args()

    target = Path(args.target)
    if not target.exists():
        raise SystemExit(f"Target not found: {target}")

    total = 0
    changed = 0
    for path in iter_annotation_files(target):
        total += 1
        if clean_annotation_file(path):
            changed += 1
            print(f"updated {path}")

    print(f"done total={total} changed={changed}")


if __name__ == "__main__":
    main()
