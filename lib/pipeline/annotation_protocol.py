"""Standard clip-level annotation protocol helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ANNOTATION_SUFFIX = ".annotation.json"
HIERARCHY_KEYS = ("level1", "level2", "level3", "level4", "level5")


@dataclass(frozen=True)
class ClipAnnotation:
    clip_id: str
    instruction: list[str]
    instruction_num: int
    language: Optional[str]
    hierarchy: dict
    source_path: str
    raw_payload: dict


def annotation_path(annotation_root: str | Path, clip_id: str, *, annotation_suffix: str = ANNOTATION_SUFFIX) -> Path:
    return Path(annotation_root) / f"{clip_id}{annotation_suffix}"


def _normalize_string_list(values) -> list[str]:
    normalized = []
    if not isinstance(values, list):
        return normalized
    for value in values:
        if value is None:
            continue
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if value:
            normalized.append(value)
    return normalized


def _normalize_hierarchy(payload: dict) -> dict:
    hierarchy = payload.get("hierarchy")
    if not isinstance(hierarchy, dict):
        hierarchy = payload.get("global_analysis")
    if not isinstance(hierarchy, dict):
        return {}

    normalized = {}
    for key in HIERARCHY_KEYS:
        value = hierarchy.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if value:
            normalized[key] = value
    return normalized


def _normalize_instruction(payload: dict, hierarchy: dict) -> list[str]:
    instruction = _normalize_string_list(payload.get("instruction"))
    if instruction:
        return instruction
    return [hierarchy[key] for key in HIERARCHY_KEYS if key in hierarchy]


def load_clip_annotation(
    annotation_root: str | Path,
    clip_id: str,
    *,
    annotation_suffix: str = ANNOTATION_SUFFIX,
) -> tuple[Optional[ClipAnnotation], Optional[str], str]:
    """Load one clip-level annotation sidecar."""
    path = annotation_path(annotation_root, clip_id, annotation_suffix=annotation_suffix)
    if not path.exists():
        return None, "missing_annotation", str(path)

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None, "invalid_json", str(path)

    status = str(payload.get("status", "Valid")).strip()
    if status != "Valid":
        return None, "invalid_status", str(path)

    hierarchy = _normalize_hierarchy(payload)
    instruction = _normalize_instruction(payload, hierarchy)
    if not instruction:
        return None, "empty_instruction", str(path)

    language = payload.get("language")
    if language is not None:
        language = str(language).strip() or None

    annotation = ClipAnnotation(
        clip_id=clip_id,
        instruction=instruction,
        instruction_num=len(instruction),
        language=language,
        hierarchy=hierarchy,
        source_path=str(path),
        raw_payload=payload,
    )
    return annotation, None, str(path)
