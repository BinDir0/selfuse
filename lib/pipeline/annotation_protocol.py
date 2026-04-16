"""Standard clip-level annotation protocol helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ANNOTATION_SUFFIX = ".annotation.json"
HIERARCHY_KEYS = ("level1", "level2", "level3", "level4", "level5")
_FACTORY_CLIP_ID_PATTERN = re.compile(r"^f(\d{3})_")


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


def annotation_path_candidates(
    annotation_root: str | Path,
    clip_id: str,
    *,
    annotation_suffix: str = ANNOTATION_SUFFIX,
) -> list[Path]:
    root = Path(annotation_root)
    candidates = [annotation_path(root, clip_id, annotation_suffix=annotation_suffix)]

    match = _FACTORY_CLIP_ID_PATTERN.match(str(clip_id))
    if match:
        factory_dir = root / f"factory{int(match.group(1)):03d}"
        nested = factory_dir / f"{clip_id}{annotation_suffix}"
        if nested not in candidates:
            candidates.append(nested)

    return candidates


def resolve_annotation_path(
    annotation_root: str | Path,
    clip_id: str,
    *,
    annotation_suffix: str = ANNOTATION_SUFFIX,
) -> tuple[Optional[Path], str]:
    candidates = annotation_path_candidates(
        annotation_root,
        clip_id,
        annotation_suffix=annotation_suffix,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate, str(candidate)
    return None, str(candidates[-1])


def build_annotation_issue(clip_id: str, error_code: str, resolved_path: str) -> dict:
    return {
        "clip_id": str(clip_id),
        "error_code": str(error_code),
        "resolved_path": str(resolved_path),
    }


def summarize_annotation_issues(issues: list[dict]) -> dict:
    summary = {
        "total": len(issues),
        "missing_annotation": 0,
        "invalid_json": 0,
        "invalid_status": 0,
        "empty_instruction": 0,
        "other": 0,
    }
    for item in issues:
        code = str(item.get("error_code") or "")
        if code in summary:
            summary[code] += 1
        else:
            summary["other"] += 1
    return summary


def write_annotation_issue_report(
    report_path: str | Path,
    *,
    annotation_root: str | Path | None,
    annotation_suffix: str,
    issues: list[dict],
    context: Optional[dict] = None,
) -> str:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "annotation_root": None if annotation_root is None else str(Path(annotation_root).resolve()),
        "annotation_suffix": str(annotation_suffix),
        "summary": summarize_annotation_issues(issues),
        "issues": list(issues),
        "context": dict(context or {}),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path.resolve())


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
    path, resolved_path = resolve_annotation_path(
        annotation_root,
        clip_id,
        annotation_suffix=annotation_suffix,
    )
    if path is None:
        return None, "missing_annotation", resolved_path

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
