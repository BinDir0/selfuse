"""Annotation helpers for WebDataset export and rewrite flows."""

import json
import os

ANNOTATION_LEVEL_KEYS = ("level1", "level2", "level3", "level4", "level5")
DEFAULT_ANNOTATION_SUFFIX = "_qwen-annotation.json"


def get_episode_annotation_path(ep, annotation_suffix=DEFAULT_ANNOTATION_SUFFIX):
    """Return the qwen-annotation JSON path for an episode."""
    parent_dir = os.path.dirname(ep["crop_dir"])
    return os.path.join(parent_dir, f"{ep['episode_id']}{annotation_suffix}")


def normalize_instruction(global_analysis):
    """Normalize level1..level5 annotation text into an instruction list."""
    if not isinstance(global_analysis, dict):
        return []

    instruction = []
    for level_key in ANNOTATION_LEVEL_KEYS:
        value = global_analysis.get(level_key)
        if value is None:
            continue
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if value:
            instruction.append(value)
    return instruction


def load_episode_instruction(ep, annotation_suffix=DEFAULT_ANNOTATION_SUFFIX):
    """Load and validate one episode's qwen annotation."""
    annotation_path = get_episode_annotation_path(ep, annotation_suffix=annotation_suffix)
    if not os.path.exists(annotation_path):
        return None, "missing_annotation", annotation_path

    try:
        with open(annotation_path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None, "invalid_json", annotation_path

    status = str(payload.get("status", "")).strip()
    if status != "Valid":
        return None, "invalid_status", annotation_path

    instruction = normalize_instruction(payload.get("global_analysis"))
    if not instruction:
        return None, "empty_instruction", annotation_path

    return instruction, None, annotation_path


def attach_or_filter_episode_instructions(episodes, annotation_suffix=DEFAULT_ANNOTATION_SUFFIX, allow_missing_annotation=False):
    """Attach instruction to episodes or drop invalid entries."""
    kept = []
    stats = {
        "kept": 0,
        "filtered": 0,
        "missing_annotation": 0,
        "invalid_json": 0,
        "invalid_status": 0,
        "empty_instruction": 0,
    }

    for ep in episodes:
        instruction, error_code, annotation_path = load_episode_instruction(ep, annotation_suffix=annotation_suffix)
        ep_copy = dict(ep)
        ep_copy["annotation_path"] = annotation_path

        if instruction is None:
            stats[error_code] = stats.get(error_code, 0) + 1
            if allow_missing_annotation:
                ep_copy["instruction"] = []
                kept.append(ep_copy)
                stats["kept"] += 1
            else:
                stats["filtered"] += 1
            continue

        ep_copy["instruction"] = instruction
        kept.append(ep_copy)
        stats["kept"] += 1

    return kept, stats
