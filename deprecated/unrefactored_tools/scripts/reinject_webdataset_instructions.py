#!/usr/bin/env python3
"""Rewrite existing WebDataset shards with instruction metadata from clip annotations."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
from multiprocessing import get_context
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.annotation_protocol import (  # noqa: E402
    build_annotation_issue_from_candidates,
    load_clip_annotation,
    write_annotation_issue_report,
)
from lib.pipeline.exporters.webdataset_rewriter import (  # noqa: E402
    build_updated_meta_from_meta,
    iter_shard_paths,
    iter_shard_samples,
    split_sample_member_name,
    validate_sample_record,
    write_sample_to_tar,
)

_WORKER_ARGS = None


def _default_annotation_issue_report_path(output_dir: str | Path, shard_start: int, shard_end: int | None) -> Path:
    end_label = "all" if shard_end is None else f"{int(shard_end):06d}"
    return Path(output_dir) / f"_annotation_issues.shards_{int(shard_start):06d}_{end_label}.json"


def build_parser():
    parser = argparse.ArgumentParser(description="Reinject instruction metadata into existing WebDataset shards")
    parser.add_argument("--source_shard_dir", required=True, help="Directory containing source shard tar files")
    parser.add_argument("--output_dir", required=True, help="Directory for rewritten shard tar files")
    parser.add_argument("--annotation_root", required=True, help="Root directory containing clip annotations")
    parser.add_argument(
        "--mix_plan",
        default=None,
        help=(
            "Optional frozen mix plan JSONL. When provided, anonymized clip_id/episode_id values "
            "are mapped back to original_clip_id before loading annotations."
        ),
    )
    parser.add_argument(
        "--source_name",
        default=None,
        help=(
            "Optional source name filter used together with --mix_plan. Only plan entries from this "
            "source will be reinjected; all other samples keep existing meta unchanged."
        ),
    )
    parser.add_argument(
        "--annotation_suffix",
        default=".annotation.json",
        help="Annotation suffix, e.g. .annotation.json",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of shard rewrite workers",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip output shard files that already exist and are non-empty",
    )
    parser.add_argument("--shard_start", type=int, default=0, help="Start shard index (inclusive)")
    parser.add_argument("--shard_end", type=int, default=None, help="End shard index (exclusive)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail a shard on missing/invalid/empty annotations instead of keeping empty instructions",
    )
    parser.add_argument(
        "--drop_missing_annotation",
        action="store_true",
        help="Drop frames/clips with missing/invalid/empty annotations instead of writing them with empty instructions",
    )
    parser.add_argument(
        "--drop_missing_episodes",
        action="store_true",
        help="Drop whole episodes when annotation is missing or invalid_status; keep strict handling for other annotation errors.",
    )
    parser.add_argument(
        "--annotation_issue_report_out",
        default=None,
        help="Optional JSON path for missing/invalid annotation report; defaults to a shard-range-specific file in output_dir when issues exist",
    )
    return parser


def _load_mix_plan_lookup(mix_plan_path: str, source_name: str | None) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    with Path(mix_plan_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            planned_episode_id = str(record.get("episode_id") or "")
            if not planned_episode_id:
                continue
            record_source_name = None if record.get("source_name") is None else str(record.get("source_name"))
            if source_name is not None and record_source_name != str(source_name):
                continue
            lookup[planned_episode_id] = {
                "original_clip_id": str(record["original_clip_id"]),
                "source_name": record_source_name,
            }
    return lookup


def _load_mix_plan_target_shard_names(mix_plan_path: str, source_name: str | None) -> set[str]:
    shard_names: set[str] = set()
    with Path(mix_plan_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_source_name = None if record.get("source_name") is None else str(record.get("source_name"))
            if source_name is not None and record_source_name != str(source_name):
                continue
            shard_idx = record.get("output_shard_idx")
            if shard_idx is None:
                continue
            shard_names.add(f"shard-{int(shard_idx):06d}.tar")
    return shard_names


def _should_drop_annotation_issue(
    error_code: str | None,
    *,
    drop_missing_annotation: bool,
    drop_missing_episodes: bool,
) -> bool:
    code = str(error_code or "unknown")
    if drop_missing_annotation:
        return True
    if drop_missing_episodes and code in {"missing_annotation", "invalid_status"}:
        return True
    return False


def _link_or_copy_unchanged_shard(source_path: str, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)
    try:
        os.link(source_path, output_path)
        return "hardlink"
    except OSError:
        shutil.copy2(source_path, output_path)
        return "copy"


def _resolve_original_clip_id(meta: dict, mix_plan_lookup: dict[str, dict] | None) -> tuple[str | None, str | None]:
    lookup_clip_id = meta.get("clip_id") or meta.get("episode_id")
    if not lookup_clip_id:
        return None, None
    if mix_plan_lookup is None:
        return str(lookup_clip_id), str(lookup_clip_id)
    plan_entry = mix_plan_lookup.get(str(lookup_clip_id))
    if plan_entry is None:
        return None, str(lookup_clip_id)
    return str(plan_entry["original_clip_id"]), str(lookup_clip_id)


def _inspect_shard_actions(
    shard_path: str,
    *,
    annotation_root: str,
    mix_plan_lookup: dict[str, dict] | None,
    annotation_suffix: str,
    drop_missing_annotation: bool,
    drop_missing_episodes: bool,
) -> dict:
    annotation_cache: dict[str, tuple[object, str | None, str]] = {}
    clip_issue_details: dict[str, dict] = {}
    clip_stats: dict[str, str] = {}
    changed = False
    preserved_frames = 0
    dropped_frames = 0
    rewritten_frames = 0
    empty_frames = 0

    with tarfile.open(shard_path, "r|") as tar_reader:
        for member in tar_reader:
            if not member.isfile() or not member.name.endswith(".meta.json"):
                continue
            sample_key, _, _field_name = split_sample_member_name(member.name)
            if sample_key is None:
                continue
            member_file = tar_reader.extractfile(member)
            if member_file is None:
                raise RuntimeError(f"Failed to extract shard member: {member.name}")
            original_meta_bytes = member_file.read()
            meta = json.loads(original_meta_bytes.decode("utf-8"))
            original_clip_id, _lookup_clip_id = _resolve_original_clip_id(meta, mix_plan_lookup)
            if original_clip_id is None:
                preserved_frames += 1
                continue

            cached = annotation_cache.get(original_clip_id)
            if cached is None:
                cached = load_clip_annotation(
                    annotation_root,
                    original_clip_id,
                    annotation_suffix=annotation_suffix,
                )
                annotation_cache[original_clip_id] = cached
            annotation, error_code, annotation_path = cached
            if annotation is None:
                clip_stats.setdefault(original_clip_id, error_code or "unknown")
                if original_clip_id not in clip_issue_details:
                    clip_issue_details[original_clip_id] = build_annotation_issue_from_candidates(
                        annotation_root,
                        original_clip_id,
                        error_code or "unknown",
                        annotation_suffix=annotation_suffix,
                        resolved_path=annotation_path,
                    )
                should_drop = _should_drop_annotation_issue(
                    error_code,
                    drop_missing_annotation=drop_missing_annotation,
                    drop_missing_episodes=drop_missing_episodes,
                )
                if should_drop:
                    changed = True
                    dropped_frames += 1
                else:
                    updated_meta = build_updated_meta_from_meta(meta, [], language=meta.get("language"))
                    if updated_meta != original_meta_bytes:
                        changed = True
                        empty_frames += 1
                    else:
                        preserved_frames += 1
                continue

            updated_meta = build_updated_meta_from_meta(
                meta,
                annotation.instruction,
                language=annotation.language,
            )
            clip_stats.setdefault(original_clip_id, "updated")
            if updated_meta != original_meta_bytes:
                changed = True
                rewritten_frames += 1
            else:
                preserved_frames += 1

    return {
        "changed": bool(changed),
        "rewritten_frames": int(rewritten_frames),
        "empty_frames": int(empty_frames),
        "dropped_frames": int(dropped_frames),
        "preserved_frames": int(preserved_frames),
        "clip_counts": {
            "updated": sum(1 for value in clip_stats.values() if value == "updated"),
            "missing_annotation": sum(1 for value in clip_stats.values() if value == "missing_annotation"),
            "invalid_json": sum(1 for value in clip_stats.values() if value == "invalid_json"),
            "invalid_status": sum(1 for value in clip_stats.values() if value == "invalid_status"),
            "empty_instruction": sum(1 for value in clip_stats.values() if value == "empty_instruction"),
        },
        "annotation_issues": list(clip_issue_details.values()),
    }


def select_shard_paths(shard_dir: str, *, shard_start: int, shard_end: int | None, output_dir: str, resume: bool) -> list[str]:
    shard_paths = list(iter_shard_paths(shard_dir))
    selected = shard_paths[shard_start:shard_end]
    if not resume:
        return selected

    kept = []
    for shard_path in selected:
        output_path = os.path.join(output_dir, os.path.basename(shard_path))
        if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            continue
        kept.append(shard_path)
    return kept


def rewrite_shard(
    shard_path: str,
    *,
    output_dir: str,
    annotation_root: str,
    mix_plan_lookup: dict[str, dict] | None,
    target_shard_names: set[str] | None,
    source_name: str | None,
    annotation_suffix: str,
    strict: bool,
    drop_missing_annotation: bool,
    drop_missing_episodes: bool,
) -> dict:
    shard_name = os.path.basename(shard_path)
    output_path = os.path.join(output_dir, shard_name)
    tmp_path = output_path + ".tmp"
    if target_shard_names is not None and shard_name not in target_shard_names:
        link_mode = _link_or_copy_unchanged_shard(shard_path, output_path)
        return {
            "shard": shard_name,
            "output_path": output_path,
            "frames_written": 0,
            "rewritten_frames": 0,
            "empty_frames": 0,
            "preserved_frames": 0,
            "dropped_frames": 0,
            "dropped_clips": 0,
            "clip_counts": {
                "updated": 0,
                "missing_annotation": 0,
                "invalid_json": 0,
                "invalid_status": 0,
                "empty_instruction": 0,
            },
            "annotation_issues": [],
            "status": f"non_target_{link_mode}",
        }
    inspection = _inspect_shard_actions(
        shard_path,
        annotation_root=annotation_root,
        mix_plan_lookup=mix_plan_lookup,
        annotation_suffix=annotation_suffix,
        drop_missing_annotation=drop_missing_annotation,
        drop_missing_episodes=drop_missing_episodes,
    )
    if not inspection["changed"]:
        link_mode = _link_or_copy_unchanged_shard(shard_path, output_path)
        return {
            "shard": shard_name,
            "output_path": output_path,
            "frames_written": 0,
            "rewritten_frames": 0,
            "empty_frames": 0,
            "preserved_frames": int(inspection["preserved_frames"]),
            "dropped_frames": 0,
            "dropped_clips": 0,
            "clip_counts": dict(inspection["clip_counts"]),
            "annotation_issues": list(inspection["annotation_issues"]),
            "status": f"unchanged_{link_mode}",
        }
    frames_written = 0
    rewritten_frames = 0
    empty_frames = 0
    dropped_frames = 0
    preserved_frames = 0
    clip_stats: dict[str, str] = {}
    clip_issue_details: dict[str, dict] = {}
    annotation_cache: dict[str, tuple[object, str | None, str]] = {}
    dropped_clip_ids: set[str] = set()

    tar_writer = None
    try:
        for sample in iter_shard_samples(shard_path):
            validate_sample_record(sample)
            meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            original_clip_id, lookup_clip_id = _resolve_original_clip_id(meta, mix_plan_lookup)
            if lookup_clip_id is None:
                raise RuntimeError(f"Sample {sample['key']} missing clip_id/episode_id in meta")

            if original_clip_id is None:
                updated_meta = sample["meta_bytes"]
                preserved_frames += 1
                if tar_writer is None:
                    os.makedirs(output_dir, exist_ok=True)
                    tar_writer = tarfile.open(tmp_path, "w")
                write_sample_to_tar(
                    tar_writer,
                    sample["key"],
                    sample["image_bytes"],
                    sample["lowdim_bytes"],
                    updated_meta,
                    mano_bytes=sample.get("mano_bytes"),
                    depth_bytes=sample.get("depth_bytes"),
                )
                frames_written += 1
                continue

            cached = annotation_cache.get(original_clip_id)
            if cached is None:
                cached = load_clip_annotation(
                    annotation_root,
                    original_clip_id,
                    annotation_suffix=annotation_suffix,
                )
                annotation_cache[original_clip_id] = cached
            annotation, error_code, annotation_path = cached
            if annotation is None:
                clip_stats.setdefault(original_clip_id, error_code or "unknown")
                if original_clip_id not in clip_issue_details:
                    issue = build_annotation_issue_from_candidates(
                        annotation_root,
                        original_clip_id,
                        error_code or "unknown",
                        annotation_suffix=annotation_suffix,
                        resolved_path=annotation_path,
                    )
                    clip_issue_details[original_clip_id] = issue
                should_drop = _should_drop_annotation_issue(
                    error_code,
                    drop_missing_annotation=drop_missing_annotation,
                    drop_missing_episodes=drop_missing_episodes,
                )
                if strict and not should_drop:
                    raise RuntimeError(
                        f"Failed to load annotation for clip_id={original_clip_id}: {error_code} ({annotation_path})"
                    )
                if should_drop:
                    dropped_frames += 1
                    dropped_clip_ids.add(str(original_clip_id))
                    continue
                updated_meta = build_updated_meta_from_meta(meta, [], language=meta.get("language"))
                empty_frames += 1
            else:
                clip_stats.setdefault(original_clip_id, "updated")
                updated_meta = build_updated_meta_from_meta(
                    meta,
                    annotation.instruction,
                    language=annotation.language,
                )
                rewritten_frames += 1

            if tar_writer is None:
                os.makedirs(output_dir, exist_ok=True)
                tar_writer = tarfile.open(tmp_path, "w")

            write_sample_to_tar(
                tar_writer,
                sample["key"],
                sample["image_bytes"],
                sample["lowdim_bytes"],
                updated_meta,
                mano_bytes=sample.get("mano_bytes"),
                depth_bytes=sample.get("depth_bytes"),
            )
            frames_written += 1
    except Exception:
        if tar_writer is not None:
            tar_writer.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if tar_writer is not None:
        tar_writer.close()
        os.replace(tmp_path, output_path)

    counts = {
        "updated": sum(1 for value in clip_stats.values() if value == "updated"),
        "missing_annotation": sum(1 for value in clip_stats.values() if value == "missing_annotation"),
        "invalid_json": sum(1 for value in clip_stats.values() if value == "invalid_json"),
        "invalid_status": sum(1 for value in clip_stats.values() if value == "invalid_status"),
        "empty_instruction": sum(1 for value in clip_stats.values() if value == "empty_instruction"),
    }
    return {
        "shard": os.path.basename(shard_path),
        "output_path": output_path,
        "frames_written": frames_written,
        "rewritten_frames": rewritten_frames,
        "empty_frames": empty_frames,
        "preserved_frames": preserved_frames,
        "dropped_frames": dropped_frames,
        "dropped_clips": len(dropped_clip_ids),
        "clip_counts": counts,
        "annotation_issues": list(clip_issue_details.values()),
        "status": "rewritten",
    }


def _worker_init(args_dict: dict):
    global _WORKER_ARGS
    _WORKER_ARGS = dict(args_dict)


def _worker_rewrite_shard(shard_path: str) -> dict:
    return rewrite_shard(shard_path, **_WORKER_ARGS)


def main():
    args = build_parser().parse_args()
    shard_paths = select_shard_paths(
        args.source_shard_dir,
        shard_start=args.shard_start,
        shard_end=args.shard_end,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    if not shard_paths:
        print("No shard files selected.")
        return

    worker_args = {
        "output_dir": args.output_dir,
        "annotation_root": args.annotation_root,
        "mix_plan_lookup": None if args.mix_plan is None else _load_mix_plan_lookup(args.mix_plan, args.source_name),
        "target_shard_names": None if args.mix_plan is None else _load_mix_plan_target_shard_names(args.mix_plan, args.source_name),
        "source_name": None if args.source_name is None else str(args.source_name),
        "annotation_suffix": args.annotation_suffix,
        "strict": args.strict,
        "drop_missing_annotation": args.drop_missing_annotation,
        "drop_missing_episodes": args.drop_missing_episodes,
    }
    totals = {
        "shards": 0,
        "frames_written": 0,
        "rewritten_frames": 0,
        "empty_frames": 0,
        "preserved_frames": 0,
        "dropped_frames": 0,
        "dropped_clips": 0,
        "updated_clips": 0,
        "missing_annotation": 0,
        "invalid_json": 0,
        "invalid_status": 0,
        "empty_instruction": 0,
        "non_target_hardlink": 0,
        "non_target_copy": 0,
        "unchanged_hardlink": 0,
        "unchanged_copy": 0,
        "rewritten_shards": 0,
    }
    annotation_issue_map = {}

    target_shard_names = worker_args["target_shard_names"]
    if target_shard_names is not None:
        target_shard_paths = [path for path in shard_paths if os.path.basename(path) in target_shard_names]
        passthrough_shard_paths = [path for path in shard_paths if os.path.basename(path) not in target_shard_names]
        if passthrough_shard_paths:
            progress = tqdm(passthrough_shard_paths, desc="Link non-target shards")
            for shard_path in progress:
                output_path = os.path.join(args.output_dir, os.path.basename(shard_path))
                link_mode = _link_or_copy_unchanged_shard(shard_path, output_path)
                totals["shards"] += 1
                totals[f"non_target_{link_mode}"] += 1
    else:
        target_shard_paths = list(shard_paths)

    if not target_shard_paths:
        print(json.dumps(totals, ensure_ascii=False, indent=2))
        return

    if args.workers <= 1:
        result_iter = (
            rewrite_shard(
                shard_path,
                output_dir=args.output_dir,
                annotation_root=args.annotation_root,
                mix_plan_lookup=worker_args["mix_plan_lookup"],
                target_shard_names=worker_args["target_shard_names"],
                source_name=worker_args["source_name"],
                annotation_suffix=args.annotation_suffix,
                strict=args.strict,
                drop_missing_annotation=args.drop_missing_annotation,
                drop_missing_episodes=args.drop_missing_episodes,
            )
            for shard_path in target_shard_paths
        )
    else:
        mp_context = get_context()
        pool = mp_context.Pool(args.workers, initializer=_worker_init, initargs=(worker_args,))
        result_iter = pool.imap_unordered(_worker_rewrite_shard, target_shard_paths, chunksize=1)

    try:
        for result in tqdm(result_iter, total=len(target_shard_paths), desc="Rewrite target shards"):
            totals["shards"] += 1
            totals["frames_written"] += result["frames_written"]
            totals["rewritten_frames"] += result["rewritten_frames"]
            totals["empty_frames"] += result["empty_frames"]
            totals["preserved_frames"] += result.get("preserved_frames", 0)
            totals["dropped_frames"] += result.get("dropped_frames", 0)
            totals["dropped_clips"] += result.get("dropped_clips", 0)
            status = str(result.get("status") or "")
            if status == "rewritten":
                totals["rewritten_shards"] += 1
            elif status == "unchanged_hardlink":
                totals["unchanged_hardlink"] += 1
            elif status == "unchanged_copy":
                totals["unchanged_copy"] += 1
            for issue in result.get("annotation_issues", []):
                key = (issue.get("clip_id"), issue.get("error_code"), issue.get("resolved_path"))
                annotation_issue_map.setdefault(key, issue)
            for key, value in result["clip_counts"].items():
                if key == "updated":
                    totals["updated_clips"] += value
                else:
                    totals[key] += value
    finally:
        if args.workers > 1:
            pool.close()
            pool.join()

    report_path = None
    annotation_issues = list(annotation_issue_map.values())
    if annotation_issues or args.annotation_issue_report_out:
        report_path = write_annotation_issue_report(
            args.annotation_issue_report_out
            or _default_annotation_issue_report_path(args.output_dir, args.shard_start, args.shard_end),
            annotation_root=args.annotation_root,
            annotation_suffix=args.annotation_suffix,
            issues=annotation_issues,
            context={
                "source_shard_dir": str(Path(args.source_shard_dir).resolve()),
                "output_dir": str(Path(args.output_dir).resolve()),
                "strict": bool(args.strict),
                "drop_missing_annotation": bool(args.drop_missing_annotation),
            },
        )
        if annotation_issues:
            print(
                "Warning: "
                f"{len(annotation_issues)} clip(s) have missing/invalid/empty annotations; "
                f"report written to {report_path}"
            )
    if report_path is not None:
        totals["annotation_issue_report_path"] = report_path
        totals["annotation_issue_count"] = len(annotation_issues)

    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
