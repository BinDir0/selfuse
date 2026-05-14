#!/usr/bin/env python3
"""Repair legacy WebDataset lowdim rot6d flatten order.

Older HaWoR WDS shards may store wrist rot6d as a row-major flatten of the
first two rotation-matrix columns:

  old = rotmat[:, :2].reshape(6)          # [r00, r01, r10, r11, r20, r21]

Current code expects the column-major/HMR2 layout:

  new = rotmat[:, :2].T.reshape(6)        # [r00, r10, r20, r01, r11, r21]

This script rewrites only the four wrist rot6d slots in each lowdim.npy:

  state.left   lowdim[6:12]
  state.right  lowdim[12:18]
  action.left  lowdim[54:60]
  action.right lowdim[60:66]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **_kwargs):
        return iterable if iterable is not None else ()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.webdataset_rewriter import (  # noqa: E402
    iter_shard_paths,
    iter_shard_samples,
    validate_sample_record,
    write_sample_to_tar,
)


ROT6D_SLICES = (
    (6, 12),
    (12, 18),
    (54, 60),
    (60, 66),
)
WRIST_TRANSLATION_SCALE_SLICES = (
    (0, 6),
    (48, 54),
)
HAND_STATE_ACTION_SCALE_SLICES = (
    (18, 48),
    (66, 96),
)
STATE_ACTION_SCALE_SLICES = WRIST_TRANSLATION_SCALE_SLICES + HAND_STATE_ACTION_SCALE_SLICES
REPAIR_MARKER_KEY = "lowdim_rot6d_repair"
REPAIR_MARKER_VALUE = "legacy_3x2_row_major_to_2x3_column_major"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rewrite WDS shards, repairing legacy lowdim rot6d flatten order.",
    )
    parser.add_argument("--source-shard-dir", required=True, help="Input directory containing shard-*.tar files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for repaired shards.")
    parser.add_argument("--shard-start", type=int, default=0, help="Inclusive shard index in sorted shard order.")
    parser.add_argument("--shard-end", type=int, default=None, help="Exclusive shard index in sorted shard order.")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)), help="Parallel shard workers.")
    parser.add_argument(
        "--executor",
        choices=("process", "thread"),
        default="process",
        help="Parallel backend. process is faster for np.load/np.save-heavy rewrites; thread can be gentler on memory.",
    )
    parser.add_argument("--report-out", default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--progress-out",
        default=None,
        help="Optional JSONL progress file with per-shard start/sample/finish events.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Sample interval for --progress-out heartbeats. Default: 1000.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip shards whose output tar already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Scan and report planned rewrites without writing output shards.",
    )
    parser.add_argument(
        "--dirty-seed",
        default="0",
        help="Seed string for deterministic episode-level dirty sampling.",
    )
    parser.add_argument(
        "--keep-legacy-rot6d-episode-fraction",
        type=float,
        default=0.0,
        help="Fraction of episodes to leave with legacy broken rot6d instead of repairing.",
    )
    parser.add_argument(
        "--dirty-instruction-episode-fraction",
        type=float,
        default=0.0,
        help="Fraction of episodes whose instructions should be dirtied.",
    )
    parser.add_argument(
        "--dirty-instruction-mode",
        choices=("empty", "generic"),
        default="empty",
        help="Instruction dirtying mode: empty clears instructions; generic uses --generic-instruction.",
    )
    parser.add_argument(
        "--generic-instruction",
        default="do something useful",
        help="Replacement instruction for --dirty-instruction-mode generic.",
    )
    parser.add_argument(
        "--dirty-state-action-scale-episode-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of episodes whose hand state/action dims and wrist translation dims "
            "should be randomly scaled. Wrist rot6d dims are not scaled."
        ),
    )
    parser.add_argument(
        "--dirty-state-action-scale-min",
        type=float,
        default=0.9,
        help="Minimum per-sample, per-dimension state/action scale factor.",
    )
    parser.add_argument(
        "--dirty-state-action-scale-max",
        type=float,
        default=1.1,
        help="Maximum per-sample, per-dimension state/action scale factor.",
    )
    return parser


def _is_same_or_nested(path_a: Path, path_b: Path) -> bool:
    if path_a == path_b:
        return True
    try:
        path_a.relative_to(path_b)
        return True
    except ValueError:
        return False


def validate_io_dirs(source_dir: Path, output_dir: Path) -> None:
    source_resolved = source_dir.resolve()
    output_resolved = output_dir.resolve()
    if source_resolved == output_resolved:
        raise SystemExit("--output-dir must be different from --source-shard-dir")
    if _is_same_or_nested(output_resolved, source_resolved):
        raise SystemExit("--output-dir must not be inside --source-shard-dir")
    if _is_same_or_nested(source_resolved, output_resolved):
        raise SystemExit("--source-shard-dir must not be inside --output-dir")


def decode_npy(payload: bytes) -> np.ndarray:
    return np.load(io.BytesIO(payload), allow_pickle=False)


def encode_npy(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return buffer.getvalue()


def repair_legacy_rot6d_vector(rot6d: np.ndarray) -> np.ndarray:
    arr = np.asarray(rot6d, dtype=np.float32)
    if arr.shape != (6,):
        raise ValueError(f"Expected rot6d shape (6,), got {arr.shape}")
    return arr.reshape(3, 2).T.reshape(6).astype(np.float32, copy=False)


def repair_lowdim(lowdim: np.ndarray) -> np.ndarray:
    repaired = np.asarray(lowdim, dtype=np.float32).copy()
    if repaired.ndim != 1 or repaired.shape[0] < 66:
        raise ValueError(f"Expected lowdim with at least 66 dims, got {repaired.shape}")
    for start, end in ROT6D_SLICES:
        repaired[start:end] = repair_legacy_rot6d_vector(repaired[start:end])
    return repaired


def stable_u64(seed: str, namespace: str, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def dirty_state_action_scale_lowdim(
    lowdim: np.ndarray,
    *,
    seed: str,
    sample_key: str,
    scale_min: float,
    scale_max: float,
) -> np.ndarray:
    scaled = np.asarray(lowdim, dtype=np.float32).copy()
    if scaled.ndim != 1 or scaled.shape[0] < 96:
        raise ValueError(f"Expected lowdim with at least 96 dims, got {scaled.shape}")
    rng = np.random.default_rng(stable_u64(seed, "dirty_state_action_scale_values", sample_key))
    for start, end in STATE_ACTION_SCALE_SLICES:
        factors = rng.uniform(float(scale_min), float(scale_max), size=end - start).astype(np.float32)
        scaled[start:end] *= factors
    return scaled


def episode_id_for_sample(sample_key: str, meta: dict[str, Any]) -> str:
    for key in ("clip_id", "episode_id"):
        value = meta.get(key)
        if value:
            return str(value)
    if "_f" in sample_key:
        return sample_key.rsplit("_f", 1)[0]
    return sample_key


def stable_episode_selected(seed: str, namespace: str, episode_id: str, fraction: float) -> bool:
    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    value = stable_u64(seed, namespace, episode_id) / float(1 << 64)
    return value < fraction


def validate_fraction(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise SystemExit(f"{name} must be in [0, 1], got {value}")


def validate_scale_range(scale_min: float, scale_max: float) -> None:
    if not np.isfinite(scale_min) or not np.isfinite(scale_max):
        raise SystemExit("--dirty-state-action-scale-min/max must be finite")
    if scale_min <= 0.0 or scale_max <= 0.0:
        raise SystemExit("--dirty-state-action-scale-min/max must be positive")
    if scale_min > scale_max:
        raise SystemExit("--dirty-state-action-scale-min must be <= --dirty-state-action-scale-max")


def update_meta_for_repair(
    meta: dict[str, Any],
    *,
    rot6d_repaired: bool,
    dirty_instruction_mode: str | None,
    generic_instruction: str,
    dirty_state_action_scale: bool = False,
    dirty_state_action_scale_min: float = 0.9,
    dirty_state_action_scale_max: float = 1.1,
) -> bytes:
    meta = dict(meta)
    existing_marker = meta.get(REPAIR_MARKER_KEY)
    if existing_marker:
        raise ValueError(f"sample already has {REPAIR_MARKER_KEY}={existing_marker!r}")

    dirty_flags = []
    if rot6d_repaired:
        meta[REPAIR_MARKER_KEY] = REPAIR_MARKER_VALUE
        meta["lowdim_rot6d_layout"] = "column_major_2x3"
    else:
        meta[REPAIR_MARKER_KEY] = "skipped_for_dirty_ablation"
        meta["lowdim_rot6d_layout"] = "legacy_3x2_row_major"
        dirty_flags.append("legacy_rot6d")

    if dirty_instruction_mode == "empty":
        meta["instruction"] = []
        meta["instruction_num"] = 0
        dirty_flags.append("empty_instruction")
    elif dirty_instruction_mode == "generic":
        meta["instruction"] = [generic_instruction]
        meta["instruction_num"] = 1
        meta["language"] = generic_instruction
        dirty_flags.append("generic_instruction")

    if dirty_state_action_scale:
        meta["dirty_state_action_scale_range"] = [
            float(dirty_state_action_scale_min),
            float(dirty_state_action_scale_max),
        ]
        meta["dirty_state_action_scale_fields"] = [
            "wrist_state.translation",
            "wrist_action.translation",
            "hand_state",
            "hand_action",
        ]
        dirty_flags.append("state_action_scale")

    if dirty_flags:
        meta["dirty_ablation_flags"] = sorted(set(meta.get("dirty_ablation_flags", []) + dirty_flags))
    return json.dumps(meta, ensure_ascii=False).encode("utf-8")


def mark_meta_repaired(meta_bytes: bytes) -> bytes:
    meta = json.loads(meta_bytes.decode("utf-8"))
    return update_meta_for_repair(
        meta,
        rot6d_repaired=True,
        dirty_instruction_mode=None,
        generic_instruction="do something useful",
    )


def _write_progress(progress_out: str | None, record: dict[str, Any]) -> None:
    if not progress_out:
        return
    with open(progress_out, "a", encoding="utf-8") as progress_file:
        progress_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _progress_record(
    *,
    run_id: str | None,
    event: str,
    source_path: Path,
    output_path: Path,
    samples: int,
    status: str,
    last_key: str = "",
) -> dict[str, Any]:
    return {
        "time": time.time(),
        "run_id": run_id,
        "event": event,
        "shard": str(source_path),
        "output": str(output_path),
        "samples": int(samples),
        "status": status,
        "last_key": last_key,
    }


def repair_shard(
    source_shard: str,
    output_dir: str,
    *,
    resume: bool,
    dry_run: bool,
    progress_out: str | None = None,
    progress_interval: int = 1000,
    progress_run_id: str | None = None,
    dirty_seed: str = "0",
    keep_legacy_rot6d_episode_fraction: float = 0.0,
    dirty_instruction_episode_fraction: float = 0.0,
    dirty_instruction_mode: str = "empty",
    generic_instruction: str = "do something useful",
    dirty_state_action_scale_episode_fraction: float = 0.0,
    dirty_state_action_scale_min: float = 0.9,
    dirty_state_action_scale_max: float = 1.1,
) -> dict[str, Any]:
    source_path = Path(source_shard)
    output_path = Path(output_dir) / source_path.name
    if output_path.exists() and resume:
        _write_progress(
            progress_out,
            _progress_record(
                run_id=progress_run_id,
                event="skip",
                source_path=source_path,
                output_path=output_path,
                samples=0,
                status="skipped_existing",
            ),
        )
        return {
            "shard": str(source_path),
            "output": str(output_path),
            "status": "skipped_existing",
            "samples": 0,
            "lowdim_repaired": 0,
        }
    if output_path.exists() and not resume and not dry_run:
        output_path.unlink()

    samples = 0
    lowdim_repaired = 0
    legacy_rot6d_samples = 0
    dirty_instruction_samples = 0
    dirty_state_action_scale_samples = 0
    legacy_rot6d_episodes: set[str] = set()
    dirty_instruction_episodes: set[str] = set()
    dirty_state_action_scale_episodes: set[str] = set()
    progress_interval = max(1, int(progress_interval))
    _write_progress(
        progress_out,
        _progress_record(
            run_id=progress_run_id,
            event="start",
            source_path=source_path,
            output_path=output_path,
            samples=samples,
            status="dry_run" if dry_run else "rewriting",
        ),
    )
    if dry_run:
        for sample in iter_shard_samples(str(source_path)):
            validate_sample_record(sample)
            meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            episode_id = episode_id_for_sample(sample["key"], meta)
            keep_legacy_rot6d = stable_episode_selected(
                dirty_seed,
                "legacy_rot6d",
                episode_id,
                keep_legacy_rot6d_episode_fraction,
            )
            dirty_instruction = stable_episode_selected(
                dirty_seed,
                "dirty_instruction",
                episode_id,
                dirty_instruction_episode_fraction,
            )
            dirty_state_action_scale = stable_episode_selected(
                dirty_seed,
                "dirty_state_action_scale",
                episode_id,
                dirty_state_action_scale_episode_fraction,
            )
            if keep_legacy_rot6d:
                legacy_rot6d_samples += 1
                legacy_rot6d_episodes.add(episode_id)
            else:
                _ = repair_lowdim(decode_npy(sample["lowdim_bytes"]))
                lowdim_repaired += 1
            if dirty_instruction:
                dirty_instruction_samples += 1
                dirty_instruction_episodes.add(episode_id)
            if dirty_state_action_scale:
                dirty_state_action_scale_samples += 1
                dirty_state_action_scale_episodes.add(episode_id)
            samples += 1
            if samples % progress_interval == 0:
                _write_progress(
                    progress_out,
                    _progress_record(
                        run_id=progress_run_id,
                        event="sample",
                        source_path=source_path,
                        output_path=output_path,
                        samples=samples,
                        status="dry_run",
                        last_key=sample["key"],
                    ),
                )
        _write_progress(
            progress_out,
            _progress_record(
                run_id=progress_run_id,
                event="finish",
                source_path=source_path,
                output_path=output_path,
                samples=samples,
                status="dry_run",
            ),
        )
        return {
            "shard": str(source_path),
            "output": str(output_path),
            "status": "dry_run",
            "samples": samples,
            "lowdim_repaired": lowdim_repaired,
            "legacy_rot6d_samples": legacy_rot6d_samples,
            "dirty_instruction_samples": dirty_instruction_samples,
            "dirty_state_action_scale_samples": dirty_state_action_scale_samples,
            "legacy_rot6d_episodes": len(legacy_rot6d_episodes),
            "dirty_instruction_episodes": len(dirty_instruction_episodes),
            "dirty_state_action_scale_episodes": len(dirty_state_action_scale_episodes),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    import tarfile

    with tarfile.open(tmp_path, "w") as tar_writer:
        for sample in iter_shard_samples(str(source_path)):
            validate_sample_record(sample)
            meta = json.loads(sample["meta_bytes"].decode("utf-8"))
            episode_id = episode_id_for_sample(sample["key"], meta)
            keep_legacy_rot6d = stable_episode_selected(
                dirty_seed,
                "legacy_rot6d",
                episode_id,
                keep_legacy_rot6d_episode_fraction,
            )
            dirty_instruction = stable_episode_selected(
                dirty_seed,
                "dirty_instruction",
                episode_id,
                dirty_instruction_episode_fraction,
            )
            dirty_state_action_scale = stable_episode_selected(
                dirty_seed,
                "dirty_state_action_scale",
                episode_id,
                dirty_state_action_scale_episode_fraction,
            )
            if keep_legacy_rot6d:
                repaired_lowdim = decode_npy(sample["lowdim_bytes"]).astype(np.float32, copy=False)
                legacy_rot6d_samples += 1
                legacy_rot6d_episodes.add(episode_id)
            else:
                repaired_lowdim = repair_lowdim(decode_npy(sample["lowdim_bytes"]))
                lowdim_repaired += 1
            if dirty_instruction:
                dirty_instruction_samples += 1
                dirty_instruction_episodes.add(episode_id)
            if dirty_state_action_scale:
                repaired_lowdim = dirty_state_action_scale_lowdim(
                    repaired_lowdim,
                    seed=dirty_seed,
                    sample_key=sample["key"],
                    scale_min=dirty_state_action_scale_min,
                    scale_max=dirty_state_action_scale_max,
                )
                dirty_state_action_scale_samples += 1
                dirty_state_action_scale_episodes.add(episode_id)
            write_sample_to_tar(
                tar_writer,
                sample["key"],
                sample["image_bytes"],
                encode_npy(repaired_lowdim),
                update_meta_for_repair(
                    meta,
                    rot6d_repaired=not keep_legacy_rot6d,
                    dirty_instruction_mode=dirty_instruction_mode if dirty_instruction else None,
                    generic_instruction=generic_instruction,
                    dirty_state_action_scale=dirty_state_action_scale,
                    dirty_state_action_scale_min=dirty_state_action_scale_min,
                    dirty_state_action_scale_max=dirty_state_action_scale_max,
                ),
                mano_bytes=sample.get("mano_bytes"),
                depth_bytes=sample.get("depth_bytes"),
            )
            samples += 1
            if samples % progress_interval == 0:
                _write_progress(
                    progress_out,
                    _progress_record(
                        run_id=progress_run_id,
                        event="sample",
                        source_path=source_path,
                        output_path=output_path,
                        samples=samples,
                        status="rewriting",
                        last_key=sample["key"],
                    ),
                )
    tmp_path.replace(output_path)
    _write_progress(
        progress_out,
        _progress_record(
            run_id=progress_run_id,
            event="finish",
            source_path=source_path,
            output_path=output_path,
            samples=samples,
            status="rewritten",
        ),
    )
    return {
        "shard": str(source_path),
        "output": str(output_path),
        "status": "rewritten",
        "samples": samples,
        "lowdim_repaired": lowdim_repaired,
        "legacy_rot6d_samples": legacy_rot6d_samples,
        "dirty_instruction_samples": dirty_instruction_samples,
        "dirty_state_action_scale_samples": dirty_state_action_scale_samples,
        "legacy_rot6d_episodes": len(legacy_rot6d_episodes),
        "dirty_instruction_episodes": len(dirty_instruction_episodes),
        "dirty_state_action_scale_episodes": len(dirty_state_action_scale_episodes),
    }


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for item in items:
        status = str(item.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "shards_total": len(items),
        "status_counts": status_counts,
        "samples": int(sum(int(item.get("samples", 0)) for item in items)),
        "lowdim_repaired": int(sum(int(item.get("lowdim_repaired", 0)) for item in items)),
        "legacy_rot6d_samples": int(sum(int(item.get("legacy_rot6d_samples", 0)) for item in items)),
        "dirty_instruction_samples": int(sum(int(item.get("dirty_instruction_samples", 0)) for item in items)),
        "dirty_state_action_scale_samples": int(sum(int(item.get("dirty_state_action_scale_samples", 0)) for item in items)),
        "legacy_rot6d_episodes_shard_local": int(sum(int(item.get("legacy_rot6d_episodes", 0)) for item in items)),
        "dirty_instruction_episodes_shard_local": int(sum(int(item.get("dirty_instruction_episodes", 0)) for item in items)),
        "dirty_state_action_scale_episodes_shard_local": int(sum(int(item.get("dirty_state_action_scale_episodes", 0)) for item in items)),
    }


def main() -> None:
    args = build_parser().parse_args()
    source_dir = Path(args.source_shard_dir)
    output_dir = Path(args.output_dir)
    validate_io_dirs(source_dir, output_dir)
    shard_paths = list(iter_shard_paths(str(source_dir)))
    selected = shard_paths[int(args.shard_start): args.shard_end]
    if not selected:
        raise SystemExit("No shards selected")
    workers = max(1, int(args.workers))
    validate_fraction("--keep-legacy-rot6d-episode-fraction", float(args.keep_legacy_rot6d_episode_fraction))
    validate_fraction("--dirty-instruction-episode-fraction", float(args.dirty_instruction_episode_fraction))
    validate_fraction(
        "--dirty-state-action-scale-episode-fraction",
        float(args.dirty_state_action_scale_episode_fraction),
    )
    validate_scale_range(float(args.dirty_state_action_scale_min), float(args.dirty_state_action_scale_max))
    executor_cls = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
    progress_run_id = uuid.uuid4().hex if args.progress_out else None
    if args.progress_out:
        progress_path = Path(args.progress_out)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("", encoding="utf-8")

    print(
        f"Repairing legacy rot6d WDS: shards={len(selected)} workers={workers} "
        f"executor={args.executor} dry_run={bool(args.dry_run)} output={output_dir}",
        flush=True,
    )

    items: list[dict[str, Any]] = []
    if workers == 1:
        iterator = (
            repair_shard(
                path,
                str(output_dir),
                resume=bool(args.resume),
                dry_run=bool(args.dry_run),
                progress_out=args.progress_out,
                progress_interval=int(args.progress_interval),
                progress_run_id=progress_run_id,
                dirty_seed=str(args.dirty_seed),
                keep_legacy_rot6d_episode_fraction=float(args.keep_legacy_rot6d_episode_fraction),
                dirty_instruction_episode_fraction=float(args.dirty_instruction_episode_fraction),
                dirty_instruction_mode=str(args.dirty_instruction_mode),
                generic_instruction=str(args.generic_instruction),
                dirty_state_action_scale_episode_fraction=float(args.dirty_state_action_scale_episode_fraction),
                dirty_state_action_scale_min=float(args.dirty_state_action_scale_min),
                dirty_state_action_scale_max=float(args.dirty_state_action_scale_max),
            )
            for path in selected
        )
        items = list(tqdm(iterator, total=len(selected), desc="Repair rot6d", unit="shard"))
    else:
        with executor_cls(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    repair_shard,
                    path,
                    str(output_dir),
                    resume=bool(args.resume),
                    dry_run=bool(args.dry_run),
                    progress_out=args.progress_out,
                    progress_interval=int(args.progress_interval),
                    progress_run_id=progress_run_id,
                    dirty_seed=str(args.dirty_seed),
                    keep_legacy_rot6d_episode_fraction=float(args.keep_legacy_rot6d_episode_fraction),
                    dirty_instruction_episode_fraction=float(args.dirty_instruction_episode_fraction),
                    dirty_instruction_mode=str(args.dirty_instruction_mode),
                    generic_instruction=str(args.generic_instruction),
                    dirty_state_action_scale_episode_fraction=float(args.dirty_state_action_scale_episode_fraction),
                    dirty_state_action_scale_min=float(args.dirty_state_action_scale_min),
                    dirty_state_action_scale_max=float(args.dirty_state_action_scale_max),
                )
                for path in selected
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Repair rot6d", unit="shard"):
                items.append(future.result())
        items.sort(key=lambda item: item["shard"])

    report = {"summary": summarize(items), "items": items}
    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
