#!/usr/bin/env python3
"""Fill reserved source slots in an existing mixed WDS after a delay.

This version does not require the reserved source to be rebuilt into WDS first.
Instead it:

1. waits for an optional delay
2. loads the real clip manifest for the reserved source
3. keeps only episodes whose current seq_folder outputs are complete enough to
   export and whose exported frame_count exactly matches the reserved manifest
4. rewrites only the affected mixed output shards

Rewritten shards preserve already-written non-reserved episodes by reading the
current mixed shard as the source of truth for those samples.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import shutil
import tarfile
import threading
import time
from pathlib import Path

import io
import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, **kwargs):
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.clip_manifest import load_clip_manifest, remap_descriptor_seq_folders  # noqa: E402
from lib.pipeline.depth_artifacts import DEPTH_EXPORT_ENCODING, DEPTH_EXPORT_SCHEMA, encode_depth_npy  # noqa: E402
from lib.pipeline.exporters.mano_codec import mano_meta_fields  # noqa: E402
from lib.pipeline.exporters.manifest_build.episodes import (  # noqa: E402
    load_descriptor_episode_features,
    load_manifest_record_prediction,
    prepare_manifest_record_for_build,
)
from lib.pipeline.exporters.webdataset_rewriter import iter_shard_samples, write_sample_to_tar  # noqa: E402
from lib.pipeline.frame_sources import build_frame_bytes_reader  # noqa: E402


_THREAD_LOCAL = threading.local()


def _stable_hash_hex(*parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _episode_uid(source_name: str, original_clip_id: str) -> str:
    return _stable_hash_hex("episode", source_name, original_clip_id)


def _coalesce_indices(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ordered = sorted(set(int(value) for value in indices))
    ranges: list[tuple[int, int]] = []
    start = ordered[0]
    prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev + 1))
        start = value
        prev = value
    ranges.append((start, prev + 1))
    return ranges


def _encode_npy(array, *, dtype=np.float32) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=dtype), allow_pickle=False)
    return buffer.getvalue()


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _write_text_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(str(line))
            handle.write("\n")


def _rewrite_resume_dir(state_dir: Path) -> Path:
    return state_dir / "_rewrite_resume"


def _rewrite_resume_marker_path(state_dir: Path, shard_name: str) -> Path:
    return _rewrite_resume_dir(state_dir) / f"{shard_name}.done.json"


def _load_rewrite_resume_marker(state_dir: Path, shard_name: str) -> dict | None:
    marker_path = _rewrite_resume_marker_path(state_dir, shard_name)
    if not marker_path.is_file():
        return None
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("status")) != "written":
        return None
    if str(payload.get("output_shard_name")) != shard_name:
        return None
    return payload


def _write_rewrite_resume_marker(state_dir: Path, shard_result: dict) -> None:
    shard_name = str(shard_result["output_shard_name"])
    marker_path = _rewrite_resume_marker_path(state_dir, shard_name)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_payload = dict(shard_result)
    marker_payload["resume_marker_version"] = 1
    marker_path.write_text(json.dumps(marker_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sample_episode_key(sample_key: str) -> str:
    if "_f" not in sample_key:
        raise ValueError(f"Sample key does not contain frame suffix: {sample_key}")
    return sample_key.rsplit("_f", 1)[0]


def _build_final_meta(
    episode: dict,
    *,
    dataset_name: str,
    split: str,
    episode_id: str,
    episode_index: int,
    presence: int,
    export_depth: bool,
) -> bytes:
    descriptor = episode["descriptor"]
    descriptor_extra = getattr(descriptor, "extra", None) or {}
    instruction = list(episode.get("instruction", []) or [])

    payload: dict[str, object] = {
        "dataset_name": str(dataset_name),
        "clip_id": str(episode_id),
        "episode_id": str(episode_id),
        "episode_index": int(episode_index),
        "split": str(split),
        "instruction": instruction,
        "instruction_num": int(episode.get("instruction_num", len(instruction))),
        "presence": int(presence),
        "lowdim_schema": descriptor_extra.get("lowdim_schema") or "hawor_wrist_world_v2",
        "wrist_translation_semantics": "mano_joint_0_world",
        "camera_extrinsic_convention": "w2c",
    }

    language = episode.get("language")
    if language is not None:
        payload["language"] = language

    native_feature_source = descriptor_extra.get("native_feature_source")
    if native_feature_source:
        payload["native_feature_source"] = native_feature_source

    mano_fields = mano_meta_fields()
    if descriptor_extra.get("mano_schema"):
        mano_fields["mano_schema"] = descriptor_extra["mano_schema"]
    payload.update(mano_fields)

    if export_depth:
        payload["depth_schema"] = DEPTH_EXPORT_SCHEMA
        payload["depth_encoding"] = DEPTH_EXPORT_ENCODING

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _load_existing_shard_episode_samples(path: Path) -> dict[str, list[dict]]:
    if not path.is_file():
        return {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in iter_shard_samples(str(path)):
        grouped[_sample_episode_key(str(sample["key"]))].append(sample)
    return grouped


def _build_tmp_shard_path(base_dir: Path, shard_name: str) -> Path:
    return base_dir / f".{shard_name}.tmp.{os.getpid()}.{threading.get_ident()}"


def _publish_completed_shard(
    *,
    built_tmp_path: Path,
    output_path: Path,
) -> None:
    publish_tmp_path = output_path.parent / f".{output_path.name}.publish.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        if built_tmp_path.parent == output_path.parent:
            os.replace(built_tmp_path, publish_tmp_path)
        else:
            shutil.copyfile(built_tmp_path, publish_tmp_path)
        os.replace(publish_tmp_path, output_path)
    finally:
        if publish_tmp_path.exists():
            publish_tmp_path.unlink()
        if built_tmp_path.exists():
            built_tmp_path.unlink()


def _get_thread_runtime(device_str: str, mano_dir: str | None):
    import torch
    from lib.pipeline.exporters.webdataset_features import build_mano_models

    runtime = getattr(_THREAD_LOCAL, "runtime", None)
    if runtime is not None:
        cached_device = runtime["device"]
        cached_mano_dir = runtime["mano_dir"]
        if cached_device == device_str and cached_mano_dir == mano_dir:
            return runtime

    device = torch.device(device_str)
    mano_right, mano_left = build_mano_models(device, mano_dir=mano_dir)
    mano_right.eval()
    mano_left.eval()
    runtime = {
        "device": str(device),
        "mano_dir": mano_dir,
        "torch_device": device,
        "mano_right": mano_right,
        "mano_left": mano_left,
        "shard_fd_cache": {},
        "shard_tar_cache": {},
    }
    _THREAD_LOCAL.runtime = runtime
    return runtime


def _build_reserved_episode_samples(
    episode: dict,
    *,
    output_episode_id: str,
    output_episode_index: int,
    dataset_name: str,
    split: str,
    feature_cache_dir: str | None,
    mano_dir: str | None,
    device_str: str,
    export_depth: bool,
) -> list[dict]:
    runtime = _get_thread_runtime(device_str, mano_dir)
    episode_data = load_descriptor_episode_features(
        episode,
        runtime["mano_right"],
        runtime["mano_left"],
        runtime["torch_device"],
        feature_cache_dir,
        mano_dir,
        source_fps=float(episode.get("source_fps", 5.0)),
        target_fps=float(episode.get("target_fps", 30.0)),
        interpolate_labels=bool(episode.get("interpolate_labels", True)),
        export_depth=bool(export_depth),
    )
    if episode_data is None:
        raise RuntimeError(f"Failed to load descriptor episode features for {episode['clip_id']}")

    frame_count = int(episode.get("num_valid_frames", 0))
    if frame_count <= 0:
        frame_count = int(episode_data["frame_count"])
    if int(episode_data["frame_count"]) < frame_count:
        raise RuntimeError(
            f"Episode feature frame_count too small for {episode['clip_id']}: "
            f"episode_data={int(episode_data['frame_count'])} expected={frame_count}"
        )

    read_frame_bytes = build_frame_bytes_reader(
        episode["descriptor"],
        shard_fd_cache=runtime["shard_fd_cache"],
        shard_tar_cache=runtime["shard_tar_cache"],
    )
    lowdim_all = episode_data["lowdim_all"]
    mano_all = episode_data["mano_all"]
    presence_per_frame = episode_data["presence_per_frame"]
    depth_all = episode_data.get("depth_all")

    output_samples: list[dict] = []
    for frame_idx in range(frame_count):
        depth_bytes = None
        if export_depth:
            if depth_all is None:
                raise RuntimeError(f"Depth export requested but depth data missing for {episode['clip_id']}")
            depth_bytes = encode_depth_npy(depth_all[frame_idx])
        output_samples.append(
            {
                "key": f"{output_episode_id}_f{frame_idx:06d}",
                "image_bytes": read_frame_bytes(frame_idx),
                "lowdim_bytes": _encode_npy(lowdim_all[frame_idx]),
                "mano_bytes": _encode_npy(mano_all[frame_idx]),
                "depth_bytes": depth_bytes,
                "meta_bytes": _build_final_meta(
                    episode,
                    dataset_name=dataset_name,
                    split=split,
                    episode_id=output_episode_id,
                    episode_index=output_episode_index,
                    presence=int(presence_per_frame[frame_idx]),
                    export_depth=bool(export_depth),
                ),
            }
        )
    return output_samples


def _inspect_reserved_record(
    record,
    *,
    source_name: str,
    expected_frame_count_by_uid: dict[str, int],
    annotation_root: str | None,
    annotation_suffix: str,
    require_annotation: bool,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
) -> tuple[dict | None, dict]:
    episode_uid = _episode_uid(source_name, record.clip_id)
    expected_frame_count = expected_frame_count_by_uid.get(episode_uid)
    base = {
        "episode_uid": str(episode_uid),
        "source_name": str(source_name),
        "original_clip_id": str(record.clip_id),
        "seq_folder": str(Path(record.descriptor.seq_folder).resolve()),
        "expected_frame_count": None if expected_frame_count is None else int(expected_frame_count),
        "actual_frame_count": None,
        "status": None,
    }

    if expected_frame_count is None:
        base["status"] = "not_reserved"
        return None, base

    prediction, prediction_status = load_manifest_record_prediction(record)
    if prediction is None and prediction_status not in (None, "native_features"):
        base["status"] = str(prediction_status)
        return None, base

    episode, error_code = prepare_manifest_record_for_build(
        record,
        require_annotation=bool(require_annotation),
        annotation_root=annotation_root,
        annotation_suffix=annotation_suffix,
        source_fps=float(source_fps),
        target_fps=float(target_fps),
        interpolate_labels=bool(interpolate_labels),
        prediction=None if prediction_status == "native_features" else prediction,
    )
    if episode is None:
        base["status"] = str(error_code)
        return None, base

    actual_frame_count = int(episode["num_valid_frames"])
    base["actual_frame_count"] = actual_frame_count
    if actual_frame_count != int(expected_frame_count):
        base["status"] = "frame_count_mismatch"
        return None, base

    base["status"] = "completed"
    return episode, base


def _rewrite_single_output_shard(
    *,
    shard_idx: int,
    shard_entries: list[dict],
    output_dir: Path,
    local_tmp_dir: Path | None,
    source_name: str,
    completed_episode_lookup: dict[str, dict],
    dataset_name: str,
    split: str,
    feature_cache_dir: str | None,
    mano_dir: str | None,
    device_str: str,
    export_depth: bool,
) -> dict:
    shard_name = f"shard-{int(shard_idx):06d}.tar"
    output_path = output_dir / shard_name
    tmp_base_dir = local_tmp_dir if local_tmp_dir is not None else output_dir
    tmp_base_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = _build_tmp_shard_path(tmp_base_dir, shard_name)
    has_nonreserved_entries = any(str(entry["source_name"]) != source_name for entry in shard_entries)
    if has_nonreserved_entries and not output_path.is_file():
        return {
            "output_shard_idx": int(shard_idx),
            "output_shard_name": shard_name,
            "status": "pending_output_missing",
            "planned_episode_count": len(shard_entries),
            "written_episode_count": 0,
            "written_frame_count": 0,
            "missing_reserved_episode_count": 0,
        }
    existing_by_episode_id = _load_existing_shard_episode_samples(output_path)

    missing_existing_nonreserved: list[str] = []
    missing_reserved_still_unavailable: list[str] = []
    written_episode_count = 0
    written_frame_count = 0

    try:
        with tarfile.open(tmp_path, "w") as tar_writer:
            for entry in shard_entries:
                planned_episode_id = str(entry["episode_id"])
                planned_source_name = str(entry["source_name"])
                planned_episode_uid = str(entry["episode_uid"])

                output_samples: list[dict] | None = None
                if planned_source_name == source_name:
                    episode = completed_episode_lookup.get(planned_episode_uid)
                    if episode is None:
                        existing_samples = existing_by_episode_id.get(planned_episode_id)
                        if existing_samples:
                            output_samples = existing_samples
                        else:
                            missing_reserved_still_unavailable.append(planned_episode_id)
                            continue
                    else:
                        output_samples = _build_reserved_episode_samples(
                            episode,
                            output_episode_id=planned_episode_id,
                            output_episode_index=int(entry["output_episode_index"]),
                            dataset_name=dataset_name,
                            split=split,
                            feature_cache_dir=feature_cache_dir,
                            mano_dir=mano_dir,
                            device_str=device_str,
                            export_depth=bool(export_depth),
                        )
                else:
                    output_samples = existing_by_episode_id.get(planned_episode_id)
                    if not output_samples:
                        missing_existing_nonreserved.append(planned_episode_id)
                        continue

                for sample in output_samples:
                    write_sample_to_tar(
                        tar_writer,
                        sample["key"],
                        sample["image_bytes"],
                        sample["lowdim_bytes"],
                        sample["meta_bytes"],
                        mano_bytes=sample.get("mano_bytes"),
                        depth_bytes=sample.get("depth_bytes"),
                    )
                written_episode_count += 1
                written_frame_count += len(output_samples)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    if missing_existing_nonreserved:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Output shard {shard_name} is missing already-built non-reserved episode(s): "
            f"{missing_existing_nonreserved[:8]}"
        )

    if written_episode_count <= 0:
        if tmp_path.exists():
            tmp_path.unlink()
        return {
            "output_shard_idx": int(shard_idx),
            "output_shard_name": shard_name,
            "status": "pending_missing_all",
            "planned_episode_count": len(shard_entries),
            "written_episode_count": 0,
            "written_frame_count": 0,
            "missing_reserved_episode_count": int(len(missing_reserved_still_unavailable)),
        }

    _publish_completed_shard(
        built_tmp_path=tmp_path,
        output_path=output_path,
    )
    return {
        "output_shard_idx": int(shard_idx),
        "output_shard_name": shard_name,
        "status": "written",
        "planned_episode_count": len(shard_entries),
        "written_episode_count": int(written_episode_count),
        "written_frame_count": int(written_frame_count),
        "missing_reserved_episode_count": int(len(missing_reserved_still_unavailable)),
    }


def _materialize_finalized_shard(
    *,
    source_path: Path,
    dest_path: Path,
    mode: str,
) -> str:
    if dest_path.exists():
        try:
            if dest_path.samefile(source_path):
                return "existing"
        except FileNotFoundError:
            pass
        if dest_path.is_dir():
            raise RuntimeError(f"Destination path is a directory, expected file: {dest_path}")
        dest_path.unlink()

    if mode == "hardlink":
        os.link(source_path, dest_path)
    elif mode == "symlink":
        os.symlink(source_path, dest_path)
    elif mode == "copy":
        shutil.copy2(source_path, dest_path)
    else:
        raise ValueError(f"Unsupported finalized link mode: {mode}")
    return "created"


def _sync_materialized_finalized_shard(
    *,
    shard_idx: int,
    output_dir: Path,
    finalized_shard_dir: Path | None,
    finalized_link_mode: str,
    finalized_shards_set: set[int],
    materialized_records: list[dict],
) -> None:
    if finalized_shard_dir is None:
        return
    if int(shard_idx) not in finalized_shards_set:
        return
    shard_name = f"shard-{int(shard_idx):06d}.tar"
    source_path = output_dir / shard_name
    if not source_path.is_file():
        return
    dest_path = finalized_shard_dir / shard_name
    action = _materialize_finalized_shard(
        source_path=source_path,
        dest_path=dest_path,
        mode=finalized_link_mode,
    )
    materialized_records.append(
        {
            "output_shard_idx": int(shard_idx),
            "output_shard_name": shard_name,
            "status": action,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "After a delay, inspect seq_folder outputs for a reserved source and fill the "
            "corresponding positions in an already-written mixed WDS without requiring the "
            "reserved source to be explicitly built into WDS first."
        )
    )
    parser.add_argument("--delay-hours", type=float, default=7.0, help="Delay before starting the fill job")
    parser.add_argument("--source-name", default="buildai0324", help="Reserved source name in the frozen mix plan")
    parser.add_argument("--source-clip-manifest", required=True, help="Clip manifest JSONL for the reserved source")
    parser.add_argument(
        "--seq-folder-root",
        default=None,
        help="Optional root used to remap descriptor.seq_folder before scanning actual completion",
    )
    parser.add_argument("--reserved-episode-manifest", required=True, help="Reserved episode manifest JSONL used during planning")
    parser.add_argument("--mix-plan", required=True, help="Frozen mix plan JSONL")
    parser.add_argument("--output-dir", required=True, help="Mixed WDS output directory to patch")
    parser.add_argument("--state-dir", required=True, help="Working directory for generated manifests/reports")
    parser.add_argument("--shard_start", type=int, default=0, help="Optional inclusive output shard index lower bound")
    parser.add_argument("--shard_end", type=int, default=None, help="Optional exclusive output shard index upper bound")
    parser.add_argument(
        "--finalized-shard-dir",
        default=None,
        help=(
            "Optional directory that will be populated with finalized shard tar files. "
            "This is intended for incremental sanity-check and transfer."
        ),
    )
    parser.add_argument(
        "--finalized-link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How finalized shard tar files are materialized into --finalized-shard-dir",
    )
    parser.add_argument("--dataset-name", default="mixed_vla", help="dataset_name written into final meta.json")
    parser.add_argument("--split", default="train", help="split written into final meta.json")
    parser.add_argument("--scan-workers", type=int, default=8, help="Thread workers for reserved-source completion scan")
    parser.add_argument("--write-workers", type=int, default=1, help="Parallel output-shard writers")
    parser.add_argument("--annotation-root", default=None, help="Optional annotation sidecar directory")
    parser.add_argument(
        "--annotation-suffix",
        default=".annotation.json",
        help="Annotation sidecar suffix when --annotation-root is set",
    )
    parser.add_argument("--require-annotation", action="store_true", help="Drop reserved clips with missing/invalid annotations")
    parser.add_argument("--source-fps", type=float, default=5.0, help="FPS of seq_folder stage outputs")
    parser.add_argument("--target-fps", type=float, default=30.0, help="FPS of RGB frames referenced by the descriptor manifest")
    parser.add_argument(
        "--interpolate-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interpolate source labels onto descriptor frames instead of truncating",
    )
    parser.add_argument("--mano-device", default="cpu", help="Device for MANO feature generation during patching")
    parser.add_argument("--mano-dir", default=None, help="Optional MANO model directory")
    parser.add_argument(
        "--feature-cache-dir",
        default=None,
        help="Optional feature cache directory. Defaults to <state-dir>/_episode_feature_cache",
    )
    parser.add_argument(
        "--disable-feature-cache",
        action="store_true",
        help="Disable per-episode feature cache for one-shot fill runs",
    )
    parser.add_argument(
        "--local-tmp-dir",
        default=None,
        help=(
            "Optional local filesystem directory used for shard construction before publishing "
            "the completed tar into --output-dir."
        ),
    )
    parser.add_argument("--export-depth", action="store_true", help="Also export depth from seq_folder artifacts when available")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; do not rewrite any mixed shards")
    parser.add_argument("--report-out", default=None, help="Optional final report path")
    return parser


def main() -> None:
    import torch

    args = build_parser().parse_args()
    if args.delay_hours < 0:
        raise ValueError("--delay-hours must be >= 0")
    if args.scan_workers < 1:
        raise ValueError("--scan-workers must be >= 1")
    if args.write_workers < 1:
        raise ValueError("--write-workers must be >= 1")
    if int(args.shard_start) < 0:
        raise ValueError("--shard_start must be >= 0")
    if args.shard_end is not None and int(args.shard_end) < int(args.shard_start):
        raise ValueError("--shard_end must be >= --shard_start")

    resolved_device = str(torch.device(args.mano_device if torch.cuda.is_available() else "cpu"))
    resolved_write_workers = int(args.write_workers)

    state_dir = Path(args.state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir).resolve()
    finalized_shard_dir = (
        Path(args.finalized_shard_dir).resolve() if args.finalized_shard_dir else None
    )
    local_tmp_dir = (
        Path(args.local_tmp_dir).resolve() if args.local_tmp_dir else None
    )
    feature_cache_dir = None
    if not bool(args.disable_feature_cache):
        feature_cache_dir = (
            str(Path(args.feature_cache_dir).resolve())
            if args.feature_cache_dir
            else str((state_dir / "_episode_feature_cache").resolve())
        )
        Path(feature_cache_dir).mkdir(parents=True, exist_ok=True)
    if local_tmp_dir is not None:
        local_tmp_dir.mkdir(parents=True, exist_ok=True)

    source_name = str(args.source_name)
    actual_scan_manifest_out = state_dir / f"{source_name}.actual_scan.jsonl"
    completed_manifest_out = state_dir / f"{source_name}.completed_only.jsonl"
    finalized_list_out = state_dir / f"{source_name}.finalized_shards.list"
    pending_list_out = state_dir / f"{source_name}.pending_shards.list"
    fill_report_out = Path(args.report_out).resolve() if args.report_out else (state_dir / f"{source_name}.fill.report.json")

    delay_seconds = float(args.delay_hours) * 3600.0
    if delay_seconds > 0:
        print(f"Sleeping for {delay_seconds:.0f} seconds before filling {source_name} ...", flush=True)
        time.sleep(delay_seconds)

    reserved_records = _read_jsonl(Path(args.reserved_episode_manifest).resolve())
    expected_frame_count_by_uid = {
        str(record["episode_uid"]): int(record["frame_count"])
        for record in reserved_records
        if str(record.get("source_name")) == source_name
    }
    reserved_clip_ids = {
        str(record["original_clip_id"])
        for record in reserved_records
        if str(record.get("source_name")) == source_name
    }

    clip_manifest_records = load_clip_manifest(str(Path(args.source_clip_manifest).resolve()))
    if args.seq_folder_root:
        remap_descriptor_seq_folders([record.descriptor for record in clip_manifest_records], args.seq_folder_root)
    clip_manifest_records = [record for record in clip_manifest_records if str(record.clip_id) in reserved_clip_ids]

    actual_scan_records: list[dict] = []
    completed_records: list[dict] = []
    completed_episode_lookup: dict[str, dict] = {}

    if int(args.scan_workers) <= 1:
        iterator = (
            _inspect_reserved_record(
                record,
                source_name=source_name,
                expected_frame_count_by_uid=expected_frame_count_by_uid,
                annotation_root=args.annotation_root,
                annotation_suffix=args.annotation_suffix,
                require_annotation=bool(args.require_annotation),
                source_fps=float(args.source_fps),
                target_fps=float(args.target_fps),
                interpolate_labels=bool(args.interpolate_labels),
            )
            for record in clip_manifest_records
        )
        iterator = tqdm(iterator, total=len(clip_manifest_records), desc=f"Scan {source_name}")
        for episode, info in iterator:
            actual_scan_records.append(info)
            if episode is None:
                continue
            completed_episode_lookup[info["episode_uid"]] = episode
            completed_records.append(
                {
                    "manifest_type": "descriptor_episode",
                    "source_name": source_name,
                    "original_clip_id": info["original_clip_id"],
                    "episode_uid": info["episode_uid"],
                    "frame_count": int(info["actual_frame_count"]),
                    "has_depth": bool(args.export_depth),
                    "source_shard_path": None,
                    "source_episode_key": info["original_clip_id"],
                    "reserved_only": False,
                    "seq_folder": info["seq_folder"],
                }
            )
    else:
        with ThreadPoolExecutor(max_workers=int(args.scan_workers)) as executor:
            future_to_clip = {
                executor.submit(
                    _inspect_reserved_record,
                    record,
                    source_name=source_name,
                    expected_frame_count_by_uid=expected_frame_count_by_uid,
                    annotation_root=args.annotation_root,
                    annotation_suffix=args.annotation_suffix,
                    require_annotation=bool(args.require_annotation),
                    source_fps=float(args.source_fps),
                    target_fps=float(args.target_fps),
                    interpolate_labels=bool(args.interpolate_labels),
                ): str(record.clip_id)
                for record in clip_manifest_records
            }
            progress = tqdm(total=len(future_to_clip), desc=f"Scan {source_name} x{int(args.scan_workers)}")
            try:
                for future in as_completed(future_to_clip):
                    episode, info = future.result()
                    actual_scan_records.append(info)
                    if episode is not None:
                        completed_episode_lookup[info["episode_uid"]] = episode
                        completed_records.append(
                            {
                                "manifest_type": "descriptor_episode",
                                "source_name": source_name,
                                "original_clip_id": info["original_clip_id"],
                                "episode_uid": info["episode_uid"],
                                "frame_count": int(info["actual_frame_count"]),
                                "has_depth": bool(args.export_depth),
                                "source_shard_path": None,
                                "source_episode_key": info["original_clip_id"],
                                "reserved_only": False,
                                "seq_folder": info["seq_folder"],
                            }
                        )
                    progress.update(1)
            finally:
                progress.close()

    actual_scan_records.sort(key=lambda item: (str(item["status"]), str(item["original_clip_id"])))
    completed_records.sort(key=lambda item: str(item["original_clip_id"]))
    _write_jsonl(actual_scan_manifest_out, actual_scan_records)
    _write_jsonl(completed_manifest_out, completed_records)

    completed_uids = set(completed_episode_lookup)
    plan_records = _read_jsonl(Path(args.mix_plan).resolve())
    target_shards: list[int] = []
    shard_entries_by_idx: dict[int, list[dict]] = defaultdict(list)
    reserved_uids_by_shard: dict[int, set[str]] = defaultdict(set)
    for record in plan_records:
        shard_idx = int(record["output_shard_idx"])
        if shard_idx < int(args.shard_start):
            continue
        if args.shard_end is not None and shard_idx >= int(args.shard_end):
            continue
        shard_entries_by_idx[shard_idx].append(record)
        if str(record.get("source_name")) == source_name:
            reserved_uids_by_shard[shard_idx].add(str(record["episode_uid"]))
            if str(record.get("episode_uid")) in completed_uids:
                target_shards.append(shard_idx)

    target_shards = sorted(set(target_shards))
    target_ranges = _coalesce_indices(target_shards)
    target_shards_set = set(int(value) for value in target_shards)
    finalized_shards_set = {
        int(shard_idx)
        for shard_idx, reserved_uids in shard_entries_by_idx.items()
        if all(uid in completed_uids for uid in reserved_uids_by_shard.get(int(shard_idx), set()))
    }

    rewrite_reports: list[dict] = []
    materialized_finalized: list[dict] = []
    skipped_existing_reports: list[dict] = []
    if finalized_shard_dir is not None and not args.dry_run:
        finalized_shard_dir.mkdir(parents=True, exist_ok=True)
        for shard_idx in sorted(finalized_shards_set):
            if int(shard_idx) in target_shards_set:
                continue
            _sync_materialized_finalized_shard(
                shard_idx=int(shard_idx),
                output_dir=output_dir,
                finalized_shard_dir=finalized_shard_dir,
                finalized_link_mode=str(args.finalized_link_mode),
                finalized_shards_set=finalized_shards_set,
                materialized_records=materialized_finalized,
            )
    pending_target_shards: list[int] = []
    for shard_idx in target_shards:
        shard_name = f"shard-{int(shard_idx):06d}.tar"
        if int(shard_idx) in finalized_shards_set:
            finalized_path = None if finalized_shard_dir is None else (finalized_shard_dir / shard_name)
            if finalized_path is not None and finalized_path.is_file():
                skipped_existing_reports.append(
                    {
                        "output_shard_idx": int(shard_idx),
                        "output_shard_name": shard_name,
                        "status": "skipped_existing_finalized",
                    }
                )
                continue
            marker_payload = _load_rewrite_resume_marker(state_dir, shard_name)
            if marker_payload is not None:
                skipped_existing_reports.append(
                    {
                        "output_shard_idx": int(shard_idx),
                        "output_shard_name": shard_name,
                        "status": "skipped_existing_marker",
                    }
                )
                continue
        pending_target_shards.append(int(shard_idx))
    if not args.dry_run and target_shards:
        output_dir.mkdir(parents=True, exist_ok=True)
        if resolved_write_workers <= 1:
            iterator = tqdm(pending_target_shards, desc="Rewrite mixed shards")
            for shard_idx in iterator:
                item = _rewrite_single_output_shard(
                    shard_idx=shard_idx,
                    shard_entries=shard_entries_by_idx[shard_idx],
                    output_dir=output_dir,
                    local_tmp_dir=local_tmp_dir,
                    source_name=source_name,
                    completed_episode_lookup=completed_episode_lookup,
                    dataset_name=str(args.dataset_name),
                    split=str(args.split),
                    feature_cache_dir=feature_cache_dir,
                    mano_dir=args.mano_dir,
                    device_str=resolved_device,
                    export_depth=bool(args.export_depth),
                )
                rewrite_reports.append(item)
                if str(item.get("status")) == "written":
                    _sync_materialized_finalized_shard(
                        shard_idx=shard_idx,
                        output_dir=output_dir,
                        finalized_shard_dir=finalized_shard_dir,
                        finalized_link_mode=str(args.finalized_link_mode),
                        finalized_shards_set=finalized_shards_set,
                        materialized_records=materialized_finalized,
                    )
                    if int(shard_idx) in finalized_shards_set:
                        _write_rewrite_resume_marker(state_dir, item)
        else:
            with ThreadPoolExecutor(max_workers=resolved_write_workers) as executor:
                future_to_shard = {
                    executor.submit(
                        _rewrite_single_output_shard,
                        shard_idx=shard_idx,
                        shard_entries=shard_entries_by_idx[shard_idx],
                        output_dir=output_dir,
                        local_tmp_dir=local_tmp_dir,
                        source_name=source_name,
                        completed_episode_lookup=completed_episode_lookup,
                        dataset_name=str(args.dataset_name),
                        split=str(args.split),
                        feature_cache_dir=feature_cache_dir,
                        mano_dir=args.mano_dir,
                        device_str=resolved_device,
                        export_depth=bool(args.export_depth),
                    ): shard_idx
                    for shard_idx in pending_target_shards
                }
                progress = tqdm(total=len(future_to_shard), desc=f"Rewrite mixed shards x{resolved_write_workers}")
                try:
                    for future in as_completed(future_to_shard):
                        item = future.result()
                        rewrite_reports.append(item)
                        if str(item.get("status")) == "written":
                            _sync_materialized_finalized_shard(
                                shard_idx=int(item["output_shard_idx"]),
                                output_dir=output_dir,
                                finalized_shard_dir=finalized_shard_dir,
                                finalized_link_mode=str(args.finalized_link_mode),
                                finalized_shards_set=finalized_shards_set,
                                materialized_records=materialized_finalized,
                            )
                            if int(item["output_shard_idx"]) in finalized_shards_set:
                                _write_rewrite_resume_marker(state_dir, item)
                        progress.update(1)
                finally:
                    progress.close()

    rewrite_reports.extend(skipped_existing_reports)
    rewrite_reports.sort(key=lambda item: int(item["output_shard_idx"]))
    rewrite_report_by_shard = {
        int(item["output_shard_idx"]): item
        for item in rewrite_reports
    }

    finalized_shards: list[int] = []
    pending_shards: list[int] = []
    for shard_idx, shard_entries in sorted(shard_entries_by_idx.items()):
        output_path = output_dir / f"shard-{int(shard_idx):06d}.tar"
        if not output_path.is_file():
            pending_shards.append(int(shard_idx))
            continue
        reserved_uids = reserved_uids_by_shard.get(int(shard_idx), set())
        if all(uid in completed_uids for uid in reserved_uids):
            finalized_shards.append(int(shard_idx))
            continue
        pending_shards.append(int(shard_idx))

    finalized_ranges = _coalesce_indices(finalized_shards)
    pending_ranges = _coalesce_indices(pending_shards)
    finalized_shard_names = [f"shard-{int(shard_idx):06d}.tar" for shard_idx in finalized_shards]
    pending_shard_names = [f"shard-{int(shard_idx):06d}.tar" for shard_idx in pending_shards]
    _write_text_lines(finalized_list_out, finalized_shard_names)
    _write_text_lines(pending_list_out, pending_shard_names)

    if finalized_shard_dir is not None and not args.dry_run:
        already_materialized = {int(item["output_shard_idx"]) for item in materialized_finalized}
        for shard_idx in finalized_shards:
            if int(shard_idx) in already_materialized:
                continue
            _sync_materialized_finalized_shard(
                shard_idx=int(shard_idx),
                output_dir=output_dir,
                finalized_shard_dir=finalized_shard_dir,
                finalized_link_mode=str(args.finalized_link_mode),
                finalized_shards_set=finalized_shards_set,
                materialized_records=materialized_finalized,
            )

    status_counts: dict[str, int] = defaultdict(int)
    for item in actual_scan_records:
        status_counts[str(item["status"])] += 1

    report = {
        "source_name": source_name,
        "delay_hours": float(args.delay_hours),
        "source_clip_manifest": str(Path(args.source_clip_manifest).resolve()),
        "seq_folder_root": None if args.seq_folder_root is None else str(Path(args.seq_folder_root).resolve()),
        "reserved_episode_manifest": str(Path(args.reserved_episode_manifest).resolve()),
        "mix_plan": str(Path(args.mix_plan).resolve()),
        "output_dir": str(output_dir),
        "state_dir": str(state_dir),
        "shard_start": int(args.shard_start),
        "shard_end": None if args.shard_end is None else int(args.shard_end),
        "finalized_shard_dir": None if finalized_shard_dir is None else str(finalized_shard_dir),
        "finalized_link_mode": str(args.finalized_link_mode),
        "local_tmp_dir": None if local_tmp_dir is None else str(local_tmp_dir),
        "feature_cache_dir": None if feature_cache_dir is None else str(Path(feature_cache_dir).resolve()),
        "disable_feature_cache": bool(args.disable_feature_cache),
        "mano_device": resolved_device,
        "write_workers_requested": int(args.write_workers),
        "write_workers_effective": int(resolved_write_workers),
        "export_depth": bool(args.export_depth),
        "actual_scan_manifest": str(actual_scan_manifest_out),
        "completed_manifest": str(completed_manifest_out),
        "finalized_shards_list": str(finalized_list_out),
        "pending_shards_list": str(pending_list_out),
        "reserved_episode_count": len(reserved_clip_ids),
        "actual_manifest_clip_count": len(clip_manifest_records),
        "completed_episode_count": len(completed_records),
        "scan_status_counts": dict(sorted(status_counts.items())),
        "target_shard_count": len(target_shards),
        "pending_target_shard_count": len(pending_target_shards),
        "target_shard_ranges": [
            {"shard_start": int(shard_start), "shard_end": int(shard_end)}
            for shard_start, shard_end in target_ranges
        ],
        "skipped_existing_rewrite_count": len(skipped_existing_reports),
        "skipped_existing_rewrites": skipped_existing_reports,
        "rewrite_report_count": len(rewrite_reports),
        "rewrite_reports": rewrite_reports,
        "rewrite_report_by_shard_count": len(rewrite_report_by_shard),
        "finalized_shard_count": len(finalized_shards),
        "finalized_shard_ranges": [
            {"shard_start": int(shard_start), "shard_end": int(shard_end)}
            for shard_start, shard_end in finalized_ranges
        ],
        "pending_shard_count": len(pending_shards),
        "pending_shard_ranges": [
            {"shard_start": int(shard_start), "shard_end": int(shard_end)}
            for shard_start, shard_end in pending_ranges
        ],
        "materialized_finalized_shard_count": len(materialized_finalized),
        "materialized_finalized_shards": materialized_finalized,
        "dry_run": bool(args.dry_run),
    }
    fill_report_out.parent.mkdir(parents=True, exist_ok=True)
    fill_report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
