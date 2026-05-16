#!/usr/bin/env python3
"""Mix multiple WebDataset sources into one anonymized shuffled dataset."""

from __future__ import annotations

import atexit
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import io
import json
import os
import pickle
import random
import shutil
import tarfile
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import heapq
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, **kwargs):
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.webdataset_rewriter import (  # noqa: E402
    iter_shard_paths,
    iter_shard_samples,
    split_sample_member_name,
    validate_sample_record,
    write_sample_to_tar,
)
from lib.pipeline.quality_metrics import (  # noqa: E402
    INTRINSIC_SLICE,
    decode_lowdim,
)


SHARD_SAMPLE_INDEX_FORMAT_VERSION = 1
SHARD_SAMPLE_INDEX_MEM_CACHE_MAX = 16
SHARD_SOURCE_FD_CACHE_MAX = 32
DEFAULT_SOURCE_INDEX_CACHE_ROOT = Path("/share_data/guantianrui/datasets/_tmp") / "mix_wds_source_index_v1"
_shard_sample_index_mem_cache: OrderedDict[str, dict] = OrderedDict()
_shard_sample_index_mem_cache_lock = threading.Lock()
_source_index_cache_root = DEFAULT_SOURCE_INDEX_CACHE_ROOT
_source_index_cache_root_lock = threading.Lock()
_source_shard_fd_cache: OrderedDict[str, int] = OrderedDict()
_source_shard_fd_cache_lock = threading.Lock()


def _close_source_shard_fd_cache() -> None:
    with _source_shard_fd_cache_lock:
        while _source_shard_fd_cache:
            _path, fd = _source_shard_fd_cache.popitem(last=False)
            try:
                os.close(fd)
            except OSError:
                pass


atexit.register(_close_source_shard_fd_cache)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mix existing WebDataset sources into one anonymized dataset. "
            "Supports a two-phase workflow: stage on worker machines, then finalize on one coordinator."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    stage_parser = subparsers.add_parser(
        "stage",
        help="Stage per-episode tar files and append stage records for later finalize.",
    )
    _add_source_args(stage_parser, required=True)
    _add_resize_args(stage_parser)
    stage_parser.add_argument(
        "--staging_dir",
        required=True,
        help="Root directory for staged per-episode tar files.",
    )
    stage_parser.add_argument(
        "--stage_manifest_out",
        required=True,
        help="Append-only JSONL manifest describing staged episodes from this run.",
    )
    stage_parser.add_argument(
        "--stage_tag",
        default="default",
        help="Unique tag for this worker/run. Used in staging paths and stage_uids to avoid collisions.",
    )
    stage_parser.add_argument("--shard_start", type=int, default=0, help="Inclusive shard index in sorted shard order")
    stage_parser.add_argument("--shard_end", type=int, default=None, help="Exclusive shard index in sorted shard order")
    stage_parser.add_argument("--max-episodes", type=int, default=None, help="Optional episode cap for smoke testing")
    stage_parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality used when re-encoding resized frames",
    )
    stage_parser.add_argument("--report-out", default=None, help="Optional JSON report path")

    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan source WDS shards into a lightweight per-episode manifest without staging episode tar files.",
    )
    _add_source_args(scan_parser, required=True)
    scan_parser.add_argument("--episode_manifest_out", required=True, help="Output JSONL manifest with one record per input episode.")
    scan_parser.add_argument("--shard_start", type=int, default=0, help="Inclusive shard index in sorted shard order")
    scan_parser.add_argument("--shard_end", type=int, default=None, help="Exclusive shard index in sorted shard order")
    scan_parser.add_argument("--max-episodes", type=int, default=None, help="Optional episode cap for smoke testing")
    scan_parser.add_argument("--report-out", default=None, help="Optional JSON report path")

    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Convert an existing clip manifest into an episode manifest without rescanning WDS tar members.",
    )
    manifest_parser.add_argument("--clip_manifest", required=True, help="Input clip manifest JSONL for an existing WDS dataset.")
    manifest_parser.add_argument(
        "--source-name",
        default=None,
        help="Optional source name override. Defaults to the clip manifest source_id.",
    )
    manifest_parser.add_argument(
        "--has-depth",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional has_depth override. Defaults to clip-manifest metadata hint if present, else false.",
    )
    manifest_parser.add_argument("--episode_manifest_out", required=True, help="Output JSONL manifest with one record per input episode.")
    manifest_parser.add_argument("--report-out", default=None, help="Optional JSON report path")

    reserve_parser = subparsers.add_parser(
        "reserve",
        help="Reserve output positions for a future dataset by emitting placeholder per-episode records from a clip manifest.",
    )
    reserve_parser.add_argument("--clip_manifest", required=True, help="Clip manifest JSONL used to reserve future episode positions.")
    reserve_parser.add_argument(
        "--source-name",
        required=True,
        help="Stable source name for the reserved dataset. Must match the eventual scan source name used later.",
    )
    reserve_parser.add_argument("--episode_manifest_out", required=True, help="Output JSONL manifest with one reserved record per episode.")
    reserve_parser.add_argument(
        "--drop-last-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reserve frame_count-1 to match action-conditioned WDS export semantics.",
    )
    reserve_parser.add_argument("--report-out", default=None, help="Optional JSON report path")

    plan_parser = subparsers.add_parser(
        "plan",
        help="Freeze a deterministic global mix plan from scanned and reserved episode manifests.",
    )
    plan_parser.add_argument(
        "--episode_manifest",
        action="append",
        required=True,
        help="Repeatable episode-manifest JSONL produced by scan/reserve.",
    )
    plan_parser.add_argument("--mix_plan_out", required=True, help="Output JSONL plan with one record per output episode copy.")
    plan_parser.add_argument(
        "--frames_per_shard",
        type=int,
        default=10000,
        help="Approximate frame budget per output shard in the frozen plan.",
    )
    plan_parser.add_argument("--repeat-min", type=int, default=1, help="Minimum repeat count per episode")
    plan_parser.add_argument("--repeat-max", type=int, default=3, help="Maximum repeat count per episode")
    plan_parser.add_argument(
        "--fixed-repeat-source",
        action="append",
        default=[],
        help="Optional override <source_name>=<count>. Episodes from this source will repeat exactly that many times.",
    )
    plan_parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic repeat-count sampling and ordering")
    plan_parser.add_argument("--report-out", default=None, help="Optional JSON report path")

    finalize_parser = subparsers.add_parser(
        "finalize",
        help="Read staged episode manifests, globally repeat/shuffle/anonymize, and write final mixed shards.",
    )
    finalize_parser.add_argument(
        "--stage_manifest",
        action="append",
        required=True,
        help="Repeatable path to a stage-manifest JSONL produced by the stage command.",
    )
    finalize_parser.add_argument(
        "--staging_dir",
        required=True,
        help="Root directory containing staged episode tar files referenced by --stage_manifest.",
    )
    _add_finalize_args(finalize_parser)

    all_parser = subparsers.add_parser(
        "all",
        help="Single-machine convenience mode: stage and finalize in one process.",
    )
    _add_source_args(all_parser, required=True)
    _add_resize_args(all_parser)
    all_parser.add_argument(
        "--staging_dir",
        required=True,
        help="Directory for staged per-episode tar files. This will be large; place it on a disk with enough free space.",
    )
    all_parser.add_argument("--max-episodes", type=int, default=None, help="Optional episode cap for smoke testing")
    all_parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality used when re-encoding resized frames",
    )
    _add_finalize_args(all_parser)

    write_parser = subparsers.add_parser(
        "write",
        help="Write final mixed shards for a selected output-shard range from a frozen mix plan.",
    )
    write_parser.add_argument("--mix_plan", required=True, help="Frozen mix plan JSONL produced by the plan command.")
    write_parser.add_argument(
        "--episode_manifest",
        action="append",
        required=True,
        help="Repeatable scanned episode-manifest JSONL containing currently available source episodes.",
    )
    write_parser.add_argument("--output_dir", required=True, help="Output directory for mixed shard tar files")
    write_parser.add_argument("--dataset-name", default="mixed_vla", help="Generic dataset_name to store in meta.json")
    write_parser.add_argument("--split", default="train", help="Generic split to store in meta.json")
    write_parser.add_argument("--shard_start", type=int, default=0, help="Inclusive output shard index")
    write_parser.add_argument("--shard_end", type=int, default=None, help="Exclusive output shard index")
    write_parser.add_argument("--workers", type=int, default=1, help="Number of output shards to write in parallel")
    write_parser.add_argument(
        "--index-cache-dir",
        default=str(DEFAULT_SOURCE_INDEX_CACHE_ROOT),
        help="Directory for cached source-shard sample indexes. Override if you want a different scratch location.",
    )
    write_parser.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        help="Optional source name(s) to skip even if present in the available episode manifests.",
    )
    write_parser.add_argument(
        "--allow-missing-episodes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow planned episodes with unavailable data (for reserved future sources). Incomplete shards are still written.",
    )
    write_parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip output shards that already exist. Disable this when rewriting shards after a reserved source becomes available.",
    )
    write_parser.add_argument("--report-out", default=None, help="Optional JSON report path")

    warm_index_parser = subparsers.add_parser(
        "warm-index",
        help="Prebuild cached source-shard sample indexes from one or more available episode manifests.",
    )
    warm_index_parser.add_argument(
        "--episode_manifest",
        action="append",
        required=True,
        help="Repeatable scanned episode-manifest JSONL containing available WDS episodes.",
    )
    warm_index_parser.add_argument("--workers", type=int, default=8, help="Number of source shards to index in parallel")
    warm_index_parser.add_argument("--shard_start", type=int, default=0, help="Inclusive source-shard index in sorted unique source-shard order")
    warm_index_parser.add_argument("--shard_end", type=int, default=None, help="Exclusive source-shard index in sorted unique source-shard order")
    warm_index_parser.add_argument(
        "--index-cache-dir",
        default=str(DEFAULT_SOURCE_INDEX_CACHE_ROOT),
        help="Directory for cached source-shard sample indexes. Override if you want a different scratch location.",
    )
    warm_index_parser.add_argument("--report-out", default=None, help="Optional JSON report path")
    return parser


def _add_source_args(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        "--source",
        action="append",
        required=required,
        help="Repeatable source spec: <name>=<wds_dir>",
    )
    parser.add_argument(
        "--resize-source",
        action="append",
        default=[],
        help="Source name(s) whose RGB frames should be resized to --resize-width/--resize-height with intrinsic scaling.",
    )


def _add_resize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resize-width", type=int, default=456, help="Target width for resized sources")
    parser.add_argument("--resize-height", type=int, default=256, help="Target height for resized sources")


def _add_finalize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output_dir", required=True, help="Output directory for mixed shard tar files")
    parser.add_argument(
        "--frames_per_shard",
        type=int,
        default=10000,
        help="Approximate frame budget per output shard",
    )
    parser.add_argument("--repeat-min", type=int, default=1, help="Minimum repeat count per episode")
    parser.add_argument("--repeat-max", type=int, default=3, help="Maximum repeat count per episode")
    parser.add_argument(
        "--fixed-repeat-source",
        action="append",
        default=[],
        help="Optional override <source_name>=<count>. Episodes from this source will repeat exactly that many times.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeat-count sampling and episode shuffle")
    parser.add_argument("--dataset-name", default="mixed_vla", help="Generic dataset_name to store in meta.json")
    parser.add_argument("--split", default="train", help="Generic split to store in meta.json")
    parser.add_argument(
        "--cleanup-staging",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete --staging_dir after output shards are written successfully",
    )
    parser.add_argument("--report-out", default=None, help="Optional JSON report path")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    shard_dir: Path
    resize: bool


class MixedShardWriter:
    def __init__(self, output_dir: Path, frames_per_shard: int):
        self.output_dir = output_dir
        self.frames_per_shard = frames_per_shard
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._shard_idx = 0
        self._current_tar = None
        self._current_tmp_path = None
        self._current_output_path = None
        self._current_frame_count = 0
        self._current_clip_count = 0
        self.output_shards: list[dict] = []

    def _open_shard(self) -> None:
        output_name = f"shard-{self._shard_idx:06d}.tar"
        self._current_output_path = self.output_dir / output_name
        self._current_tmp_path = self.output_dir / f"{output_name}.tmp"
        self._current_tar = tarfile.open(self._current_tmp_path, "w")
        self._current_frame_count = 0
        self._current_clip_count = 0

    def _close_shard(self) -> None:
        if self._current_tar is None:
            return
        self._current_tar.close()
        if self._current_frame_count <= 0:
            if self._current_tmp_path and self._current_tmp_path.exists():
                self._current_tmp_path.unlink()
        else:
            os.replace(self._current_tmp_path, self._current_output_path)
            self.output_shards.append(
                {
                    "name": self._current_output_path.name,
                    "frames": self._current_frame_count,
                    "clips": self._current_clip_count,
                }
            )
            self._shard_idx += 1

        self._current_tar = None
        self._current_tmp_path = None
        self._current_output_path = None
        self._current_frame_count = 0
        self._current_clip_count = 0

    def add_episode(self, samples: list[dict]) -> None:
        if not samples:
            return
        clip_frame_count = len(samples)
        if self._current_tar is None:
            self._open_shard()
        elif self._current_frame_count > 0 and self._current_frame_count + clip_frame_count > self.frames_per_shard:
            self._close_shard()
            self._open_shard()

        for sample in samples:
            write_sample_to_tar(
                self._current_tar,
                sample["key"],
                sample["image_bytes"],
                sample["lowdim_bytes"],
                sample["meta_bytes"],
                mano_bytes=sample.get("mano_bytes"),
                depth_bytes=sample.get("depth_bytes"),
            )
        self._current_frame_count += clip_frame_count
        self._current_clip_count += 1

    def finish(self) -> list[dict]:
        self._close_shard()
        return list(self.output_shards)

    def abort(self) -> None:
        if self._current_tar is not None:
            self._current_tar.close()
        if self._current_tmp_path and self._current_tmp_path.exists():
            self._current_tmp_path.unlink()
        self._current_tar = None
        self._current_tmp_path = None
        self._current_output_path = None


def _parse_source_spec(raw_value: str, resize_sources: set[str]) -> SourceSpec:
    if "=" not in raw_value:
        raise ValueError(f"Invalid --source spec {raw_value!r}; expected <name>=<wds_dir>")
    name, path_str = raw_value.split("=", 1)
    name = name.strip()
    path = Path(path_str.strip()).resolve()
    if not name:
        raise ValueError(f"Invalid --source spec {raw_value!r}; empty source name")
    if not path.is_dir():
        raise FileNotFoundError(f"Source WDS dir not found: {path}")
    return SourceSpec(name=name, shard_dir=path, resize=name in resize_sources)


def _parse_repeat_override(raw_value: str) -> tuple[str, int]:
    if "=" not in raw_value:
        raise ValueError(f"Invalid --fixed-repeat-source spec {raw_value!r}; expected <name>=<count>")
    name, count_str = raw_value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid --fixed-repeat-source spec {raw_value!r}; empty source name")
    count = int(count_str.strip())
    if count < 1:
        raise ValueError(f"Invalid --fixed-repeat-source spec {raw_value!r}; count must be >= 1")
    return name, count


def _sample_clip_id(sample: dict, meta: dict | None) -> str:
    if meta is not None:
        clip_id = meta.get("clip_id")
        if clip_id:
            return str(clip_id)
    return sample["key"].rsplit("_f", 1)[0]


def _sample_clip_id_fast(sample: dict) -> str:
    sample_key = str(sample["key"])
    if "_f" in sample_key:
        return sample_key.rsplit("_f", 1)[0]
    try:
        meta = json.loads(sample["meta_bytes"].decode("utf-8"))
    except Exception:
        return sample_key
    return _sample_clip_id(sample, meta)


def _encode_npy(array) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return buffer.getvalue()


def _resize_sample(
    image_bytes: bytes,
    lowdim_bytes: bytes,
    *,
    target_width: int,
    target_height: int,
    jpeg_quality: int,
) -> tuple[bytes, bytes]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    src_width, src_height = image.size
    if src_width == target_width and src_height == target_height:
        return image_bytes, lowdim_bytes

    scale_x = float(target_width) / float(src_width)
    scale_y = float(target_height) / float(src_height)

    resized = image.resize((target_width, target_height), resample=Image.BILINEAR)
    image_buffer = io.BytesIO()
    resized.save(image_buffer, format="JPEG", quality=jpeg_quality)

    lowdim = decode_lowdim(lowdim_bytes).copy()
    lowdim[INTRINSIC_SLICE][0] *= scale_x
    lowdim[INTRINSIC_SLICE][2] *= scale_x
    lowdim[INTRINSIC_SLICE][1] *= scale_y
    lowdim[INTRINSIC_SLICE][3] *= scale_y
    return image_buffer.getvalue(), _encode_npy(lowdim)


def _stage_source_root(staging_dir: Path, source_name: str, stage_tag: str) -> Path:
    return staging_dir / source_name / stage_tag


def _stage_episode_path(staging_dir: Path, source_name: str, stage_tag: str, staged_index: int) -> Path:
    return _stage_source_root(staging_dir, source_name, stage_tag) / f"episode-{staged_index:08d}.tar"


def _stage_uid(source_name: str, stage_tag: str, staged_index: int) -> str:
    return f"{source_name}:{stage_tag}:{staged_index:08d}"


def _append_jsonl_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def _read_jsonl_records(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _select_shard_paths(shard_paths: list[str], shard_start: int, shard_end: int | None) -> list[str]:
    total = len(shard_paths)
    if shard_start < 0:
        raise ValueError("--shard_start must be >= 0")
    if shard_end is not None and shard_end < 0:
        raise ValueError("--shard_end must be >= 0")
    if shard_end is not None and shard_end < shard_start:
        raise ValueError("--shard_end must be >= --shard_start")
    if shard_start > total:
        raise ValueError(f"--shard_start ({shard_start}) exceeds shard count ({total})")
    selected_end = total if shard_end is None else min(int(shard_end), total)
    return shard_paths[shard_start:selected_end]


def _sample_episode_key(sample_key: str) -> str:
    if "_f" not in sample_key:
        raise ValueError(f"Sample key does not contain frame suffix: {sample_key}")
    return sample_key.rsplit("_f", 1)[0]


def _stable_hash_hex(*parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _episode_uid(source_name: str, original_clip_id: str) -> str:
    return _stable_hash_hex("episode", source_name, original_clip_id)


def _stable_randint(seed: int, episode_uid: str, low: int, high: int) -> int:
    if high < low:
        raise ValueError(f"Invalid randint range: [{low}, {high}]")
    span = int(high) - int(low) + 1
    value = int(_stable_hash_hex("repeat", seed, episode_uid)[:16], 16)
    return int(low) + (value % span)


def _planned_episode_id(seed: int, output_episode_index: int, episode_uid: str, repeat_index: int) -> str:
    digest = _stable_hash_hex("episode_id", seed, output_episode_index, episode_uid, repeat_index)[:16]
    return f"ep_{digest}"


def _load_json_meta(meta_bytes: bytes) -> dict | None:
    try:
        return json.loads(meta_bytes.decode("utf-8"))
    except Exception:
        return None


def _extract_presence_fast(meta_bytes: bytes) -> int:
    pattern = b'"presence":'
    try:
        start = meta_bytes.index(pattern) + len(pattern)
        end = start
        length = len(meta_bytes)
        while end < length and meta_bytes[end] in b" \t\r\n":
            end += 1
        value_start = end
        while end < length and meta_bytes[end] in b"-0123456789":
            end += 1
        if end > value_start:
            return int(meta_bytes[value_start:end].decode("ascii"))
    except Exception:
        pass
    meta = _load_json_meta(meta_bytes)
    return int((meta or {}).get("presence", 0) or 0)


def _iter_manifest_records(paths: list[str]) -> list[dict]:
    records: list[dict] = []
    for raw_path in paths:
        manifest_path = Path(raw_path).resolve()
        records.extend(_read_jsonl_records(manifest_path))
    return records


def _build_episode_record(
    *,
    manifest_type: str,
    source_name: str,
    original_clip_id: str,
    frame_count: int,
    has_depth: bool,
    source_shard_path: str | None,
    source_episode_key: str | None,
    reserved_only: bool,
) -> dict:
    return {
        "manifest_type": manifest_type,
        "source_name": str(source_name),
        "original_clip_id": str(original_clip_id),
        "episode_uid": _episode_uid(str(source_name), str(original_clip_id)),
        "frame_count": int(frame_count),
        "has_depth": bool(has_depth),
        "source_shard_path": None if source_shard_path is None else str(Path(source_shard_path).resolve()),
        "source_episode_key": None if source_episode_key is None else str(source_episode_key),
        "reserved_only": bool(reserved_only),
    }


def _write_jsonl_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _iter_lightweight_wds_episodes(shard_path: str):
    current_episode_key = None
    current_clip_id = None
    current_frame_count = 0
    current_has_depth = False
    saw_image_for_sample: set[str] = set()

    def _flush_current():
        nonlocal current_episode_key, current_clip_id, current_frame_count, current_has_depth, saw_image_for_sample
        if current_episode_key is None:
            return None
        payload = {
            "source_episode_key": current_episode_key,
            "original_clip_id": current_clip_id or current_episode_key,
            "frame_count": int(current_frame_count),
            "has_depth": bool(current_has_depth),
        }
        current_episode_key = None
        current_clip_id = None
        current_frame_count = 0
        current_has_depth = False
        saw_image_for_sample = set()
        return payload

    with tarfile.open(shard_path, "r|") as tar_reader:
        for member in tar_reader:
            if not member.isfile():
                continue

            sample_key, _suffix, field_name = split_sample_member_name(member.name)
            if sample_key is None:
                raise ValueError(f"Unsupported shard member: {member.name}")
            episode_key = _sample_episode_key(str(sample_key))

            if current_episode_key is None:
                current_episode_key = episode_key
            elif episode_key != current_episode_key:
                payload = _flush_current()
                if payload is not None:
                    yield payload
                current_episode_key = episode_key

            if field_name == "image_bytes" and sample_key not in saw_image_for_sample:
                saw_image_for_sample.add(str(sample_key))
                current_frame_count += 1
            elif field_name == "depth_bytes":
                current_has_depth = True
            elif field_name == "meta_bytes" and current_clip_id is None:
                member_file = tar_reader.extractfile(member)
                if member_file is not None:
                    try:
                        meta = json.loads(member_file.read().decode("utf-8"))
                    except Exception:
                        meta = None
                    if isinstance(meta, dict) and meta.get("clip_id"):
                        current_clip_id = str(meta["clip_id"])

    payload = _flush_current()
    if payload is not None:
        yield payload


def scan_sources_to_episode_manifest(
    *,
    sources: list[SourceSpec],
    shard_start: int,
    shard_end: int | None,
    max_episodes: int | None,
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    report = {
        "sources": {},
        "episode_count": 0,
        "frame_count": 0,
    }

    for source in sources:
        shard_paths = list(iter_shard_paths(str(source.shard_dir)))
        if not shard_paths:
            raise RuntimeError(f"No shard tar files found in {source.shard_dir}")
        shard_paths = _select_shard_paths(shard_paths, shard_start, shard_end)
        source_stats = {
            "source_dir": str(source.shard_dir),
            "shards": len(shard_paths),
            "episodes": 0,
            "frames": 0,
            "episodes_with_depth": 0,
        }
        report["sources"][source.name] = source_stats

        stop_early = False
        for shard_path in tqdm(shard_paths, desc=f"Scan {source.name}"):
            for episode_summary in _iter_lightweight_wds_episodes(shard_path):
                record = _build_episode_record(
                    manifest_type="wds_episode",
                    source_name=source.name,
                    original_clip_id=str(episode_summary["original_clip_id"]),
                    frame_count=int(episode_summary["frame_count"]),
                    has_depth=bool(episode_summary["has_depth"]),
                    source_shard_path=shard_path,
                    source_episode_key=str(episode_summary["source_episode_key"]),
                    reserved_only=False,
                )
                records.append(record)
                source_stats["episodes"] += 1
                source_stats["frames"] += int(episode_summary["frame_count"])
                if bool(episode_summary["has_depth"]):
                    source_stats["episodes_with_depth"] += 1
                report["episode_count"] = int(report["episode_count"]) + 1
                report["frame_count"] = int(report["frame_count"]) + int(episode_summary["frame_count"])
                if max_episodes is not None and int(report["episode_count"]) >= int(max_episodes):
                    stop_early = True
                    break
            if stop_early:
                break

        if stop_early:
            break

    return records, report


def reserve_from_clip_manifest(
    *,
    clip_manifest_path: str,
    source_name: str,
    drop_last_frame: bool,
) -> tuple[list[dict], dict]:
    from lib.pipeline.clip_manifest import load_clip_manifest

    records: list[dict] = []
    total_frames = 0
    for record in load_clip_manifest(clip_manifest_path):
        frame_count = int(record.descriptor.frame_count)
        if drop_last_frame:
            frame_count = max(frame_count - 1, 0)
        records.append(
            _build_episode_record(
                manifest_type="reserved_episode",
                source_name=source_name,
                original_clip_id=record.clip_id,
                frame_count=frame_count,
                has_depth=False,
                source_shard_path=None,
                source_episode_key=None,
                reserved_only=True,
            )
        )
        total_frames += frame_count

    report = {
        "source_name": source_name,
        "clip_manifest": str(Path(clip_manifest_path).resolve()),
        "drop_last_frame": bool(drop_last_frame),
        "episode_count": len(records),
        "frame_count": int(total_frames),
    }
    return records, report


def import_from_clip_manifest(
    *,
    clip_manifest_path: str,
    source_name: str | None,
    has_depth: bool | None,
) -> tuple[list[dict], dict]:
    from lib.pipeline.clip_manifest import load_clip_manifest

    manifest_records = load_clip_manifest(clip_manifest_path)
    if not manifest_records:
        raise RuntimeError(f"No clip records found in {clip_manifest_path}")

    records: list[dict] = []
    total_frames = 0
    resolved_sources: set[str] = set()
    tar_shard_records = 0
    for record in manifest_records:
        resolved_source_name = str(source_name or record.source_id)
        resolved_sources.add(resolved_source_name)
        descriptor = record.descriptor
        if not descriptor.is_tar_shard or not descriptor.shard_path:
            raise ValueError(
                f"Clip manifest record {record.clip_id} is not a tar-shard descriptor; "
                f"storage_kind={descriptor.storage_kind!r} shard_path={descriptor.shard_path!r}"
            )
        tar_shard_records += 1
        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        record_has_depth = bool(metadata.get("has_depth", False)) if has_depth is None else bool(has_depth)
        frame_count = int(descriptor.frame_count)
        records.append(
            _build_episode_record(
                manifest_type="wds_episode",
                source_name=resolved_source_name,
                original_clip_id=record.clip_id,
                frame_count=frame_count,
                has_depth=record_has_depth,
                source_shard_path=str(Path(descriptor.shard_path).resolve()),
                source_episode_key=str(record.clip_id),
                reserved_only=False,
            )
        )
        total_frames += frame_count

    report = {
        "clip_manifest": str(Path(clip_manifest_path).resolve()),
        "source_names": sorted(resolved_sources),
        "episode_count": len(records),
        "frame_count": int(total_frames),
        "tar_shard_records": int(tar_shard_records),
        "has_depth_override": has_depth,
    }
    return records, report


def _load_unique_episode_records(manifest_paths: list[str]) -> tuple[list[dict], set[str]]:
    by_uid: dict[str, dict] = {}
    source_names: set[str] = set()
    for record in _iter_manifest_records(manifest_paths):
        episode_uid = str(record["episode_uid"])
        source_name = str(record["source_name"])
        source_names.add(source_name)
        existing = by_uid.get(episode_uid)
        if existing is None:
            by_uid[episode_uid] = dict(record)
            continue
        if json.dumps(existing, sort_keys=True) != json.dumps(record, sort_keys=True):
            raise ValueError(f"Conflicting episode records for episode_uid={episode_uid}")
    records = sorted(by_uid.values(), key=lambda item: (str(item["source_name"]), str(item["original_clip_id"])))
    return records, source_names


def _deadjacent_planned_copies(planned_copies: list[dict], *, seed: int) -> list[dict]:
    if len(planned_copies) < 2:
        return list(planned_copies)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in planned_copies:
        grouped[str(item["episode_uid"])].append(item)
    for episode_uid, items in grouped.items():
        grouped[episode_uid] = sorted(items, key=lambda item: (str(item["order_key"]), int(item["repeat_index"])))

    heap: list[tuple[int, int, str]] = []
    for episode_uid, items in grouped.items():
        tie = int(_stable_hash_hex("heap", seed, episode_uid)[:16], 16)
        heapq.heappush(heap, (-len(items), tie, episode_uid))

    ordered: list[dict] = []
    prev_episode_uid: str | None = None
    while heap:
        count_a, tie_a, episode_uid_a = heapq.heappop(heap)
        if prev_episode_uid is not None and episode_uid_a == prev_episode_uid:
            if not heap:
                raise RuntimeError(
                    "Unable to separate repeated copies of the same episode. "
                    "Add more distinct episodes or lower the repeat count."
                )
            count_b, tie_b, episode_uid_b = heapq.heappop(heap)
            ordered.append(grouped[episode_uid_b].pop(0))
            prev_episode_uid = episode_uid_b
            count_b += 1
            if count_b < 0:
                heapq.heappush(heap, (count_b, tie_b, episode_uid_b))
            heapq.heappush(heap, (count_a, tie_a, episode_uid_a))
            continue

        ordered.append(grouped[episode_uid_a].pop(0))
        prev_episode_uid = episode_uid_a
        count_a += 1
        if count_a < 0:
            heapq.heappush(heap, (count_a, tie_a, episode_uid_a))

    return ordered


def build_frozen_mix_plan(
    *,
    episode_records: list[dict],
    repeat_min: int,
    repeat_max: int,
    seed: int,
    fixed_repeat_by_source: dict[str, int],
    frames_per_shard: int,
) -> tuple[list[dict], dict]:
    planned_copies: list[dict] = []
    repeat_histogram: dict[int, int] = {}
    source_repeat_histogram: dict[str, dict[int, int]] = defaultdict(dict)

    for record in episode_records:
        source_name = str(record["source_name"])
        episode_uid = str(record["episode_uid"])
        repeat_count = int(
            fixed_repeat_by_source.get(
                source_name,
                _stable_randint(int(seed), episode_uid, int(repeat_min), int(repeat_max)),
            )
        )
        repeat_histogram[repeat_count] = repeat_histogram.get(repeat_count, 0) + 1
        source_repeat_histogram[source_name][repeat_count] = source_repeat_histogram[source_name].get(repeat_count, 0) + 1
        for repeat_index in range(repeat_count):
            planned_copies.append(
                {
                    "episode_uid": episode_uid,
                    "source_name": source_name,
                    "original_clip_id": str(record["original_clip_id"]),
                    "frame_count": int(record["frame_count"]),
                    "has_depth": bool(record["has_depth"]),
                    "reserved_only": bool(record.get("reserved_only", False)),
                    "repeat_index": int(repeat_index),
                    "order_key": _stable_hash_hex("order", seed, episode_uid, repeat_index),
                }
            )

    ordered = _deadjacent_planned_copies(planned_copies, seed=seed)

    plan_records: list[dict] = []
    current_shard_idx = 0
    current_shard_frames = 0
    source_shard_spans: dict[str, dict[str, int]] = {}
    source_output_shards: dict[str, set[int]] = defaultdict(set)
    for output_episode_index, item in enumerate(ordered):
        frame_count = int(item["frame_count"])
        if current_shard_frames > 0 and (current_shard_frames + frame_count) > int(frames_per_shard):
            current_shard_idx += 1
            current_shard_frames = 0
        episode_id = _planned_episode_id(int(seed), int(output_episode_index), str(item["episode_uid"]), int(item["repeat_index"]))
        record = {
            "output_episode_index": int(output_episode_index),
            "output_shard_idx": int(current_shard_idx),
            "output_shard_name": f"shard-{int(current_shard_idx):06d}.tar",
            "episode_id": episode_id,
            "episode_uid": str(item["episode_uid"]),
            "source_name": str(item["source_name"]),
            "original_clip_id": str(item["original_clip_id"]),
            "frame_count": frame_count,
            "has_depth": bool(item["has_depth"]),
            "reserved_only": bool(item["reserved_only"]),
            "repeat_index": int(item["repeat_index"]),
        }
        plan_records.append(record)
        current_shard_frames += frame_count
        span = source_shard_spans.setdefault(
            str(item["source_name"]),
            {"shard_min": int(current_shard_idx), "shard_max": int(current_shard_idx), "episodes": 0},
        )
        span["shard_min"] = min(int(span["shard_min"]), int(current_shard_idx))
        span["shard_max"] = max(int(span["shard_max"]), int(current_shard_idx))
        span["episodes"] = int(span["episodes"]) + 1
        source_output_shards[str(item["source_name"])].add(int(current_shard_idx))

    report = {
        "input_episode_count": len(episode_records),
        "planned_episode_count": len(plan_records),
        "planned_shard_count": 0 if not plan_records else int(plan_records[-1]["output_shard_idx"]) + 1,
        "repeat_histogram": {str(key): int(value) for key, value in sorted(repeat_histogram.items())},
        "source_repeat_histogram": {
            source_name: {str(key): int(value) for key, value in sorted(hist.items())}
            for source_name, hist in sorted(source_repeat_histogram.items())
        },
        "source_output_shard_spans": source_shard_spans,
        "source_output_shards": {
            source_name: sorted(int(value) for value in shard_ids)
            for source_name, shard_ids in sorted(source_output_shards.items())
        },
    }
    return plan_records, report


def _load_mix_plan(path: str) -> list[dict]:
    records = _read_jsonl_records(Path(path).resolve())
    return sorted(records, key=lambda item: int(item["output_episode_index"]))


def _select_output_plan_shards(plan_records: list[dict], shard_start: int, shard_end: int | None) -> list[dict]:
    if shard_start < 0:
        raise ValueError("--shard_start must be >= 0")
    if shard_end is not None and shard_end < shard_start:
        raise ValueError("--shard_end must be >= --shard_start")
    selected_end = None if shard_end is None else int(shard_end)
    selected = []
    for record in plan_records:
        shard_idx = int(record["output_shard_idx"])
        if shard_idx < int(shard_start):
            continue
        if selected_end is not None and shard_idx >= selected_end:
            continue
        selected.append(record)
    return selected


def _load_available_wds_episode_lookup(manifest_paths: list[str]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for record in _iter_manifest_records(manifest_paths):
        if str(record.get("manifest_type")) != "wds_episode":
            continue
        episode_uid = str(record["episode_uid"])
        existing = lookup.get(episode_uid)
        if existing is None:
            lookup[episode_uid] = dict(record)
            continue
        if json.dumps(existing, sort_keys=True) != json.dumps(record, sort_keys=True):
            raise ValueError(f"Conflicting available records for episode_uid={episode_uid}")
    return lookup


def _configure_source_index_cache_root(path: str | Path) -> Path:
    global _source_index_cache_root
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    with _source_index_cache_root_lock:
        _source_index_cache_root = resolved
    return resolved


def _shard_sample_index_cache_path(shard_path: str) -> Path:
    resolved = str(Path(shard_path).resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()
    with _source_index_cache_root_lock:
        cache_root = _source_index_cache_root
    return cache_root / f"{Path(shard_path).name}.{digest}.pkl.gz"


def _load_shard_sample_index_from_disk(cache_path: Path, shard_path: str) -> dict | None:
    if not cache_path.is_file():
        return None
    try:
        with gzip.open(cache_path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    try:
        stat = Path(shard_path).stat()
    except FileNotFoundError:
        return None
    if int(payload.get("format_version", 0)) != SHARD_SAMPLE_INDEX_FORMAT_VERSION:
        return None
    if str(payload.get("shard_path")) != str(Path(shard_path).resolve()):
        return None
    if int(payload.get("shard_size", -1)) != int(stat.st_size):
        return None
    if int(payload.get("shard_mtime_ns", -1)) != int(stat.st_mtime_ns):
        return None
    return payload


def _write_shard_sample_index_to_disk(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with gzip.open(tmp_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)


def _build_shard_sample_index(shard_path: str) -> dict:
    shard_path = str(Path(shard_path).resolve())
    stat = Path(shard_path).stat()
    by_sample_key: dict[str, dict[str, tuple[int, int]]] = {}
    by_episode_key: dict[str, list[str]] = defaultdict(list)

    with tarfile.open(shard_path, "r|") as tar_reader:
        for member in tar_reader:
            if not member.isfile():
                continue

            sample_key, _suffix, field_name = split_sample_member_name(member.name)
            if sample_key is None:
                raise ValueError(f"Unsupported shard member: {member.name}")
            sample_key = str(sample_key)
            fields = by_sample_key.setdefault(sample_key, {})
            fields[str(field_name)] = (int(member.offset_data), int(member.size))

    for sample_key in by_sample_key:
        by_episode_key[_sample_episode_key(sample_key)].append(sample_key)
    for sample_keys in by_episode_key.values():
        sample_keys.sort()

    return {
        "format_version": SHARD_SAMPLE_INDEX_FORMAT_VERSION,
        "shard_path": shard_path,
        "shard_size": int(stat.st_size),
        "shard_mtime_ns": int(stat.st_mtime_ns),
        "episodes": dict(by_episode_key),
        "samples": by_sample_key,
    }


def _get_shard_sample_index(shard_path: str) -> dict:
    shard_path = str(Path(shard_path).resolve())
    with _shard_sample_index_mem_cache_lock:
        cached = _shard_sample_index_mem_cache.pop(shard_path, None)
        if cached is not None:
            _shard_sample_index_mem_cache[shard_path] = cached
            return cached

    cache_path = _shard_sample_index_cache_path(shard_path)
    index = _load_shard_sample_index_from_disk(cache_path, shard_path)
    if index is None:
        index = _build_shard_sample_index(shard_path)
        _write_shard_sample_index_to_disk(cache_path, index)

    with _shard_sample_index_mem_cache_lock:
        _shard_sample_index_mem_cache[shard_path] = index
        while len(_shard_sample_index_mem_cache) > SHARD_SAMPLE_INDEX_MEM_CACHE_MAX:
            _shard_sample_index_mem_cache.popitem(last=False)
    return index


def _pread_exact(fd: int, offset: int, size: int, *, shard_path: str, sample_key: str, field_name: str) -> bytes:
    payload = os.pread(fd, int(size), int(offset))
    if len(payload) != int(size):
        raise RuntimeError(
            f"Short read from shard {shard_path}: sample={sample_key} field={field_name} "
            f"expected={int(size)} got={len(payload)}"
        )
    return payload


def _get_source_shard_fd(shard_path: str) -> int:
    resolved = str(Path(shard_path).resolve())
    with _source_shard_fd_cache_lock:
        cached = _source_shard_fd_cache.pop(resolved, None)
        if cached is not None:
            _source_shard_fd_cache[resolved] = cached
            return cached

    fd = os.open(resolved, os.O_RDONLY)
    evicted_fd = None
    with _source_shard_fd_cache_lock:
        prior = _source_shard_fd_cache.pop(resolved, None)
        if prior is not None:
            _source_shard_fd_cache[resolved] = prior
            evicted_fd = fd
            fd = prior
        else:
            _source_shard_fd_cache[resolved] = fd
            while len(_source_shard_fd_cache) > SHARD_SOURCE_FD_CACHE_MAX:
                _evicted_path, evicted_fd = _source_shard_fd_cache.popitem(last=False)
                break
    if evicted_fd is not None:
        try:
            os.close(evicted_fd)
        except OSError:
            pass
    return fd


def _load_needed_episodes_from_shard(shard_path: str, needed_episode_keys: set[str]) -> dict[str, list[dict]]:
    index = _get_shard_sample_index(shard_path)
    loaded_by_sample_key: dict[str, dict] = {}
    pending_reads: list[tuple[int, int, str, str]] = []

    for episode_key in sorted(str(value) for value in needed_episode_keys):
        sample_keys = index["episodes"].get(episode_key) or []
        if not sample_keys:
            continue
        for sample_key in sample_keys:
            loaded_by_sample_key[sample_key] = {
                "key": sample_key,
                "image_bytes": None,
                "lowdim_bytes": None,
                "mano_bytes": None,
                "depth_bytes": None,
                "meta_bytes": None,
            }
            fields = index["samples"].get(sample_key) or {}
            for field_name in ("image_bytes", "lowdim_bytes", "mano_bytes", "depth_bytes", "meta_bytes"):
                offset_size = fields.get(field_name)
                if offset_size is None:
                    continue
                pending_reads.append(
                    (
                        int(offset_size[0]),
                        int(offset_size[1]),
                        sample_key,
                        field_name,
                    )
                )

    pending_reads.sort(key=lambda item: item[0])
    fd = _get_source_shard_fd(shard_path)
    for offset, size, sample_key, field_name in pending_reads:
        loaded_by_sample_key[sample_key][field_name] = _pread_exact(
            fd,
            offset,
            size,
            shard_path=shard_path,
            sample_key=sample_key,
            field_name=field_name,
        )

    loaded: dict[str, list[dict]] = {}
    for episode_key in sorted(str(value) for value in needed_episode_keys):
        sample_keys = index["episodes"].get(episode_key) or []
        if not sample_keys:
            continue
        episode_samples: list[dict] = []
        for sample_key in sample_keys:
            sample = loaded_by_sample_key.get(sample_key)
            if sample is None:
                continue
            validate_sample_record(sample)
            episode_samples.append(sample)
        loaded[episode_key] = episode_samples
    return loaded


def _write_single_output_shard(
    *,
    shard_idx: int,
    shard_entries: list[dict],
    available_lookup: dict[str, dict],
    output_dir: Path,
    dataset_name: str,
    split: str,
    exclude_sources: set[str],
    allow_missing_episodes: bool,
    resume: bool,
) -> dict:
    shard_name = f"shard-{int(shard_idx):06d}.tar"
    output_path = output_dir / shard_name
    tmp_path = output_dir / f"{shard_name}.tmp"
    if bool(resume) and output_path.is_file():
        return {
            "output_shard_idx": int(shard_idx),
            "output_shard_name": shard_name,
            "status": "skipped_existing",
            "planned_episode_count": len(shard_entries),
            "written_episode_count": 0,
            "missing_episode_count": 0,
            "written_frame_count": 0,
        }

    needed_by_input_shard: dict[str, set[str]] = defaultdict(set)
    available_entries: list[dict] = []
    missing_entries: list[dict] = []

    for entry in shard_entries:
        source_name = str(entry["source_name"])
        if source_name in exclude_sources:
            missing_entries.append(entry)
            continue
        actual = available_lookup.get(str(entry["episode_uid"]))
        if actual is None:
            missing_entries.append(entry)
            continue
        source_shard_path = str(actual["source_shard_path"])
        source_episode_key = str(actual["source_episode_key"])
        needed_by_input_shard[source_shard_path].add(source_episode_key)
        joined = dict(entry)
        joined["_actual"] = actual
        available_entries.append(joined)

    if missing_entries and not bool(allow_missing_episodes):
        missing_uids = [str(item["episode_uid"]) for item in missing_entries[:16]]
        raise RuntimeError(
            f"Output shard {shard_name} has {len(missing_entries)} unavailable planned episode(s): {missing_uids}"
        )

    loaded_samples_by_input: dict[tuple[str, str], list[dict]] = {}
    for source_shard_path, needed_episode_keys in needed_by_input_shard.items():
        payload = _load_needed_episodes_from_shard(source_shard_path, needed_episode_keys)
        for episode_key, samples in payload.items():
            loaded_samples_by_input[(source_shard_path, episode_key)] = samples

    written_episode_count = 0
    written_frame_count = 0
    try:
        with tarfile.open(tmp_path, "w") as tar_writer:
            for entry in available_entries:
                actual = entry["_actual"]
                source_shard_path = str(actual["source_shard_path"])
                source_episode_key = str(actual["source_episode_key"])
                source_samples = loaded_samples_by_input.get((source_shard_path, source_episode_key))
                if not source_samples:
                    if bool(allow_missing_episodes):
                        continue
                    raise RuntimeError(
                        f"Failed to load planned episode_uid={entry['episode_uid']} from {source_shard_path}"
                    )
                if len(source_samples) != int(actual["frame_count"]):
                    raise RuntimeError(
                        f"Episode frame_count mismatch for episode_uid={entry['episode_uid']}: "
                        f"manifest={int(actual['frame_count'])} loaded={len(source_samples)}"
                    )

                base_meta = _load_json_meta(source_samples[0]["meta_bytes"]) if source_samples else None
                meta_prefix = _build_final_meta_prefix(
                    base_meta,
                    dataset_name=dataset_name,
                    split=split,
                    episode_id=str(entry["episode_id"]),
                    episode_index=int(entry["output_episode_index"]),
                )

                frame_count_written = 0
                for frame_idx, sample in enumerate(source_samples):
                    meta_bytes = meta_prefix + str(_extract_presence_fast(sample["meta_bytes"])).encode("ascii") + b"}"
                    write_sample_to_tar(
                        tar_writer,
                        f"{entry['episode_id']}_f{frame_idx:06d}",
                        sample["image_bytes"],
                        sample["lowdim_bytes"],
                        meta_bytes,
                        mano_bytes=sample.get("mano_bytes"),
                        depth_bytes=sample.get("depth_bytes"),
                    )
                    frame_count_written += 1
                written_episode_count += 1
                written_frame_count += frame_count_written
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    if written_episode_count <= 0:
        if tmp_path.exists():
            tmp_path.unlink()
        return {
            "output_shard_idx": int(shard_idx),
            "output_shard_name": shard_name,
            "status": "pending_missing_all",
            "planned_episode_count": len(shard_entries),
            "written_episode_count": 0,
            "missing_episode_count": int(len(missing_entries)),
            "written_frame_count": 0,
        }

    os.replace(tmp_path, output_path)
    return {
        "output_shard_idx": int(shard_idx),
        "output_shard_name": shard_name,
        "status": "written",
        "planned_episode_count": len(shard_entries),
        "written_episode_count": int(written_episode_count),
        "missing_episode_count": int(len(missing_entries)),
        "written_frame_count": int(written_frame_count),
    }


def write_from_frozen_plan(
    *,
    plan_records: list[dict],
    available_lookup: dict[str, dict],
    output_dir: Path,
    dataset_name: str,
    split: str,
    exclude_sources: set[str],
    allow_missing_episodes: bool,
    resume: bool,
    workers: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_shard: dict[int, list[dict]] = defaultdict(list)
    for record in plan_records:
        by_shard[int(record["output_shard_idx"])].append(record)

    report_items: list[dict] = []
    shard_indices = sorted(by_shard)
    resolved_workers = max(1, min(int(workers), len(shard_indices)))

    if resolved_workers == 1:
        iterator = shard_indices
        if len(shard_indices) > 1:
            iterator = tqdm(shard_indices, desc="Write mix shards")
        for shard_idx in iterator:
            report_items.append(
                _write_single_output_shard(
                    shard_idx=shard_idx,
                    shard_entries=by_shard[shard_idx],
                    available_lookup=available_lookup,
                    output_dir=output_dir,
                    dataset_name=dataset_name,
                    split=split,
                    exclude_sources=exclude_sources,
                    allow_missing_episodes=allow_missing_episodes,
                    resume=resume,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
            future_to_shard = {
                executor.submit(
                    _write_single_output_shard,
                    shard_idx=shard_idx,
                    shard_entries=by_shard[shard_idx],
                    available_lookup=available_lookup,
                    output_dir=output_dir,
                    dataset_name=dataset_name,
                    split=split,
                    exclude_sources=exclude_sources,
                    allow_missing_episodes=allow_missing_episodes,
                    resume=resume,
                ): shard_idx
                for shard_idx in shard_indices
            }
            progress = tqdm(total=len(future_to_shard), desc=f"Write mix shards x{resolved_workers}")
            try:
                for future in as_completed(future_to_shard):
                    report_items.append(future.result())
                    progress.update(1)
            finally:
                progress.close()

    report_items.sort(key=lambda item: int(item["output_shard_idx"]))
    total_written_frames = int(sum(int(item.get("written_frame_count", 0)) for item in report_items))
    total_missing_episodes = int(sum(int(item.get("missing_episode_count", 0)) for item in report_items))

    return {
        "selected_shards": len(by_shard),
        "written_frames": int(total_written_frames),
        "missing_episodes": int(total_missing_episodes),
        "items": report_items,
    }


def _flush_staged_episode(
    *,
    staging_dir: Path,
    source: SourceSpec,
    stage_tag: str,
    staged_index: int,
    original_clip_id: str,
    episode_samples: list[dict],
    stage_records: list[dict],
    source_stats: dict,
    stage_manifest_out: Path | None,
) -> None:
    if not episode_samples:
        return

    stage_path = _stage_episode_path(staging_dir, source.name, stage_tag, staged_index)
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = stage_path.with_suffix(".tar.tmp")

    with tarfile.open(tmp_path, "w") as tar_writer:
        for sample in episode_samples:
            write_sample_to_tar(
                tar_writer,
                sample["key"],
                sample["image_bytes"],
                sample["lowdim_bytes"],
                sample["meta_bytes"],
                mano_bytes=sample.get("mano_bytes"),
                depth_bytes=sample.get("depth_bytes"),
            )
    os.replace(tmp_path, stage_path)

    frame_count = len(episode_samples)
    has_depth = any(sample.get("depth_bytes") is not None for sample in episode_samples)
    record = {
        "stage_index": int(staged_index),
        "stage_tag": stage_tag,
        "stage_uid": _stage_uid(source.name, stage_tag, staged_index),
        "source_name": source.name,
        "stage_relpath": str(stage_path.relative_to(staging_dir)),
        "frame_count": int(frame_count),
        "has_depth": bool(has_depth),
        "original_clip_id_sha1": hashlib.sha1(original_clip_id.encode("utf-8")).hexdigest(),
    }
    stage_records.append(record)
    if stage_manifest_out is not None:
        _append_jsonl_record(stage_manifest_out, record)
    source_stats["episodes"] += 1
    source_stats["frames"] += frame_count
    if has_depth:
        source_stats["episodes_with_depth"] += 1


def stage_sources(
    *,
    sources: list[SourceSpec],
    staging_dir: Path,
    stage_tag: str,
    resize_width: int,
    resize_height: int,
    jpeg_quality: int,
    max_episodes: int | None,
    stage_manifest_out: Path | None,
    shard_start: int,
    shard_end: int | None,
) -> tuple[list[dict], dict]:
    staging_dir.mkdir(parents=True, exist_ok=True)

    stage_records: list[dict] = []
    report = {
        "stage_tag": stage_tag,
        "shard_start": int(shard_start),
        "shard_end": None if shard_end is None else int(shard_end),
        "sources": {},
        "staged_episodes": 0,
        "staged_frames": 0,
    }
    staged_index = 0

    for source in sources:
        source_stage_root = _stage_source_root(staging_dir, source.name, stage_tag)
        if source_stage_root.exists() and any(source_stage_root.iterdir()):
            raise RuntimeError(f"Stage target must be empty before running: {source_stage_root}")
        shard_paths = list(iter_shard_paths(str(source.shard_dir)))
        if not shard_paths:
            raise RuntimeError(f"No shard tar files found in {source.shard_dir}")
        shard_paths = _select_shard_paths(shard_paths, shard_start, shard_end)
        if not shard_paths:
            continue

        source_stats = {
            "source_dir": str(source.shard_dir),
            "resize_applied": bool(source.resize),
            "shards": len(shard_paths),
            "episodes": 0,
            "frames": 0,
            "episodes_with_depth": 0,
        }
        report["sources"][source.name] = source_stats

        stop_early = False
        for shard_path in tqdm(shard_paths, desc=f"Stage {source.name}"):
            current_clip_id = None
            current_episode_samples: list[dict] = []

            for sample in iter_shard_samples(shard_path):
                validate_sample_record(sample)
                clip_id = _sample_clip_id_fast(sample)

                if current_clip_id is None:
                    current_clip_id = clip_id
                elif clip_id != current_clip_id:
                    flushed_frame_count = len(current_episode_samples)
                    _flush_staged_episode(
                        staging_dir=staging_dir,
                        source=source,
                        stage_tag=stage_tag,
                        staged_index=staged_index,
                        original_clip_id=current_clip_id,
                        episode_samples=current_episode_samples,
                        stage_records=stage_records,
                        source_stats=source_stats,
                        stage_manifest_out=stage_manifest_out,
                    )
                    staged_index += 1
                    current_clip_id = clip_id
                    current_episode_samples = []
                    report["staged_episodes"] = int(report["staged_episodes"]) + 1
                    report["staged_frames"] = int(report["staged_frames"]) + flushed_frame_count
                    if max_episodes is not None and len(stage_records) >= max_episodes:
                        stop_early = True
                        break

                image_bytes = sample["image_bytes"]
                lowdim_bytes = sample["lowdim_bytes"]
                if source.resize:
                    image_bytes, lowdim_bytes = _resize_sample(
                        image_bytes,
                        lowdim_bytes,
                        target_width=resize_width,
                        target_height=resize_height,
                        jpeg_quality=jpeg_quality,
                    )

                frame_idx = len(current_episode_samples)
                current_episode_samples.append(
                    {
                        "key": f"f{frame_idx:06d}",
                        "image_bytes": image_bytes,
                        "lowdim_bytes": lowdim_bytes,
                        "mano_bytes": sample.get("mano_bytes"),
                        "depth_bytes": sample.get("depth_bytes"),
                        "meta_bytes": sample["meta_bytes"],
                    }
                )

            if stop_early:
                break

            if current_episode_samples:
                _flush_staged_episode(
                    staging_dir=staging_dir,
                    source=source,
                    stage_tag=stage_tag,
                    staged_index=staged_index,
                    original_clip_id=current_clip_id or f"unknown-{staged_index}",
                    episode_samples=current_episode_samples,
                    stage_records=stage_records,
                    source_stats=source_stats,
                    stage_manifest_out=stage_manifest_out,
                )
                staged_index += 1
                report["staged_episodes"] = int(report["staged_episodes"]) + 1
                report["staged_frames"] = int(report["staged_frames"]) + len(current_episode_samples)
                if max_episodes is not None and len(stage_records) >= max_episodes:
                    stop_early = True
                    break

        if stop_early:
            break

    return stage_records, report


def _build_final_meta_prefix(
    original_meta: dict | None,
    *,
    dataset_name: str,
    split: str,
    episode_id: str,
    episode_index: int,
) -> bytes:
    meta = original_meta if isinstance(original_meta, dict) else {}
    instruction = meta.get("instruction", [])
    if isinstance(instruction, str):
        instruction = [instruction]
    elif not isinstance(instruction, list):
        instruction = list(instruction) if isinstance(instruction, tuple) else []

    payload: dict[str, object] = {}
    for key, value in meta.items():
        if key.startswith("mano_"):
            payload[key] = value
            continue
        if key in {"cameras", "language", "depth_schema", "depth_encoding"}:
            payload[key] = value
            continue
        if key.endswith("_schema") or key.endswith("_convention"):
            payload[key] = value

    payload.update(
        {
        "dataset_name": dataset_name,
        "clip_id": episode_id,
        "episode_id": episode_id,
        "episode_index": int(episode_index),
        "split": split,
        "instruction": instruction,
        "instruction_num": int(meta.get("instruction_num", len(instruction) if instruction else 0)),
        "lowdim_schema": meta.get("lowdim_schema", "hawor_wrist_world_v2"),
        "wrist_translation_semantics": meta.get("wrist_translation_semantics", "mano_joint_0_world"),
        "camera_extrinsic_convention": meta.get("camera_extrinsic_convention", "w2c"),
        }
    )
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (encoded[:-1] + ',"presence":').encode("utf-8")


def _build_final_meta(
    original_meta: dict | None,
    *,
    dataset_name: str,
    split: str,
    episode_id: str,
    episode_index: int,
    presence: int,
) -> bytes:
    return _build_final_meta_prefix(
        original_meta,
        dataset_name=dataset_name,
        split=split,
        episode_id=episode_id,
        episode_index=episode_index,
    ) + str(int(presence)).encode("ascii") + b"}"


def _make_opaque_clip_id(seed: int, output_episode_index: int, stage_index: int, repeat_index: int) -> str:
    digest = hashlib.sha1(
        f"{seed}:{output_episode_index}:{stage_index}:{repeat_index}".encode("utf-8")
    ).hexdigest()[:16]
    return f"ep_{digest}"


def _iter_stage_samples(stage_tar_path: str):
    return iter_shard_samples(stage_tar_path)


def _deadjacent_plan(plan: list[dict], *, seed: int) -> list[dict]:
    if len(plan) < 2:
        return list(plan)

    rng = random.Random(seed + 17)
    grouped: dict[str, list[dict]] = {}
    for item in plan:
        grouped.setdefault(str(item["stage_uid"]), []).append(item)

    for items in grouped.values():
        rng.shuffle(items)

    heap: list[tuple[int, float, str]] = []
    for stage_uid, items in grouped.items():
        heapq.heappush(heap, (-len(items), rng.random(), stage_uid))

    arranged: list[dict] = []
    prev_stage_uid: str | None = None

    while heap:
        count_a, tie_a, stage_uid_a = heapq.heappop(heap)
        if prev_stage_uid is not None and stage_uid_a == prev_stage_uid:
            if not heap:
                raise RuntimeError(
                    "Unable to separate repeated copies of the same episode. "
                    "Add more distinct episodes or lower the repeat count."
                )
            count_b, tie_b, stage_uid_b = heapq.heappop(heap)
            arranged.append(grouped[stage_uid_b].pop())
            prev_stage_uid = stage_uid_b
            count_b += 1
            if count_b < 0:
                heapq.heappush(heap, (count_b, rng.random(), stage_uid_b))
            heapq.heappush(heap, (count_a, tie_a, stage_uid_a))
            continue

        arranged.append(grouped[stage_uid_a].pop())
        prev_stage_uid = stage_uid_a
        count_a += 1
        if count_a < 0:
            heapq.heappush(heap, (count_a, rng.random(), stage_uid_a))

    return arranged


def build_output_plan(
    stage_records: list[dict],
    *,
    repeat_min: int,
    repeat_max: int,
    seed: int,
    fixed_repeat_by_source: dict[str, int],
) -> list[dict]:
    rng = random.Random(seed)
    planned: list[dict] = []
    repeat_histogram: dict[int, int] = {}
    source_repeat_histogram: dict[str, dict[int, int]] = {}

    for record in stage_records:
        source_name = str(record["source_name"])
        repeat_count = fixed_repeat_by_source.get(source_name, rng.randint(repeat_min, repeat_max))
        repeat_histogram[repeat_count] = repeat_histogram.get(repeat_count, 0) + 1
        source_repeat_histogram.setdefault(source_name, {})
        source_repeat_histogram[source_name][repeat_count] = (
            source_repeat_histogram[source_name].get(repeat_count, 0) + 1
        )
        for repeat_index in range(repeat_count):
            planned.append(
                {
                    "stage_index": int(record.get("stage_index", 0)),
                    "stage_uid": str(record.get("stage_uid") or f"{source_name}:{int(record.get('stage_index', 0)):08d}"),
                    "source_name": source_name,
                    "stage_tar": str(record["stage_tar"]),
                    "frame_count": int(record["frame_count"]),
                    "has_depth": bool(record["has_depth"]),
                    "repeat_index": repeat_index,
                }
            )

    rng.shuffle(planned)
    planned = _deadjacent_plan(planned, seed=seed)
    return planned, repeat_histogram, source_repeat_histogram


def write_mixed_dataset(
    *,
    plan: list[dict],
    output_dir: Path,
    frames_per_shard: int,
    dataset_name: str,
    split: str,
    seed: int,
) -> dict:
    writer = MixedShardWriter(output_dir, frames_per_shard)
    source_episode_counts: dict[str, int] = {}
    source_frame_counts: dict[str, int] = {}

    try:
        for output_episode_index, episode_ref in enumerate(tqdm(plan, desc="Write mixed WDS")):
            episode_id = _make_opaque_clip_id(
                seed,
                output_episode_index,
                int(hashlib.sha1(str(episode_ref["stage_uid"]).encode("utf-8")).hexdigest()[:8], 16),
                int(episode_ref["repeat_index"]),
            )
            output_samples: list[dict] = []
            meta_prefix: bytes | None = None

            for sample in _iter_stage_samples(episode_ref["stage_tar"]):
                validate_sample_record(sample)
                if meta_prefix is None:
                    meta = _load_json_meta(sample["meta_bytes"])
                    meta_prefix = _build_final_meta_prefix(
                        meta,
                        dataset_name=dataset_name,
                        split=split,
                        episode_id=episode_id,
                        episode_index=output_episode_index,
                    )

                frame_idx = len(output_samples)
                output_samples.append(
                    {
                        "key": f"{episode_id}_f{frame_idx:06d}",
                        "image_bytes": sample["image_bytes"],
                        "lowdim_bytes": sample["lowdim_bytes"],
                        "mano_bytes": sample.get("mano_bytes"),
                        "depth_bytes": sample.get("depth_bytes"),
                        "meta_bytes": meta_prefix + str(_extract_presence_fast(sample["meta_bytes"])).encode("ascii") + b"}",
                    }
                )

            writer.add_episode(output_samples)
            source_name = str(episode_ref["source_name"])
            source_episode_counts[source_name] = source_episode_counts.get(source_name, 0) + 1
            source_frame_counts[source_name] = source_frame_counts.get(source_name, 0) + len(output_samples)

        output_shards = writer.finish()
    except Exception:
        writer.abort()
        raise

    return {
        "episodes_written": len(plan),
        "frames_written": int(sum(int(item["frame_count"]) for item in plan)),
        "source_episode_counts": source_episode_counts,
        "source_frame_counts": source_frame_counts,
        "output_shards": output_shards,
    }


def _resolve_sources_from_args(args: argparse.Namespace) -> tuple[list[SourceSpec], set[str]]:
    resize_sources = set(args.resize_source or [])
    sources = [_parse_source_spec(raw_value, resize_sources) for raw_value in args.source]
    source_names = [spec.name for spec in sources]
    if len(source_names) != len(set(source_names)):
        raise ValueError(f"Duplicate source names are not allowed: {source_names}")
    return sources, resize_sources


def _validate_common_stage_args(args: argparse.Namespace) -> None:
    if args.resize_width < 1 or args.resize_height < 1:
        raise ValueError("resize target must be positive")
    if not (1 <= args.jpeg_quality <= 100):
        raise ValueError("--jpeg-quality must be in [1, 100]")


def _validate_common_finalize_args(args: argparse.Namespace, known_source_names: set[str] | None = None) -> dict[str, int]:
    if args.frames_per_shard < 1:
        raise ValueError("--frames_per_shard must be >= 1")
    if args.repeat_min < 1:
        raise ValueError("--repeat-min must be >= 1")
    if args.repeat_max < args.repeat_min:
        raise ValueError("--repeat-max must be >= --repeat-min")
    fixed_repeat_by_source = dict(_parse_repeat_override(raw_value) for raw_value in args.fixed_repeat_source)
    if known_source_names is not None:
        unknown_repeat_sources = sorted(set(fixed_repeat_by_source) - set(known_source_names))
        if unknown_repeat_sources:
            raise ValueError(
                f"--fixed-repeat-source referenced unknown source(s): {unknown_repeat_sources}; known={sorted(known_source_names)}"
            )
    return fixed_repeat_by_source


def _load_stage_records_from_manifests(manifest_paths: list[str], staging_dir: Path) -> tuple[list[dict], set[str]]:
    records: list[dict] = []
    source_names: set[str] = set()
    seen_uids: set[str] = set()
    for raw_path in manifest_paths:
        manifest_path = Path(raw_path).resolve()
        for record in _read_jsonl_records(manifest_path):
            stage_uid = str(record.get("stage_uid") or "")
            if not stage_uid:
                raise ValueError(f"Stage manifest record is missing stage_uid: {manifest_path}")
            if stage_uid in seen_uids:
                raise ValueError(f"Duplicate stage_uid across stage manifests: {stage_uid}")
            seen_uids.add(stage_uid)
            source_name = str(record["source_name"])
            source_names.add(source_name)
            relpath = str(record["stage_relpath"])
            stage_tar = (staging_dir / relpath).resolve()
            if not stage_tar.is_file():
                raise FileNotFoundError(f"Staged episode tar not found: {stage_tar}")
            merged = dict(record)
            merged["stage_tar"] = str(stage_tar)
            records.append(merged)
    return records, source_names


def _write_report(path: str | None, payload: dict) -> None:
    if not path:
        return
    report_path = Path(path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _command_scan(args: argparse.Namespace) -> dict:
    sources, _resize_sources = _resolve_sources_from_args(args)
    episode_manifest_out = Path(args.episode_manifest_out).resolve()
    if episode_manifest_out.exists():
        raise RuntimeError(f"--episode_manifest_out must not already exist: {episode_manifest_out}")
    records, scan_report = scan_sources_to_episode_manifest(
        sources=sources,
        shard_start=int(args.shard_start),
        shard_end=args.shard_end,
        max_episodes=args.max_episodes,
    )
    _write_jsonl_records(episode_manifest_out, records)
    report = {
        "command": "scan",
        "sources": [spec.name for spec in sources],
        "episode_manifest_out": str(episode_manifest_out),
        "scan": scan_report,
    }
    _write_report(args.report_out, report)
    return report


def _command_manifest(args: argparse.Namespace) -> dict:
    episode_manifest_out = Path(args.episode_manifest_out).resolve()
    if episode_manifest_out.exists():
        raise RuntimeError(f"--episode_manifest_out must not already exist: {episode_manifest_out}")
    records, import_report = import_from_clip_manifest(
        clip_manifest_path=str(args.clip_manifest),
        source_name=None if args.source_name is None else str(args.source_name),
        has_depth=args.has_depth,
    )
    _write_jsonl_records(episode_manifest_out, records)
    report = {
        "command": "manifest",
        "episode_manifest_out": str(episode_manifest_out),
        "import": import_report,
    }
    _write_report(args.report_out, report)
    return report


def _command_reserve(args: argparse.Namespace) -> dict:
    episode_manifest_out = Path(args.episode_manifest_out).resolve()
    if episode_manifest_out.exists():
        raise RuntimeError(f"--episode_manifest_out must not already exist: {episode_manifest_out}")
    records, reserve_report = reserve_from_clip_manifest(
        clip_manifest_path=str(args.clip_manifest),
        source_name=str(args.source_name),
        drop_last_frame=bool(args.drop_last_frame),
    )
    _write_jsonl_records(episode_manifest_out, records)
    report = {
        "command": "reserve",
        "episode_manifest_out": str(episode_manifest_out),
        "reserve": reserve_report,
    }
    _write_report(args.report_out, report)
    return report


def _command_plan(args: argparse.Namespace) -> dict:
    fixed_repeat_by_source = _validate_common_finalize_args(args, known_source_names=None)
    mix_plan_out = Path(args.mix_plan_out).resolve()
    if mix_plan_out.exists():
        raise RuntimeError(f"--mix_plan_out must not already exist: {mix_plan_out}")
    episode_records, source_names = _load_unique_episode_records(list(args.episode_manifest))
    plan_records, plan_report = build_frozen_mix_plan(
        episode_records=episode_records,
        repeat_min=int(args.repeat_min),
        repeat_max=int(args.repeat_max),
        seed=int(args.seed),
        fixed_repeat_by_source=fixed_repeat_by_source,
        frames_per_shard=int(args.frames_per_shard),
    )
    _write_jsonl_records(mix_plan_out, plan_records)
    report = {
        "command": "plan",
        "sources": sorted(source_names),
        "episode_manifest": [str(Path(path).resolve()) for path in args.episode_manifest],
        "mix_plan_out": str(mix_plan_out),
        "repeat_min": int(args.repeat_min),
        "repeat_max": int(args.repeat_max),
        "frames_per_shard": int(args.frames_per_shard),
        "seed": int(args.seed),
        "fixed_repeat_by_source": {key: int(value) for key, value in sorted(fixed_repeat_by_source.items())},
        "plan": plan_report,
    }
    _write_report(args.report_out, report)
    return report


def _command_warm_index(args: argparse.Namespace) -> dict:
    if int(args.workers) < 1:
        raise ValueError("--workers must be >= 1")
    cache_root = _configure_source_index_cache_root(args.index_cache_dir)
    all_shard_paths = sorted(
        {
            str(Path(record["source_shard_path"]).resolve())
            for record in _iter_manifest_records(list(args.episode_manifest))
            if str(record.get("manifest_type")) == "wds_episode" and record.get("source_shard_path")
        }
    )
    if not all_shard_paths:
        raise RuntimeError("No source_shard_path entries found in the provided episode manifests")
    shard_paths = _select_shard_paths(
        all_shard_paths,
        shard_start=int(args.shard_start),
        shard_end=args.shard_end,
    )
    if not shard_paths:
        raise RuntimeError("No source shards matched the requested warm-index shard range")

    warmed: list[dict] = []
    if int(args.workers) <= 1:
        iterator = tqdm(shard_paths, desc="Warm source index")
        for shard_path in iterator:
            index = _get_shard_sample_index(shard_path)
            warmed.append(
                {
                    "shard_path": shard_path,
                    "episodes": len(index.get("episodes", {})),
                    "samples": len(index.get("samples", {})),
                    "cache_path": str(_shard_sample_index_cache_path(shard_path)),
                }
            )
    else:
        with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
            future_to_path = {executor.submit(_get_shard_sample_index, shard_path): shard_path for shard_path in shard_paths}
            progress = tqdm(total=len(future_to_path), desc=f"Warm source index x{int(args.workers)}")
            try:
                for future in as_completed(future_to_path):
                    shard_path = future_to_path[future]
                    index = future.result()
                    warmed.append(
                        {
                            "shard_path": shard_path,
                            "episodes": len(index.get("episodes", {})),
                            "samples": len(index.get("samples", {})),
                            "cache_path": str(_shard_sample_index_cache_path(shard_path)),
                        }
                    )
                    progress.update(1)
            finally:
                progress.close()

    warmed.sort(key=lambda item: item["shard_path"])
    report = {
        "command": "warm-index",
        "episode_manifest": [str(Path(path).resolve()) for path in args.episode_manifest],
        "index_cache_dir": str(cache_root),
        "workers": int(args.workers),
        "shard_start": int(args.shard_start),
        "shard_end": None if args.shard_end is None else int(args.shard_end),
        "source_shard_count_total": len(all_shard_paths),
        "source_shard_count": len(shard_paths),
        "items": warmed,
    }
    _write_report(args.report_out, report)
    return report


def _command_write(args: argparse.Namespace) -> dict:
    if int(args.workers) < 1:
        raise ValueError("--workers must be >= 1")
    cache_root = _configure_source_index_cache_root(args.index_cache_dir)
    exclude_sources = {str(item).strip() for item in (args.exclude_source or []) if str(item).strip()}
    available_lookup = _load_available_wds_episode_lookup(list(args.episode_manifest))
    all_plan_records = _load_mix_plan(str(args.mix_plan))
    selected_plan_records = _select_output_plan_shards(
        all_plan_records,
        shard_start=int(args.shard_start),
        shard_end=args.shard_end,
    )
    if not selected_plan_records:
        raise RuntimeError("No plan records matched the requested shard range")
    write_report = write_from_frozen_plan(
        plan_records=selected_plan_records,
        available_lookup=available_lookup,
        output_dir=Path(args.output_dir).resolve(),
        dataset_name=str(args.dataset_name),
        split=str(args.split),
        exclude_sources=exclude_sources,
        allow_missing_episodes=bool(args.allow_missing_episodes),
        resume=bool(args.resume),
        workers=int(args.workers),
    )
    report = {
        "command": "write",
        "mix_plan": str(Path(args.mix_plan).resolve()),
        "episode_manifest": [str(Path(path).resolve()) for path in args.episode_manifest],
        "output_dir": str(Path(args.output_dir).resolve()),
        "dataset_name": str(args.dataset_name),
        "split": str(args.split),
        "index_cache_dir": str(cache_root),
        "exclude_sources": sorted(exclude_sources),
        "allow_missing_episodes": bool(args.allow_missing_episodes),
        "resume": bool(args.resume),
        "workers": int(args.workers),
        "shard_start": int(args.shard_start),
        "shard_end": None if args.shard_end is None else int(args.shard_end),
        "write": write_report,
    }
    _write_report(args.report_out, report)
    return report


def _command_stage(args: argparse.Namespace) -> dict:
    _validate_common_stage_args(args)
    sources, resize_sources = _resolve_sources_from_args(args)
    staging_dir = Path(args.staging_dir).resolve()
    stage_manifest_out = Path(args.stage_manifest_out).resolve()
    if stage_manifest_out.exists():
        raise RuntimeError(f"--stage_manifest_out must not already exist: {stage_manifest_out}")

    stage_records, stage_report = stage_sources(
        sources=sources,
        staging_dir=staging_dir,
        stage_tag=str(args.stage_tag),
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        jpeg_quality=args.jpeg_quality,
        max_episodes=args.max_episodes,
        stage_manifest_out=stage_manifest_out,
        shard_start=int(args.shard_start),
        shard_end=args.shard_end,
    )
    report = {
        "command": "stage",
        "sources": [spec.name for spec in sources],
        "resize_sources": sorted(resize_sources),
        "resize_target": {
            "width": int(args.resize_width),
            "height": int(args.resize_height),
        },
        "stage_manifest_out": str(stage_manifest_out),
        "staging_dir": str(staging_dir),
        "stage": stage_report,
        "stage_records_written": len(stage_records),
    }
    _write_report(args.report_out, report)
    return report


def _command_finalize(args: argparse.Namespace) -> dict:
    staging_dir = Path(args.staging_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir == staging_dir:
        raise ValueError("--output_dir and --staging_dir must be different")
    stage_records, source_names = _load_stage_records_from_manifests(list(args.stage_manifest), staging_dir)
    fixed_repeat_by_source = _validate_common_finalize_args(args, known_source_names=source_names)
    if not stage_records:
        raise RuntimeError("No staged episodes found")

    plan, repeat_histogram, source_repeat_histogram = build_output_plan(
        stage_records,
        repeat_min=args.repeat_min,
        repeat_max=args.repeat_max,
        seed=args.seed,
        fixed_repeat_by_source=fixed_repeat_by_source,
    )
    write_report = write_mixed_dataset(
        plan=plan,
        output_dir=output_dir,
        frames_per_shard=args.frames_per_shard,
        dataset_name=args.dataset_name,
        split=args.split,
        seed=args.seed,
    )
    report = {
        "command": "finalize",
        "sources": sorted(source_names),
        "repeat_min": int(args.repeat_min),
        "repeat_max": int(args.repeat_max),
        "fixed_repeat_by_source": {key: int(value) for key, value in sorted(fixed_repeat_by_source.items())},
        "seed": int(args.seed),
        "dataset_name": args.dataset_name,
        "split": args.split,
        "frames_per_shard": int(args.frames_per_shard),
        "staging_dir": str(staging_dir),
        "output_dir": str(output_dir),
        "stage_manifest": [str(Path(path).resolve()) for path in args.stage_manifest],
        "staged_episode_count": len(stage_records),
        "repeat_histogram": {str(key): int(value) for key, value in sorted(repeat_histogram.items())},
        "source_repeat_histogram": {
            source_name: {str(key): int(value) for key, value in sorted(hist.items())}
            for source_name, hist in sorted(source_repeat_histogram.items())
        },
        "write": write_report,
    }
    _write_report(args.report_out, report)
    if args.cleanup_staging:
        shutil.rmtree(staging_dir)
    return report


def _command_all(args: argparse.Namespace) -> dict:
    _validate_common_stage_args(args)
    sources, resize_sources = _resolve_sources_from_args(args)
    fixed_repeat_by_source = _validate_common_finalize_args(
        args,
        known_source_names={spec.name for spec in sources},
    )
    output_dir = Path(args.output_dir).resolve()
    staging_dir = Path(args.staging_dir).resolve()
    if output_dir == staging_dir:
        raise ValueError("--output_dir and --staging_dir must be different")
    stage_records, stage_report = stage_sources(
        sources=sources,
        staging_dir=staging_dir,
        stage_tag="all",
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        jpeg_quality=args.jpeg_quality,
        max_episodes=args.max_episodes,
        stage_manifest_out=None,
        shard_start=0,
        shard_end=None,
    )
    if not stage_records:
        raise RuntimeError("No staged episodes found")
    for record in stage_records:
        record["stage_tar"] = str((staging_dir / record["stage_relpath"]).resolve())

    plan, repeat_histogram, source_repeat_histogram = build_output_plan(
        stage_records,
        repeat_min=args.repeat_min,
        repeat_max=args.repeat_max,
        seed=args.seed,
        fixed_repeat_by_source=fixed_repeat_by_source,
    )
    write_report = write_mixed_dataset(
        plan=plan,
        output_dir=output_dir,
        frames_per_shard=args.frames_per_shard,
        dataset_name=args.dataset_name,
        split=args.split,
        seed=args.seed,
    )
    report = {
        "command": "all",
        "sources": [spec.name for spec in sources],
        "resize_sources": sorted(resize_sources),
        "resize_target": {
            "width": int(args.resize_width),
            "height": int(args.resize_height),
        },
        "repeat_min": int(args.repeat_min),
        "repeat_max": int(args.repeat_max),
        "fixed_repeat_by_source": {key: int(value) for key, value in sorted(fixed_repeat_by_source.items())},
        "seed": int(args.seed),
        "dataset_name": args.dataset_name,
        "split": args.split,
        "frames_per_shard": int(args.frames_per_shard),
        "staging_dir": str(staging_dir),
        "output_dir": str(output_dir),
        "stage": stage_report,
        "repeat_histogram": {str(key): int(value) for key, value in sorted(repeat_histogram.items())},
        "source_repeat_histogram": {
            source_name: {str(key): int(value) for key, value in sorted(hist.items())}
            for source_name, hist in sorted(source_repeat_histogram.items())
        },
        "write": write_report,
    }
    _write_report(args.report_out, report)
    if args.cleanup_staging:
        shutil.rmtree(staging_dir)
    return report


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] not in {"scan", "manifest", "reserve", "plan", "warm-index", "write", "stage", "finalize", "all"}:
        argv = ["all", *argv]
    args = build_parser().parse_args(argv)

    if args.command == "scan":
        report = _command_scan(args)
    elif args.command == "manifest":
        report = _command_manifest(args)
    elif args.command == "reserve":
        report = _command_reserve(args)
    elif args.command == "plan":
        report = _command_plan(args)
    elif args.command == "warm-index":
        report = _command_warm_index(args)
    elif args.command == "write":
        report = _command_write(args)
    elif args.command == "stage":
        report = _command_stage(args)
    elif args.command == "finalize":
        report = _command_finalize(args)
    else:
        report = _command_all(args)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
