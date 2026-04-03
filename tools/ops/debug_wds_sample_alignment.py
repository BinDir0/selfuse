#!/usr/bin/env python3
"""Diagnose whether a WebDataset sample is aligned with its source episode."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.clip_manifest import load_clip_manifest
from lib.pipeline.exporters.manifest_vla import load_descriptor_episode_features
from lib.pipeline.exporters.webdataset_discovery import discover_episodes, load_episode_stats
from lib.pipeline.exporters.webdataset_features import build_mano_models, load_episode_features
from lib.pipeline.exporters.webdataset_rewriter import iter_shard_paths, iter_shard_samples, validate_sample_record
from lib.pipeline.frame_sources import read_frame_bytes_from_descriptor
from lib.pipeline.quality_metrics import decode_lowdim, parse_frame_index


def build_parser():
    parser = argparse.ArgumentParser(description="Debug image/lowdim alignment for one WebDataset sample")
    parser.add_argument("--input", required=True, help="Shard tar file or directory containing shard tar files")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sample-key", default=None, help="Exact sample key to inspect")
    group.add_argument("--sample-index", type=int, default=0, help="Sequential sample index to inspect when --sample-key is omitted")
    parser.add_argument("--processed-root", default=None, help="Processed episode root for legacy builders that only store episode_index")
    parser.add_argument("--descriptor-manifest", default=None, help="Clip manifest for manifest-based builders")
    parser.add_argument("--mano-dir", default=None, help="Optional MANO model directory override for lowdim recomputation")
    parser.add_argument("--device", default="cpu", help="Torch device for MANO forward, e.g. cpu or cuda:0")
    parser.add_argument("--neighbor-window", type=int, default=5, help="Search +/-N source frames for the best lowdim/image match")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser


def md5_bytes(payload: bytes | None) -> str | None:
    if payload is None:
        return None
    return hashlib.md5(payload).hexdigest()


def resolve_tar_paths(input_path: str) -> list[str]:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        return [str(path)]
    return list(iter_shard_paths(str(path)))


def find_sample(tar_paths: list[str], *, sample_key: str | None, sample_index: int) -> tuple[dict, str, int]:
    seen = 0
    for shard_path in tar_paths:
        for sample in iter_shard_samples(shard_path):
            if sample_key is not None:
                if sample["key"] != sample_key:
                    continue
            else:
                if seen != sample_index:
                    seen += 1
                    continue
            validate_sample_record(sample)
            return sample, shard_path, seen
            seen += 1
    if sample_key is not None:
        raise KeyError(f"Sample key not found: {sample_key}")
    raise IndexError(f"Sample index out of range: {sample_index}")


def _build_legacy_episode_lookup(processed_root: str) -> dict[int, dict]:
    episodes = discover_episodes(processed_root, require_world_res=True)
    lookup = {}
    for ep in episodes:
        stats = load_episode_stats(ep, rescan_frame_index=False)
        if stats is not None:
            lookup[int(stats["episode_index"])] = stats
    return lookup


def _resolve_source_episode(meta: dict, *, processed_root: str | None, descriptor_manifest: str | None) -> tuple[str, dict]:
    if meta.get("clip_id") and descriptor_manifest:
        records = load_clip_manifest(descriptor_manifest)
        by_clip = {record.clip_id: record for record in records}
        record = by_clip.get(str(meta["clip_id"]))
        if record is None:
            raise KeyError(f"clip_id not found in descriptor manifest: {meta['clip_id']}")
        return "manifest", {
            "clip_id": record.clip_id,
            "seq_folder": record.descriptor.seq_folder,
            "descriptor": record.descriptor,
            "episode_id": record.clip_id,
            "num_valid_frames": int(record.metadata.get("num_valid_frames", 0)) if record.metadata else None,
        }

    if processed_root is None:
        raise ValueError("Need --processed-root for legacy samples without clip manifest support")

    if "episode_index" not in meta:
        raise KeyError("meta.json does not contain episode_index")
    lookup = _build_legacy_episode_lookup(processed_root)
    episode = lookup.get(int(meta["episode_index"]))
    if episode is None:
        raise KeyError(f"episode_index not found under processed_root: {meta['episode_index']}")
    return "legacy", episode


def _load_recomputed_episode(source_kind: str, episode_info: dict, *, device: str, mano_dir: str | None):
    import torch

    torch_device = torch.device(device)
    mano_right, mano_left = build_mano_models(torch_device, mano_dir=mano_dir)
    mano_right.eval()
    mano_left.eval()

    if source_kind == "legacy":
        return load_episode_features(
            episode_info,
            mano_right,
            mano_left,
            torch_device,
            rescan_frame_index=False,
            feature_cache_dir=None,
            require_cache=False,
            mano_dir=mano_dir,
        )

    feature_request = {
        "seq_folder": episode_info["seq_folder"],
        "episode_id": episode_info["episode_id"],
        "num_valid_frames": episode_info.get("num_valid_frames"),
    }
    return load_descriptor_episode_features(
        feature_request,
        mano_right,
        mano_left,
        torch_device,
        feature_cache_dir=None,
        mano_dir=mano_dir,
        source_fps=5.0,
        target_fps=5.0,
        interpolate_labels=False,
    )


def _load_source_image_bytes(source_kind: str, episode_info: dict, frame_idx: int) -> bytes | None:
    if source_kind == "legacy":
        frame_path = episode_info["frame_index"].get(frame_idx)
        if frame_path is None or not os.path.exists(frame_path):
            return None
        with open(frame_path, "rb") as handle:
            return handle.read()
    return read_frame_bytes_from_descriptor(episode_info["descriptor"], frame_idx)


def compare_sample(sample: dict, meta: dict, source_kind: str, episode_info: dict, recomputed: dict, neighbor_window: int) -> dict:
    wds_frame_idx = parse_frame_index(sample["key"])
    wds_lowdim = decode_lowdim(sample["lowdim_bytes"])
    image_hash = md5_bytes(sample["image_bytes"])

    source_lowdim_all = np.asarray(recomputed["lowdim_all"], dtype=np.float32)
    source_presence = np.asarray(recomputed["presence_per_frame"])
    source_frame_count = int(source_lowdim_all.shape[0])
    if wds_frame_idx >= source_frame_count:
        raise IndexError(
            f"WDS frame_idx={wds_frame_idx} exceeds recomputed frame count {source_frame_count}"
        )

    direct_lowdim = source_lowdim_all[wds_frame_idx]
    direct_lowdim_max_abs = float(np.max(np.abs(direct_lowdim - wds_lowdim)))

    lowdim_neighbors = []
    image_neighbors = []
    best_lowdim = None
    best_image = None
    for source_frame_idx in range(max(0, wds_frame_idx - neighbor_window), min(source_frame_count, wds_frame_idx + neighbor_window + 1)):
        candidate_lowdim = source_lowdim_all[source_frame_idx]
        lowdim_max_abs = float(np.max(np.abs(candidate_lowdim - wds_lowdim)))
        lowdim_record = {
            "source_frame_idx": source_frame_idx,
            "lowdim_max_abs_diff": lowdim_max_abs,
            "presence": int(source_presence[source_frame_idx]),
        }
        lowdim_neighbors.append(lowdim_record)
        if best_lowdim is None or lowdim_max_abs < best_lowdim["lowdim_max_abs_diff"]:
            best_lowdim = lowdim_record

        source_image = _load_source_image_bytes(source_kind, episode_info, source_frame_idx)
        source_hash = md5_bytes(source_image)
        image_record = {
            "source_frame_idx": source_frame_idx,
            "image_md5": source_hash,
            "image_exact_match": bool(source_hash is not None and source_hash == image_hash),
        }
        image_neighbors.append(image_record)
        if image_record["image_exact_match"] and best_image is None:
            best_image = image_record

    return {
        "sample_key": sample["key"],
        "wds_frame_idx": wds_frame_idx,
        "wds_image_md5": image_hash,
        "wds_presence": None if meta is None else meta.get("presence"),
        "direct_frame_check": {
            "source_frame_idx": wds_frame_idx,
            "lowdim_max_abs_diff": direct_lowdim_max_abs,
            "presence": int(source_presence[wds_frame_idx]),
            "image_exact_match": any(item["source_frame_idx"] == wds_frame_idx and item["image_exact_match"] for item in image_neighbors),
        },
        "best_lowdim_match": best_lowdim,
        "best_image_match": best_image,
        "lowdim_neighbor_scan": lowdim_neighbors,
        "image_neighbor_scan": image_neighbors,
    }


def infer_alignment_issue(result: dict) -> str:
    direct = result["direct_frame_check"]
    best_lowdim = result["best_lowdim_match"]
    best_image = result["best_image_match"]

    image_ok = bool(direct["image_exact_match"])
    lowdim_ok = bool(direct["lowdim_max_abs_diff"] < 1e-5)

    if image_ok and lowdim_ok:
        return "sample appears aligned: image and lowdim both match the same source frame"
    if image_ok and best_lowdim and best_lowdim["source_frame_idx"] != direct["source_frame_idx"]:
        return (
            f"image matches source frame {direct['source_frame_idx']}, but lowdim matches source frame "
            f"{best_lowdim['source_frame_idx']} better; lowdim is likely shifted"
        )
    if lowdim_ok and best_image and best_image["source_frame_idx"] != direct["source_frame_idx"]:
        return (
            f"lowdim matches source frame {direct['source_frame_idx']}, but image bytes match source frame "
            f"{best_image['source_frame_idx']} better; image is likely shifted"
        )
    if not image_ok and not lowdim_ok:
        return "neither image nor lowdim matches the expected frame directly; investigate source frame mapping and build semantics"
    return "sample mismatch is present, but the cause is ambiguous from this local window"


def main():
    args = build_parser().parse_args()
    tar_paths = resolve_tar_paths(args.input)
    if not tar_paths:
        raise SystemExit("No shard tar files found")

    sample, shard_path, sample_position = find_sample(
        tar_paths,
        sample_key=args.sample_key,
        sample_index=args.sample_index,
    )
    meta = json.loads(sample["meta_bytes"].decode("utf-8"))
    source_kind, episode_info = _resolve_source_episode(
        meta,
        processed_root=args.processed_root,
        descriptor_manifest=args.descriptor_manifest,
    )
    recomputed = _load_recomputed_episode(
        source_kind,
        episode_info,
        device=args.device,
        mano_dir=args.mano_dir,
    )
    if recomputed is None:
        raise RuntimeError("Failed to recompute episode features")

    comparison = compare_sample(
        sample,
        meta,
        source_kind,
        episode_info,
        recomputed,
        neighbor_window=args.neighbor_window,
    )
    report = {
        "sample_position": sample_position,
        "shard_path": shard_path,
        "source_kind": source_kind,
        "source_episode": {
            "episode_id": episode_info.get("episode_id"),
            "crop_dir": episode_info.get("crop_dir"),
            "seq_folder": episode_info.get("seq_folder"),
            "clip_id": episode_info.get("clip_id"),
        },
        "meta": meta,
        "comparison": comparison,
        "inference": infer_alignment_issue(comparison),
    }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        Path(args.output).expanduser().resolve().write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
