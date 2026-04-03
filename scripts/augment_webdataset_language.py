#!/usr/bin/env python3
"""Rewrite existing WebDataset shards with instruction metadata."""

import argparse
import os
import sys
import tarfile
from multiprocessing import get_context
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.webdataset_annotation import (  # noqa: E402
    DEFAULT_ANNOTATION_SUFFIX,
    load_episode_instruction,
)
from lib.pipeline.exporters.webdataset_discovery import (  # noqa: E402
    discover_episode_stats,
    discover_episodes,
    parse_factory_range,
    repeat_episode_stats,
)
from lib.pipeline.exporters.webdataset_rewriter import (  # noqa: E402
    build_updated_meta,
    iter_shard_paths,
    iter_shard_samples,
    parse_episode_index,
    validate_sample_record,
    write_sample_to_tar,
)

_worker_episode_lookup = None
_worker_output_dir = None


def build_parser():
    parser = argparse.ArgumentParser(description="Rewrite WebDataset shards with instruction metadata")
    parser.add_argument("--source_shard_dir", required=True, help="Directory containing source shard tar files")
    parser.add_argument("--output_dir", required=True, help="Directory for rewritten shard tar files")
    parser.add_argument("--input_dir", required=True, help="BuildAI processed root used to create the shards")
    parser.add_argument("--factory_range", default=None, help="Inclusive factory range like 1-50 for 10K BuildAI layout")
    parser.add_argument("--episode_list", default=None, help="Optional episode list used during the original build")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes like the original build")
    parser.add_argument("--repeat_episodes", type=int, default=1, help="Repeat count used during the original build")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of shard rewrite workers",
    )
    parser.add_argument("--rescan", action="store_true", help="Force rescan of frame indexes when rebuilding episode order")
    parser.add_argument(
        "--annotation_suffix",
        default=DEFAULT_ANNOTATION_SUFFIX,
        help="Suffix for per-episode annotation JSON files",
    )
    parser.add_argument(
        "--allow_missing_annotation",
        action="store_true",
        help="Keep samples with empty instruction when annotation is missing or invalid",
    )
    return parser


def build_episode_lookup(args):
    cache_file = os.path.join(args.input_dir, "_vla_episodes_cache.json")
    episodes = discover_episodes(
        args.input_dir,
        episode_list=args.episode_list,
        max_episodes=args.max_episodes,
        cache_file=cache_file,
        factory_range=args.factory_range,
    )
    if not episodes:
        raise RuntimeError("No episodes found while rebuilding episode index mapping")

    episode_stats = discover_episode_stats(episodes, rescan_frame_index=args.rescan)
    if not episode_stats:
        raise RuntimeError("No valid episodes with extracted frames found while rebuilding episode index mapping")

    episode_stats = repeat_episode_stats(episode_stats, args.repeat_episodes)
    lookup = {}
    stats = {
        "episodes_total": len(episode_stats),
        "episodes_writable": 0,
        "episodes_skipped": 0,
        "missing_annotation": 0,
        "invalid_json": 0,
        "invalid_status": 0,
        "empty_instruction": 0,
    }

    for ep in episode_stats:
        instruction, error_code, annotation_path = load_episode_instruction(
            ep,
            annotation_suffix=args.annotation_suffix,
        )
        should_skip = instruction is None and not args.allow_missing_annotation
        if instruction is None:
            stats[error_code] = stats.get(error_code, 0) + 1
            if should_skip:
                stats["episodes_skipped"] += 1
            instruction = []
        else:
            stats["episodes_writable"] += 1

        lookup[ep["episode_index"]] = {
            "episode_id": ep["episode_id"],
            "instruction": instruction,
            "skip": should_skip,
            "annotation_path": annotation_path,
        }

    if args.allow_missing_annotation:
        stats["episodes_writable"] = len(episode_stats)

    return lookup, stats


def rewrite_shard(shard_path, output_dir, episode_lookup):
    output_path = os.path.join(output_dir, os.path.basename(shard_path))
    tmp_path = output_path + ".tmp"
    frames_written = 0
    skipped_frames = 0
    skipped_episodes = set()
    written_episodes = set()
    tar_writer = None

    try:
        for sample in iter_shard_samples(shard_path):
            validate_sample_record(sample)
            episode_index = parse_episode_index(sample["key"])
            episode_info = episode_lookup.get(episode_index)
            if episode_info is None:
                raise RuntimeError(
                    f"Sample {sample['key']} references episode_index={episode_index}, "
                    "but the rebuilt episode mapping does not contain it"
                )

            if episode_info["skip"]:
                skipped_frames += 1
                skipped_episodes.add(episode_index)
                continue

            updated_meta = build_updated_meta(sample["meta_bytes"], episode_info["instruction"])
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
            )
            frames_written += 1
            written_episodes.add(episode_index)
    except Exception:
        if tar_writer is not None:
            tar_writer.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if tar_writer is not None:
        tar_writer.close()

    if frames_written == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    else:
        os.replace(tmp_path, output_path)

    return {
        "output_path": output_path,
        "frames_written": frames_written,
        "skipped_frames": skipped_frames,
        "written_episodes": len(written_episodes),
        "skipped_episodes": len(skipped_episodes),
        "shard_written": 1 if frames_written > 0 else 0,
    }


def _worker_init(output_dir, episode_lookup):
    global _worker_episode_lookup, _worker_output_dir

    _worker_episode_lookup = episode_lookup
    _worker_output_dir = output_dir


def _worker_rewrite_shard(shard_path):
    return rewrite_shard(shard_path, _worker_output_dir, _worker_episode_lookup)


def print_lookup_summary(stats):
    print(
        "Episode annotation map:"
        f" total={stats['episodes_total']}"
        f" writable={stats['episodes_writable']}"
        f" skipped={stats['episodes_skipped']}"
        f" missing={stats['missing_annotation']}"
        f" invalid_json={stats['invalid_json']}"
        f" invalid_status={stats['invalid_status']}"
        f" empty_instruction={stats['empty_instruction']}"
    )


def print_final_summary(output_dir, totals):
    print("\nDone!")
    print(f"  Output: {output_dir}")
    print(f"  Shards written: {totals['shards_written']}")
    print(f"  Frames written: {totals['frames_written']}")
    print(f"  Frames skipped: {totals['skipped_frames']}")
    print(f"  Episodes written: {totals['written_episodes']}")
    print(f"  Episodes skipped: {totals['skipped_episodes']}")


def run_rewrite_workers(shard_paths, output_dir, episode_lookup, workers):
    totals = {
        "shards_written": 0,
        "frames_written": 0,
        "skipped_frames": 0,
        "written_episodes": 0,
        "skipped_episodes": 0,
    }

    if workers <= 1:
        result_iter = (rewrite_shard(shard_path, output_dir, episode_lookup) for shard_path in shard_paths)
    else:
        mp_context = get_context()
        with mp_context.Pool(
            workers,
            initializer=_worker_init,
            initargs=(output_dir, episode_lookup),
        ) as pool:
            result_iter = pool.imap_unordered(_worker_rewrite_shard, shard_paths, chunksize=1)
            for result in tqdm(result_iter, total=len(shard_paths), desc="Rewrite shards"):
                totals["shards_written"] += result["shard_written"]
                totals["frames_written"] += result["frames_written"]
                totals["skipped_frames"] += result["skipped_frames"]
                totals["written_episodes"] += result["written_episodes"]
                totals["skipped_episodes"] += result["skipped_episodes"]
        return totals

    for result in tqdm(result_iter, total=len(shard_paths), desc="Rewrite shards"):
        totals["shards_written"] += result["shard_written"]
        totals["frames_written"] += result["frames_written"]
        totals["skipped_frames"] += result["skipped_frames"]
        totals["written_episodes"] += result["written_episodes"]
        totals["skipped_episodes"] += result["skipped_episodes"]

    return totals


def main():
    args = build_parser().parse_args()
    parse_factory_range(args.factory_range)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    os.makedirs(args.output_dir, exist_ok=True)

    episode_lookup, lookup_stats = build_episode_lookup(args)
    print_lookup_summary(lookup_stats)

    shard_paths = list(iter_shard_paths(args.source_shard_dir))
    if not shard_paths:
        raise RuntimeError(f"No .tar shards found in {args.source_shard_dir}")

    print(f"Rewriting {len(shard_paths)} shards with {args.workers} worker(s)...")
    totals = run_rewrite_workers(shard_paths, args.output_dir, episode_lookup, args.workers)

    print_final_summary(args.output_dir, totals)


if __name__ == "__main__":
    main()
