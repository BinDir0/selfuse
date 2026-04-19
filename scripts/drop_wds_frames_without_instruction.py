#!/usr/bin/env python3
"""Rewrite WebDataset shards while dropping whole clips with bad instruction frames."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from multiprocessing import get_context
from pathlib import Path

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    class tqdm:  # noqa: N801
        def __init__(self, iterable=None, total=None, desc=None, unit=None):
            self.iterable = iterable
            self.total = total if total is not None else (len(iterable) if iterable is not None else None)
            self.desc = desc or "Progress"
            self.unit = unit or "item"
            self.count = 0

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            for item in self.iterable:
                yield item
                self.count += 1
                self._print()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n=1):
            self.count += int(n)
            self._print()

        def set_postfix(self, refresh=False, **kwargs):
            self.postfix = kwargs
            self._print()

        def _print(self):
            total = "?" if self.total is None else str(self.total)
            postfix = getattr(self, "postfix", {})
            extra = (" " + " ".join(f"{key}={value}" for key, value in postfix.items())) if postfix else ""
            print(f"{self.desc}: {self.count}/{total} {self.unit}{extra}")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.webdataset_rewriter import iter_shard_paths, iter_shard_samples, write_sample_to_tar  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drop whole WDS clips containing missing/empty instruction frames")
    parser.add_argument("--source_shard_dir", required=True, help="Source directory containing shard tar files")
    parser.add_argument("--output_dir", required=True, help="Output directory for rewritten shard tar files")
    parser.add_argument("--report_out", default=None, help="Optional JSON report path")
    parser.add_argument("--start_shard", type=int, default=0, help="Start shard index in sorted order (inclusive)")
    parser.add_argument("--end_shard", type=int, default=None, help="End shard index in sorted order (exclusive)")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)), help="Shard-level workers")
    parser.add_argument("--keep_invalid_meta", action="store_true", help="Keep frames whose meta.json is invalid")
    parser.add_argument("--keep_missing_meta", action="store_true", help="Keep frames missing meta.json")
    parser.add_argument(
        "--drop_instruction_num_mismatch",
        action="store_true",
        help="Also drop whole clips where any frame has instruction_num > non-empty instruction slots",
    )
    return parser


def parse_instruction_entries(meta: dict | None) -> tuple[int, list[str], list[object]]:
    if not isinstance(meta, dict):
        return 0, [], []
    raw_instruction_num = meta.get("instruction_num", 0)
    try:
        instruction_num = max(0, int(raw_instruction_num))
    except Exception:
        instruction_num = 0
    raw_instruction = meta.get("instruction", [])
    if isinstance(raw_instruction, str):
        slots = [raw_instruction]
    elif isinstance(raw_instruction, (list, tuple)):
        slots = list(raw_instruction)
    else:
        slots = []
    effective_slots = slots[:instruction_num]
    cleaned = [str(item).strip() for item in effective_slots if str(item).strip()]
    return instruction_num, cleaned, effective_slots


def select_shards(source_shard_dir: str, start_shard: int, end_shard: int | None) -> tuple[list[str], int, int]:
    shard_paths = list(iter_shard_paths(source_shard_dir))
    if not shard_paths:
        raise RuntimeError(f"No shard tar files found in {source_shard_dir}")
    total_shards = len(shard_paths)
    if start_shard < 0 or start_shard >= total_shards:
        raise ValueError(f"--start_shard {start_shard} is out of range [0, {total_shards})")
    resolved_end = total_shards if end_shard is None else int(end_shard)
    if resolved_end < start_shard or resolved_end > total_shards:
        raise ValueError(f"--end_shard {resolved_end} is out of range [{start_shard}, {total_shards}]")
    selected = shard_paths[start_shard:resolved_end]
    if not selected:
        raise RuntimeError(f"No shards selected in range [{start_shard}, {resolved_end}) from {source_shard_dir}")
    return selected, total_shards, resolved_end


def validate_io_dirs(source_dir: Path, output_dir: Path) -> None:
    source_resolved = source_dir.resolve()
    output_resolved = output_dir.resolve()
    if source_resolved == output_resolved:
        raise ValueError("--output_dir must be different from --source_shard_dir")
    try:
        output_resolved.relative_to(source_resolved)
    except ValueError:
        ...
    else:
        raise ValueError("--output_dir must not be inside --source_shard_dir")
    try:
        source_resolved.relative_to(output_resolved)
    except ValueError:
        ...
    else:
        raise ValueError("--source_shard_dir must not be inside --output_dir")


def should_drop_sample(meta_bytes, args_dict: dict) -> tuple[bool, str | None]:
    if meta_bytes is None:
        return (not args_dict["keep_missing_meta"]), "missing_meta"
    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
    except Exception:
        return (not args_dict["keep_invalid_meta"]), "invalid_meta"

    instruction_num, instructions, effective_slots = parse_instruction_entries(meta)
    if instruction_num <= 0:
        return True, "missing_instruction"
    if not instructions:
        return True, "empty_instruction"
    if args_dict["drop_instruction_num_mismatch"] and len(instructions) != min(instruction_num, len(effective_slots)):
        return True, "instruction_num_mismatch"
    return False, None


def sample_clip_id(sample: dict) -> str:
    meta_bytes = sample.get("meta_bytes")
    if meta_bytes is not None:
        try:
            meta = json.loads(meta_bytes.decode("utf-8"))
        except Exception:
            meta = None
        if isinstance(meta, dict):
            clip_id = meta.get("clip_id")
            if clip_id:
                return str(clip_id)
    return sample["key"].rsplit("_f", 1)[0]


def rewrite_shard(task: tuple[str, str, dict]) -> dict:
    shard_path, output_dir, args_dict = task
    shard_name = os.path.basename(shard_path)
    output_path = os.path.join(output_dir, shard_name)
    tmp_path = f"{output_path}.tmp"
    result = {
        "shard_name": shard_name,
        "output_path": output_path,
        "samples_total": 0,
        "clips_total": 0,
        "clips_written": 0,
        "clips_dropped": 0,
        "frames_written": 0,
        "frames_dropped": 0,
        "drop_reasons": {
            "missing_meta": 0,
            "invalid_meta": 0,
            "missing_instruction": 0,
            "empty_instruction": 0,
            "instruction_num_mismatch": 0,
        },
    }

    tar_writer = None

    current_clip_id = None
    current_samples = []
    current_drop = False
    current_drop_reasons = set()

    def flush_current_clip() -> None:
        nonlocal tar_writer, current_clip_id, current_samples, current_drop, current_drop_reasons
        if current_clip_id is None:
            return
        result["clips_total"] += 1
        if current_drop:
            result["clips_dropped"] += 1
            result["frames_dropped"] += len(current_samples)
            for reason in current_drop_reasons:
                result["drop_reasons"][reason] += len(current_samples)
        else:
            if tar_writer is None:
                os.makedirs(output_dir, exist_ok=True)
                tar_writer = tarfile.open(tmp_path, "w")
            for buffered_sample in current_samples:
                write_sample_to_tar(
                    tar_writer,
                    buffered_sample["key"],
                    buffered_sample["image_bytes"],
                    buffered_sample["lowdim_bytes"],
                    buffered_sample["meta_bytes"],
                    mano_bytes=buffered_sample.get("mano_bytes"),
                )
                result["frames_written"] += 1
            result["clips_written"] += 1
        current_clip_id = None
        current_samples = []
        current_drop = False
        current_drop_reasons = set()

    try:
        for sample in iter_shard_samples(shard_path):
            result["samples_total"] += 1
            clip_id = sample_clip_id(sample)
            if current_clip_id is None:
                current_clip_id = clip_id
            elif clip_id != current_clip_id:
                flush_current_clip()
                current_clip_id = clip_id

            drop, reason = should_drop_sample(sample.get("meta_bytes"), args_dict)
            if drop:
                if reason:
                    current_drop_reasons.add(reason)
                current_drop = True
            current_samples.append(sample)
        flush_current_clip()
    except Exception:
        if tar_writer is not None:
            tar_writer.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if tar_writer is not None:
        tar_writer.close()
        os.replace(tmp_path, output_path)
    elif os.path.exists(tmp_path):
        os.remove(tmp_path)

    return result


def merge_results(report: dict, shard_result: dict) -> None:
    report["summary"]["samples_total"] += int(shard_result["samples_total"])
    report["summary"]["clips_total"] += int(shard_result["clips_total"])
    report["summary"]["clips_written"] += int(shard_result["clips_written"])
    report["summary"]["clips_dropped"] += int(shard_result["clips_dropped"])
    report["summary"]["frames_written"] += int(shard_result["frames_written"])
    report["summary"]["frames_dropped"] += int(shard_result["frames_dropped"])
    for key, value in shard_result["drop_reasons"].items():
        report["drop_reasons"][key] += int(value)


def main() -> None:
    args = build_parser().parse_args()
    source_dir = Path(args.source_shard_dir)
    output_dir = Path(args.output_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source shard dir not found: {source_dir}")
    validate_io_dirs(source_dir, output_dir)

    shard_paths, total_shards, resolved_end = select_shards(str(source_dir), args.start_shard, args.end_shard)
    worker_args = {
        "keep_invalid_meta": bool(args.keep_invalid_meta),
        "keep_missing_meta": bool(args.keep_missing_meta),
        "drop_instruction_num_mismatch": bool(args.drop_instruction_num_mismatch),
    }
    report = {
        "source_shard_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "shard_selection": {
            "start_shard": int(args.start_shard),
            "end_shard": int(resolved_end),
            "selected_shards": len(shard_paths),
            "total_shards_available": int(total_shards),
        },
        "summary": {
            "samples_total": 0,
            "clips_total": 0,
            "clips_written": 0,
            "clips_dropped": 0,
            "frames_written": 0,
            "frames_dropped": 0,
        },
        "drop_reasons": {
            "missing_meta": 0,
            "invalid_meta": 0,
            "missing_instruction": 0,
            "empty_instruction": 0,
            "instruction_num_mismatch": 0,
        },
    }

    tasks = [(shard_path, str(output_dir), worker_args) for shard_path in shard_paths]
    pool = None
    if int(args.workers) <= 1:
        iterator = (rewrite_shard(task) for task in tasks)
    else:
        mp_context = get_context()
        pool = mp_context.Pool(int(args.workers))
        iterator = pool.imap_unordered(rewrite_shard, tasks, chunksize=1)

    try:
        with tqdm(total=len(tasks), desc="Rewrite shards", unit="shard") as progress:
            for shard_result in iterator:
                merge_results(report, shard_result)
                progress.update(1)
                progress.set_postfix(
                    clips_dropped=int(report["summary"]["clips_dropped"]),
                    dropped=int(report["summary"]["frames_dropped"]),
                    written=int(report["summary"]["frames_written"]),
                    refresh=False,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
