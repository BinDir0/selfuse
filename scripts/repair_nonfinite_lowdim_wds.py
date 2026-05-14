#!/usr/bin/env python3
"""Rewrite WDS shards, replacing non-finite lowdim values with random finite values.

Input is the full bad-keys JSONL from filter_and_check_datasets.py. Only records
with reason == NonFiniteDataError are used. The script rewrites the matching
shards already present in --dst-dir, replacing NaN/+Inf/-Inf entries in
<key>.lowdim.npy with deterministic random values.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import io
import json
import os
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    class _NoOpTqdm:
        def __init__(self, iterable=None, **_kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable if self.iterable is not None else ())

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

        def update(self, _n=1):
            return None

    def tqdm(iterable=None, **_kwargs):
        return _NoOpTqdm(iterable)


REPAIR_REASON = "NonFiniteDataError"
REPAIR_FLAG = "nonfinite_lowdim_random_repair"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace non-finite lowdim values for WDS samples listed in a bad-keys JSONL.",
    )
    parser.add_argument(
        "--bad-keys",
        required=True,
        help="JSONL produced by filter_and_check_datasets.py --bad-keys-output.",
    )
    parser.add_argument(
        "--dst-dir",
        required=True,
        help="Dataset directory containing the shards to rewrite.",
    )
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    parser.add_argument("--seed", default="nonfinite-lowdim-repair-v1", help="Deterministic random seed string.")
    parser.add_argument("--replacement-min", type=float, default=-1.0, help="Minimum random replacement value.")
    parser.add_argument("--replacement-max", type=float, default=1.0, help="Maximum random replacement value.")
    parser.add_argument("--dry-run", action="store_true", help="Report planned rewrites without writing shards.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing destination shards.")
    parser.add_argument("--workers", type=int, default=1, help="Number of shards to rewrite in parallel.")
    parser.add_argument(
        "--executor",
        choices=("process", "thread"),
        default="process",
        help="Parallel executor to use when --workers > 1.",
    )
    return parser.parse_args()


def stable_u64(seed: str, sample_key: str) -> int:
    digest = hashlib.sha256(f"{seed}:nonfinite_lowdim:{sample_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def encode_npy(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return buffer.getvalue()


def decode_npy(payload: bytes) -> np.ndarray:
    return np.load(io.BytesIO(payload), allow_pickle=False)


def replace_nonfinite_values(
    lowdim: np.ndarray,
    *,
    seed: str,
    sample_key: str,
    replacement_min: float,
    replacement_max: float,
) -> tuple[np.ndarray, int]:
    repaired = np.asarray(lowdim, dtype=np.float32).copy()
    mask = ~np.isfinite(repaired)
    count = int(mask.sum())
    if count == 0:
        return repaired, 0
    rng = np.random.default_rng(stable_u64(seed, sample_key))
    repaired[mask] = rng.uniform(float(replacement_min), float(replacement_max), size=count).astype(np.float32)
    return repaired, count


def update_meta(meta_payload: bytes, *, values_replaced: int, replacement_min: float, replacement_max: float) -> bytes:
    meta = json.loads(meta_payload.decode("utf-8"))
    flags = list(meta.get("dirty_ablation_flags", []))
    flags.append(REPAIR_FLAG)
    meta["dirty_ablation_flags"] = sorted(set(flags))
    meta["nonfinite_lowdim_repair"] = {
        "method": "random_uniform",
        "values_replaced": int(values_replaced),
        "replacement_range": [float(replacement_min), float(replacement_max)],
    }
    return json.dumps(meta, ensure_ascii=False).encode("utf-8")


def load_nonfinite_keys(path: str | Path) -> dict[str, set[str]]:
    keys_by_shard: dict[str, set[str]] = defaultdict(set)
    with open(path, "r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, 1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if record.get("reason") != REPAIR_REASON:
                continue
            key = record.get("key")
            shard = record.get("shard")
            if not key or not shard:
                raise ValueError(f"{path}:{line_no}: NonFiniteDataError record must contain key and shard")
            keys_by_shard[str(shard)].add(str(key))
    return dict(keys_by_shard)


def sample_key_for_member(member_name: str, keys: set[str]) -> str | None:
    for key in keys:
        if member_name == f"{key}.lowdim.npy" or member_name == f"{key}.meta.json":
            return key
    return None


def add_payload(tar_writer: tarfile.TarFile, member: tarfile.TarInfo, payload: bytes) -> None:
    updated = copy.copy(member)
    updated.size = len(payload)
    tar_writer.addfile(updated, io.BytesIO(payload))


def rewrite_shard(
    *,
    src_shard: str,
    dst_dir: Path,
    sample_keys: set[str],
    seed: str,
    replacement_min: float,
    replacement_max: float,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    src_path = Path(src_shard)
    dst_path = dst_dir / src_path.name
    if not dst_path.exists():
        raise FileNotFoundError(f"destination shard not found: {dst_path}")
    if not overwrite and not dry_run:
        raise FileExistsError(f"destination shard exists: {dst_path}. Pass --overwrite to rewrite it.")

    stats: dict[str, Any] = {
        "source": str(src_path),
        "destination": str(dst_path),
        "sample_keys_requested": len(sample_keys),
        "sample_keys_repaired": [],
        "sample_keys_without_nonfinite": [],
        "values_replaced": 0,
        "members_total": 0,
        "dry_run": bool(dry_run),
    }
    repaired_counts_by_key: dict[str, int] = {}
    tmp_path = dst_path.with_name(dst_path.name + ".tmp")
    if tmp_path.exists() and not dry_run:
        tmp_path.unlink()

    started = time.time()
    with tarfile.open(dst_path, "r:*") as src:
        dst = None if dry_run else tarfile.open(tmp_path, "w")
        try:
            for member in src:
                stats["members_total"] += 1
                if not member.isfile():
                    if dst is not None:
                        dst.addfile(member)
                    continue

                file_obj = src.extractfile(member)
                payload = file_obj.read() if file_obj is not None else b""
                sample_key = sample_key_for_member(member.name, sample_keys)

                if sample_key is not None and member.name.endswith(".lowdim.npy"):
                    lowdim = decode_npy(payload)
                    repaired, count = replace_nonfinite_values(
                        lowdim,
                        seed=seed,
                        sample_key=sample_key,
                        replacement_min=replacement_min,
                        replacement_max=replacement_max,
                    )
                    repaired_counts_by_key[sample_key] = count
                    if count:
                        payload = encode_npy(repaired)
                        stats["values_replaced"] += count

                elif sample_key is not None and member.name.endswith(".meta.json"):
                    count = repaired_counts_by_key.get(sample_key, 0)
                    if count:
                        payload = update_meta(
                            payload,
                            values_replaced=count,
                            replacement_min=replacement_min,
                            replacement_max=replacement_max,
                        )

                if dst is not None:
                    add_payload(dst, member, payload)
        finally:
            if dst is not None:
                dst.close()

    repaired = sorted(key for key, count in repaired_counts_by_key.items() if count > 0)
    without = sorted(key for key, count in repaired_counts_by_key.items() if count == 0)
    missing = sorted(set(sample_keys) - set(repaired_counts_by_key))
    stats["sample_keys_repaired"] = repaired
    stats["sample_keys_without_nonfinite"] = without
    stats["sample_keys_missing_in_shard"] = missing
    stats["elapsed_sec"] = time.time() - started

    if not dry_run:
        os.replace(tmp_path, dst_path)
    return stats


def validate_args(args: argparse.Namespace) -> None:
    if not np.isfinite(args.replacement_min) or not np.isfinite(args.replacement_max):
        raise SystemExit("--replacement-min/max must be finite")
    if float(args.replacement_min) > float(args.replacement_max):
        raise SystemExit("--replacement-min must be <= --replacement-max")
    if int(args.workers) < 1:
        raise SystemExit("--workers must be >= 1")


def build_rewrite_kwargs(
    *,
    src_shard: str,
    sample_keys: set[str],
    dst_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "src_shard": src_shard,
        "dst_dir": dst_dir,
        "sample_keys": sample_keys,
        "seed": str(args.seed),
        "replacement_min": float(args.replacement_min),
        "replacement_max": float(args.replacement_max),
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
    }


def rewrite_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return rewrite_shard(**kwargs)


def rewrite_shards(
    *,
    keys_by_shard: dict[str, set[str]],
    dst_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    jobs = [
        build_rewrite_kwargs(
            src_shard=src_shard,
            sample_keys=sample_keys,
            dst_dir=dst_dir,
            args=args,
        )
        for src_shard, sample_keys in sorted(keys_by_shard.items())
    ]
    workers = max(1, int(args.workers))
    if workers == 1:
        iterator = tqdm(jobs, desc="Repair nonfinite lowdim", unit="shard")
        return [rewrite_from_kwargs(job) for job in iterator]

    executor_cls = (
        concurrent.futures.ProcessPoolExecutor
        if str(args.executor) == "process"
        else concurrent.futures.ThreadPoolExecutor
    )
    results: list[dict[str, Any]] = []
    pending_jobs = iter(jobs)
    futures: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}

    def submit_next(executor: concurrent.futures.Executor) -> bool:
        try:
            job = next(pending_jobs)
        except StopIteration:
            return False
        futures[executor.submit(rewrite_from_kwargs, job)] = job
        return True

    with executor_cls(max_workers=workers) as executor:
        for _ in range(min(workers, len(jobs))):
            submit_next(executor)
        with tqdm(total=len(jobs), desc="Repair nonfinite lowdim", unit="shard") as progress:
            while futures:
                done, _ = concurrent.futures.wait(
                    futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    futures.pop(future)
                    results.append(future.result())
                    progress.update(1)
                    submit_next(executor)

    results.sort(key=lambda item: item["destination"])
    return results


def main() -> None:
    args = parse_args()
    validate_args(args)
    dst_dir = Path(args.dst_dir)
    report_path = Path(args.report) if args.report else dst_dir / "repair_nonfinite_lowdim_report.json"
    keys_by_shard = load_nonfinite_keys(args.bad_keys)

    report: dict[str, Any] = {
        "bad_keys": args.bad_keys,
        "dst_dir": str(dst_dir),
        "reason": REPAIR_REASON,
        "seed": args.seed,
        "replacement_range": [float(args.replacement_min), float(args.replacement_max)],
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "shards_to_rewrite": len(keys_by_shard),
        "shards": [],
    }

    report["workers"] = int(args.workers)
    report["executor"] = str(args.executor)
    report["shards"] = rewrite_shards(keys_by_shard=keys_by_shard, dst_dir=dst_dir, args=args)

    report["sample_keys_requested"] = sum(item["sample_keys_requested"] for item in report["shards"])
    report["sample_keys_repaired"] = sum(len(item["sample_keys_repaired"]) for item in report["shards"])
    report["sample_keys_without_nonfinite"] = sum(len(item["sample_keys_without_nonfinite"]) for item in report["shards"])
    report["sample_keys_missing_in_shard"] = sum(len(item["sample_keys_missing_in_shard"]) for item in report["shards"])
    report["values_replaced"] = sum(int(item["values_replaced"]) for item in report["shards"])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")
    print(
        json.dumps(
            {
                "shards_to_rewrite": report["shards_to_rewrite"],
                "sample_keys_repaired": report["sample_keys_repaired"],
                "values_replaced": report["values_replaced"],
                "missing": report["sample_keys_missing_in_shard"],
            },
            ensure_ascii=False,
        )
    )
    if report["sample_keys_missing_in_shard"]:
        raise SystemExit(f"Some requested sample keys were not found: {report['sample_keys_missing_in_shard']}")


if __name__ == "__main__":
    main()
