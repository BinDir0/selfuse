"""
Verify data completeness between WebDataset shards and original HF datasets.

Counts expected samples from hf_list sources (arrow/parquet), scans all WDS
samples, and compares per-dataset counts to detect missing or extra samples.

Usage:
    python data/webdataset/verify_hf_wds_completeness.py \
        --hf_list /path/to/hf_paths.txt \
        --wds_dir /cfs/data/vlm_wds/ \
        --split train \
        --num_workers 16
"""

import glob
import json
import argparse
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

import webdataset as wds
from datasets import load_dataset


def collect_data_files(dataset_path, split):
    arrow_files = sorted(glob.glob(f"{dataset_path}/{split}/*.arrow"))
    if arrow_files:
        return arrow_files, "arrow"

    parquet_files = sorted(glob.glob(f"{dataset_path}/{split}/*.parquet"))
    if parquet_files:
        return parquet_files, "parquet"

    return [], None


def parse_meta(meta):
    try:
        if isinstance(meta, dict):
            return meta
        if isinstance(meta, bytes):
            return json.loads(meta.decode("utf-8"))
        if isinstance(meta, str):
            return json.loads(meta)
    except Exception:
        return None
    return None


def count_hf_dataset(args):
    dataset_path, dataset_name, split = args
    files, ext = collect_data_files(dataset_path, split)
    if not files:
        return {
            "dataset_name": dataset_name,
            "expected_count": 0,
            "missing_source": True,
            "num_files": 0,
        }

    stream = load_dataset(ext, data_files=files, split="train", streaming=True)
    count = 0
    for _ in stream:
        count += 1

    return {
        "dataset_name": dataset_name,
        "expected_count": count,
        "missing_source": False,
        "num_files": len(files),
    }


def scan_wds_shard(shard_path):
    dataset = wds.WebDataset(str(shard_path), shardshuffle=False)
    counts = defaultdict(int)
    samples = 0
    bad_meta = 0

    for sample in dataset:
        samples += 1
        meta = parse_meta(sample.get("meta.json"))
        if meta is None:
            bad_meta += 1
            continue
        dataset_name = meta.get("dataset_name")
        if dataset_name is None:
            bad_meta += 1
            continue
        counts[str(dataset_name)] += 1

    return {
        "shard": shard_path.name,
        "samples": samples,
        "bad_meta": bad_meta,
        "counts": dict(counts),
    }


def load_hf_entries(hf_list):
    with open(hf_list) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    entries = []
    for line in lines:
        parts = line.split()
        dataset_path = parts[0]
        dataset_name = parts[1] if len(parts) > 1 else Path(dataset_path).stem
        entries.append((dataset_path, dataset_name))
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Verify HF->WDS completeness by per-dataset sample counts"
    )
    parser.add_argument(
        "--hf_list",
        type=str,
        required=True,
        help="Text file: each line is 'dataset_path [dataset_name]'",
    )
    parser.add_argument(
        "--wds_dir",
        type=str,
        required=True,
        help="Root directory containing WebDataset shards",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to verify, e.g. train/test",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--shard_pattern",
        type=str,
        default="*.tar",
        help="Shard filename glob pattern under wds_dir",
    )
    args = parser.parse_args()

    entries = load_hf_entries(args.hf_list)
    if not entries:
        print(f"No valid entries in {args.hf_list}")
        raise SystemExit(1)

    print("Counting expected samples from HF source files...")
    count_args = [(dataset_path, dataset_name, args.split) for dataset_path, dataset_name in entries]
    expected_counts = {}
    missing_sources = []
    with mp.Pool(min(args.num_workers, len(count_args))) as pool:
        for result in pool.imap_unordered(count_hf_dataset, count_args):
            dataset_name = result["dataset_name"]
            expected_counts[dataset_name] = result["expected_count"]
            if result["missing_source"]:
                missing_sources.append(dataset_name)
            print(
                f"  {dataset_name}: {result['expected_count']} samples "
                f"({result['num_files']} files)",
                flush=True,
            )

    all_shards = sorted(Path(args.wds_dir).rglob(args.shard_pattern))
    if not all_shards:
        print(f"No shards found under {args.wds_dir} with pattern {args.shard_pattern}")
        raise SystemExit(1)

    print(f"\nScanning {len(all_shards)} shards...")
    actual_counts = defaultdict(int)
    total_samples = 0
    total_bad_meta = 0
    with mp.Pool(args.num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(scan_wds_shard, all_shards)):
            total_samples += result["samples"]
            total_bad_meta += result["bad_meta"]
            for ds_name, cnt in result["counts"].items():
                actual_counts[ds_name] += cnt
            if (i + 1) % 100 == 0 or (i + 1) == len(all_shards):
                print(f"  [{i+1}/{len(all_shards)}] scanned", flush=True)

    print(f"\n{'='*60}")
    print("Completeness check:\n")

    all_ok = True
    for ds_name in sorted(expected_counts.keys()):
        expected = expected_counts[ds_name]
        actual = actual_counts.get(ds_name, 0)
        ok = (expected == actual)
        if not ok:
            all_ok = False
        status = "OK" if ok else "INCOMPLETE"
        delta = actual - expected
        print(f"[{status}] {ds_name}: actual={actual}, expected={expected}, delta={delta}")

    unknown_ds = sorted(set(actual_counts.keys()) - set(expected_counts.keys()))
    if unknown_ds:
        all_ok = False
        print(f"\nUnknown datasets in WDS (not in hf_list): {unknown_ds}")

    if missing_sources:
        all_ok = False
        print(f"\nDatasets missing source files: {sorted(missing_sources)}")

    print(f"\n{'='*60}")
    print(f"Total scanned samples: {total_samples}")
    print(f"Bad meta samples: {total_bad_meta}")
    if all_ok and total_bad_meta == 0:
        print("All datasets complete.")
    else:
        print("Some datasets have completeness issues.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
