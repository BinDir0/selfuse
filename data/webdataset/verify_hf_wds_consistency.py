"""
Verify sample consistency for HF-converted WebDataset shards.

Checks sample-level invariants:
- meta.json is valid and has required fields
- image_*.jpg exists and can be decoded
- meta.n_images matches number of image_*.jpg entries
- texts and rating arrays are aligned

Usage:
    python data/webdataset/verify_hf_wds_consistency.py \
        --wds_dir /cfs/data/vlm_wds/ \
        --split train \
        --num_shards 100 \
        --num_workers 16
"""

import io
import json
import random
import argparse
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

import webdataset as wds
from PIL import Image


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


def verify_sample(sample, expected_split):
    checks = {
        "meta": False,
        "dataset_name": False,
        "images": False,
        "n_images": False,
        "texts": False,
        "ratings": False,
        "split": False,
    }

    errors = []

    meta = parse_meta(sample.get("meta.json"))
    if meta is None:
        errors.append("invalid meta.json")
        return checks, errors
    checks["meta"] = True

    dataset_name = meta.get("dataset_name")
    if dataset_name is None or str(dataset_name) == "":
        errors.append("missing dataset_name")
    else:
        checks["dataset_name"] = True

    image_keys = sorted(
        key for key in sample.keys()
        if key.startswith("image_") and key.endswith(".jpg")
    )
    if not image_keys:
        errors.append("missing image_*.jpg")
    else:
        image_decode_ok = True
        for key in image_keys:
            try:
                Image.open(io.BytesIO(sample[key])).convert("RGB")
            except Exception:
                image_decode_ok = False
                errors.append(f"invalid image bytes: {key}")
                break
        if image_decode_ok:
            checks["images"] = True

    n_images = meta.get("n_images")
    if isinstance(n_images, int) and n_images == len(image_keys):
        checks["n_images"] = True
    else:
        errors.append(f"n_images mismatch: meta={n_images}, files={len(image_keys)}")

    texts = meta.get("texts")
    if not isinstance(texts, list) or len(texts) == 0:
        errors.append("invalid texts")
    else:
        valid_texts = True
        for text in texts:
            if not isinstance(text, dict):
                valid_texts = False
                break
            if "user" not in text or "assistant" not in text:
                valid_texts = False
                break
        if valid_texts:
            checks["texts"] = True
        else:
            errors.append("texts entries must include user/assistant")

    rating_fields = [
        "formatting_ratings",
        "visual_dependency_ratings",
        "relevance_ratings",
    ]
    ratings_ok = True
    if isinstance(texts, list):
        expected_len = len(texts)
        for field in rating_fields:
            values = meta.get(field)
            if not isinstance(values, list) or len(values) != expected_len:
                ratings_ok = False
                errors.append(f"{field} length mismatch")
                break
    else:
        ratings_ok = False
    if ratings_ok:
        checks["ratings"] = True

    if expected_split is not None:
        if meta.get("split") == expected_split:
            checks["split"] = True
        else:
            errors.append(f"split mismatch: {meta.get('split')} != {expected_split}")
    else:
        checks["split"] = True

    return checks, errors


def verify_shard(args):
    shard_path, expected_split = args
    dataset = wds.WebDataset(str(shard_path), shardshuffle=False)

    total_samples = 0
    total_errors = 0
    check_totals = defaultdict(int)
    error_msgs = []

    for sample in dataset:
        total_samples += 1
        checks, errors = verify_sample(sample, expected_split)
        for field, ok in checks.items():
            if ok:
                check_totals[field] += 1
        if errors:
            total_errors += 1
            if len(error_msgs) < 10:
                key = sample.get("__key__", "<no_key>")
                error_msgs.append(f"{key}: {errors[0]}")

    return {
        "shard": shard_path.name,
        "samples": total_samples,
        "errors": total_errors,
        "check_totals": dict(check_totals),
        "error_msgs": error_msgs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify HF->WDS sample consistency"
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
        help="Expected split in meta.json (set empty string to skip)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=100,
        help="Number of shards to randomly sample; <=0 means all shards",
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

    all_shards = []
    for sub_wds_path in Path(args.wds_dir).iterdir():
        if sub_wds_path.is_dir():
            split_path = Path(sub_wds_path) / args.split
            sub_wds_shards = sorted(split_path.rglob(args.shard_pattern))
            if not sub_wds_shards:
                print(f"No shards found under {split_path} with pattern {args.shard_pattern}")
                raise SystemExit(1)
            print(f"\nScanning {len(sub_wds_shards)} shards in {sub_wds_path}, split {args.split}...")
            all_shards.extend(sub_wds_shards)
    if not all_shards:
        print(f"No shards found under {args.wds_dir} with pattern {args.shard_pattern}")
        raise SystemExit(1)

    if args.num_shards > 0 and args.num_shards < len(all_shards):
        random.seed(0)
        shards = random.sample(all_shards, args.num_shards)
    else:
        shards = all_shards

    expected_split = args.split if args.split != "" else None
    work_args = [(shard_path, expected_split) for shard_path in shards]

    print(f"Verifying {len(shards)} shards with {args.num_workers} workers...")
    total_samples = 0
    total_errors = 0
    all_check_totals = defaultdict(int)
    all_error_msgs = []

    with mp.Pool(args.num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(verify_shard, work_args)):
            total_samples += result["samples"]
            total_errors += result["errors"]
            for field, cnt in result["check_totals"].items():
                all_check_totals[field] += cnt
            all_error_msgs.extend(result["error_msgs"])
            if (i + 1) % 100 == 0 or (i + 1) == len(work_args):
                print(
                    f"  [{i+1}/{len(work_args)}] scanned, "
                    f"samples={total_samples}, errors={total_errors}",
                    flush=True,
                )

    print(f"\n{'='*60}")
    print("Consistency check:\n")

    fields = [
        "meta",
        "dataset_name",
        "images",
        "n_images",
        "texts",
        "ratings",
        "split",
    ]
    for field in fields:
        print(f"  {field}: {all_check_totals.get(field, 0)}/{total_samples}")

    print(f"\nTotal samples: {total_samples}")
    print(f"Samples with errors: {total_errors}")
    if total_errors > 0:
        print("\nExample errors:")
        for msg in all_error_msgs[:20]:
            print(f"  - {msg}")
        raise SystemExit(1)

    print("All checked samples are consistent.")


if __name__ == "__main__":
    main()
