"""
Verify sample consistency for HF-converted WebDataset shards.

Checks sample-level invariants:
- meta.json is valid and has required fields
- image_*.jpg exists and can be decoded
- meta.n_images matches number of image_*.jpg entries
- texts and rating arrays are aligned

Path protocol:
- wds_dir: root directory containing one subdir per dataset,
  each subdir contains {split}/*.tar shards.
- hf_list (optional): each line is "dataset_path [dataset_name]".
  dataset_path must contain split subdir with {split}/*.arrow or {split}/*.parquet.

Usage:
    python data/webdataset/verify_hf_wds_consistency.py \
        --wds_dir /cfs/data/vlm_wds/ \
        --split train \
        --num_shards 100 \
        --num_workers 16
    # Optional HF<->WDS comparison:
    #   --hf_list /path/to/hf_paths.txt \
    #   --image_size 32 \
    #   --image_diff_threshold 2.0
"""

import io
import json
import random
import argparse
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

import webdataset as wds
import numpy as np
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


def to_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, list):
        return [to_python(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    return obj


def image_to_signature(image, size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def bytes_to_signature(image_bytes, size):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image_to_signature(image, size)


def mean_abs_diff(a, b):
    return np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))


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


def collect_data_files(dataset_path, split):
    arrow_files = sorted(Path(dataset_path).glob(f"{split}/*.arrow"))
    if arrow_files:
        arrow_files = [p for p in arrow_files if p.stat().st_size > 0]
        return [str(p) for p in arrow_files], "arrow"

    parquet_files = sorted(Path(dataset_path).glob(f"{split}/*.parquet"))
    if parquet_files:
        parquet_files = [p for p in parquet_files if p.stat().st_size > 0]
        return [str(p) for p in parquet_files], "parquet"

    return [], None


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


def scan_wds_for_compare(shards, expected_split, image_size):
    total_samples = 0
    total_errors = 0
    all_check_totals = defaultdict(int)
    all_error_msgs = []
    wds_index = defaultdict(dict)
    duplicates = defaultdict(list)

    for i, shard_path in enumerate(shards):
        dataset = wds.WebDataset(str(shard_path), shardshuffle=False)
        for sample in dataset:
            total_samples += 1
            checks, errors = verify_sample(sample, expected_split)
            for field, ok in checks.items():
                if ok:
                    all_check_totals[field] += 1
            if errors:
                total_errors += 1
                if len(all_error_msgs) < 20:
                    key = sample.get("__key__", "<no_key>")
                    all_error_msgs.append(f"{key}: {errors[0]}")
                continue

            meta = parse_meta(sample.get("meta.json"))
            if meta is None:
                continue
            dataset_name = meta.get("dataset_name")
            sample_idx = meta.get("sample_idx")
            if dataset_name is None or sample_idx is None:
                continue
            try:
                sample_idx = int(sample_idx)
            except Exception:
                continue

            if sample_idx in wds_index[dataset_name]:
                duplicates[dataset_name].append(sample_idx)
                continue

            image_keys = sorted(
                key for key in sample.keys()
                if key.startswith("image_") and key.endswith(".jpg")
            )
            image_sigs = [bytes_to_signature(sample[k], image_size) for k in image_keys]
            wds_index[dataset_name][sample_idx] = {
                "meta": {
                    "dataset_name": dataset_name,
                    "source": meta.get("source", dataset_name),
                    "sample_idx": sample_idx,
                    "n_images": meta.get("n_images"),
                    "texts": to_python(meta.get("texts")),
                    "formatting_ratings": to_python(meta.get("formatting_ratings")),
                    "visual_dependency_ratings": to_python(meta.get("visual_dependency_ratings")),
                    "relevance_ratings": to_python(meta.get("relevance_ratings")),
                },
                "image_sigs": image_sigs,
                "__key__": sample.get("__key__", "<no_key>"),
            }

        if (i + 1) % 100 == 0 or (i + 1) == len(shards):
            print(
                f"  [{i+1}/{len(shards)}] scanned, "
                f"samples={total_samples}, errors={total_errors}",
                flush=True,
            )

    return {
        "total_samples": total_samples,
        "total_errors": total_errors,
        "check_totals": dict(all_check_totals),
        "error_msgs": all_error_msgs,
        "wds_index": wds_index,
        "duplicates": duplicates,
    }


def compare_hf_to_wds(entries, split, wds_index, image_size, diff_threshold):
    from datasets import load_dataset

    total_compared = 0
    meta_mismatch = 0
    image_mismatch = 0
    missing_in_hf = defaultdict(list)
    worst_diff = 0.0

    for dataset_path, dataset_name in entries:
        needed = wds_index.get(dataset_name, {})
        if not needed:
            continue

        files, ext = collect_data_files(dataset_path, split)
        if not files:
            missing_in_hf[dataset_name].extend(sorted(needed.keys()))
            continue

        found = set()
        sample_idx = 0
        for fp in files:
            stream = load_dataset(ext, data_files=[fp], split="train", streaming=True)
            for sample in stream:
                if sample_idx in needed:
                    found.add(sample_idx)
                    wds_info = needed[sample_idx]

                    hf_meta = {
                        "dataset_name": dataset_name,
                        "source": sample.get("source", dataset_name),
                        "sample_idx": sample_idx,
                        "n_images": int(len(sample["images"])),
                        "texts": to_python(sample.get("texts")),
                        "formatting_ratings": to_python(sample.get("formatting_ratings")),
                        "visual_dependency_ratings": to_python(sample.get("visual_dependency_ratings")),
                        "relevance_ratings": to_python(sample.get("relevance_ratings")),
                    }

                    if hf_meta != wds_info["meta"]:
                        meta_mismatch += 1

                    hf_images = sample["images"]
                    if len(hf_images) != len(wds_info["image_sigs"]):
                        image_mismatch += 1
                    else:
                        for i, hf_img in enumerate(hf_images):
                            hf_sig = image_to_signature(hf_img, image_size)
                            wds_sig = wds_info["image_sigs"][i]
                            diff = mean_abs_diff(hf_sig, wds_sig)
                            worst_diff = max(worst_diff, diff)
                            if diff > diff_threshold:
                                image_mismatch += 1
                                break

                    total_compared += 1

                sample_idx += 1

        missing = sorted(set(needed.keys()) - found)
        if missing:
            missing_in_hf[dataset_name].extend(missing)

    return {
        "total_compared": total_compared,
        "meta_mismatch": meta_mismatch,
        "image_mismatch": image_mismatch,
        "missing_in_hf": missing_in_hf,
        "worst_diff": worst_diff,
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
        "--hf_list",
        type=str,
        default=None,
        help="Optional HF list for HF<->WDS comparison",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=32,
        help="Image resize for diff (square size)",
    )
    parser.add_argument(
        "--image_diff_threshold",
        type=float,
        default=2.0,
        help="Mean abs diff threshold on resized images",
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
    if args.hf_list:
        print(f"Verifying {len(shards)} shards (single-process for HF compare)...")
        scan_result = scan_wds_for_compare(shards, expected_split, args.image_size)
        total_samples = scan_result["total_samples"]
        total_errors = scan_result["total_errors"]
        all_check_totals = scan_result["check_totals"]
        all_error_msgs = scan_result["error_msgs"]
        wds_index = scan_result["wds_index"]
        duplicates = scan_result["duplicates"]
    else:
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

    if args.hf_list:
        entries = []
        with open(args.hf_list) as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        for line in lines:
            parts = line.split()
            dataset_path = parts[0]
            dataset_name = parts[1] if len(parts) > 1 else Path(dataset_path).stem
            entries.append((dataset_path, dataset_name))

        compare = compare_hf_to_wds(
            entries,
            args.split,
            wds_index,
            args.image_size,
            args.image_diff_threshold,
        )
        print(f"\n{'='*60}")
        print("HF <-> WDS check:\n")
        print(f"Total compared: {compare['total_compared']}")
        print(f"Meta mismatches: {compare['meta_mismatch']}")
        print(f"Image mismatches: {compare['image_mismatch']}")
        print(f"Worst image diff: {compare['worst_diff']:.4f}")

        if duplicates:
            print("\nDuplicate sample_idx in WDS:")
            for ds, idxs in duplicates.items():
                uniq = sorted(set(idxs))
                print(f"  {ds}: {len(uniq)} duplicates (examples: {uniq[:10]})")

        missing = compare["missing_in_hf"]
        if missing:
            print("\nMissing in HF (present in WDS but not found in HF):")
            for ds, idxs in missing.items():
                uniq = sorted(set(idxs))
                print(f"  {ds}: {len(uniq)} missing (examples: {uniq[:10]})")

        if (compare["meta_mismatch"] > 0 or compare["image_mismatch"] > 0
                or duplicates or missing):
            raise SystemExit(1)

    print("All checked samples are consistent.")


if __name__ == "__main__":
    main()
