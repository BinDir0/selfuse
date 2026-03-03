"""
Convert Hugging Face datasets (arrow/parquet) to WebDataset shard format.

Each sample stores:
- image_XXX.jpg: one JPEG per image in the sample
- meta.json: text candidates, ratings, dataset info

Usage:
    python data/convert_hf_to_wds.py \
        --hf_list /path/to/hf_paths.txt \
        --output_dir /cfs/data/vlm_wds/ \
        --split train \
        --num_workers 32
"""

import io
import json
import time
import glob
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import webdataset as wds
from datasets import load_dataset
from PIL import Image


def to_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, list):
        return [to_python(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    return obj


def collect_data_files(dataset_path, split):
    """Collect arrow/parquet files under one HF dataset path."""
    arrow_files = sorted(glob.glob(f"{dataset_path}/{split}/*.arrow"))
    if arrow_files:
        return arrow_files, "arrow"

    parquet_files = sorted(glob.glob(f"{dataset_path}/{split}/*.parquet"))
    if parquet_files:
        return parquet_files, "parquet"

    return [], None


def sanitize_name(name):
    return str(name).replace("/", "_").replace(" ", "_")


def process_task_batch(
    task_batch,
    output_pattern,
    split,
    maxcount,
    maxsize,
    image_quality,
    worker_id=0,
):
    """Process a batch of (file, ext, dataset_name) tasks and write WDS shards."""
    shard_idx = 0
    shard_count = 0
    shard_size = 0
    sample_idx = 0
    writer = wds.TarWriter(output_pattern % shard_idx)
    source_sample_idx = {}

    t0 = time.time()
    report_interval = 10000

    for file_path, ext, dataset_name in task_batch:
        safe_dataset_name = sanitize_name(dataset_name)
        local_sample_idx = source_sample_idx.get(dataset_name, 0)
        stream = load_dataset(ext, data_files=[file_path], split="train", streaming=True)

        for sample in stream:
            if shard_count > 0 and (shard_count + 1 > maxcount or shard_size > maxsize):
                writer.close()
                shard_idx += 1
                writer = wds.TarWriter(output_pattern % shard_idx)
                shard_count = 0
                shard_size = 0

            images = sample["images"]
            image_bytes = {}
            image_total_bytes = 0
            for i, image in enumerate(images):
                if image.mode != "RGB":
                    image = image.convert("RGB")
                buf = io.BytesIO()
                image.save(buf, format="JPEG", quality=image_quality)
                key = f"image_{i:03d}.jpg"
                data = buf.getvalue()
                image_bytes[key] = data
                image_total_bytes += len(data)

            meta_dict = {
                "dataset_name": dataset_name,
                "source": sample.get("source", dataset_name),
                "split": split,
                "sample_idx": int(local_sample_idx),
                "n_images": int(len(images)),
                "texts": sample["texts"],
                "formatting_ratings": sample["formatting_ratings"],
                "visual_dependency_ratings": sample["visual_dependency_ratings"],
                "relevance_ratings": sample["relevance_ratings"],
            }
            meta_dict = to_python(meta_dict)
            meta_size = len(json.dumps(meta_dict, ensure_ascii=False).encode("utf-8"))

            wds_sample = {
                "__key__": f"{safe_dataset_name}_w{worker_id:04d}_{sample_idx:010d}",
                "meta.json": meta_dict,
            }
            wds_sample.update(image_bytes)

            writer.write(wds_sample)
            shard_count += 1
            shard_size += image_total_bytes + meta_size + 512 * (2 + len(image_bytes))
            sample_idx += 1
            local_sample_idx += 1

            if sample_idx % report_interval == 0:
                elapsed = time.time() - t0
                sps = sample_idx / elapsed if elapsed > 0 else 0
                print(
                    f"  [w{worker_id:03d}] {sample_idx} samples, "
                    f"{elapsed:.1f}s ({sps:.1f} samples/s)",
                    flush=True,
                )

        source_sample_idx[dataset_name] = local_sample_idx

    writer.close()
    elapsed = time.time() - t0
    sps = sample_idx / elapsed if elapsed > 0 else 0
    print(
        f"  [w{worker_id:03d}] done, "
        f"{sample_idx} samples, {elapsed:.1f}s ({sps:.1f} samples/s)",
        flush=True,
    )


def load_hf_entries(hf_list_path):
    with open(hf_list_path) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    entries = []
    for line in lines:
        parts = line.split()
        dataset_path = parts[0]
        dataset_name = parts[1] if len(parts) > 1 else Path(dataset_path).stem
        entries.append((dataset_path, dataset_name))
    return entries


def collect_all_tasks(
    entries,
    split,
):
    tasks = []
    for dataset_path, dataset_name in entries:
        files, ext = collect_data_files(dataset_path, split)
        if not files:
            print(f"Warning: no files found for {dataset_path} split={split}, skipping.")
            continue
        for fp in files:
            tasks.append((fp, ext, dataset_name))
    return tasks


def convert_hf_list(
    hf_list,
    output_dir,
    split="train",
    num_workers=32,
    maxcount=20000,
    maxsize=int(1e9),
    image_quality=95,
    shard_prefix="shard",
):
    """Convert all paths in hf_list into one merged WebDataset."""
    entries = load_hf_entries(hf_list)
    tasks = collect_all_tasks(entries, split)
    if not tasks:
        print("No valid input files found, exiting.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_workers = min(num_workers, len(tasks))
    chunk_size = max(1, (len(tasks) + actual_workers - 1) // actual_workers)
    chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]

    args_list = []
    for worker_id, task_chunk in enumerate(chunks):
        pattern = str(output_dir / f"{shard_prefix}-w{worker_id:04d}-%06d.tar")
        args_list.append(
            (
                task_chunk,
                pattern,
                split,
                maxcount,
                maxsize,
                image_quality,
                worker_id,
            )
        )

    source_names = sorted(set(dataset_name for _, dataset_name in entries))
    print(
        f"Converting merged HF list: {len(tasks)} files from "
        f"{len(source_names)} datasets, "
        f"{len(chunks)} workers"
    )
    if len(chunks) == 1:
        process_task_batch(*args_list[0])
    else:
        with mp.Pool(len(chunks)) as pool:
            pool.starmap(process_task_batch, args_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face arrow/parquet datasets to merged WebDataset shards"
    )
    parser.add_argument(
        "--hf_list",
        type=str,
        required=True,
        help="Text file: each line is 'dataset_path [dataset_name]'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for WebDataset shards",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split directory to convert, e.g. train/test",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of workers",
    )
    parser.add_argument(
        "--maxcount",
        type=int,
        default=20000,
        help="Max samples per shard",
    )
    parser.add_argument(
        "--maxsize",
        type=int,
        default=int(1e9),
        help="Max approximate bytes per shard",
    )
    parser.add_argument(
        "--image_quality",
        type=int,
        default=95,
        help="JPEG quality for image encoding",
    )
    parser.add_argument(
        "--shard_prefix",
        type=str,
        default="shard",
        help="Shard file prefix",
    )
    args = parser.parse_args()

    convert_hf_list(
        hf_list=args.hf_list,
        output_dir=args.output_dir,
        split=args.split,
        num_workers=args.num_workers,
        maxcount=args.maxcount,
        maxsize=args.maxsize,
        image_quality=args.image_quality,
        shard_prefix=args.shard_prefix,
    )
