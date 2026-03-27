"""
Convert RoboRefit-Grasping dataset to WebDataset format.

RoboRefit-Grasping is a grasping point selection dataset with:
- question: str (grasping instruction)
- options: list of [x, y] pixel coordinates
- answer: int (index of correct grasping point)
- image: PIL Image (640x480)
- output_image: PIL Image with visualizations

Output: WebDataset shards with JPEG images and JSON metadata.
Coordinates are normalized to [0, 1] range.
"""

import os
import shutil
import argparse
import multiprocessing as mp

# CRITICAL: Set HuggingFace cache BEFORE importing datasets
# This must be done before any HuggingFace library imports to avoid filling up local disk
HF_CACHE_DIR = "/share_data/zengfanlian/.cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# NOW import datasets after environment variables are set
from datasets import load_dataset
from wds_utils import ShardWriter, split_train_test

# ================= Configuration =================
ROBOREFIT_ROOT = "/share_data/guantianrui/datasets/VLM/roborefit-grasping"
OUTPUT_DIR = "/share_data/zengfanlian/datasets/VLM/Webdataset/roborefit"

# Image dimensions (all images are 640x480)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Test mode: Set to a number to only process first N samples (None = process all)
MAX_SAMPLES = None  # Set to None to process all samples, or a number like 100 for testing

# Dataset split configuration
VAL_RATIO = 0.001  # 0.1% for validation (same as other datasets)
SEED = 42

# ================================================


def process_chunk(args):
    """Process a chunk of samples and write to WebDataset shards."""
    chunk, output_dir, worker_id, split, image_quality, maxcount, maxsize = args
    sw = ShardWriter(output_dir, split=split, worker_id=worker_id,
                     maxcount=maxcount, maxsize=maxsize, image_quality=image_quality)
    for local_idx, sample in enumerate(chunk):
        question = sample['question']
        options = sample['options']
        answer_idx = sample['answer']

        # Get the correct grasping point and convert to 0-1000 Qwen3-VL format
        correct_x, correct_y = options[answer_idx]
        qx = round(correct_x / IMAGE_WIDTH * 1000)
        qy = round(correct_y / IMAGE_HEIGHT * 1000)

        user_text = f"Where should the robot grasp to {question}?"
        assistant_text = f'[{{"point_2d": [{qx}, {qy}]}}]'

        img = sample['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        key = f"roborefit_w{worker_id:04d}_{local_idx:010d}"
        sw.write(key, [img], [{"user": user_text, "assistant": assistant_text}],
                 source="roborefit-grasping", sample_idx=local_idx)
    sw.close()


def main():
    parser = argparse.ArgumentParser(description="Convert RoboRefit-Grasping to WebDataset format")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers")
    parser.add_argument("--maxcount", type=int, default=20000, help="Max samples per shard")
    parser.add_argument("--maxsize", type=float, default=1e9, help="Max bytes per shard")
    parser.add_argument("--image_quality", type=int, default=95, help="JPEG quality")
    args = parser.parse_args()

    print("=" * 80)
    print("RoboRefit-Grasping to WebDataset Format Conversion")
    print("=" * 80)

    # Load dataset
    print(f"\nLoading dataset from {ROBOREFIT_ROOT}...")
    ds = load_dataset(ROBOREFIT_ROOT, split='train')

    total_samples = len(ds)
    print(f"Total samples: {total_samples}")

    # Limit samples if MAX_SAMPLES is set
    if MAX_SAMPLES is not None:
        ds = ds.select(range(min(MAX_SAMPLES, total_samples)))
        print(f"TEST MODE: Processing only first {MAX_SAMPLES} samples")

    # Convert to list of dicts for splitting
    print("\nConverting dataset to list...")
    data = [ds[i] for i in range(len(ds))]
    print(f"Loaded {len(data)} samples into memory")

    # Split into train/test
    print(f"\nSplitting dataset (val_ratio={VAL_RATIO})...")
    train_data, test_data = split_train_test(data, val_ratio=VAL_RATIO, seed=SEED)
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Free the original data
    del data, ds

    # Prepare output directory
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each split
    for split_name, split_data in [("train", train_data), ("test", test_data)]:
        print(f"\n{'=' * 80}")
        print(f"Writing {split_name} split: {len(split_data)} samples")
        print(f"{'=' * 80}")

        num_workers = min(args.num_workers, len(split_data))
        if num_workers == 0:
            continue

        # Split data into chunks for workers
        chunk_size = (len(split_data) + num_workers - 1) // num_workers
        chunks = []
        for i in range(num_workers):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(split_data))
            if start < end:
                chunks.append((
                    split_data[start:end],
                    OUTPUT_DIR,
                    i,
                    split_name,
                    args.image_quality,
                    args.maxcount,
                    int(args.maxsize),
                ))

        print(f"Using {len(chunks)} workers...")
        with mp.Pool(len(chunks)) as pool:
            pool.map(process_chunk, chunks)

        print(f"Finished writing {split_name} split")

    print("\n" + "=" * 80)
    print(f"Conversion completed successfully!")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
