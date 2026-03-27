"""
Convert Robo2VLM-1 dataset (Parquet) to WebDataset format.

Robo2VLM-1 is a multiple-choice VQA dataset with:
- question: str
- choices: list of strings
- correct_answer: int (index of correct choice)
- image: PIL Image

Output: WebDataset shards with JPEG images and JSON metadata.
"""

import os
import glob
import shutil
import ast
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
from wds_utils import ShardWriter

# ================= Configuration =================
ROBO2VLM_ROOT = "/share_data/guantianrui/datasets/VLM/Robo2VLM-1/data"
OUTPUT_DIR = "/share_data/zengfanlian/datasets/VLM/Webdataset/robo2vlm"

# Test mode: Set to a number to only process first N files (None = process all)
MAX_TRAIN_FILES = None  # Set to None to process all train files, or a number like 2 for testing
MAX_TEST_FILES = None   # Set to None to process all test files

# Text quality filtering
MAX_TURN_CHARS = 800  # drop samples where user+assistant combined > this

# ================================================


def format_multiple_choice(question, choices):
    """
    Format question and choices with letter labels (A, B, C, D, ...).
    Supports up to 26 choices (A-Z).

    Args:
        question: str - the question text
        choices: list of str - the answer choices

    Returns:
        str - formatted question with letter-labeled choices
    """
    # Dynamically generate letter labels A-Z (supports up to 26 choices)
    formatted_choices = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
    formatted_question = f"Question: {question}\nChoices:\n{formatted_choices}\nAnswer with the letter."
    return formatted_question


def format_answer(correct_answer_idx, choices):
    """
    Format the answer using the letter label (A, B, C, D, ...).
    Supports up to 26 choices (A-Z).

    Args:
        correct_answer_idx: int - index of correct answer
        choices: list of str - the answer choices (not used, kept for compatibility)

    Returns:
        str - formatted answer with letter label
    """
    # Dynamically generate letter label (A=65 in ASCII)
    return f"Answer: {chr(65 + correct_answer_idx)}"


def process_chunk(args):
    """Process a chunk of samples and write to WebDataset shards."""
    parquet_files, start_idx, end_idx, output_dir, worker_id, split, image_quality, maxcount, maxsize = args
    
    # Worker 内部只加载属于自己的那部分数据切片
    from datasets import load_dataset
    chunk = load_dataset('parquet', data_files=parquet_files, split=f'train[{start_idx}:{end_idx}]')
    
    sw = ShardWriter(output_dir, split=split, worker_id=worker_id,
                     maxcount=maxcount, maxsize=maxsize, image_quality=image_quality)
    for local_idx, sample in enumerate(chunk):
        choices = sample['choices']
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        user_text = format_multiple_choice(sample['question'], choices)
        assistant_text = format_answer(sample['correct_answer'], choices)
        if len(user_text) + len(assistant_text) > MAX_TURN_CHARS:
            continue
        img = sample['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        key = f"robo2vlm_w{worker_id:04d}_{local_idx:010d}"
        sw.write(key, [img], [{"user": user_text, "assistant": assistant_text}],
                 source="robo2vlm", sample_idx=local_idx)
    sw.close()


def main():
    parser = argparse.ArgumentParser(description="Convert Robo2VLM-1 to WebDataset format")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--maxcount", type=int, default=20000, help="Max samples per shard")
    parser.add_argument("--maxsize", type=float, default=1e9, help="Max bytes per shard")
    parser.add_argument("--image_quality", type=int, default=95, help="JPEG quality")
    args = parser.parse_args()

    print("=" * 80)
    print("Robo2VLM-1 to WebDataset Format Conversion")
    print("=" * 80)

    # Find parquet files
    train_files = sorted(glob.glob(os.path.join(ROBO2VLM_ROOT, "train-*.parquet")))
    test_files = sorted(glob.glob(os.path.join(ROBO2VLM_ROOT, "test-*.parquet")))

    print(f"\nFound:")
    print(f"  Train files: {len(train_files)}")
    print(f"  Test files: {len(test_files)}")

    # Apply file limits for testing
    if MAX_TRAIN_FILES is not None:
        train_files = train_files[:MAX_TRAIN_FILES]
        print(f"\nTEST MODE: Using only first {MAX_TRAIN_FILES} train files")

    if MAX_TEST_FILES is not None:
        test_files = test_files[:MAX_TEST_FILES]
        print(f"TEST MODE: Using only first {MAX_TEST_FILES} test files")

    # Prepare output directory
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each split
    for split_name, parquet_files in [("train", train_files), ("test", test_files)]:
        if not parquet_files:
            print(f"\nNo parquet files found for {split_name} split, skipping.")
            continue

        print(f"\n{'=' * 80}")
        print(f"Processing {split_name} split")
        print(f"{'=' * 80}")
        print(f"Loading from {len(parquet_files)} parquet files...")

        # Load all parquet files for this split
        ds = load_dataset(
            'parquet',
            data_files=parquet_files,
            split='train'  # Parquet files don't have split info, we organize manually
        )

        total_samples = len(ds)
        print(f"Total samples in {split_name}: {total_samples}")
        
        # 释放主进程的 dataset 对象，避免占用内存
        del ds

        num_workers = min(args.num_workers, total_samples)
        if num_workers == 0:
            continue

        # Split data into chunks for workers
        chunk_size = (total_samples + num_workers - 1) // num_workers
        chunks = []
        for i in range(num_workers):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_samples)
            if start < end:
                chunks.append((
                    parquet_files,
                    start,
                    end,
                    OUTPUT_DIR,
                    i,
                    split_name,
                    args.image_quality,
                    args.maxcount,
                    int(args.maxsize),
                ))

        print(f"Writing {split_name} with {len(chunks)} workers...")
        with mp.Pool(len(chunks)) as pool:
            pool.map(process_chunk, chunks)

        print(f"Finished writing {split_name} split: {total_samples} samples")

    print("\n" + "=" * 80)
    print(f"Conversion completed successfully!")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
