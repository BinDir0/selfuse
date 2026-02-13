"""
Convert Robo2VLM-1 dataset (Parquet) to FineVision Arrow format.

Robo2VLM-1 is a multiple-choice VQA dataset with:
- question: str
- choices: list of strings
- correct_answer: int (index of correct choice)
- image: PIL Image

Output: FineVision Arrow format with text conversations.
"""

import os
import glob
import shutil
import ast

# CRITICAL: Set HuggingFace cache BEFORE importing datasets
# This must be done before any HuggingFace library imports to avoid filling up local disk
HF_CACHE_DIR = "/share_data/zengfanlian/.cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# NOW import datasets after environment variables are set
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, List as HFList, Image as HFImage, concatenate_datasets
from tqdm import tqdm

# ================= Configuration =================
ROBO2VLM_ROOT = "/share_data/guantianrui/datasets/VLM/Robo2VLM-1/data"
OUTPUT_DIR = "/share_data/zengfanlian/datasets/VLM/FineVision_Arrow_Format/robo2vlm"
TEMP_DIR = "/share_data/zengfanlian/datasets/VLM/FineVision_Arrow_Format/robo2vlm_temp"

# Test mode: Set to a number to only process first N files (None = process all)
MAX_TRAIN_FILES = None  # Set to None to process all train files, or a number like 2 for testing
MAX_TEST_FILES = None   # Set to None to process all test files

# Batch size for incremental processing (process and save this many samples at a time)
BATCH_SIZE = 50000  # Process 50k samples at a time to avoid memory issues

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


def convert_sample(sample):
    """
    Convert a single Robo2VLM sample to FineVision format.
    
    Args:
        sample: dict with keys ['id', 'question', 'choices', 'correct_answer', 'image']
    
    Returns:
        dict in FineVision format
    """
    # Parse choices if it's a string (from Parquet)
    choices = sample['choices']
    if isinstance(choices, str):
        choices = ast.literal_eval(choices)
    
    # Format question with choices
    user_text = format_multiple_choice(sample['question'], choices)
    
    # Format answer
    assistant_text = format_answer(sample['correct_answer'], choices)
    
    # Create conversation
    text_entry = {
        "user": user_text,
        "assistant": assistant_text
    }
    
    # Return in FineVision format
    return {
        "images": [sample['image']],
        "texts": [text_entry],
        "source": "robo2vlm",
        
        # Metadata (default values)
        "image_correspondence_ratings": [],
        "image_correspondence_min": 0,
        "visual_dependency_ratings": [],
        "visual_dependency_min": 0,
        "formatting_ratings": [],
        "formatting_min": 0,
        "relevance_ratings": [],
        "relevance_min": 0
    }


def convert_batch(batch):
    """
    Batch conversion for better performance.
    
    Args:
        batch: dict with lists of samples
    
    Returns:
        dict with lists in FineVision format
    """
    batch_size = len(batch['question'])
    
    # Initialize output lists
    images_list = []
    texts_list = []
    sources = []
    
    # Metadata lists
    image_corr_ratings = []
    image_corr_min = []
    visual_dep_ratings = []
    visual_dep_min = []
    formatting_ratings = []
    formatting_min = []
    relevance_ratings = []
    relevance_min = []
    
    # Process each sample in the batch
    for i in range(batch_size):
        # Parse choices if it's a string (from Parquet)
        choices = batch['choices'][i]
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        
        # Format question with choices
        user_text = format_multiple_choice(batch['question'][i], choices)
        
        # Format answer
        assistant_text = format_answer(batch['correct_answer'][i], choices)
        
        # Append to lists
        images_list.append([batch['image'][i]])
        texts_list.append([{"user": user_text, "assistant": assistant_text}])
        sources.append("robo2vlm")
        
        # Append metadata (defaults)
        # Ratings correspond to texts (conversations), not images - always 1 conversation per sample
        image_corr_ratings.append([0])
        image_corr_min.append(0)
        visual_dep_ratings.append([0])
        visual_dep_min.append(0)
        formatting_ratings.append([0])
        formatting_min.append(0)
        relevance_ratings.append([0])
        relevance_min.append(0)
    
    return {
        "images": images_list,
        "texts": texts_list,
        "source": sources,
        "image_correspondence_ratings": image_corr_ratings,
        "image_correspondence_min": image_corr_min,
        "visual_dependency_ratings": visual_dep_ratings,
        "visual_dependency_min": visual_dep_min,
        "formatting_ratings": formatting_ratings,
        "formatting_min": formatting_min,
        "relevance_ratings": relevance_ratings,
        "relevance_min": relevance_min
    }


def load_and_convert_split(split_name, parquet_files, temp_subdir):
    """
    Load and convert a split (train or test) in batches to avoid OOM.
    
    Args:
        split_name: str - 'train' or 'test'
        parquet_files: list of str - paths to parquet files
        temp_subdir: str - temporary directory for this split
    
    Returns:
        Dataset - converted dataset
    """
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
    
    # Process in batches
    print(f"\nProcessing in batches of {BATCH_SIZE} samples...")
    batch_paths = []
    num_batches = (total_samples + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, total_samples)
        
        print(f"\n[{split_name} Batch {batch_idx + 1}/{num_batches}] Processing samples {start_idx} to {end_idx}...")
        
        # Select this batch
        batch_ds = ds.select(range(start_idx, end_idx))
        
        # Convert to FineVision format
        converted_batch = batch_ds.map(
            convert_batch,
            batched=True,
            batch_size=1000,
            num_proc=64,  # Use 64 parallel processes for conversion
            remove_columns=batch_ds.column_names,
            desc=f"Converting {split_name} batch {batch_idx + 1}"
        )
        
        print(f"  Converted {len(converted_batch)} samples")
        
        # Save this batch immediately
        batch_path = os.path.join(temp_subdir, f"batch_{batch_idx:04d}")
        print(f"  Saving to {batch_path}...")
        converted_batch.save_to_disk(batch_path, num_proc=4)
        print(f"  ✓ Saved and freed from memory")
        
        batch_paths.append(batch_path)
        
        # Free memory
        del batch_ds, converted_batch
    
    # Concatenate all batches for this split
    print(f"\nConcatenating {len(batch_paths)} batches for {split_name}...")
    from datasets import load_from_disk
    all_batches = []
    for batch_path in batch_paths:
        all_batches.append(load_from_disk(batch_path))
    
    final_ds = concatenate_datasets(all_batches)
    print(f"✓ {split_name}: {len(final_ds)} samples")
    
    del all_batches
    return final_ds


def main():
    print("=" * 80)
    print("Robo2VLM-1 to FineVision Arrow Format Conversion (Optimized)")
    print("=" * 80)
    
    # Clean up and create temp directories
    if os.path.exists(TEMP_DIR):
        print(f"Removing existing temp directory: {TEMP_DIR}")
        shutil.rmtree(TEMP_DIR)
    
    train_temp = os.path.join(TEMP_DIR, "train")
    test_temp = os.path.join(TEMP_DIR, "test")
    os.makedirs(train_temp, exist_ok=True)
    os.makedirs(test_temp, exist_ok=True)
    
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
    
    # Define schema
    finevision_features = Features({
        "images": HFList(HFImage()),
        "texts": HFList({
            "user": Value("string"),
            "assistant": Value("string")
        }),
        "source": Value("string"),
        "image_correspondence_ratings": HFList(Value("int64")),
        "image_correspondence_min": Value("int64"),
        "visual_dependency_ratings": HFList(Value("int64")),
        "visual_dependency_min": Value("int64"),
        "formatting_ratings": HFList(Value("int64")),
        "formatting_min": Value("int64"),
        "relevance_ratings": HFList(Value("int64")),
        "relevance_min": Value("int64")
    })
    
    # Convert train split
    train_ds = load_and_convert_split("train", train_files, train_temp)
    
    # Convert test split  
    test_ds = load_and_convert_split("test", test_files, test_temp)
    
    # Create DatasetDict
    ds_dict = DatasetDict({
        "train": train_ds,
        "test": test_ds
    })
    
    print("\n" + "=" * 80)
    print(f"Final dataset: train={len(train_ds)}, test={len(test_ds)}")
    print("=" * 80)
    
    # Free memory before final save
    del train_ds, test_ds
    
    # Save to disk
    print("\n" + "=" * 80)
    print(f"Saving final dataset to: {OUTPUT_DIR}")
    print("=" * 80)
    
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Use adaptive num_proc for ~1GB per shard
    train_size = len(ds_dict['train'])
    test_size = len(ds_dict['test'])
    
    # Target: ~10,000 samples per shard (~1GB based on typical VLM data)
    target_samples_per_shard = 10000
    num_shards = max(1, train_size // target_samples_per_shard)
    num_proc = min(num_shards, 16)  # Max 16 to avoid OOM
    
    print(f"Train size: {train_size} samples")
    print(f"Test size: {test_size} samples")
    print(f"Target: ~{target_samples_per_shard} samples per shard (~1GB)")
    print(f"Will create ~{num_shards} shards")
    print(f"Using num_proc={num_proc} parallel writers (to avoid OOM)")
    print(f"Estimated total size: ~{train_size * 104 / 1024 / 1024:.1f} GB")
    
    ds_dict.save_to_disk(OUTPUT_DIR, num_proc=num_proc)
    
    print(f"\n✅ Saved successfully to {OUTPUT_DIR}")
    
    # Clean up temp directory
    print("\n" + "=" * 80)
    print("Cleaning up temporary files...")
    print("=" * 80)
    shutil.rmtree(TEMP_DIR)
    print(f"✓ Removed {TEMP_DIR}")
    
    # Verification
    print("\n" + "=" * 80)
    print("Verifying first training sample...")
    print("=" * 80)
    from datasets import load_from_disk
    loaded_ds = load_from_disk(OUTPUT_DIR)
    sample = loaded_ds['train'][0]
    
    print("Keys:", sample.keys())
    print("Images type:", type(sample['images']), "Count:", len(sample['images']))
    print("Texts count:", len(sample['texts']))
    print("\nFirst conversation:")
    print("  User:")
    print("   ", sample['texts'][0]['user'][:200], "...")
    print("  Assistant:")
    print("   ", sample['texts'][0]['assistant'])
    print("Source:", sample['source'])
    
    print("\n" + "=" * 80)
    print("✅ Conversion completed successfully!")
    print(f"Train samples: {len(loaded_ds['train'])}")
    print(f"Test samples: {len(loaded_ds['test'])}")
    print("=" * 80)


if __name__ == "__main__":
    main()
