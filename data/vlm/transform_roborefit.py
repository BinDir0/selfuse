"""
Convert RoboRefit-Grasping dataset to FineVision Arrow format.

RoboRefit-Grasping is a grasping point selection dataset with:
- question: str (grasping instruction)
- options: list of [x, y] pixel coordinates
- answer: int (index of correct grasping point)
- image: PIL Image (640x480)
- output_image: PIL Image with visualizations

Output: FineVision Arrow format with text conversations.
Coordinates are normalized to [0, 1] range.
"""

import os
import shutil

# CRITICAL: Set HuggingFace cache BEFORE importing datasets
# This must be done before any HuggingFace library imports to avoid filling up local disk
HF_CACHE_DIR = "/share_data/zengfanlian/.cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# NOW import datasets after environment variables are set
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, List as HFList, Image as HFImage
from tqdm import tqdm

# ================= Configuration =================
ROBOREFIT_ROOT = "/share_data/guantianrui/datasets/VLM/roborefit-grasping"
OUTPUT_DIR = "/share_data/zengfanlian/datasets/VLM/FineVision_Arrow_Format/roborefit"

# Image dimensions (all images are 640x480)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Test mode: Set to a number to only process first N samples (None = process all)
MAX_SAMPLES = None  # Set to None to process all samples, or a number like 100 for testing

# Dataset split configuration
VAL_RATIO = 0.001  # 0.1% for validation (same as other datasets)
SEED = 42

# ================================================


def normalize_coordinates(x, y, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """
    Normalize pixel coordinates to [0, 1] range.
    
    Args:
        x: int - pixel x coordinate
        y: int - pixel y coordinate
        width: int - image width
        height: int - image height
    
    Returns:
        tuple of (normalized_x, normalized_y) rounded to 3 decimal places
    """
    norm_x = round(x / width, 4)
    norm_y = round(y / height, 4)
    return (norm_x, norm_y)


def format_grasping_question(question, options):
    """
    Format grasping question with letter-labeled coordinate choices.
    Coordinates are normalized to [0, 1] range.
    
    Args:
        question: str - the grasping instruction
        options: list of [x, y] - pixel coordinates of candidate grasping points
    
    Returns:
        str - formatted question with letter-labeled choices
    """
    # Normalize all coordinates
    normalized_options = [normalize_coordinates(x, y) for x, y in options]
    
    # Create letter-labeled choices (A, B, C, D, ...)
    formatted_choices = "\n".join([
        f"{chr(65 + i)}. {coord}"
        for i, coord in enumerate(normalized_options)
    ])
    
    formatted_question = f"Question: {question}\nChoices:\n{formatted_choices}\nAnswer with the letter."
    return formatted_question


def format_answer(correct_answer_idx):
    """
    Format the answer using the letter label (A, B, C, D, ...).
    
    Args:
        correct_answer_idx: int - index of correct answer
    
    Returns:
        str - formatted answer with letter label
    """
    return f"Answer: {chr(65 + correct_answer_idx)}"


def convert_batch(batch):
    """
    Batch conversion function for parallel processing.
    
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
        # Format question with normalized coordinate choices
        user_text = format_grasping_question(batch['question'][i], batch['options'][i])
        
        # Format answer
        assistant_text = format_answer(batch['answer'][i])
        
        # Append to lists
        # Note: We only use the input image, not the output_image with annotations
        images_list.append([batch['image'][i]])
        texts_list.append([{"user": user_text, "assistant": assistant_text}])
        sources.append("roborefit-grasping")
        
        # Append metadata (defaults) - fill with 0s based on number of images (always 1)
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


def main():
    print("=" * 80)
    print("RoboRefit-Grasping to FineVision Arrow Format Conversion")
    print("=" * 80)
    
    # Define Features schema to avoid type inference issues
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
    
    # Load dataset
    print(f"\nLoading dataset from {ROBOREFIT_ROOT}...")
    ds = load_dataset(ROBOREFIT_ROOT, split='train')
    
    total_samples = len(ds)
    print(f"Total samples: {total_samples}")
    
    # Limit samples if MAX_SAMPLES is set
    if MAX_SAMPLES is not None:
        ds = ds.select(range(min(MAX_SAMPLES, total_samples)))
        print(f"TEST MODE: Processing only first {MAX_SAMPLES} samples")
    
    # Remove columns we don't need
    print("\nConverting to FineVision format...")
    columns_to_remove = ['source_id', 'split', 'image_path', 'output_image_path', 
                        'source', 'scene', 'class', 'output_image']
    
    # Keep only the columns we need for conversion
    ds = ds.remove_columns([col for col in columns_to_remove if col in ds.column_names])
    
    # Convert to FineVision format
    converted_ds = ds.map(
        convert_batch,
        batched=True,
        batch_size=100,
        num_proc=64,  # Use parallel processing
        remove_columns=ds.column_names,
        desc="Converting samples"
    )
    
    print(f"✓ Converted {len(converted_ds)} samples")
    
    # Split into train/test
    print("\n" + "=" * 80)
    print(f"Splitting dataset (val_ratio={VAL_RATIO})...")
    print("=" * 80)
    
    ds_dict = converted_ds.train_test_split(test_size=VAL_RATIO, seed=SEED)
    
    print(f"Train samples: {len(ds_dict['train'])}")
    print(f"Test samples: {len(ds_dict['test'])}")
    print("=" * 80)
    
    # Save to disk
    print("\n" + "=" * 80)
    print(f"Saving dataset to: {OUTPUT_DIR}")
    print("=" * 80)
    
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Calculate optimal sharding for saving
    train_size = len(ds_dict['train'])
    target_samples_per_shard = 10000  # ~1GB per shard (same as other datasets)
    num_shards = max(1, train_size // target_samples_per_shard)
    num_proc = min(num_shards, 16)  # Max 16 processes
    
    print(f"Train size: {train_size} samples")
    print(f"Target: ~{target_samples_per_shard} samples per shard (~1GB)")
    print(f"Will create ~{num_shards} shards")
    print(f"Using num_proc={num_proc} for saving")
    
    ds_dict.save_to_disk(OUTPUT_DIR, num_proc=num_proc)
    
    print(f"\n✅ Saved successfully to {OUTPUT_DIR}")
    
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
    for line in sample['texts'][0]['user'].split('\n'):
        print("   ", line)
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
