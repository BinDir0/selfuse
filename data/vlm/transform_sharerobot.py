"""
Convert ShareRobot Affordance and Trajectory datasets to FineVision format.

Converts bbox and trajectory coordinates to normalized [0,1] format for PaliGemma.
"""
import os
import json
import shutil
from PIL import Image
from datasets import Dataset, Features, Value, Image as HFImage, concatenate_datasets, load_from_disk
from datasets.features import List

# ================= Configuration Section =================

# 1. ShareRobot root directory
SHAREROBOT_ROOT = "/share_data/guantianrui/datasets/VLM/ShareRobot"

# 2. Output directory for Arrow format
OUTPUT_DIR = "/share_data/zengfanlian/datasets/VLM/FineVision_Arrow_Format/sharerobot"
TEMP_DIR = "/share_data/zengfanlian/datasets/VLM/FineVision_Arrow_Format/sharerobot_temp"

# 3. Only process affordance and trajectory (planning has multiple images per sample)
PROCESS_SUBDATASETS = ['affordance', 'trajectory']

# 4. Train/test split ratio
VAL_RATIO = 0.001  # 0.1% for validation
SEED = 42

# 5. Test mode: Set to a number to only process first N samples (None = process all)
MAX_SAMPLES = None  # Set to None to process all samples, or a number like 100 for testing

# 6. Batch size for incremental processing (process and save this many samples at a time)
BATCH_SIZE = 5000  # Process 5k samples at a time

# =========================================================


def normalize_bbox(bbox, width, height):
    """
    Normalize bbox coordinates to [0, 1] range.
    
    Input: {x, y, width, height} (top-left corner + size)
    Output: (y_min, x_min, y_max, x_max) format
    """
    x = bbox['x']
    y = bbox['y']
    w = bbox['width']
    h = bbox['height']
    
    # Convert to (y_min, x_min, y_max, x_max) and normalize
    y_min = y / height
    x_min = x / width
    y_max = (y + h) / height
    x_max = (x + w) / width
    
    return y_min, x_min, y_max, x_max


def normalize_point(point, width, height):
    """Normalize a single point to [0, 1] range."""
    x_norm = point[0] / width
    y_norm = point[1] / height
    return x_norm, y_norm


def bbox_to_text(y_min, x_min, y_max, x_max):
    """
    Convert normalized bbox to text format (RoboAfford style).
    
    Format: [(y_min, x_min, y_max, x_max)]
    """
    return f"[({y_min:.2f}, {x_min:.2f}, {y_max:.2f}, {x_max:.2f})]"


def trajectory_to_text(points):
    """
    Convert normalized trajectory points to text format (RoboPoint style).
    
    Format: [(x1, y1), (x2, y2), ...]
    """
    point_strs = [f"({x:.3f}, {y:.3f})" for x, y in points]
    return "[" + ", ".join(point_strs) + "]"


def load_image_safe(base_path, image_path):
    """Safely load an image."""
    possible_paths = [
        os.path.join(base_path, image_path),
        os.path.join(SHAREROBOT_ROOT, image_path),
        os.path.join(SHAREROBOT_ROOT, 'images', image_path),
    ]
    
    for full_path in possible_paths:
        if os.path.exists(full_path):
            try:
                return Image.open(full_path).convert('RGB')
            except Exception:
                continue
    return None


def convert_affordance_batch(batch, image_base_path):
    """
    Batch conversion function for affordance samples.
    
    Args:
        batch: dict with lists of sample data
        image_base_path: base path for images
    
    Returns:
        dict with lists in FineVision format
    """
    batch_size = len(batch['image_path'])
    
    # Initialize output lists
    images_out = []
    texts_out = []
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
    
    for i in range(batch_size):
        try:
            # Load image
            img = load_image_safe(image_base_path, batch['image_path'][i])
            if img is None:
                continue
            
            # Get dimensions
            orig_width = batch['meta_data'][i]['original_width']
            orig_height = batch['meta_data'][i]['original_height']
            
            # Normalize bbox
            bbox = batch['affordance'][i]
            y_min, x_min, y_max, x_max = normalize_bbox(bbox, orig_width, orig_height)
            
            # Create conversation (RoboAfford style)
            instruction = batch['instruction'][i]
            user_text = f"Please provide the bounding box coordinate of the region to {instruction}."
            assistant_text = bbox_to_text(y_min, x_min, y_max, x_max)
            
            # Append outputs
            images_out.append([img])
            texts_out.append([{"user": user_text, "assistant": assistant_text}])
            sources.append("affordance")
            
            # Metadata (defaults)
            image_corr_ratings.append([0])
            image_corr_min.append(0)
            visual_dep_ratings.append([0])
            visual_dep_min.append(0)
            formatting_ratings.append([0])
            formatting_min.append(0)
            relevance_ratings.append([0])
            relevance_min.append(0)
            
        except Exception:
            continue
    
    return {
        "images": images_out,
        "texts": texts_out,
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


def convert_trajectory_batch(batch, image_base_path):
    """
    Batch conversion function for trajectory samples.
    
    Args:
        batch: dict with lists of sample data
        image_base_path: base path for images
    
    Returns:
        dict with lists in FineVision format
    """
    batch_size = len(batch['image_path'])
    
    # Initialize output lists
    images_out = []
    texts_out = []
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
    
    for i in range(batch_size):
        try:
            # Load image
            img = load_image_safe(image_base_path, batch['image_path'][i])
            if img is None:
                continue
            
            # Get dimensions
            orig_width = batch['meta_data'][i]['original_width']
            orig_height = batch['meta_data'][i]['original_height']
            
            # Normalize trajectory points
            trajectory = batch['trajectory'][i]
            normalized_points = [normalize_point(point, orig_width, orig_height) for point in trajectory]
            
            # Create conversation (RoboPoint style)
            instruction = batch['instruction'][i]
            user_text = (
                f"Predict the trajectory points to {instruction}. "
                f"Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], "
                f"where each tuple contains the x and y coordinates of a point in the trajectory. "
                f"The coordinates should be between 0 and 1, indicating the normalized pixel locations."
            )
            assistant_text = trajectory_to_text(normalized_points)
            
            # Append outputs
            images_out.append([img])
            texts_out.append([{"user": user_text, "assistant": assistant_text}])
            sources.append("trajectory")
            
            # Metadata (defaults)
            image_corr_ratings.append([0])
            image_corr_min.append(0)
            visual_dep_ratings.append([0])
            visual_dep_min.append(0)
            formatting_ratings.append([0])
            formatting_min.append(0)
            relevance_ratings.append([0])
            relevance_min.append(0)
            
        except Exception:
            continue
    
    return {
        "images": images_out,
        "texts": texts_out,
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


def process_subdataset(subdataset_name, convert_fn):
    """Process a single subdataset (affordance or trajectory)."""
    print(f"\n{'='*80}")
    print(f"Processing {subdataset_name.upper()} dataset")
    print(f"{'='*80}")
    
    json_path = os.path.join(SHAREROBOT_ROOT, subdataset_name, f'{subdataset_name}.json')
    image_base_path = os.path.join(SHAREROBOT_ROOT, subdataset_name, 'images')
    
    # Load JSON data
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    total_samples = len(data)
    print(f"Found {total_samples:,} samples")
    
    # Limit samples if MAX_SAMPLES is set
    if MAX_SAMPLES is not None:
        data = data[:MAX_SAMPLES]
        print(f"TEST MODE: Processing only first {MAX_SAMPLES} samples")
    
    # Process in batches
    print(f"Processing in batches of {BATCH_SIZE} samples...")
    
    batch_datasets = []
    num_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(data))
        batch_data = data[start_idx:end_idx]
        
        print(f"[Batch {batch_idx + 1}/{num_batches}] Processing samples {start_idx} to {end_idx}...")
        
        # Create dataset from this batch
        initial_ds = Dataset.from_list(batch_data)
        
        # Convert to FineVision format
        ds = initial_ds.map(
            lambda batch: convert_fn(batch, image_base_path),
            batched=True,
            batch_size=500,
            num_proc=32,  # Use 32 parallel processes
            remove_columns=initial_ds.column_names,
            desc=f"Converting {subdataset_name} batch {batch_idx + 1}"
        )
        
        print(f"  Converted {len(ds)} samples (filtered from {len(batch_data)})")
        
        # Save this batch immediately
        batch_path = os.path.join(TEMP_DIR, f"{subdataset_name}_batch_{batch_idx:04d}")
        print(f"  Saving to {batch_path}...")
        ds.save_to_disk(batch_path, num_proc=4)
        print(f"  ✓ Batch {batch_idx + 1} saved and freed from memory")
        
        # Keep reference for later concatenation
        batch_datasets.append(batch_path)
        
        # Free memory
        del initial_ds, ds
    
    # Concatenate all batches for this subdataset
    print(f"\nConcatenating {len(batch_datasets)} batches for {subdataset_name}...")
    all_datasets = []
    for i, batch_path in enumerate(batch_datasets):
        print(f"  Loading batch {i+1}/{len(batch_datasets)}: {batch_path}...")
        ds = load_from_disk(batch_path)
        all_datasets.append(ds)
    
    print(f"  Concatenating {subdataset_name} datasets...")
    full_ds = concatenate_datasets(all_datasets)
    print(f"  Total {subdataset_name} samples: {len(full_ds):,}")
    
    # Free memory
    del all_datasets
    
    return full_ds


def main():
    print("=" * 80)
    print("ShareRobot to FineVision Arrow Format Conversion (Incremental)")
    print("=" * 80)
    
    # --- Define Schema (Features) ---
    finevision_features = Features({
        "images": List(HFImage()),
        "texts": List({
            "user": Value("string"),
            "assistant": Value("string")
        }),
        "source": Value("string"),
        "image_correspondence_ratings": List(Value("int64")), 
        "image_correspondence_min": Value("int64"),
        "visual_dependency_ratings": List(Value("int64")),
        "visual_dependency_min": Value("int64"),
        "formatting_ratings": List(Value("int64")),
        "formatting_min": Value("int64"),
        "relevance_ratings": List(Value("int64")),
        "relevance_min": Value("int64")
    })

    # Clean up temp directory if exists
    if os.path.exists(TEMP_DIR):
        print(f"Removing existing temp directory: {TEMP_DIR}")
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Process subdatasets
    all_datasets = []
    
    if 'affordance' in PROCESS_SUBDATASETS:
        affordance_ds = process_subdataset('affordance', convert_affordance_batch)
        all_datasets.append(affordance_ds)
    
    if 'trajectory' in PROCESS_SUBDATASETS:
        trajectory_ds = process_subdataset('trajectory', convert_trajectory_batch)
        all_datasets.append(trajectory_ds)
    
    if not all_datasets:
        print("\n✗ No datasets processed")
        return
    
    # Concatenate all subdatasets
    print("\n" + "=" * 80)
    print("Concatenating all subdatasets...")
    print("=" * 80)
    
    full_ds = concatenate_datasets(all_datasets)
    print(f"Total samples after concatenation: {len(full_ds)}")
    
    # Free memory
    del all_datasets
    
    # Split into train/test
    print("\n" + "=" * 80)
    print(f"Splitting dataset (val_ratio={VAL_RATIO})...")
    print("=" * 80)
    ds_dict = full_ds.train_test_split(test_size=VAL_RATIO, seed=SEED)
    
    print(f"Train samples: {len(ds_dict['train'])}")
    print(f"Test samples: {len(ds_dict['test'])}")
    
    # Free memory
    del full_ds
    
    # Save final dataset
    print("\n" + "=" * 80)
    print(f"Saving final dataset to: {OUTPUT_DIR}")
    print("=" * 80)
    
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Calculate optimal sharding
    train_size = len(ds_dict['train'])
    target_samples_per_shard = 10000
    num_shards = max(1, train_size // target_samples_per_shard)
    num_proc = min(num_shards, 16)  # Max 16 processes to avoid OOM
    
    print(f"Dataset size: {train_size} samples")
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
    loaded_ds = load_from_disk(OUTPUT_DIR)
    sample = loaded_ds['train'][0]
    
    print("Keys:", sample.keys())
    print("Images type:", type(sample['images']), "Count:", len(sample['images']))
    print("Texts count:", len(sample['texts']))
    print("First text entry:")
    print("  User:", sample['texts'][0]['user'][:100], "...")
    print("  Assistant:", sample['texts'][0]['assistant'][:100], "...")
    print("Source:", sample['source'])
    
    print("\n" + "=" * 80)
    print("✅ Conversion completed successfully!")
    print("=" * 80) 


if __name__ == "__main__":
    main()
