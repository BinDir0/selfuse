import os
import json
import io
from PIL import Image
from datasets import Dataset, Features, Sequence, Value, Image as HFImage, concatenate_datasets
from datasets.features import List
import shutil

# ================= Configuration Section =================

# 1. Path to the RoboPoint JSON file
ROBOPOINT_JSON_PATH = "/share_data/guantianrui/datasets/VLM/robopoint-data/robopoint_1432k.json"

# 2. Root directory for images
# The 'image' field in RoboPoint JSON usually contains relative paths (e.g., "region_ref/xxx.png")
# You must specify the root folder where these relative paths are located.
IMAGE_ROOT_DIR = "/share_data/guantianrui/datasets/VLM/robopoint-data/images"

# 3. Output directory for Arrow format
OUTPUT_DIR = "/share_data/zengfanlian/datasets/VLM/FineVision_Arrow_Format/robopoint_cleaned"
TEMP_DIR = "/share_data/zengfanlian/datasets/VLM/FineVision_Arrow_Format/robopoint_temp"

# 4. Train/test split ratio
VAL_RATIO = 0.001  # 0.1% for validation
SEED = 42

# 5. Test mode: Set to a number to only process first N samples (None = process all)
MAX_SAMPLES = None  # Set to None to process all samples, or a number like 100 for testing

# 6. Batch size for incremental processing (process and save this many samples at a time)
BATCH_SIZE = 50000  # Process 50k samples at a time to avoid memory issues

# 7. Text quality filtering
MAX_TURN_CHARS = 500  # drop samples where user+assistant combined > this

# =========================================================

def clean_image_metadata(img):
    """
    Remove all PNG metadata by re-encoding the image.
    This prevents MAX_TEXT_CHUNK errors during training.
    
    Args:
        img: PIL Image object
        
    Returns:
        PIL Image object without metadata
    """
    if img is None:
        return None
    
    # Convert to RGB if needed
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    
    # Re-encode to remove all metadata
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    buffer.seek(0)
    clean_img = Image.open(buffer)
    clean_img.load()  # Force load the image data
    
    return clean_img


def load_robopoint_json():
    """Load RoboPoint JSON data into memory, filtering out samples without 'image' field."""
    print(f"Loading RoboPoint data from {ROBOPOINT_JSON_PATH}...")
    
    with open(ROBOPOINT_JSON_PATH, 'r') as f:
        data = json.load(f)
    
    total = len(data)
    print(f"Found {total} samples in JSON")
    
    # Filter out samples without 'image' field (these are pure text conversations)
    print("Filtering samples with 'image' field...")
    data = [sample for sample in data if 'image' in sample]
    
    filtered_count = total - len(data)
    print(f"Filtered out {filtered_count} samples without 'image' field")
    print(f"Remaining: {len(data)} samples with images")
    
    # Limit samples if MAX_SAMPLES is set
    if MAX_SAMPLES is not None:
        data = data[:MAX_SAMPLES]
        print(f"TEST MODE: Processing only first {MAX_SAMPLES} samples")
    
    return data


def convert_robopoint_batch(batch):
    """
    Batch conversion function for parallel processing.
    
    Args:
        batch: dict with lists of ['id', 'image', 'conversations']
    
    Returns:
        dict with lists in FineVision format
    """
    # Handle missing keys gracefully
    if 'image' not in batch:
        # Return empty results if no image field
        return {
            "images": [],
            "texts": [],
            "source": [],
            "image_correspondence_ratings": [],
            "image_correspondence_min": [],
            "visual_dependency_ratings": [],
            "visual_dependency_min": [],
            "formatting_ratings": [],
            "formatting_min": [],
            "relevance_ratings": [],
            "relevance_min": []
        }
    
    batch_size = len(batch['image'])
    
    # Initialize output lists
    images_out = []
    texts_out = []
    sources = []
    
    # Metadata lists (all defaults)
    image_corr_ratings = []
    image_corr_min = []
    visual_dep_ratings = []
    visual_dep_min = []
    formatting_ratings = []
    formatting_min = []
    relevance_ratings = []
    relevance_min = []
    
    for i in range(batch_size):
        rel_path = batch['image'][i]
        conversations = batch.get('conversations', [None] * batch_size)[i]
        
        # Skip invalid samples
        if not rel_path or not conversations or len(conversations) < 2:
            continue
        
        full_img_path = os.path.join(IMAGE_ROOT_DIR, rel_path)
        
        try:
            # Load image and remove metadata
            img_obj = Image.open(full_img_path).convert("RGB")
            img_obj = clean_image_metadata(img_obj)  # Clean PNG metadata
            
            # Extract conversation
            human_text = conversations[0].get('value', '')
            human_text = human_text.replace("<image>\n", "").replace("<image>", "").strip()
            gpt_text = conversations[1].get('value', '')
            
            if len(human_text) + len(gpt_text) > MAX_TURN_CHARS:
                continue
            
            # Append to output
            images_out.append([img_obj])
            texts_out.append([{"user": human_text, "assistant": gpt_text}])
            sources.append("robopoint")
            
            # Metadata (defaults)
            # Ratings correspond to texts (conversations), not images - always 1 conversation per sample
            image_corr_ratings.append([0])
            image_corr_min.append(0)
            visual_dep_ratings.append([0])
            visual_dep_min.append(0)
            formatting_ratings.append([0])
            formatting_min.append(0)
            relevance_ratings.append([0])
            relevance_min.append(0)
            
        except Exception as e:
            # Skip failed samples
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

def main():
    print("=" * 80)
    print("RoboPoint to FineVision Arrow Format Conversion (Incremental)")
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

    # Load all JSON data
    print("\n" + "=" * 80)
    print("Loading JSON data...")
    print("=" * 80)
    robopoint_data = load_robopoint_json()
    total_samples = len(robopoint_data)
    
    # Process in batches and save incrementally
    print("\n" + "=" * 80)
    print(f"Processing in batches of {BATCH_SIZE} samples...")
    print("=" * 80)
    
    batch_datasets = []
    num_batches = (total_samples + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, total_samples)
        batch_data = robopoint_data[start_idx:end_idx]
        
        print(f"\n[Batch {batch_idx + 1}/{num_batches}] Processing samples {start_idx} to {end_idx}...")
        
        # Create dataset from this batch
        initial_ds = Dataset.from_list(batch_data)
        
        # Convert to FineVision format
        ds = initial_ds.map(
            convert_robopoint_batch,
            batched=True,
            batch_size=1000,
            num_proc=64,  # Use 64 parallel processes for each batch
            remove_columns=initial_ds.column_names,
            desc=f"Converting batch {batch_idx + 1}"
        )
        
        print(f"  Converted {len(ds)} samples (filtered from {len(batch_data)})")
        
        # Save this batch immediately with minimal num_proc to avoid OOM
        batch_path = os.path.join(TEMP_DIR, f"batch_{batch_idx:04d}")
        print(f"  Saving to {batch_path}...")
        ds.save_to_disk(batch_path, num_proc=4)  # Only 4 processes for saving
        print(f"  ✓ Batch {batch_idx + 1} saved and freed from memory")
        
        # Keep reference for later concatenation
        batch_datasets.append(batch_path)
        
        # Free memory immediately
        del initial_ds, ds
    
    # Concatenate all batches
    print("\n" + "=" * 80)
    print("Concatenating all batches...")
    print("=" * 80)
    
    from datasets import load_from_disk
    all_datasets = []
    for i, batch_path in enumerate(batch_datasets):
        print(f"Loading batch {i+1}/{len(batch_datasets)}: {batch_path}...")
        ds = load_from_disk(batch_path)
        all_datasets.append(ds)
    
    print("Concatenating datasets...")
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
    