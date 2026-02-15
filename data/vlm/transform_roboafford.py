"""
Convert RoboAfford dataset to FineVision Arrow format.

Processing 4 data sources (skipping PixMo-Points):
- LVIS_absxy_513K.json: Visual Genome images
- object_affordance_prediction_absxy_561K.json: PACO-LVIS
- object_ref_max_points_10_absxy_347K.json: RoboPoint
- region_ref_max_points_10_absxy_320K.json: RoboPoint

Features:
- Multi-turn conversations split into separate samples
- Coordinates normalized to [0, 1] range
- FineVision Arrow format output
"""

import os
import json
import shutil
import re
import time
import io
from PIL import Image, PngImagePlugin
from tqdm import tqdm

# CRITICAL: Increase PIL's limit for large PNG iCCP chunks BEFORE loading any images
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)  # 100MB (default is 1MB)

# CRITICAL: Set HuggingFace cache BEFORE importing datasets
HF_CACHE_DIR = "/share_data/zengfanlian/.cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# NOW import datasets after environment variables are set
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets, Features, Value, List as HFList, Image as HFImage

# ================= Configuration =================
ROBOAFFORD_ROOT = "/share_data/guantianrui/datasets/VLM/RoboAfford"
OUTPUT_DIR = "/share_data/zengfanlian/datasets/VLM/FineVision_Arrow_Format/roboafford"
TEMP_DIR = "/share_data/zengfanlian/datasets/VLM/FineVision_Arrow_Format/roboafford_temp"

# Image root directories
IMAGE_ROOTS = {
    'robopoint': '/share_data/guantianrui/datasets/VLM/robopoint-data/images',
    'coco': '/share_data/guantianrui/datasets/VLM/COCO2017/coco2017',
    'visual_genome': '/share_data/guantianrui/datasets/VLM/robopoint-data/images',
}

# JSON files to process
JSON_FILES = [
    'LVIS_absxy_513K.json',
    'object_affordance_prediction_absxy_561K.json',
    'object_ref_max_points_10_absxy_347K.json',
    'region_ref_max_points_10_absxy_320K.json',
]

# Batch size for processing (process this many original samples at a time)
BATCH_SIZE = 50000

# Train/test split ratio
VAL_RATIO = 0.001  # 0.1% for validation
SEED = 42

# ================================================


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


def normalize_coordinates_in_text(text, img_width, img_height):
    """Normalize absolute pixel coordinates to [0, 1] range."""
    def normalize_match(match):
        coords_str = match.group(1)
        coords = []
        for coord_match in re.finditer(r'\(([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+),\s*([\d.]+))?\)', coords_str):
            if coord_match.group(3):  # Bounding box
                x1, y1, x2, y2 = [float(coord_match.group(i)) for i in range(1, 5)]
                coords.append(f"({round(x1/img_width, 4)}, {round(y1/img_height, 4)}, {round(x2/img_width, 4)}, {round(y2/img_height, 4)})")
            else:  # Point
                x, y = float(coord_match.group(1)), float(coord_match.group(2))
                coords.append(f"({round(x/img_width, 4)}, {round(y/img_height, 4)})")
        return f"[{', '.join(coords)}]"
    
    return re.sub(r'\[((?:\([^)]+\)(?:,\s*)?)+)\]', normalize_match, text)


def load_image(image_path):
    """Load image from file path."""
    if image_path.startswith('region_ref/') or image_path.startswith('object_ref/'):
        full_path = os.path.join(IMAGE_ROOTS['robopoint'], image_path)
    elif image_path.startswith(('train2017/', 'val2017/', 'test2017/')):
        full_path = os.path.join(IMAGE_ROOTS['coco'], image_path)
    elif image_path.startswith('VG_100K/'):
        full_path = os.path.join(IMAGE_ROOTS['visual_genome'], image_path)
    else:
        return None
    
    if os.path.exists(full_path):
        img = Image.open(full_path).convert('RGB')
        return clean_image_metadata(img)  # Clean PNG metadata
    return None


def convert_batch(batch, source):
    """
    Convert a batch of samples from RoboAfford to FineVision format.
    Uses Dataset.map() style batched processing.
    Randomly selects ONE conversation turn per image to avoid data duplication.
    """
    import random
    random.seed(SEED)  # Set seed for reproducibility
    
    batch_size = len(batch['id'])
    
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
        sample_id = batch['id'][i]
        image_path = batch['image'][i]
        conversations = batch['conversations'][i]
        
        # Load image
        img = load_image(image_path)
        if img is None:
            continue
        
        img_width, img_height = img.size
        
        # Extract all valid conversation pairs
        valid_pairs = []
        for j in range(0, len(conversations), 2):
            if j + 1 >= len(conversations):
                break
            
            user_msg = conversations[j]
            assistant_msg = conversations[j + 1]
            
            if user_msg['from'] == 'human' and assistant_msg['from'] == 'gpt':
                valid_pairs.append((user_msg, assistant_msg))
        
        # Randomly select ONE conversation pair
        if not valid_pairs:
            continue
        
        user_msg, assistant_msg = random.choice(valid_pairs)
        
        # Normalize coordinates
        user_text = user_msg['value'].replace('<image>\n', '').strip()
        assistant_text = assistant_msg['value'].strip()
        
        user_text_normalized = normalize_coordinates_in_text(user_text, img_width, img_height)
        assistant_text_normalized = normalize_coordinates_in_text(assistant_text, img_width, img_height)
        
        # Append to lists
        images_list.append([img])
        texts_list.append([{"user": user_text_normalized, "assistant": assistant_text_normalized}])
        sources.append(f"roboafford-{source}")
        
        # Append metadata (defaults) - fill with 0s based on number of conversations (always 1)
        # Note: ratings correspond to texts (conversations), not images
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


def process_json_file(json_file, batch_counter):
    """Process a single JSON file in batches, like robopoint."""
    print(f"\n{'='*80}")
    print(f"Processing: {json_file}")
    print(f"{'='*80}")
    
    # Load all data
    with open(os.path.join(ROBOAFFORD_ROOT, json_file), 'r') as f:
        data = json.load(f)
    
    total_samples = len(data)
    print(f"Total samples: {total_samples}")
    
    # Determine source name
    if 'LVIS' in json_file:
        source = 'lvis'
    elif 'affordance' in json_file:
        source = 'paco-lvis'
    elif 'object_ref' in json_file:
        source = 'robopoint-obj'
    elif 'region_ref' in json_file:
        source = 'robopoint-reg'
    else:
        source = 'unknown'
    
    # Calculate number of batches
    num_batches = (total_samples + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing in {num_batches} batches of {BATCH_SIZE} samples each")
    
    # Process in batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, total_samples)
        batch_data = data[start_idx:end_idx]
        
        print(f"\n[Batch {batch_idx + 1}/{num_batches}] Processing samples {start_idx} to {end_idx}...")
        
        # Create dataset from this batch (original data)
        initial_ds = Dataset.from_list(batch_data)
        
        # Convert to FineVision format using map (with progress bar!)
        converted_batch = initial_ds.map(
            lambda batch: convert_batch(batch, source),
            batched=True,
            batch_size=1000,
            num_proc=64,  # Use 32 parallel processes
            remove_columns=initial_ds.column_names,
            desc=f"Converting batch {batch_idx + 1}/{num_batches}"
        )
        
        print(f"  Converted {len(converted_batch)} samples from {len(batch_data)} original samples")
        
        # Save this batch immediately
        batch_path = os.path.join(TEMP_DIR, f"batch_{batch_counter:04d}")
        print(f"  Saving to {batch_path}...")
        converted_batch.save_to_disk(batch_path, num_proc=4)
        print(f"  ✓ Batch {batch_counter} saved and freed from memory")
        
        batch_counter += 1
        
        # Free memory
        del batch_data, initial_ds, converted_batch
    
    return batch_counter


def main():
    print("=" * 80)
    print("RoboAfford to FineVision Arrow Format Conversion")
    print("=" * 80)
    
    # Create temp directory
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Process each JSON file in batches
    batch_counter = 0
    
    for json_file in JSON_FILES:
        batch_counter = process_json_file(json_file, batch_counter)
    
    print(f"\n✓ All files processed. Total batches: {batch_counter}")
    
    # Concatenate all batches
    print("\n" + "=" * 80)
    print("Concatenating all batches...")
    print("=" * 80)
    
    batch_paths = sorted([os.path.join(TEMP_DIR, d) for d in os.listdir(TEMP_DIR)])
    all_batches = [load_from_disk(p) for p in batch_paths]
    final_ds = concatenate_datasets(all_batches)
    
    print(f"✓ Total samples: {len(final_ds)}")
    
    # Free memory
    del all_batches
    
    # Split into train/test
    print("\n" + "=" * 80)
    print(f"Splitting dataset (val_ratio={VAL_RATIO})...")
    print("=" * 80)
    
    ds_dict = final_ds.train_test_split(test_size=VAL_RATIO, seed=SEED)
    print(f"Train samples: {len(ds_dict['train'])}")
    print(f"Test samples: {len(ds_dict['test'])}")
    
    # Free memory
    del final_ds
    
    # Save to disk
    print("\n" + "=" * 80)
    print(f"Saving dataset to: {OUTPUT_DIR}")
    print("=" * 80)
    
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Calculate optimal sharding
    train_size = len(ds_dict['train'])
    target_samples_per_shard = 10000  # ~1GB per shard (same as robopoint)
    num_shards = max(1, train_size // target_samples_per_shard)
    num_proc = min(num_shards, 16)  # Max 16 to avoid OOM
    
    print(f"Train size: {train_size} samples")
    print(f"Test size: {len(ds_dict['test'])} samples")
    print(f"Target: ~{target_samples_per_shard} samples per shard (~1GB)")
    print(f"Will create ~{num_shards} shards")
    print(f"Using num_proc={num_proc} for saving")
    
    ds_dict.save_to_disk(OUTPUT_DIR, num_proc=num_proc)
    
    print(f"\n✅ Saved successfully!")
    
    # Clean up
    shutil.rmtree(TEMP_DIR)
    print(f"✓ Cleaned up temp files")
    
    # Verification
    print("\n" + "=" * 80)
    print("Verification")
    print("=" * 80)
    loaded_ds = load_from_disk(OUTPUT_DIR)
    sample = loaded_ds['train'][0]
    
    print("First sample:")
    print(f"  User: {sample['texts'][0]['user'][:200]}...")
    print(f"  Assistant: {sample['texts'][0]['assistant'][:100]}...")
    print(f"  Source: {sample['source']}")
    print(f"\n✅ Conversion completed!")
    print(f"Train samples: {len(loaded_ds['train'])}")
    print(f"Test samples: {len(loaded_ds['test'])}")
    print("=" * 80)


if __name__ == "__main__":
    main()
