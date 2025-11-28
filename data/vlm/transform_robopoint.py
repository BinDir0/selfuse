import os
import json
from PIL import Image
from datasets import Dataset, Features, Sequence, Value, Image as HFImage
from datasets.features import List

# ================= Configuration Section =================

# 1. Path to the RoboPoint JSON file
ROBOPOINT_JSON_PATH = "/share_data/guantianrui/datasets/VLM/robopoint-data/robopoint_1432k.json"

# 2. Root directory for images
# The 'image' field in RoboPoint JSON usually contains relative paths (e.g., "region_ref/xxx.png")
# You must specify the root folder where these relative paths are located.
IMAGE_ROOT_DIR = "/share_data/guantianrui/datasets/VLM/robopoint-data/images"

# 3. Output path for the generated Parquet file
OUTPUT_PATH = "/share_data/guantianrui/datasets/VLM/robopoint-data/robopoint_finevision_format.parquet"

# =========================================================

def generate_finevision_data():
    """
    Generator function: Iterates through the local RoboPoint JSON and images,
    yielding data strictly in the FineVision schema.
    """
    print(f"Loading RoboPoint data from {ROBOPOINT_JSON_PATH}...")
    
    with open(ROBOPOINT_JSON_PATH, 'r') as f:
        # If the file is extremely large, consider reading line-by-line. 
        # For standard JSON lists, loading into memory is fine.
        data = json.load(f)

    total = len(data)
    print(f"Found {total} samples. Starting conversion...")

    for idx, item in enumerate(data):
        # --- 1. Process Image ---
        # RoboPoint has a single image path string.
        # FineVision requires a List of PIL Image objects: List[Image]
        rel_path = item.get("image", "")
        
        # Skip if image path is empty
        if not rel_path or not rel_path.strip():
            print(f"Warning: Sample {idx} has empty image path, skipping.")
            continue
        
        full_img_path = os.path.join(IMAGE_ROOT_DIR, rel_path)
        
        try:
            # Open image and force convert to RGB.
            # This prevents errors with RGBA PNGs when converting to tensors later.
            img_obj = Image.open(full_img_path).convert("RGB")
            images_list = [img_obj]
        except Exception as e:
            print(f"Warning: Failed to load image {full_img_path}, skipping. Error: {e}")
            continue

        # --- 2. Process Text (Conversations -> Texts) ---
        # Input Format (RoboPoint): list of dicts [{"from": "human", ...}, {"from": "gpt", ...}]
        # Target Format (FineVision): list of dicts [{'user': ..., 'assistant': ...}]
        
        conversations = item.get("conversations", [])
        if len(conversations) < 2:
            print(f"Warning: Sample {idx} has insufficient conversations (< 2), skipping.")
            continue

        # Verify conversation roles (optional check, but helpful for debugging)
        if conversations[0].get("from") != "human":
            print(f"Warning: Sample {idx} first conversation is not from 'human', but proceeding anyway.")
        if conversations[1].get("from") != "gpt":
            print(f"Warning: Sample {idx} second conversation is not from 'gpt', but proceeding anyway.")

        # Extract Human text
        human_text = conversations[0].get('value', '')
        # Clean the <image> token (FineVision style usually relies on the processor, not raw text tags)
        human_text = human_text.replace("<image>\n", "").replace("<image>", "").strip()

        # Extract GPT text
        gpt_text = conversations[1].get('value', '')

        # Construct the text entry for FineVision
        text_entry = {
            "user": human_text,
            "assistant": gpt_text
        }

        # --- 3. Yield the final dictionary ---
        yield {
            # Core Data
            "images": images_list,       # Must be a List
            "texts": [text_entry],       # Must be a List of dicts
            "source": "robopoint",       # Track the origin

            # Metadata (Fill with empty lists/default values for compatibility)
            # Note: Sequence types cannot be None, so we use empty lists.
            # For Value(int64) types, we use 0 as default (real FineVision data uses integers, not None).
            "image_correspondence_ratings": [],
            "image_correspondence_min": 0,
            "visual_dependency_ratings": [],
            "visual_dependency_min": 0,
            "formatting_ratings": [],
            "formatting_min": 0,
            "relevance_ratings": [],
            "relevance_min": 0
        }

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{total} samples")

def main():
    # --- Define Schema (Features) ---
    # Crucial Step: We must explicitly define the Features.
    # Since RoboPoint has 'None' for all rating fields, without this definition,
    # PyArrow might infer the column type as 'Null'.
    # This would cause a schema mismatch error if you later try to concatenate 
    # this dataset with the real FineVision dataset (where these columns are int64).
    finevision_features = Features({
        "images": List(HFImage()),  # List of Images (use List instead of Sequence to match FineVision format)
        "texts": List({
            "user": Value("string"),
            "assistant": Value("string")
        }),
        "source": Value("string"),
        # Define metadata columns as List(int64) or int64, matching FineVision format
        "image_correspondence_ratings": List(Value("int64")), 
        "image_correspondence_min": Value("int64"),
        "visual_dependency_ratings": List(Value("int64")),
        "visual_dependency_min": Value("int64"),
        "formatting_ratings": List(Value("int64")),
        "formatting_min": Value("int64"),
        "relevance_ratings": List(Value("int64")),
        "relevance_min": Value("int64")
    })

    # Create dataset from generator
    ds = Dataset.from_generator(generate_finevision_data, features=finevision_features)

    print("Conversion finished. Saving to Parquet...")
    
    # Save to Parquet
    # This automatically serializes the PIL Image objects into binary bytes inside the Parquet file.
    ds.to_parquet(OUTPUT_PATH)
    
    print(f"Saved successfully to {OUTPUT_PATH}")

    # --- Verification ---
    print("\n--- Verifying first sample ---")
    from datasets import load_dataset
    check_ds = load_dataset("parquet", data_files=OUTPUT_PATH, split="train")
    sample = check_ds[0]
    
    print("Keys:", sample.keys())
    print("Images type:", type(sample['images']), "Count:", len(sample['images']))
    print("Texts:", sample['texts'])
    print("Visual Dependency Ratings (Should be None):", sample['visual_dependency_ratings']) 

if __name__ == "__main__":
    main()
    