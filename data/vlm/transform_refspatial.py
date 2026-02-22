import os
import json
import io
from PIL import Image
from datasets import Dataset, Features, Value, Image as HFImage, concatenate_datasets
from datasets.features import List
import shutil

# ================= Configuration Section =================

# 1. Root directory of RefSpatial dataset
REFSPATIAL_ROOT = "/share_data/guantianrui/datasets/VLM/RefSpatial"

# 2. Output directory for Arrow format
OUTPUT_DIR = "/share_data/zengfanlian/datasets/VLM/FineVision_Arrow_Format/refspatial_cleaned"
TEMP_DIR = "/share_data/zengfanlian/datasets/VLM/FineVision_Arrow_Format/refspatial_temp"

# 3. Train/test split ratio
VAL_RATIO = 0.001  # 0.1% for validation
SEED = 42

# 4. Test mode: Set to a number to only process first N samples (None = process all)
MAX_SAMPLES = None

# 5. Batch size for incremental processing
BATCH_SIZE = 50000

# 6. Sub-dataset configs
#    Each entry describes one JSON file within RefSpatial.
#    image_dir: directory where the filenames in sample['image'] live
#    depth_dir:  directory where the filenames in sample['depth'] live
DATASET_CONFIGS = [
    {
        "json_path": f"{REFSPATIAL_ROOT}/2D/choice_qa.json",
        "image_dir": f"{REFSPATIAL_ROOT}/2D/image",
        "depth_dir": f"{REFSPATIAL_ROOT}/2D/depth",
        "source": "refspatial_2d_choice",
    },
    {
        "json_path": f"{REFSPATIAL_ROOT}/2D/reasoning_template_qa.json",
        "image_dir": f"{REFSPATIAL_ROOT}/2D/image",
        "depth_dir": f"{REFSPATIAL_ROOT}/2D/depth",
        "source": "refspatial_2d_reasoning",
    },
    {
        "json_path": f"{REFSPATIAL_ROOT}/3D/choice_qa.json",
        "image_dir": f"{REFSPATIAL_ROOT}/3D/image",
        "depth_dir": f"{REFSPATIAL_ROOT}/3D/depth",
        "source": "refspatial_3d_choice",
    },
    {
        "json_path": f"{REFSPATIAL_ROOT}/3D/reasoning_template_qa.json",
        "image_dir": f"{REFSPATIAL_ROOT}/3D/image",
        "depth_dir": f"{REFSPATIAL_ROOT}/3D/depth",
        "source": "refspatial_3d_reasoning",
    },
    # multi_view_qa skipped: model does not support multiple images per sample
    {
        "json_path": f"{REFSPATIAL_ROOT}/3D/vacant_qa.json",
        "image_dir": f"{REFSPATIAL_ROOT}/3D/image",
        "depth_dir": f"{REFSPATIAL_ROOT}/3D/depth",
        "source": "refspatial_3d_vacant",
    },
    {
        "json_path": f"{REFSPATIAL_ROOT}/3D/visual_choice_qa.json",
        # RGB images (with bboxes) live in image_visual_choice/; depth still in depth/
        "image_dir": f"{REFSPATIAL_ROOT}/3D/image_visual_choice",
        "depth_dir": f"{REFSPATIAL_ROOT}/3D/depth",
        "source": "refspatial_3d_visual_choice",
    },
    {
        "json_path": f"{REFSPATIAL_ROOT}/Simulator/metadata.json",
        "image_dir": f"{REFSPATIAL_ROOT}/Simulator/image",
        "depth_dir": f"{REFSPATIAL_ROOT}/Simulator/depth",
        "source": "refspatial_simulator",
    },
]

# 7. Text quality filtering
MAX_TURN_CHARS = 500          # drop turns where user+assistant combined > this
SLASH_REPEAT_THRESHOLD = 5    # flag if any token repeats this many times consecutively in '/' splits

# =========================================================


def has_abnormal_slash_repeat(text, threshold=SLASH_REPEAT_THRESHOLD):
    """
    Detect corrupted object names like 'yellow matte/yellow matte/yellow matte/...'.
    Splits by '/' and checks for consecutive identical tokens repeating >= threshold times.
    Short or normal patterns like 'left/right' won't trigger this.
    """
    parts = text.split("/")
    if len(parts) < threshold:
        return False
    streak = 1
    for i in range(1, len(parts)):
        cur = parts[i].strip().lower()
        prev = parts[i - 1].strip().lower()
        if cur and cur == prev:
            streak += 1
            if streak >= threshold:
                return True
        else:
            streak = 1
    return False


def is_turn_clean(user_text, assistant_text):
    """Return True if a single QA turn passes quality filters."""
    combined_len = len(user_text) + len(assistant_text)
    if combined_len > MAX_TURN_CHARS:
        return False
    if has_abnormal_slash_repeat(user_text) or has_abnormal_slash_repeat(assistant_text):
        return False
    return True


def clean_image_metadata(img):
    """Remove all PNG metadata by re-encoding the image to prevent MAX_TEXT_CHUNK errors."""
    if img is None:
        return None
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    buffer.seek(0)
    clean_img = Image.open(buffer)
    clean_img.load()
    return clean_img


def conversations_to_texts(conversations):
    """
    Convert a list of {'from': 'human'/'gpt', 'value': str} pairs
    into a list of {'user': str, 'assistant': str} dicts (one per turn).

    Turns that are too long or contain corrupted repetitive text are dropped.
    Returns None if no valid turns remain.
    """
    if not conversations or len(conversations) < 2 or len(conversations) % 2 != 0:
        return None

    texts = []
    for i in range(0, len(conversations), 2):
        human_turn = conversations[i]
        gpt_turn = conversations[i + 1]
        if human_turn.get('from') != 'human' or gpt_turn.get('from') != 'gpt':
            return None
        user_text = human_turn.get('value', '').strip()
        assistant_text = gpt_turn.get('value', '').strip()
        if not user_text or not assistant_text:
            continue
        if not is_turn_clean(user_text, assistant_text):
            continue
        texts.append({"user": user_text, "assistant": assistant_text})

    return texts if texts else None


def load_images_from_dir(filenames, directory):
    """
    Load a list of image files from a directory as cleaned RGB PIL images.

    Returns (loaded_images, success). On any IO failure returns ([], False).
    """
    loaded = []
    for fname in filenames:
        full_path = os.path.join(directory, fname)
        try:
            img = Image.open(full_path).convert("RGB")
            img = clean_image_metadata(img)
            loaded.append(img)
        except Exception:
            return [], False
    return loaded, True


def convert_refspatial_batch(batch, image_dir, depth_dir, source):
    """
    Batch conversion for RefSpatial samples.

    Args:
        batch: dict of lists with keys ['id', 'image', 'depth', 'conversations', ...]
        image_dir: directory containing the RGB image files
        depth_dir: directory containing the depth image files
        source: source name string for this sub-dataset

    Returns:
        dict in FineVision format
    """
    batch_size = len(batch['image'])

    images_out = []
    depth_images_out = []
    texts_out = []
    sources = []
    image_corr_ratings = []
    image_corr_min = []
    visual_dep_ratings = []
    visual_dep_min = []
    formatting_ratings = []
    formatting_min = []
    relevance_ratings = []
    relevance_min = []

    for i in range(batch_size):
        image_filenames = batch['image'][i]       # list of RGB filename strings
        depth_filenames = batch.get('depth', [None] * batch_size)[i]  # may be absent
        conversations = batch.get('conversations', [None] * batch_size)[i]

        if not image_filenames or not conversations:
            continue

        # Parse multi-turn conversations into [{user, assistant}, ...]
        texts = conversations_to_texts(conversations)
        if not texts:
            continue

        # Load RGB images
        loaded_images, ok = load_images_from_dir(image_filenames, image_dir)
        if not ok or not loaded_images:
            continue

        # Load depth images (best effort: skip sample only if depth_dir is set but files fail)
        loaded_depths = []
        if depth_filenames:
            loaded_depths, ok = load_images_from_dir(depth_filenames, depth_dir)
            if not ok:
                continue  # depth files listed but failed to load → skip sample

        num_turns = len(texts)
        images_out.append(loaded_images)
        depth_images_out.append(loaded_depths)
        texts_out.append(texts)
        sources.append(source)

        # Dummy metadata ratings (one entry per turn in texts)
        image_corr_ratings.append([0] * num_turns)
        image_corr_min.append(0)
        visual_dep_ratings.append([0] * num_turns)
        visual_dep_min.append(0)
        formatting_ratings.append([0] * num_turns)
        formatting_min.append(0)
        relevance_ratings.append([0] * num_turns)
        relevance_min.append(0)

    return {
        "images": images_out,
        "depth": depth_images_out,
        "texts": texts_out,
        "source": sources,
        "image_correspondence_ratings": image_corr_ratings,
        "image_correspondence_min": image_corr_min,
        "visual_dependency_ratings": visual_dep_ratings,
        "visual_dependency_min": visual_dep_min,
        "formatting_ratings": formatting_ratings,
        "formatting_min": formatting_min,
        "relevance_ratings": relevance_ratings,
        "relevance_min": relevance_min,
    }


def load_json_data(json_path, source_name):
    """Load a RefSpatial JSON file and return the list of samples."""
    print(f"Loading {source_name} from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {json_path}, got {type(data)}")

    print(f"  Found {len(data)} samples")

    if MAX_SAMPLES is not None:
        data = data[:MAX_SAMPLES]
        print(f"  TEST MODE: Using first {MAX_SAMPLES} samples")

    return data


def process_sub_dataset(cfg, global_batch_idx, finevision_features):
    """
    Process one RefSpatial sub-dataset (one JSON file) in BATCH_SIZE chunks.

    Returns a list of saved batch paths.
    """
    json_path = cfg["json_path"]
    image_dir = cfg["image_dir"]
    source = cfg["source"]

    data = load_json_data(json_path, source)
    total_samples = len(data)
    num_batches = (total_samples + BATCH_SIZE - 1) // BATCH_SIZE

    saved_paths = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, total_samples)

        batch_path = os.path.join(TEMP_DIR, f"batch_{global_batch_idx:06d}")

        # Resume: skip if this batch was already saved
        if os.path.exists(batch_path) and os.path.exists(os.path.join(batch_path, "dataset_info.json")):
            print(f"\n  [Sub-batch {batch_idx + 1}/{num_batches}] SKIP (already exists: {batch_path})")
            saved_paths.append(batch_path)
            global_batch_idx += 1
            continue

        batch_data = data[start_idx:end_idx]
        print(f"\n  [Sub-batch {batch_idx + 1}/{num_batches}] samples {start_idx}–{end_idx}...")

        initial_ds = Dataset.from_list(batch_data)
        ds = initial_ds.map(
            convert_refspatial_batch,
            batched=True,
            batch_size=500,
            num_proc=32,
            fn_kwargs={
                "image_dir": image_dir,
                "depth_dir": cfg["depth_dir"],
                "source": source,
            },
            remove_columns=initial_ds.column_names,
            desc=f"Converting {source} batch {batch_idx + 1}",
            features=finevision_features,
        )

        print(f"  Converted {len(ds)} samples (from {len(batch_data)})")

        ds.save_to_disk(batch_path, num_proc=4)
        print(f"  Saved to {batch_path}")

        saved_paths.append(batch_path)
        global_batch_idx += 1

        del initial_ds, ds

    return saved_paths, global_batch_idx


def main():
    print("=" * 80)
    print("RefSpatial to FineVision Arrow Format Conversion")
    print(f"  Filtering: drop turns with combined length > {MAX_TURN_CHARS} chars")
    print(f"  Filtering: drop turns with slash-repeat >= {SLASH_REPEAT_THRESHOLD}")
    print("=" * 80)

    finevision_features = Features({
        "images": List(HFImage()),
        "depth": List(HFImage()),
        "texts": List({
            "user": Value("string"),
            "assistant": Value("string"),
        }),
        "source": Value("string"),
        "image_correspondence_ratings": List(Value("int64")),
        "image_correspondence_min": Value("int64"),
        "visual_dependency_ratings": List(Value("int64")),
        "visual_dependency_min": Value("int64"),
        "formatting_ratings": List(Value("int64")),
        "formatting_min": Value("int64"),
        "relevance_ratings": List(Value("int64")),
        "relevance_min": Value("int64"),
    })

    # Resume support: keep existing temp batches, only recreate missing ones
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Process each sub-dataset
    all_batch_paths = []
    global_batch_idx = 0

    for cfg_idx, cfg in enumerate(DATASET_CONFIGS):
        print("\n" + "=" * 80)
        print(f"[{cfg_idx + 1}/{len(DATASET_CONFIGS)}] Processing sub-dataset: {cfg['source']}")
        print("=" * 80)

        batch_paths, global_batch_idx = process_sub_dataset(
            cfg, global_batch_idx, finevision_features
        )
        all_batch_paths.extend(batch_paths)
        print(f"  Sub-dataset done, total batches so far: {len(all_batch_paths)}")

    # Concatenate all batches
    print("\n" + "=" * 80)
    print(f"Concatenating {len(all_batch_paths)} batch shards...")
    print("=" * 80)

    from datasets import load_from_disk
    all_datasets = []
    for i, batch_path in enumerate(all_batch_paths):
        print(f"Loading shard {i + 1}/{len(all_batch_paths)}: {batch_path}")
        ds = load_from_disk(batch_path)
        all_datasets.append(ds)

    print("Concatenating datasets...")
    full_ds = concatenate_datasets(all_datasets)
    print(f"Total samples: {len(full_ds)}")
    del all_datasets

    # Split into train/val
    print("\n" + "=" * 80)
    print(f"Splitting dataset (val_ratio={VAL_RATIO})...")
    print("=" * 80)
    ds_dict = full_ds.train_test_split(test_size=VAL_RATIO, seed=SEED)
    print(f"Train: {len(ds_dict['train'])}, Val: {len(ds_dict['test'])}")
    del full_ds

    # Save final dataset
    print("\n" + "=" * 80)
    print(f"Saving to: {OUTPUT_DIR}")
    print("=" * 80)

    if os.path.exists(OUTPUT_DIR):
        print(f"Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_size = len(ds_dict['train'])
    target_samples_per_shard = 10000
    num_shards = max(1, train_size // target_samples_per_shard)
    num_proc = min(num_shards, 16)
    print(f"Saving {train_size} train samples (~{num_shards} shards, {num_proc} workers)")

    ds_dict.save_to_disk(OUTPUT_DIR, num_proc=num_proc)
    print(f"\n✅ Saved to {OUTPUT_DIR}")

    # Clean up temp directory
    print("\n" + "=" * 80)
    print("Cleaning up temp files...")
    shutil.rmtree(TEMP_DIR)
    print(f"✓ Removed {TEMP_DIR}")

    # Verification
    print("\n" + "=" * 80)
    print("Verifying first training sample...")
    print("=" * 80)
    loaded_ds = load_from_disk(OUTPUT_DIR)
    sample = loaded_ds['train'][0]

    print("Keys:", list(sample.keys()))
    print("Images count:", len(sample['images']))
    print("Depth images count:", len(sample['depth']))
    print("Texts count:", len(sample['texts']))
    print("First turn:")
    print("  User:     ", sample['texts'][0]['user'][:120], "...")
    print("  Assistant:", sample['texts'][0]['assistant'][:120], "...")
    print("Source:", sample['source'])

    print("\n" + "=" * 80)
    print("✅ Conversion completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
