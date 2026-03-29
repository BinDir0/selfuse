import os
import json
import io
import re
import argparse
import multiprocessing as mp
from PIL import Image
from wds_utils import ShardWriter, split_train_test

# ================= Configuration Section =================
HF_CACHE_DIR = "/share_data/zengfanlian/.cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
# 1. Root directory of RefSpatial dataset
REFSPATIAL_ROOT = "/share_data/guantianrui/datasets/VLM/RefSpatial"

# 2. Output directory for WebDataset format
OUTPUT_DIR = "/share_data/zengfanlian/datasets/VLM/Webdataset/refspatial"

# 3. Train/test split ratio
VAL_RATIO = 0.001  # 0.1% for validation
SEED = 42

# 4. Test mode: Set to a number to only process first N samples (None = process all)
MAX_SAMPLES = None

# 5. Sub-dataset configs
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

# 6. Text quality filtering
MAX_TURN_CHARS = 800          # drop turns where user+assistant combined > this
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


# Regex to match [0,1] normalized coordinate tuples like (0.567, 0.086)
_NORM_COORD_RE = re.compile(r'\((\d+\.\d+),\s*(\d+\.\d+)\)')

# Regex to detect a choice answer starting with (A), (B), (C), ...
_CHOICE_ANSWER_RE = re.compile(r'^\(([A-Z])\)')

# Regex to detect inline choice questions like "... (A) opt1 (B) opt2"
_FIRST_CHOICE_RE = re.compile(r'\([A-Z]\)')


def scale_coords_to_1000(text):
    """Convert [0,1] normalized coordinate tuples to 0-1000 Qwen3 bracket format.

    (0.567, 0.086) → [567, 86]
    Used for coordinates in choice options and referring expressions.
    """
    def _replace(m):
        x = round(float(m.group(1)) * 1000)
        y = round(float(m.group(2)) * 1000)
        return f'[{x}, {y}]'
    return _NORM_COORD_RE.sub(_replace, text)


def extract_choice_letter(text):
    """If assistant answer starts with (A)/(B)/... extract just the letter.

    (A) Correct, ... → A
    (B) black matte door at center → B
    """
    m = _CHOICE_ANSWER_RE.match(text)
    if m:
        return m.group(1)
    return text


def reformat_choice_question(text):
    """Convert inline choice format to robo2vlm multi-line format.

    'Question? (A) opt1 (B) opt2' →
    'Question: Question?\nChoices:\nA. opt1\nB. opt2\nAnswer with the letter.'

    Non-choice questions are returned unchanged.
    """
    first = _FIRST_CHOICE_RE.search(text)
    if not first:
        return text

    question = text[:first.start()].strip()
    choices_str = text[first.start():]

    # Split by (X) markers → ['', 'A', 'text_a ', 'B', 'text_b', ...]
    parts = re.split(r'\(([A-Z])\)\s*', choices_str)
    choices = []
    for i in range(1, len(parts), 2):
        letter = parts[i]
        choice_text = parts[i + 1].strip() if i + 1 < len(parts) else ''
        choices.append((letter, choice_text))

    if not choices:
        return text

    formatted_choices = '\n'.join(f'{letter}. {choice}' for letter, choice in choices)
    return f'Question: {question}\nChoices:\n{formatted_choices}\nAnswer with the letter.'


def conversations_to_texts(conversations):
    """
    Convert a list of {'from': 'human'/'gpt', 'value': str} pairs
    into a list of {'user': str, 'assistant': str} dicts (one per turn).

    Coordinates in [0,1] range are scaled to 0-1000 Qwen3 bracket format.
    Choice answers are reduced to a single letter (A/B/C/D).
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
        user_text = reformat_choice_question(
            scale_coords_to_1000(human_turn.get('value', '').strip())
        )
        assistant_text = extract_choice_letter(
            scale_coords_to_1000(gpt_turn.get('value', '').strip())
        )
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


def process_sample(sample, image_dir, depth_dir, source):
    """Process a single RefSpatial sample."""
    image_filenames = sample.get('image')
    depth_filenames = sample.get('depth')
    conversations = sample.get('conversations')

    if not image_filenames or not conversations:
        return None

    texts = conversations_to_texts(conversations)
    if not texts:
        return None

    loaded_images, ok = load_images_from_dir(image_filenames, image_dir)
    if not ok or not loaded_images:
        return None

    # Load depth images (best effort)
    loaded_depths = []
    if depth_filenames:
        loaded_depths, ok = load_images_from_dir(depth_filenames, depth_dir)
        if not ok:
            return None

    return loaded_images, loaded_depths, texts, source


def process_chunk(args):
    """Worker function: process a chunk of samples and write to WebDataset shards."""
    chunk, output_dir, worker_id, split, image_quality, maxcount, maxsize = args
    sw = ShardWriter(output_dir, split=split, worker_id=worker_id,
                     maxcount=maxcount, maxsize=maxsize, image_quality=image_quality)
    for local_idx, (sample, image_dir, depth_dir, source) in enumerate(chunk):
        result = process_sample(sample, image_dir, depth_dir, source)
        if result is None:
            continue
        images, depths, texts, src = result

        # Encode depth images as extra files (depth_000.jpg, depth_001.jpg, ...)
        extra_images = {}
        for i, d in enumerate(depths):
            extra_images[f"depth_{i:03d}.jpg"] = d

        key = f"refspatial_{src}_w{worker_id:04d}_{local_idx:010d}"
        sw.write(key, images, texts, source=src, sample_idx=local_idx,
                 extra_images=extra_images if extra_images else None)
    sw.close()


def main():
    parser = argparse.ArgumentParser(description="RefSpatial to WebDataset conversion")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers")
    parser.add_argument("--maxcount", type=int, default=20000, help="Max samples per shard")
    parser.add_argument("--maxsize", type=float, default=1e9, help="Max shard size in bytes")
    parser.add_argument("--image_quality", type=int, default=95, help="JPEG quality")
    args = parser.parse_args()

    print("=" * 80)
    print("RefSpatial to WebDataset Format Conversion")
    print(f"  Filtering: drop turns with combined length > {MAX_TURN_CHARS} chars")
    print(f"  Filtering: drop turns with slash-repeat >= {SLASH_REPEAT_THRESHOLD}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Workers: {args.num_workers}, maxcount: {args.maxcount}, "
          f"maxsize: {args.maxsize:.0f}, quality: {args.image_quality}")
    print("=" * 80)

    # Load all sub-datasets
    all_data = []
    for cfg_idx, cfg in enumerate(DATASET_CONFIGS):
        print(f"\n[{cfg_idx + 1}/{len(DATASET_CONFIGS)}] Loading sub-dataset: {cfg['source']}")
        json_path = cfg["json_path"]
        source = cfg["source"]
        data = load_json_data(json_path, source)
        for sample in data:
            all_data.append((sample, cfg["image_dir"], cfg["depth_dir"], source))

    print(f"\nTotal samples loaded: {len(all_data)}")

    # Split into train/test
    print(f"\nSplitting dataset (val_ratio={VAL_RATIO}, seed={SEED})...")
    train_data, test_data = split_train_test(all_data, VAL_RATIO, SEED)
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each split with multiprocessing
    for split_name, split_data in [("train", train_data), ("test", test_data)]:
        if not split_data:
            print(f"\nSkipping {split_name}: no data")
            continue

        n_workers = min(args.num_workers, len(split_data))
        chunk_size = (len(split_data) + n_workers - 1) // n_workers
        chunks = []
        for w in range(n_workers):
            start = w * chunk_size
            end = min((w + 1) * chunk_size, len(split_data))
            if start >= len(split_data):
                break
            chunks.append((
                split_data[start:end],
                OUTPUT_DIR,
                w,
                split_name,
                args.image_quality,
                args.maxcount,
                int(args.maxsize),
            ))

        print(f"\nWriting {split_name} split: {len(split_data)} samples with {len(chunks)} workers...")
        with mp.Pool(len(chunks)) as pool:
            pool.map(process_chunk, chunks)

    print("\n" + "=" * 80)
    print(f"Conversion completed successfully!")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
