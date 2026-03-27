import os
import re
import json
import io
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
# 1. Path to the RoboPoint JSON file
ROBOPOINT_JSON_PATH = "/share_data/guantianrui/datasets/VLM/robopoint-data/robopoint_1432k.json"

# 2. Root directory for images
# The 'image' field in RoboPoint JSON usually contains relative paths (e.g., "region_ref/xxx.png")
# You must specify the root folder where these relative paths are located.
IMAGE_ROOT_DIR = "/share_data/guantianrui/datasets/VLM/robopoint-data/images"

# 3. Output directory for WebDataset format
OUTPUT_DIR = "/share_data/zengfanlian/datasets/VLM/Webdataset/robopoint"

# 4. Train/test split ratio
VAL_RATIO = 0.001  # 0.1% for validation
SEED = 42

# 5. Test mode: Set to a number to only process first N samples (None = process all)
MAX_SAMPLES = None  # Set to None to process all samples, or a number like 100 for testing

# 6. Text quality filtering
MAX_TURN_CHARS = 800  # drop samples where user+assistant combined > this

# =========================================================

_FORMAT_INSTR_RE = re.compile(
    r'\s*Your answer should be formatted as.*?'
    r'(?:in the image|pixel locations)\.\s*',
    re.DOTALL | re.IGNORECASE,
)

# Regex to match coordinate patterns in RoboPoint assistant text
# Matches [(0.461, 0.527), ...] or [(0.12, 0.34, 0.56, 0.78)]
_BRACKET_COORDS_RE = re.compile(
    r'\[((?:\(\s*\d+\.?\d*\s*,\s*\d+\.?\d*(?:\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*)?\s*\)(?:\s*,\s*)?)+)\]'
)
_POINT_RE = re.compile(r'\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)')
_BBOX_TUPLE_RE = re.compile(
    r'\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)'
)


def _to_qwen3vl_json(text):
    """Convert [0,1] normalised coordinate text to Qwen3-VL JSON format with 0-1000 coords."""
    def _replace(m):
        inner = m.group(1)
        # Check for 4-element bbox tuple first
        bbox_m = _BBOX_TUPLE_RE.search(inner)
        if bbox_m:
            a, b, c, d = (float(bbox_m.group(i)) for i in range(1, 5))
            return (f'[{{"bbox_2d": [{round(a*1000)}, {round(b*1000)}, '
                    f'{round(c*1000)}, {round(d*1000)}]}}]')
        # 2-element point tuples
        points = _POINT_RE.findall(inner)
        if not points:
            return m.group(0)
        items = ", ".join(
            f'{{"point_2d": [{round(float(x)*1000)}, {round(float(y)*1000)}]}}'
            for x, y in points
        )
        return f"[{items}]"

    return _BRACKET_COORDS_RE.sub(_replace, text)


def strip_format_instructions(text):
    """Remove coordinate-format boilerplate from question text."""
    return _FORMAT_INSTR_RE.sub(' ', text).strip()


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


def process_sample(sample):
    """
    Process a single RoboPoint sample.

    Args:
        sample: dict with keys like 'image', 'conversations'

    Returns:
        (PIL.Image, [{"user": str, "assistant": str}]) or None on failure
    """
    rel_path = sample.get('image')
    conversations = sample.get('conversations')

    if not rel_path or not conversations or len(conversations) < 2:
        return None

    full_img_path = os.path.join(IMAGE_ROOT_DIR, rel_path)

    try:
        img_obj = Image.open(full_img_path).convert("RGB")
        img_obj = clean_image_metadata(img_obj)

        human_text = conversations[0].get('value', '')
        human_text = human_text.replace("<image>\n", "").replace("<image>", "").strip()
        human_text = strip_format_instructions(human_text)
        gpt_text = _to_qwen3vl_json(conversations[1].get('value', '').strip())

        if len(human_text) + len(gpt_text) > MAX_TURN_CHARS:
            return None

        return (img_obj, [{"user": human_text, "assistant": gpt_text}])
    except Exception:
        return None


def process_chunk(args):
    """Worker function that processes a chunk of samples and writes WebDataset shards."""
    chunk, output_dir, worker_id, split, image_quality, maxcount, maxsize = args
    sw = ShardWriter(output_dir, split=split, worker_id=worker_id,
                     maxcount=maxcount, maxsize=maxsize, image_quality=image_quality)
    for local_idx, sample in enumerate(chunk):
        result = process_sample(sample)
        if result is None:
            continue
        img, texts = result
        key = f"robopoint_w{worker_id:04d}_{local_idx:010d}"
        sw.write(key, [img], texts, source="robopoint", sample_idx=local_idx)
    sw.close()


def main():
    parser = argparse.ArgumentParser(description="Convert RoboPoint to WebDataset format")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers")
    parser.add_argument("--maxcount", type=int, default=20000, help="Max samples per shard")
    parser.add_argument("--maxsize", type=float, default=1e9, help="Max shard size in bytes")
    parser.add_argument("--image_quality", type=int, default=95, help="JPEG quality for images")
    args = parser.parse_args()

    print("=" * 80)
    print("RoboPoint to WebDataset Format Conversion")
    print("=" * 80)

    # Load all JSON data
    print("\n" + "=" * 80)
    print("Loading JSON data...")
    print("=" * 80)
    robopoint_data = load_robopoint_json()

    # Split into train/test
    print("\n" + "=" * 80)
    print(f"Splitting dataset (val_ratio={VAL_RATIO})...")
    print("=" * 80)
    train_data, test_data = split_train_test(robopoint_data, VAL_RATIO, SEED)
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Free original data
    del robopoint_data

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split_name, split_data in [("train", train_data), ("test", test_data)]:
        print(f"\nWriting {split_name} split: {len(split_data)} samples")
        num_workers = min(args.num_workers, len(split_data))
        chunk_size = (len(split_data) + num_workers - 1) // num_workers
        chunks = [split_data[i:i+chunk_size] for i in range(0, len(split_data), chunk_size)]

        worker_args = [
            (chunk, OUTPUT_DIR, wid, split_name, args.image_quality, args.maxcount, args.maxsize)
            for wid, chunk in enumerate(chunks)
        ]

        if len(worker_args) == 1:
            process_chunk(worker_args[0])
        else:
            with mp.Pool(len(worker_args)) as pool:
                pool.map(process_chunk, worker_args)

    print("\n" + "=" * 80)
    print("Conversion completed successfully!")
    print(f"Output saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
