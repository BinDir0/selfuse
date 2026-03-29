"""
Convert RoboAfford dataset to WebDataset format.

Processing 4 data sources (skipping PixMo-Points):
- LVIS_absxy_513K.json: Visual Genome images
- object_affordance_prediction_absxy_561K.json: PACO-LVIS
- object_ref_max_points_10_absxy_347K.json: RoboPoint
- region_ref_max_points_10_absxy_320K.json: RoboPoint

Features:
- Multi-turn conversations split into separate samples
- Coordinates normalized to [0, 1] range
- WebDataset format output
"""

import os
import json
import re
import random
import argparse
import io
import multiprocessing as mp
from PIL import Image, PngImagePlugin
from tqdm import tqdm

from wds_utils import ShardWriter, split_train_test
HF_CACHE_DIR = "/share_data/zengfanlian/.cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
# CRITICAL: Increase PIL's limit for large PNG iCCP chunks BEFORE loading any images
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)  # 100MB (default is 1MB)

# ================= Configuration =================
ROBOAFFORD_ROOT = "/share_data/guantianrui/datasets/VLM/RoboAfford"
OUTPUT_DIR = "/share_data/zengfanlian/datasets/VLM/Webdataset/roboafford"

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

# Text quality filtering
MAX_TURN_CHARS = 800  # drop turns where user+assistant combined > this

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


_FORMAT_INSTR_RE = re.compile(
    r'\s*Your answer should be formatted as.*?'
    r'(?:in the image|pixel locations)\.\s*',
    re.DOTALL | re.IGNORECASE,
)


def strip_format_instructions(text):
    """Remove coordinate-format boilerplate from question text."""
    return _FORMAT_INSTR_RE.sub(' ', text).strip()


def normalize_coords_to_tuple(text, img_width, img_height):
    """Convert absolute pixel coordinates to Qwen3 bracket format with 0-1000 coords.

    For use in question/input text (referring expressions, choice options).
    Bbox  [(x1, y1, x2, y2)] → [qx1, qy1, qx2, qy2]
    Points [(x, y), ...]      → [[qx, qy], ...]
    """
    def normalize_match(match):
        coords_str = match.group(1)
        tuples = list(re.finditer(
            r'\(([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+),\s*([\d.]+))?\)', coords_str
        ))
        if not tuples:
            return match.group(0)

        first = tuples[0]
        if first.group(3):  # Bbox (x1, y1, x2, y2)
            x1, y1, x2, y2 = (float(first.group(i)) for i in range(1, 5))
            qx1 = round(x1 / img_width * 1000)
            qy1 = round(y1 / img_height * 1000)
            qx2 = round(x2 / img_width * 1000)
            qy2 = round(y2 / img_height * 1000)
            return f'[{qx1}, {qy1}, {qx2}, {qy2}]'
        else:  # Points (x, y), ...
            items = []
            for t in tuples:
                x, y = float(t.group(1)), float(t.group(2))
                qx = round(x / img_width * 1000)
                qy = round(y / img_height * 1000)
                items.append(f'[{qx}, {qy}]')
            return "[" + ", ".join(items) + "]"

    return re.sub(r'\[((?:\([^)]+\)(?:,\s*)?)+)\]', normalize_match, text)


def normalize_coords_to_json(text, img_width, img_height):
    """Convert absolute pixel coordinates to Qwen3-VL JSON format with 0-1000 coords.

    For use in assistant/output text (grounding results).
    Bbox  (x1, y1, x2, y2) → [{"bbox_2d": [x1, y1, x2, y2]}]
    Points (x, y), ...      → [{"point_2d": [x, y]}, ...]
    """
    def normalize_match(match):
        coords_str = match.group(1)
        tuples = list(re.finditer(
            r'\(([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+),\s*([\d.]+))?\)', coords_str
        ))
        if not tuples:
            return match.group(0)

        first = tuples[0]
        if first.group(3):  # Bbox (x1, y1, x2, y2)
            x1, y1, x2, y2 = (float(first.group(i)) for i in range(1, 5))
            qx1 = round(x1 / img_width * 1000)
            qy1 = round(y1 / img_height * 1000)
            qx2 = round(x2 / img_width * 1000)
            qy2 = round(y2 / img_height * 1000)
            return f'[{{"bbox_2d": [{qx1}, {qy1}, {qx2}, {qy2}]}}]'
        else:  # Points (x, y), ...
            items = []
            for t in tuples:
                x, y = float(t.group(1)), float(t.group(2))
                qx = round(x / img_width * 1000)
                qy = round(y / img_height * 1000)
                items.append(f'{{"point_2d": [{qx}, {qy}]}}')
            return "[" + ", ".join(items) + "]"

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


def process_sample(sample_dict, source_name, rng=None):
    """
    Process a single RoboAfford sample into WebDataset format.

    Args:
        sample_dict: dict with keys: id, image, conversations
        source_name: one of lvis, paco-lvis, robopoint-obj, robopoint-reg
        rng: random.Random instance for reproducibility

    Returns:
        (img, texts_list, source_name) or None if sample is invalid
    """
    if rng is None:
        rng = random.Random(SEED)

    image_path = sample_dict['image']
    conversations = sample_dict['conversations']

    # Load image
    img = load_image(image_path)
    if img is None:
        return None

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
        return None

    user_msg, assistant_msg = rng.choice(valid_pairs)

    # Normalize coordinates
    user_text = user_msg['value'].replace('<image>\n', '').strip()
    assistant_text = assistant_msg['value'].strip()

    user_text_normalized = strip_format_instructions(
        normalize_coords_to_tuple(user_text, img_width, img_height)
    )
    assistant_text_normalized = normalize_coords_to_json(assistant_text, img_width, img_height)

    if len(user_text_normalized) + len(assistant_text_normalized) > MAX_TURN_CHARS:
        return None

    texts = [{"user": user_text_normalized, "assistant": assistant_text_normalized}]
    return (img, texts, source_name)


def process_chunk(args):
    """Worker function: process a chunk of samples and write to WebDataset shards."""
    chunk, output_dir, worker_id, split, image_quality, maxcount, maxsize = args
    sw = ShardWriter(output_dir, split=split, worker_id=worker_id,
                     maxcount=maxcount, maxsize=maxsize, image_quality=image_quality)
    rng = random.Random(SEED + worker_id)
    for local_idx, (sample, source) in enumerate(chunk):
        result = process_sample(sample, source, rng)
        if result is None:
            continue
        img, texts, src = result
        key = f"roboafford_{src}_w{worker_id:04d}_{local_idx:010d}"
        sw.write(key, [img], texts, source=f"roboafford-{src}", sample_idx=local_idx)
    sw.close()


def determine_source(json_file):
    """Determine source name from JSON filename."""
    if 'LVIS' in json_file:
        return 'lvis'
    elif 'affordance' in json_file:
        return 'paco-lvis'
    elif 'object_ref' in json_file:
        return 'robopoint-obj'
    elif 'region_ref' in json_file:
        return 'robopoint-reg'
    else:
        return 'unknown'


def main():
    parser = argparse.ArgumentParser(description="Convert RoboAfford to WebDataset format")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers")
    parser.add_argument("--maxcount", type=int, default=20000, help="Max samples per shard")
    parser.add_argument("--maxsize", type=float, default=1e9, help="Max shard size in bytes")
    parser.add_argument("--image_quality", type=int, default=95, help="JPEG quality")
    args = parser.parse_args()

    print("=" * 80)
    print("RoboAfford to WebDataset Format Conversion")
    print("=" * 80)

    # Load all JSON files into a combined list with source tags
    all_data = []  # list of (sample_dict, source_name)
    for json_file in JSON_FILES:
        source = determine_source(json_file)
        json_path = os.path.join(ROBOAFFORD_ROOT, json_file)
        print(f"Loading {json_file} (source={source})...")
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"  Loaded {len(data)} samples")
        for sample in data:
            all_data.append((sample, source))

    print(f"\nTotal combined samples: {len(all_data)}")

    # Split into train/test
    print(f"Splitting dataset (val_ratio={VAL_RATIO}, seed={SEED})...")
    train_data, test_data = split_train_test(all_data, VAL_RATIO, SEED)
    print(f"  Train: {len(train_data)} samples")
    print(f"  Test:  {len(test_data)} samples")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each split
    for split, split_data in [("train", train_data), ("test", test_data)]:
        print(f"\n{'='*80}")
        print(f"Processing {split} split ({len(split_data)} samples) with {args.num_workers} workers...")
        print(f"{'='*80}")

        # Divide data into chunks for workers
        num_workers = min(args.num_workers, len(split_data))
        chunk_size = (len(split_data) + num_workers - 1) // num_workers
        chunks = []
        for w in range(num_workers):
            start = w * chunk_size
            end = min((w + 1) * chunk_size, len(split_data))
            if start >= len(split_data):
                break
            chunk = split_data[start:end]
            chunks.append((chunk, OUTPUT_DIR, w, split, args.image_quality,
                           args.maxcount, int(args.maxsize)))

        # Process chunks in parallel
        with mp.Pool(processes=len(chunks)) as pool:
            pool.map(process_chunk, chunks)

        print(f"  {split} split complete.")

    print(f"\nConversion completed!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
