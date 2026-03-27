"""
Convert ShareRobot Affordance and Trajectory datasets to WebDataset format.

Converts bbox and trajectory coordinates to normalized [0,1] format for PaliGemma.
"""
import os
import json
import glob
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
# 1. ShareRobot root directory
SHAREROBOT_ROOT = "/share_data/guantianrui/datasets/VLM/ShareRobot"

# 2. Output directory for WebDataset format
OUTPUT_DIR = "/share_data/zengfanlian/datasets/VLM/Webdataset/sharerobot"

# 3. Only process affordance and trajectory (planning has multiple images per sample)
PROCESS_SUBDATASETS = ['affordance', 'trajectory', 'planning']

# 4. Train/test split ratio
VAL_RATIO = 0.001  # 0.1% for validation
SEED = 42

# 5. Test mode: Set to a number to only process first N samples (None = process all)
MAX_SAMPLES = None  # Set to None to process all samples, or a number like 100 for testing

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
    Convert normalized [0,1] bbox to Qwen3-VL JSON format.
    Input order: (y_min, x_min, y_max, x_max)
    Output: [{"bbox_2d": [x1, y1, x2, y2]}] with 0-1000 coords
    """
    x1 = round(x_min * 1000)
    y1 = round(y_min * 1000)
    x2 = round(x_max * 1000)
    y2 = round(y_max * 1000)
    return f'[{{"bbox_2d": [{x1}, {y1}, {x2}, {y2}]}}]'


def trajectory_to_text(points):
    """
    Convert normalized [0,1] trajectory points to Qwen3-VL JSON format.
    Input: list of (x, y) in [0,1]
    Output: [{"point_2d": [x1, y1]}, {"point_2d": [x2, y2]}, ...] with 0-1000 coords
    """
    items = ", ".join(
        f'{{"point_2d": [{round(x * 1000)}, {round(y * 1000)}]}}'
        for x, y in points
    )
    return f"[{items}]"


def load_image_safe(base_path, image_path):
    """Safely load an image."""
    possible_paths = [
        os.path.join(base_path, image_path),
        os.path.join(SHAREROBOT_ROOT, image_path),
        os.path.join(SHAREROBOT_ROOT, 'images', image_path),
    ]
    
    # Fallback for planning images that might be extracted in trajectory/images
    if image_path.startswith('rt_frames_success/'):
        possible_paths.append(
            os.path.join(SHAREROBOT_ROOT, 'trajectory', 'images', image_path.replace('rt_frames_success/', ''))
        )

    for full_path in possible_paths:
        if os.path.exists(full_path):
            try:
                return Image.open(full_path).convert('RGB')
            except Exception:
                continue
    return None


def process_affordance_sample(sample, image_base_path):
    """Process a single affordance sample."""
    try:
        img = load_image_safe(image_base_path, sample['image_path'])
        if img is None:
            return None
        orig_width = sample['meta_data']['original_width']
        orig_height = sample['meta_data']['original_height']
        bbox = sample['affordance']
        y_min, x_min, y_max, x_max = normalize_bbox(bbox, orig_width, orig_height)
        instruction = sample['instruction']
        user_text = f"Please provide the bounding box coordinate of the region to {instruction}."
        assistant_text = bbox_to_text(y_min, x_min, y_max, x_max)
        return [img], [{"user": user_text, "assistant": assistant_text}]
    except Exception:
        return None


def process_trajectory_sample(sample, image_base_path):
    """Process a single trajectory sample."""
    try:
        img = load_image_safe(image_base_path, sample['image_path'])
        if img is None:
            return None
        orig_width = sample['meta_data']['original_width']
        orig_height = sample['meta_data']['original_height']
        trajectory = sample['trajectory']
        normalized_points = [normalize_point(point, orig_width, orig_height) for point in trajectory]
        instruction = sample['instruction']
        user_text = f"Predict the trajectory points to {instruction}."
        assistant_text = trajectory_to_text(normalized_points)
        return [img], [{"user": user_text, "assistant": assistant_text}]
    except Exception:
        return None


def process_planning_sample(sample, image_base_path):
    """Process a single planning sample (multi-image)."""
    try:
        image_paths = sample.get('image', [])
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        images = []
        for img_path in image_paths:
            img = load_image_safe(image_base_path, img_path)
            if img is None:
                return None
            images.append(img)
            
        if not images:
            return None
            
        # Parse conversation
        human_msg = ""
        gpt_msg = ""
        for conv in sample.get('conversations', []):
            if conv['from'] == 'human':
                human_msg = conv['value']
            elif conv['from'] == 'gpt':
                gpt_msg = conv['value']
                
        # Clean up <image> tags
        human_msg = human_msg.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "").strip()
        
        return images, [{"user": human_msg, "assistant": gpt_msg}]
    except Exception:
        return None


def process_chunk(args):
    """Worker function to process a chunk of samples and write to WebDataset shards."""
    chunk, output_dir, worker_id, split, image_quality, maxcount, maxsize = args
    sw = ShardWriter(output_dir, split=split, worker_id=worker_id,
                     maxcount=maxcount, maxsize=maxsize, image_quality=image_quality)
    for local_idx, (sample, subdataset, image_base_path) in enumerate(chunk):
        if subdataset == 'affordance':
            result = process_affordance_sample(sample, image_base_path)
            source = "affordance"
        elif subdataset == 'trajectory':
            result = process_trajectory_sample(sample, image_base_path)
            source = "trajectory"
        elif subdataset == 'planning':
            result = process_planning_sample(sample, image_base_path)
            source = "planning"
        else:
            continue
            
        if result is None:
            continue
        images, texts = result
        key = f"sharerobot_{source}_w{worker_id:04d}_{local_idx:010d}"
        sw.write(key, images, texts, source=source, sample_idx=local_idx)
    sw.close()


def main():
    parser = argparse.ArgumentParser(description="Convert ShareRobot to WebDataset format")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--maxcount", type=int, default=20000, help="Max samples per shard")
    parser.add_argument("--maxsize", type=float, default=1e9, help="Max shard size in bytes")
    parser.add_argument("--image_quality", type=int, default=95, help="JPEG quality")
    args = parser.parse_args()

    print("=" * 80)
    print("ShareRobot to WebDataset Format Conversion")
    print("=" * 80)

    # Load all data
    all_data = []
    for subdataset_name in PROCESS_SUBDATASETS:
        if subdataset_name == 'planning':
            json_dir = os.path.join(SHAREROBOT_ROOT, subdataset_name, 'jsons')
            json_paths = glob.glob(os.path.join(json_dir, '*.json'))
            image_base_path = os.path.join(SHAREROBOT_ROOT, subdataset_name, 'images')
        else:
            json_paths = [os.path.join(SHAREROBOT_ROOT, subdataset_name, f'{subdataset_name}.json')]
            image_base_path = os.path.join(SHAREROBOT_ROOT, subdataset_name, 'images')
            
        for json_path in json_paths:
            if not os.path.exists(json_path):
                continue
            with open(json_path, 'r') as f:
                data = json.load(f)
            if MAX_SAMPLES is not None:
                data = data[:MAX_SAMPLES]
            for sample in data:
                all_data.append((sample, subdataset_name, image_base_path))
            print(f"Loaded {len(data)} {subdataset_name} samples from {os.path.basename(json_path)}")

    print(f"\nTotal samples: {len(all_data)}")

    # Split into train/test
    train_data, test_data = split_train_test(all_data, VAL_RATIO, SEED)
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process train and test splits
    for split_name, split_data in [("train", train_data), ("test", test_data)]:
        if not split_data:
            print(f"\nNo {split_name} data to process")
            continue

        print(f"\n{'='*80}")
        print(f"Processing {split_name} split ({len(split_data)} samples)")
        print(f"{'='*80}")

        num_workers = min(args.num_workers, len(split_data))
        chunk_size = (len(split_data) + num_workers - 1) // num_workers
        chunks = []
        for i in range(num_workers):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(split_data))
            if start >= len(split_data):
                break
            chunks.append((
                split_data[start:end],
                OUTPUT_DIR,
                i,
                split_name,
                args.image_quality,
                args.maxcount,
                int(args.maxsize),
            ))

        print(f"Using {len(chunks)} workers")
        with mp.Pool(len(chunks)) as pool:
            pool.map(process_chunk, chunks)

    print(f"\nSaved WebDataset shards to {OUTPUT_DIR}")
    print("=" * 80)
    print("Conversion completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
