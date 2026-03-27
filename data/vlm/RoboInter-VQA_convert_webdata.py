"""
Convert RoboInter-VQA dataset to FineVision WebDataset format.

This script processes the RoboInter-VQA dataset (Generation, Understanding, Task Planning)
and converts it into WebDataset shards with FineVision-compatible metadata.
"""

import io
import os
import re
import json
import glob
import time
import random
import argparse
import multiprocessing as mp
import numpy as np
import webdataset as wds
from PIL import Image
from pathlib import Path
from tqdm import tqdm


HF_CACHE_DIR = "/share_data/zengfanlian/.cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# ================= Configuration =================
ROBOINTER_ROOT = "/share_data/guantianrui/datasets/VLM/RoboInter-VQA"
OUTPUT_ROOT = "/share_data/zengfanlian/datasets/VLM/Webdataset/RoboInter-VQA"

# Task mapping for Generation
GENERATION_TASKS = {
    "traj_qa": "trajectory",
    "traj_qa_wo_init_pos": "trajectory",
    "gripper_det_qa": "bbox",
    "contact_point_qa": "point",
    "contact_box_qa": "bbox",
    "current_box_qa": "bbox",
    "final_box_qa": "bbox"
}

# Task mapping for Understanding
UNDERSTANDING_TASKS = {
    "contact_decide": "classification",
    "grasppose_choice": "classification",
    "grounding_choice": "classification",
    "traj_choice": "classification",
    "trajlang_choice": "classification",
    "traj_direction_choice": "classification"
}

# Task mapping for Task Planning
PLANNING_TASKS = {
    "task_planning": "planning",
    "Scene_Understanding": "planning",
    "Temporal_Understanding": "planning"
}

# ================= Helper Functions =================

def normalize_bbox(bbox, width, height):
    """
    Normalize bbox coordinates to [0, 1] range.
    Input: [[x1, y1], [x2, y2]] (top-left, bottom-right)
    Output: (y_min, x_min, y_max, x_max)
    """
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    
    y_min = y1 / height
    x_min = x1 / width
    y_max = y2 / height
    x_max = x2 / width
    
    # Clip to [0, 1] just in case
    return (
        max(0.0, min(1.0, y_min)),
        max(0.0, min(1.0, x_min)),
        max(0.0, min(1.0, y_max)),
        max(0.0, min(1.0, x_max))
    )

def normalize_point(point, width, height):
    """Normalize a single point [x, y] to [0, 1] range."""
    x_norm = point[0] / width
    y_norm = point[1] / height
    return max(0.0, min(1.0, x_norm)), max(0.0, min(1.0, y_norm))

_FORMAT_INSTR_RE = re.compile(
    r'\s*Your answer should be formatted as.*?'
    r'(?:in the image|pixel locations)\.\s*',
    re.DOTALL | re.IGNORECASE,
)


def strip_format_instructions(text):
    """Remove coordinate-format boilerplate from question text."""
    return _FORMAT_INSTR_RE.sub(' ', text).strip()


# Regex to match absolute pixel coordinate patterns in text:
# Matches (320, 240) or [320, 240] or [[320, 240], [450, 100]] style coords
_ABS_COORD_TUPLE_RE = re.compile(r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)')
_ABS_COORD_BRACKET_RE = re.compile(r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]')


def normalize_abs_coords_in_text(text, img_width, img_height):
    """Convert absolute pixel coordinate tuples/brackets in text to 0-1000 plain tuple format.

    (320, 240) → (500, 500)  for a 640x480 image
    [320, 240] → [500, 500]
    Used for coordinates in choice options and question text.
    """
    def _replace_paren(m):
        x = round(float(m.group(1)) / img_width * 1000)
        y = round(float(m.group(2)) / img_height * 1000)
        return f'({x}, {y})'

    def _replace_bracket(m):
        x = round(float(m.group(1)) / img_width * 1000)
        y = round(float(m.group(2)) / img_height * 1000)
        return f'[{x}, {y}]'

    text = _ABS_COORD_TUPLE_RE.sub(_replace_paren, text)
    text = _ABS_COORD_BRACKET_RE.sub(_replace_bracket, text)
    return text


def bbox_to_text(y_min, x_min, y_max, x_max):
    """Qwen3-VL JSON format: [{"bbox_2d": [x1, y1, x2, y2]}] in 0-1000."""
    x1 = round(x_min * 1000)
    y1 = round(y_min * 1000)
    x2 = round(x_max * 1000)
    y2 = round(y_max * 1000)
    return f'[{{"bbox_2d": [{x1}, {y1}, {x2}, {y2}]}}]'

def point_to_text(x, y):
    """Single point JSON in 0-1000."""
    return f'{{"point_2d": [{round(x * 1000)}, {round(y * 1000)}]}}'

def trajectory_to_text(points):
    """Qwen3-VL JSON format: [{"point_2d": [x, y]}, ...] in 0-1000."""
    items = ", ".join(point_to_text(x, y) for x, y in points)
    return f"[{items}]"

def load_image_safe(image_path):
    """Safely load an image."""
    full_path = os.path.join(ROBOINTER_ROOT, image_path)
    if os.path.exists(full_path):
        try:
            return Image.open(full_path).convert('RGB')
        except Exception:
            return None
    return None

def process_entry(entry, task_type):
    """Process a single entry based on task type."""
    try:
        # Load images
        image_paths = entry['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        images = []
        for path in image_paths:
            img = load_image_safe(path)
            if img is None:
                return None
            images.append(img)
            
        if not images:
            return None

        # Get dimensions from first image (assume all same size for video/multi-frame)
        orig_w, orig_h = images[0].size
        
        # If 'h' and 'w' are provided in JSON, use them for normalization logic
        # But for loading, we use actual image size.
        # Check if JSON dimensions match image dimensions (sanity check)
        json_h = entry.get('h')
        json_w = entry.get('w')
        
        # Use JSON dimensions for normalization if available, otherwise image size
        norm_h = json_h if json_h else orig_h
        norm_w = json_w if json_w else orig_w

        # Parse GT and create assistant response
        gt_str = entry.get('gt', "")
        assistant_text = ""
        
        if task_type == "bbox":
            # GT format: [[x1, y1], [x2, y2]] or string representation
            if isinstance(gt_str, str):
                try:
                    gt_data = json.loads(gt_str)
                except:
                    gt_data = eval(gt_str)
            else:
                gt_data = gt_str
                
            y_min, x_min, y_max, x_max = normalize_bbox(gt_data, norm_w, norm_h)
            assistant_text = bbox_to_text(y_min, x_min, y_max, x_max)
            
        elif task_type == "point":
            if isinstance(gt_str, str):
                try:
                    gt_data = json.loads(gt_str)
                except:
                    gt_data = eval(gt_str)
            else:
                gt_data = gt_str
                
            if isinstance(gt_data[0], list):
                points = [normalize_point(p, norm_w, norm_h) for p in gt_data]
                items = ", ".join(point_to_text(x, y) for x, y in points)
                assistant_text = f"[{items}]"
            else:
                x, y = normalize_point(gt_data, norm_w, norm_h)
                assistant_text = f"[{point_to_text(x, y)}]"
                
        elif task_type == "trajectory":
            if isinstance(gt_str, str):
                try:
                    gt_data = json.loads(gt_str)
                except:
                    gt_data = eval(gt_str)
            else:
                gt_data = gt_str
                
            points = [normalize_point(p, norm_w, norm_h) for p in gt_data]
            assistant_text = trajectory_to_text(points)
            
        elif task_type == "classification":
            # Classification tasks (Understanding)
            # The GT is usually just a letter (A, B, C, D) or a simple word
            assistant_text = gt_str
            
        else:
            # Planning / Text generation
            # Normalize any absolute pixel coordinates in the answer text
            assistant_text = normalize_abs_coords_in_text(gt_str, norm_w, norm_h)

        # Create conversation
        # Some datasets use 'human', others use 'huamn' (typo)
        human_msg = ""
        for msg in entry['conversations']:
            if msg['from'] in ['human', 'huamn']:
                human_msg = msg['value']
                break
                
        # Clean up <image> tags and format instructions
        # Handle various image tag formats and ensure text is preserved
        human_msg = human_msg.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
        human_msg = strip_format_instructions(human_msg)
        # Normalize any absolute pixel coordinates in question/choices to 0-1000 tuples
        human_msg = normalize_abs_coords_in_text(human_msg, norm_w, norm_h).strip()
        
        return {
            "images": images,
            "texts": [{"user": human_msg, "assistant": assistant_text}],
            "source": "RoboInter-VQA",
            "task": entry.get('task', 'unknown'),
            "id": entry.get('id', 'unknown')
        }

    except Exception as e:
        # print(f"Error processing entry {entry.get('id', 'unknown')}: {e}")
        return None

def get_task_type(filename, mapping):
    for key, val in mapping.items():
        if key in filename:
            return val
    return None

def get_worker_next_shard_idx(output_dir, worker_id, shard_prefix="shard"):
    """Find the next available shard index for a specific worker."""
    if not os.path.exists(output_dir):
        return 0
    
    # Look for files like shard-w005-000012.tar
    pattern = os.path.join(output_dir, f"{shard_prefix}-w{worker_id:03d}-*.tar")
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return 0
        
    max_idx = -1
    for f in existing_files:
        try:
            # Parse filename to get the sequence number
            basename = os.path.basename(f)
            # Expected: prefix-wXXX-YYYYYY.tar
            parts = basename.replace(".tar", "").split("-")
            seq_part = parts[-1]
            if seq_part.isdigit():
                max_idx = max(max_idx, int(seq_part))
        except:
            continue
            
    return max_idx + 1

def process_batch_wrapper(samples, output_dir, worker_id, maxcount, maxsize, image_quality, shard_prefix):
    """Wrapper to handle the task_type extraction from item and writing."""
    
    # Determine start index
    shard_idx = get_worker_next_shard_idx(output_dir, worker_id, shard_prefix)
    
    current_count = 0
    current_size = 0
    
    def get_writer(idx):
        pattern = os.path.join(output_dir, f"{shard_prefix}-w{worker_id:03d}-{idx:06d}.tar")
        return wds.TarWriter(pattern)

    writer = get_writer(shard_idx)
    
    for entry in samples:
        # Extract task type which we attached earlier
        task_type = entry.pop('_task_type_internal', 'unknown')
        
        processed = process_entry(entry, task_type)
        if not processed:
            continue
            
        # Convert images
        image_bytes = {}
        sample_bytes_size = 0
        try:
            for i, img in enumerate(processed['images']):
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=image_quality)
                data = buf.getvalue()
                image_bytes[f"image_{i:03d}.jpg"] = data
                sample_bytes_size += len(data)
        except Exception:
            continue

        meta = {
            "dataset_name": "RoboInter-VQA",
            "source": processed['source'],
            "task": processed['task'],
            "id": processed['id'],
            "n_images": len(processed['images']),
            "texts": processed['texts'],
            "formatting_ratings": [0],
            "visual_dependency_ratings": [0],
            "relevance_ratings": [0]
        }
        
        meta_json = json.dumps(meta).encode('utf-8')
        sample_bytes_size += len(meta_json)
        
        # Check limits and rotate if needed
        if current_count >= maxcount or (current_size > 0 and current_size + sample_bytes_size > maxsize):
            writer.close()
            shard_idx += 1
            current_count = 0
            current_size = 0
            writer = get_writer(shard_idx)
            
        sample = {
            "__key__": f"{processed['id'].replace('/', '_').replace('#', '_')}",
            "meta.json": meta
        }
        sample.update(image_bytes)
        writer.write(sample)
        
        current_count += 1
        current_size += sample_bytes_size
        
    writer.close()

def process_split(files, output_dir, args, task_mapping, shard_prefix="shard"):
    if not files:
        return

    print(f"\n=== Processing {os.path.basename(output_dir)} Split ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load ALL data into memory first
    all_samples = []
    print("Loading metadata...")
    for json_file in tqdm(files, desc="Reading JSONs"):
        filename = os.path.basename(json_file)
        task_type = get_task_type(filename, task_mapping)
        if not task_type:
            continue
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Attach task_type to each entry so we don't lose it after flattening
                for item in data:
                    item['_task_type_internal'] = task_type
                all_samples.extend(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    total_samples = len(all_samples)
    print(f"Total samples loaded: {total_samples}")
    
    if total_samples == 0:
        return

    # 2. Split into N chunks
    chunk_size = (total_samples + args.num_workers - 1) // args.num_workers
    chunks = [all_samples[i:i + chunk_size] for i in range(0, total_samples, chunk_size)]
    
    print(f"Split into {len(chunks)} tasks for workers.")
    
    # 3. Prepare args
    worker_args = []
    for i, chunk in enumerate(chunks):
        worker_args.append((
            chunk,
            output_dir,
            i, # worker_id
            args.maxcount,
            args.maxsize,
            args.image_quality,
            shard_prefix
        ))

    # 4. Run Pool
    with mp.Pool(args.num_workers) as pool:
        pool.starmap(process_batch_wrapper, worker_args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=64)
    
    # Standard arguments matching user's request
    parser.add_argument("--split", type=str, default="both", help="Split directory to convert (train/val/both)")
    parser.add_argument("--maxcount", type=int, default=20000, help="Max samples per shard")
    parser.add_argument("--maxsize", type=int, default=int(1e9), help="Max approximate bytes per shard")
    parser.add_argument("--image_quality", type=int, default=95, help="JPEG quality for image encoding")
    
    args = parser.parse_args()

    print("Processing Generation, Understanding, and Task_planning...")
    
    # Determine splits to process
    splits = []
    if args.split == "both":
        splits = ["train", "val"]
    else:
        splits = [args.split]
        
    # Collect paths and task mappings for all categories
    all_tasks = []
    
    # 1. Generation
    gen_paths = []
    if "train" in splits:
        gen_paths.extend([
            os.path.join(ROBOINTER_ROOT, "Generation/meta/train/droid/origin_format/*.json"),
            os.path.join(ROBOINTER_ROOT, "Generation/meta/train/rh20t/origin_format/*.json")
        ])
    if "val" in splits:
        gen_paths.extend([
            os.path.join(ROBOINTER_ROOT, "Generation/meta/val/origin_format/*.json")
        ])
    all_tasks.append((gen_paths, GENERATION_TASKS))
    
    # 2. Understanding
    und_paths = []
    if "train" in splits:
        und_paths.extend([
            os.path.join(ROBOINTER_ROOT, "Understanding/meta/train/droid/*.json"),
            os.path.join(ROBOINTER_ROOT, "Understanding/meta/train/rh20t/*.json")
        ])
    if "val" in splits:
        und_paths.extend([
            os.path.join(ROBOINTER_ROOT, "Understanding/meta/val/*.json")
        ])
    all_tasks.append((und_paths, UNDERSTANDING_TASKS))
    
    # 3. Task_planning
    plan_paths = []
    if "train" in splits:
        plan_paths.extend([
            os.path.join(ROBOINTER_ROOT, "Task_planning/meta/train/manipvqa/*.json")
        ])
    if "val" in splits:
        plan_paths.extend([
            os.path.join(ROBOINTER_ROOT, "Task_planning/meta/val/*/*.json")
        ])
    all_tasks.append((plan_paths, PLANNING_TASKS))

    # Process each category sequentially to avoid mixing task mappings
    for search_paths, task_mapping in all_tasks:
        json_files = []
        for path in search_paths:
            json_files.extend(glob.glob(path))
        
        if not json_files:
            continue

        print(f"Found {len(json_files)} JSON files for current category.")
        
        # Group files by split (train/test)
        train_files = []
        test_files = []
        
        for json_file in json_files:
            if "/val/" in json_file or "/val" in os.path.dirname(json_file):
                test_files.append(json_file)
            else:
                train_files.append(json_file)
                
        # Extract category name from the first path to use as prefix
        category_name = "unknown"
        if "Generation" in search_paths[0]:
            category_name = "generation"
        elif "Understanding" in search_paths[0]:
            category_name = "understanding"
        elif "Task_planning" in search_paths[0]:
            category_name = "planning"
            
        shard_prefix = f"shard_{category_name}"
        
        # Process train files
        if train_files:
            output_dir = os.path.join(OUTPUT_ROOT, "train")
            process_split(train_files, output_dir, args, task_mapping, shard_prefix)

        # Process test files
        if test_files:
            output_dir = os.path.join(OUTPUT_ROOT, "test")
            process_split(test_files, output_dir, args, task_mapping, shard_prefix)
            
    print("Done!")

if __name__ == "__main__":
    main()
