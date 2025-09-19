from datasets import load_dataset, get_dataset_config_names
import time

hub_dataset_name = "HuggingFaceM4/FineVision"
subset_names = [
    "image_textualization(filtered)",
    "sharegpt4v(llava)",
    "sharegpt4v(sam)",
    "textcaps",
    "alfworldgpt",
    "cambrian(filtered)_processed",
    "cocoqa",
]
local_data_dir = "/share_data/datasets/VLM/FineVision"
cache_dir = "/share_data/datasets/VLM/FineVision/.cache"

# --- 2. In first run, generate cache from local original files ---

print("--- First run ---")
print(f"Using '{hub_dataset_name}' official script...")
print(f"From local original data directory: {local_data_dir}")
print(f"Generate Arrow cache to: {cache_dir}")

# We only load the first 1000 samples of the training set as a subset
# The library will automatically calculate which Parquet files to read to satisfy this requirement
time_start = time.time()
subset_datasets_path = [f"{local_data_dir}/{name}/*.parquet" for name in subset_names]
print(subset_datasets_path)

subset_dataset = load_dataset(
    "parquet",
    data_files=subset_datasets_path,   # <--- Core parameter: pointing to the local original data
    split="train",      # <--- Specify the subset you want to load
    cache_dir=cache_dir # <--- Specify the location of the cache
)
print(f"Subset dataset information:")
print(subset_dataset)

print("\nFirst run completed!")
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")

# --- 3. Subsequent loading: experience fast loading ---

print("\n\n--- Simulate second run of the script, for subsequent loading ---")

# Again call the exact same command
# This time, it will find that the cache already exists, and load directly at lightning speed
time_start = time.time()
subset_dataset = load_dataset(
    "parquet",
    data_files=subset_datasets_path,   # <--- Core parameter: pointing to the local original data
    split="train",      # <--- Specify the subset you want to load
    cache_dir=cache_dir # <--- Specify the location of the cache
)
print(f"Subset dataset information:")
print(subset_dataset)

print("\nSubsequent loading completed!")
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")

# You will find that the output speed of the second run is extremely fast

print("Calculating max length...")
start_time = time.time()
max_length = 0
for idx in range(len(subset_dataset)):
    for text in subset_dataset[idx]['texts']:
        max_length = max(max_length, len(text['user']) + len(text['assistant']))
print(f"Max length: {max_length}")
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

print("Calculating max image num...")
start_time = time.time()
max_image_num = 0
for idx in range(len(subset_dataset)):
    max_image_num = max(max_image_num, len(subset_dataset[idx]['images']))
print(f"Max image num: {max_image_num}")
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
