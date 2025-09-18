from datasets import load_dataset, get_dataset_config_names

# Get all subset names and load the first one
with open('list.txt', 'r') as f:
    available_subsets = f.read().splitlines()

available_subsets = get_dataset_config_names('HuggingFaceM4/FineVision')
print(available_subsets)

ds = load_dataset("HuggingFaceM4/FineVision", "vqav2")
ds.save_to_disk("/share_data/datasets/VLM")
