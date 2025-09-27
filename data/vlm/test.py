import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


ORIGINAL_DATA_PATHS = [
    # "/share_data/datasets/VLM/FineVision/objects365_qa",  # localization & grounding
    # "/share_data/datasets/VLM/FineVision/spatialsense",  # spatialsense
    # "/share_data/datasets/VLM/FineVision/alfworldgpt",  # high level subtask action
    # "/share_data/datasets/VLM/FineVision/indoor_qa", 
    # "/share_data/datasets/VLM/FineVision/LLaVA_Instruct_150K", 
    # "/share_data/datasets/VLM/FineVision/cocoqa", 

    # "/share_data/datasets/VLM/FineVision/cambrian(filtered)_processed",
    # "/share_data/datasets/VLM/FineVision/lnqa",
    # "/share_data/datasets/VLM/FineVision/lrv_normal(filtered)",
    "/share_data/datasets/VLM/FineVision/lvis_instruct4v",
    "/share_data/datasets/VLM/FineVision/vqav2",

    # "/share_data/datasets/VLM/FineVision/sharegpt4v(llava)", 
    # "/share_data/datasets/VLM/FineVision/sharegpt4v(sam)", 
    # "/share_data/datasets/VLM/FineVision/textcaps", 
    # "/share_data/datasets/VLM/FineVision/image_textualization(filtered)", 
]

SPLIT = "train" 
CACHE_DIR = "/share_data/datasets/VLM/FineVision/.cache"

if __name__ == "__main__":
    for data_path in ORIGINAL_DATA_PATHS:
        print(f"Loading original dataset from {data_path}...")
        original_dataset = load_dataset(
            "parquet",
            data_files=f"{data_path}/*.parquet",
            split=SPLIT,
            cache_dir=CACHE_DIR,
        )
        filtered_dataset = load_dataset(
            "parquet",
            data_files=f"{data_path}_filtered/*.parquet",
            split=SPLIT,
            cache_dir=CACHE_DIR,
        )
        for i in tqdm(range(len(original_dataset))):
            data = original_dataset[i]
            formatting_ratings = data['formatting_ratings']
            visual_dependency_ratings = data['visual_dependency_ratings']
            relevance_ratings = data['relevance_ratings']
            for rating in formatting_ratings:
                if rating is None:
                    print(f"Formatting rating {rating} is None for sample {i}!!!!!!!!!!!!!")
                    print(formatting_ratings)
            for rating in visual_dependency_ratings:
                if rating is None:
                    print(f"Visual dependency rating {rating} is None for sample {i}!!!!!!!!!!!!!")
                    print(visual_dependency_ratings)
            for rating in relevance_ratings:
                if rating is None:
                    print(f"Relevance rating {rating} is None for sample {i}!!!!!!!!!!!!!")
                    print(relevance_ratings)
        

        for i in tqdm(range(len(filtered_dataset))):
            data = filtered_dataset[i]
            formatting_ratings = data['formatting_ratings']
            visual_dependency_ratings = data['visual_dependency_ratings']
            relevance_ratings = data['relevance_ratings']
            for rating in formatting_ratings:
                if rating is None:
                    print(f"Formatting rating {rating} is None for sample {i} in filtered dataset!!!!!!!!!!!!!")
            for rating in visual_dependency_ratings:
                if rating is None:
                    print(f"Visual dependency rating {rating} is None for sample {i} in filtered dataset!!!!!!!!!!!!!")
            for rating in relevance_ratings:
                if rating is None:
                    print(f"Relevance rating {rating} is None for sample {i} in filtered dataset!!!!!!!!!!!!!")
    