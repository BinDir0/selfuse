import os
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

TOKENIZER_NAME = "/share_data/checkpoints/paligemma-3b-pt-224" 
MAX_TOKEN_LENGTH = 500 # CLIP's default max length
SPLIT = "train" 
CACHE_DIR = "/share_data/datasets/VLM/FineVision/.cache"

print(f"Loading tokenizer: {TOKENIZER_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, padding_side="right") 

def filter_by_token_length(examples: dict) -> list[bool]:
    texts = examples['texts']
    lengths = []
    for text in texts: 
        max_length = 0
        for diag in text: 
            content = diag['user'] + diag['assistant']
            tokenized_content = tokenizer(content, truncation=False, padding=False)['input_ids']
            max_length = max(max_length, len(tokenized_content))
        lengths.append(max_length)
    
    # print only after processing a batch
    valid_list = [length <= MAX_TOKEN_LENGTH for length in lengths]
    valid_count = sum(valid_list)
    print(f"Processed {len(lengths)} texts, valid: {valid_count}, ratio: {valid_count/len(lengths):.3f}")
    return valid_list


if __name__ == "__main__":
    print("Start processing dataset...")
    
    # for summary statistics
    total_original = 0
    total_filtered = 0
    dataset_stats = []

    for data_path in ORIGINAL_DATA_PATHS:
        print(f"Loading original dataset from {data_path}...")
        original_dataset = load_dataset(
            "parquet",
            data_files=f"{data_path}/*.parquet",
            split=SPLIT,
            cache_dir=CACHE_DIR,
        )
        original_size = len(original_dataset)
        print(f"Original dataset size: {original_size}")
        
        print("Start filtering long text sequences...")
        filtered_dataset = original_dataset.filter(
            filter_by_token_length,
            batched=True,
            batch_size=1000,  
            num_proc=96,  
        )
        
        filtered_size = len(filtered_dataset)
        retention_rate = filtered_size / original_size if original_size > 0 else 0
        
        # record statistics
        dataset_name = data_path.split('/')[-1]
        dataset_stats.append({
            'name': dataset_name,
            'original': original_size,
            'filtered': filtered_size,
            'retention_rate': retention_rate
        })
        
        total_original += original_size
        total_filtered += filtered_size
        
        print(f"Filtered dataset size: {filtered_size}")
        print(f"Retention rate: {retention_rate:.3f} ({filtered_size}/{original_size})")

        save_path = f"{data_path}_filtered"
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Saving filtered dataset to {save_path}...")
        
        # split and save the dataset into multiple files
        total_samples = len(filtered_dataset)
        samples_per_file = 10000  # number of samples per file
        num_files = (total_samples + samples_per_file - 1) // samples_per_file
        
        for i in range(num_files):
            start_idx = i * samples_per_file
            end_idx = min((i + 1) * samples_per_file, total_samples)
            subset = filtered_dataset.select(range(start_idx, end_idx))
            
            file_path = os.path.join(save_path, f"{SPLIT}-{i:05d}-of-{num_files:05d}.parquet")
            subset.to_parquet(file_path)
            print(f"Saved {len(subset)} samples to {file_path}")
        
        print(f"Total files created: {num_files}")
        
        print("Processing completed!")
        print(f"Now you can load '{save_path}/*.parquet' to use the filtered dataset.")
    
    # print summary statistics
    print("\n" + "="*80)
    print("FILTERING SUMMARY")
    print("="*80)
    print(f"{'Dataset':<25} {'Original':<12} {'Filtered':<12} {'Retention':<12}")
    print("-"*80)
    
    for stat in dataset_stats:
        print(f"{stat['name']:<25} {stat['original']:<12,} {stat['filtered']:<12,} {stat['retention_rate']:<12.3f}")
    
    print("-"*80)
    overall_retention = total_filtered / total_original if total_original > 0 else 0
    print(f"{'TOTAL':<25} {total_original:<12,} {total_filtered:<12,} {overall_retention:<12.3f}")
    print("="*80)
    print(f"Overall retention rate: {overall_retention:.3f} ({total_filtered:,}/{total_original:,})")
    print(f"Total samples removed: {total_original - total_filtered:,}")
    print("="*80)
