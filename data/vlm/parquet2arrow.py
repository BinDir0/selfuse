import os
import shutil
from datasets import load_dataset, disable_progress_bar
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- 配置区域 ---
root_path = "/share_data/guantianrui/datasets/VLM/FineVision"
output_root = "/share_data/guantianrui/datasets/VLM/FineVision_Arrow_Format"
local_tmp_cache_root = "/DATA/chenzhang/.cache" # 临时总目录

val_ratio = 0.001
seed = 42
max_parallel_subsets = 4  # 同时并行转换多少个子集
num_proc_per_subset = 16  # 每个子集内部使用的进程数

# 禁用 Hugging Face 默认的进度条，防止多进程下满屏乱跳
disable_progress_bar()
# ----------------

def convert_single_subset(s):
    """单个子集的转换逻辑"""
    subset_out_path = os.path.join(output_root, s)
    if os.path.exists(subset_out_path):
        return f"SKIP: {s}"

    my_cache_dir = os.path.join(local_tmp_cache_root, s)
    os.makedirs(my_cache_dir, exist_ok=True)
    
    # 每个进程需要独立的环境变量配置
    os.environ["HF_DATASETS_CACHE"] = my_cache_dir

    try:
        # 1. 加载
        ds = load_dataset(
            "parquet", 
            data_files=f"{root_path}/{s}/*.parquet", 
            split="train",
            cache_dir=my_cache_dir
        )
        
        # 2. 切分
        ds_dict = ds.train_test_split(test_size=val_ratio, seed=seed)
        
        # 3. 保存
        ds_dict.save_to_disk(subset_out_path, num_proc=num_proc_per_subset)
        
        # 4. 清理临时缓存
        if os.path.exists(my_cache_dir):
            shutil.rmtree(my_cache_dir)
        return f"DONE: {s}"
    except Exception as e:
        return f"ERROR: {s} - {str(e)}"

def main():
    os.makedirs(output_root, exist_ok=True)
    subsets = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    
    print(f"开始并行转换任务...")
    print(f"总子集数: {len(subsets)}, 并行任务数: {max_parallel_subsets}")

    # 使用 as_completed 来驱动 tqdm 进度条
    with ProcessPoolExecutor(max_workers=max_parallel_subsets) as executor:
        # 提交所有任务
        future_to_subset = {executor.submit(convert_single_subset, s): s for s in subsets}
        
        # 使用 tqdm 包裹 as_completed 迭代器
        with tqdm(total=len(subsets), desc="Overall Progress") as pbar:
            for future in as_completed(future_to_subset):
                subset_name = future_to_subset[future]
                try:
                    result = future.result()
                    # 使用 tqdm.write 而不是 print，这样不会破坏进度条渲染
                    tqdm.write(result)
                except Exception as exc:
                    tqdm.write(f"子集 {subset_name} 产生了未预料的异常: {exc}")
                
                pbar.update(1)

if __name__ == "__main__":
    main()