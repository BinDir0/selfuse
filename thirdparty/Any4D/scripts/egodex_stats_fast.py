import argparse
import json
import os
import h5py
import decord
import multiprocessing
import concurrent.futures
from tqdm import tqdm

# ==========================================
# 核心处理函数 (必须放在主程序块之外)
# ==========================================
def process_one_episode(base_path):
    """
    处理单个 Episode：读取 MP4 帧数和 HDF5 姿态数
    """
    try:
        mp4_path = f"{base_path}.mp4"
        hdf5_path = f"{base_path}.hdf5"
        episode_name = os.path.basename(base_path)

        # 1. 读取视频长度
        # 优化：只初始化 VideoReader 获取 len，不进行 get_batch 解码，速度极快
        if not os.path.exists(mp4_path):
            return None
        vr = decord.VideoReader(mp4_path, ctx=decord.cpu(0))
        video_len = len(vr)

        # 2. 读取 HDF5 长度
        if not os.path.exists(hdf5_path):
            return None
        with h5py.File(hdf5_path, 'r') as file:
            # 获取 transforms/camera 的第一维长度
            extrinsics_len = file['transforms']['camera'].shape[0]

        # 3. 计算统计信息
        minlen = min(video_len, extrinsics_len)
        maxlen = max(video_len, extrinsics_len)
        gap = maxlen - minlen

        return (episode_name, minlen, gap)

    except Exception as e:
        # 如果文件损坏或无法读取，打印错误但不中断程序
        print(f"\nError processing {base_path}: {e}")
        return None

# ==========================================
# 主逻辑
# ==========================================
def main():
    # 1. 准备路径列表
    print("Loading dataset list...")
    # 注意：这里假设 JSON 路径是固定的，如果变动请修改这里
    json_path = "/share_data/guantianrui/datasets/EgoDex/episode_names.json"
    episode_list_str = json.load(open(json_path))
    
    all_base_paths = []
    for episode_name in episode_list_str:
        # 解析路径逻辑 (保持原逻辑不变)
        group, remainder = episode_name.split("_", 1)
        task, id = remainder.rsplit("_", 1)
        full_path = f"/share_data/guantianrui/datasets/EgoDex/{group}/{task}/{id}"
        all_base_paths.append(full_path)

    print(f"Total episodes found: {len(all_base_paths)}")

    # 2. 配置多进程
    # 留几个核给系统，其余全部利用
    max_workers = max(1, multiprocessing.cpu_count() - 4)
    print(f"Starting processing with {max_workers} processes...")

    results = []
    
    # 3. 并行执行
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 使用 tqdm 显示进度条
        # executor.map 会按输入顺序返回结果
        futures = list(tqdm(executor.map(process_one_episode, all_base_paths), 
                            total=len(all_base_paths), 
                            desc="Stats Processing"))
        
        # 过滤掉 None (即报错的文件)
        results = [r for r in futures if r is not None]

    print(f"\nSuccessfully processed {len(results)} videos.")

    # 4. 排序并保存结果 - 按 minlen 排序
    print("Saving egodex_video_list_len.txt ...")
    results.sort(key=lambda x: x[1]) # sort by minlen
    with open("egodex_video_list_len.txt", "w") as f:
        for name, minlen, gap in results:
            f.write(f"{name} minlen: {minlen} gap: {gap}\n")

    # 5. 排序并保存结果 - 按 gap 排序
    print("Saving egodex_video_list_gap.txt ...")
    results.sort(key=lambda x: x[2]) # sort by gap
    with open("egodex_video_list_gap.txt", "w") as f:
        for name, minlen, gap in results:
            f.write(f"{name} gap: {gap} minlen: {minlen}\n")
    
    print("Done!")

if __name__ == "__main__":
    main()