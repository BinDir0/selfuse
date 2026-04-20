import os
import json
import decord
import multiprocessing
import time
import argparse
from tqdm import tqdm
import cv2
import numpy as np

# ================= 配置区域 =================

# 数据集根目录
VIDEO_ROOT = "/share_data/guantianrui/datasets/VITRA-1M/video_root"

# JSON 路径列表 (Train 和 Test)
JSON_PATHS = [
    "/share_data/guantianrui/datasets/VITRA-1M/zarr_episodes_statistics.json",
    "/share_data/guantianrui/datasets/VITRA-1M/zarr_episodes_statistics_testset.json"
]

# ===========================================

def get_video_filename(video_name_key):
    """
    根据 video_name 解析实际的文件名
    Ego4D_... -> .mp4
    somethingsomethingv2_... -> .webm
    epic_kitchens_... -> .MP4
    """
    if video_name_key.startswith("Ego4D_"):
        return f"{video_name_key[6:]}.mp4"
    elif video_name_key.startswith("somethingsomethingv2_"):
        return f"{video_name_key[21:]}.webm"
    elif video_name_key.startswith("epic_kitchens_"):
        return f"{video_name_key[14:]}.MP4"
    else:
        # 如果有其他前缀，可以在这里补充，或者返回 None
        return None

def read_video_cv2(video_path, start_idx, end_idx):
    """
    使用 OpenCV 读取视频片段 (作为 Decord 的备选方案)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"OpenCV failed to open {video_path}")
    
    frames = []
    # 1. 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    
    current_idx = start_idx
    while current_idx < end_idx:
        ret, frame = cap.read()
        if not ret:
            # 如果读不到（比如视频意外结束），就停止
            break
        
        # OpenCV 默认 BGR，转 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        current_idx += 1
        
    cap.release()

    if len(frames) == 0:
        raise IOError(f"OpenCV read 0 frames from {video_path}")

    mid_idx = len(frames) // 2
    frames_final = [frames[mid_idx]] + frames
        
    return np.array(frames_final)

def process(tasks):
    for task in tasks:
        ep_name = task['ep_name']
        print(f"Checking {task['ep_name']} ...\n")
        task_len = task["end"] - task["start"] + 1
        if task_len > 180:
            print(f"[LONG] {task['ep_name']} has {task_len} frames.\n")
            
        video_key = task['video_key']
        # --- 2. 文件存在性检查 (轻量 IO) ---
        filename = get_video_filename(video_key)
        if filename is None:
            print(f"[ERROR] {ep_name}: Unknown Prefix: {video_key}\n")
            
            continue
            
        video_path = os.path.join(VIDEO_ROOT, filename)
        
        if not os.path.exists(video_path):
            print(f"[ERROR] {ep_name}: File Missing\n")
            
            continue

        s = task['start']
        e = task['end']
        duration = e - s + 1

        # --- 3. 视频解码检查 (重 IO + CPU) ---
        try:
            # 使用 cpu(0) 避免 CUDA 初始化，num_threads=0 让 decord 自动调度
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=0)

            if s > e:
                print(f"[ERROR] {ep_name}: Invalid duration: {duration} ({s}-{e})\n")
                
                continue
            
            # 检查越界
            if e >= len(vr):
                print(f"[ERROR] {ep_name}: IndexOOB: req_end={e}, video_len={len(vr)}\n")
                
                continue
            
            indices = list(range(s, e + 1))
            frames = vr.get_batch(indices).asnumpy()
            
            # 检查读取出来的帧数是否对齐
            if frames.shape[0] != duration:
                print(f"[ERROR] {ep_name}: Frame mismatch: req={duration}, got={frames.shape[0]}\n")
                

        except Exception as exc:
            try:
                # 注意：OpenCV 读取比较慢，如果大量触发这里会拖慢速度
                _ = read_video_cv2(video_path, s, e)
                # 如果 OpenCV 成功了，说明是 decord 的问题，但视频文件本身“可能”是好的
                # 可以选择记录一下，或者直接 pass
                # print(f"[WARN] {ep_name}: Decord failed ({decord_err}), but OpenCV succeeded.\n")
                pass 
                
            except Exception as cv2_err:
                # OpenCV 也失败了，说明视频彻底坏了
                print(f"[FATAL] {ep_name}: Both Decord & OpenCV failed. Err: {cv2_err}\n")
                


def load_all_tasks(cpu_id, total_cpus):
    all_tasks = []
    print("-" * 50)
    for json_path in JSON_PATHS:
        if not os.path.exists(json_path):
            print(f"[WARN] JSON not found: {json_path}")
            continue
            
        print(f"Loading {os.path.basename(json_path)} ...")
        data = json.load(open(json_path))
        # 根据你提供的结构，root 下就是 video_statistics
        stats = data.get('video_statistics', [])
        
        count = 0
        for vid_item in stats:
            video_key = vid_item['video_name']
            for ep in vid_item['episodes']:
                try:
                    all_tasks.append({
                        "video_key": video_key,
                        "ep_name": ep['zarr_episode_name'],
                        "start": ep['original_start_frame'],
                        "end": ep['original_end_frame']
                    })
                    count += 1
                except:
                    print(ep)
        print(f"  -> Found {count} episodes.")
    
    tasks_per_cpu = len(all_tasks) // total_cpus
    start_idx = cpu_id * tasks_per_cpu
    if cpu_id == total_cpus - 1:
        end_idx = len(all_tasks)
    else:
        end_idx = (cpu_id + 1) * tasks_per_cpu
    my_tasks = all_tasks[start_idx:end_idx]
    print(f"CPU {cpu_id} processing tasks {start_idx} to {end_idx} (total {len(my_tasks)})")
            
    print("-" * 50)
    return my_tasks

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu_id", type=int, default=0)
    parser.add_argument("--total_cpus", type=int, default=512)
    return parser.parse_args()

def main():
    args = parse_args()
    tasks = load_all_tasks(args.cpu_id, args.total_cpus)
    process(tasks)

if __name__ == "__main__":
    main()