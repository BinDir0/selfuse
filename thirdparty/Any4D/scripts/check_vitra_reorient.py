import os
import json
import decord
import argparse
import cv2
import numpy as np
import sys

# ================= 配置区域 =================
VIDEO_ROOT = "/share_data/guantianrui/datasets/VITRA-1M/video_root_new"
JSON_PATHS = [
    "/share_data/guantianrui/datasets/VITRA-1M/zarr_episode_stat_new/ego4d_cooking_zarr_episode_statistics.json",
    "/share_data/guantianrui/datasets/VITRA-1M/zarr_episode_stat_new/epic_zarr_episode_statistics.json",
    "/share_data/guantianrui/datasets/VITRA-1M/zarr_episode_stat_new/ego4d_other_zarr_episode_statistics.json",
    "/share_data/guantianrui/datasets/VITRA-1M/zarr_episode_stat_new/ssv2_zarr_episode_statistics.json"
]
LOG_ROOT = "/share_data/yifan/projects/depth/Any4D/vitra_check_results"
# ===========================================

def setup_logging(cpu_id):
    """
    【核心修改】
    将当前进程的所有输出（Stdout + Stderr）重定向到日志文件。
    包括 C++ 库 (FFmpeg/Decord) 的输出。
    """
    os.makedirs(LOG_ROOT, exist_ok=True)
    log_path = os.path.join(LOG_ROOT, f"{cpu_id}.txt")
    
    # 1. 打开日志文件
    # buffering=1 表示行缓冲，每打印一行就写入磁盘，防止程序崩了日志没存下来
    log_file = open(log_path, "w", buffering=1, encoding='utf-8')
    
    # 2. 刷新 Python 缓冲区，防止之前的输出乱序
    sys.stdout.flush()
    sys.stderr.flush()
    
    # 3. 获取文件描述符
    fd_log = log_file.fileno()
    fd_stdout = 1
    fd_stderr = 2
    
    # 4. 【魔法时刻】使用 os.dup2 强行修改底层指向
    # 以后所有往 stdout(1) 写的内容，都会去 fd_log
    # 以后所有往 stderr(2) 写的内容，也都会去 fd_log
    os.dup2(fd_log, fd_stdout)
    os.dup2(fd_log, fd_stderr)
    
    # 5. 为了保险，把 Python 对象的指向也更新一下（虽然 dup2 已经接管了底层）
    sys.stdout = log_file
    sys.stderr = log_file
    
    print(f"Process started for CPU {cpu_id}. Logs redirected to {log_path}")

def get_video_filename(video_name_key):
    if video_name_key.startswith("Ego4D_"):
        return f"{video_name_key[6:]}.mp4"
    elif video_name_key.startswith("somethingsomethingv2_"):
        return f"{video_name_key[21:]}.webm"
    elif video_name_key.startswith("epic_kitchens_"):
        return f"{video_name_key[14:]}.MP4"
    else:
        return None

def read_video_cv2(video_path, start_idx, end_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"OpenCV failed to open {video_path}")
    
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    current_idx = start_idx
    
    while current_idx <= end_idx:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        current_idx += 1
    cap.release()
    
    if len(frames) == 0:
        raise IOError("OpenCV read 0 frames")
    return np.array(frames)

def process(tasks):
    # 注意：现在不需要传 log_file 进来了，直接用 print 就会自动写到文件
    
    # 为了避免日志文件里全是进度条刷屏，这里把 tqdm 的 output 设为 None 或者 file=sys.stdout
    # 既然已经重定向了，tqdm 也会写到文件里。
    # 建议：在重定向模式下，尽量不要用 tqdm，或者设置 mininterval 大一点，否则日志文件会全是 \r 符号
    for i, task in enumerate(tasks):
        if i % 100 == 0:
            print(f"Processing task {i}/{len(tasks)}...", flush=True)

        ep_name = task['ep_name']
        print(ep_name)
        s = task['start']
        e = task['end']
        duration = e - s + 1
        
        # 1. 检查长度
        if duration > 180:
            print(f"[LONG] {ep_name} has {duration} frames.", flush=True)
        
        # 2. 检查前缀和路径
        video_key = task['video_key']
        filename = get_video_filename(video_key)
        if filename is None:
            print(f"[ERROR] {ep_name}: Unknown Prefix: {video_key}", flush=True)
            continue
            
        video_path = os.path.join(VIDEO_ROOT, filename)
        if not os.path.exists(video_path):
            print(f"[ERROR] {ep_name}: File Missing", flush=True)
            continue

        # 3. 视频解码检查
        try:
            if s > e:
                print(f"[ERROR] {ep_name}: Invalid range ({s}-{e})", flush=True)
                continue

            # decord 可能会往 stderr 打印红字，现在会被 os.dup2 捕获进文件
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=0)
            
            if e >= len(vr):
                print(f"[ERROR] {ep_name}: IndexOOB: req_end={e}, video_len={len(vr)}", flush=True)
                continue
            
            indices = list(range(s, e + 1))
            frames = vr.get_batch(indices).asnumpy()
            
            if frames.shape[0] != duration:
                print(f"[ERROR] {ep_name}: Decord Frame mismatch: req={duration}, got={frames.shape[0]}", flush=True)

        except Exception as decord_err:
            try:
                # 尝试 OpenCV
                _ = read_video_cv2(video_path, s, e)
                # OpenCV 成功，忽略 Decord 错误，或者打个 Warning
                print(f"[WARN] {ep_name}: Decord failed but OpenCV ok.", flush=True)
                pass
            except Exception as cv2_err:
                # 彻底挂了
                print(f"[FATAL] {ep_name}: Read Failed. Decord: {decord_err} | OpenCV: {cv2_err}", flush=True)

def load_all_tasks(cpu_id, total_cpus):
    all_tasks = []
    for json_path in JSON_PATHS:
        if not os.path.exists(json_path):
            print(f"[WARN] JSON not found: {json_path}") # 这也会进日志
            continue
            
        data = json.load(open(json_path))
        stats = data.get('video_statistics', [])
        
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
                except:
                    pass
    
    total_len = len(all_tasks)
    tasks_per_cpu = total_len // total_cpus
    start_idx = cpu_id * tasks_per_cpu
    if cpu_id == total_cpus - 1:
        end_idx = total_len
    else:
        end_idx = (cpu_id + 1) * tasks_per_cpu
        
    my_tasks = all_tasks[start_idx:end_idx]
    
    print(f"CPU {cpu_id} assigned {len(my_tasks)} tasks (Index {start_idx}-{end_idx})", flush=True)
    return my_tasks

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu_id", type=int, default=0)
    parser.add_argument("--total_cpus", type=int, default=512)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 这一行必须最先执行！执行后，终端里就什么都看不到了，全在文件里
    setup_logging(args.cpu_id)
    
    # 2. 加载任务
    tasks = load_all_tasks(args.cpu_id, args.total_cpus)
    
    # 3. 处理
    process(tasks)

if __name__ == "__main__":
    main()