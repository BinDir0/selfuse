import os
import json
import decord
import multiprocessing
import time
from tqdm import tqdm

# ================= 配置区域 =================

# 数据集根目录
VIDEO_ROOT = "/share_data/guantianrui/datasets/VITRA-1M/video_root"

# JSON 路径列表 (Train 和 Test)
JSON_PATHS = [
    "/share_data/guantianrui/datasets/VITRA-1M/zarr_episodes_statistics.json",
    "/share_data/guantianrui/datasets/VITRA-1M/zarr_episodes_statistics_testset.json"
]

# 输出日志文件
LOG_ERROR = "vitra_check_errors.txt"   # 记录损坏、缺失、无法解码的
LOG_LONG  = "vitra_check_long.txt"     # 记录 > 180 帧的

# 并发数 (根据你的 64 核服务器设置)
NUM_WORKERS = 64
# 帧数阈值
MAX_FRAMES = 180

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

def worker_check(task):
    """
    工作进程：对单个 Episode 进行检查
    """
    ep_name = task['ep_name']
    video_key = task['video_key']
    s = task['start']
    e = task['end']
    
    # --- 1. 逻辑检查 (速度最快) ---
    # 计算帧数 (闭区间)
    duration = e - s + 1
    
    if duration > MAX_FRAMES:
        return {
            "status": "long",
            "ep_name": ep_name,
            "info": f"{duration}"
        }
    
    # 简单的逻辑错误检查
    if duration <= 0:
        return {
            "status": "error",
            "ep_name": ep_name,
            "info": f"Invalid duration: {duration} ({s}-{e})"
        }

    # --- 2. 文件存在性检查 (轻量 IO) ---
    filename = get_video_filename(video_key)
    if filename is None:
        return {"status": "error", "ep_name": ep_name, "info": f"Unknown Prefix: {video_key}"}
        
    video_path = os.path.join(VIDEO_ROOT, filename)
    
    if not os.path.exists(video_path):
        return {
            "status": "error", 
            "ep_name": ep_name, 
            "info": f"File Missing" # 简化日志，不打印全路径，省空间
        }

    # --- 3. 视频解码检查 (重 IO + CPU) ---
    try:
        # 使用 cpu(0) 避免 CUDA 初始化，num_threads=0 让 decord 自动调度
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=0)
        
        # 检查越界
        if e >= len(vr):
            return {
                "status": "error",
                "ep_name": ep_name,
                "info": f"IndexOOB: req_end={e}, video_len={len(vr)}"
            }
        
        # 【关键】尝试读取像素
        # 这一步会触发 FFmpeg 的底层解码。
        # 如果文件损坏（Invalid NAL unit 等），这里可能会抛出异常，
        # 或者虽然不抛异常但 FFmpeg 会在 stderr 打印红字。
        # 我们只能捕获抛出的异常。
        indices = list(range(s, e + 1))
        frames = vr.get_batch(indices).asnumpy()
        
        # 检查读取出来的帧数是否对齐
        if frames.shape[0] != duration:
             return {
                "status": "error",
                "ep_name": ep_name,
                "info": f"Frame mismatch: req={duration}, got={frames.shape[0]}"
            }

        return None # 一切正常

    except Exception as exc:
        # 捕获所有解码错误
        return {
            "status": "error",
            "ep_name": ep_name,
            "info": str(exc).replace('\n', ' ') # 保持一行
        }

def load_all_tasks():
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
            
    print("-" * 50)
    return all_tasks

def main():
    # 1. 准备任务
    tasks = load_all_tasks()
    total_tasks = len(tasks)
    
    if total_tasks == 0:
        print("No tasks found. Exiting.")
        return

    print(f"Starting check on {total_tasks} episodes.")
    print(f"Workers: {NUM_WORKERS} | Video Root: {VIDEO_ROOT}")
    print(f"Logs: {LOG_ERROR} / {LOG_LONG}")
    
    # 2. 初始化日志文件 (清空)
    with open(LOG_ERROR, 'w') as f:
        f.write("episode_name\terror_info\n")
    with open(LOG_LONG, 'w') as f:
        f.write("episode_name\tframe_count\n")

    # 3. 多进程处理
    # chunksize 设置大一点 (例如 100)，可以减少主进程和子进程之间的通信开销
    # 因为任务总数有 100万+
    chunk_size = 100 
    
    error_cnt = 0
    long_cnt = 0
    
    print("Processing... (This may take a while)")
    
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        # 使用 imap_unordered 提高吞吐
        iterator = pool.imap_unordered(worker_check, tasks, chunksize=chunk_size)
        
        with open(LOG_ERROR, 'a') as f_err, open(LOG_LONG, 'a') as f_long:
            # tqdm 显示进度条
            for res in tqdm(iterator, total=total_tasks, unit="ep"):
                if res is None:
                    continue
                
                status = res['status']
                line = f"{res['ep_name']}\t{res['info']}\n"
                
                if status == 'error':
                    error_cnt += 1
                    f_err.write(line)
                elif status == 'long':
                    long_cnt += 1
                    f_long.write(line)
                    
    print("\n" + "=" * 50)
    print("Check Completed.")
    print(f"Total Checked: {total_tasks}")
    print(f"Errors Found : {error_cnt}  -> Saved to {LOG_ERROR}")
    print(f"Too Long     : {long_cnt}  -> Saved to {LOG_LONG}")
    print("=" * 50)
    print("提示：")
    print("1. 如果终端出现红色 FFmpeg 报错，说明该文件存在底层损坏，但如果未抛出 Python 异常，")
    print("   脚本可能不会记录到 error log 中 (视 decord 健壮性而定)。")
    print("2. 你可以使用 cat vitra_check_errors.txt | awk '{print $1}' 来获取所有坏 Episode 的名字。")

if __name__ == "__main__":
    main()