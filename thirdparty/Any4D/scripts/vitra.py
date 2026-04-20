import tarfile
import io
import os
import cv2
import decord
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import resize_intrinsics, load_images_vitra

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


class VitraDataset(Dataset):
    def __init__(self, episode_list, output_root, chunk_len, high_level_config):
        self.chunk_len = chunk_len
        self.high_level_config = high_level_config
        self.output_root = output_root
        
        self.tasks = []

        skip_task = False
        
        for ep_info in tqdm(episode_list, desc="Preparing Tasks"):
            zarr_ep_name = ep_info['zarr_episode_name']
            video_path = ep_info['video_path']
            start_frame = ep_info['original_start_frame']
            end_frame = ep_info['original_end_frame'] # 闭区间 [start, end]
            
            npz_out_path = os.path.join(self.output_root, f"{zarr_ep_name}.npz")
            if os.path.exists(npz_out_path):
                skip_task = True
                continue
            
            # 【修正】闭区间计算总帧数
            # e.g. 4-25 -> 25-4+1 = 22 frames
            total_frames_in_ep = end_frame - start_frame + 1
            
            if total_frames_in_ep <= 0:
                print(f"[WARN] Skipped {zarr_ep_name} due to duration <= 0")
                continue
            
            # 切分 Chunk
            num_chunks = (total_frames_in_ep + self.chunk_len - 1) // self.chunk_len
            
            for i in range(num_chunks):
                # 计算相对偏移 (相对于 start_frame)
                if i == num_chunks - 1:
                    chunk_end_rel = total_frames_in_ep # 这里作为 slice end，直接用总长度
                else:
                    chunk_end_rel = (i + 1) * self.chunk_len
                
                chunk_start_rel = max(0, chunk_end_rel - self.chunk_len)
                
                # 转换为绝对帧索引 (用于 Decord 读取)
                # abs_start_idx 是 Inclusive Start
                abs_start_idx = start_frame + chunk_start_rel
                
                # abs_end_idx 是 Exclusive End (用于 Python range)
                # start_frame + chunk_end_rel
                # 举例: total=22. Last chunk: chunk_end_rel=22.
                # abs_end = 4 + 22 = 26.
                # range(4, 26) -> 4, ..., 25 (共22帧，包含25). 正确！
                abs_end_idx = start_frame + chunk_end_rel
                
                self.tasks.append({
                    "zarr_episode_name": zarr_ep_name,
                    "video_path": video_path,
                    "npz_path": npz_out_path,
                    "abs_start_idx": abs_start_idx,
                    "abs_end_idx": abs_end_idx,
                    "total_len": total_frames_in_ep,
                    "chunk_seq_id": i,
                    "is_last_chunk": (i == num_chunks - 1)
                })
        
        print(f"Skip task in Dataset? {skip_task}")

        print(f"Total chunks to process: {len(self.tasks)}")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        
        try:
            s = task['abs_start_idx']
            e = task['abs_end_idx']
            video_path = task['video_path']
            
            # =================================================
            # 尝试 1: Decord (速度快，但对坏视频敏感)
            # =================================================
            try:
                # 显式指定 num_threads=0 让 ffmpeg 自动管理，有时候能增加稳定性
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=0)
                
                # 边界保护
                if e > len(vr):
                    print(f"[WARN] Adjusting end index for {task['zarr_episode_name']} from {e} to {len(vr)}")
                    e = len(vr)
                if s >= e: 
                    print(f"[WARN] Adjusting start index for {task['zarr_episode_name']} from {s} to {e-1}")
                    s = max(0, e - 1)
                
                indices_ori = list(range(s, e))
                mid_idx = (s + e) // 2
                indices = [mid_idx] + indices_ori
                video_images_chunk = vr.get_batch(indices).asnumpy()

            except Exception as decord_error:
                # =================================================
                # 尝试 2: OpenCV (速度慢，但鲁棒性强)
                # =================================================
                # print(f"[WARN] Decord failed for {task['zarr_episode_name']}, trying OpenCV... Error: {decord_error}")
                print("Switching to OpenCV for video reading...")
                video_images_chunk = read_video_cv2(video_path, s, e)
            
            # =================================================
            
            # 2. 预处理
            views = load_images_vitra(
                video_images_chunk,
                norm_type=self.high_level_config["data_norm_type"]
            )
            
            return views, task
            
        except Exception as e:
            # 如果 OpenCV 也挂了，那是真的没救了
            print(f"[ERROR] Failed to process task {task['zarr_episode_name']}")
            print(f"        Video: {task['video_path']}")
            print(f"        Error: {e}")
            return None, None