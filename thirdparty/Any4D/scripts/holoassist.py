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
from utils import resize_intrinsics, load_images_holoassist, find_closest_aspect_ratio

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

class HoloAssistDataset(Dataset):
    def __init__(self, task_list, output_root, chunk_len, high_level_config):
        self.chunk_len = chunk_len
        self.overlap = 10  # 【新增】设置重叠帧数
        self.high_level_config = high_level_config
        self.output_root = output_root
        
        self.video_root_base = "/share_data/guantianrui/datasets/HoloAssist/raw_data/new_video"
        self.cam_info_base = "/share_data/guantianrui/datasets/HoloAssist/raw_data/cam_info"
        
        self.tasks = []
        
        print(f"Scanning assigned tasks ({len(task_list)})...")
        
        for task_name in tqdm(task_list, desc="Preparing Tasks"):

            if not task_name.startswith("R") and not task_name.startswith("z"):
                print(f"[WARN] Skipping unexpected task name: {task_name}")
                continue
            
            # 1. 构造路径
            video_path = os.path.join(self.video_root_base, task_name, "Export_py", "Video_pitchshift.mp4")
            intr_path = os.path.join(self.cam_info_base, task_name, "Export_py", "Video", "Intrinsics.txt")
            npz_out_path = os.path.join(self.output_root, f"{task_name}.npz")
            
            # 2. 断点续传
            if os.path.exists(npz_out_path):
                continue
            
            # 3. 严格检查
            if not os.path.exists(video_path):
                print(f"[ERROR] Video missing: {video_path}")
                continue
            if not os.path.exists(intr_path):
                print(f"[ERROR] Intrinsics missing: {intr_path}")
                continue
                
            # 4. 解析内参
            try:
                intr_data = np.loadtxt(intr_path)
                assert intr_data.shape == (25,), f"Intrinsics shape mismatch {task_name}"
                fx, fy, cx, cy = intr_data[-7:-3]
                assert fx > 0 and fy > 0 and cx > 0 and cy > 0, "Invalid intrinsics values"
                
                K_ori = np.eye(3)
                K_ori[0, 0] = fx
                K_ori[1, 1] = fy
                K_ori[0, 2] = cx
                K_ori[1, 2] = cy
            except Exception as e:
                print(f"[ERROR] Intrinsics parse error {task_name}: {e}")
                continue
            
            # 5. 读取视频长度
            try:
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
                video_len = len(vr)
                h, w, _ = vr[0].shape
                assert h == 504 and w == 896, f"Resolution mismatch {task_name}: {w}x{h}"
            except Exception as e:
                print(f"[ERROR] Video read error {task_name}: {e}")
                continue
                
            if video_len == 0: continue
                
            # 6. 切分 Chunk (使用 Overlap)
            # 步长 stride
            stride = self.chunk_len - self.overlap
            
            # 计算需要多少个 chunk
            # 这种切分方式确保覆盖 [0, video_len]
            if video_len <= self.chunk_len:
                num_chunks = 1
            else:
                num_chunks = int(np.ceil((video_len - self.overlap) / stride))
            
            for i in range(num_chunks):
                # 计算 start
                chunk_start = i * stride
                
                # 计算 end
                chunk_end = chunk_start + self.chunk_len
                
                # 最后一个 chunk 特殊处理：贴着视频末尾
                # 注意：这样会导致最后一个 chunk 和倒数第二个 chunk 的重叠可能 > 10
                # 但这是保证不越界的常用做法
                if chunk_end > video_len:
                    chunk_end = video_len
                    chunk_start = max(0, chunk_end - self.chunk_len)
                
                self.tasks.append({
                    "task_name": task_name,
                    "video_path": video_path,
                    "intrinsics_ori": K_ori,
                    "npz_path": npz_out_path,
                    "start_idx": chunk_start,
                    "end_idx": chunk_end,
                    "total_len": video_len,
                    "chunk_seq_id": i,
                    "is_last_chunk": (i == num_chunks - 1)
                })

        print(f"Total chunks to process: {len(self.tasks)}")
        
    # __len__ 和 __getitem__ 保持不变，
    # 因为 __getitem__ 只需要根据 task 里的 start_idx/end_idx 读数据即可
    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        
        try:
            s = task['start_idx']
            e = task['end_idx']
            video_path = task['video_path']
            
            # =================== 1. 读取视频 (双保险) ===================
            try:
                # 尝试 Decord
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=0)
                if e > len(vr):
                    print(f"[WARN] Adjusting end index from {e} to {len(vr)} for {task['task_name']}")
                    e = len(vr)
                if s >= e:
                    print(f"[WARN] Adjusting start index from {s} to {e-1} for {task['task_name']}")
                    s = max(0, e - 1)
                indices_range = list(range(s, e))
                mid = (s + e) // 2
                indices = [mid] + indices_range
                video_images_chunk = vr.get_batch(indices).asnumpy()
                
            except Exception:
                # Fallback to OpenCV
                # 假设 read_video_cv2 已经在类外定义好了 (复用 VITRA 的逻辑)
                video_images_chunk = read_video_cv2(video_path, s, e)
            
            # =================== 2. 处理内参 ===================
            # 原始尺寸: 896 (W) x 504 (H)
            # 目标尺寸: resize_intrinsics 会根据目标 518 计算
            # 注意: resize_intrinsics(K, old_size=(H, W), new_size=(target_H, target_W))
            
            # 计算 target size (复用 load_images 里的逻辑 or calc here)
            # 我们需要在调用 load_images 之前准备好 resize 后的内参
            # find_closest_aspect_ratio 已经在外部定义
            
            # 计算 Aspect Ratio
            # W=896, H=504 -> 1.7777 (16:9)
            # find_closest_aspect_ratio(1.777, 518) -> likely (518, 294)
            
            target_w, target_h = find_closest_aspect_ratio(896/504, 518)
            
            K_resized = resize_intrinsics(
                task['intrinsics_ori'],
                old_size=(504, 896), # (H, W)
                new_size=(target_h, target_w) # (H, W)
            )
            
            # =================== 3. 预处理 ===================
            views = load_images_holoassist(
                video_images_chunk,
                intrinsics=K_resized,
                norm_type=self.high_level_config["data_norm_type"]
            )
            
            return views, task
            
        except Exception as e:
            print(f"[ERROR] Failed task {task['task_name']} chunk {task['chunk_seq_id']}: {e}")
            return None, None