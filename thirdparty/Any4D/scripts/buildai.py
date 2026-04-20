import os
import cv2
import numpy as np
import joblib
import decord
from torch.utils.data import Dataset
from utils import resize_intrinsics, load_images, find_closest_aspect_ratio

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

class BuildAIDataset(Dataset):
    def __init__(self, episode_list, output_root, chunk_len, high_level_config):
        self.episode_list = episode_list # 传入已经匹配好 mp4 和 pkl 的列表
        self.chunk_len = chunk_len
        self.high_level_config = high_level_config
        self.output_root = output_root

        # 固定内参
        self.K_ori = np.array([
            [1030.59, 0,       966.69],
            [0,       1032.82, 539.69],
            [0,       0,       1]
        ])

        self.tasks = []
        self._prepare_tasks()

    def _prepare_tasks(self):
        import cv2  # 确保导入了 cv2
        print(f"Generating chunks for {len(self.episode_list)} BuildAI episodes...")
        for ep in self.episode_list:
            video_path = ep['video_path']
            pose_path = ep['pose_path']
            ep_name = ep['ep_name'] 
            if os.path.exists(os.path.join(self.output_root, f"{ep_name}.npz")):
                print(f"[INFO] Skipping {ep_name}, npz already exists.")
                continue
            
            video_len = 0
            # 1. 优先尝试使用 decord 读取长度
            try:
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
                video_len = len(vr)
            except Exception as e:
                # 2. 如果 decord 失败，回退到 OpenCV
                try:
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print(f"[WARN] Failed to open video with both decord and opencv: {video_path}")
                        continue
                    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                except Exception as e2:
                    print(f"[ERROR] Could not get video length for {video_path}: {e2}")
                    continue

            # 检查获取到的长度是否有效
            if video_len <= 0:
                print(f"[WARN] Invalid video length ({video_len}) for: {video_path}")
                continue

            # 3. 分 Chunk 逻辑
            num_chunks = (video_len + self.chunk_len - 1) // self.chunk_len
            for i in range(num_chunks):
                chunk_end = min((i + 1) * self.chunk_len, video_len)
                chunk_start = max(0, chunk_end - self.chunk_len)
                
                self.tasks.append({
                    "ep_name": ep_name,
                    "video_path": video_path,
                    "pose_path": pose_path,
                    "npz_path": os.path.join(self.output_root, f"{ep_name}.npz"),
                    "start_idx": chunk_start,
                    "end_idx": chunk_end,
                    "total_len": video_len,
                    "chunk_seq_id": i,
                    "is_last_chunk": (i == num_chunks - 1)
                })

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        try:
            s, e = task['start_idx'], task['end_idx']
            
            # 1. Load Video
            try:
                vr = decord.VideoReader(task['video_path'], ctx=decord.cpu(0), num_threads=0)
                mid = (s + e) // 2
                indices = [mid] + list(range(s, e))
                video_images = vr.get_batch(indices).asnumpy()
            except:
                print("Switching to OpenCV for video reading...")
                video_images = read_video_cv2(task['video_path'], s, e)

            # 2. Load Poses (World2Cam)
            pose_data = joblib.load(task['pose_path'])
            Rs = pose_data["slam_R_w2c"][s:e]
            ts = pose_data["slam_t_w2c"][s:e]
            R_mid = pose_data["slam_R_w2c"][mid]
            t_mid = pose_data["slam_t_w2c"][mid]

            extrinsics = []
            def make_4x4(R, t):
                m = np.eye(4)
                m[:3, :3]= R
                m[:3, 3] = t
                return m
            
            extrinsics.append(make_4x4(R_mid, t_mid)) # Ref
            for i in range(len(Rs)):
                extrinsics.append(make_4x4(Rs[i], ts[i])) # Views

            # 3. Resize Intrinsics
            # BuildAI 原始 1920x1080. Any4D 常用宽度 518
            target_w, target_h = find_closest_aspect_ratio(1920/1080, 518)
            K_res = resize_intrinsics(self.K_ori, old_size=(1080, 1920), new_size=(target_h, target_w))

            views = load_images(video_images, extrinsics, K_res, 
                                norm_type=self.high_level_config["data_norm_type"])
            return views, task
        except Exception as e:
            print(f"Error loading {task['ep_name']}: {e}")
            return None, None