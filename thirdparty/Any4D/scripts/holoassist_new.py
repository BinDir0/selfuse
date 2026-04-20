import tarfile
import io
import os
import cv2
import decord
import pickle
import numpy as np
import joblib
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import resize_intrinsics, load_images, find_closest_aspect_ratio

# ==============================================================================
# Helper: Video Reader
# ==============================================================================
def read_video_cv2(video_path, start_idx, end_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"OpenCV failed to open {video_path}")
    
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    
    current_idx = start_idx
    while current_idx < end_idx:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        current_idx += 1
        
    cap.release()
    if len(frames) == 0:
        raise IOError(f"OpenCV read 0 frames from {video_path}")

    mid_idx = len(frames) // 2
    frames_final = [frames[mid_idx]] + frames
    return np.array(frames_final)

# ==============================================================================
# Dataset: HoloAssist (Updated with GT Extrinsics)
# ==============================================================================
class HoloAssistNewDataset(Dataset):
    def __init__(self, task_list, output_root, chunk_len, high_level_config):
        self.chunk_len = chunk_len
        self.high_level_config = high_level_config
        self.output_root = output_root
        
        self.video_root_base = "/share_data/guantianrui/datasets/HoloAssist/raw_data/video"
        self.cam_info_base = "/share_data/guantianrui/datasets/HoloAssist/raw_data/cam_info"
        self.pose_root_base = "/share_data/guantianrui/datasets/HoloAssist/output/holoassist_results"
        
        self.tasks = []
        
        print(f"Scanning assigned tasks ({len(task_list)})...")
        
        for task_name in tqdm(task_list, desc="Preparing Tasks"):
            
            # Filter non-task folders
            if not (task_name.startswith("R") or task_name.startswith("z")):
                continue
            
            # 1. Construct Paths
            video_path = os.path.join(self.video_root_base, task_name, "Export_py", "Video_pitchshift.mp4")
            intr_path = os.path.join(self.cam_info_base, task_name, "Export_py", "Video", "Intrinsics.txt")
            pose_path = os.path.join(self.pose_root_base, task_name, "hawor_results.pkl")
            npz_out_path = os.path.join(self.output_root, f"{task_name}.npz")
            
            # 2. Check Existance
            if os.path.exists(npz_out_path):
                continue

            if not os.path.exists(video_path):
                print(f"[WARN] Video missing: {video_path}")
                continue
            if not os.path.exists(intr_path):
                print(f"[WARN] Intrinsics missing: {intr_path}")
                continue
            if not os.path.exists(pose_path):
                # User instruction: ignore episodes without hawor_results.pkl
                # print(f"[WARN] Pose pkl missing (Quality check failed): {pose_path}")
                continue
                
            # 3. Parse Metadata (Intrinsics & Length Check)
            try:
                # A. Intrinsics
                intr_data = np.loadtxt(intr_path)
                if intr_data.shape != (25,):
                    print(f"[WARN] Intrinsics shape mismatch {task_name}: {intr_data.shape}")
                    continue
                fx, fy, cx, cy = intr_data[-7:-3]
                if not (fx > 0 and fy > 0 and cx > 0 and cy > 0):
                    print(f"[WARN] Invalid intrinsics values {task_name}")
                    continue
                
                K_ori = np.eye(3)
                K_ori[0, 0] = fx
                K_ori[1, 1] = fy
                K_ori[0, 2] = cx
                K_ori[1, 2] = cy

                # B. Poses (Load just to check length/validity)
                pose_data = joblib.load(pose_path)
                
                if "slam_R_w2c" not in pose_data or "slam_t_w2c" not in pose_data:
                    print(f"[WARN] Missing keys in pickle {task_name}")
                    continue
                    
                num_poses = len(pose_data["slam_R_w2c"])
                # C. Video Length
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
                video_len = len(vr)
                h, w, _ = vr[0].shape
                
                if h != 504 or w != 896:
                    print(f"[WARN] Resolution mismatch {task_name}: {w}x{h} (Expected 896x504)")
                    continue

                # D. Alignment Check
                if num_poses != video_len:
                    print(f"[WARN] Length mismatch {task_name}: Poses {num_poses} vs Video {video_len}")
                    continue

            except Exception as e:
                print(f"[ERROR] Init failed for {task_name}: {e}")
                continue
                
            # 4. Chunking
            # Simple [0, 180], [180, 360]...
            num_chunks = (video_len + self.chunk_len - 1) // self.chunk_len
            
            for i in range(num_chunks):
                if i == num_chunks - 1:
                    chunk_end = video_len
                else:
                    chunk_end = (i + 1) * self.chunk_len
                
                chunk_start = max(0, chunk_end - self.chunk_len)
                
                # Double check bounds
                if chunk_start >= chunk_end:
                    continue

                self.tasks.append({
                    "task_name": task_name,
                    "video_path": video_path,
                    "pose_path": pose_path,
                    "intrinsics_ori": K_ori,
                    "npz_path": npz_out_path,
                    "start_idx": chunk_start,
                    "end_idx": chunk_end,
                    "total_len": video_len,
                    "chunk_seq_id": i,
                    "is_last_chunk": (i == num_chunks - 1)
                })

        print(f"Total chunks to process: {len(self.tasks)}")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        
        try:
            s = task['start_idx']
            e = task['end_idx']
            video_path = task['video_path']
            pose_path = task['pose_path']
            
            # =================== 1. Load Video ===================
            try:
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=0)
                # Bounds check inside getitem just in case
                if e > len(vr):
                    print(f"[WARN] Adjusting end index to video length: {task['task_name']}")
                    e = len(vr)
                if s >= e:
                    print(f"[WARN] Adjusting start index to valid range: {task['task_name']}")
                    s = max(0, e - 1)
                
                indices_range = list(range(s, e))
                mid = (s + e) // 2
                indices = [mid] + indices_range # [Ref, Src0, Src1...]
                video_images_chunk = vr.get_batch(indices).asnumpy()
            except Exception:
                video_images_chunk = read_video_cv2(video_path, s, e)
            
            # =================== 2. Load Extrinsics (GT) ===================
            # Need to load pickle inside getitem to be multi-process safe/friendly
            pose_data = joblib.load(pose_path)
            
            # Extract Chunk
            # Data is World2Cam
            Rs = pose_data["slam_R_w2c"][s:e] # (T, 3, 3)
            ts = pose_data["slam_t_w2c"][s:e] # (T, 3)
            
            # Need mid frame for Ref
            mid_idx_rel = (s + e) // 2
            
            R_mid = pose_data["slam_R_w2c"][mid_idx_rel]
            t_mid = pose_data["slam_t_w2c"][mid_idx_rel]

            extrinsics_chunk = []
            
            def get_w2c(R, t):
                w2c = np.eye(4)
                w2c[:3, :3] = R
                w2c[:3, 3] = t
                return w2c
            
            # Add Ref
            extrinsics_chunk.append(get_w2c(R_mid, t_mid))
            
            # Add Views
            for i in range(len(Rs)):
                extrinsics_chunk.append(get_w2c(Rs[i], ts[i]))

            # =================== 3. Intrinsics ===================
            # Original: 896 x 504
            # Target: 518 (Width matches aspect ratio logic in Any4D usually)
            # Logic: fit height or width?
            # 896/504 = 1.77. 518/294 = 1.76.
            target_w, target_h = find_closest_aspect_ratio(896/504, 518)
            
            K_resized = resize_intrinsics(
                task['intrinsics_ori'],
                old_size=(504, 896), # (H, W)
                new_size=(target_h, target_w)
            )

            # =================== 4. Package ===================
            views = load_images(
                video_images_chunk,
                extrinsics_chunk,
                K_resized,
                norm_type=self.high_level_config["data_norm_type"]
            )
            
            return views, task
            
        except Exception as e:
            print(f"[ERROR] Processing failure {task['task_name']} chunk {task['chunk_seq_id']}: {e}")
            return None, None