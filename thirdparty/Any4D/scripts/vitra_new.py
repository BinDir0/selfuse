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
from utils import resize_intrinsics, load_images, find_closest_aspect_ratio

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
# Dataset: VITRA (Updated with GT Geometry)
# ==============================================================================
class VitraNewDataset(Dataset):
    def __init__(self, episode_list, output_root, annotation_map, chunk_len, high_level_config):
        self.chunk_len = chunk_len
        self.high_level_config = high_level_config
        self.output_root = output_root
        self.tasks = []
        self.annotation_map = annotation_map

        print(f"Initializing Dataset with {len(episode_list)} episodes...")
        
        for ep_info in tqdm(episode_list, desc="Preparing Tasks"):
            zarr_ep_name = ep_info['zarr_episode_name']
            video_path = ep_info['video_path']
            start_frame = ep_info['original_start_frame']
            end_frame = ep_info['original_end_frame'] # Inclusive in JSON logic usually
            
            # Check if annotation exists
            if zarr_ep_name not in self.annotation_map:
                # print(f"[WARN] No .npy annotation found for {zarr_ep_name}, skipping.")
                continue
            
            npy_path = self.annotation_map[zarr_ep_name]
            
            npz_out_path = os.path.join(self.output_root, f"{zarr_ep_name}.npz")
            if os.path.exists(npz_out_path):
                continue
            
            # Calculate total frames (Inclusive start, Inclusive end)
            total_frames_in_ep = end_frame - start_frame + 1
            if total_frames_in_ep <= 0:
                continue
            
            num_chunks = (total_frames_in_ep + self.chunk_len - 1) // self.chunk_len
            
            for i in range(num_chunks):
                if i == num_chunks - 1:
                    chunk_end_rel = total_frames_in_ep 
                else:
                    chunk_end_rel = (i + 1) * self.chunk_len
                
                chunk_start_rel = max(0, chunk_end_rel - self.chunk_len)
                
                # Absolute indices for Video Reader (Raw Video)
                abs_start_idx = start_frame + chunk_start_rel
                abs_end_idx = start_frame + chunk_end_rel # Exclusive for range/decord
                
                self.tasks.append({
                    "zarr_episode_name": zarr_ep_name,
                    "video_path": video_path,
                    "npy_path": npy_path,
                    "npz_path": npz_out_path,
                    # Video Reading (Absolute)
                    "abs_start_idx": abs_start_idx,
                    "abs_end_idx": abs_end_idx,
                    # Annotation Reading (Relative to the start of the episode)
                    "rel_start_idx": chunk_start_rel, 
                    "rel_end_idx": chunk_end_rel,
                    # Meta
                    "total_len": total_frames_in_ep,
                    "chunk_seq_id": i,
                    "is_last_chunk": (i == num_chunks - 1)
                })

        print(f"Total chunks to process: {len(self.tasks)}")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        
        try:
            # 1. Load Annotation Data
            # ------------------------------------------------------------------
            # We load this first to fail fast if corrupt, and to get dimensions for intrinsics
            ep_data = np.load(task['npy_path'], allow_pickle=True).item()
            
            # Extrinsics: (T, 4, 4) - Slicing
            # The NPY is the "episodic annotation", so index 0 corresponds to 'original_start_frame'
            full_extrinsics = ep_data["extrinsics"] 
            
            rel_s = task['rel_start_idx']
            rel_e = task['rel_end_idx']
            
            # Safety clamp for extrinsics slicing
            if rel_e > len(full_extrinsics) - 1:
                print(f"[WARN] Clamping rel_e from {rel_e} to {len(full_extrinsics)} for {task['zarr_episode_name']}")
                rel_e = len(full_extrinsics)
            if rel_s >= rel_e:
                 print("[WARN] Clamping rel_s to max 0 for", task['zarr_episode_name'])
                 rel_s = max(0, rel_e - 1)
                 
            # Slice Chunk Extrinsics
            # Need to pick the middle frame for the 'ref' view logic
            mid_rel_idx = (rel_s + rel_e) // 2
            
            # Format: [Ref View, View 0, View 1, ...]
            extrinsics_chunk_data = [full_extrinsics[mid_rel_idx]] + list(full_extrinsics[rel_s:rel_e])
            
            # Intrinsics: (3, 3) - Usually constant per episode
            intrinsics_ori = ep_data["intrinsics"]
            
            # 2. Load Video Data
            # ------------------------------------------------------------------
            s = task['abs_start_idx']
            e = task['abs_end_idx']
            video_path = task['video_path']
            
            video_images_chunk = None
            
            try:
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=0)
                # Get resolution from VR for resizing intrinsics
                h_ori, w_ori, _ = vr[0].shape 
                
                # Clamp indices
                if e > len(vr):
                    print(f"[WARN] Clamping e from {e} to {len(vr)} for {task['zarr_episode_name']}")
                    e = len(vr)
                if s >= e:
                    print("[WARN] Clamping s to max 0 for", task['zarr_episode_name'])
                    s = max(0, e - 1)
                
                indices_ori = list(range(s, e))
                mid_idx_abs = (s + e) // 2
                indices = [mid_idx_abs] + indices_ori
                
                video_images_chunk = vr.get_batch(indices).asnumpy()
                
            except Exception:
                # Fallback to CV2
                # Note: CV2 logic in read_video_cv2 doesn't return (h,w), assume standard or update helper
                # For robust intrinsics resizing, we ideally need the exact resolution.
                # Assuming 1080p for fallback or try to get it from frame.
                video_images_chunk = read_video_cv2(video_path, s, e)
                h_ori, w_ori = video_images_chunk[0].shape[:2]

            # 3. Geometry Pre-processing
            # ------------------------------------------------------------------
            # Resize Intrinsics to match the network input (usually 518x294 for Any4D)
            # Standard load_images in utils usually expects 518x294 target
            aspect_ratio = w_ori / h_ori
            target_width, target_height = find_closest_aspect_ratio(
            aspect_ratio, 518
            )

            intrinsics = resize_intrinsics(
                intrinsics_ori, 
                old_size=(h_ori, w_ori), 
                new_size=(target_height, target_width) # (H, W) matches Any4D default
            )
            
            # 4. Final Data Loading
            # ------------------------------------------------------------------
            # Using the standard load_images which takes (imgs, extr, intr)
            views = load_images(
                video_images_chunk,
                extrinsics_chunk_data,
                intrinsics,
                norm_type=self.high_level_config["data_norm_type"]
            )
            
            return views, task
            
        except Exception as e:
            print(f"[ERROR] Task failed: {task['zarr_episode_name']} | {e}")
            return None, None