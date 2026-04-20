import decord
import h5py
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import tqdm
from utils import resize_intrinsics, load_images

class EgoDexDataset(Dataset):
    def __init__(self, episode_list, chunk_len, high_level_config):
        """
        episode_list: 当前 GPU 分配到的视频列表
        """
        self.chunk_len = chunk_len
        self.high_level_config = high_level_config
        
        # --- 核心：预计算所有任务 ---
        self.tasks = [] 
        
        print("Pre-scanning dataset for chunks...")
        for episode_name in tqdm(episode_list, desc="Scanning"):
            # 1. 构建路径
            mp4_path = f"{episode_name}.mp4"
            hdf5_path = f"{episode_name}.hdf5"
            npz_path = f"{episode_name}.npz"

            try:
                # 2. 快速获取长度 (Decord 只读 Header，很快)
                vr = decord.VideoReader(mp4_path, ctx=decord.cpu(0))
                video_len = len(vr)
                
                # 3. 切分 Chunk 并存入任务列表
                # 如果 video_len < self.chunk_len，range 会生成 [0]，正好一个 chunk
                # chunk_idx 用来标记这是该视频的第几个片段，方便后续排序
                num_chunks = (video_len + self.chunk_len - 1) // self.chunk_len
                
                for i in range(num_chunks):
                    # 计算这个 chunk 的起止
                    if i == num_chunks - 1:
                        end_idx = video_len
                    else:
                        end_idx = (i + 1) * self.chunk_len
                    start_idx = max(0, end_idx - self.chunk_len)
                    
                    self.tasks.append({
                        "episode_name": episode_name,
                        "mp4_path": mp4_path,
                        "hdf5_path": hdf5_path,
                        "npz_path": npz_path,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "total_len": video_len,
                        "chunk_seq_id": i,      # 这是第几个 chunk
                        "is_last_chunk": (i == num_chunks - 1) # 是否是最后一个
                    })
                    
            except Exception as e:
                print(f"Error initializing {episode_name}: {e}")
                continue
                
        print(f"Total chunks to process: {len(self.tasks)}")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        
        # 1. 读取视频 (Decord)
        vr = decord.VideoReader(task["mp4_path"], ctx=decord.cpu(0))
        
        start_idx = task["start_idx"]
        end_idx = task["end_idx"]
        img_idx = (start_idx + end_idx) // 2
        
        indices = [img_idx] + list(range(start_idx, end_idx))
        video_images_chunk = vr.get_batch(indices).asnumpy()
        
        # 2. 读取 Extrinsics/Intrinsics
        with h5py.File(task["hdf5_path"], 'r') as file:
            intrinsics_ori = file['camera']['intrinsic'][:]
            # 局部读取 Extrinsics
            # HDF5 支持切片读取，非常快
            extrinsics_slice = file['transforms']['camera'][start_idx:end_idx]
            extrinsic_mid = file['transforms']['camera'][img_idx]
            
        # 组合 Extrinsics
        extrinsics_chunk = [extrinsic_mid] + list(extrinsics_slice)
        
        # Resize Intrinsics
        intrinsics = resize_intrinsics(
            intrinsics_ori, 
            old_size=(1080, 1920), 
            new_size=(294, 518)
        )
        
        # 3. 预处理 (CPU 密集型，由 worker 承担)
        views = load_images(
            video_images_chunk,
            extrinsics_chunk,
            intrinsics,
            norm_type=self.high_level_config["data_norm_type"]
        )
        
        return views, task