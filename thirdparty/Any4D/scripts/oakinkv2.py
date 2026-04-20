import tarfile
import io
import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import resize_intrinsics, load_images


class OakInkV2Dataset(Dataset):
    def __init__(self, episode_keys, episode_info_dict, chunk_len, high_level_config):
        self.chunk_len = chunk_len
        self.high_level_config = high_level_config
        self.episode_info_dict = episode_info_dict
        
        # 基础路径
        self.data_root = "/share_data/guantianrui/datasets/OakInk-v2/data"
        self.anno_root = "/share_data/guantianrui/datasets/OakInk-v2/anno_preview"
        self.target_cam = "104422070969"
        
        self.tasks = []
        
        # ------------------------------------------------------------------
        # Init 阶段：只检查 TAR 包和 PKL 文件是否存在，不检查内部图片
        # ------------------------------------------------------------------
        print(f"Scanning OakInk-v2 dataset ({len(episode_keys)} episodes)...")
        
        for key in tqdm(episode_keys, desc="Preparing Tasks"):
            info = self.episode_info_dict[key]
            source_episode = info["source_episode"]
            
            # 1. 检查输出是否存在 (断点续传)
            npz_out_path = f"/share_data/guantianrui/datasets/OakInk-v2/Any4D_Predictions/{key}.npz"
            if os.path.exists(npz_out_path):
                continue

            # 2. 检查 TAR 和 PKL 是否存在
            tar_path = os.path.join(self.data_root, f"{source_episode}.tar")
            pkl_path = os.path.join(self.anno_root, f"{source_episode}.pkl")
            
            assert os.path.exists(tar_path) and os.path.exists(pkl_path)
                
            try:
                # 3. 读取 Pickle 获取元数据
                with open(pkl_path, "rb") as f:
                    anno_data = pickle.load(f)
                
                # 4. 获取有效帧列表
                full_frame_ids = anno_data["frame_id_list"]
                start_t, end_t = info["original_timestep_range"]
                
                # 筛选 range 内的 valid_frames
                valid_frames = [fid for fid in full_frame_ids if start_t <= fid <= end_t]
                
                # 严格单调递增检查
                assert len(valid_frames) > 0
                assert all(x < y for x, y in zip(valid_frames, valid_frames[1:])), \
                        f"Frame IDs not strictly increasing in {key}"
                
                # 5. 准备 Chunk 数据 (只存 Meta 信息)
                cam_intr_dict = anno_data["cam_intr"]["egocentric"]
                cam_extr_dict = anno_data["cam_extr"]["egocentric"]
                
                frame_data_list = []
                for fid in valid_frames:
                    assert fid in cam_intr_dict and fid in cam_extr_dict
                    frame_data_list.append({
                        "frame_id": fid,
                        "intrinsics": cam_intr_dict[fid],
                        "extrinsics": cam_extr_dict[fid]
                    })
                
                # =========================================================
                # 【新增】强校验：确保整个 Episode 内参一致
                # =========================================================
                # 1. 堆叠所有内参 -> (T, 3, 3)
                all_intrinsics = np.stack([item["intrinsics"] for item in frame_data_list])
                
                # 2. 取第一帧作为基准 -> (1, 3, 3)
                ref_k = all_intrinsics[0:1] 
                
                # 3. 计算所有帧相对于基准的差异
                #利用广播机制：(T, 3, 3) - (1, 3, 3)
                max_diff = np.max(np.abs(all_intrinsics - ref_k))
                
                # 4. 断言
                # 1e-4 是一个安全的阈值。如果焦距 float 精度丢失，通常也在 1e-5 级别。
                # 如果是变焦导致的变化，diff 通常会大于 1.0甚至几十。
                assert max_diff < 1e-4, f"Consistency Check Failed: Intrinsics are changing in {key}! Max diff: {max_diff}"
                
                # =========================================================

                total_len = len(frame_data_list)
                num_chunks = (total_len + self.chunk_len - 1) // self.chunk_len
                
                for i in range(num_chunks):
                    if i == num_chunks - 1:
                        end_idx = total_len
                    else:
                        end_idx = (i + 1) * self.chunk_len
                    start_idx = max(0, end_idx - self.chunk_len)
                    
                    self.tasks.append({
                        "episode_key": key,
                        "tar_path": tar_path,
                        "npz_path": npz_out_path,
                        "chunk_data": frame_data_list[start_idx:end_idx],
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "total_len": total_len,
                        "chunk_seq_id": i,
                        "is_last_chunk": (i == num_chunks - 1)
                    })

            except Exception as e:
                print(f"Error initializing {key}: {e}")
                continue
                
        print(f"Total chunks to process: {len(self.tasks)}")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        chunk_data = task["chunk_data"]
        tar_path = task["tar_path"]
        
        # 1. 构建 Frame ID 查找表
        # frame_id (int) -> index_in_buffer (int)
        needed_fids = {info['frame_id']: i for i, info in enumerate(chunk_data)}
        
        images_buffer = [None] * len(chunk_data)
        found_count = 0
        total_needed = len(chunk_data)
        
        try:
            # 2. 线性扫描 TAR (鲁棒匹配模式)
            with tarfile.open(tar_path, "r") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    
                    # 快速过滤：如果不包含目标相机名，直接跳过
                    # 你的相机名是 104422070969
                    if self.target_cam not in member.name:
                        continue
                    
                    # 解析 Frame ID
                    # 无论路径是 ./xxx/001.png 还是 xxx/001.png
                    # basename 都会得到 001.png
                    try:
                        filename = os.path.basename(member.name)
                        fid_str = filename.split('.')[0] # "003198"
                        fid = int(fid_str)               # 3198
                    except ValueError:
                        continue # 文件名不是数字，跳过
                    
                    # 匹配
                    if fid in needed_fids:
                        idx_in_chunk = needed_fids[fid]
                        
                        f = tar.extractfile(member)
                        if f is not None:
                            img_bytes = f.read()
                            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                            images_buffer[idx_in_chunk] = np.array(pil_img)
                            found_count += 1
                        
                        if found_count == total_needed:
                            break
                            
        except Exception as e:
            print(f"[TAR ERROR] Failed to read {tar_path}: {e}")

        # 3. 【严格校验】检查是否有缺帧
        missing_info = []
        for i in range(len(images_buffer)):
            if images_buffer[i] is None:
                missing_fid = chunk_data[i]['frame_id']
                missing_info.append(missing_fid)
                # 填充黑图防止 Crash，但已经记录了错误
                images_buffer[i] = np.zeros((480, 848, 3), dtype=np.uint8)
        
        if len(missing_info) > 0:
            # 打印醒目的报警信息
            print("\n" + "="*60)
            print(f"!! CRITICAL DATASET ERROR !!")
            print(f"Episode: {task['episode_key']}")
            print(f"TAR Path: {tar_path}")
            print(f"Target Camera: {self.target_cam}")
            print(f"Missing Frame IDs ({len(missing_info)}): {missing_info}")
            print("="*60 + "\n", flush=True)

        # -----------------------------------------------------------
        # 后续处理 (Resize, Normalize)
        # -----------------------------------------------------------
        video_images_chunk = images_buffer
        extrinsics_chunk = [item["extrinsics"] for item in chunk_data]
        
        # 内参取中间帧
        mid_idx = len(chunk_data) // 2
        video_images_chunk = [video_images_chunk[mid_idx]] + video_images_chunk
        extrinsics_chunk = [extrinsics_chunk[mid_idx]] + extrinsics_chunk
        intrinsics_ori = chunk_data[mid_idx]["intrinsics"]
        
        intrinsics = resize_intrinsics(
            intrinsics_ori, 
            old_size=(480, 848), # Height, Width
            new_size=(294, 518)
        )
        
        views = load_images(
            video_images_chunk,
            extrinsics_chunk,
            intrinsics,
            norm_type=self.high_level_config["data_norm_type"]
        )
        
        return views, task