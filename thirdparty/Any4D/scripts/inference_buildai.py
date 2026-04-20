import argparse
import json
import os
import hydra
import numpy as np
import torch
import torch.multiprocessing
import decord
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
import imageio
import pickle
import tarfile
import io
from PIL import Image

from any4d.utils.inference import loss_of_one_batch_multi_view
from any4d.utils.misc import seed_everything
from any4d.models import init_model
from any4d.utils.geometry import quaternion_to_rotation_matrix, recover_pinhole_intrinsics_from_ray_directions

from egodex import EgoDexDataset
from oakinkv2 import OakInkV2Dataset
from vitra import VitraDataset
from holoassist_new import HoloAssistNewDataset
from buildai import BuildAIDataset
from utils import resize_intrinsics, load_images, compute_alignment_transform

torch.multiprocessing.set_sharing_strategy('file_system')

def init_hydra_config(config_path, overrides=None):
    "Initialize Hydra config"
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path).split(".")[0]
    relative_path = os.path.relpath(config_dir, os.path.dirname(__file__))
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path=relative_path)
    if overrides is not None:
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    else:
        cfg = hydra.compose(config_name=config_name)

    return cfg

def init_inference_model(config, ckpt_path, device):
    "Initialize the model for inference"
    # Load the model
    if isinstance(config, dict):
        config_path = config["path"]
        overrrides = config["config_overrides"]
        model_args = init_hydra_config(config_path, overrides=overrrides)
        model = init_model(model_args.model.model_str, model_args.model.model_config)
    else:
        config_path = config
        model_args = init_hydra_config(config_path)
        model = init_model(model_args.model_str, model_args.model_config)
    model.to(device)
    if ckpt_path is not None:
        print("Loading model from: ", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(model.load_state_dict(ckpt["model"], strict=False))
        model.to(device)
    # Set the model to eval mode
    model.eval()

    return model

@torch.no_grad()
def sample_inference(model, views, device, use_amp):
    # Run inference
    result = loss_of_one_batch_multi_view(
        views,
        model,
        None,
        device,
        use_amp=use_amp,
    )

    return result


class Any4DDepthEstimator:
    def __init__(self, dataset_name, gpu_id, total_gpus):
        self.dataset_name = dataset_name
        self.gpu_id = gpu_id
        self.total_gpus = total_gpus
        self.machine = "local"
        self.config_dir = "configs"
        self.checkpoint_path = "checkpoints/any4d_4v_combined.pth"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)
        self.high_level_config = {
            "path": f"configs/train.yaml",
            "config_overrides": [
                f"machine=local",
                "model=any4d",
                "model.encoder.uses_torch_hub=false",
                "model/task=mvs",
            ],
            "checkpoint_path": self.checkpoint_path,
            "trained_with_amp": True,
            "data_norm_type": "dinov2",
        }
        if dataset_name == "taco":
            self.prepare_taco_dataset()
        elif dataset_name == "egodex":
            self.prepare_egodex_dataset()
        elif dataset_name == "oakinkv2":
            self.prepare_oakinkv2_dataset()
        elif dataset_name == "vitra":
            self.prepare_vitra_dataset()
        elif dataset_name == "holoassist":
            self.prepare_holoassist_dataset()
        elif dataset_name == "buildai":
            self.prepare_buildai_dataset()
        self.model = init_inference_model(self.high_level_config, self.checkpoint_path, self.device)
        # self.moge_model = load_moge_model(device="cuda")

    def prepare_taco_dataset(self):
        episode_list_str = json.load(open("/share_data/guantianrui/datasets/taco/episode_names.json"))
        episode_list = [name.replace(")_", ")/") for name in episode_list_str]
        episode_num = len(episode_list)
        episode_num_per_gpu = episode_num // self.total_gpus
        start_idx = self.gpu_id * episode_num_per_gpu
        if self.gpu_id == self.total_gpus - 1:
            end_idx = episode_num
        else:
            end_idx = start_idx + episode_num_per_gpu
        self.episode_list = episode_list[start_idx:end_idx]
        self.video_base_path = "/share_data/guantianrui/datasets/taco/Egocentric_RGB_Videos"
        self.extri_intri_base_path = "/share_data/guantianrui/datasets/taco/Egocentric_Camera_Parameters"
        self.output_base_path = "/share_data/guantianrui/datasets/taco/Any4D_Predictions"

    def prepare_egodex_dataset(self):
        episode_list_str = json.load(open("/share_data/guantianrui/datasets/EgoDex/episode_names.json"))
        all_base_path = []
        for episode_name in episode_list_str:
            group, remainder = episode_name.split("_", 1)
            task, id = remainder.rsplit("_", 1)
            base_path = f"/share_data/guantianrui/datasets/EgoDex/{group}/{task}/{id}"
            if os.path.exists(f"{base_path}.npz"):
                continue
            all_base_path.append(base_path)
        print(f"Total episodes for EgoDex: {len(all_base_path)}")
        episode_num = len(all_base_path)
        episode_num_per_gpu = episode_num // self.total_gpus
        start_idx = self.gpu_id * episode_num_per_gpu
        if self.gpu_id == self.total_gpus - 1:
            end_idx = episode_num
        else:
            end_idx = start_idx + episode_num_per_gpu
        self.episode_list = all_base_path[start_idx:end_idx]

    def prepare_oakinkv2_dataset(self):
        # 路径指向 json 文件
        json_path = "/share_data/guantianrui/datasets/OakInk-v2/zarr_episodes_statistics.json"
        self.episode_info_dict = json.load(open(json_path)) 
        all_episode_keys = list(self.episode_info_dict.keys())
        
        # 分配 GPU 任务
        episode_num = len(all_episode_keys)
        episode_num_per_gpu = episode_num // self.total_gpus
        start_idx = self.gpu_id * episode_num_per_gpu
        if self.gpu_id == self.total_gpus - 1:
            end_idx = episode_num
        else:
            end_idx = start_idx + episode_num_per_gpu
        self.episode_keys = all_episode_keys[start_idx:end_idx]
        
        # 输出路径 base
        self.output_base_path = "/share_data/guantianrui/datasets/OakInk-v2/Any4D_Predictions"
        os.makedirs(self.output_base_path, exist_ok=True)

    def prepare_vitra_dataset(self):
        self.video_root = "/share_data/guantianrui/datasets/VITRA-1M/video_root"
        self.output_root = "/share_data/guantianrui/datasets/VITRA-1M/Any4D_Predictions"
        os.makedirs(self.output_root, exist_ok=True)

        # --- 优化重点 1: 一次性读取已存在的文件名 ---
        print(f"Scanning existing output files in {self.output_root}...")
        # 使用 set 存储文件名，查询复杂度为 O(1)
        # 只在 GPU 0 上打印进度，或者所有进程只查一次
        existing_files = set(os.listdir(self.output_root))
        print(f"Found {len(existing_files)} existing predictions.")

        # 1. 读取大 JSON
        json_path_test = "/share_data/guantianrui/datasets/VITRA-1M/zarr_episodes_statistics_testset.json"
        print(f"Loading VITRA metadata from {json_path_test} ...")
        
        if not os.path.exists(json_path_test):
            raise FileNotFoundError(f"Metadata file not found: {json_path_test}")
        
        info_test = json.load(open(json_path_test))
        all_video_stats = info_test['video_statistics']
        
        # 2. 展平所有 Episode 并解析路径
        print("Flattening episodes and resolving paths (Strict Mode)...")
        flat_episode_list = []
        
        # 用于统计缺失情况
        missing_videos = set()
        
        for video_stat in tqdm(all_video_stats, desc="Flattening"):
            video_name_key = video_stat['video_name']
            episodes = video_stat['episodes']
            
            # --- 解析视频路径 ---
            filename = None
            
            # 1. Ego4D
            if video_name_key.startswith("Ego4D_"):
                # "Ego4D_uuid" -> "uuid.mp4"
                vid_id = video_name_key[6:]
                filename = f"{vid_id}.mp4"
                
            # 2. Something-Something V2
            elif video_name_key.startswith("somethingsomethingv2_"):
                # "somethingsomethingv2_id" -> "id.webm"
                vid_id = video_name_key[21:]
                filename = f"{vid_id}.webm"
                
            # 3. Epic-Kitchen (修正前缀为 epic_)
            elif video_name_key.startswith("epic_kitchens_"):
                # "epic_id" -> "id.MP4" (保留 .MP4 后缀)
                vid_id = video_name_key[14:]
                filename = f"{vid_id}.MP4"
            
            else:
                print(f"[WARN] Unknown video prefix: {video_name_key}")
                continue

            # 构建绝对路径
            video_path = os.path.join(self.video_root, filename)
            
            # --- 【严格校验】检查文件是否存在 ---
            if not os.path.exists(video_path):
                # 记录缺失，只打印一次 warning 避免刷屏 (如果有多个 episode 指向同一个缺失视频)
                if video_path not in missing_videos:
                    print(f"[WARN] Video file missing: {video_path} (Key: {video_name_key})")
                    missing_videos.add(video_path)
                continue

            # --- 添加该视频下的所有 Episode ---
            for ep in episodes:
                # 简单检查一下帧范围逻辑是否合法
                try:
                    s = ep['original_start_frame']
                    e = ep['original_end_frame']
                    if e <= s:
                        print(f"[WARN] Invalid frame range in JSON: {ep['zarr_episode_name']} ({s}-{e})")
                        continue

                    if f"{ep['zarr_episode_name']}.npz" in existing_files:
                        continue
                        
                    flat_episode_list.append({
                        "video_path": video_path,
                        "zarr_episode_name": ep['zarr_episode_name'],
                        "original_start_frame": s,
                        "original_end_frame": e
                    })
                except:
                    print(f"[WARN] Malformed episode entry: {ep}")
                    continue

        flat_episode_list_train = json.load(open("/share_data/yifan/projects/depth/Any4D/flat_episode_list.json"))

        for ep in flat_episode_list_train:
            if f"{ep['zarr_episode_name']}.npz" not in existing_files:
                flat_episode_list.append(ep)

        total_episodes = len(flat_episode_list)
        print(f"Total valid episodes found: {total_episodes}")
        # if len(missing_videos) > 0:
        #     print(f"Total missing videos: {len(missing_videos)}")
        
        # 3. 均匀分配给当前 GPU
        if total_episodes == 0:
            self.my_episode_list = []
            print(f"GPU {self.gpu_id} has no tasks.")
            return

        per_gpu = total_episodes // self.total_gpus
        start_idx = self.gpu_id * per_gpu
        
        if self.gpu_id == self.total_gpus - 1:
            end_idx = total_episodes
        else:
            end_idx = start_idx + per_gpu
            
        self.my_episode_list = flat_episode_list[start_idx:end_idx]
        print(f"GPU {self.gpu_id} assigned {len(self.my_episode_list)} episodes.")

    def prepare_holoassist_dataset(self):
        # 1. 基础路径
        self.video_root_base = "/share_data/guantianrui/datasets/HoloAssist/raw_data/video"
        self.pose_root_base = "/share_data/guantianrui/datasets/HoloAssist/output/holoassist_results"
        self.output_base_path = "/share_data/guantianrui/datasets/HoloAssist/Any4D_Predictions"
        
        print(f"Scanning HoloAssist tasks from {self.video_root_base} ...")
        
        if not os.path.exists(self.video_root_base):
             raise FileNotFoundError(f"Root not found: {self.video_root_base}")
        
        # 2. 扫描所有候选文件夹
        candidate_tasks = [
            d for d in os.listdir(self.video_root_base) 
            if os.path.isdir(os.path.join(self.video_root_base, d))
            and (d.startswith("R") or d.startswith("z"))
        ]
        
        # 3. 【新增】根据 hawor_results.pkl 进行过滤
        valid_tasks = []
        print("Filtering tasks based on hawor_results.pkl existence...")
        # 使用 set 避免重复 IO (虽然这里是一对一)
        # 也可以直接检查，速度通常很快
        
        for task_name in tqdm(candidate_tasks, desc="Checking PKLs"):
            pkl_path = os.path.join(self.pose_root_base, task_name, "hawor_results.pkl")
            if os.path.exists(pkl_path):
                valid_tasks.append(task_name)
        
        valid_tasks.sort() # 排序保证多卡一致性
        
        total_tasks = len(valid_tasks)
        print(f"Total VALID tasks found: {total_tasks} (dropped {len(candidate_tasks) - total_tasks})")
        
        os.makedirs(self.output_base_path, exist_ok=True)
        
        # 4. 分配 GPU
        if total_tasks == 0:
            self.my_task_list = []
            print(f"GPU {self.gpu_id} has no tasks.")
            return

        per_gpu = total_tasks // self.total_gpus
        start_idx = self.gpu_id * per_gpu
        if self.gpu_id == self.total_gpus - 1:
            end_idx = total_tasks
        else:
            end_idx = start_idx + per_gpu
            
        self.my_task_list = valid_tasks[start_idx:end_idx]
        print(f"GPU {self.gpu_id} assigned {len(self.my_task_list)} tasks.")

    def prepare_buildai_dataset(self):
        self.video_root = "/share_data/lvjianan/datasets/BuildAI-processed"
        self.pose_root = "/share_data/lvjianan/HaWoR/output/buildai_results"
        self.output_base_path = "/share_data/guantianrui/datasets/BuildAI/Any4D_Predictions" # 建议修改为你自己的输出路径
        os.makedirs(self.output_base_path, exist_ok=True)

        print("Scanning BuildAI directories...")
        all_episodes = []

        episode_list_path = "/share_data/guantianrui/datasets/BuildAI/buildai_episode_list_new.pkl"
        if not os.path.exists(episode_list_path):
            # 遍历 factory_{fid} / worker_{wid} / processed
            factories = [d for d in os.listdir(self.video_root) if d.startswith("factory")]
            for f_dir in tqdm(factories, desc="Scanning Factories"):
                f_path = os.path.join(self.video_root, f_dir)
                workers = [d for d in os.listdir(f_path) if d.startswith("worker")]
                for w_dir in workers:
                    proc_path = os.path.join(f_path, w_dir, "processed")
                    if not os.path.exists(proc_path):
                        print(f"[WARN] Missing processed dir: {proc_path}")
                        continue
                    
                    # 寻找 mp4
                    files = [f for f in os.listdir(proc_path) if f.endswith(".mp4")]
                    for f in files:
                        ep_name = f.replace(".mp4", "")
                        video_path = os.path.join(proc_path, f)
                        pose_path = os.path.join(self.pose_root, ep_name, "hawor_results.pkl")
                        
                        # 检查位姿文件是否存在
                        if os.path.exists(pose_path):
                            # 如果已经存在结果则跳过 (可选)
                            if os.path.exists(os.path.join(self.output_base_path, f"{ep_name}.npz")):
                                continue
                            all_episodes.append({
                                "ep_name": ep_name,
                                "video_path": video_path,
                                "pose_path": pose_path
                            })
                        else:
                            print(f"[WARN] Missing pose file for {ep_name}: {pose_path}")
            with open(episode_list_path, "wb") as f:
                pickle.dump(all_episodes, f)
        else:
            with open(episode_list_path, "rb") as f:
                all_episodes_old = pickle.load(f)
            for episode in all_episodes_old:
                if os.path.exists(os.path.join(self.output_base_path, f"{episode['ep_name']}.npz")):
                    continue
                all_episodes.append(episode)

        all_episodes.sort(key=lambda x: x['ep_name'])

        print(f"Total BuildAI episodes to process: {len(all_episodes)}")

        # GPU 任务分配
        total = len(all_episodes)
        per_gpu = total // self.total_gpus
        start = self.gpu_id * per_gpu
        end = total if self.gpu_id == self.total_gpus - 1 else start + per_gpu
        
        self.my_episode_list = all_episodes[start:end]
        print(f"GPU {self.gpu_id} assigned {len(self.my_episode_list)} episodes.")

    def load_mp4(self, path):
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        frames = vr.get_batch(range(len(vr))).asnumpy()
        return frames

    def estimate_depth(self):
        if self.dataset_name == "taco":
            self.taco_loop()
        elif self.dataset_name == "egodex":
            self.egodex_loop()
        elif self.dataset_name == "oakinkv2":
            self.oakinkv2_loop()
        elif self.dataset_name == "vitra":
            self.vitra_loop()
        elif self.dataset_name == "holoassist":
            self.holoassist_loop()
        elif self.dataset_name == "buildai":
            self.buildai_loop()
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
    def buildai_loop(self):
        def collate_fn(batch):
            if batch[0][0] is None: return None
            return batch[0]
        
        chunk_len = 180
        dataset = BuildAIDataset(
            episode_list=self.my_episode_list,
            output_root=self.output_base_path,
            chunk_len=chunk_len,
            high_level_config=self.high_level_config
        )
        
        if len(dataset) == 0:
            print(f"GPU {self.gpu_id} has no tasks.")
            return
            
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, 
                            collate_fn=collate_fn, pin_memory=True)
        
        depth_buffer = defaultdict(list)
        
        for batch in tqdm(loader, desc=f"GPU {self.gpu_id} BuildAI Progress"):
            if batch is None:
                print("[WARN] Skipping a batch due to loading issues.")
                continue
            views, task = batch
            
            # 推理
            pred_result = sample_inference(self.model, views, self.device, 
                                          use_amp=self.high_level_config["trained_with_amp"])
            
            # 解析索引与重叠
            s, e, chunk_id = task["start_idx"], task["end_idx"], task["chunk_seq_id"]
            name, total_len = task["ep_name"], task["total_len"]
            
            current_len = e - s
            start_save_i = 0
            if chunk_id > 0 and task["is_last_chunk"]:
                saved = chunk_id * chunk_len
                start_save_i = current_len - (total_len - saved)

            chunk_depths = []
            for i in range(start_save_i, current_len):
                d = pred_result[f'pred{i + 2}']['pts3d_cam'][0, :, :, 2].cpu().numpy()
                chunk_depths.append(np.round(d * 1000).astype(np.uint16))
            
            depth_buffer[name].append((chunk_id, chunk_depths))
            
            if task["is_last_chunk"]:
                chunks = depth_buffer.pop(name)
                chunks.sort(key=lambda x: x[0])
                full_depth = []
                for _, data in chunks: full_depth.extend(data)

                # Validation
                if len(full_depth) != total_len:
                    print(f"[WARN] Length mismatch saving {name}: Got {len(full_depth)}, Exp {total_len}")
                    # Still save what we have
                
                np.savez_compressed(task["npz_path"], depth=full_depth)
            
            del pred_result
            torch.cuda.empty_cache()
    
    def taco_loop(self):
        for episode_name in tqdm(self.episode_list, desc="Estimating depth"):
            video_path = os.path.join(self.video_base_path, episode_name, "color.mp4")
            extri_path = os.path.join(self.extri_intri_base_path, episode_name, "egocentric_frame_extrinsic.npy")
            intri_path = os.path.join(self.extri_intri_base_path, episode_name, "egocentric_intrinsic.txt")
            extrinsics = list(np.load(extri_path))  # (T, 4, 4)
            intrinsics_ori = np.loadtxt(intri_path)  # (3, 3)
            intrinsics = resize_intrinsics(
                intrinsics_ori, 
                old_size=(1080, 1920), 
                new_size=(294, 518)
            )
            video_images = list(self.load_mp4(video_path))  # (T, 1080, 1920, 3)
            assert video_images[0].shape[0] == 1080 and video_images[0].shape[1] == 1920
            minlen = min(len(video_images), len(extrinsics))
            chunk_len = 180
            if minlen % chunk_len == 0:
                chunk_num = minlen // chunk_len
            else:
                chunk_num = minlen // chunk_len + 1
            depth = []
            for i in range(chunk_num):
                if i == chunk_num - 1:
                    end_idx = minlen
                else:
                    end_idx = (i + 1) * chunk_len
                start_idx = max(0, end_idx - chunk_len)
                img_idx = (start_idx + end_idx) // 2
                video_images_chunk = [video_images[img_idx]] + video_images[start_idx:end_idx]
                extrinsics_chunk = [extrinsics[img_idx]] + extrinsics[start_idx:end_idx]
                views = load_images(
                    video_images_chunk, 
                    extrinsics_chunk, 
                    intrinsics,
                    norm_type=self.high_level_config["data_norm_type"],)
            
                pred_result = sample_inference(
                    self.model,
                    views,
                    self.device,
                    use_amp=self.high_level_config["trained_with_amp"],
                )

                if i == chunk_num - 1:
                    for id in range(i * chunk_len, minlen):
                        depth.append(np.round(pred_result[f'pred{id - start_idx + 2}']['pts3d_cam'][0, :, :, 2].cpu().numpy() * 1000).astype(np.uint16))
                else:
                    for id in range(chunk_len):
                        depth.append(np.round(pred_result[f'pred{id + 2}']['pts3d_cam'][0, :, :, 2].cpu().numpy() * 1000).astype(np.uint16))
                del pred_result
                torch.cuda.empty_cache()
            assert len(depth) == minlen
            output_dir = os.path.join(self.output_base_path, episode_name)
            os.makedirs(output_dir, exist_ok=True)
            np.savez_compressed(os.path.join(output_dir, "depth.npz"), depth=depth)

    def egodex_loop(self):
        # 只需要一个 collate_fn
        def collate_fn(batch):
            # batch[0][0] 是 views, batch[0][1] 是 task info
            return batch[0]
        
        chunk_len = 180
        
        # 1. 初始化全局 Dataset
        # 这里把所有视频都扫了一遍，生成了所有要做的小任务
        dataset = EgoDexDataset(
            episode_list=self.episode_list,
            chunk_len=chunk_len,
            high_level_config=self.high_level_config
        )
        
        if len(dataset) == 0:
            print("No tasks found (all done or empty).")
            return

        # 2. 初始化持久化的 DataLoader
        # 这里的 num_workers 进程会一直存活，直到所有视频跑完
        loader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, # 建议 False，按顺序处理视频，减小 Buffer 压力
            num_workers=8, # 建议 8-12
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=2
        )

        # 3. 结果缓存 Buffer
        # 结构: { "episode_name": [ (chunk_id, data_array), ... ] }
        results_buffer = defaultdict(list)

        # 4. 跑起来！
        for views, task in tqdm(loader, desc="Total Progress"):
            
            # 推理
            pred_result = sample_inference(
                self.model,
                views,
                self.device,
                use_amp=self.high_level_config["trained_with_amp"],
            )

            # 解析 Task 信息
            s_idx = task["start_idx"]
            e_idx = task["end_idx"]
            chunk_id = task["chunk_seq_id"]
            name = task["episode_name"]
            total_len = task["total_len"]
            
            # 提取深度 (转 numpy)
            chunk_depths = []
            
            # 逻辑：如果是最后一段，要把重叠/剩余部分处理好
            # 注意：views 的长度 = 1 (middle) + chunk_frames
            # pred_result 的 key 是 'pred2', 'pred3'... 对应 views 里的 index
            
            # 计算当前 chunk 实际包含的帧数 (例如 10)
            current_len = e_idx - s_idx
            
            # 【核心修正】计算从哪里开始保存
            # chunk_id * chunk_len 是这个 chunk 理论上的起始点 (绝对坐标)
            # s_idx 是这个 chunk 实际加载的起始点 (绝对坐标)
            # 它们的差值，就是重叠部分的长度，也就是我们需要跳过的部分
            start_save_i = (chunk_id * chunk_len) - s_idx
            
            # 此时：
            # 如果是普通 chunk，start_save_i 会是 0
            # 如果是最后一段 (有重叠)，start_save_i 会是重叠的帧数 (比如 8)
            
            for i in range(start_save_i, current_len):
                # pred key 的逻辑保持不变：i 是相对于当前 views 的索引
                # pred{i+2} 对应 views[i+1]，即第 i 帧 (views[0]是Ref)
                d = pred_result[f'pred{i + 2}']['pts3d_cam'][0, :, :, 2].cpu().numpy()
                chunk_depths.append(np.round(d * 1000).astype(np.uint16))

            # 存入 Buffer
            results_buffer[name].append((chunk_id, chunk_depths))

            del pred_result
            torch.cuda.empty_cache()
            
            # 5. 检查是否该视频已完成
            if task["is_last_chunk"]:
                # 取出该视频的所有片段
                chunks = results_buffer.pop(name)
                
                # 按 chunk_id 排序 (虽然 loader shuffle=False 通常是有序的，但多进程下保险起见)
                chunks.sort(key=lambda x: x[0])
                
                # 拼接所有列表
                full_depth = []
                for _, data in chunks:
                    full_depth.extend(data)
                
                # 校验长度
                if len(full_depth) == total_len:
                    # os.makedirs(output_dir, exist_ok=True)
                    np.savez_compressed(task["npz_path"], depth=full_depth)
                else:
                    print(f"Error: Length mismatch for {name}. Exp:{total_len}, Got:{len(full_depth)}")

    def save_debug_visualization(self, episode_key):
        """
        保存指定 Episode 的原始数据用于 Debug 可视化
        """
        # 1. 准备路径
        debug_dir = os.path.join(self.output_base_path, "debug_vis", f"gpu_{self.gpu_id}", episode_key)
        
        # 检查标记文件是否存在，如果存在则跳过 (避免重复保存)
        if os.path.exists(os.path.join(debug_dir, "intrinsics.txt")):
            return

        print(f"Saving debug visualization for {episode_key} ...")
        os.makedirs(debug_dir, exist_ok=True)
        
        info = self.episode_info_dict[episode_key]
        source_episode = info["source_episode"]
        tar_path = os.path.join("/share_data/guantianrui/datasets/OakInk-v2/data", f"{source_episode}.tar")
        pkl_path = os.path.join("/share_data/guantianrui/datasets/OakInk-v2/anno_preview", f"{source_episode}.pkl")
        target_cam = "104422070969"

        if not os.path.exists(tar_path) or not os.path.exists(pkl_path):
            print(f"[Debug] Missing files for {episode_key}")
            return

        # 2. 读取元数据
        try:
            with open(pkl_path, "rb") as f:
                anno_data = pickle.load(f)
        except Exception as e:
            print(f"[Debug] Failed to load pickle: {e}")
            return
            
        full_frame_ids = anno_data["frame_id_list"]
        start_t, end_t = info["original_timestep_range"]
        valid_frames = [fid for fid in full_frame_ids if start_t <= fid <= end_t]

        if not valid_frames:
            print("[Debug] No valid frames found.")
            return

        cam_intr_dict = anno_data["cam_intr"]["egocentric"]
        cam_extr_dict = anno_data["cam_extr"]["egocentric"]
        
        # 3. 收集需要的信息
        # key: frame_id, value: list_index
        needed_fids_map = {} 
        frame_data_list = []
        
        for idx, fid in enumerate(valid_frames):
            if fid not in cam_intr_dict or fid not in cam_extr_dict:
                continue
            
            needed_fids_map[fid] = len(frame_data_list)
            
            frame_data_list.append({
                "fid": fid,
                "intrinsics": cam_intr_dict[fid],
                "extrinsics": cam_extr_dict[fid]
            })

        # 4. 读取 TAR 图片 (使用鲁棒逻辑)
        images_buffer = [None] * len(frame_data_list)
        found_count = 0
        total_needed = len(frame_data_list)
        
        try:
            with tarfile.open(tar_path, "r") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    
                    # 快速筛选相机
                    if target_cam not in member.name:
                        continue
                        
                    # 解析 Frame ID
                    try:
                        filename = os.path.basename(member.name)
                        fid_str = filename.split('.')[0]
                        fid = int(fid_str)
                    except ValueError:
                        continue
                    
                    # 匹配
                    if fid in needed_fids_map:
                        idx = needed_fids_map[fid]
                        
                        f = tar.extractfile(member)
                        if f:
                            img_bytes = f.read()
                            # 保持原始分辨率 (848x480)
                            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                            images_buffer[idx] = np.array(pil_img)
                            found_count += 1
                        
                        if found_count == total_needed:
                            break
        except Exception as e:
            print(f"[Debug] Failed reading tar: {e}")
            return

        # 过滤掉没找到的帧
        valid_indices = [i for i, img in enumerate(images_buffer) if img is not None]
        assert len(valid_indices) == total_needed
        final_images = [images_buffer[i] for i in valid_indices]
        final_extrinsics = [frame_data_list[i]["extrinsics"] for i in valid_indices]
        
        if not final_images:
            print("[Debug] No images extracted.")
            return

        # 内参只取中间一帧
        mid_idx = len(final_images) // 2
        mid_k = frame_data_list[valid_indices[mid_idx]]["intrinsics"]

        # 5. 保存文件
        
        # A. 保存 RGB 视频
        vid_path = os.path.join(debug_dir, "rgb.mp4")
        try:
            imageio.mimsave(vid_path, final_images, fps=30, quality=8)
            print(f"[Debug] Saved RGB video to {vid_path}")
        except Exception as e:
            print(f"[Debug] Failed to save video: {e}")
        
        # B. 保存 Extrinsics (T, 4, 4)
        ext_path = os.path.join(debug_dir, "extrinsics.npy")
        np.save(ext_path, np.array(final_extrinsics))
        print(f"[Debug] Saved Extrinsics to {ext_path}")
        
        # C. 保存 Intrinsics (3, 3)
        int_path = os.path.join(debug_dir, "intrinsics.txt")
        np.savetxt(int_path, mid_k)
        print(f"[Debug] Saved Intrinsics to {int_path}")

    def oakinkv2_loop(self):
        def collate_fn(batch):
            return batch[0]
        
        chunk_len = 180
        
        # 1. 初始化 Dataset
        dataset = OakInkV2Dataset(
            episode_keys=self.episode_keys,
            episode_info_dict=self.episode_info_dict,
            chunk_len=chunk_len,
            high_level_config=self.high_level_config
        )
        
        if len(dataset) == 0:
            print("No tasks found.")
            return

        # =======================================================
        # 保存第一条数据的 Debug 信息
        # =======================================================
        if len(dataset.tasks) > 0:
            # 获取该 GPU 分配到的第一个任务对应的 episode_key
            first_task = dataset.tasks[0]
            first_episode_key = first_task["episode_key"]
            
            # 调用保存函数 (函数内部已经做了 exist 检查)
            self.save_debug_visualization(first_episode_key)
        # =======================================================

        # 2. DataLoader (以下保持不变)
        loader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=8, 
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=2
        )

        # 3. Buffer
        results_buffer = defaultdict(list)
        
        # 4. 循环推理
        for views, task in tqdm(loader, desc="OakInk-v2 Progress"):
            
            # 推理
            pred_result = sample_inference(
                self.model,
                views,
                self.device,
                use_amp=self.high_level_config["trained_with_amp"],
            )
            
            # 解析
            s_idx = task["start_idx"]
            e_idx = task["end_idx"]
            chunk_id = task["chunk_seq_id"]
            key = task["episode_key"]
            total_len = task["total_len"]
            
            chunk_depths = []
            current_len = e_idx - s_idx
            
            # 处理重叠
            start_save_i = (chunk_id * chunk_len) - s_idx
            
            for i in range(start_save_i, current_len):
                d = pred_result[f'pred{i + 2}']['pts3d_cam'][0, :, :, 2].cpu().numpy()
                chunk_depths.append(np.round(d * 1000).astype(np.uint16))
                
            results_buffer[key].append((chunk_id, chunk_depths))
            
            del pred_result
            torch.cuda.empty_cache()

            # 5. 保存
            if task["is_last_chunk"]:
                chunks = results_buffer.pop(key)
                chunks.sort(key=lambda x: x[0])
                
                full_depth = []
                for _, data in chunks:
                    full_depth.extend(data)
                
                if len(full_depth) == total_len:
                    # task["npz_path"] 在 Dataset 里已经生成好了
                    np.savez_compressed(task["npz_path"], depth=full_depth)
                else:
                    print(f"Error: Length mismatch for {key}. Exp:{total_len}, Got:{len(full_depth)}")

    def save_vitra_debug_visualization(self, episode_info):
        """
        保存 VITRA 数据集的 Debug 视频片段 (Raw RGB)
        """
        zarr_ep_name = episode_info['zarr_episode_name']
        video_path = episode_info['video_path']
        start_frame = episode_info['original_start_frame']
        end_frame = episode_info['original_end_frame']
        
        # 1. 准备输出路径
        debug_dir = os.path.join(self.output_root, "debug_vis", f"gpu_{self.gpu_id}", zarr_ep_name)
        vid_save_path = os.path.join(debug_dir, "rgb.mp4")
        if os.path.exists(vid_save_path):
            return

        print(f"[Debug] Saving visualization for {zarr_ep_name} ...")
        os.makedirs(debug_dir, exist_ok=True)
        
        try:
            if not os.path.exists(video_path):
                print(f"[Debug] Video file not found: {video_path}")
                return

            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            
            # 【修正】闭区间逻辑 [start, end]
            # 这里的 end_frame 是包含在内的，所以 python range 需要 +1
            # 但首先要检查边界
            
            # 如果 end_frame (0-based) 超过了视频最大索引 (len-1)
            # len=100, valid indices 0..99.
            # if end_frame(99) >= 100 -> False. 
            # if end_frame(100) >= 100 -> True.
            if end_frame >= len(vr):
                print(f"[Debug] Frame range out of bounds. Clamping end_frame.")
                end_frame = len(vr) - 1 # 闭区间的最后一个有效索引
            
            if start_frame > end_frame:
                print(f"[Debug] Invalid frame range: {start_frame}-{end_frame}")
                return

            # range 是左闭右开，所以要 end_frame + 1
            indices = list(range(start_frame, end_frame + 1))
            
            # 读取
            frames = vr.get_batch(indices).asnumpy()
            
            # 保存
            fps = vr.get_avg_fps()
            if fps <= 0 or np.isnan(fps): fps = 30
            
            imageio.mimsave(vid_save_path, frames, fps=fps, quality=8)
            print(f"[Debug] Saved RGB video to {vid_save_path}")

        except Exception as e:
            print(f"[Debug] Failed to save debug video for {zarr_ep_name}: {e}")

    def vitra_loop(self):
        # 修正 collate_fn 以处理 __getitem__ 返回 None 的情况
        def collate_fn(batch):
            if batch[0][0] is None:
                return None
            return batch[0]
        
        chunk_len = 180

        # =======================================================
        # 【新增】保存第一条数据的 Debug 视频
        # =======================================================
        # self.my_episode_list 已经被 prepare_vitra_dataset 分配好了
        if len(self.my_episode_list) > 0:
            # 取当前 GPU 分配到的第一个 episode
            first_episode_info = self.my_episode_list[0]
            self.save_vitra_debug_visualization(first_episode_info)
        # =======================================================

        # 1. 初始化 Dataset
        dataset = VitraDataset(
            episode_list=self.my_episode_list,
            output_root=self.output_root,
            chunk_len=chunk_len,
            high_level_config=self.high_level_config
        )
        
        if len(dataset) == 0:
            print("No VITRA tasks found.")
            return

        # 2. DataLoader
        loader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=8, # Decord 读取 mp4/webm 比较耗 CPU
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=2
        )

        depth_buffer = defaultdict(list)
        extrinsics_buffer = defaultdict(list)
        intrinsics_buffer = defaultdict(list)
        
        # 3. 推理循环
        for batch in tqdm(loader, desc="VITRA Progress"):
            if batch is None:
                print("Batch is None, skipping...")
                continue
                
            views, task = batch
            
            # 推理
            pred_result = sample_inference(
                self.model,
                views, 
                self.device,
                use_amp=self.high_level_config["trained_with_amp"],
            )
            
            s_idx = task["abs_start_idx"]
            e_idx = task["abs_end_idx"]
            chunk_id = task["chunk_seq_id"]
            ep_name = task["zarr_episode_name"]
            total_len = task["total_len"]
            
            chunk_depths = []
            chunk_extrinsics = []
            chunk_intrinsics = []
            
            # 当前 chunk 里的帧数
            current_len = e_idx - s_idx
            
            # ================= 重叠处理逻辑 =================
            if chunk_id == 0:
                start_save_i = 0
            else:
                # 只有最后一个 chunk 可能会发生回溯重叠
                if task["is_last_chunk"]:
                    # 计算已经保存了多少帧
                    saved_frames_count = chunk_id * chunk_len
                    # 还需要保存多少帧
                    frames_to_save = total_len - saved_frames_count
                    # 从当前 chunk 的末尾往前推 frames_to_save 帧开始保存
                    start_save_i = current_len - frames_to_save
                else:
                    start_save_i = 0
            # ==========================================================

            for i in range(start_save_i, current_len):
                # 假设 pred2 对应输入序列的第1帧
                key_name = f'pred{i + 2}'
                d = pred_result[key_name]['pts3d_cam'][0, :, :, 2].cpu().numpy()
                chunk_depths.append(np.round(d * 1000).astype(np.uint16))
                cam_quats = pred_result[key_name]['cam_quats'][0].cpu()
                cam_trans = pred_result[key_name]['cam_trans'][0].cpu()
                cam_rot = quaternion_to_rotation_matrix(cam_quats)
                cam_pose = torch.eye(4)
                cam_pose[:3, :3] = cam_rot
                cam_pose[:3, 3] = cam_trans
                cam_pose = cam_pose.numpy()
                world2cam = np.linalg.inv(cam_pose)
                ray_directions = pred_result[key_name]['ray_directions'][0].cpu()
                cam_intrinsics = recover_pinhole_intrinsics_from_ray_directions(ray_directions)
                cam_intrinsics = cam_intrinsics.numpy()
                chunk_extrinsics.append(world2cam)
                chunk_intrinsics.append(cam_intrinsics)

            
            depth_buffer[ep_name].append((chunk_id, chunk_depths))
            extrinsics_buffer[ep_name].append((chunk_id, chunk_extrinsics))
            intrinsics_buffer[ep_name].append((chunk_id, chunk_intrinsics))
            
            del pred_result
            torch.cuda.empty_cache()
            
            # 保存
            if task["is_last_chunk"]:
                depth_chunks = depth_buffer.pop(ep_name)
                extrinsics_chunks = extrinsics_buffer.pop(ep_name)
                intrinsics_chunks = intrinsics_buffer.pop(ep_name)
                depth_chunks.sort(key=lambda x: x[0])
                extrinsics_chunks.sort(key=lambda x: x[0])
                intrinsics_chunks.sort(key=lambda x: x[0])
                
                full_depth = []
                for _, data in depth_chunks:
                    full_depth.extend(data)
                
                full_extrinsics = []
                for _, data in extrinsics_chunks:
                    full_extrinsics.extend(data)

                full_intrinsics = []
                for _, data in intrinsics_chunks:
                    full_intrinsics.extend(data)
                
                assert len(full_depth) == total_len
                assert len(full_extrinsics) == total_len
                assert len(full_intrinsics) == total_len
                np.savez_compressed(task["npz_path"], depth=full_depth, extrinsics=full_extrinsics, intrinsics=full_intrinsics)

    def holoassist_loop(self):
        def collate_fn(batch):
            if batch[0][0] is None: return None
            return batch[0]
        
        chunk_len = 180
            
        dataset = HoloAssistNewDataset(
            task_list=self.my_task_list,
            output_root=self.output_base_path,
            chunk_len=chunk_len,
            high_level_config=self.high_level_config
        )
        
        if len(dataset) == 0:
            return
            
        loader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=8, 
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=2
        )
        
        # Buffers
        depth_buffer = defaultdict(list)
        
        for batch in tqdm(loader, desc="HoloAssist Progress"):
            if batch is None: 
                print("Batch is None, skipping...")
                continue
            
            views, task = batch
            
            # Inference (uses GT Pose input)
            pred_result = sample_inference(
                self.model,
                views,
                self.device,
                use_amp=self.high_level_config["trained_with_amp"],
            )
            
            s_idx = task["start_idx"]
            e_idx = task["end_idx"]
            chunk_id = task["chunk_seq_id"]
            name = task["task_name"]
            total_len = task["total_len"]
            
            current_len = e_idx - s_idx
            
            chunk_depths = []

            # Overlap handling
            if chunk_id == 0:
                start_save_i = 0
            else:
                if task["is_last_chunk"]:
                    saved_frames_count = chunk_id * chunk_len
                    frames_to_save = total_len - saved_frames_count
                    start_save_i = current_len - frames_to_save
                else:
                    start_save_i = 0
            
            for i in range(start_save_i, current_len):
                # Depth from model
                d = pred_result[f'pred{i + 2}']['pts3d_cam'][0, :, :, 2].cpu().numpy()
                chunk_depths.append(np.round(d * 1000).astype(np.uint16))
            
            # Store
            depth_buffer[name].append((chunk_id, chunk_depths))
            
            del pred_result
            torch.cuda.empty_cache()
            
            # Merge and Save
            if task["is_last_chunk"]:
                # Pop chunks
                d_chunks = depth_buffer.pop(name)
                
                # Sort by ID
                d_chunks.sort(key=lambda x: x[0])
                
                # Flatten
                full_depth = []
                for _, data in d_chunks: full_depth.extend(data)

                # Validation
                if len(full_depth) != total_len:
                    print(f"[WARN] Length mismatch saving {name}: Got {len(full_depth)}, Exp {total_len}")
                    # Still save what we have
                
                np.savez_compressed(
                    task["npz_path"], 
                    depth=full_depth,
                )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="buildai")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--total_gpus", type=int, default=8)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(0)
    any4d_depth_estimator = Any4DDepthEstimator(args.dataset_name, args.gpu_id, args.total_gpus)
    any4d_depth_estimator.estimate_depth()