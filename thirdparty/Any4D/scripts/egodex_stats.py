import argparse
import json
import os
import hydra

import h5py
import numpy as np
import torch
import torchvision.transforms as tvf
import decord
from tqdm import tqdm
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
from PIL.ImageOps import exif_transpose
from PIL import Image

from any4d.utils.inference import loss_of_one_batch_multi_view, preprocess_input_views_for_inference
from any4d.utils.misc import seed_everything
from any4d.models import init_model
from any4d.utils.moge_inference import load_moge_model
from any4d.utils.moge_inference import run_moge_inference
from any4d.utils.cropping import crop_resize_if_necessary

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

# Fixed resolution mappings with precomputed aspect ratios as keys
RESOLUTION_MAPPINGS = {
    518: {
        1.000: (518, 518),  # 1:1
        1.321: (518, 392),  # 4:3
        1.542: (518, 336),  # 3:2
        1.762: (518, 294),  # 16:9
        2.056: (518, 252),  # 2:1
        3.083: (518, 168),  # 3.2:1
        0.757: (392, 518),  # 3:4
        0.649: (336, 518),  # 2:3
        0.567: (294, 518),  # 9:16
        0.486: (252, 518),  # 1:2
    },
    512: {
        1.000: (512, 512),  # 1:1
        1.333: (512, 384),  # 4:3
        1.524: (512, 336),  # 3:2
        1.778: (512, 288),  # 16:9
        2.000: (512, 256),  # 2:1
        3.200: (512, 160),  # 3.2:1
        0.750: (384, 512),  # 3:4
        0.656: (336, 512),  # 2:3
        0.562: (288, 512),  # 9:16
        0.500: (256, 512),  # 1:2
    },
}

# Precomputed sorted aspect ratio keys for efficient lookup
ASPECT_RATIO_KEYS = {
    518: sorted(RESOLUTION_MAPPINGS[518].keys()),
    512: sorted(RESOLUTION_MAPPINGS[512].keys()),
}

def find_closest_aspect_ratio(aspect_ratio, resolution_set):
    """
    Find the closest aspect ratio from the resolution mappings using efficient key lookup.

    Args:
        aspect_ratio (float): Target aspect ratio
        resolution_set (int): Resolution set to use (518 or 512)

    Returns:
        tuple: (target_width, target_height) from the resolution mapping
    """
    aspect_keys = ASPECT_RATIO_KEYS[resolution_set]

    # Find the closest aspect ratio key using binary search approach
    closest_key = min(aspect_keys, key=lambda x: abs(x - aspect_ratio))

    return RESOLUTION_MAPPINGS[resolution_set][closest_key]

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

def resize_intrinsics(K, old_size, new_size):
    """
    调整相机内参以适应新的图像尺寸。
    
    Args:
        K (np.array): 原始内参矩阵 3x3
        old_size (tuple): (Height, Width) 原始尺寸
        new_size (tuple): (Height, Width) 新尺寸
        
    Returns:
        np.array: 新的内参矩阵
    """
    old_h, old_w = old_size
    new_h, new_w = new_size
    
    # 1. 计算宽和高的缩放比例
    scale_x = new_w / old_w
    scale_y = new_h / old_h
    
    # 2. 复制一份 K，防止修改原数据
    K_new = K.copy()
    
    # 3. 更新焦距 (fx, fy)
    K_new[0, 0] *= scale_x  # fx
    K_new[1, 1] *= scale_y  # fy
    
    # 4. 更新光心 (cx, cy)
    K_new[0, 2] *= scale_x  # cx
    K_new[1, 2] *= scale_y  # cy
    
    return K_new

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
            all_base_path.append(f"/share_data/guantianrui/datasets/EgoDex/{group}/{task}/{id}")
        episode_num = len(all_base_path)
        episode_num_per_gpu = episode_num // self.total_gpus
        start_idx = self.gpu_id * episode_num_per_gpu
        if self.gpu_id == self.total_gpus - 1:
            end_idx = episode_num
        else:
            end_idx = start_idx + episode_num_per_gpu
        self.episode_list = all_base_path[start_idx:end_idx]

    def load_mp4(self, path):
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        frames = vr.get_batch(range(len(vr))).asnumpy()
        return frames

    def estimate_depth(self):
        if self.dataset_name == "taco":
            self.taco_loop()
        elif self.dataset_name == "egodex":
            self.egodex_loop()
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
    
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
                views = self.load_images(
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
        video_list = []
        for episode_name in tqdm(self.episode_list, desc="Estimating depth"):
            mp4_path = f"{episode_name}.mp4"
            hdf5_path = f"{episode_name}.hdf5"
            npz_path = f"{episode_name}.npz"
            video_images = list(self.load_mp4(mp4_path))  # (T, H, W, 3)
            with h5py.File(hdf5_path, 'r') as file:
                intrinsics_ori = file['camera']['intrinsic'][:]  # (3, 3)
                extrinsics = list(file['transforms']['camera'][:])  # (T, 4, 4)
            intrinsics = resize_intrinsics(
                intrinsics_ori, 
                old_size=(1080, 1920), 
                new_size=(294, 518)
            )
            assert video_images[0].shape[0] == 1080 and video_images[0].shape[1] == 1920
            minlen = min(len(video_images), len(extrinsics))
            maxlen = max(len(video_images), len(extrinsics))
            gap = maxlen - minlen
            video_list.append((episode_name, minlen, gap))
        # sort video_list by the number of frames and output to file
        video_list.sort(key=lambda x: x[1])
        with open("egodex_video_list_len.txt", "w") as f:
            for video_name in video_list:
                f.write(f"{video_name[0]} minlen: {video_name[1]} gap: {video_name[2]}\n")
        video_list.sort(key=lambda x: x[2])
        with open("egodex_video_list_gap.txt", "w") as f:
            for video_name in video_list:
                f.write(f"{video_name[0]} gap: {video_name[2]} minlen: {video_name[1]}\n")

    def load_images(self, 
        video_images, # (T, H, W, 3)
        extrinsics, # (T, 4, 4)
        intrinsics, # (3, 3)
        resize_mode="fixed_mapping",
        norm_type="dinov2",
        resolution_set=518):

        H, W, _ = video_images[0].shape
        aspect_ratio = W / H
        # Determine target size for all images based on resize mode
        if resize_mode == "fixed_mapping":
            # Resolution mappings are already compatible with their respective patch sizes
            # 518 mappings are divisible by 14, 512 mappings are divisible by 16
            target_width, target_height = find_closest_aspect_ratio(
                aspect_ratio, resolution_set
            )
            target_size = (target_width, target_height)

        # Get the image normalization function based on the norm_type
        if norm_type in IMAGE_NORMALIZATION_DICT.keys():
            img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
            ImgNorm = tvf.Compose(
                [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
            )
        else:
            raise ValueError(
                f"Unknown image normalization type: {norm_type}. Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
            )

        imgs = []
        for id, image in enumerate(video_images):
            # binary_mask = np.ones((H, W), dtype=np.float32)
            # transform = tvf.Compose([tvf.ToTensor()])
            # input_moge_img = transform(image).unsqueeze(0)  # (1, 3, H, W)
            # moge_output = run_moge_inference(moge_model, input_moge_img, device="cuda")
            # non_ambiguous_mask = moge_output["mask"].squeeze(0).cpu().numpy().astype(np.float32)  # (H, W)
            # additional_quantities = [non_ambiguous_mask, binary_mask]
            # # img, additional_quantities = crop_resize_if_necessary(img, resolution=size, additional_quantities=additional_quantities)
            # img = exif_transpose(Image.fromarray(image)).convert("RGB")
            # img, additional_quantities = crop_resize_if_necessary(img, resolution=target_size, additional_quantities=additional_quantities)
            # non_ambiguous_mask = torch.tensor(additional_quantities[0]).bool()
            # binary_mask = torch.tensor(additional_quantities[1]).bool()

            additional_quantities = None
            img = exif_transpose(Image.fromarray(image)).convert("RGB")
            # img = crop_resize_if_necessary(img, resolution=size)[0]
            img = crop_resize_if_necessary(img, resolution=target_size)[0]
            non_ambiguous_mask = torch.tensor(np.ones_like(img)).bool()  # Default mask, all pixels are valid
            binary_mask = torch.tensor(np.ones_like(img)).bool()
            imgs.append(
                dict(
                    img=ImgNorm(img)[None],
                    intrinsics=torch.from_numpy(intrinsics)[None].float(),
                    camera_poses=torch.from_numpy(np.linalg.inv(extrinsics[id]))[None].float(),
                    true_shape=np.int32([img.size[::-1]]),
                    idx=len(imgs),
                    instance=str(len(imgs)),
                    data_norm_type=[norm_type],
                    non_ambiguous_mask=non_ambiguous_mask,
                    binary_mask=binary_mask
                )
            )
        views = preprocess_input_views_for_inference(imgs)

        return views


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="taco")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--total_gpus", type=int, default=8)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(0)
    any4d_depth_estimator = Any4DDepthEstimator(args.dataset_name, args.gpu_id, args.total_gpus)
    any4d_depth_estimator.estimate_depth()