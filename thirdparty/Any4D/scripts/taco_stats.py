import argparse
import json
import os
import hydra

import numpy as np
import torch
import decord
from tqdm import tqdm

from any4d.utils.misc import seed_everything
from any4d.models import init_model
from any4d.utils.moge_inference import load_moge_model

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


class Any4DDepthEstimator:
    def __init__(self, dataset_name, gpu_id, total_gpus):
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
        # self.model = init_inference_model(self.high_level_config, self.checkpoint_path, self.device)
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

    def load_mp4(self, path):
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        frames = vr.get_batch(range(len(vr))).asnumpy()
        return frames

    def estimate_depth(self):
        video_list = []
        for episode_name in tqdm(self.episode_list, desc="Loading videos"):
            video_path = os.path.join(self.video_base_path, episode_name, "color.mp4")
            extri_path = os.path.join(self.extri_intri_base_path, episode_name, "egocentric_frame_extrinsic.npy")
            intri_path = os.path.join(self.extri_intri_base_path, episode_name, "egocentric_intrinsic.txt")
            extrinsics = np.load(extri_path)
            intrinsics = np.loadtxt(intri_path)
            video_images = self.load_mp4(video_path)
            print(f"Episode {episode_name} has {len(video_images)} frames")
            print(f"Extrinsics has {len(extrinsics)} frames")
            minlen = min(len(video_images), len(extrinsics))
            maxlen = max(len(video_images), len(extrinsics))
            gap = maxlen - minlen
            video_list.append((minlen, gap, episode_name))
        # sort video_list by the number of frames and output to file
        video_list.sort(key=lambda x: x[0])
        with open("video_list_len.txt", "w") as f:
            for video_name in video_list:
                f.write(f"{video_name[2]} minlen: {video_name[0]} gap: {video_name[1]}\n")
        video_list.sort(key=lambda x: x[1])
        with open("video_list_gap.txt", "w") as f:
            for video_name in video_list:
                f.write(f"{video_name[2]} gap: {video_name[1]} minlen: {video_name[0]}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="taco")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--total_gpus", type=int, default=8)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    any4d_depth_estimator = Any4DDepthEstimator(args.dataset_name, args.gpu_id, args.total_gpus)
    seed_everything(0)
    any4d_depth_estimator.estimate_depth()