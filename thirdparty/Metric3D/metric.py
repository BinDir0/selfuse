# standard library
from pathlib import Path
from typing import *
import sys, os
# third party
import cv2
import numpy as np
import torch
from PIL import Image
from mmengine import Config
# metric 3d
metric3d_path = Path(__file__).resolve().parent
metric3d_mono_path = metric3d_path / 'mono'
sys.path.append(str(metric3d_path))
sys.path.append(str(metric3d_mono_path))
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import transform_test_data_scalecano, get_prediction
from mono.utils.mldb import load_data_info, reset_ckpt_path
from mono.utils.transform import gray_to_colormap

__ALL__ = ['Metric3D']

def calculate_radius(mask):
    # 获取矩阵的大小 N
    N = mask.shape[0]
    
    # 找到矩阵的中心点
    center = (N - 1) / 2
    
    # 获取所有值为0的点的坐标
    y, x = np.where(mask == 0)
    
    # 计算这些点到中心的距离
    distances = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    
    # 半径是这些距离的最大值
    radius = distances.max()
    
    return radius

class Metric3D:
    cfg_: Config
    model_: torch.nn.Module

    def __init__(
        self,
        checkpoint: Union[str, Path] = './weights/metric_depth_vit_large_800k.pth',
        model_name: str = 'v2-L',
    ) -> None:
        checkpoint = Path(checkpoint).resolve()
        cfg:Config = self._load_config_(model_name, checkpoint)
        # build model
        model = get_configured_monodepth_model(cfg, )
        model = torch.nn.DataParallel(model).cuda()
        model, _, _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
        model.eval()
        # save to self
        self.cfg_ = cfg
        self.model_ = model

    @torch.no_grad()
    def __call__(
        self,
        rgb_image: Union[np.ndarray, Image.Image, str, Path],
        intrinsic: Union[str, Path, np.ndarray],
        d_max: Optional[float] = 300,
        d_min: Optional[float] = 0,
        margin_mask=None,
        crop_margin=0
    ) -> np.ndarray:
        # read image
        if isinstance(rgb_image, (str, Path)):
            rgb_image = np.array(Image.open(rgb_image))
        elif isinstance(rgb_image, Image.Image):
            rgb_image = np.array(rgb_image)

        if isinstance(intrinsic, (str, Path)):
            intrinsic = np.loadtxt(intrinsic)
        intrinsic = intrinsic[:4]

        # crop margin mask
        if crop_margin != 0:
            original_h, original_w = margin_mask.shape
            # radius = calculate_radius(margin_mask) - crop_margin
            radius = original_h // 2 - crop_margin
            left, right, up, bottom = int(original_w//2-radius), int(original_w//2+radius), int(original_h//2-radius), int(original_h//2+radius)
            rgb_image = rgb_image[up:bottom, left:right]
            h, w = rgb_image.shape[:2]
            intrinsic[2] = w/2
            intrinsic[3] = h/2
            cv2.imwrite("debug.png", rgb_image[:, :, ::-1])
        # get intrinsic
        h, w = rgb_image.shape[:2]

        input_size = (616, 1064)
        scale = min(input_size[0] / h, input_size[1] / w)
        # transform image
        rgb_input, cam_models_stacks, pad, label_scale_factor = \
            transform_test_data_scalecano(rgb_image, intrinsic, self.cfg_.data_basic)
        # predict depth
        normalize_scale = self.cfg_.data_basic.depth_range[1]
        rgb_input = rgb_input.unsqueeze(0)
        pred_depth, output = get_prediction(
            model = self.model_,
            input = rgb_input,
            cam_model = cam_models_stacks,
            pad_info = pad,
            scale_info = label_scale_factor,
            gt_depth = None,
            normalize_scale = normalize_scale,
            ori_shape=[h, w],
        )
        
        # post process
        # pred_depth = (pred_depth > 0) * (pred_depth < 300) * pred_depth

        pred_depth = pred_depth.squeeze().cpu().numpy()
        pred_depth[pred_depth > d_max] = 0
        pred_depth[pred_depth < d_min] = 0

        pred_depth = pred_depth[pad[0] : pred_depth.shape[0] - pad[1], pad[2] : pred_depth.shape[1] - pad[3]]

        canonical_to_real_scale = intrinsic[0] * scale / 1000.0 # 1000.0 is the focal length of canonical camera
        pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric

        # because crop margin at beginning
        if not margin_mask is None:
            final_depth = np.zeros((original_h, original_w))
            pred_depth = cv2.resize(pred_depth, (w, h))
            final_depth[up:bottom, left:right] = pred_depth
        return pred_depth

    def _load_config_(
        self,
        model_name: str,
        checkpoint: Union[str, Path],
    ) -> Config:
        print(f'Loading model {model_name} from {checkpoint}')
        config_path = metric3d_path / 'mono/configs/HourglassDecoder'
        assert model_name in ['v2-L', 'v2-S', 'v2-g'], f"Model {model_name} not supported"
        # load config file
        cfg = Config.fromfile(
            str(config_path / 'vit.raft5.large.py') if model_name == 'v2-L' 
            else str(config_path / 'vit.raft5.small.py') if model_name == 'v2-S' 
            else str(config_path / 'vit.raft5.giant2.py')
        )
        cfg.load_from = str(checkpoint)
        # load data info
        data_info = {}
        load_data_info('data_info', data_info=data_info)
        cfg.mldb_info = data_info
        # update check point info
        reset_ckpt_path(cfg.model, data_info)
        # set distributed
        cfg.distributed = False
        
        return cfg
    
    @torch.no_grad()
    def batch_inference(
        self,
        rgb_images: List[np.ndarray],
        intrinsic: Union[str, Path, np.ndarray],
        d_max: Optional[float] = 300,
        d_min: Optional[float] = 0,
    ) -> List[np.ndarray]:
        """
        Batch inference for multiple images with the same intrinsic.

        Args:
            rgb_images: List of RGB images (numpy arrays)
            intrinsic: Camera intrinsic parameters
            d_max: Maximum depth threshold
            d_min: Minimum depth threshold

        Returns:
            List of predicted depth maps
        """
        if isinstance(intrinsic, (str, Path)):
            intrinsic = np.loadtxt(intrinsic)
        intrinsic = intrinsic[:4]

        # Preprocess all images
        batch_inputs = []
        batch_cam_models = []
        batch_pads = []
        batch_scales = []
        batch_ori_shapes = []

        for rgb_image in rgb_images:
            h, w = rgb_image.shape[:2]
            rgb_input, cam_models_stacks, pad, label_scale_factor = \
                transform_test_data_scalecano(rgb_image, intrinsic, self.cfg_.data_basic)
            batch_inputs.append(rgb_input)
            batch_cam_models.append(cam_models_stacks)
            batch_pads.append(pad)
            batch_scales.append(label_scale_factor)
            batch_ori_shapes.append([h, w])

        # Stack inputs into batch
        batch_tensor = torch.stack(batch_inputs, dim=0)

        # Predict depth for the entire batch
        normalize_scale = self.cfg_.data_basic.depth_range[1]

        # Stack cam_models for batch processing
        # batch_cam_models[i] is a list of 5 tensors (multi-scale)
        # We need to stack each scale level across the batch
        batch_cam_models_stacked = []
        for scale_idx in range(len(batch_cam_models[0])):  # 5 scales
            scale_tensors = [batch_cam_models[i][scale_idx] for i in range(len(rgb_images))]
            batch_cam_models_stacked.append(torch.cat(scale_tensors, dim=0))

        # Single batch inference call
        data = dict(
            input=batch_tensor,
            cam_model=batch_cam_models_stacked,
        )
        with torch.cuda.amp.autocast():
            pred_depth_batch, confidence_batch, output_dict = self.model_.module.inference(data)

        # Split batch results
        pred_depths_batch = [pred_depth_batch[i:i+1] for i in range(len(rgb_images))]

        # Post-process each depth map
        results = []
        input_size = (616, 1064)
        for i, (pred_depth, rgb_image) in enumerate(zip(pred_depths_batch, rgb_images)):
            h, w = rgb_image.shape[:2]
            scale = min(input_size[0] / h, input_size[1] / w)
            pad = batch_pads[i]

            pred_depth = pred_depth.squeeze().cpu().numpy()
            pred_depth[pred_depth > d_max] = 0
            pred_depth[pred_depth < d_min] = 0

            pred_depth = pred_depth[pad[0]:pred_depth.shape[0] - pad[1], pad[2]:pred_depth.shape[1] - pad[3]]

            canonical_to_real_scale = intrinsic[0] * scale / 1000.0
            pred_depth = pred_depth * canonical_to_real_scale

            results.append(pred_depth)

        return results

    @staticmethod
    def gray_to_colormap(depth: np.ndarray) -> np.ndarray:
        return gray_to_colormap(depth)