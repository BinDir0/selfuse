import os
import random

import torch

from lib.pipeline.stage_api import StageExecutionConfig


def set_determinism(seed: int):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True is safe here because input sizes are fixed in the main stages.
    torch.backends.cudnn.benchmark = True


class WorkerRuntime:
    """Long-lived per-GPU runtime that caches models across videos."""

    def __init__(
        self,
        gpu: str,
        checkpoint: str,
        infiller_weight: str,
        img_focal: float = None,
        chunk_batch_size: int = 8,
        num_workers: int = 16,
        render_batch_size: int = 8,
        metric3d_batch_size: int = 8,
        detect_batch_size: int = 256,
        detect_io_workers: int = 16,
        detect_device: str = "cuda:0",
        detect_half_precision: bool = True,
        infiller_window_batch_size: int = 64,
        rebuild_cam_space_cache: bool = False,
    ):
        self.gpu = gpu
        self.stage_config = StageExecutionConfig(
            img_focal=img_focal,
            checkpoint=checkpoint,
            infiller_weight=infiller_weight,
            chunk_batch_size=chunk_batch_size,
            num_workers=num_workers,
            render_batch_size=render_batch_size,
            metric3d_batch_size=metric3d_batch_size,
            detect_batch_size=detect_batch_size,
            detect_io_workers=detect_io_workers,
            infiller_window_batch_size=infiller_window_batch_size,
            rebuild_cam_space_cache=rebuild_cam_space_cache,
            detect_device=detect_device,
            detect_half_precision=detect_half_precision,
        )

        self.detector_runner = None
        self.motion_runner = None
        self.metric_runner = None
        self.infiller_runner = None
        self.droid_net = None
        self.mano_right = None
        self.mano_left = None

        if self.gpu is not None and self.gpu != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

    def ensure_runner(self, stage: str):
        if stage == "detect_track" and self.detector_runner is None:
            from ultralytics import YOLO

            self.detector_runner = YOLO("./weights/external/detector.pt")

        if stage == "motion" and self.motion_runner is None:
            from hawor.utils.process import get_mano_cfg
            from lib.models.mano_wrapper import MANO
            from lib.pipeline.stages.motion import build_motion_runner

            self.motion_runner = build_motion_runner(self.stage_config.checkpoint)
            device = self.motion_runner["device"]

            self.mano_right = MANO(**get_mano_cfg(is_right=True)).to(device)
            self.mano_left = MANO(**get_mano_cfg(is_right=False)).to(device)
            self.mano_left.shapedirs[:, 0, :] *= -1

        if stage == "slam" and self.metric_runner is None:
            from lib.pipeline.stages.slam import build_metric3d_runner

            self.metric_runner = build_metric3d_runner()

        if stage == "slam" and self.droid_net is None:
            from lib.pipeline.masked_droid_slam import build_droid_net

            self.droid_net = build_droid_net()

        if stage == "infiller" and self.infiller_runner is None:
            from lib.pipeline.stages.infiller import build_infiller_runner

            self.infiller_runner = build_infiller_runner(self.stage_config.infiller_weight)
