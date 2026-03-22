"""
DPVO-SLAM adapter for HaWoR pipeline.

Uses frame_source to read frames, runs DPVO, and returns traj + disps
in the format expected by hawor_slam (Metric3D + scale estimation).
"""
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DPVO_ROOT = PROJECT_ROOT / "thirdparty" / "DPVO"
if str(DPVO_ROOT) not in sys.path:
    sys.path.insert(0, str(DPVO_ROOT))


def _frame_stream(frame_source, calib, stride=1, max_size=800):
    """
    Yield (t, image_bgr, intrinsics) for DPVO from frame_source.

    为了让 DPVO 在 80G 显存下稳定运行，这里对输入分辨率做轻量下采样：
    - 将长边限制到 max_size（默认 800 像素）
    - 按相同缩放比例调整内参 fx, fy, cx, cy
    - 再裁剪到 16 的倍数满足 DPVO 要求
    """
    fx, fy, cx, cy = np.array(calib[:4], dtype=np.float64)
    n = len(frame_source)
    for t in range(0, n, stride):
        img = frame_source.get_frame(t, rgb=False)
        if img is None:
            break
        h, w = img.shape[:2]

        # 等比缩放到长边不超过 max_size
        scale = min(max_size / max(h, w), 1.0)
        if scale < 1.0:
            new_h = int(h * scale)
            new_w = int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            fx_s, fy_s = fx * scale, fy * scale
            cx_s, cy_s = cx * scale, cy * scale
        else:
            fx_s, fy_s, cx_s, cy_s = fx, fy, cx, cy

        # 裁剪到 16 的倍数（DPVO 要求）
        img = img[: h - h % 16, : w - w % 16]
        intrinsics = np.array([fx_s, fy_s, cx_s, cy_s], dtype=np.float64)
        yield t, img, intrinsics


def _poses_to_traj(poses):
    """Convert DPVO poses [tx,ty,tz,qx,qy,qz,qw] to hawor [tx,ty,tz,qw,qx,qy,qz]."""
    traj = np.zeros_like(poses)
    traj[:, :3] = poses[:, :3]
    traj[:, 3] = poses[:, 6]  # qw
    traj[:, 4:7] = poses[:, 3:6]  # qx,qy,qz
    return traj


def _build_disps_from_patches(slam, H, W, fx, fy, cx, cy):
    """
    Build per-frame disparity maps from DPVO patches for scale estimation.
    DPVO uses sparse patches; we rasterize patch centers (x,y,disp) and interpolate.
    """
    n = slam.n
    ht, wd = slam.ht, slam.wd
    disps_list = []

    patches = slam.pg.patches_.cpu().numpy()[:n]

    for i in range(n):
        x = patches[i, :, 0, 1, 1]
        y = patches[i, :, 1, 1, 1]
        d = patches[i, :, 2, 1, 1]
        valid = d > 1e-6
        if not np.any(valid):
            disps_list.append(np.ones((ht, wd), dtype=np.float32) * 0.01)
            continue

        x, y, d = x[valid], y[valid], d[valid]

        disp_map = np.zeros((ht, wd), dtype=np.float32)
        cnt_map = np.zeros((ht, wd), dtype=np.int32)
        ix = np.clip(np.round(y).astype(int), 0, ht - 1)
        iy = np.clip(np.round(x).astype(int), 0, wd - 1)
        np.add.at(disp_map, (ix, iy), d)
        np.add.at(cnt_map, (ix, iy), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            disp_map = np.where(cnt_map > 0, disp_map / cnt_map, 0)

        med = np.median(d)
        disp_map[disp_map <= 0] = med
        disp_map[disp_map <= 0] = med
        disps_list.append(disp_map.astype(np.float32))

    if (H, W) != (ht, wd):
        disps_list = [
            cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR) for d in disps_list
        ]

    return np.stack(disps_list, axis=0)


def run_dpvo_slam(imagedir, masks, calib=None, stride=1):
    """
    Run DPVO on frame_source and return traj, disps for hawor_slam.

    Args:
        imagedir: frame_source (has get_frame, __len__) or path to video/images
        masks: torch.Tensor (T,H,W) - used for mask_list in scale est (not by DPVO)
        calib: np.ndarray [fx, fy, cx, cy]
        stride: frame stride

    Returns:
        traj: (T, 7) [tx, ty, tz, qw, qx, qy, qz] for all processed frames
        disps: (K, H, W) disparity (inverse depth) maps only for DPVO internal keyframes
        tstamps: (T,) frame indices for traj
        tstamps_disps: (K,) frame indices for disps (DPVO internal keyframe timestamps)
    """
    from dpvo.config import cfg
    from dpvo.dpvo import DPVO
    from lib.pipeline.slam_geom_utils import est_calib, get_dimention

    frame_source = imagedir
    if hasattr(imagedir, "get_frame") and hasattr(imagedir, "__len__"):
        pass
    else:
        from lib.pipeline.frame_source import build_frame_source

        frame_source = build_frame_source(imagedir)

    if calib is None:
        calib = np.array(est_calib(frame_source))

    calib = np.array(calib[:4], dtype=np.float64)
    config_path = DPVO_ROOT / "config" / "default.yaml"
    if config_path.exists():
        cfg.merge_from_file(str(config_path))

    # Optional: override DPVO keyframe/patch configs for experiments.
    # This is critical for HaWoR, because DPVO only provides patch-based
    # disparity/depth for frames it kept in its internal keyframe graph.
    # If the number of generated disps is too small, lowering KEYFRAME_THRESH
    # (or disabling its removal effect) can increase coverage.
    env_to_cfg = {
        "HAWOR_DPVO_PATCHES_PER_FRAME": "PATCHES_PER_FRAME",
        "HAWOR_DPVO_REMOVAL_WINDOW": "REMOVAL_WINDOW",
        "HAWOR_DPVO_OPTIMIZATION_WINDOW": "OPTIMIZATION_WINDOW",
        "HAWOR_DPVO_PATCH_LIFETIME": "PATCH_LIFETIME",
        "HAWOR_DPVO_KEYFRAME_INDEX": "KEYFRAME_INDEX",
        "HAWOR_DPVO_KEYFRAME_THRESH": "KEYFRAME_THRESH",
        "HAWOR_DPVO_MIXED_PRECISION": "MIXED_PRECISION",
    }
    for env_k, cfg_k in env_to_cfg.items():
        if env_k not in os.environ:
            continue
        v = os.environ.get(env_k)
        if v is None or v == "":
            continue
        cur = getattr(cfg, cfg_k)
        # bool/float/int casting based on existing type
        if isinstance(cur, bool):
            setattr(cfg, cfg_k, v.strip().lower() in ("1", "true", "yes", "y", "on"))
        elif isinstance(cur, int):
            setattr(cfg, cfg_k, int(v))
        else:
            setattr(cfg, cfg_k, float(v))

    weight_path = str(DPVO_ROOT / "models" / "dpvo.pth")

    slam = None
    with torch.inference_mode():
        for t, image, intrinsics in _frame_stream(frame_source, calib, stride, max_size=800):
            image_t = torch.from_numpy(image).permute(2, 0, 1).float().cuda()
            intrinsics_t = torch.from_numpy(intrinsics).float().cuda()

            if slam is None:
                _, H_img, W_img = image_t.shape
                slam = DPVO(cfg, weight_path, ht=H_img, wd=W_img, viz=False)

            slam(t, image_t, intrinsics_t)

        poses, tstamps = slam.terminate()
        traj = _poses_to_traj(poses)

        H_out, W_out = get_dimention(frame_source)
        fx, fy, cx, cy = calib[:4]
        disps = _build_disps_from_patches(slam, H_out, W_out, fx, fy, cx, cy)
        # disparity maps are generated only for DPVO internal keyframes (slam.n)
        # Depending on DPVO build, slam.pg.tstamps_ can be torch.Tensor or np.ndarray.
        _tst = slam.pg.tstamps_
        if hasattr(_tst, "detach"):
            _tst = _tst.detach().cpu().numpy()
        else:
            _tst = np.asarray(_tst)
        tstamps_disps = _tst[: int(slam.n)].reshape(-1)

    del slam
    torch.cuda.empty_cache()

    return traj, disps, tstamps, tstamps_disps

