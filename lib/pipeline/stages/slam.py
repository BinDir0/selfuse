import argparse
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METRIC3D_ROOT = PROJECT_ROOT / "thirdparty" / "Metric3D"
if str(METRIC3D_ROOT) not in sys.path:
    sys.path.insert(0, str(METRIC3D_ROOT))

from hawor.utils.process import block_print, enable_print
from lib.pipeline.est_scale import est_scale_hybrid, est_scale_hybrid_batch
from lib.pipeline.frame_source import build_frame_source
from lib.pipeline.slam_geom_utils import est_calib, get_dimention


QUIET_MODE = os.environ.get("HAWOR_QUIET", "0") == "1"


def vprint(*args, **kwargs):
    if not QUIET_MODE:
        print(*args, **kwargs)


def build_metric3d_runner(weight_path=None):
    from metric import Metric3D

    if weight_path is None:
        weight_path = str(METRIC3D_ROOT / "weights" / "metric_depth_vit_large_800k.pth")
    block_print()
    try:
        metric = Metric3D(weight_path)
    finally:
        enable_print()
    return metric


def _resolve_seq_folder(video_path: str, seq_folder: str = None) -> str:
    if seq_folder is not None:
        return seq_folder
    video_path_obj = Path(video_path)
    return str(video_path_obj.parent / video_path_obj.stem)


def _resolve_frame_source(video_path: str, frame_source=None):
    return frame_source or build_frame_source(video_path)


def _load_masks(seq_folder: str, start_idx: int, end_idx: int) -> torch.Tensor:
    masks_path = os.path.join(seq_folder, f"tracks_{start_idx}_{end_idx}", "model_masks.npy")
    return torch.from_numpy(np.load(masks_path, allow_pickle=True))


def _resolve_focal(seq_folder: str, requested_focal: float = None) -> float:
    if requested_focal is not None:
        return float(requested_focal)

    focal_path = os.path.join(seq_folder, "est_focal.txt")
    try:
        with open(focal_path, "r", encoding="utf-8") as handle:
            return float(handle.read())
    except Exception:
        focal = 600.0
        vprint("No focal length provided")
        with open(focal_path, "w", encoding="utf-8") as handle:
            handle.write(str(focal))
        return focal


def _build_calibration(frame_source, focal: float) -> np.ndarray:
    calib = np.asarray(est_calib(frame_source), dtype=np.float32)
    calib[:2] = float(focal)
    return calib


def _run_droid_backend(frame_source, masks, calib, droid_net=None):
    from lib.pipeline.masked_droid_slam import run_slam

    droid, traj = run_slam(frame_source, masks=masks, calib=calib, droid_net=droid_net)
    n = int(droid.video.counter.value)
    tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
    disps = droid.video.disps_up.cpu().numpy()[:n]
    vprint("DBA errors:", droid.backend.errors)

    del droid
    torch.cuda.empty_cache()

    return {
        "backend": "droid",
        "traj": np.asarray(traj, dtype=np.float32),
        "tstamp": np.asarray(tstamp, dtype=np.int32),
        "disps": np.asarray(disps, dtype=np.float32),
    }


def _run_dpvo_backend(frame_source, masks, calib):
    from lib.pipeline.dpvo_slam import run_dpvo_slam

    traj, disps, traj_tstamp, disp_tstamp = run_dpvo_slam(frame_source, masks, calib=calib)
    traj_tstamp = np.asarray(traj_tstamp, dtype=np.int32).reshape(-1)
    disp_tstamp = np.asarray(disp_tstamp, dtype=np.int32).reshape(-1)
    if len(disp_tstamp) == len(disps):
        tstamp = disp_tstamp
    else:
        tstamp = traj_tstamp

    return {
        "backend": "dpvo",
        "traj": np.asarray(traj, dtype=np.float32),
        "tstamp": tstamp,
        "disps": np.asarray(disps, dtype=np.float32),
    }


def _run_slam_backend(args, frame_source, masks, calib, droid_net=None):
    slam_backend = getattr(args, "slam_backend", "dpvo")
    if slam_backend == "droid":
        return _run_droid_backend(frame_source, masks, calib, droid_net=droid_net)
    if slam_backend == "dpvo":
        return _run_dpvo_backend(frame_source, masks, calib)
    raise ValueError(f"Unknown slam backend: {slam_backend}")


def _ordered_unique_indices(frame_indices: Sequence[int]):
    return list(dict.fromkeys(int(frame_idx) for frame_idx in frame_indices))


def _depth_frame_indices(frame_source, tstamp, predict_all_frames: bool):
    if predict_all_frames:
        return list(range(len(frame_source)))
    return _ordered_unique_indices(tstamp)


def _resize_depths(depth_batch: np.ndarray, output_hw):
    out_h, out_w = output_hw
    resized = [
        cv2.resize(depth.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        for depth in depth_batch
    ]
    return np.stack(resized, axis=0)


def _predict_metric3d_depths(frame_source, frame_indices, metric_runner, calib, batch_size, output_hw):
    pred_depths = []
    worker_count = min(8, max(1, len(frame_indices)))

    with ThreadPoolExecutor(max_workers=worker_count) as frame_loader:
        for batch_start in tqdm(
            range(0, len(frame_indices), batch_size),
            desc="Metric3D batches",
            disable=QUIET_MODE,
        ):
            batch_indices = frame_indices[batch_start : batch_start + batch_size]
            batch_frames = list(
                frame_loader.map(lambda frame_idx: frame_source.get_frame(int(frame_idx), rgb=True), batch_indices)
            )
            batch_depths = metric_runner.batch_inference(batch_frames, calib)
            pred_depths.append(_resize_depths(np.asarray(batch_depths), output_hw))

    return np.concatenate(pred_depths, axis=0) if pred_depths else np.empty((0,) + tuple(output_hw), dtype=np.float32)


def _predict_any4d_depths(frame_source, frame_indices, any4d_runner, batch_size, output_hw, args):
    from lib.pipeline.any4d_depth import predict_any4d_depth_batch

    pred_depths = []
    for batch_start in tqdm(
        range(0, len(frame_indices), batch_size),
        desc="Any4D batches",
        disable=QUIET_MODE,
    ):
        batch_indices = frame_indices[batch_start : batch_start + batch_size]
        batch_depths = predict_any4d_depth_batch(
            frame_source,
            batch_indices,
            runner=any4d_runner,
            any4d_repo_root=getattr(args, "any4d_repo_root", None),
            checkpoint_path=getattr(args, "any4d_checkpoint_path", None),
            resolution_set=getattr(args, "any4d_resolution_set", None),
            use_amp=getattr(args, "any4d_use_amp", None),
        )
        pred_depths.append(_resize_depths(np.asarray(batch_depths), output_hw))

    return np.concatenate(pred_depths, axis=0) if pred_depths else np.empty((0,) + tuple(output_hw), dtype=np.float32)


def _depth_cache_path(seq_folder: str, depth_backend: str, start_idx: int, end_idx: int, predict_all_frames: bool) -> str:
    scope = "all" if predict_all_frames else "keyframes"
    return os.path.join(seq_folder, "SLAM", f"dense_depth_{depth_backend}_{scope}_{start_idx}_{end_idx}.npz")


def _load_cached_depths(cache_path: str):
    if not os.path.exists(cache_path):
        return None

    with np.load(cache_path, allow_pickle=False) as cached:
        return {
            "frame_indices": cached["frame_indices"].astype(np.int32),
            "pred_depths": cached["pred_depths"].astype(np.float32),
        }


def _save_cached_depths(cache_path: str, frame_indices, pred_depths):
    np.savez_compressed(
        cache_path,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        pred_depths=np.asarray(pred_depths, dtype=np.float32),
    )


def _predict_depths(args, frame_source, seq_folder, start_idx, end_idx, tstamp, calib, metric_runner, any4d_runner, metric3d_batch_size):
    depth_backend = getattr(args, "depth_backend", "metric3d")
    predict_all_frames = bool(getattr(args, "depth_predict_all_frames", True))
    frame_indices = _depth_frame_indices(frame_source, tstamp, predict_all_frames)
    output_hw = get_dimention(frame_source)
    os.makedirs(os.path.join(seq_folder, "SLAM"), exist_ok=True)

    cache_path = _depth_cache_path(seq_folder, depth_backend, start_idx, end_idx, predict_all_frames)
    cached = _load_cached_depths(cache_path) if predict_all_frames else None
    if cached is not None and np.array_equal(cached["frame_indices"], np.asarray(frame_indices, dtype=np.int32)):
        return cached["frame_indices"], cached["pred_depths"], cache_path, True

    if depth_backend == "metric3d":
        metric_runner = metric_runner or build_metric3d_runner()
        pred_depths = _predict_metric3d_depths(
            frame_source,
            frame_indices,
            metric_runner,
            calib,
            metric3d_batch_size,
            output_hw,
        )
    elif depth_backend == "any4d":
        if any4d_runner is None:
            from lib.pipeline.any4d_depth import build_any4d_runner

            any4d_runner = build_any4d_runner(
                any4d_repo_root=getattr(args, "any4d_repo_root", None),
                checkpoint_path=getattr(args, "any4d_checkpoint_path", None),
                resolution_set=getattr(args, "any4d_resolution_set", None),
                use_amp=getattr(args, "any4d_use_amp", None),
            )
        pred_depths = _predict_any4d_depths(
            frame_source,
            frame_indices,
            any4d_runner,
            metric3d_batch_size,
            output_hw,
            args,
        )
    else:
        raise ValueError(f"Unknown depth backend: {depth_backend}")

    if predict_all_frames:
        _save_cached_depths(cache_path, frame_indices, pred_depths)

    return np.asarray(frame_indices, dtype=np.int32), pred_depths, cache_path, False


def _gather_keyframe_depths(tstamp, depth_frame_indices, pred_depths):
    depth_by_frame = {
        int(frame_idx): pred_depths[i]
        for i, frame_idx in enumerate(np.asarray(depth_frame_indices, dtype=np.int32).tolist())
    }
    return [depth_by_frame[int(frame_idx)] for frame_idx in tstamp]


def _estimate_scale(disps, pred_depths, masks, tstamp):
    min_threshold = 0.4
    max_threshold = 0.7

    slam_depth_list = [1.0 / disps[i] for i in range(len(tstamp))]
    mask_list = [masks[int(frame_idx)].cpu().numpy().astype(np.uint8) for frame_idx in tstamp]
    scales_ = est_scale_hybrid_batch(
        slam_depth_list,
        pred_depths,
        sigma=0.5,
        masks=mask_list,
        near_thresh=min_threshold,
        far_thresh=max_threshold,
    )

    for i in range(len(tstamp)):
        if not math.isnan(scales_[i]):
            continue
        near_thresh = min_threshold
        far_thresh = max_threshold
        for _ in range(10):
            near_thresh -= 0.1
            far_thresh += 0.1
            scales_[i] = est_scale_hybrid(
                slam_depth_list[i],
                pred_depths[i],
                sigma=0.5,
                msk=mask_list[i],
                near_thresh=near_thresh,
                far_thresh=far_thresh,
            )
            if not math.isnan(scales_[i]):
                break

    valid_scales = [scale for scale in scales_ if not math.isnan(scale)]
    if valid_scales:
        fallback = np.median(valid_scales)
        for i in range(len(scales_)):
            if math.isnan(scales_[i]):
                scales_[i] = fallback

    return float(np.median(scales_))


def _save_slam_outputs(seq_folder, start_idx, end_idx, tstamp, disps, traj, focal, calib, scale):
    os.makedirs(os.path.join(seq_folder, "SLAM"), exist_ok=True)
    save_path = os.path.join(seq_folder, "SLAM", f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    np.savez(
        save_path,
        tstamp=np.asarray(tstamp, dtype=np.int32),
        disps=np.asarray(disps, dtype=np.float32),
        traj=np.asarray(traj, dtype=np.float32),
        img_focal=float(focal),
        img_center=np.asarray(calib[-2:], dtype=np.float32),
        scale=np.float32(scale),
    )
    return save_path


def _print_timing(
    video_path: str,
    timing: dict,
    num_keyframes: int,
    depth_frame_count: int,
    *,
    slam_backend: str,
    depth_backend: str,
    predict_all_frames: bool,
    used_depth_cache: bool,
):
    total_time = timing["total"]
    print(f"\n{'=' * 60}")
    print(f"SLAM Stage Timing for {os.path.basename(video_path)}")
    print(f"{'=' * 60}")
    for key in ("1_load_masks", "2_slam", "3_depth", "4_scale_est", "5_save"):
        elapsed = timing.get(key, 0.0)
        pct = elapsed / total_time * 100 if total_time > 0 else 0
        print(f"  {key:20s}: {elapsed:7.2f}s ({pct:5.1f}%)")
    print(f"  {'total':20s}: {total_time:7.2f}s")
    print(f"  {'slam_backend':20s}: {slam_backend}")
    print(f"  {'depth_backend':20s}: {depth_backend}")
    print(f"  {'depth_scope':20s}: {'all_frames' if predict_all_frames else 'keyframes'}")
    print(f"  {'depth_cache_used':20s}: {used_depth_cache}")
    print(f"  {'keyframes':20s}: {num_keyframes}")
    print(f"  {'depth_frames':20s}: {depth_frame_count}")
    print(f"{'=' * 60}\n")


def hawor_slam(
    args,
    start_idx,
    end_idx,
    metric_runner=None,
    metric3d_batch_size=32,
    droid_net=None,
    any4d_runner=None,
    frame_source=None,
    seq_folder=None,
):
    timing = {}
    start_time = time.time()

    seq_folder = _resolve_seq_folder(args.video_path, seq_folder)
    os.makedirs(seq_folder, exist_ok=True)
    frame_source = _resolve_frame_source(args.video_path, frame_source)
    slam_backend = getattr(args, "slam_backend", "dpvo")
    depth_backend = getattr(args, "depth_backend", "metric3d")
    predict_all_frames = bool(getattr(args, "depth_predict_all_frames", True))
    vprint(
        f"Running slam on {seq_folder} "
        f"(slam_backend={slam_backend}, depth_backend={depth_backend}, "
        f"depth_scope={'all_frames' if predict_all_frames else 'keyframes'}) ..."
    )

    t0 = time.time()
    masks = _load_masks(seq_folder, start_idx, end_idx)
    focal = _resolve_focal(seq_folder, getattr(args, "img_focal", None))
    calib = _build_calibration(frame_source, focal)
    timing["1_load_masks"] = time.time() - t0

    t0 = time.time()
    slam_outputs = _run_slam_backend(args, frame_source, masks, calib, droid_net=droid_net)
    traj = slam_outputs["traj"]
    tstamp = slam_outputs["tstamp"]
    disps = slam_outputs["disps"]
    timing["2_slam"] = time.time() - t0

    t0 = time.time()
    depth_frame_indices, depth_predictions, depth_cache_path, used_cache = _predict_depths(
        args,
        frame_source,
        seq_folder,
        start_idx,
        end_idx,
        tstamp,
        calib,
        metric_runner,
        any4d_runner,
        metric3d_batch_size,
    )
    keyframe_depths = _gather_keyframe_depths(tstamp, depth_frame_indices, depth_predictions)
    if used_cache:
        vprint(f"Loaded cached dense depth from {depth_cache_path}")
    elif predict_all_frames:
        vprint(f"Saved dense depth cache to {depth_cache_path}")
    timing["3_depth"] = time.time() - t0

    t0 = time.time()
    scale = _estimate_scale(disps, keyframe_depths, masks, tstamp)
    vprint(f"estimated scale: {scale}")
    timing["4_scale_est"] = time.time() - t0

    t0 = time.time()
    _save_slam_outputs(seq_folder, start_idx, end_idx, tstamp, disps, traj, focal, calib, scale)
    timing["5_save"] = time.time() - t0

    timing["total"] = time.time() - start_time
    _print_timing(
        args.video_path,
        timing,
        len(tstamp),
        len(depth_frame_indices),
        slam_backend=slam_backend,
        depth_backend=depth_backend,
        predict_all_frames=predict_all_frames,
        used_depth_cache=used_cache,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default="")
    parser.add_argument("--input_type", type=str, default="file")
    parser.add_argument("--slam_backend", type=str, default="dpvo", choices=["droid", "dpvo"])
    parser.add_argument("--depth_backend", type=str, default="metric3d", choices=["metric3d", "any4d"])
    parser.add_argument(
        "--depth_predict_all_frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Predict depth for all frames and cache the dense sidecar under SLAM/.",
    )
    parser.add_argument("--any4d_repo_root", type=str, default=None)
    parser.add_argument("--any4d_checkpoint_path", type=str, default=None)
    parser.add_argument("--any4d_resolution_set", type=int, default=None)
    parser.add_argument(
        "--any4d_use_amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override Any4D AMP usage when depth_backend=any4d.",
    )
    args = parser.parse_args()

    from lib.pipeline.stages.detect_track import detect_track_video

    start_idx, end_idx, _, _ = detect_track_video(args)
    hawor_slam(args, start_idx, end_idx)
