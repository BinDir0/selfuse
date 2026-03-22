import math
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import argparse
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
import cv2
from pycocotools import mask as masktool
from lib.pipeline.est_scale import *
from lib.pipeline.frame_source import build_frame_source
from lib.pipeline.slam_geom_utils import est_calib, get_dimention
from hawor.utils.process import block_print, enable_print

sys.path.insert(0, os.path.dirname(__file__) + '/../../thirdparty/Metric3D')
from metric import Metric3D

# Check if we should suppress verbose output
QUIET_MODE = os.environ.get("HAWOR_QUIET", "0") == "1"

def vprint(*args, **kwargs):
    """Print only if not in quiet mode."""
    if not QUIET_MODE:
        print(*args, **kwargs)


def get_all_mp4_files(folder_path):
    # Ensure the folder path is absolute
    folder_path = os.path.abspath(folder_path)

    # Recursively search for all .mp4 files in the folder and its subfolders
    mp4_files = glob(os.path.join(folder_path, '**', '*.mp4'), recursive=True)

    return mp4_files

def split_list_by_interval(lst, interval=1000):
    start_indices = []
    end_indices = []
    split_lists = []

    for i in range(0, len(lst), interval):
        start_indices.append(i)
        end_indices.append(min(i + interval, len(lst)))
        split_lists.append(lst[i:i + interval])

    return start_indices, end_indices, split_lists

def build_metric3d_runner(weight_path='thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth'):
    block_print()
    metric = Metric3D(weight_path)
    enable_print()
    return metric


def _resolve_any4d_paths(
    project_root: str,
    any4d_repo_root: Optional[str],
    any4d_checkpoint_path: Optional[str],
    any4d_resolution_set: Optional[int],
    any4d_use_amp: bool,
):
    """Resolve Any4D repo / checkpoint / options (same rules as DPVO branch)."""
    _repo = any4d_repo_root
    if _repo is None or str(_repo).strip() == "":
        _repo = os.environ.get(
            "HAWOR_ANY4D_REPO_ROOT",
            os.path.join(project_root, "thirdparty", "Any4D"),
        ).strip()
    else:
        _repo = str(_repo).strip()
    if not os.path.isabs(_repo):
        _repo = os.path.abspath(os.path.join(project_root, _repo))
    any4d_repo_root_resolved = _repo

    _ckpt = any4d_checkpoint_path
    if _ckpt is None or str(_ckpt).strip() == "":
        _ckpt = os.environ.get(
            "HAWOR_ANY4D_CHECKPOINT_PATH",
            os.path.join(project_root, "checkpoints", "any4d_4v_combined.pth"),
        ).strip()
    else:
        _ckpt = str(_ckpt).strip()
    if not os.path.isabs(_ckpt):
        _ckpt = os.path.abspath(os.path.join(project_root, _ckpt))
    any4d_ckpt_path = _ckpt

    if any4d_resolution_set is None:
        any4d_resolution_set = int(os.environ.get("HAWOR_ANY4D_RESOLUTION", "518"))
    # AMP: default on; set HAWOR_ANY4D_USE_AMP=0 to disable. Explicit True from API stays on.
    if not any4d_use_amp:
        from scripts.scripts_test_video.run_any4d_depth import _env_flag_on

        any4d_use_amp = _env_flag_on("HAWOR_ANY4D_USE_AMP", default_on=True)

    return (
        any4d_repo_root_resolved,
        any4d_ckpt_path,
        int(any4d_resolution_set),
        bool(any4d_use_amp),
    )


def _depth_predict_all_frames_enabled(explicit: Optional[bool] = None) -> bool:
    """
    When True: run Metric3D/Any4D on every frame in [start_idx, end_idx], save uint16 npz, then subset keyframes.

    Default **on**. Disable with ``HAWOR_DEPTH_PREDICT_ALL_FRAMES=0`` or CLI ``--no_depth_predict_all_frames``.
    """
    if explicit is not None:
        return bool(explicit)
    v = os.environ.get("HAWOR_DEPTH_PREDICT_ALL_FRAMES", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


def _segment_frame_ids(start_idx: int, end_idx: int, n_frames_src: int) -> np.ndarray:
    a, b = int(start_idx), int(end_idx)
    ids = np.arange(a, b + 1, dtype=np.int64)
    return ids[(ids >= 0) & (ids < int(n_frames_src))]


def _save_dense_depth_uint16_npz(
    out_path: str,
    frame_indices: np.ndarray,
    depths_list,
    depth_backend: str,
):
    """Save per-frame metric depth maps as uint16 + global scale (decode: d = u16 * depth_scale)."""
    stack = np.stack([np.asarray(d, dtype=np.float32) for d in depths_list], axis=0)
    valid = np.isfinite(stack)
    dmax = float(np.nanmax(stack[valid])) if valid.any() else 1.0
    dmax = max(dmax, 1e-6)
    clipped = np.clip(stack, 0.0, dmax)
    u16 = np.round(clipped / dmax * 65535.0).astype(np.uint16)
    scale = np.float64(dmax / 65535.0)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(
        out_path,
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        depths_uint16=u16,
        depth_scale=scale,
        height=np.int32(stack.shape[1]),
        width=np.int32(stack.shape[2]),
        depth_backend=np.array([str(depth_backend)]),
    )


def _gather_keyframe_depths_from_dense(
    dense_depths: list,
    segment_frame_ids: np.ndarray,
    keyframe_tstamps: np.ndarray,
) -> list:
    seg = np.asarray(segment_frame_ids, dtype=np.int64).reshape(-1)
    id_to_i = {int(seg[i]): i for i in range(len(seg))}
    out = []
    for ts in np.asarray(keyframe_tstamps, dtype=np.int64).reshape(-1):
        ts = int(ts)
        if ts not in id_to_i:
            raise ValueError(
                f"SLAM keyframe frame id {ts} is outside dense depth segment "
                f"[{int(seg.min())}..{int(seg.max())}] — widen start_idx/end_idx or use "
                f"--no_depth_predict_all_frames / HAWOR_DEPTH_PREDICT_ALL_FRAMES=0."
            )
        out.append(dense_depths[id_to_i[ts]])
    return out


def _predict_depths_for_frames(
    frame_ids: np.ndarray,
    *,
    depth_backend: str,
    frame_source,
    metric3d_batch_size: int,
    calib,
    metric_runner,
    video_path: str,
    seq_folder: str,
    start_idx: int,
    end_idx: int,
    cache_tag: str,
    project_root: str,
    any4d_repo_root: Optional[str],
    any4d_checkpoint_path: Optional[str],
    any4d_resolution_set: Optional[int],
    any4d_use_amp: bool,
    W: int,
    H: int,
    any4d_cache_suffix: str = "",
):
    """
    Metric3D or Any4D depth for an ordered list of frame indices (global ids matching extracted_images).

    Any4D 整段写单个缓存 ``SLAM/any4d_depth_{cache_tag}_{start}_{end}{suffix}.npz``
    （``depths``、``frame_indices``）。``any4d_cache_suffix`` 用 ``_allframes`` 与关键帧模式区分。
    """
    depth_backend = str(depth_backend).lower().strip()
    if depth_backend not in ("metric3d", "any4d"):
        raise ValueError(f"Unknown depth backend: {depth_backend}")

    frame_ids = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
    n = int(frame_ids.shape[0])
    pred_depths = []

    if depth_backend == "metric3d":
        metric = metric_runner or build_metric3d_runner()
        max_workers = min(8, os.cpu_count() or 8)
        desc = (
            "Metric3D batches (all frames)"
            if any4d_cache_suffix
            else "Metric3D batches"
        )
        with ThreadPoolExecutor(max_workers=max_workers) as frame_loader:
            for batch_start in tqdm(
                range(0, n, metric3d_batch_size),
                desc=desc,
                disable=QUIET_MODE,
            ):
                batch_end = min(batch_start + metric3d_batch_size, n)
                batch_indices = frame_ids[batch_start:batch_end]

                batch_frames = list(
                    frame_loader.map(
                        lambda t: frame_source.get_frame(int(t), rgb=True), batch_indices
                    )
                )

                with torch.inference_mode():
                    batch_depths = metric.batch_inference(batch_frames, calib)

                for pred_depth in batch_depths:
                    pred_depth = cv2.resize(pred_depth, (W, H))
                    pred_depths.append(pred_depth)
        return pred_depths

    (
        any4d_repo_root_resolved,
        any4d_ckpt_path,
        res_set,
        use_amp,
    ) = _resolve_any4d_paths(
        project_root,
        any4d_repo_root,
        any4d_checkpoint_path,
        any4d_resolution_set,
        any4d_use_amp,
    )

    from scripts.scripts_test_video.run_any4d_depth import run_any4d_depth_batch

    suf = any4d_cache_suffix or ""
    _any4d_model = None
    slam_dir = os.path.join(seq_folder, "SLAM")
    merged_npz = os.path.join(
        slam_dir,
        f"any4d_depth_{cache_tag}_{start_idx}_{end_idx}{suf}.npz",
    )
    force = os.environ.get("HAWOR_ANY4D_FORCE_RERUN", "0") == "1"
    if force and os.path.isfile(merged_npz):
        try:
            os.remove(merged_npz)
        except OSError:
            pass

    if not force and os.path.isfile(merged_npz):
        depth_npz = np.load(merged_npz, allow_pickle=False)
        depth_stack = depth_npz["depths"]
        if "frame_indices" in getattr(depth_npz, "files", ()):
            cached_ids = np.asarray(depth_npz["frame_indices"], dtype=np.int64).reshape(-1)
            if cached_ids.shape[0] != n or not np.array_equal(cached_ids, frame_ids):
                depth_npz.close()
                depth_stack = None
            else:
                for i in range(n):
                    pred_depth = cv2.resize(depth_stack[i].astype(np.float32), (W, H))
                    pred_depths.append(pred_depth)
                depth_npz.close()
                return pred_depths
        else:
            # Old merged file without frame_indices: trust length only
            if depth_stack.shape[0] == n:
                for i in range(n):
                    pred_depth = cv2.resize(depth_stack[i].astype(np.float32), (W, H))
                    pred_depths.append(pred_depth)
                depth_npz.close()
                return pred_depths
            depth_npz.close()

    desc = (
        f"Any4D batches ({depth_backend}, {cache_tag}, all-frames)"
        if suf
        else f"Any4D batches ({depth_backend}, {cache_tag})"
    )
    for batch_start in tqdm(
        range(0, n, metric3d_batch_size),
        desc=desc,
        disable=QUIET_MODE,
    ):
        batch_end = min(batch_start + metric3d_batch_size, n)
        batch_indices = frame_ids[batch_start:batch_end]

        _any4d_model, depth_stack = run_any4d_depth_batch(
            video_path=str(video_path),
            frame_indices=batch_indices.tolist(),
            any4d_repo_root=any4d_repo_root_resolved,
            output_depth_npz=None,
            checkpoint_path=any4d_ckpt_path,
            resolution_set=res_set,
            use_amp=use_amp,
            model=_any4d_model,
        )
        for i in range(int(depth_stack.shape[0])):
            pred_depth = cv2.resize(depth_stack[i].astype(np.float32), (W, H))
            pred_depths.append(pred_depth)

    os.makedirs(slam_dir, exist_ok=True)
    depths_arr = np.stack(
        [np.asarray(d, dtype=np.float32) for d in pred_depths], axis=0
    )
    np.savez(
        merged_npz,
        depths=depths_arr,
        frame_indices=np.asarray(frame_ids, dtype=np.int64),
    )

    return pred_depths


def _predict_depths_batched(
    *,
    depth_backend: str,
    frame_source,
    tstamp_metric: np.ndarray,
    kf_idx: np.ndarray,
    metric3d_batch_size: int,
    calib,
    metric_runner,
    video_path: str,
    seq_folder: str,
    start_idx: int,
    end_idx: int,
    cache_tag: str,
    project_root: str,
    any4d_repo_root: Optional[str],
    any4d_checkpoint_path: Optional[str],
    any4d_resolution_set: Optional[int],
    any4d_use_amp: bool,
    W: int,
    H: int,
):
    """
    Metric3D or Any4D depth for each SLAM keyframe (frame indices in ``tstamp_metric``).

    ``cache_tag`` should be ``droid`` or ``dpvo`` so Any4D batch caches do not collide.
    """
    tstamp_metric = np.asarray(tstamp_metric, dtype=np.int64).reshape(-1)
    kf_idx = np.asarray(kf_idx, dtype=np.int64).reshape(-1)
    if tstamp_metric.shape[0] != kf_idx.shape[0]:
        raise ValueError(
            f"tstamp_metric len {tstamp_metric.shape[0]} != kf_idx len {kf_idx.shape[0]}"
        )
    frame_ids = tstamp_metric[kf_idx]
    return _predict_depths_for_frames(
        frame_ids,
        depth_backend=depth_backend,
        frame_source=frame_source,
        metric3d_batch_size=metric3d_batch_size,
        calib=calib,
        metric_runner=metric_runner,
        video_path=video_path,
        seq_folder=seq_folder,
        start_idx=start_idx,
        end_idx=end_idx,
        cache_tag=cache_tag,
        project_root=project_root,
        any4d_repo_root=any4d_repo_root,
        any4d_checkpoint_path=any4d_checkpoint_path,
        any4d_resolution_set=any4d_resolution_set,
        any4d_use_amp=any4d_use_amp,
        W=W,
        H=H,
        any4d_cache_suffix="",
    )


def _effective_depth_backend(depth_backend: Optional[str]) -> str:
    if depth_backend is not None:
        s = str(depth_backend).lower().strip()
    else:
        s = os.environ.get("HAWOR_DEPTH_BACKEND", "metric3d").lower().strip()
    if s not in ("metric3d", "any4d"):
        raise ValueError(f"Unknown depth backend: {s}")
    return s


def hawor_slam(
    args,
    start_idx,
    end_idx,
    metric_runner=None,
    metric3d_batch_size=48,
    droid_net=None,
    slam_backend: str = "droid",
    depth_backend: Optional[str] = None,
    any4d_repo_root: Optional[str] = None,
    any4d_checkpoint_path: Optional[str] = None,
    any4d_resolution_set: Optional[int] = None,
    any4d_use_amp: bool = False,
    depth_predict_all_frames: Optional[bool] = None,
):
    import time
    timing = {}
    t_start = time.time()

    video_path = args.video_path
    video_root = os.path.dirname(video_path)
    video = os.path.basename(video_path).split('.')[0]
    seq_folder = os.path.join(video_root, video)
    os.makedirs(seq_folder, exist_ok=True)
    video_folder = os.path.join(video_root, video)

    frame_source = build_frame_source(video_path)
    metric3d_batch_size = int(os.environ.get("HAWOR_METRIC3D_BATCH_SIZE", metric3d_batch_size))
    do_dense_depth = _depth_predict_all_frames_enabled(depth_predict_all_frames)

    first_img = frame_source.get_frame(0, rgb=False)
    height, width, _ = first_img.shape

    vprint(f'Running slam on {video_folder} ...')

    ##### Load masks #####
    t0 = time.time()
    masks = np.load(f'{video_folder}/tracks_{start_idx}_{end_idx}/model_masks.npy', allow_pickle=True)
    masks = torch.from_numpy(masks)
    vprint(masks.shape)

    focal = args.img_focal
    if focal is None:
        try:
            est_focal_path = os.path.join(video_folder, 'est_focal.txt')
            with open(est_focal_path, 'r') as f_est:
                focal = float(f_est.read())
        except Exception:
            vprint('No focal length provided')
            focal = 600
            with open(os.path.join(video_folder, 'est_focal.txt'), 'w') as f_est_out:
                f_est_out.write(str(focal))
    calib = np.array(est_calib(frame_source))
    center = calib[2:]
    calib[:2] = focal
    timing['1_load_masks'] = time.time() - t0

    # -------------------------------------------------------------------------
    # DROID 分支（深度可选 metric3d / any4d，与 CLI --depth_backend 一致）
    # -------------------------------------------------------------------------
    if slam_backend == "droid":
        # Lazy import: `masked_droid_slam` pulls in `lietorch/droid` which can
        # break at import-time if the environment is mismatched.
        from lib.pipeline.masked_droid_slam import run_slam

        t0 = time.time()
        print("Running DROID SLAM...")
        droid, traj = run_slam(frame_source, masks=masks, calib=calib, droid_net=droid_net)
        n = droid.video.counter.value
        tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
        disps = droid.video.disps_up.cpu().numpy()[:n]
        vprint('DBA errors:', droid.backend.errors)

        del droid
        torch.cuda.empty_cache()
        timing['2_droid_slam'] = time.time() - t0

        t0 = time.time()
        min_threshold = 0.4
        max_threshold = 0.7

        _db = _effective_depth_backend(depth_backend)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        H, W = get_dimention(frame_source)
        n = len(tstamp)
        kf_idx = np.arange(n, dtype=np.int64)
        tstamp_metric = np.asarray(tstamp, dtype=np.int64).reshape(-1)

        if do_dense_depth:
            fr_len = len(frame_source)
            seg_ids = _segment_frame_ids(start_idx, end_idx, fr_len)
            if seg_ids.size == 0:
                raise ValueError(
                    "dense depth: empty frame id range — check start_idx/end_idx vs len(frame_source)"
                )
            vprint(
                f"Predicting depth ({_db}) for ALL segment frames "
                f"[{int(seg_ids[0])}..{int(seg_ids[-1])}] (n={len(seg_ids)}) ..."
            )
            dense_list = _predict_depths_for_frames(
                seg_ids,
                depth_backend=_db,
                frame_source=frame_source,
                metric3d_batch_size=metric3d_batch_size,
                calib=calib,
                metric_runner=metric_runner,
                video_path=video_path,
                seq_folder=seq_folder,
                start_idx=start_idx,
                end_idx=end_idx,
                cache_tag="droid",
                project_root=project_root,
                any4d_repo_root=any4d_repo_root,
                any4d_checkpoint_path=any4d_checkpoint_path,
                any4d_resolution_set=any4d_resolution_set,
                any4d_use_amp=any4d_use_amp,
                W=W,
                H=H,
                any4d_cache_suffix="_allframes",
            )
            dense_npz = os.path.join(
                seq_folder, "SLAM", f"dense_depth_{_db}_{start_idx}_{end_idx}.npz"
            )
            _save_dense_depth_uint16_npz(dense_npz, seg_ids, dense_list, _db)
            vprint(f"Saved dense depth (uint16): {dense_npz}")
            pred_depths = _gather_keyframe_depths_from_dense(
                dense_list, seg_ids, tstamp_metric
            )
        else:
            vprint(f"Predicting depth ({_db}) for DROID keyframes ...")
            pred_depths = _predict_depths_batched(
                depth_backend=_db,
                frame_source=frame_source,
                tstamp_metric=tstamp_metric,
                kf_idx=kf_idx,
                metric3d_batch_size=metric3d_batch_size,
                calib=calib,
                metric_runner=metric_runner,
                video_path=video_path,
                seq_folder=seq_folder,
                start_idx=start_idx,
                end_idx=end_idx,
                cache_tag="droid",
                project_root=project_root,
                any4d_repo_root=any4d_repo_root,
                any4d_checkpoint_path=any4d_checkpoint_path,
                any4d_resolution_set=any4d_resolution_set,
                any4d_use_amp=any4d_use_amp,
                W=W,
                H=H,
            )
        timing['3_metric3d'] = time.time() - t0

        t0 = time.time()
        vprint('Estimating Metric Scale ...')
        slam_depth_list = [1.0 / disps[i] for i in range(n)]
        mask_list = [masks[tstamp[i]].numpy().astype(np.uint8) for i in range(n)]

        scales_ = est_scale_hybrid_batch(
            slam_depth_list, pred_depths, sigma=0.5,
            masks=mask_list, near_thresh=min_threshold, far_thresh=max_threshold)

        for i in range(n):
            if math.isnan(scales_[i]):
                nt, ft = min_threshold, max_threshold
                while math.isnan(scales_[i]):
                    nt -= 0.1
                    ft += 0.1
                    scales_[i] = est_scale_hybrid(
                        slam_depth_list[i], pred_depths[i], sigma=0.5,
                        msk=mask_list[i], near_thresh=nt, far_thresh=ft)

        median_s = np.median(scales_)
        vprint(f"estimated scale: {median_s}")
        timing['4_scale_est'] = time.time() - t0

        t0 = time.time()
        os.makedirs(f"{seq_folder}/SLAM", exist_ok=True)
        save_path = f'{seq_folder}/SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz'
        np.savez(
            save_path,
            tstamp=tstamp, disps=disps, traj=traj,
            img_focal=focal, img_center=calib[-2:],
            scale=median_s,
        )
        with open(os.path.join(seq_folder, "SLAM", "slam_backend.txt"), "w") as bf:
            bf.write("droid\n")
        timing['5_save'] = time.time() - t0

        total_time = time.time() - t_start
        timing['total'] = total_time
        video_name = os.path.basename(args.video_path)
        print(f"\n{'='*60}")
        print(f"SLAM Stage Timing for {video_name}")
        print(f"{'='*60}")
        for key in ['1_load_masks', '2_droid_slam', '3_metric3d', '4_scale_est', '5_save']:
            t = timing.get(key, 0)
            pct = t / total_time * 100 if total_time > 0 else 0
            print(f"  {key:20s}: {t:7.2f}s ({pct:5.1f}%)")
        print(f"  {'total':20s}: {total_time:7.2f}s")
        print(f"  {'keyframes':20s}: {n}")
        print(f"{'='*60}\n")
        return

    # -------------------------------------------------------------------------
    # DPVO 分支
    # -------------------------------------------------------------------------
    if slam_backend != "dpvo":
        raise ValueError(f"Unknown SLAM backend: {slam_backend}")

    t0 = time.time()
    try:
        import torch as _torch_gc
        _torch_gc.cuda.empty_cache()
    except Exception:
        pass

    print("Running DPVO SLAM...")
    dpvo_npz_path = os.path.join(
        seq_folder, "SLAM", f"dpvo_raw_{start_idx}_{end_idx}.npz"
    )
    os.makedirs(os.path.dirname(dpvo_npz_path), exist_ok=True)

    if os.environ.get("HAWOR_DPVO_FORCE_RERUN", "0") == "1" and os.path.exists(
        dpvo_npz_path
    ):
        os.remove(dpvo_npz_path)
        vprint("HAWOR_DPVO_FORCE_RERUN=1: removed cached dpvo_raw, will rerun DPVO.")

    # 与 DROID 一致：在当前进程内跑 VO（lazy import，避免 droid 路径加载 DPVO）
    dpvo_ran_fresh = not os.path.exists(dpvo_npz_path)
    if dpvo_ran_fresh:
        from lib.pipeline.dpvo_slam import run_dpvo_slam

        t_vo0 = time.time()
        traj, disps, tstamp, tstamp_disps = run_dpvo_slam(
            frame_source, masks=masks, calib=calib
        )
        dpvo_vo_sec = time.time() - t_vo0
        _dpvo_sec = np.array([dpvo_vo_sec], dtype=np.float64)
        np.savez(
            dpvo_npz_path,
            tstamp=tstamp,
            disps=disps,
            traj=traj,
            tstamp_disps=tstamp_disps,
            dpvo_vo_wall_sec=_dpvo_sec,
            dpvo_subprocess_sec=_dpvo_sec,
        )
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    dpvo_data = np.load(dpvo_npz_path)
    dpvo_cached_vo_sec = None
    if not dpvo_ran_fresh:
        if "dpvo_vo_wall_sec" in dpvo_data.files:
            dpvo_cached_vo_sec = float(
                np.asarray(dpvo_data["dpvo_vo_wall_sec"]).reshape(-1)[0]
            )
        elif "dpvo_subprocess_sec" in dpvo_data.files:
            dpvo_cached_vo_sec = float(
                np.asarray(dpvo_data["dpvo_subprocess_sec"]).reshape(-1)[0]
            )
    traj = dpvo_data["traj"]
    disps = dpvo_data["disps"]
    tstamp = dpvo_data["tstamp"].astype(np.int32)
    dpvo_used_cache = not dpvo_ran_fresh

    t_slam_block = time.time() - t0
    if dpvo_ran_fresh:
        timing["2_droid_slam"] = t_slam_block
    elif dpvo_cached_vo_sec is not None:
        timing["2_droid_slam"] = dpvo_cached_vo_sec
    else:
        timing["2_droid_slam"] = t_slam_block
        if dpvo_used_cache:
            vprint(
                "DPVO 缓存缺少 dpvo_vo_wall_sec / dpvo_subprocess_sec，2_droid_slam 仅为本段墙钟。"
            )

    t0 = time.time()
    _db = _effective_depth_backend(depth_backend)
    min_threshold = 0.4
    max_threshold = 0.7

    H, W = get_dimention(frame_source)

    tstamp_full = np.asarray(tstamp, dtype=np.int64)
    disps_full = np.asarray(disps)
    traj_full = np.asarray(traj)

    # Use DPVO internal keyframes for depth nets (no interpolation):
    if "tstamp_disps" in dpvo_data.files:
        tstamp_metric = np.asarray(dpvo_data["tstamp_disps"], dtype=np.int64).reshape(-1)
        disps_metric = disps_full
        if tstamp_metric.shape[0] != disps_metric.shape[0]:
            raise ValueError(
                f"DPVO mismatch: tstamp_disps={tstamp_metric.shape[0]} vs disps={disps_metric.shape[0]}"
            )

        order = np.argsort(tstamp_full)
        tstamp_sorted = tstamp_full[order]
        traj_sorted = traj_full[order]
        idx_sorted = np.searchsorted(tstamp_sorted, tstamp_metric)
        if (idx_sorted >= tstamp_sorted.shape[0]).any() or not np.all(
            tstamp_sorted[idx_sorted] == tstamp_metric
        ):
            raise ValueError("DPVO tstamp_disps contains timestamps missing from traj/tstamp")
        traj_metric = traj_sorted[idx_sorted]
    else:
        n_save = min(len(tstamp_full), len(disps_full), traj_full.shape[0])
        tstamp_metric = tstamp_full[:n_save]
        disps_metric = disps_full[:n_save]
        traj_metric = traj_full[:n_save]

    n_save = int(disps_metric.shape[0])
    kf_idx = np.arange(n_save, dtype=np.int64)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if do_dense_depth:
        fr_len = len(frame_source)
        seg_ids = _segment_frame_ids(start_idx, end_idx, fr_len)
        if seg_ids.size == 0:
            raise ValueError(
                "dense depth: empty frame id range — check start_idx/end_idx vs len(frame_source)"
            )
        vprint(
            f"Predicting depth ({_db}) for ALL segment frames "
            f"[{int(seg_ids[0])}..{int(seg_ids[-1])}] (n={len(seg_ids)}) ..."
        )
        dense_list = _predict_depths_for_frames(
            seg_ids,
            depth_backend=_db,
            frame_source=frame_source,
            metric3d_batch_size=metric3d_batch_size,
            calib=calib,
            metric_runner=metric_runner,
            video_path=video_path,
            seq_folder=seq_folder,
            start_idx=start_idx,
            end_idx=end_idx,
            cache_tag="dpvo",
            project_root=project_root,
            any4d_repo_root=any4d_repo_root,
            any4d_checkpoint_path=any4d_checkpoint_path,
            any4d_resolution_set=any4d_resolution_set,
            any4d_use_amp=any4d_use_amp,
            W=W,
            H=H,
            any4d_cache_suffix="_allframes",
        )
        dense_npz = os.path.join(
            seq_folder, "SLAM", f"dense_depth_{_db}_{start_idx}_{end_idx}.npz"
        )
        _save_dense_depth_uint16_npz(dense_npz, seg_ids, dense_list, _db)
        vprint(f"Saved dense depth (uint16): {dense_npz}")
        pred_depths = _gather_keyframe_depths_from_dense(
            dense_list, seg_ids, tstamp_metric
        )
    else:
        vprint(f"Predicting depth ({_db}) for DPVO keyframes ...")
        pred_depths = _predict_depths_batched(
            depth_backend=_db,
            frame_source=frame_source,
            tstamp_metric=tstamp_metric,
            kf_idx=kf_idx,
            metric3d_batch_size=metric3d_batch_size,
            calib=calib,
            metric_runner=metric_runner,
            video_path=video_path,
            seq_folder=seq_folder,
            start_idx=start_idx,
            end_idx=end_idx,
            cache_tag="dpvo",
            project_root=project_root,
            any4d_repo_root=any4d_repo_root,
            any4d_checkpoint_path=any4d_checkpoint_path,
            any4d_resolution_set=any4d_resolution_set,
            any4d_use_amp=any4d_use_amp,
            W=W,
            H=H,
        )
    timing['3_metric3d'] = time.time() - t0

    t0 = time.time()
    vprint('Estimating Metric Scale ...')
    # Replace with DPVO keyframe-only depth evidence for scale estimation.
    tstamp = tstamp_metric
    disps = disps_metric
    traj = traj_metric

    slam_depth_list = [1.0 / disps[int(i)] for i in kf_idx]
    mask_list = [masks[int(tstamp_metric[i])].numpy().astype(np.uint8) for i in kf_idx]

    scales_ = est_scale_hybrid_batch(
        slam_depth_list, pred_depths, sigma=0.5,
        masks=mask_list, near_thresh=min_threshold, far_thresh=max_threshold)

    for i in range(len(scales_)):
        if math.isnan(scales_[i]):
            nt, ft = min_threshold, max_threshold
            while math.isnan(scales_[i]):
                nt -= 0.1
                ft += 0.1
                scales_[i] = est_scale_hybrid(
                    slam_depth_list[i], pred_depths[i], sigma=0.5,
                    msk=mask_list[i], near_thresh=nt, far_thresh=ft)

    median_s = np.median(scales_)
    vprint(f"estimated scale: {median_s}")
    timing['4_scale_est'] = time.time() - t0

    t0 = time.time()
    os.makedirs(f"{seq_folder}/SLAM", exist_ok=True)
    save_path = f'{seq_folder}/SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz'
    np.savez(
        save_path,
        tstamp=tstamp, disps=disps, traj=traj,
        img_focal=focal, img_center=calib[-2:],
        scale=median_s,
    )
    with open(os.path.join(seq_folder, "SLAM", "slam_backend.txt"), "w") as bf:
        bf.write("dpvo\n")
    timing['5_save'] = time.time() - t0

    stage_keys = ["1_load_masks", "2_droid_slam", "3_metric3d", "4_scale_est", "5_save"]
    breakdown_total = sum(timing.get(k, 0) for k in stage_keys)
    wall_total = time.time() - t_start
    timing["total"] = breakdown_total
    video_name = os.path.basename(args.video_path)
    print(f"\n{'='*60}")
    print(f"SLAM Stage Timing for {video_name}")
    print(f"{'='*60}")
    for key in stage_keys:
        t = timing.get(key, 0)
        pct = t / breakdown_total * 100 if breakdown_total > 0 else 0
        print(f"  {key:20s}: {t:7.2f}s ({pct:5.1f}%)")
    print(f"  {'total (阶段合计)':20s}: {breakdown_total:7.2f}s")
    if dpvo_used_cache and wall_total + 1e-3 < breakdown_total:
        print(
            f"  {'wall_clock (本进程)':20s}: {wall_total:7.2f}s "
            f"（DPVO 命中缓存时阶段合计含已记录的 VO 耗时）"
        )
    print(f"  {'keyframes':20s}: {n_save}")
    if len(kf_idx) != n_save:
        print(f"  {'Metric3D subsampled':20s}: {len(kf_idx)}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default='')
    parser.add_argument("--input_type", type=str, default='file')
    parser.add_argument(
        "--depth_backend",
        type=str,
        default=None,
        choices=["metric3d", "any4d"],
        help="Depth for scale (DROID/DPVO): metric3d (default) or any4d; env HAWOR_DEPTH_BACKEND if omitted",
    )
    parser.add_argument(
        "--any4d",
        action="store_true",
        help="Same as --depth_backend any4d",
    )
    args = parser.parse_args()

    depth_backend_kw = None
    if args.any4d:
        depth_backend_kw = "any4d"
    elif args.depth_backend is not None:
        depth_backend_kw = args.depth_backend

    from scripts.scripts_test_video.detect_track_video import detect_track_video
    start_idx, end_idx, _, _ = detect_track_video(args)
    hawor_slam(args, start_idx, end_idx, depth_backend=depth_backend_kw)
