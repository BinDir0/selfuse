import argparse
import hashlib
import math
import os
import shutil
import sys
import time
import zipfile
import zlib
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.any4d_depth import (
    build_any4d_runner,
    build_any4d_views,
    iter_any4d_depth_sequence_batches,
)
from lib.pipeline.errors import CorruptStageDataError
from lib.pipeline.dpvo_slam import run_dpvo_slam
from lib.pipeline.est_scale import est_scale_hybrid, est_scale_hybrid_batch
from lib.pipeline.frame_source import ImageFolderFrameSource, build_frame_source
from lib.pipeline.slam_geom_utils import est_calib, get_dimention


QUIET_MODE = os.environ.get("HAWOR_QUIET", "0") == "1"
CORRUPT_STAGE_ERROR_TOKENS = (
    "bad crc-32",
    "invalid block type",
    "failed to decode image from tar",
    "failed to read image",
    "failed to write stage3 frame cache image",
    "failed to decode",
    "no such file or directory",
    "truncated",
    "unexpected end of data",
    "cannot identify image file",
    "failed to write stage3 frame cache file",
)


def vprint(*args, **kwargs):
    if not QUIET_MODE:
        print(*args, **kwargs)


def _resolve_seq_folder(video_path: str, seq_folder: str = None) -> str:
    if seq_folder is not None:
        return seq_folder
    video_path_obj = Path(video_path)
    return str(video_path_obj.parent / video_path_obj.stem)


def _resolve_frame_source(video_path: str, frame_source=None):
    return frame_source or build_frame_source(video_path)


def _load_masks(seq_folder: str, start_idx: int, end_idx: int) -> torch.Tensor:
    masks_path = os.path.join(seq_folder, f"tracks_{start_idx}_{end_idx}", "model_masks.npy")
    try:
        return torch.from_numpy(np.load(masks_path, allow_pickle=True))
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, zlib.error) as error:
        raise CorruptStageDataError(f"Corrupt masks file: {masks_path} ({error})") from error


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


def _depth_predict_all_frames_enabled(explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return bool(explicit)
    value = os.environ.get("HAWOR_DEPTH_PREDICT_ALL_FRAMES", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def _resolve_any4d_batch_size(default_batch_size: int) -> int:
    return int(os.environ.get("HAWOR_ANY4D_BATCH_SIZE", default_batch_size))


def _resolve_stage3_tmp_root(args) -> str:
    tmp_root = (
        getattr(args, "stage3_tmp_root", None)
        or os.environ.get("HAWOR_STAGE3_TMP_ROOT")
        or os.environ.get("HAWOR_BATCH_TMPDIR")
        or "/DATA/guantianrui/tmp"
    )
    tmp_root = os.path.abspath(os.path.expanduser(tmp_root))
    if not os.path.isdir(tmp_root):
        raise FileNotFoundError(f"Stage3 tmp root does not exist: {tmp_root}")
    if not os.access(tmp_root, os.W_OK | os.X_OK):
        raise PermissionError(f"Stage3 tmp root is not writable: {tmp_root}")
    return tmp_root


def _keep_stage3_tmp() -> bool:
    value = os.environ.get("HAWOR_STAGE3_KEEP_TMP", "").strip().lower()
    return value in ("1", "true", "yes", "y", "on")


def _direct_frame_path(frame_source, frame_idx: int):
    image_paths = getattr(frame_source, "image_paths", None)
    if image_paths is None:
        return None
    if frame_idx < 0 or frame_idx >= len(image_paths):
        return None
    path = image_paths[frame_idx]
    return path if os.path.exists(path) else None


def _raw_frame_bytes(frame_source, frame_idx: int):
    getter = getattr(frame_source, "get_frame_bytes", None)
    if not callable(getter):
        return None
    return getter(frame_idx)


def _frame_output_extension(frame_source, frame_idx: int):
    image_paths = getattr(frame_source, "image_paths", None)
    if image_paths is not None and 0 <= frame_idx < len(image_paths):
        suffix = Path(image_paths[frame_idx]).suffix.lower()
        if suffix:
            return suffix

    frame_names = getattr(frame_source, "frame_names", None)
    if frame_names is not None and 0 <= frame_idx < len(frame_names):
        suffix = Path(frame_names[frame_idx]).suffix.lower()
        if suffix:
            return suffix

    return ".png"


def _stage3_frame_cache_dir(tmp_root: str, seq_folder: str, start_idx: int, end_idx: int) -> str:
    seq_hash = hashlib.sha1(os.path.abspath(seq_folder).encode("utf-8")).hexdigest()[:12]
    seq_name = Path(seq_folder).name
    return os.path.join(tmp_root, "hawor_stage3_frames", f"{seq_name}_{seq_hash}_{start_idx}_{end_idx}")


def _stage3_frame_cache_marker(cache_dir: str) -> str:
    return os.path.join(cache_dir, ".ready")


def _build_stage3_workspace(frame_source, frame_ids: np.ndarray, seq_folder: str, start_idx: int, end_idx: int, tmp_root: str):
    frame_id_list = [int(frame_id) for frame_id in np.asarray(frame_ids, dtype=np.int64).tolist()]
    direct_paths = {}
    use_direct_paths = True
    for frame_id in frame_id_list:
        path = _direct_frame_path(frame_source, frame_id)
        if path is None:
            use_direct_paths = False
            break
        direct_paths[frame_id] = path
    if use_direct_paths:
        ordered_paths = [direct_paths[frame_id] for frame_id in frame_id_list]
        force_stable_decode = bool(getattr(frame_source, "force_stage3_stable_decode", False))
        return {
            "frame_path_map": direct_paths,
            "frame_source": ImageFolderFrameSource(ordered_paths, use_turbojpeg=not force_stable_decode),
            "workspace_dir": None,
            "ready_marker": None,
            "materialized": False,
        }

    cache_dir = _stage3_frame_cache_dir(tmp_root, seq_folder, start_idx, end_idx)
    ready_marker = _stage3_frame_cache_marker(cache_dir)
    expected_paths = {
        frame_id: os.path.join(cache_dir, f"{frame_id:06d}{_frame_output_extension(frame_source, frame_id)}")
        for frame_id in frame_id_list
    }

    if os.path.isfile(ready_marker):
        if all(os.path.isfile(path) for path in expected_paths.values()):
            ordered_paths = [expected_paths[frame_id] for frame_id in frame_id_list]
            return {
                "frame_path_map": expected_paths,
                "frame_source": ImageFolderFrameSource(ordered_paths, use_turbojpeg=False),
                "workspace_dir": cache_dir,
                "ready_marker": ready_marker,
                "materialized": True,
            }
        try:
            os.remove(ready_marker)
        except OSError:
            pass

    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

    for frame_id in frame_id_list:
        out_path = expected_paths[frame_id]
        try:
            payload = _raw_frame_bytes(frame_source, frame_id)
            if payload is not None:
                with open(out_path, "wb") as handle:
                    handle.write(payload)
                continue

            image = frame_source.get_frame(frame_id, rgb=False)
        except Exception as error:
            raise CorruptStageDataError(
                f"Failed to materialize stage3 frame {frame_id} for {seq_folder}: {error}"
            ) from error
        if not cv2.imwrite(out_path, image):
            raise CorruptStageDataError(f"Failed to write stage3 frame cache file: {out_path}")

    Path(ready_marker).touch()
    ordered_paths = [expected_paths[frame_id] for frame_id in frame_id_list]
    return {
        "frame_path_map": expected_paths,
        "frame_source": ImageFolderFrameSource(ordered_paths, use_turbojpeg=False),
        "workspace_dir": cache_dir,
        "ready_marker": ready_marker,
        "materialized": True,
    }


def _cleanup_stage3_workspace(workspace: dict, *, success: bool):
    workspace_dir = workspace.get("workspace_dir")
    ready_marker = workspace.get("ready_marker")
    if not workspace_dir or not os.path.isdir(workspace_dir):
        return
    if success:
        if _keep_stage3_tmp():
            return
        shutil.rmtree(workspace_dir, ignore_errors=True)
        return
    if ready_marker is None or not os.path.isfile(ready_marker):
        shutil.rmtree(workspace_dir, ignore_errors=True)


def _dpvo_cache_path(seq_folder: str, start_idx: int, end_idx: int) -> str:
    return os.path.join(seq_folder, "SLAM", f"dpvo_raw_{start_idx}_{end_idx}.npz")


def _run_dpvo_with_cache(frame_source, masks, calib, seq_folder: str, start_idx: int, end_idx: int, frame_indices: Optional[np.ndarray] = None):
    cache_path = _dpvo_cache_path(seq_folder, start_idx, end_idx)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.environ.get("HAWOR_DPVO_FORCE_RERUN", "0") == "1" and os.path.exists(cache_path):
        os.remove(cache_path)
        vprint("HAWOR_DPVO_FORCE_RERUN=1: removed cached dpvo_raw, will rerun DPVO.")

    ran_fresh = not os.path.exists(cache_path)
    if ran_fresh:
        t0 = time.time()
        traj, disps, traj_tstamp, disp_tstamp = run_dpvo_slam(frame_source, masks=masks, calib=calib, frame_indices=frame_indices)
        dpvo_vo_sec = time.time() - t0
        wall_sec = np.array([dpvo_vo_sec], dtype=np.float64)
        np.savez(
            cache_path,
            tstamp=np.asarray(traj_tstamp, dtype=np.int32),
            disps=np.asarray(disps, dtype=np.float32),
            traj=np.asarray(traj, dtype=np.float32),
            tstamp_disps=np.asarray(disp_tstamp, dtype=np.int32),
            dpvo_vo_wall_sec=wall_sec,
            dpvo_subprocess_sec=wall_sec,
        )
        torch.cuda.empty_cache()

    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            traj_full = cached["traj"].astype(np.float32)
            disps_full = cached["disps"].astype(np.float32)
            tstamp_full = cached["tstamp"].astype(np.int64).reshape(-1)
            tstamp_disps = cached["tstamp_disps"].astype(np.int64).reshape(-1) if "tstamp_disps" in cached.files else None
            cached_vo_sec = None
            if "dpvo_vo_wall_sec" in cached.files:
                cached_vo_sec = float(np.asarray(cached["dpvo_vo_wall_sec"]).reshape(-1)[0])
            elif "dpvo_subprocess_sec" in cached.files:
                cached_vo_sec = float(np.asarray(cached["dpvo_subprocess_sec"]).reshape(-1)[0])
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, zlib.error) as error:
        _drop_corrupt_cache(cache_path, error)
        return _run_dpvo_with_cache(frame_source, masks, calib, seq_folder, start_idx, end_idx, frame_indices=frame_indices)

    if tstamp_disps is not None and tstamp_disps.shape[0] == disps_full.shape[0]:
        order = np.argsort(tstamp_full)
        tstamp_sorted = tstamp_full[order]
        traj_sorted = traj_full[order]
        idx_sorted = np.searchsorted(tstamp_sorted, tstamp_disps)
        if (idx_sorted >= tstamp_sorted.shape[0]).any() or not np.all(tstamp_sorted[idx_sorted] == tstamp_disps):
            raise ValueError("DPVO tstamp_disps contains timestamps missing from traj/tstamp")
        tstamp_metric = tstamp_disps.astype(np.int32)
        traj_metric = traj_sorted[idx_sorted].astype(np.float32)
        disps_metric = disps_full.astype(np.float32)
    else:
        n_save = min(len(tstamp_full), len(disps_full), traj_full.shape[0])
        tstamp_metric = tstamp_full[:n_save].astype(np.int32)
        traj_metric = traj_full[:n_save].astype(np.float32)
        disps_metric = disps_full[:n_save].astype(np.float32)

    return {
        "traj": traj_metric,
        "tstamp": tstamp_metric,
        "disps": disps_metric,
        "used_cache": not ran_fresh,
        "cache_path": cache_path,
        "cached_vo_sec": cached_vo_sec,
    }


def _segment_frame_ids(start_idx: int, end_idx: int, num_frames: int) -> np.ndarray:
    frame_ids = np.arange(int(start_idx), int(end_idx), dtype=np.int64)
    return frame_ids[(frame_ids >= 0) & (frame_ids < int(num_frames))]


def _dense_depth_cache_path(seq_folder: str, start_idx: int, end_idx: int) -> str:
    return os.path.join(seq_folder, "SLAM", f"dense_depth_any4d_{start_idx}_{end_idx}.npz")


def _legacy_dense_depth_cache_paths(seq_folder: str, start_idx: int, end_idx: int):
    return [
        os.path.join(seq_folder, "SLAM", f"dense_depth_any4d_all_{start_idx}_{end_idx}.npz"),
        os.path.join(seq_folder, "SLAM", f"dense_depth_any4d_keyframes_{start_idx}_{end_idx}.npz"),
    ]


def _any4d_cache_path(seq_folder: str, start_idx: int, end_idx: int, suffix: str = "") -> str:
    return os.path.join(seq_folder, "SLAM", f"any4d_depth_dpvo_{start_idx}_{end_idx}{suffix}.npz")


def _save_dense_depth_uint16_npz(out_path: str, frame_indices, depths):
    depth_stack = np.asarray(depths, dtype=np.float32)
    depth_stack = np.nan_to_num(depth_stack, nan=0.0, posinf=0.0, neginf=0.0)
    depth_stack = np.clip(depth_stack, 0.0, None)
    depth_mm = np.clip(np.round(depth_stack * 1000.0), 0.0, 65535.0).astype(np.uint16)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(
        out_path,
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        depths_uint16=depth_mm,
        height=np.int32(depth_stack.shape[1]),
        width=np.int32(depth_stack.shape[2]),
    )


def _drop_corrupt_cache(cache_path: str, error: Exception):
    vprint(f"Corrupt cache ignored: {cache_path} ({error})")
    try:
        os.remove(cache_path)
        vprint(f"Removed corrupt cache: {cache_path}")
    except OSError:
        pass


def _iter_exception_chain(error: Exception):
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = current.__cause__ or current.__context__


def _is_corrupt_stage_data_error(error: Exception) -> bool:
    for item in _iter_exception_chain(error):
        if isinstance(item, CorruptStageDataError):
            return True
        if isinstance(item, (EOFError, zipfile.BadZipFile, zlib.error, cv2.error)):
            return True
        message = str(item).strip().lower()
        if any(token in message for token in CORRUPT_STAGE_ERROR_TOKENS):
            return True
    return False


def _load_dense_depth_cache(cache_path: str):
    if not os.path.exists(cache_path):
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            frame_indices = cached["frame_indices"].astype(np.int64).reshape(-1)
            if "depths_uint16" in cached.files:
                depths = cached["depths_uint16"].astype(np.float32) * 1e-3
            elif "pred_depths" in cached.files:
                depths = cached["pred_depths"].astype(np.float32)
            else:
                return None
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, zlib.error) as error:
        _drop_corrupt_cache(cache_path, error)
        return None
    return frame_indices, depths


def _load_matching_dense_depth_cache(seq_folder: str, start_idx: int, end_idx: int, frame_ids: np.ndarray):
    candidate_paths = [_dense_depth_cache_path(seq_folder, start_idx, end_idx), *_legacy_dense_depth_cache_paths(seq_folder, start_idx, end_idx)]
    for cache_path in candidate_paths:
        cached = _load_dense_depth_cache(cache_path)
        if cached is None:
            continue
        cached_ids, cached_depths = cached
        if np.array_equal(cached_ids, frame_ids):
            return cached_ids, cached_depths, cache_path
    return None


def _load_matching_any4d_cache(cache_path: str, frame_ids: np.ndarray, output_hw):
    if not os.path.exists(cache_path):
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            if "depths" not in cached.files:
                return None
            if "frame_indices" in cached.files:
                cached_ids = cached["frame_indices"].astype(np.int64).reshape(-1)
                if not np.array_equal(cached_ids, frame_ids):
                    return None
            elif cached["depths"].shape[0] != frame_ids.shape[0]:
                return None
            depth_stack = cached["depths"].astype(np.float32)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, zlib.error) as error:
        _drop_corrupt_cache(cache_path, error)
        return None
    return _resize_depths(depth_stack, output_hw)


def _resize_depths(depth_batch: np.ndarray, output_hw):
    out_h, out_w = output_hw
    resized = [
        cv2.resize(depth.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        for depth in depth_batch
    ]
    return np.stack(resized, axis=0)


def _predict_any4d_depths_for_frames(
    frame_source,
    frame_ids: np.ndarray,
    *,
    any4d_runner,
    any4d_batch_size: int,
    output_hw,
    args,
    seq_folder: str,
    start_idx: int,
    end_idx: int,
    frame_path_map,
    any4d_cache_suffix: str = "",
    timing: dict | None = None,
):
    cache_path = _any4d_cache_path(seq_folder, start_idx, end_idx, suffix=any4d_cache_suffix)
    force = os.environ.get("HAWOR_ANY4D_FORCE_RERUN", "0") == "1"
    if force and os.path.isfile(cache_path):
        try:
            os.remove(cache_path)
        except OSError:
            pass

    if not force:
        t_cache_lookup = time.time()
        cached_depths = _load_matching_any4d_cache(cache_path, frame_ids, output_hw)
        if timing is not None:
            timing["3b_depth_cache_lookup"] = timing.get("3b_depth_cache_lookup", 0.0) + (time.time() - t_cache_lookup)
        if cached_depths is not None:
            return cached_depths, cache_path, True

    pred_depths = np.empty((len(frame_ids),) + tuple(output_hw), dtype=np.float32)
    desc = "Any4D batches (all frames)" if any4d_cache_suffix else "Any4D batches"

    def _prepare_views(batch_indices, ref_frame_idx):
        batch_image_paths = [frame_path_map[int(ref_frame_idx)], *[frame_path_map[int(frame_idx)] for frame_idx in batch_indices]]
        return build_any4d_views(
            frame_source,
            list(batch_indices),
            runner=any4d_runner,
            any4d_repo_root=getattr(args, "any4d_repo_root", None),
            checkpoint_path=getattr(args, "any4d_checkpoint_path", None),
            resolution_set=getattr(args, "any4d_resolution_set", None),
            use_amp=getattr(args, "any4d_use_amp", None),
            image_paths=batch_image_paths,
        )

    def _record_any4d_timing(name: str, elapsed: float) -> None:
        if timing is None:
            return
        if name == "view_prep":
            key = "3c_any4d_view_prep"
        elif name == "forward":
            key = "3d_any4d_forward"
        else:
            key = f"3_any4d_{name}"
        timing[key] = timing.get(key, 0.0) + float(elapsed)

    for batch_result in iter_any4d_depth_sequence_batches(
        frame_ids.tolist(),
        any4d_batch_size=any4d_batch_size,
        build_views_for_chunk=_prepare_views,
        runner=any4d_runner,
        progress_desc=desc,
        progress_disable=QUIET_MODE,
        timing_callback=_record_any4d_timing,
        prediction_view_offset=2,
    ):
        batch_start = int(batch_result["batch_start"])
        batch_indices = list(batch_result["batch_indices"])
        batch_size = len(batch_indices)
        t_resize = time.time()
        pred_depths[batch_start : batch_start + batch_size] = _resize_depths(
            np.asarray(batch_result["depths"], dtype=np.float32),
            output_hw,
        )
        if timing is not None:
            timing["3e_any4d_resize"] = timing.get("3e_any4d_resize", 0.0) + (time.time() - t_resize)

    os.makedirs(os.path.join(seq_folder, "SLAM"), exist_ok=True)
    t_cache_save = time.time()
    np.savez(
        cache_path,
        depths=np.asarray(pred_depths, dtype=np.float32),
        frame_indices=np.asarray(frame_ids, dtype=np.int64),
    )
    if timing is not None:
        timing["3f_any4d_cache_save"] = timing.get("3f_any4d_cache_save", 0.0) + (time.time() - t_cache_save)
    return pred_depths, cache_path, False


def _gather_keyframe_depths_from_dense(dense_depths: np.ndarray, segment_frame_ids: np.ndarray, keyframe_tstamps: np.ndarray):
    index_by_frame = {int(frame_id): idx for idx, frame_id in enumerate(np.asarray(segment_frame_ids, dtype=np.int64).tolist())}
    gathered = []
    for frame_id in np.asarray(keyframe_tstamps, dtype=np.int64).tolist():
        if int(frame_id) not in index_by_frame:
            raise ValueError(
                f"SLAM keyframe frame id {int(frame_id)} is outside dense depth segment. "
                "Check detect-track frame range or disable full-frame depth."
            )
        gathered.append(dense_depths[index_by_frame[int(frame_id)]])
    return gathered


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
    slam_dir = os.path.join(seq_folder, "SLAM")
    os.makedirs(slam_dir, exist_ok=True)
    save_path = os.path.join(slam_dir, f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    np.savez(
        save_path,
        tstamp=np.asarray(tstamp, dtype=np.int32),
        disps=np.asarray(disps, dtype=np.float32),
        traj=np.asarray(traj, dtype=np.float32),
        img_focal=float(focal),
        img_center=np.asarray(calib[-2:], dtype=np.float32),
        scale=np.float32(scale),
    )
    with open(os.path.join(slam_dir, "slam_backend.txt"), "w", encoding="utf-8") as handle:
        handle.write("dpvo\n")
    return save_path


def _print_timing(video_path: str, timing: dict, num_keyframes: int, depth_frame_count: int, *, predict_all_frames: bool, used_depth_cache: bool):
    total_time = timing["total"]
    print(f"\n{'=' * 60}")
    print(f"SLAM Stage Timing for {os.path.basename(video_path)}")
    print(f"{'=' * 60}")
    summary_keys = (
        "0_stage3_workspace",
        "1_load_masks",
        "2_slam",
        "3a_any4d_init",
        "3_depth",
        "4_scale_est",
        "5_save",
    )
    for key in summary_keys:
        elapsed = float(timing.get(key, 0.0))
        pct = elapsed / total_time * 100 if total_time > 0 else 0
        print(f"  {key:20s}: {elapsed:7.2f}s ({pct:5.1f}%)")
    cached_slam_sec = timing.get("2_slam_cached_source")
    if cached_slam_sec is not None:
        print(f"  {'2_slam_cached_src':20s}: {cached_slam_sec:7.2f}s (metadata)")
    for key in (
        "3b_dense_depth_cache_lookup",
        "3b_depth_cache_lookup",
        "3c_any4d_view_prep",
        "3d_any4d_forward",
        "3e_any4d_resize",
        "3f_any4d_cache_save",
        "3g_dense_depth_cache_save",
        "3h_gather_keyframe_depths",
    ):
        if key not in timing:
            continue
        elapsed = float(timing[key])
        pct = elapsed / total_time * 100 if total_time > 0 else 0
        print(f"  {key:20s}: {elapsed:7.2f}s ({pct:5.1f}%)")
    for key in ("0_stage3_materialized", "0_stage3_frame_count"):
        if key in timing:
            print(f"  {key:20s}: {int(timing[key])}")
    print(f"  {'total':20s}: {total_time:7.2f}s")
    print(f"  {'slam_backend':20s}: dpvo")
    print(f"  {'depth_backend':20s}: any4d")
    print(f"  {'depth_scope':20s}: {'all_frames' if predict_all_frames else 'keyframes'}")
    print(f"  {'depth_cache_used':20s}: {used_depth_cache}")
    print(f"  {'keyframes':20s}: {num_keyframes}")
    print(f"  {'depth_frames':20s}: {depth_frame_count}")
    print(f"{'=' * 60}\n")


def hawor_slam(
    args,
    start_idx,
    end_idx,
    any4d_runner=None,
    any4d_batch_size=32,
    frame_source=None,
    seq_folder=None,
    return_timing=False,
):
    timing = {}
    start_time = time.time()
    success = False

    seq_folder = _resolve_seq_folder(args.video_path, seq_folder)
    os.makedirs(seq_folder, exist_ok=True)
    frame_source = _resolve_frame_source(args.video_path, frame_source)
    segment_frame_ids = _segment_frame_ids(start_idx, end_idx, len(frame_source))
    if segment_frame_ids.size == 0:
        raise ValueError("stage3: empty frame range after clipping to available frames")
    stage3_tmp_root = _resolve_stage3_tmp_root(args)
    t_workspace = time.time()
    workspace = _build_stage3_workspace(frame_source, segment_frame_ids, seq_folder, start_idx, end_idx, stage3_tmp_root)
    timing["0_stage3_workspace"] = time.time() - t_workspace
    stage3_frame_source = workspace["frame_source"]
    stage3_frame_path_map = workspace["frame_path_map"]
    timing["0_stage3_materialized"] = int(bool(workspace.get("materialized")))
    timing["0_stage3_frame_count"] = int(segment_frame_ids.shape[0])
    predict_all_frames = _depth_predict_all_frames_enabled(getattr(args, "depth_predict_all_frames", None))
    any4d_batch_size = _resolve_any4d_batch_size(any4d_batch_size)
    vprint(
        f"Running slam on {seq_folder} "
        f"(slam_backend=dpvo, depth_backend=any4d, "
        f"depth_scope={'all_frames' if predict_all_frames else 'keyframes'}) ..."
    )

    try:
        t0 = time.time()
        masks = _load_masks(seq_folder, start_idx, end_idx)
        focal = _resolve_focal(seq_folder, getattr(args, "img_focal", None))
        calib = _build_calibration(stage3_frame_source, focal)
        timing["1_load_masks"] = time.time() - t0

        t0 = time.time()
        slam_outputs = _run_dpvo_with_cache(
            stage3_frame_source,
            masks,
            calib,
            seq_folder,
            start_idx,
            end_idx,
            frame_indices=segment_frame_ids,
        )
        traj = slam_outputs["traj"]
        tstamp = slam_outputs["tstamp"]
        disps = slam_outputs["disps"]
        timing["2_slam"] = time.time() - t0
        if slam_outputs["used_cache"] and slam_outputs["cached_vo_sec"] is not None:
            timing["2_slam_cached_source"] = float(slam_outputs["cached_vo_sec"])

        output_hw = get_dimention(stage3_frame_source)
        depth_cache_used = False

        t0 = time.time()
        if any4d_runner is None:
            any4d_runner = build_any4d_runner(
                any4d_repo_root=getattr(args, "any4d_repo_root", None),
                checkpoint_path=getattr(args, "any4d_checkpoint_path", None),
                resolution_set=getattr(args, "any4d_resolution_set", None),
                use_amp=getattr(args, "any4d_use_amp", None),
            )
        timing["3a_any4d_init"] = time.time() - t0

        t0 = time.time()
        if predict_all_frames:
            frame_ids = segment_frame_ids

            force_any4d_rerun = os.environ.get("HAWOR_ANY4D_FORCE_RERUN", "0") == "1"
            if force_any4d_rerun:
                dense_cache_path = _dense_depth_cache_path(seq_folder, start_idx, end_idx)
                if os.path.exists(dense_cache_path):
                    try:
                        os.remove(dense_cache_path)
                    except OSError:
                        pass
            cached_dense = None if force_any4d_rerun else _load_matching_dense_depth_cache(seq_folder, start_idx, end_idx, frame_ids)
            timing["3b_dense_depth_cache_lookup"] = time.time() - t0
            if cached_dense is not None:
                depth_frame_indices, depth_predictions, depth_cache_path = cached_dense
                depth_cache_used = True
            else:
                depth_predictions, any4d_cache_path, used_any4d_cache = _predict_any4d_depths_for_frames(
                    stage3_frame_source,
                    frame_ids,
                    any4d_runner=any4d_runner,
                    any4d_batch_size=any4d_batch_size,
                    output_hw=output_hw,
                    args=args,
                    seq_folder=seq_folder,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    frame_path_map=stage3_frame_path_map,
                    any4d_cache_suffix="_allframes",
                    timing=timing,
                )
                depth_frame_indices = frame_ids.astype(np.int64)
                dense_cache_path = _dense_depth_cache_path(seq_folder, start_idx, end_idx)
                t_dense_save = time.time()
                _save_dense_depth_uint16_npz(dense_cache_path, depth_frame_indices, depth_predictions)
                timing["3g_dense_depth_cache_save"] = time.time() - t_dense_save
                depth_cache_path = any4d_cache_path if used_any4d_cache else dense_cache_path
                depth_cache_used = used_any4d_cache
            t_gather = time.time()
            keyframe_depths = _gather_keyframe_depths_from_dense(depth_predictions, depth_frame_indices, tstamp)
            timing["3h_gather_keyframe_depths"] = time.time() - t_gather
        else:
            depth_frame_indices = np.asarray(tstamp, dtype=np.int64)
            depth_predictions, depth_cache_path, depth_cache_used = _predict_any4d_depths_for_frames(
                stage3_frame_source,
                depth_frame_indices,
                any4d_runner=any4d_runner,
                any4d_batch_size=any4d_batch_size,
                output_hw=output_hw,
                args=args,
                seq_folder=seq_folder,
                start_idx=start_idx,
                end_idx=end_idx,
                frame_path_map=stage3_frame_path_map,
                any4d_cache_suffix="",
                timing=timing,
            )
            keyframe_depths = [depth_predictions[i] for i in range(len(depth_predictions))]

        if depth_cache_used:
            vprint(f"Loaded cached Any4D depth from {depth_cache_path}")
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
        success = True
    except Exception as error:
        if isinstance(error, CorruptStageDataError):
            raise
        if _is_corrupt_stage_data_error(error):
            raise CorruptStageDataError(f"Corrupt stage data for {seq_folder}: {error}") from error
        raise
    finally:
        _cleanup_stage3_workspace(workspace, success=success)

    timing["total"] = time.time() - start_time
    _print_timing(
        args.video_path,
        timing,
        len(tstamp),
        len(depth_frame_indices),
        predict_all_frames=predict_all_frames,
        used_depth_cache=depth_cache_used,
    )
    if return_timing:
        return timing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default="")
    parser.add_argument("--input_type", type=str, default="file")
    parser.add_argument("--any4d_batch_size", type=int, default=32)
    parser.add_argument(
        "--depth_predict_all_frames",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Predict dense depth for all frames; defaults to env HAWOR_DEPTH_PREDICT_ALL_FRAMES or on.",
    )
    parser.add_argument("--any4d_repo_root", type=str, default=None)
    parser.add_argument("--any4d_checkpoint_path", type=str, default=None)
    parser.add_argument("--any4d_resolution_set", type=int, default=None)
    parser.add_argument("--stage3_tmp_root", type=str, default=None)
    parser.add_argument(
        "--any4d_use_amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override Any4D AMP usage.",
    )
    args = parser.parse_args()

    from lib.pipeline.stages.detect_track import detect_track_video

    start_idx, end_idx, _, _ = detect_track_video(args)
    hawor_slam(args, start_idx, end_idx, any4d_batch_size=args.any4d_batch_size)
