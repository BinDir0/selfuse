import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.frame_source import build_frame_source
from lib.pipeline.tools import parse_chunks
from lib.eval_utils.custom_utils import interpolate_bboxes, validate_motion_velocity
from hawor.utils.process import get_mano_cfg, get_mano_faces, run_mano, run_mano_left
from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis

from .hawor_cache import (
    _get_motion_output_paths,
    _get_tracks_dir,
    _invalidate_cam_space_cache,
    _save_cam_space_json,
    _save_motion_outputs,
)
from .hawor_common import MANO_FACE_EXTRA, vprint
from .hawor_runtime import build_motion_runner


@dataclass
class MotionStageContext:
    model: object
    device: torch.device
    mano_right: object
    mano_left: object
    frame_source: object
    tracks: dict
    img_focal: float
    img_center: list
    height: int
    width: int
    model_masks_tensor: torch.Tensor
    faces_right: np.ndarray
    faces_left: np.ndarray
    save_executor: ThreadPoolExecutor
    save_futures: list
    video_path: str
    seq_folder: str
    output_dir: str
    frame_chunks_file: str
    model_masks_file: str


def _resolve_img_focal(args, seq_folder):
    img_focal = args.img_focal
    if img_focal is not None:
        return img_focal

    try:
        with open(os.path.join(seq_folder, "est_focal.txt"), "r") as handle:
            return float(handle.read())
    except Exception:
        img_focal = 600
        vprint(f"No focal length provided, use default {img_focal}")
        with open(os.path.join(seq_folder, "est_focal.txt"), "w") as handle:
            handle.write(str(img_focal))
        return img_focal


def _load_motion_inputs(args, seq_folder, start_idx, end_idx, prefetched_data=None, frame_source=None):
    if prefetched_data is not None:
        return prefetched_data["frame_source"], prefetched_data["tracks"]

    if frame_source is None:
        frame_source = build_frame_source(args.video_path)
    tracks_dir = _get_tracks_dir(seq_folder, start_idx, end_idx)
    tracks = np.load(os.path.join(tracks_dir, "model_tracks.npy"), allow_pickle=True).item()
    return frame_source, tracks


def _sanitize_tracks_for_available_frames(tracks, num_frames):
    if len(tracks) == 0:
        return tracks

    try:
        max_frame_in_tracks = max(
            max(track["frame"] for track in track_data)
            for track_data in tracks.values()
            if len(track_data) > 0
        )
    except ValueError:
        return tracks

    if max_frame_in_tracks < num_frames:
        return tracks

    vprint(f"WARNING: Track data references frame {max_frame_in_tracks} but only {num_frames} frames available.")
    vprint("         This usually means extracted_images is incomplete.")
    vprint(f"         Auto-fixing: Filtering out track entries with frame >= {num_frames}")

    fixed_tracks = {}
    total_removed = 0
    for track_id, track_data in tracks.items():
        original_len = len(track_data)
        filtered_track = [track for track in track_data if track["frame"] < num_frames]
        total_removed += original_len - len(filtered_track)
        if len(filtered_track) >= 5:
            fixed_tracks[track_id] = filtered_track

    vprint(f"         Removed {total_removed} track entries referencing unavailable frames")
    vprint(f"         {len(tracks) - len(fixed_tracks)} tracks dropped (too short after filtering)")
    vprint(f"         {len(fixed_tracks)} tracks remain")
    return fixed_tracks


def _split_tracks_by_hand(tracks):
    left_track = []
    right_track = []

    for track_id in np.array([track_key for track_key in tracks]):
        track = tracks[track_id]
        if len(track) < 5:
            continue

        confidences = [item["det_box"][0, 4] for item in track if item["det"]]
        if len(confidences) == 0 or np.mean(confidences) < 0.3:
            continue

        if "is_near_edge" in track[0]:
            edge_ratio = sum(1 for item in track if item.get("is_near_edge", False)) / len(track)
            if edge_ratio > 0.7:
                continue

        valid = np.array([item["det"] for item in track])
        is_right = np.concatenate([item["det_handedness"] for item in track])[valid]
        if is_right.sum() / len(is_right) < 0.5:
            left_track.extend(track)
        else:
            right_track.extend(track)

    return {
        0: sorted(left_track, key=lambda item: item["frame"]),
        1: sorted(right_track, key=lambda item: item["frame"]),
    }


def _build_hand_faces():
    faces = get_mano_faces()
    faces_right = np.concatenate([faces, MANO_FACE_EXTRA], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]
    return faces_right, faces_left


def _build_mano_models(device, mano_models=None):
    if mano_models is not None:
        return mano_models["right"], mano_models["left"]

    from lib.models.mano_wrapper import MANO

    mano_right = MANO(**get_mano_cfg(is_right=True)).to(device)
    mano_left = MANO(**get_mano_cfg(is_right=False)).to(device)
    mano_left.shapedirs[:, 0, :] *= -1
    return mano_right, mano_left


def _resolve_cached_motion_output(args, seq_folder, start_idx, end_idx, force=False):
    output_dir, frame_chunks_file, model_masks_file = _get_motion_output_paths(seq_folder, start_idx, end_idx)
    if os.path.exists(frame_chunks_file) and not os.path.exists(model_masks_file):
        vprint(f"Warning: Incomplete output detected. Removing {frame_chunks_file} to force re-run")
        os.remove(frame_chunks_file)

    if (not force) and os.path.exists(frame_chunks_file) and os.path.exists(model_masks_file):
        vprint("skip hawor motion estimation")
        img_focal = args.img_focal
        if img_focal is None:
            try:
                with open(os.path.join(seq_folder, "est_focal.txt"), "r") as handle:
                    img_focal = float(handle.read())
            except Exception:
                img_focal = 600
        frame_chunks_all = joblib.load(frame_chunks_file)
        return output_dir, frame_chunks_file, model_masks_file, frame_chunks_all, img_focal

    return output_dir, frame_chunks_file, model_masks_file, None, None


def _build_motion_context(
    args,
    seq_folder,
    start_idx,
    end_idx,
    output_dir,
    frame_chunks_file,
    model_masks_file,
    motion_runner=None,
    mano_models=None,
    prefetched_data=None,
    frame_source=None,
):

    _invalidate_cam_space_cache(seq_folder)

    motion_runner = motion_runner or build_motion_runner(args.checkpoint)
    model = motion_runner["model"]
    device = motion_runner["device"]
    mano_right, mano_left = _build_mano_models(device, mano_models=mano_models)

    video_path = args.video_path
    frame_source, tracks = _load_motion_inputs(
        args,
        seq_folder,
        start_idx,
        end_idx,
        prefetched_data=prefetched_data,
        frame_source=frame_source,
    )
    tracks = _sanitize_tracks_for_available_frames(tracks, len(frame_source))
    img_focal = _resolve_img_focal(args, seq_folder)

    first_frame = frame_source.get_frame(0, rgb=False)
    img_center = [first_frame.shape[1] / 2, first_frame.shape[0] / 2]
    height, width = first_frame.shape[:2]
    model_masks_tensor = torch.zeros((len(frame_source), height, width), dtype=torch.bool)
    faces_right, faces_left = _build_hand_faces()

    context = MotionStageContext(
        model=model,
        device=device,
        mano_right=mano_right,
        mano_left=mano_left,
        frame_source=frame_source,
        tracks=tracks,
        img_focal=img_focal,
        img_center=img_center,
        height=height,
        width=width,
        model_masks_tensor=model_masks_tensor,
        faces_right=faces_right,
        faces_left=faces_left,
        save_executor=ThreadPoolExecutor(max_workers=1),
        save_futures=[],
        video_path=video_path,
        seq_folder=seq_folder,
        output_dir=output_dir,
        frame_chunks_file=frame_chunks_file,
        model_masks_file=model_masks_file,
    )
    return context


def _render_chunk_masks(context, frame_ck, data_out, do_flip):
    if do_flip:
        outputs = run_mano_left(
            data_out["init_trans"],
            data_out["init_root_orient"],
            data_out["init_hand_pose"],
            betas=data_out["init_betas"],
            mano_model=context.mano_left,
        )
    else:
        outputs = run_mano(
            data_out["init_trans"],
            data_out["init_root_orient"],
            data_out["init_hand_pose"],
            betas=data_out["init_betas"],
            mano_model=context.mano_right,
        )

    vertices = outputs["vertices"][0]
    faces_np = context.faces_left if do_flip else context.faces_right
    verts_2d = torch.zeros(vertices.shape[0], vertices.shape[1], 2, device=vertices.device)
    verts_2d[..., 0] = vertices[..., 0] / (vertices[..., 2] + 1e-8) * context.img_focal + context.img_center[0]
    verts_2d[..., 1] = vertices[..., 1] / (vertices[..., 2] + 1e-8) * context.img_focal + context.img_center[1]
    verts_2d_np = verts_2d.cpu().numpy().astype(np.int32)

    batch_masks = np.zeros((len(frame_ck), context.height, context.width), dtype=np.uint8)
    for mask_index, _frame_idx in enumerate(frame_ck):
        triangles = verts_2d_np[mask_index][faces_np]
        cv2.fillPoly(batch_masks[mask_index], triangles, 1)
    batch_masks_tensor = torch.from_numpy(batch_masks.view(np.bool_))
    for mask_index, frame_idx in enumerate(frame_ck):
        context.model_masks_tensor[frame_idx] |= batch_masks_tensor[mask_index]


def _prepare_track_inference_inputs(track):
    valid = np.array([item["det"] for item in track])
    if valid.sum() < 2:
        return None

    boxes = np.concatenate([item["det_box"] for item in track])
    non_zero_indices = np.where(np.any(boxes != 0, axis=1))[0]
    if len(non_zero_indices) == 0:
        return None
    first_non_zero = non_zero_indices[0]
    last_non_zero = non_zero_indices[-1]

    boxes[first_non_zero:last_non_zero + 1] = interpolate_bboxes(boxes[first_non_zero:last_non_zero + 1])
    velocity_valid = validate_motion_velocity(boxes[first_non_zero:last_non_zero + 1])
    valid[first_non_zero:last_non_zero + 1] = velocity_valid

    slice_valid = valid[first_non_zero:last_non_zero + 1]
    boxes = boxes[first_non_zero:last_non_zero + 1]
    frames = np.array([item["frame"] for item in track])[first_non_zero:last_non_zero + 1]
    handedness = np.concatenate([item["det_handedness"] for item in track])[first_non_zero:last_non_zero + 1]

    geom_valid = np.isfinite(boxes[:, :4]).all(axis=1)
    geom_valid &= boxes[:, 2] > boxes[:, 0]
    geom_valid &= boxes[:, 3] > boxes[:, 1]
    valid_mask = slice_valid & geom_valid
    if valid_mask.sum() == 0:
        return None

    boxes = boxes[valid_mask]
    frame = frames[valid_mask]
    is_right = handedness[valid_mask]

    if is_right.sum() / len(is_right) < 0.5:
        is_right = np.zeros((len(boxes), 1))
    else:
        is_right = np.ones((len(boxes), 1))

    frame_chunks, boxes_chunks = parse_chunks(frame, boxes, min_len=1)
    if len(frame_chunks) == 0:
        return None

    all_frame_indices = []
    all_boxes_list = []
    chunk_boundaries = [0]
    for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
        all_frame_indices.extend(frame_ck)
        all_boxes_list.append(boxes_ck)
        chunk_boundaries.append(len(all_frame_indices))

    if len(all_frame_indices) == 0:
        return None

    return {
        "frame_chunks": frame_chunks,
        "all_frame_indices": np.array(all_frame_indices, dtype=np.int64),
        "all_boxes": np.concatenate(all_boxes_list, axis=0) if len(all_boxes_list) > 1 else all_boxes_list[0],
        "chunk_boundaries": chunk_boundaries,
        "do_flip": bool(is_right[0] <= 0),
    }


def _save_and_render_chunk(context, idx, frame_ck, chunk_results, do_flip):
    data_out = {
        "init_root_orient": chunk_results["pred_rotmat"][None, :, 0],
        "init_hand_pose": chunk_results["pred_rotmat"][None, :, 1:],
        "init_trans": chunk_results["pred_trans"][None, :, 0],
        "init_betas": chunk_results["pred_shape"][None, :],
    }

    init_root = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
    init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
    if do_flip:
        init_root[..., 1] *= -1
        init_root[..., 2] *= -1
        init_hand_pose[..., 1] *= -1
        init_hand_pose[..., 2] *= -1
    data_out["init_root_orient"] = angle_axis_to_rotation_matrix(init_root)
    data_out["init_hand_pose"] = angle_axis_to_rotation_matrix(init_hand_pose)

    data_out_for_save = {key: value.clone().cpu() for key, value in data_out.items()}
    context.save_futures.append(
        context.save_executor.submit(
            _save_cam_space_json,
            data_out_for_save,
            context.seq_folder,
            idx,
            frame_ck[0],
            frame_ck[-1],
        )
    )

    data_out["init_root_orient"] = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
    data_out["init_hand_pose"] = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
    _render_chunk_masks(context, frame_ck, data_out, do_flip=do_flip)


def _process_hand_track(args, idx, track, context, profiler=None):
    track_inputs = _prepare_track_inference_inputs(track)
    if track_inputs is None:
        return [], 0.0, 0.0, 0.0

    frame_chunks = track_inputs["frame_chunks"]
    all_frame_indices = track_inputs["all_frame_indices"]
    all_boxes = track_inputs["all_boxes"]
    chunk_boundaries = track_inputs["chunk_boundaries"]
    do_flip = track_inputs["do_flip"]

    vprint(
        f"inference from frame {all_frame_indices[0]} to {all_frame_indices[-1]} "
        f"({len(frame_chunks)} chunks merged)"
    )

    inference_time = 0.0
    postprocess_time = 0.0
    render_time = 0.0

    t_inference = time.time()
    if profiler:
        print(f"[PROFILER] Step before inference (track {idx})")
        profiler.step()
    results = context.model.inference(
        context.frame_source,
        all_frame_indices,
        all_boxes,
        img_focal=context.img_focal,
        img_center=context.img_center,
        device=context.device,
        do_flip=do_flip,
        chunk_batch_size=getattr(args, "chunk_batch_size", 4),
        num_workers=getattr(args, "num_workers", 16),
        output_device=context.device,
    )
    if profiler:
        print(f"[PROFILER] Step after inference (track {idx})")
        profiler.step()
    inference_time += time.time() - t_inference

    t_post = time.time()
    for chunk_idx, frame_ck in enumerate(frame_chunks):
        start_idx = chunk_boundaries[chunk_idx]
        end_idx = chunk_boundaries[chunk_idx + 1]
        chunk_results = {
            "pred_rotmat": results["pred_rotmat"][start_idx:end_idx],
            "pred_trans": results["pred_trans"][start_idx:end_idx],
            "pred_shape": results["pred_shape"][start_idx:end_idx],
        }
        t_render = time.time()
        _save_and_render_chunk(context, idx, frame_ck, chunk_results, do_flip)
        render_time += time.time() - t_render

    postprocess_time += time.time() - t_post
    return frame_chunks, inference_time, postprocess_time, render_time


def _finalize_motion_outputs(context, frame_chunks_all, profiler=None):
    for future in context.save_futures:
        future.result()
    context.save_executor.shutdown(wait=False)

    if profiler:
        print("[PROFILER] Final step after all tracks processed")
        profiler.step()

    model_masks = context.model_masks_tensor.cpu().numpy()
    del context.model_masks_tensor
    torch.cuda.empty_cache()

    os.makedirs(context.output_dir, exist_ok=True)
    _save_motion_outputs(
        model_masks,
        frame_chunks_all,
        context.model_masks_file,
        context.frame_chunks_file,
        context.output_dir,
    )


def run_motion_for_video(
    args,
    start_idx,
    end_idx,
    seq_folder,
    motion_runner=None,
    profiler=None,
    mano_models=None,
    prefetched_data=None,
    frame_source=None,
    force=False,
):
    timing = {}
    t_start_total = time.time()

    output_dir, frame_chunks_file, model_masks_file, cached_frame_chunks_all, img_focal = _resolve_cached_motion_output(
        args,
        seq_folder,
        start_idx,
        end_idx,
        force=force,
    )
    if cached_frame_chunks_all is not None:
        return cached_frame_chunks_all, img_focal

    context = _build_motion_context(
        args,
        seq_folder,
        start_idx,
        end_idx,
        output_dir,
        frame_chunks_file,
        model_masks_file,
        motion_runner=motion_runner,
        mano_models=mano_models,
        prefetched_data=prefetched_data,
        frame_source=frame_source,
    )

    timing["1_load_data"] = time.time() - t_start_total

    vprint(f"Running hawor on {os.path.basename(context.video_path)} ...")
    t_setup = time.time()
    final_tracks = _split_tracks_by_hand(context.tracks)
    timing["2_setup"] = time.time() - t_setup

    t_tracks = time.time()
    frame_chunks_all = defaultdict(list)
    timing_inference = 0.0
    timing_postprocess = 0.0
    timing_render = 0.0

    for idx in [0, 1]:
        vprint(f"tracklet {idx}:")
        track = final_tracks[idx]
        frame_chunks, inference_time, postprocess_time, render_time = _process_hand_track(
            args,
            idx,
            track,
            context,
            profiler=profiler,
        )
        frame_chunks_all[idx] = frame_chunks
        timing_inference += inference_time
        timing_postprocess += postprocess_time
        timing_render += render_time

    timing["3_track_processing"] = time.time() - t_tracks
    timing["3a_inference"] = timing_inference
    timing["3b_postprocess"] = timing_postprocess
    timing["3c_render"] = timing_render

    t_save = time.time()
    _finalize_motion_outputs(context, frame_chunks_all, profiler=profiler)
    timing["4_save_results"] = time.time() - t_save
    timing["total"] = time.time() - t_start_total

    print(f"\n{'=' * 60}")
    print(f"Motion Stage Timing for {os.path.basename(context.video_path)}")
    print(f"{'=' * 60}")
    for key in sorted(timing.keys()):
        if key == "total":
            continue
        pct = (timing[key] / timing["total"]) * 100
        print(f"  {key:25s}: {timing[key]:6.2f}s ({pct:5.1f}%)")
    print(f"  {'total':25s}: {timing['total']:6.2f}s")
    print(f"{'=' * 60}\n")
    print(f"Motion stage completed successfully for {os.path.basename(context.video_path)}")

    return frame_chunks_all, context.img_focal


def hawor_motion_estimation(args, start_idx, end_idx, seq_folder, profiler=None):
    return run_motion_for_video(args, start_idx, end_idx, seq_folder, motion_runner=None, profiler=profiler)
