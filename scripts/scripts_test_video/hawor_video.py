from collections import defaultdict

import json
import os
import sys
import joblib
import numpy as np
import torch
from tqdm import tqdm

from lib.pipeline.frame_source import build_frame_source
from lib.pipeline.tools import parse_chunks, parse_chunks_hand_frame
from lib.models.hawor import HAWOR
from lib.eval_utils.custom_utils import cam2world_convert, load_slam_cam
from lib.eval_utils.custom_utils import interpolate_bboxes, validate_motion_velocity
from lib.eval_utils.filling_utils import filling_postprocess, filling_preprocess
import cv2
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
from infiller.lib.model.network import TransformerModel

# Check if we should suppress verbose output
QUIET_MODE = os.environ.get("HAWOR_QUIET", "0") == "1"

def vprint(*args, **kwargs):
    """Print only if not in quiet mode."""
    if not QUIET_MODE:
        print(*args, **kwargs)

def load_hawor(checkpoint_path):
    from pathlib import Path
    from hawor.configs import get_config
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()

    model = HAWOR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg

def build_motion_runner(checkpoint_path, device=None):
    model, model_cfg = load_hawor(checkpoint_path)
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device)
    model.eval()
    # torch.compile with CUDA graphs for ViT backbone (10-30% speedup)
    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            model.backbone = torch.compile(model.backbone, mode="reduce-overhead")
            vprint("[torch.compile] ViT backbone compiled with mode='reduce-overhead'")
        except Exception as e:
            vprint(f"[torch.compile] Skipping backbone compilation: {e}")
    return {
        'model': model,
        'model_cfg': model_cfg,
        'device': device,
    }


def build_infiller_runner(weight_path, device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    ckpt = torch.load(weight_path, map_location=device)
    pos_dim = 3
    shape_dim = 10
    num_joints = 15
    rot_dim = (num_joints + 1) * 6 # rot6d
    repr_dim = 2 * (pos_dim + shape_dim + rot_dim)
    nhead = 8 # repr_dim = 154
    horizon = 120
    filling_model = TransformerModel(seq_len=horizon, input_dim=repr_dim, d_model=384, nhead=nhead, d_hid=2048, nlayers=8, dropout=0.05, out_dim=repr_dim, masked_attention_stage=True)
    filling_model.to(device)
    filling_model.load_state_dict(ckpt['transformer_encoder_state_dict'])
    filling_model.eval()
    return {
        'model': filling_model,
        'device': device,
        'horizon': horizon,
    }


def run_motion_for_video(args, start_idx, end_idx, seq_folder, motion_runner=None, profiler=None, mano_models=None, prefetched_data=None):
    import time
    timing = {}
    t_start_total = time.time()

    # Early skip check - before any expensive operations
    frame_chunks_file = f'{seq_folder}/tracks_{start_idx}_{end_idx}/frame_chunks_all.npy'
    model_masks_file = f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_masks.npy'

    # Auto-fix incomplete outputs: if frame_chunks exists but model_masks doesn't, remove frame_chunks
    if os.path.exists(frame_chunks_file) and not os.path.exists(model_masks_file):
        vprint(f"Warning: Incomplete output detected. Removing {frame_chunks_file} to force re-run")
        os.remove(frame_chunks_file)

    if os.path.exists(frame_chunks_file) and os.path.exists(model_masks_file):
        vprint("skip hawor motion estimation")
        # Need to load img_focal for return value
        img_focal = args.img_focal
        if img_focal is None:
            try:
                with open(os.path.join(seq_folder, 'est_focal.txt'), 'r') as f:
                    img_focal = float(f.read())
            except:
                img_focal = 600
        frame_chunks_all = joblib.load(frame_chunks_file)
        return frame_chunks_all, img_focal

    # If not skipping, proceed with full initialization
    t0 = time.time()
    motion_runner = motion_runner or build_motion_runner(args.checkpoint)
    model = motion_runner['model']

    # Create MANO models once for reuse (avoid recreation overhead)
    device = motion_runner['device']

    if mano_models is not None:
        # Reuse cached MANO models from WorkerRuntime
        mano_right = mano_models['right']
        mano_left = mano_models['left']
    else:
        # Create MANO models (standalone/backward-compatible path)
        from lib.models.mano_wrapper import MANO

        # Right hand MANO model
        MANO_cfg_right = {
            'data_dir': '_DATA/data/',
            'model_path': '_DATA/data/mano',
            'gender': 'neutral',
            'num_hand_joints': 15,
            'create_body_pose': False
        }
        mano_right = MANO(**MANO_cfg_right).to(device)

        # Left hand MANO model
        MANO_cfg_left = {
            'data_dir': '_DATA/data_left/',
            'model_path': '_DATA/data_left/mano_left',
            'gender': 'neutral',
            'num_hand_joints': 15,
            'create_body_pose': False,
            'is_rhand': False
        }
        mano_left = MANO(**MANO_cfg_left).to(device)
        # Fix MANO shapedirs of the left hand bug
        mano_left.shapedirs[:, 0, :] *= -1

    video_path = args.video_path

    # Use prefetched data if available, otherwise load fresh
    if prefetched_data is not None:
        frame_source = prefetched_data['frame_source']
        tracks = prefetched_data['tracks']
    else:
        frame_source = build_frame_source(video_path)
        tracks = np.load(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_tracks.npy', allow_pickle=True).item()

    # Validate and auto-fix tracks that reference frames beyond available frames
    num_frames = len(frame_source)
    if len(tracks) > 0:
        try:
            max_frame_in_tracks = max(
                max(t['frame'] for t in track_data)
                for track_data in tracks.values()
                if len(track_data) > 0
            )
            if max_frame_in_tracks >= num_frames:
                vprint(f"WARNING: Track data references frame {max_frame_in_tracks} but only {num_frames} frames available.")
                vprint(f"         This usually means extracted_images is incomplete.")
                vprint(f"         Auto-fixing: Filtering out track entries with frame >= {num_frames}")

                # Auto-fix: Remove track entries that reference unavailable frames
                fixed_tracks = {}
                total_removed = 0
                for track_id, track_data in tracks.items():
                    original_len = len(track_data)
                    filtered_track = [t for t in track_data if t['frame'] < num_frames]
                    total_removed += original_len - len(filtered_track)

                    # Keep track only if it has at least 5 valid frames (same threshold as line 183)
                    if len(filtered_track) >= 5:
                        fixed_tracks[track_id] = filtered_track

                vprint(f"         Removed {total_removed} track entries referencing unavailable frames")
                vprint(f"         {len(tracks) - len(fixed_tracks)} tracks dropped (too short after filtering)")
                vprint(f"         {len(fixed_tracks)} tracks remain")

                tracks = fixed_tracks
        except ValueError:
            # All tracks are empty, skip validation
            pass

    img_focal = args.img_focal
    timing['1_load_data'] = time.time() - t0
    if img_focal is None:
        try:
            with open(os.path.join(seq_folder, 'est_focal.txt'), 'r') as f:
                img_focal = f.read()
                img_focal = float(img_focal)
        except:
            img_focal = 600
            vprint(f'No focal length provided, use default {img_focal}')
            with open(os.path.join(seq_folder, 'est_focal.txt'), 'w') as f:
                f.write(str(img_focal))

    tid = np.array([tr for tr in tracks])

    vprint(f'Running hawor on {os.path.basename(video_path)} ...')

    t0 = time.time()
    left_trk = []
    right_trk = []
    for k, idx in enumerate(tid):
        trk = tracks[idx]

        # Filter out very short tracks (likely false positives)
        if len(trk) < 5:  # Require at least 5 frames
            continue

        # Check average confidence
        confs = [t['det_box'][0, 4] for t in trk if t['det']]
        if len(confs) == 0 or np.mean(confs) < 0.3:  # Require avg confidence >= 0.3
            continue

        # Check if track is mostly near edges (likely hand leaving/entering frame)
        if 'is_near_edge' in trk[0]:
            edge_ratio = sum(1 for t in trk if t.get('is_near_edge', False)) / len(trk)
            if edge_ratio > 0.7:  # If >70% detections are near edge, likely unstable
                continue

        valid = np.array([t['det'] for t in trk])
        is_right = np.concatenate([t['det_handedness'] for t in trk])[valid]

        if is_right.sum() / len(is_right) < 0.5:
            left_trk.extend(trk)
        else:
            right_trk.extend(trk)
    left_trk = sorted(left_trk, key=lambda x: x['frame'])
    right_trk = sorted(right_trk, key=lambda x: x['frame'])
    final_tracks = {
        0: left_trk,
        1: right_trk
    }
    tid = [0, 1]

    img = frame_source.get_frame(0, rgb=False)
    img_center = [img.shape[1] / 2, img.shape[0] / 2]# w/2, h/2
    H, W = img.shape[:2]

    # Use GPU tensor for model_masks to avoid CPU-GPU transfers
    model_masks_gpu = torch.zeros((len(frame_source), H, W), device='cuda', dtype=torch.bool)

    # get faces
    faces = get_mano_faces()
    faces_new = np.array([[92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)
    faces_left = faces_right[:,[0,2,1]]

    timing['2_setup'] = time.time() - t0

    t0 = time.time()
    frame_chunks_all = defaultdict(list)
    timing_inference = 0
    timing_postprocess = 0
    timing_render = 0

    # Background thread for IO-bound saves (JSON serialization + write)
    from concurrent.futures import ThreadPoolExecutor
    save_executor = ThreadPoolExecutor(max_workers=1)
    save_futures = []

    def _save_cam_space_json(data_out_cpu, seq_folder, idx, frame_ck_first, frame_ck_last):
        """Serialize data_out to JSON and save to disk (CPU/IO-bound)."""
        pred_dict = {k: v.tolist() for k, v in data_out_cpu.items()}
        pred_path = os.path.join(seq_folder, 'cam_space', str(idx), f"{frame_ck_first}_{frame_ck_last}.json")
        cam_dir = os.path.join(seq_folder, 'cam_space', str(idx))
        if not os.path.exists(cam_dir):
            os.makedirs(cam_dir)
        with open(pred_path, "w") as f:
            json.dump(pred_dict, f, indent=1)

    for idx in tid:
        vprint(f"tracklet {idx}:")
        trk = final_tracks[idx]

        # interp bboxes
        valid = np.array([t['det'] for t in trk])
        if valid.sum() < 2:
            continue
        boxes = np.concatenate([t['det_box'] for t in trk])
        non_zero_indices = np.where(np.any(boxes != 0, axis=1))[0]
        first_non_zero = non_zero_indices[0]
        last_non_zero = non_zero_indices[-1]

        # Interpolate bboxes with size consistency check
        boxes[first_non_zero:last_non_zero+1] = interpolate_bboxes(boxes[first_non_zero:last_non_zero+1])

        # Apply motion velocity validation to filter implausible movements
        velocity_valid = validate_motion_velocity(boxes[first_non_zero:last_non_zero+1])

        # Update valid mask: only frames that pass both interpolation and velocity check
        valid[first_non_zero:last_non_zero+1] = velocity_valid


        boxes = boxes[first_non_zero:last_non_zero+1]
        is_right = np.concatenate([t['det_handedness'] for t in trk])[valid]
        frame = np.array([t['frame'] for t in trk])[valid]
        
        if is_right.sum() / len(is_right) < 0.5:
            is_right = np.zeros((len(boxes), 1))
        else:
            is_right = np.ones((len(boxes), 1))

        frame_chunks, boxes_chunks = parse_chunks(frame, boxes, min_len=1)
        frame_chunks_all[idx] = frame_chunks

        if len(frame_chunks) == 0:
            continue

        # Optimization: Merge all chunks for this hand into a single inference call
        # This reduces overhead and improves GPU utilization
        if is_right[0] > 0:
            do_flip = False
        else:
            do_flip = True

        # Collect all frame indices and boxes from all chunks
        all_frame_indices = []
        all_boxes_list = []
        chunk_boundaries = [0]  # Track where each chunk starts for later splitting

        for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
            all_frame_indices.extend(frame_ck)
            all_boxes_list.append(boxes_ck)
            chunk_boundaries.append(len(all_frame_indices))

        if len(all_frame_indices) == 0:
            continue

        vprint(f"inference from frame {all_frame_indices[0]} to {all_frame_indices[-1]} ({len(frame_chunks)} chunks merged)")

        # Single inference call for all chunks
        all_frame_indices = np.array(all_frame_indices, dtype=np.int64)
        all_boxes = np.concatenate(all_boxes_list, axis=0) if len(all_boxes_list) > 1 else all_boxes_list[0]

        t_inf = time.time()
        if profiler:
            print(f"[PROFILER] Step before inference (track {idx})")
            profiler.step()  # Profile this inference call
        results = model.inference(
            frame_source,
            all_frame_indices,
            all_boxes,
            img_focal=img_focal,
            img_center=img_center,
            do_flip=do_flip,
            chunk_batch_size=getattr(args, 'chunk_batch_size', 4),
            num_workers=getattr(args, 'num_workers', 16),
        )
        if profiler:
            print(f"[PROFILER] Step after inference (track {idx})")
            profiler.step()  # Profile post-inference
        timing_inference += time.time() - t_inf

        # Process results for each original chunk
        t_post = time.time()
        for chunk_idx, (frame_ck, boxes_ck) in enumerate(zip(frame_chunks, boxes_chunks)):
            start_idx = chunk_boundaries[chunk_idx]
            end_idx = chunk_boundaries[chunk_idx + 1]

            # Extract results for this chunk
            chunk_results = {
                "pred_rotmat": results["pred_rotmat"][start_idx:end_idx],
                "pred_trans": results["pred_trans"][start_idx:end_idx],
                "pred_shape": results["pred_shape"][start_idx:end_idx],
            }

            data_out = {
                "init_root_orient": chunk_results["pred_rotmat"][None, :, 0], # (B, T, 3, 3)
                "init_hand_pose": chunk_results["pred_rotmat"][None, :, 1:], # (B, T, 15, 3, 3)
                "init_trans": chunk_results["pred_trans"][None, :, 0],  # (B, T, 3)
                "init_betas": chunk_results["pred_shape"][None, :]  # (B, T, 10)
            }

            # flip left hand
            init_root = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
            init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
            if do_flip:
                init_root[..., 1] *= -1
                init_root[..., 2] *= -1
                init_hand_pose[..., 1] *= -1
                init_hand_pose[..., 2] *= -1
            data_out["init_root_orient"] = angle_axis_to_rotation_matrix(init_root)
            data_out["init_hand_pose"] = angle_axis_to_rotation_matrix(init_hand_pose)

            # save camera-space results (background thread for IO)
            # Clone tensors to CPU for safe background serialization
            data_out_for_save = {k: v.clone().cpu() for k, v in data_out.items()}
            save_futures.append(save_executor.submit(
                _save_cam_space_json, data_out_for_save, seq_folder, idx,
                frame_ck[0], frame_ck[-1]
            ))


            # get hand mask
            t_rend = time.time()
            data_out["init_root_orient"] = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
            data_out["init_hand_pose"] = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
            if do_flip: # left
                outputs = run_mano_left(data_out["init_trans"], data_out["init_root_orient"], data_out["init_hand_pose"], betas=data_out["init_betas"], mano_model=mano_left)
            else: # right
                outputs = run_mano(data_out["init_trans"], data_out["init_root_orient"], data_out["init_hand_pose"], betas=data_out["init_betas"], mano_model=mano_right)

            # Keep vertices on GPU to avoid CPU-GPU transfer
            vertices = outputs["vertices"][0]  # (T, N, 3) - stays on GPU
            frame_indices = np.array(frame_ck, dtype=np.int64)

            # Generate binary masks via 2D projection + cv2.fillPoly
            # (replaces PyTorch3D GPU rasterizer — only binary mask is needed)
            faces_np = faces_left if do_flip else faces_right  # (F, 3) numpy

            # Project all vertices to 2D in one batch on GPU, then transfer once
            verts_2d = torch.zeros(vertices.shape[0], vertices.shape[1], 2, device=vertices.device)
            verts_2d[..., 0] = vertices[..., 0] / (vertices[..., 2] + 1e-8) * img_focal + img_center[0]
            verts_2d[..., 1] = vertices[..., 1] / (vertices[..., 2] + 1e-8) * img_focal + img_center[1]
            verts_2d_np = verts_2d.cpu().numpy().astype(np.int32)  # (T, 778, 2)

            # Vectorized triangle lookup: batch all fillPoly on CPU, single GPU transfer
            batch_masks = np.zeros((len(frame_ck), H, W), dtype=np.uint8)
            for i, fi in enumerate(frame_ck):
                tris = verts_2d_np[i][faces_np]  # (F, 3, 2)
                cv2.fillPoly(batch_masks[i], tris, 1)
            batch_masks_gpu = torch.from_numpy(batch_masks.view(np.bool_)).cuda()
            for i, fi in enumerate(frame_ck):
                model_masks_gpu[fi] |= batch_masks_gpu[i]
            del batch_masks_gpu

            timing_render += time.time() - t_rend
        timing_postprocess += time.time() - t_post

    timing['3_track_processing'] = time.time() - t0
    timing['3a_inference'] = timing_inference
    timing['3b_postprocess'] = timing_postprocess
    timing['3c_render'] = timing_render

    # Wait for all background JSON saves to complete
    for future in save_futures:
        future.result()  # Raises if any save failed
    save_executor.shutdown(wait=False)

    # Final profiler step to ensure trace export completes
    if profiler:
        print(f"[PROFILER] Final step after all tracks processed")
        profiler.step()

    t0 = time.time()
    # Transfer to CPU only once at the end
    model_masks = model_masks_gpu.cpu().numpy()  # bool tensor to numpy
    del model_masks_gpu
    torch.cuda.empty_cache()

    # Ensure output directory exists
    output_dir = f'{seq_folder}/tracks_{start_idx}_{end_idx}'
    os.makedirs(output_dir, exist_ok=True)

    # Save model_masks and frame_chunks in parallel (both are independent IO operations)
    def _save_masks():
        np.save(model_masks_file, model_masks)
        if not os.path.exists(model_masks_file):
            raise IOError(f"File not found after save: {model_masks_file}")
        file_size = os.path.getsize(model_masks_file)
        if file_size == 0:
            raise IOError(f"File is empty after save: {model_masks_file}")
        vprint(f"✓ Saved model_masks.npy ({model_masks.shape}, {model_masks.dtype}, {file_size} bytes)")

    def _save_chunks():
        joblib.dump(frame_chunks_all, frame_chunks_file)
        if not os.path.exists(frame_chunks_file):
            raise IOError(f"File not found after save: {frame_chunks_file}")
        file_size = os.path.getsize(frame_chunks_file)
        if file_size == 0:
            raise IOError(f"File is empty after save: {frame_chunks_file}")
        vprint(f"✓ Saved frame_chunks_all.npy ({file_size} bytes)")

    from concurrent.futures import ThreadPoolExecutor as _TPE
    with _TPE(max_workers=2) as save_pool:
        mask_future = save_pool.submit(_save_masks)
        chunks_future = save_pool.submit(_save_chunks)
        # Wait for both, propagate any errors
        try:
            mask_future.result()
        except Exception as e:
            print(f"ERROR: Failed to save model_masks.npy: {e}", file=sys.stderr)
            print(f"  Path: {model_masks_file}", file=sys.stderr)
            print(f"  Directory exists: {os.path.exists(output_dir)}", file=sys.stderr)
            raise
        try:
            chunks_future.result()
        except Exception as e:
            print(f"ERROR: Failed to save frame_chunks_all.npy: {e}", file=sys.stderr)
            print(f"  Path: {frame_chunks_file}", file=sys.stderr)
            raise

    timing['4_save_results'] = time.time() - t0

    timing['total'] = time.time() - t_start_total

    # Print timing summary
    print(f"\n{'='*60}")
    print(f"Motion Stage Timing for {os.path.basename(video_path)}")
    print(f"{'='*60}")
    for key in sorted(timing.keys()):
        if key == 'total':
            continue
        pct = (timing[key] / timing['total']) * 100
        print(f"  {key:25s}: {timing[key]:6.2f}s ({pct:5.1f}%)")
    print(f"  {'total':25s}: {timing['total']:6.2f}s")
    print(f"{'='*60}\n")

    print(f"✓ Motion stage completed successfully for {os.path.basename(video_path)}")

    return frame_chunks_all, img_focal

def hawor_motion_estimation(args, start_idx, end_idx, seq_folder, profiler=None):
    return run_motion_for_video(args, start_idx, end_idx, seq_folder, motion_runner=None, profiler=profiler)


def run_infiller_for_video(args, start_idx, end_idx, frame_chunks_all, infiller_runner=None):
    # load infiller
    infiller_runner = infiller_runner or build_infiller_runner(args.infiller_weight)
    filling_model = infiller_runner['model']
    device = infiller_runner['device']
    horizon = infiller_runner['horizon']

    video_path = args.video_path
    frame_source = build_frame_source(video_path)
    seq_folder = os.path.join(os.path.dirname(video_path), os.path.basename(video_path).split('.')[0])

    # Previous steps
    num_frames = len(frame_source)

    idx2hand = ['left', 'right']
    filling_length = 120

    fpath = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(fpath)

    pred_trans = torch.zeros(2, num_frames, 3)
    pred_rot = torch.zeros(2, num_frames, 3)
    pred_hand_pose = torch.zeros(2, num_frames, 45)
    pred_betas = torch.zeros(2, num_frames, 10)
    pred_valid = torch.zeros((2, pred_betas.size(1)))

    max_slam_frames = min(pred_trans.shape[1], R_c2w_sla_all.shape[0], t_c2w_sla_all.shape[0])

    # camera space to world space
    tid = [0, 1]            
    for k, idx in enumerate(tid):
        frame_chunks = frame_chunks_all[idx]

        if len(frame_chunks) == 0:
            continue

        for frame_ck in frame_chunks:
            frame_ck = np.asarray(frame_ck)
            valid_frame_mask = frame_ck < max_slam_frames
            if valid_frame_mask.sum() == 0:
                continue
            frame_ck = frame_ck[valid_frame_mask]
            vprint(f"from frame {frame_ck[0]} to {frame_ck[-1]}")
            pred_path = os.path.join(seq_folder, 'cam_space', str(idx), f"{frame_ck[0]}_{frame_ck[-1]}.json")
            with open(pred_path, "r") as f:
                pred_dict = json.load(f)
            data_out = {
                k:torch.tensor(v) for k, v in pred_dict.items()
                }

            R_c2w_sla = R_c2w_sla_all[frame_ck]
            t_c2w_sla = t_c2w_sla_all[frame_ck]

            data_world = cam2world_convert(R_c2w_sla, t_c2w_sla, data_out, 'right' if idx > 0 else 'left')

            pred_trans[[idx], frame_ck] = data_world["init_trans"]
            pred_rot[[idx], frame_ck] = data_world["init_root_orient"]
            pred_hand_pose[[idx], frame_ck] = data_world["init_hand_pose"].flatten(-2)
            pred_betas[[idx], frame_ck] = data_world["init_betas"]
            pred_valid[[idx], frame_ck] = 1
            
        
    # runing fillingnet for this video
    frame_list = torch.tensor(list(range(pred_trans.size(1))))
    pred_valid = (pred_valid > 0).numpy()
    for k, idx in enumerate([1, 0]):
        missing = ~pred_valid[idx]

        frame = frame_list[missing]
        frame_chunks = parse_chunks_hand_frame(frame)

        vprint(f"run infiller on {idx2hand[idx]} hand ...")
        for frame_ck in tqdm(frame_chunks, disable=QUIET_MODE):
            start_shift = -1
            while frame_ck[0] + start_shift >= 0 and pred_valid[:, frame_ck[0] + start_shift].sum() != 2:
                start_shift -= 1  # Shift to find the previous valid frame as start
            vprint(f"run infiller on frame {frame_ck[0] + start_shift} to frame {min(num_frames-1, frame_ck[0] + start_shift + filling_length)}")

            frame_start = frame_ck[0]
            filling_net_start = max(0, frame_start + start_shift)
            filling_net_end = min(num_frames-1, filling_net_start + filling_length)
            seq_valid = pred_valid[:, filling_net_start:filling_net_end]
            filling_seq = {}
            filling_seq['trans'] = pred_trans[:, filling_net_start:filling_net_end].numpy()
            filling_seq['rot'] = pred_rot[:, filling_net_start:filling_net_end].numpy()
            filling_seq['hand_pose'] = pred_hand_pose[:, filling_net_start:filling_net_end].numpy()
            filling_seq['betas'] = pred_betas[:, filling_net_start:filling_net_end].numpy()
            filling_seq['valid'] = seq_valid
            # preprocess (convert to canonical + slerp)
            filling_input, transform_w_canon = filling_preprocess(filling_seq)
            src_mask = torch.zeros((filling_length, filling_length), device=device).type(torch.bool)
            src_mask = src_mask.to(device)
            filling_input = torch.from_numpy(filling_input).unsqueeze(0).to(device).permute(1,0,2) # (seq_len, B, in_dim)
            T_original = len(filling_input)
            filling_length = 120
            if T_original < filling_length:
                pad_length = filling_length - T_original
                last_time_step = filling_input[-1, :, :]
                padding = last_time_step.unsqueeze(0).repeat(pad_length, 1, 1)
                filling_input = torch.cat([filling_input, padding], dim=0) 
                seq_valid_padding = np.ones((2, filling_length - T_original))
                seq_valid_padding = np.concatenate([seq_valid, seq_valid_padding], axis=1) 
            else:
                seq_valid_padding = seq_valid
                

            T, B, _ = filling_input.shape

            valid = torch.from_numpy(seq_valid_padding).unsqueeze(0).all(dim=1).permute(1, 0) # (T,B)
            valid_atten = torch.from_numpy(seq_valid_padding).unsqueeze(0).all(dim=1).unsqueeze(1) # (B,1,T)
            data_mask = torch.zeros((horizon, B, 1), device=device, dtype=filling_input.dtype)
            data_mask[valid] = 1
            atten_mask = torch.ones((B, 1, horizon),
                        device=device, dtype=torch.bool)
            atten_mask[valid_atten] = False
            atten_mask = atten_mask.unsqueeze(2).repeat(1, 1, T, 1) # (B,1,T,T)

            output_ck = filling_model(filling_input, src_mask, data_mask, atten_mask)

            output_ck = output_ck.permute(1,0,2).reshape(T, 2, -1).cpu().detach() #  two hands

            output_ck = output_ck[:T_original]

            filling_output = filling_postprocess(output_ck, transform_w_canon)

            # repalce the missing prediciton with infiller output
            filling_seq['trans'][~seq_valid] = filling_output['trans'][~seq_valid]
            filling_seq['rot'][~seq_valid] = filling_output['rot'][~seq_valid]
            filling_seq['hand_pose'][~seq_valid] = filling_output['hand_pose'][~seq_valid]
            filling_seq['betas'][~seq_valid] = filling_output['betas'][~seq_valid]

            pred_trans[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq['trans'][:])
            pred_rot[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq['rot'][:])
            pred_hand_pose[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq['hand_pose'][:])
            pred_betas[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq['betas'][:])
            pred_valid[:, filling_net_start:filling_net_end] = 1
    save_path = os.path.join(seq_folder, "world_space_res.pth")
    joblib.dump([pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid], save_path)
    return pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid


def hawor_infiller(args, start_idx, end_idx, frame_chunks_all):
    return run_infiller_for_video(args, start_idx, end_idx, frame_chunks_all, infiller_runner=None)
    