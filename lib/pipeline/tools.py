import cv2
from tqdm import tqdm
import numpy as np
import torch
import os

from ultralytics import YOLO
import supervision as sv

# Check if we should suppress verbose output
QUIET_MODE = os.environ.get("HAWOR_QUIET", "0") == "1"


def detect_track(
    frame_source,
    thresh=0.35,
    edge_margin_ratio=0.1,
    min_edge_conf=0.4,
    hand_det_model=None,
    detect_batch_size=128,
    num_io_workers=8,
    device='cuda:0',
    half_precision=True,
):
    """
    Detect and track hands using batched YOLO inference + post-hoc ByteTrack.

    Phase 1: Batch detection - YOLO.predict() with large batch sizes for high GPU utilization
    Phase 2: Sequential tracking - supervision.ByteTrack on CPU for track assignment

    Args:
        frame_source: ImageFolderFrameSource with pre-extracted frames
        thresh: Base confidence threshold for detection
        edge_margin_ratio: Ratio of image size to define edge region (default 0.1 = 10%)
        min_edge_conf: Minimum confidence required for detections near edges
        hand_det_model: Optional preloaded YOLO detector for reuse
        detect_batch_size: Batch size for YOLO.predict() (default 128)
        num_io_workers: Number of DataLoader workers for parallel frame loading
        device: Device for YOLO detector (e.g., 'cuda:0')
        half_precision: Use FP16 for YOLO inference
    """
    from lib.pipeline.frame_source import FrameDataset, _numpy_collate, _frame_dataset_worker_init

    hand_det_model = hand_det_model or YOLO('./weights/external/detector.pt')

    if device:
        hand_det_model.to(device)
    if half_precision and device and 'cuda' in device:
        hand_det_model.model.half()

    num_frames = len(frame_source)
    img_h, img_w = frame_source.get_size()

    # --- Phase 1: Batch Detection (GPU) ---
    all_detections = [None] * num_frames  # (xyxy, confs, class_ids) per frame
    all_boxes_raw = [np.array([]).reshape(0, 5)] * num_frames  # boxes with conf for output

    dataset = FrameDataset(frame_source)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=detect_batch_size,
        shuffle=False,
        num_workers=num_io_workers,
        collate_fn=_numpy_collate,
        worker_init_fn=_frame_dataset_worker_init,
        pin_memory=False,
        prefetch_factor=2 if num_io_workers > 0 else None,
        persistent_workers=num_io_workers > 0,
    )

    for batch_indices, batch_frames in tqdm(loader, disable=QUIET_MODE, desc="Detect (batched)"):
        with torch.no_grad():
            results_list = hand_det_model.predict(
                batch_frames, conf=thresh, verbose=False
            )

        for frame_idx, result in zip(batch_indices, results_list):
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            all_detections[frame_idx] = (boxes, confs, class_ids)
            if len(boxes) > 0:
                all_boxes_raw[frame_idx] = np.hstack([boxes, confs[:, None]])

    # --- Phase 2: Sequential Tracking (CPU) ---
    tracker = sv.ByteTrack()

    tracks = {}
    fallback_counter = 0
    track_last_seen = {}

    edge_left = img_w * edge_margin_ratio
    edge_right = img_w * (1 - edge_margin_ratio)
    edge_top = img_h * edge_margin_ratio
    edge_bottom = img_h * (1 - edge_margin_ratio)

    for t in tqdm(range(num_frames), disable=QUIET_MODE, desc="Track (sequential)"):
        det = all_detections[t]
        if det is None:
            continue

        boxes_xyxy, confs, class_ids = det

        if len(boxes_xyxy) > 0:
            sv_detections = sv.Detections(
                xyxy=boxes_xyxy,
                confidence=confs,
                class_id=class_ids.astype(int),
            )
            tracked = tracker.update_with_detections(sv_detections)

            t_boxes = tracked.xyxy
            t_confs = tracked.confidence
            t_track_ids = tracked.tracker_id if tracked.tracker_id is not None else np.full(len(t_boxes), -1)
            t_class_ids = tracked.class_id if tracked.class_id is not None else np.zeros(len(t_boxes), dtype=int)
        else:
            t_boxes = np.array([]).reshape(0, 4)
            t_confs = np.array([])
            t_track_ids = np.array([])
            t_class_ids = np.array([])

        find_right = False
        find_left = False

        for idx in range(len(t_boxes)):
            x1, y1, x2, y2 = t_boxes[idx]
            conf = t_confs[idx]
            track_id_val = t_track_ids[idx]
            handedness = t_class_ids[idx]

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            is_near_edge = (cx < edge_left or cx > edge_right or
                           cy < edge_top or cy > edge_bottom)

            if is_near_edge and conf < min_edge_conf:
                continue

            if track_id_val == -1:
                if handedness > 0:
                    id = int(10000 + fallback_counter)
                else:
                    id = int(5000 + fallback_counter)
                fallback_counter += 1
            else:
                id = int(track_id_val)

            if id in track_last_seen:
                frames_since_last = t - track_last_seen[id]
                if frames_since_last > 10 and conf < min_edge_conf:
                    continue

            subj = {
                'frame': t,
                'det': True,
                'det_box': np.array([[x1, y1, x2, y2, conf]]),
                'det_handedness': np.array([handedness]),
                'is_near_edge': is_near_edge,
            }

            if (not find_right and handedness > 0) or (not find_left and handedness == 0):
                if id in tracks:
                    tracks[id].append(subj)
                else:
                    tracks[id] = [subj]

                track_last_seen[id] = t

                if handedness > 0:
                    find_right = True
                elif handedness == 0:
                    find_left = True

    tracks = np.array(tracks, dtype=object)
    boxes_ = np.array(all_boxes_raw, dtype=object)

    return boxes_, tracks


def parse_chunks(frame, boxes, min_len=16):
    """ If a track disappear in the middle,
     we separate it to different segments to estimate the HPS independently.
     If a segment is less than 16 frames, we get rid of it for now.
     """
    frame_chunks = []
    boxes_chunks = []
    step = frame[1:] - frame[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]

    start = 0
    for bk in breaks:
        f_chunk = frame[start:bk]
        b_chunk = boxes[start:bk]
        start = bk
        if len(f_chunk)>=min_len:
            frame_chunks.append(f_chunk)
            boxes_chunks.append(b_chunk)

        if bk==breaks[-1]:  # last chunk
            f_chunk = frame[bk:]
            b_chunk = boxes[bk:]
            if len(f_chunk)>=min_len:
                frame_chunks.append(f_chunk)
                boxes_chunks.append(b_chunk)

    return frame_chunks, boxes_chunks

def parse_chunks_hand_frame(frame):
    """ If a track disappear in the middle,
     we separate it to different segments to estimate the HPS independently.
     If a segment is less than 16 frames, we get rid of it for now.
     """
    frame_chunks = []
    step = frame[1:] - frame[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]

    start = 0
    for bk in breaks:
        f_chunk = frame[start:bk]
        start = bk
        if len(f_chunk) > 0:
            frame_chunks.append(f_chunk)

        if bk==breaks[-1]:  # last chunk
            f_chunk = frame[bk:]
            if len(f_chunk) > 0:
                frame_chunks.append(f_chunk)

    return frame_chunks
