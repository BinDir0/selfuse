#!/usr/bin/env python3
"""Interactive review tool for detect_track boxes with optional manual overrides."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LEFT_COLOR = (255, 0, 0)
RIGHT_COLOR = (0, 0, 255)
TEXT_COLOR = (245, 245, 245)
PANEL_COLOR = (18, 18, 18)


@dataclass
class ReviewedFrame:
    frame_idx: int
    left_box: list[float] | None
    right_box: list[float] | None


def build_parser():
    parser = argparse.ArgumentParser(description="Interactively review detect_track boxes and export reviewed tracks")
    parser.add_argument("--video-path", required=True, help="Original video path used by HaWoR; seq_folder is inferred from it")
    parser.add_argument("--frame-step", type=int, default=5, help="Review every Nth frame")
    parser.add_argument("--start-frame", type=int, default=0, help="Optional first frame to review")
    parser.add_argument("--end-frame", type=int, default=None, help="Optional exclusive end frame")
    parser.add_argument("--window-max-width", type=int, default=1600, help="Max display width for the review window")
    parser.add_argument("--window-max-height", type=int, default=1000, help="Max display height for the review window")
    parser.add_argument("--edge-margin-ratio", type=float, default=0.1, help="Same edge heuristic ratio used in detect_track")
    parser.add_argument("--edited-conf", type=float, default=1.0, help="Confidence assigned to manually edited boxes")
    parser.add_argument("--output-tag", default="manual_review", help="Suffix for reviewed track directory when not applying in place")
    parser.add_argument("--report-out", default=None, help="Optional JSON report path")
    parser.add_argument("--apply-in-place", action="store_true", help="Overwrite the detected tracks folder after backing it up")
    return parser


def _fit_scale(width: int, height: int, max_width: int, max_height: int) -> float:
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    return float(max(scale, 1e-6))


def _scale_box(box: np.ndarray | None, scale: float):
    if box is None:
        return None
    result = np.asarray(box, dtype=np.float32).copy()
    result[:4] *= scale
    return result


def _unscale_xywh(roi, scale: float) -> np.ndarray | None:
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return None
    inv = 1.0 / max(scale, 1e-8)
    x1 = float(x) * inv
    y1 = float(y) * inv
    x2 = float(x + w) * inv
    y2 = float(y + h) * inv
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _draw_box(image: np.ndarray, box: np.ndarray | None, color, label: str):
    if box is None:
        return
    x1, y1, x2, y2 = np.round(np.asarray(box[:4], dtype=np.float32)).astype(np.int32)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
    cv2.putText(
        image,
        label,
        (x1 + 4, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def _overlay_ui(image: np.ndarray, *, frame_idx: int, review_idx: int, total_reviews: int):
    overlay = image.copy()
    panel_h = 96
    cv2.rectangle(overlay, (0, 0), (image.shape[1], panel_h), PANEL_COLOR, -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0, dst=image)

    lines = [
        f"frame={frame_idx} review={review_idx + 1}/{total_reviews}",
        "space/enter: accept  p: prev  l: redraw-left  r: redraw-right  b: redraw-both",
        "x: clear-left  y: clear-right  c: clear-both  s: save  q: quit-without-save",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (14, 28 + idx * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65 if idx == 0 else 0.55,
            TEXT_COLOR,
            2 if idx == 0 else 1,
            cv2.LINE_AA,
        )


def _decode_track_side(track_item: dict) -> str:
    handed = np.asarray(track_item["det_handedness"]).reshape(-1)
    value = int(handed[0]) if handed.size else 0
    return "right" if value > 0 else "left"


def _load_existing_tracks(tracks_path: Path) -> dict:
    return np.load(str(tracks_path), allow_pickle=True).item()


def _frame_boxes_from_tracks(tracks: dict, num_frames: int) -> dict[str, list[np.ndarray | None]]:
    per_side = {
        "left": [None] * num_frames,
        "right": [None] * num_frames,
    }
    for _track_id, items in tracks.items():
        for item in items:
            if not item.get("det", False):
                continue
            frame_idx = int(item["frame"])
            if frame_idx < 0 or frame_idx >= num_frames:
                continue
            side = _decode_track_side(item)
            box = np.asarray(item["det_box"], dtype=np.float32).reshape(-1, 5)[0]
            current = per_side[side][frame_idx]
            if current is None or float(box[4]) >= float(current[4]):
                per_side[side][frame_idx] = box.copy()
    return per_side


def _edge_flag(box: np.ndarray, width: int, height: int, margin_ratio: float) -> bool:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    edge_left = width * margin_ratio
    edge_right = width * (1.0 - margin_ratio)
    edge_top = height * margin_ratio
    edge_bottom = height * (1.0 - margin_ratio)
    return bool(cx < edge_left or cx > edge_right or cy < edge_top or cy > edge_bottom)


def _build_reviewed_tracks(
    frame_boxes: dict[str, list[np.ndarray | None]],
    *,
    image_width: int,
    image_height: int,
    edge_margin_ratio: float,
) -> tuple[np.ndarray, dict]:
    num_frames = len(frame_boxes["left"])
    model_boxes = np.empty((num_frames,), dtype=object)
    tracks = {}

    left_items = []
    right_items = []
    for frame_idx in range(num_frames):
        frame_entries = []
        for side_name, handedness, target in (("left", 0, left_items), ("right", 1, right_items)):
            box = frame_boxes[side_name][frame_idx]
            if box is None:
                continue
            arr = np.asarray(box, dtype=np.float32).reshape(5)
            frame_entries.append(arr.copy())
            target.append(
                {
                    "frame": int(frame_idx),
                    "det": True,
                    "det_box": arr[None, :].astype(np.float32),
                    "det_handedness": np.array([handedness], dtype=np.int64),
                    "is_near_edge": _edge_flag(arr, image_width, image_height, edge_margin_ratio),
                }
            )
        model_boxes[frame_idx] = np.stack(frame_entries, axis=0).astype(np.float32) if frame_entries else np.zeros((0, 5), dtype=np.float32)

    if left_items:
        tracks[0] = left_items
    if right_items:
        tracks[1] = right_items
    return model_boxes, tracks


def _write_review_outputs(
    output_dir: Path,
    *,
    model_boxes: np.ndarray,
    tracks: dict,
    reviewed_frames: list[ReviewedFrame],
    track_range: tuple[int, int],
    source_track_dir: Path,
    report_out: str | None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(output_dir / "model_boxes.npy"), model_boxes)
    np.save(str(output_dir / "model_tracks.npy"), tracks)

    payload = {
        "source_track_dir": str(source_track_dir.resolve()),
        "output_track_dir": str(output_dir.resolve()),
        "track_range": [int(track_range[0]), int(track_range[1])],
        "reviewed_frames": [asdict(item) for item in reviewed_frames],
    }
    (output_dir / "reviewed_boxes.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if report_out:
        Path(report_out).expanduser().resolve().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = build_parser().parse_args()
    from lib.pipeline.stage_api import get_seq_folder, get_track_range
    from lib.pipeline.frame_source import build_frame_source

    seq_folder = get_seq_folder(video_path=args.video_path)
    start_idx, end_idx = get_track_range(seq_folder, fast=True)
    tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"
    tracks_path = tracks_dir / "model_tracks.npy"
    if not tracks_path.exists():
        raise SystemExit(f"Missing detect_track output: {tracks_path}")

    frame_source = build_frame_source(args.video_path)
    num_frames = len(frame_source)
    tracks = _load_existing_tracks(tracks_path)
    frame_boxes = _frame_boxes_from_tracks(tracks, num_frames=num_frames)

    review_start = max(0, int(args.start_frame))
    review_end = num_frames if args.end_frame is None else min(num_frames, int(args.end_frame))
    review_indices = list(range(review_start, review_end, max(1, int(args.frame_step))))
    if not review_indices:
        raise SystemExit("No frames selected for review")

    window_name = "review_detect_tracks"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(args.window_max_width, 1400), min(args.window_max_height, 900))

    pos = 0
    while 0 <= pos < len(review_indices):
        frame_idx = int(review_indices[pos])
        image = frame_source.get_frame(frame_idx, rgb=False).copy()
        scale = _fit_scale(image.shape[1], image.shape[0], args.window_max_width, args.window_max_height)
        display = cv2.resize(
            image,
            (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
            interpolation=cv2.INTER_LINEAR,
        )

        left_box = _scale_box(frame_boxes["left"][frame_idx], scale)
        right_box = _scale_box(frame_boxes["right"][frame_idx], scale)
        _draw_box(display, left_box, LEFT_COLOR, "L")
        _draw_box(display, right_box, RIGHT_COLOR, "R")
        _overlay_ui(display, frame_idx=frame_idx, review_idx=pos, total_reviews=len(review_indices))
        cv2.imshow(window_name, display)

        key = cv2.waitKeyEx(0)
        if key in (13, 32, ord("a"), ord("A")):
            pos += 1
            continue
        if key in (ord("p"), ord("P"), 2424832):
            pos = max(0, pos - 1)
            continue
        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyAllWindows()
            raise SystemExit("Quit without saving")
        if key in (ord("s"), ord("S")):
            break

        def edit_side(side_name: str):
            fresh = frame_source.get_frame(frame_idx, rgb=False).copy()
            preview = cv2.resize(
                fresh,
                (max(1, int(round(fresh.shape[1] * scale))), max(1, int(round(fresh.shape[0] * scale)))),
                interpolation=cv2.INTER_LINEAR,
            )
            other_side = "right" if side_name == "left" else "left"
            _draw_box(preview, _scale_box(frame_boxes[other_side][frame_idx], scale), RIGHT_COLOR if other_side == "right" else LEFT_COLOR, "R" if other_side == "right" else "L")
            roi = cv2.selectROI(window_name, preview, showCrosshair=True, fromCenter=False)
            box_xyxy = _unscale_xywh(roi, scale)
            if box_xyxy is not None:
                conf = float(args.edited_conf)
                frame_boxes[side_name][frame_idx] = np.array([box_xyxy[0], box_xyxy[1], box_xyxy[2], box_xyxy[3], conf], dtype=np.float32)

        if key in (ord("l"), ord("L")):
            edit_side("left")
            continue
        if key in (ord("r"), ord("R")):
            edit_side("right")
            continue
        if key in (ord("b"), ord("B")):
            edit_side("left")
            edit_side("right")
            continue
        if key in (ord("x"), ord("X")):
            frame_boxes["left"][frame_idx] = None
            continue
        if key in (ord("y"), ord("Y")):
            frame_boxes["right"][frame_idx] = None
            continue
        if key in (ord("c"), ord("C"), 255):
            frame_boxes["left"][frame_idx] = None
            frame_boxes["right"][frame_idx] = None
            continue

    cv2.destroyAllWindows()

    sample_frame = frame_source.get_frame(0, rgb=False)
    model_boxes, reviewed_tracks = _build_reviewed_tracks(
        frame_boxes,
        image_width=sample_frame.shape[1],
        image_height=sample_frame.shape[0],
        edge_margin_ratio=float(args.edge_margin_ratio),
    )
    reviewed_frames = [
        ReviewedFrame(
            frame_idx=int(frame_idx),
            left_box=None if frame_boxes["left"][frame_idx] is None else frame_boxes["left"][frame_idx].astype(np.float32).tolist(),
            right_box=None if frame_boxes["right"][frame_idx] is None else frame_boxes["right"][frame_idx].astype(np.float32).tolist(),
        )
        for frame_idx in review_indices
    ]

    if args.apply_in_place:
        backup_dir = tracks_dir.parent / f"{tracks_dir.name}.backup_manual_review"
        if backup_dir.exists():
            raise SystemExit(f"Backup dir already exists, refusing to overwrite: {backup_dir}")
        shutil.copytree(tracks_dir, backup_dir)
        output_dir = tracks_dir
    else:
        output_dir = tracks_dir.parent / f"{tracks_dir.name}.{args.output_tag}"

    _write_review_outputs(
        output_dir,
        model_boxes=model_boxes,
        tracks=reviewed_tracks,
        reviewed_frames=reviewed_frames,
        track_range=(start_idx, end_idx),
        source_track_dir=tracks_dir,
        report_out=args.report_out,
    )
    print(f"Saved reviewed detect outputs to: {output_dir}", flush=True)
    if args.apply_in_place:
        print(f"Original detect outputs backed up at: {backup_dir}", flush=True)


if __name__ == "__main__":
    main()
