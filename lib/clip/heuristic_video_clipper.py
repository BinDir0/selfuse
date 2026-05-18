#!/usr/bin/env python3
"""Heuristic temporal clipping for raw videos.

This is the generic pipeline-facing form of the BuildAI temporal filtering
idea: sample frames, score hand/object interaction with inexpensive visual
gates, merge valid spans, and write clipped MP4 files plus a JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.clip.clip_config import get_pipeline_config  # noqa: E402


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


@dataclass(frozen=True)
class ClipInterval:
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "start_sec": float(self.start_sec),
            "end_sec": float(self.end_sec),
            "score": float(self.score),
        }


def _read_yaml(path: str | Path | None) -> dict:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_clip_config(config_path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = get_pipeline_config(config_path)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def _heuristic_section(cfg: dict) -> dict:
    return cfg.get("heuristic") or cfg.get("stage1") or {}


def discover_videos(video_root: str | Path) -> list[Path]:
    root = Path(video_root).expanduser().resolve()
    if root.is_file() and root.suffix.lower() in VIDEO_EXTENSIONS:
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"video_root not found: {root}")
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(root.rglob(f"*{ext}"))
    return sorted(videos)


def _roi_bounds(width: int, height: int, roi) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in roi]
    return (
        max(0, min(width, int(round(x1 * width)))),
        max(0, min(height, int(round(y1 * height)))),
        max(0, min(width, int(round(x2 * width)))),
        max(0, min(height, int(round(y2 * height)))),
    )


def _load_yolo(model_path: str | None):
    if not model_path:
        return None
    path = Path(model_path).expanduser()
    if not path.is_file():
        return None
    try:
        from ultralytics import YOLO

        return YOLO(str(path))
    except Exception:
        return None


def _box_intersects_roi(box, roi_px) -> bool:
    bx1, by1, bx2, by2 = [float(v) for v in box]
    rx1, ry1, rx2, ry2 = roi_px
    ix1 = max(bx1, rx1)
    iy1 = max(by1, ry1)
    ix2 = min(bx2, rx2)
    iy2 = min(by2, ry2)
    return ix2 > ix1 and iy2 > iy1


def _detect_gate(model, frame_bgr, *, gate_a: dict) -> bool:
    if model is None:
        return True
    height, width = frame_bgr.shape[:2]
    roi_px = _roi_bounds(width, height, gate_a.get("roi", [0.0, 0.0, 1.0, 1.0]))
    min_area = float(gate_a.get("min_area_ratio", 0.0)) * width * height
    max_area = float(gate_a.get("max_area_ratio", 1.0)) * width * height
    conf_thresh = float(gate_a.get("conf_thresh", gate_a.get("box_conf_thresh", 0.25)))
    try:
        result = model.predict(frame_bgr, verbose=False, conf=conf_thresh)[0]
        boxes = result.boxes
        if boxes is None:
            return False
        for xyxy, conf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
            if float(conf) < conf_thresh:
                continue
            bx1, by1, bx2, by2 = [float(v) for v in xyxy]
            area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            if area < min_area or area > max_area:
                continue
            if _box_intersects_roi((bx1, by1, bx2, by2), roi_px):
                return True
    except Exception:
        return True
    return False


def _motion_gate(prev_gray, gray, *, gate_b: dict, gate_c: dict, roi_px) -> tuple[bool, float]:
    import cv2

    if prev_gray is None:
        return False, 0.0

    x1, y1, x2, y2 = roi_px
    roi_prev = prev_gray[y1:y2, x1:x2]
    roi_gray = gray[y1:y2, x1:x2]
    if roi_prev.size == 0 or roi_gray.size == 0:
        return False, 0.0

    diff_score = float(np.mean(cv2.absdiff(roi_prev, roi_gray))) / 255.0
    hand_motion = diff_score >= float(gate_c.get("hand_motion_thresh", 0.012))

    max_corners = int(gate_b.get("flow_max_corners", 128))
    points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=max_corners,
        qualityLevel=float(gate_b.get("flow_quality_level", 0.01)),
        minDistance=float(gate_b.get("flow_min_distance", 7)),
        blockSize=int(gate_b.get("flow_block_size", 7)),
    )
    if points is None or len(points) < int(gate_b.get("flow_min_tracked", 24)):
        return hand_motion, diff_score

    next_points, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None)
    if next_points is None or status is None:
        return hand_motion, diff_score
    valid = status.reshape(-1) > 0
    if int(valid.sum()) < int(gate_b.get("flow_min_tracked", 24)):
        return hand_motion, diff_score
    flow = next_points[valid].reshape(-1, 2) - points[valid].reshape(-1, 2)
    median_disp = float(np.median(np.linalg.norm(flow, axis=1)))
    camera_disp_thresh = float(gate_b.get("camera_disp_thresh", 0.2))
    stable_camera = median_disp <= camera_disp_thresh * max(prev_gray.shape[:2])
    return bool(hand_motion and stable_camera), diff_score


def _merge_valid_samples(samples: list[tuple[int, bool, float]], *, fps: float, skip_frames: int, min_keep_sec: float) -> list[ClipInterval]:
    intervals: list[ClipInterval] = []
    start = None
    scores = []
    last_frame = None
    min_frames = max(1, int(round(float(min_keep_sec) * fps)))

    for frame_idx, is_valid, score in samples:
        if is_valid:
            if start is None:
                start = frame_idx
                scores = []
            scores.append(score)
            last_frame = frame_idx
            continue
        if start is not None and last_frame is not None:
            end = last_frame + skip_frames
            if end - start >= min_frames:
                intervals.append(
                    ClipInterval(start, end, start / fps, end / fps, float(np.mean(scores) if scores else 0.0))
                )
            start = None
            scores = []
            last_frame = None

    if start is not None and last_frame is not None:
        end = last_frame + skip_frames
        if end - start >= min_frames:
            intervals.append(ClipInterval(start, end, start / fps, end / fps, float(np.mean(scores) if scores else 0.0)))
    return intervals


def analyze_video_intervals(video_path: str | Path, cfg: dict, *, model=None) -> tuple[list[ClipInterval], dict]:
    import cv2

    video_path = Path(video_path)
    heuristic = _heuristic_section(cfg)
    gate_a = heuristic.get("gate_a") or {}
    gate_b = heuristic.get("gate_b") or {}
    gate_c = heuristic.get("gate_c") or {}
    skip_frames = max(1, int(heuristic.get("skip_frames", 15)))
    decode_size = (
        int(heuristic.get("decode_width", 448)),
        int(heuristic.get("decode_height", 256)),
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    roi_px = _roi_bounds(decode_size[0], decode_size[1], gate_a.get("roi", [0.0, 0.0, 1.0, 1.0]))

    samples = []
    prev_gray = None
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue
            frame_small = cv2.resize(frame, decode_size)
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            detect_ok = _detect_gate(model, frame_small, gate_a=gate_a)
            motion_ok, score = _motion_gate(prev_gray, gray, gate_b=gate_b, gate_c=gate_c, roi_px=roi_px)
            samples.append((frame_idx, bool(detect_ok and motion_ok), score))
            prev_gray = gray
            frame_idx += 1
    finally:
        cap.release()

    intervals = _merge_valid_samples(
        samples,
        fps=fps,
        skip_frames=skip_frames,
        min_keep_sec=float(gate_c.get("min_keep_sec", 2.0)),
    )
    return intervals, {
        "video": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "sample_count": len(samples),
        "valid_sample_count": sum(1 for _idx, valid, _score in samples if valid),
    }


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in path.stem).strip("._") or "clip"


def write_video_clip(source_video: str | Path, output_path: str | Path, interval: ClipInterval, *, output_size=None) -> dict:
    import cv2

    source_video = Path(source_video)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for clipping: {source_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if output_size:
        width, height = int(output_size[0]), int(output_size[1])
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create clipped video: {output_path}")
    written = 0
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(interval.start_frame))
        for frame_idx in range(int(interval.start_frame), int(interval.end_frame)):
            ok, frame = cap.read()
            if not ok:
                break
            if output_size:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        cap.release()
    return {"path": str(output_path), "frames": written, "fps": fps}


def run_heuristic_clipping(
    *,
    video_paths: Iterable[str | Path],
    output_root: str | Path,
    config: dict,
    source_root: str | Path | None = None,
    report_out: str | Path | None = None,
) -> dict:
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_root_path = Path(source_root).expanduser().resolve() if source_root else None
    paths_cfg = config.get("paths") or {}
    heuristic = _heuristic_section(config)
    model = _load_yolo(paths_cfg.get("model_path") or heuristic.get("model_path"))
    fallback_full_video = bool(heuristic.get("fallback_full_video", False))
    output_size = heuristic.get("output_size")

    videos = [Path(path).expanduser().resolve() for path in video_paths]
    records = []
    for video_path in tqdm(videos, desc="Heuristic clipping"):
        intervals, metrics = analyze_video_intervals(video_path, config, model=model)
        if not intervals and fallback_full_video:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            if total > 0:
                intervals = [ClipInterval(0, total, 0.0, total / fps, 0.0)]

        rel_parent = Path()
        if source_root_path and video_path.is_relative_to(source_root_path):
            rel_parent = video_path.relative_to(source_root_path).parent
        out_dir = output_root / rel_parent
        clip_records = []
        for idx, interval in enumerate(intervals):
            out_name = f"{_safe_stem(video_path)}_clip{idx:03d}.mp4"
            out_path = out_dir / out_name
            write_meta = write_video_clip(video_path, out_path, interval, output_size=output_size)
            if write_meta["frames"] <= 0:
                out_path.unlink(missing_ok=True)
                continue
            clip_records.append({**interval.to_dict(), **write_meta})
        records.append({**metrics, "clips": clip_records, "kept": len(clip_records)})

    report = {
        "output_root": str(output_root),
        "source_root": str(source_root_path) if source_root_path else None,
        "videos": records,
        "summary": {
            "source_videos": len(records),
            "kept_clips": sum(len(item["clips"]) for item in records),
        },
    }
    if report_out:
        report_path = Path(report_out)
    else:
        report_path = output_root / "_clip_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run generic heuristic video clipping.")
    parser.add_argument("--video", default=None, help="Single input video.")
    parser.add_argument("--video_root", default=None, help="Root containing input videos.")
    parser.add_argument("--output_root", required=True, help="Directory for clipped MP4 files.")
    parser.add_argument("--config", default=None, help="Heuristic clipping YAML config path.")
    parser.add_argument("--override_config", default=None, help="Optional YAML overrides.")
    parser.add_argument("--report_out", default=None, help="Optional JSON report path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = get_parser().parse_args(argv)
    if not args.video and not args.video_root:
        raise SystemExit("Provide --video or --video_root")
    override = _read_yaml(args.override_config)
    cfg = load_clip_config(args.config, override)
    if args.video:
        videos = [Path(args.video)]
        source_root = Path(args.video).parent
    else:
        videos = discover_videos(args.video_root)
        source_root = Path(args.video_root)
    report = run_heuristic_clipping(
        video_paths=videos,
        output_root=args.output_root,
        config=cfg,
        source_root=source_root,
        report_out=args.report_out,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
