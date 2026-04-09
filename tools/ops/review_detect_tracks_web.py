#!/usr/bin/env python3
"""Browser-based reviewer for detect_track boxes on headless servers."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


JPEG_QUALITY = 90


@dataclass
class ReviewedFrame:
    frame_idx: int
    left_accepted: bool
    right_accepted: bool
    left_box: list[float] | None
    right_box: list[float] | None


def build_parser():
    parser = argparse.ArgumentParser(description="Review detect_track boxes in a browser")
    parser.add_argument("--video-path", required=True, help="Original video path used by HaWoR; seq_folder is inferred from it")
    parser.add_argument("--frame-step", type=int, default=5, help="Review every Nth frame")
    parser.add_argument("--start-frame", type=int, default=0, help="Optional first frame to review")
    parser.add_argument("--end-frame", type=int, default=None, help="Optional exclusive end frame")
    parser.add_argument("--edge-margin-ratio", type=float, default=0.1, help="Same edge heuristic ratio used in detect_track")
    parser.add_argument("--edited-conf", type=float, default=1.0, help="Confidence assigned to manually edited boxes")
    parser.add_argument("--output-tag", default="manual_review", help="Suffix for reviewed track directory when not applying in place")
    parser.add_argument("--report-out", default=None, help="Optional JSON report path")
    parser.add_argument("--apply-in-place", action="store_true", help="Overwrite the detected tracks folder after backing it up")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8765, help="HTTP bind port")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    return parser


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
    for items in tracks.values():
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


def _clone_frame_boxes(frame_boxes: dict[str, list[np.ndarray | None]]) -> dict[str, list[np.ndarray | None]]:
    return {
        side: [None if box is None else np.asarray(box, dtype=np.float32).copy() for box in boxes]
        for side, boxes in frame_boxes.items()
    }


def _interpolate_box(box_a: np.ndarray, box_b: np.ndarray, alpha: float) -> np.ndarray:
    box_a = np.asarray(box_a, dtype=np.float32)
    box_b = np.asarray(box_b, dtype=np.float32)
    return ((1.0 - alpha) * box_a + alpha * box_b).astype(np.float32)


def _build_sparse_manual_boxes(
    original_frame_boxes: dict[str, list[np.ndarray | None]],
    current_frame_boxes: dict[str, list[np.ndarray | None]],
    *,
    review_indices: list[int],
    accepted_frames_by_side: dict[str, set[int]],
) -> dict[str, list[np.ndarray | None]]:
    result = _clone_frame_boxes(original_frame_boxes)
    if not review_indices:
        return result

    interval_start = int(review_indices[0])
    interval_end = int(review_indices[-1])

    for side in ("left", "right"):
        accepted = sorted(int(frame_idx) for frame_idx in accepted_frames_by_side[side] if frame_idx in review_indices)
        for frame_idx in range(interval_start, interval_end + 1):
            result[side][frame_idx] = None
        for frame_idx in accepted:
            box = current_frame_boxes[side][frame_idx]
            result[side][frame_idx] = None if box is None else np.asarray(box, dtype=np.float32).copy()
        for start_frame, end_frame in zip(accepted[:-1], accepted[1:]):
            start_box = current_frame_boxes[side][start_frame]
            end_box = current_frame_boxes[side][end_frame]
            if start_box is None or end_box is None:
                continue
            gap = end_frame - start_frame
            if gap <= 1:
                continue
            for frame_idx in range(start_frame + 1, end_frame):
                alpha = float(frame_idx - start_frame) / float(gap)
                result[side][frame_idx] = _interpolate_box(start_box, end_box, alpha)

    return result


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


def _box_to_json(box: np.ndarray | None):
    if box is None:
        return None
    arr = np.asarray(box, dtype=np.float32).reshape(5)
    return {
        "x1": float(arr[0]),
        "y1": float(arr[1]),
        "x2": float(arr[2]),
        "y2": float(arr[3]),
        "conf": float(arr[4]),
    }


class ReviewSession:
    def __init__(self, args):
        from lib.pipeline.frame_source import build_frame_source
        from lib.pipeline.stage_api import get_seq_folder, get_track_range

        self.args = args
        self.seq_folder = get_seq_folder(video_path=args.video_path)
        self.start_idx, self.end_idx = get_track_range(self.seq_folder, fast=True)
        self.tracks_dir = self.seq_folder / f"tracks_{self.start_idx}_{self.end_idx}"
        self.tracks_path = self.tracks_dir / "model_tracks.npy"
        if not self.tracks_path.exists():
            raise SystemExit(f"Missing detect_track output: {self.tracks_path}")

        self.frame_source = build_frame_source(args.video_path)
        self.num_frames = len(self.frame_source)
        self.tracks = _load_existing_tracks(self.tracks_path)
        self.original_frame_boxes = _frame_boxes_from_tracks(self.tracks, num_frames=self.num_frames)
        self.frame_boxes = _clone_frame_boxes(self.original_frame_boxes)

        review_start = max(0, int(args.start_frame))
        review_end = self.num_frames if args.end_frame is None else min(self.num_frames, int(args.end_frame))
        self.review_indices = list(range(review_start, review_end, max(1, int(args.frame_step))))
        if not self.review_indices:
            raise SystemExit("No frames selected for review")
        self.review_index_set = set(int(frame_idx) for frame_idx in self.review_indices)
        self.accepted_frames_by_side = {
            "left": set(),
            "right": set(),
        }

        sample_frame = self.frame_source.get_frame(0, rgb=False)
        self.image_height = int(sample_frame.shape[0])
        self.image_width = int(sample_frame.shape[1])
        self.position = 0

    def current_frame_idx(self) -> int:
        return int(self.review_indices[self.position])

    def get_frame_image(self, frame_idx: int) -> np.ndarray:
        return self.frame_source.get_frame(int(frame_idx), rgb=False).copy()

    def get_frame_payload(self, position: int | None = None):
        if position is None:
            position = self.position
        position = max(0, min(len(self.review_indices) - 1, int(position)))
        frame_idx = int(self.review_indices[position])
        return {
            "position": position,
            "total": len(self.review_indices),
            "frame_idx": frame_idx,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "left_box": _box_to_json(self.frame_boxes["left"][frame_idx]),
            "right_box": _box_to_json(self.frame_boxes["right"][frame_idx]),
            "left_accepted": frame_idx in self.accepted_frames_by_side["left"],
            "right_accepted": frame_idx in self.accepted_frames_by_side["right"],
            "accepted_left_total": len(self.accepted_frames_by_side["left"]),
            "accepted_right_total": len(self.accepted_frames_by_side["right"]),
            "frame_url": f"/api/frame/{frame_idx}.jpg",
        }

    def set_position(self, position: int):
        self.position = max(0, min(len(self.review_indices) - 1, int(position)))
        return self.get_frame_payload()

    def step(self, delta: int):
        return self.set_position(self.position + int(delta))

    def accept_frame(self, frame_idx: int):
        if frame_idx not in self.review_index_set:
            raise ValueError(f"frame {frame_idx} is not in the review set")
        self.accepted_frames_by_side["left"].add(int(frame_idx))
        self.accepted_frames_by_side["right"].add(int(frame_idx))

    def accept_side(self, frame_idx: int, side: str):
        side = side.lower()
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported side: {side}")
        if frame_idx not in self.review_index_set:
            raise ValueError(f"frame {frame_idx} is not in the review set")
        self.accepted_frames_by_side[side].add(int(frame_idx))

    def set_box(self, frame_idx: int, side: str, box_payload: dict):
        side = side.lower()
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported side: {side}")
        x1 = float(box_payload["x1"])
        y1 = float(box_payload["y1"])
        x2 = float(box_payload["x2"])
        y2 = float(box_payload["y2"])
        conf = float(box_payload.get("conf", self.args.edited_conf))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid box: x2/y2 must be larger than x1/y1")
        x1 = min(max(0.0, x1), float(self.image_width - 1))
        y1 = min(max(0.0, y1), float(self.image_height - 1))
        x2 = min(max(0.0, x2), float(self.image_width))
        y2 = min(max(0.0, y2), float(self.image_height))
        self.frame_boxes[side][frame_idx] = np.array([x1, y1, x2, y2, conf], dtype=np.float32)
        self.accept_side(frame_idx, side)

    def clear_box(self, frame_idx: int, side: str):
        side = side.lower()
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported side: {side}")
        self.frame_boxes[side][frame_idx] = None
        self.accept_side(frame_idx, side)

    def save(self):
        final_frame_boxes = _build_sparse_manual_boxes(
            self.original_frame_boxes,
            self.frame_boxes,
            review_indices=self.review_indices,
            accepted_frames_by_side=self.accepted_frames_by_side,
        )
        model_boxes, reviewed_tracks = _build_reviewed_tracks(
            final_frame_boxes,
            image_width=self.image_width,
            image_height=self.image_height,
            edge_margin_ratio=float(self.args.edge_margin_ratio),
        )
        reviewed_frames = [
            ReviewedFrame(
                frame_idx=int(frame_idx),
                left_accepted=int(frame_idx) in self.accepted_frames_by_side["left"],
                right_accepted=int(frame_idx) in self.accepted_frames_by_side["right"],
                left_box=None if self.frame_boxes["left"][frame_idx] is None else self.frame_boxes["left"][frame_idx].astype(np.float32).tolist(),
                right_box=None if self.frame_boxes["right"][frame_idx] is None else self.frame_boxes["right"][frame_idx].astype(np.float32).tolist(),
            )
            for frame_idx in self.review_indices
        ]

        if self.args.apply_in_place:
            backup_dir = self.tracks_dir.parent / f"{self.tracks_dir.name}.backup_manual_review"
            if backup_dir.exists():
                raise RuntimeError(f"Backup dir already exists, refusing to overwrite: {backup_dir}")
            shutil.copytree(self.tracks_dir, backup_dir)
            output_dir = self.tracks_dir
        else:
            output_dir = self.tracks_dir.parent / f"{self.tracks_dir.name}.{self.args.output_tag}"

        _write_review_outputs(
            output_dir,
            model_boxes=model_boxes,
            tracks=reviewed_tracks,
            reviewed_frames=reviewed_frames,
            track_range=(self.start_idx, self.end_idx),
            source_track_dir=self.tracks_dir,
            report_out=self.args.report_out,
        )
        return {
            "output_dir": str(output_dir),
            "backup_dir": str(backup_dir) if self.args.apply_in_place else None,
            "accepted_left_total": len(self.accepted_frames_by_side["left"]),
            "accepted_right_total": len(self.accepted_frames_by_side["right"]),
        }


def _encode_jpeg(image_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return encoded.tobytes()


def build_html():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Detect Track Review</title>
  <style>
    :root {
      --bg: #111418;
      --panel: #1a1f26;
      --panel-2: #232a33;
      --text: #eef3f8;
      --muted: #9fb0c0;
      --blue: #2b7fff;
      --red: #ff4b5c;
      --accent: #3bd18b;
      --line: #334050;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.4 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 360px 1fr;
    }
    .sidebar {
      background: var(--panel);
      border-right: 1px solid var(--line);
      padding: 18px;
      overflow-y: auto;
    }
    .main {
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    h1 {
      font-size: 18px;
      margin: 0 0 12px;
    }
    h2 {
      font-size: 13px;
      color: var(--muted);
      margin: 18px 0 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .meta, .box-grid {
      display: grid;
      gap: 8px;
    }
    .meta-row, .box-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 8px 10px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 10px;
    }
    .buttons {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .buttons.triple {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    button {
      appearance: none;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel-2);
      color: var(--text);
      padding: 10px 12px;
      cursor: pointer;
      font: inherit;
    }
    button:hover { filter: brightness(1.08); }
    button.primary { background: #17314b; border-color: #29598a; }
    button.good { background: #103325; border-color: #1a6d46; }
    button.warn { background: #3b1c21; border-color: #85404a; }
    .canvas-wrap {
      position: relative;
      display: inline-block;
      max-width: min(96vw, 1400px);
      max-height: calc(100vh - 80px);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      background: #000;
    }
    .canvas-wrap img, .canvas-wrap canvas {
      display: block;
      width: 100%;
      height: auto;
    }
    .canvas-wrap canvas {
      position: absolute;
      inset: 0;
      cursor: crosshair;
    }
    .status {
      color: var(--muted);
      min-height: 20px;
    }
    .legend {
      display: flex;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
    }
    .legend span::before {
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 6px;
      vertical-align: baseline;
    }
    .legend .left::before { background: var(--blue); }
    .legend .right::before { background: var(--red); }
    .kbd {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      color: var(--text);
    }
    @media (max-width: 1100px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <h1>Detect Track Review</h1>
      <div class="meta" id="meta"></div>
      <h2>Mode</h2>
      <div class="meta">
        <div class="meta-row"><span>Inside review interval</span><span>only accepted anchors + interpolation</span></div>
        <div class="meta-row"><span>Unaccepted sampled frames</span><span>ignored on save</span></div>
        <div class="meta-row"><span>Outside review interval</span><span>keep original detect</span></div>
      </div>
      <h2>Navigation</h2>
      <div class="buttons triple">
        <button id="prevBtn">Prev</button>
        <button id="skipBtn">Next</button>
        <button id="nextBtn" class="primary">Accept / Next</button>
      </div>
      <h2>Editing</h2>
      <div class="buttons">
        <button id="drawLeftBtn">Draw Left</button>
        <button id="drawRightBtn">Draw Right</button>
        <button id="clearLeftBtn" class="warn">Clear Left</button>
        <button id="clearRightBtn" class="warn">Clear Right</button>
      </div>
      <h2>Save</h2>
      <div class="buttons">
        <button id="saveBtn" class="good">Save</button>
        <button id="refreshBtn">Refresh</button>
      </div>
      <h2>Current Boxes</h2>
      <div class="box-grid" id="boxGrid"></div>
      <h2>Shortcuts</h2>
      <div class="meta">
        <div class="meta-row"><span class="kbd">A / Space / Enter</span><span>Accept and next</span></div>
        <div class="meta-row"><span class="kbd">N / Right</span><span>Next without accepting</span></div>
        <div class="meta-row"><span class="kbd">P</span><span>Previous</span></div>
        <div class="meta-row"><span class="kbd">L / R</span><span>Draw left / right</span></div>
        <div class="meta-row"><span class="kbd">X / Y</span><span>Clear left / right</span></div>
        <div class="meta-row"><span class="kbd">S</span><span>Save</span></div>
      </div>
    </aside>
    <main class="main">
      <div class="legend">
        <span class="left">Left hand</span>
        <span class="right">Right hand</span>
      </div>
      <div class="status" id="status">Loading...</div>
      <div class="canvas-wrap">
        <img id="frameImage" alt="frame">
        <canvas id="overlay"></canvas>
      </div>
    </main>
  </div>
  <script>
    const state = {
      frame: null,
      drawMode: null,
      dragStart: null,
      dragCurrent: null,
      imageReady: false,
    };

    const metaEl = document.getElementById("meta");
    const boxGridEl = document.getElementById("boxGrid");
    const statusEl = document.getElementById("status");
    const imageEl = document.getElementById("frameImage");
    const canvasEl = document.getElementById("overlay");
    const ctx = canvasEl.getContext("2d");

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function boxToText(box) {
      if (!box) return "None";
      return `${box.x1.toFixed(1)}, ${box.y1.toFixed(1)}, ${box.x2.toFixed(1)}, ${box.y2.toFixed(1)}  conf=${box.conf.toFixed(2)}`;
    }

    function renderMeta() {
      const f = state.frame;
      metaEl.innerHTML = `
        <div class="meta-row"><span>Frame</span><span>${f.frame_idx}</span></div>
        <div class="meta-row"><span>Review Index</span><span>${f.position + 1} / ${f.total}</span></div>
        <div class="meta-row"><span>Image Size</span><span>${f.image_width} x ${f.image_height}</span></div>
        <div class="meta-row"><span>Accepted Left Anchors</span><span>${f.accepted_left_total}</span></div>
        <div class="meta-row"><span>Accepted Right Anchors</span><span>${f.accepted_right_total}</span></div>
      `;
      boxGridEl.innerHTML = `
        <div class="box-row"><span>Left${f.left_accepted ? " [accepted]" : ""}</span><span>${boxToText(f.left_box)}</span></div>
        <div class="box-row"><span>Right${f.right_accepted ? " [accepted]" : ""}</span><span>${boxToText(f.right_box)}</span></div>
      `;
    }

    function syncCanvasSize() {
      const rect = imageEl.getBoundingClientRect();
      canvasEl.width = Math.max(1, Math.round(rect.width));
      canvasEl.height = Math.max(1, Math.round(rect.height));
      canvasEl.style.width = `${canvasEl.width}px`;
      canvasEl.style.height = `${canvasEl.height}px`;
    }

    function imageToCanvasBox(box) {
      if (!box || !state.frame) return null;
      const sx = canvasEl.width / state.frame.image_width;
      const sy = canvasEl.height / state.frame.image_height;
      return {
        x1: box.x1 * sx,
        y1: box.y1 * sy,
        x2: box.x2 * sx,
        y2: box.y2 * sy,
      };
    }

    function canvasToImageBox(rect) {
      const sx = state.frame.image_width / canvasEl.width;
      const sy = state.frame.image_height / canvasEl.height;
      const x1 = Math.min(rect.x1, rect.x2) * sx;
      const y1 = Math.min(rect.y1, rect.y2) * sy;
      const x2 = Math.max(rect.x1, rect.x2) * sx;
      const y2 = Math.max(rect.y1, rect.y2) * sy;
      return { x1, y1, x2, y2, conf: 1.0 };
    }

    function drawOneBox(box, color, label) {
      if (!box) return;
      const b = imageToCanvasBox(box);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
      ctx.fillStyle = color;
      ctx.font = "16px sans-serif";
      ctx.fillText(label, b.x1 + 4, Math.max(18, b.y1 - 6));
    }

    function redraw() {
      ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
      if (!state.frame) return;
      drawOneBox(state.frame.left_box, "#2b7fff", "L");
      drawOneBox(state.frame.right_box, "#ff4b5c", "R");
      if (state.dragStart && state.dragCurrent && state.drawMode) {
        const x1 = state.dragStart.x;
        const y1 = state.dragStart.y;
        const x2 = state.dragCurrent.x;
        const y2 = state.dragCurrent.y;
        ctx.strokeStyle = state.drawMode === "left" ? "#2b7fff" : "#ff4b5c";
        ctx.lineWidth = 2;
        ctx.setLineDash([8, 6]);
        ctx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
        ctx.setLineDash([]);
      }
    }

    async function api(url, options = {}) {
      const resp = await fetch(url, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      return data;
    }

    async function loadFrame(payload) {
      state.frame = payload;
      renderMeta();
      setStatus(`Frame ${payload.frame_idx}  review ${payload.position + 1}/${payload.total}`);
      state.imageReady = false;
      imageEl.src = `${payload.frame_url}?ts=${Date.now()}`;
    }

    async function refreshFrame() {
      const payload = await api("/api/frame");
      await loadFrame(payload);
    }

    async function step(delta) {
      const payload = await api("/api/nav", { method: "POST", body: JSON.stringify({ delta }) });
      await loadFrame(payload);
    }

    async function acceptAndStep(delta = 1) {
      const payload = await api("/api/accept", { method: "POST", body: JSON.stringify({ delta }) });
      await loadFrame(payload);
    }

    async function save() {
      setStatus("Saving...");
      const data = await api("/api/save", { method: "POST", body: JSON.stringify({}) });
      setStatus(`Saved to ${data.output_dir}${data.backup_dir ? `  backup=${data.backup_dir}` : ""}  left=${data.accepted_left_total} right=${data.accepted_right_total}`);
    }

    async function clearSide(side) {
      const data = await api("/api/clear_box", {
        method: "POST",
        body: JSON.stringify({ frame_idx: state.frame.frame_idx, side }),
      });
      await loadFrame(data);
    }

    async function commitDraw() {
      if (!state.dragStart || !state.dragCurrent || !state.drawMode) return;
      const rect = {
        x1: state.dragStart.x,
        y1: state.dragStart.y,
        x2: state.dragCurrent.x,
        y2: state.dragCurrent.y,
      };
      const w = Math.abs(rect.x2 - rect.x1);
      const h = Math.abs(rect.y2 - rect.y1);
      if (w < 3 || h < 3) {
        setStatus("Ignored tiny box");
        state.dragStart = null;
        state.dragCurrent = null;
        redraw();
        return;
      }
      const box = canvasToImageBox(rect);
      const data = await api("/api/set_box", {
        method: "POST",
        body: JSON.stringify({ frame_idx: state.frame.frame_idx, side: state.drawMode, box }),
      });
      state.drawMode = null;
      state.dragStart = null;
      state.dragCurrent = null;
      await loadFrame(data);
    }

    function pointerPos(event) {
      const rect = canvasEl.getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(rect.width, event.clientX - rect.left)),
        y: Math.max(0, Math.min(rect.height, event.clientY - rect.top)),
      };
    }

    imageEl.addEventListener("load", () => {
      syncCanvasSize();
      state.imageReady = true;
      redraw();
    });
    window.addEventListener("resize", () => {
      if (!state.imageReady) return;
      syncCanvasSize();
      redraw();
    });

    canvasEl.addEventListener("mousedown", (event) => {
      if (!state.drawMode) return;
      state.dragStart = pointerPos(event);
      state.dragCurrent = state.dragStart;
      redraw();
    });
    canvasEl.addEventListener("mousemove", (event) => {
      if (!state.dragStart) return;
      state.dragCurrent = pointerPos(event);
      redraw();
    });
    canvasEl.addEventListener("mouseup", async (event) => {
      if (!state.dragStart) return;
      state.dragCurrent = pointerPos(event);
      redraw();
      await commitDraw();
    });

    document.getElementById("prevBtn").addEventListener("click", () => step(-1));
    document.getElementById("skipBtn").addEventListener("click", () => step(1));
    document.getElementById("nextBtn").addEventListener("click", () => acceptAndStep(1));
    document.getElementById("drawLeftBtn").addEventListener("click", () => { state.drawMode = "left"; setStatus("Draw left box"); });
    document.getElementById("drawRightBtn").addEventListener("click", () => { state.drawMode = "right"; setStatus("Draw right box"); });
    document.getElementById("clearLeftBtn").addEventListener("click", () => clearSide("left"));
    document.getElementById("clearRightBtn").addEventListener("click", () => clearSide("right"));
    document.getElementById("saveBtn").addEventListener("click", save);
    document.getElementById("refreshBtn").addEventListener("click", refreshFrame);

    window.addEventListener("keydown", async (event) => {
      if (!state.frame) return;
      if (event.key === " " || event.key === "Enter" || event.key.toLowerCase() === "a") {
        event.preventDefault();
        await acceptAndStep(1);
        return;
      }
      if (event.key.toLowerCase() === "n" || event.key === "ArrowRight") {
        await step(1);
        return;
      }
      if (event.key.toLowerCase() === "p") {
        await step(-1);
        return;
      }
      if (event.key.toLowerCase() === "l") {
        state.drawMode = "left";
        setStatus("Draw left box");
        return;
      }
      if (event.key.toLowerCase() === "r") {
        state.drawMode = "right";
        setStatus("Draw right box");
        return;
      }
      if (event.key.toLowerCase() === "x") {
        await clearSide("left");
        return;
      }
      if (event.key.toLowerCase() === "y") {
        await clearSide("right");
        return;
      }
      if (event.key.toLowerCase() === "s") {
        await save();
      }
    });

    refreshFrame().catch((error) => {
      setStatus(String(error));
      console.error(error);
    });
  </script>
</body>
</html>
"""


def create_app(args):
    session = ReviewSession(args)
    app = Flask(__name__)

    @app.get("/")
    def index():
        return Response(build_html(), mimetype="text/html")

    @app.get("/api/frame")
    def api_frame():
        return jsonify(session.get_frame_payload())

    @app.get("/api/frame/<int:frame_idx>.jpg")
    def api_frame_jpg(frame_idx: int):
        image = session.get_frame_image(frame_idx)
        return Response(_encode_jpeg(image), mimetype="image/jpeg")

    @app.post("/api/nav")
    def api_nav():
        payload = request.get_json(force=True, silent=False) or {}
        delta = int(payload.get("delta", 0))
        return jsonify(session.step(delta))

    @app.post("/api/accept")
    def api_accept():
        payload = request.get_json(force=True, silent=False) or {}
        delta = int(payload.get("delta", 1))
        session.accept_frame(session.current_frame_idx())
        return jsonify(session.step(delta))

    @app.post("/api/set_box")
    def api_set_box():
        payload = request.get_json(force=True, silent=False) or {}
        frame_idx = int(payload["frame_idx"])
        if frame_idx not in session.review_indices:
            return jsonify({"error": f"frame {frame_idx} is not in the review set"}), 400
        session.set_box(frame_idx, payload["side"], payload["box"])
        return jsonify(session.get_frame_payload(session.position))

    @app.post("/api/clear_box")
    def api_clear_box():
        payload = request.get_json(force=True, silent=False) or {}
        frame_idx = int(payload["frame_idx"])
        if frame_idx not in session.review_indices:
            return jsonify({"error": f"frame {frame_idx} is not in the review set"}), 400
        session.clear_box(frame_idx, payload["side"])
        return jsonify(session.get_frame_payload(session.position))

    @app.post("/api/save")
    def api_save():
        try:
            return jsonify(session.save())
        except Exception as error:
            return jsonify({"error": str(error)}), 400

    @app.get("/api/health")
    def api_health():
        return jsonify(
            {
                "ok": True,
                "video_path": args.video_path,
                "review_frames": len(session.review_indices),
                "current_position": session.position,
                "current_frame_idx": session.current_frame_idx(),
                "accepted_left_total": len(session.accepted_frames_by_side["left"]),
                "accepted_right_total": len(session.accepted_frames_by_side["right"]),
            }
        )

    return app


def main():
    args = build_parser().parse_args()
    app = create_app(args)
    print(f"Serving detect review UI at http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
