#!/usr/bin/env python3
"""Visualize the heuristic *filtering* stage for pipeline figures.

This produces figure material for the first pipeline stage (heuristic clipping).
It reuses the real gate logic / thresholds from
``lib/clip/heuristic_video_clipper.py`` so the overlays match what the pipeline
actually decides, and it auto-scans video(s) to find one representative frame
for each of three cases:

  A) gate_a reject by SIZE  -> a YOLO hand box is detected (conf ok, inside the
     central ROI) but rejected because its area is outside
     [min_area_ratio, max_area_ratio]  (e.g. a bystander hand too small / a hand
     too close/too big). Draws every YOLO box (green = qualifies, red = rejected
     with reason) plus the central judging ROI box.
  B) gate_b reject by FLOW  -> camera moves too much: LK optical-flow median
     displacement exceeds camera_disp_thresh * max(H, W). Draws the flow vectors
     coloured by magnitude plus the median / threshold annotation.
  C) PASS                   -> a frame that clears all gates. Draws both the
     boxes + ROI and the flow vectors.

It runs the detector + optical flow only on the few sampled frames (same
sampling stride as the pipeline), so it is light; still, run it on the GPU /
dataset machine where ultralytics + the detector checkpoint live.

Typical use:
  python scripts/viz_filter_stage.py \
      --video_root /path/to/raw_videos \
      --out_dir docs/figure_assets/filter_stage \
      --topk 3 --clip

Outputs into --out_dir:
  <stem>_f<idx>_caseA_size.png / _caseB_flow.png / _caseC_pass.png
  (and *_clip.mp4 per case if --clip)
  _filter_viz_report.json   (every candidate frame + its gate metrics)

Range mode (export every frame in [start, end) as annotated jpgs so the user
can pick the demo frame themselves):
  python scripts/viz_filter_stage.py \
      --video /path/to/video.mp4 \
      --start 0 --end 600 \
      --out_dir docs/figure_assets/filter_stage_range

Outputs into --out_dir/<stem>/:
  frame_000123.jpg ...           (caseC-style overlay: ROI+box+flow+banner)
  stats.jsonl                    (per-frame gate metrics, one json per line)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.clip.heuristic_video_clipper import (  # noqa: E402
    _heuristic_section,
    _load_yolo,
    _roi_bounds,
    discover_videos,
    load_clip_config,
)

DEFAULT_MODEL = PROJECT_ROOT / "weights" / "external" / "detector.pt"

# ---- colours (BGR) ----
C_ROI = (0, 215, 255)       # amber  -> central judging ROI
C_OK = (80, 220, 80)        # green  -> qualifying box
C_BAD = (60, 60, 235)       # red    -> rejected box
C_TEXT = (255, 255, 255)
C_PASS = (90, 200, 90)
C_REJECT = (60, 60, 235)


# --------------------------------------------------------------------------- #
# instrumented per-frame analysis (mirrors _detect_gate / _motion_gate)
# --------------------------------------------------------------------------- #
def _detect_boxes(model, frame_small, gate_a):
    """Return (boxes, qualified_count, gateA_pass). Each box mirrors the gate's
    per-box accept/reject decision and records *why* it was rejected."""
    h, w = frame_small.shape[:2]
    roi_px = _roi_bounds(w, h, gate_a.get("roi", [0.0, 0.0, 1.0, 1.0]))
    frame_area = float(w * h)
    min_area = float(gate_a.get("min_area_ratio", 0.0)) * frame_area
    max_area = float(gate_a.get("max_area_ratio", 1.0)) * frame_area
    conf_thresh = float(gate_a.get("conf_thresh", gate_a.get("box_conf_thresh", 0.25)))
    min_hands = max(1, int(gate_a.get("min_hands", 2)))

    boxes = []
    qualified = 0
    if model is None:
        return boxes, qualified, True
    try:
        result = model.predict(frame_small, verbose=False, conf=conf_thresh)[0]
        raw = result.boxes
        if raw is None:
            return boxes, 0, False
        rx1, ry1, rx2, ry2 = roi_px
        for xyxy, conf in zip(raw.xyxy.cpu().numpy(), raw.conf.cpu().numpy()):
            bx1, by1, bx2, by2 = [float(v) for v in xyxy]
            conf = float(conf)
            area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            area_ratio = area / frame_area
            ix1, iy1 = max(bx1, rx1), max(by1, ry1)
            ix2, iy2 = min(bx2, rx2), min(by2, ry2)
            in_roi = ix2 > ix1 and iy2 > iy1
            reason = None
            if conf < conf_thresh:
                reason = "low_conf"
            elif area < min_area:
                reason = "too_small"
            elif area > max_area:
                reason = "too_big"
            elif not in_roi:
                reason = "outside_roi"
            qualifies = reason is None
            if qualifies:
                qualified += 1
            boxes.append({
                "xyxy": [bx1, by1, bx2, by2],
                "conf": conf,
                "area_ratio": area_ratio,
                "in_roi": in_roi,
                "qualifies": qualifies,
                "reason": reason,
            })
    except Exception as exc:  # pragma: no cover - matches gate's permissive except
        return boxes, qualified, True
    return boxes, qualified, qualified >= min_hands


def _flow(prev_gray, gray, gate_b, gate_c, roi_px):
    """Return flow detail dict mirroring _motion_gate's numbers."""
    out = {
        "have_flow": False,
        "pts": None, "nxt": None, "valid": None,
        "median_disp": 0.0, "thresh_px": 0.0,
        "stable_camera": False, "diff_score": 0.0, "hand_motion": False,
        "gateB_pass": False,
    }
    if prev_gray is None:
        return out
    x1, y1, x2, y2 = roi_px
    roi_prev = prev_gray[y1:y2, x1:x2]
    roi_gray = gray[y1:y2, x1:x2]
    if roi_prev.size == 0 or roi_gray.size == 0:
        return out
    diff_score = float(np.mean(cv2.absdiff(roi_prev, roi_gray))) / 255.0
    out["diff_score"] = diff_score
    out["hand_motion"] = diff_score >= float(gate_c.get("hand_motion_thresh", 0.012))

    min_tracked = int(gate_b.get("flow_min_tracked", 24))
    pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=int(gate_b.get("flow_max_corners", 128)),
        qualityLevel=float(gate_b.get("flow_quality_level", 0.01)),
        minDistance=float(gate_b.get("flow_min_distance", 7)),
        blockSize=int(gate_b.get("flow_block_size", 7)),
    )
    thresh_px = float(gate_b.get("camera_disp_thresh", 0.2)) * max(prev_gray.shape[:2])
    out["thresh_px"] = thresh_px
    if pts is None or len(pts) < min_tracked:
        # gate returns (hand_motion, diff): camera assumed stable for stitching
        out["gateB_pass"] = out["hand_motion"]
        return out
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
    if nxt is None or status is None:
        out["gateB_pass"] = out["hand_motion"]
        return out
    valid = status.reshape(-1) > 0
    if int(valid.sum()) < min_tracked:
        out["gateB_pass"] = out["hand_motion"]
        return out
    flow = nxt[valid].reshape(-1, 2) - pts[valid].reshape(-1, 2)
    median_disp = float(np.median(np.linalg.norm(flow, axis=1)))
    stable = median_disp <= thresh_px
    out.update({
        "have_flow": True,
        "pts": pts[valid].reshape(-1, 2),
        "nxt": nxt[valid].reshape(-1, 2),
        "median_disp": median_disp,
        "stable_camera": stable,
        "gateB_pass": bool(out["hand_motion"] and stable),
    })
    # extra: single-step LK on a fixed 4x7 grid spread across the FULL frame
    # (visualisation only; gate B's decision still uses goodFeaturesToTrack
    # restricted to the ROI in the path above).
    try:
        Hf, Wf = prev_gray.shape[:2]
        rows, cols = 4, 7
        xs = np.linspace(Wf / (2 * cols), Wf - Wf / (2 * cols), cols, dtype=np.float32)
        ys = np.linspace(Hf / (2 * rows), Hf - Hf / (2 * rows), rows, dtype=np.float32)
        gj, gi = np.meshgrid(xs, ys)
        grid_in = np.stack([gj.ravel(), gi.ravel()], axis=-1).reshape(-1, 1, 2).astype(np.float32)
        grid_nxt, gst, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, grid_in, None)
        if grid_nxt is not None and gst is not None:
            out["grid_pts"] = grid_in.reshape(-1, 2)
            out["grid_nxt"] = grid_nxt.reshape(-1, 2)
            out["grid_valid"] = (gst.reshape(-1) > 0)
            out["grid_shape"] = (rows, cols)
    except Exception:
        pass
    return out


def analyze(video_path, cfg, model):
    heuristic = _heuristic_section(cfg)
    gate_a = heuristic.get("gate_a") or {}
    gate_b = heuristic.get("gate_b") or {}
    gate_c = heuristic.get("gate_c") or {}
    skip = max(1, int(heuristic.get("skip_frames", 15)))
    dw = int(heuristic.get("decode_width", 448))
    dh = int(heuristic.get("decode_height", 256))
    roi_px = _roi_bounds(dw, dh, gate_a.get("roi", [0.0, 0.0, 1.0, 1.0]))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    Wf = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or dw)
    Hf = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or dh)

    samples = []
    prev_gray = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % skip != 0:
            idx += 1
            continue
        small = cv2.resize(frame, (dw, dh))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        boxes, qcount, gateA = _detect_boxes(model, small, gate_a)
        fl = _flow(prev_gray, gray, gate_b, gate_c, roi_px)
        samples.append({
            "frame_idx": idx,
            "boxes": boxes,
            "qualified": qcount,
            "gateA_pass": gateA,
            "flow": fl,
            "overall": bool(gateA and fl["gateB_pass"]),
        })
        prev_gray = gray
        idx += 1
    cap.release()
    meta = {
        "video": str(video_path), "fps": fps, "skip": skip,
        "decode": (dw, dh), "full": (Wf, Hf), "roi_px_small": roi_px,
        "gate_a": gate_a, "gate_b": gate_b, "gate_c": gate_c,
    }
    return samples, meta


# --------------------------------------------------------------------------- #
# candidate selection
# --------------------------------------------------------------------------- #
def _size_reject_score(s):
    """Higher = clearer 'detected hand rejected purely by size' example."""
    if s["gateA_pass"]:
        return -1.0
    best = -1.0
    for b in s["boxes"]:
        if b["reason"] in ("too_small", "too_big") and b["in_roi"]:
            # a confident, in-ROI hand killed only by size is the clearest story
            best = max(best, b["conf"])
    return best


def select_candidates(samples, topk):
    size_rej, flow_rej, passed = [], [], []
    for s in samples:
        sc = _size_reject_score(s)
        if sc >= 0:
            size_rej.append((sc, s))
        fl = s["flow"]
        if fl["have_flow"] and not fl["stable_camera"]:
            flow_rej.append((fl["median_disp"], s))
        if s["overall"]:
            passed.append((s["flow"]["diff_score"], s))
    size_rej.sort(key=lambda t: -t[0])
    flow_rej.sort(key=lambda t: -t[0])
    passed.sort(key=lambda t: -t[0])
    return {
        "caseA_size": [s for _, s in size_rej[:topk]],
        "caseB_flow": [s for _, s in flow_rej[:topk]],
        "caseC_pass": [s for _, s in passed[:topk]],
    }


# --------------------------------------------------------------------------- #
# drawing (overlays on the FULL-res frame; coords scaled from decode size)
# --------------------------------------------------------------------------- #
def _scaler(meta):
    dw, dh = meta["decode"]
    Wf, Hf = meta["full"]
    return Wf / dw, Hf / dh


def _label(img, text, org, bg, fg=C_TEXT, scale=0.5, thick=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), bg, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


def _banner(img, lines):
    pad = 8
    scale, thick = 0.55, 1
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0] for t, _ in lines]
    h = pad + sum(s[1] + 8 for s in sizes) + pad
    w = max(s[0] for s in sizes) + 2 * pad
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    y = pad
    for (t, col), (tw, th) in zip(lines, sizes):
        y += th + 4
        cv2.putText(img, t, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)
        y += 4


def _draw_roi(img, meta):
    sx, sy = _scaler(meta)
    x1, y1, x2, y2 = meta["roi_px_small"]
    p1 = (int(x1 * sx), int(y1 * sy))
    p2 = (int(x2 * sx), int(y2 * sy))
    # dashed-ish rectangle: just a thick amber box + label
    cv2.rectangle(img, p1, p2, C_ROI, 3)
    _label(img, "Central ROI (gate A)", (p1[0], max(p1[1], 22)), C_ROI, (0, 0, 0))


def _draw_boxes(img, sample, meta):
    sx, sy = _scaler(meta)
    reason_txt = {
        "too_small": "size<min", "too_big": "size>max",
        "outside_roi": "outside ROI", "low_conf": "low conf",
    }
    for b in sample["boxes"]:
        x1, y1, x2, y2 = b["xyxy"]
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        col = C_OK if b["qualifies"] else C_BAD
        cv2.rectangle(img, p1, p2, col, 2)
        tag = f"hand {b['conf']:.2f} | {b['area_ratio'] * 100:.1f}%"
        if not b["qualifies"]:
            tag += f" REJ:{reason_txt.get(b['reason'], b['reason'])}"
        _label(img, tag, (p1[0], max(p1[1], 18)), col, (0, 0, 0), scale=0.45)


def _flow_wheel_legend(img, *, radius=42, margin=18):
    """Paint a small Middlebury HSV color wheel (direction=hue, magnitude=sat)
    in the bottom-right corner so readers can decode the arrow colors."""
    H, W = img.shape[:2]
    cx = W - radius - margin
    cy = H - radius - margin - 14
    if cx - radius < 0 or cy - radius < 0:
        return
    rr = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(rr, rr, indexing="ij")
    rho = np.sqrt(xx * xx + yy * yy)
    inside = rho <= radius
    ang = np.arctan2(yy, xx)
    hue = ((ang + np.pi) / (2.0 * np.pi)) * 180.0
    sat = np.clip(rho / max(1.0, radius), 0.0, 1.0) * 255.0
    val = np.full_like(hue, 255.0)
    hsv = np.stack([hue, sat, val], axis=-1).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    y0, x0 = cy - radius, cx - radius
    roi = img[y0:y0 + 2 * radius + 1, x0:x0 + 2 * radius + 1]
    mask3 = np.repeat(inside[..., None], 3, axis=2)
    np.copyto(roi, bgr, where=mask3)
    cv2.circle(img, (cx, cy), radius, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 1, (240, 240, 240), -1, cv2.LINE_AA)
    label = "flow: hue=dir  sat=mag"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    tx = max(2, cx - tw // 2)
    ty = min(H - 4, cy + radius + 14)
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (240, 240, 240), 1, cv2.LINE_AA)


def _grid_seed_pts(roi_px, rows=4, cols=7):
    """Return (N, 2) float32 grid points (decode coords) inside the ROI."""
    x1, y1, x2, y2 = roi_px
    xs = np.linspace(x1 + (x2 - x1) / (2 * cols),
                     x2 - (x2 - x1) / (2 * cols), cols, dtype=np.float32)
    ys = np.linspace(y1 + (y2 - y1) / (2 * rows),
                     y2 - (y2 - y1) / (2 * rows), rows, dtype=np.float32)
    gj, gi = np.meshgrid(xs, ys)
    return np.stack([gj.ravel(), gi.ravel()], axis=-1).astype(np.float32)


def _grid_lk_trail(grays, *, rows=4, cols=7):
    """Backward LK chain so head dots stay anchored on the grid.

    Seed grid spans the FULL decoded frame (uniform 4x7 over [0, W) x [0, H))
    rather than just the gate's ROI - the visualisation is meant to read
    the whole image. Gate B's own median-flow decision is unaffected (it
    still runs on goodFeaturesToTrack inside the ROI in `_motion_gate`).

    Result:
      positions:  (T+1, N, 2) float32; positions[T] == seed grid; earlier
                  rows hold the backward-LK-tracked locations, frozen at
                  the last good spot once a tracker dies.
      first_idx:  (N,) int; the earliest step index for which a valid
                  backward position exists. first_idx[n] == T means tracking
                  died immediately and the caller should draw only a dot.
    """
    if len(grays) < 2:
        return None, None
    H, W = grays[0].shape[:2]
    seed = _grid_seed_pts((0, 0, W, H), rows=rows, cols=cols)       # (N, 2)
    N = seed.shape[0]
    T = len(grays) - 1
    positions = np.tile(seed[None, :, :], (T + 1, 1, 1)).astype(np.float32)
    first_idx = np.full(N, T, dtype=np.int32)
    alive = np.ones(N, dtype=bool)
    cur = seed.reshape(-1, 1, 2).copy()
    # walk grays[T] -> grays[T-1] -> ... -> grays[0]
    for t in range(T - 1, -1, -1):
        if not alive.any():
            break
        try:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                grays[t + 1], grays[t], cur, None,
                winSize=(21, 21), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
        except Exception:
            break
        if nxt is None or status is None:
            break
        s = (status.reshape(-1) > 0) & alive
        nxt_flat = nxt.reshape(-1, 2)
        # alive points walk back; freshly-dead and already-dead points freeze
        positions[t] = np.where(s[:, None], nxt_flat, positions[t + 1])
        first_idx[s] = t
        alive = s
        cur = positions[t].reshape(-1, 1, 2).astype(np.float32)
    return positions, first_idx


# BGR corner colors for the 4x7 grid (each point gets a bilinear blend of
# these so its colour identifies its origin in the grid).
_GRID_TL = np.array([ 80,  80, 240], dtype=np.float32)  # warm red
_GRID_TR = np.array([ 80, 230, 230], dtype=np.float32)  # yellow
_GRID_BL = np.array([240, 120,  80], dtype=np.float32)  # blue
_GRID_BR = np.array([220, 220, 100], dtype=np.float32)  # cyan/teal


def _grid_color_field(rows, cols):
    """Return (rows, cols, 3) BGR uint8 grid: bilinear blend of corner colours."""
    u = np.linspace(0.0, 1.0, cols, dtype=np.float32)[None, :, None]   # (1, C, 1)
    v = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None, None]   # (R, 1, 1)
    top = (1 - u) * _GRID_TL + u * _GRID_TR
    bot = (1 - u) * _GRID_BL + u * _GRID_BR
    col = (1 - v) * top + v * bot
    return np.clip(col, 0, 255).astype(np.uint8)


def _draw_flow(img, sample, meta, *, style=None, target_stroke_px=24,
               alpha=0.9, draw_legend=None):
    """Overlay sparse LK flow.

    Default style ("grid") draws 28 colored brush strokes from a fixed 4x7
    grid; each grid cell's colour is a unique bilinear blend of four corner
    colours so the painting itself is the legend (top-left red, top-right
    yellow, bottom-left blue, bottom-right cyan). No arrowheads, no white
    dots - just smooth thick antialiased strokes with a small filled dot at
    the origin to anchor the eye.

    Older styles still selectable for A/B:
      "hsv" - Middlebury wheel (direction=hue, magnitude=sat) on goodFeatures pts.
      "mag" - legacy green->red ramp by magnitude on goodFeatures pts.

    target_stroke_px auto-scales stroke length so the median visible stroke
    on the output frame is ~24 px regardless of underlying flow magnitude.
    """
    cfg = meta.get("_flow_viz", {}) if isinstance(meta, dict) else {}
    style = style or cfg.get("style", "grid")
    if draw_legend is None:
        draw_legend = cfg.get("legend", True)

    sx, sy = _scaler(meta)
    fl = sample["flow"]
    if not fl["have_flow"]:
        return

    if style == "grid":
        max_stroke_px = float(cfg.get("max_stroke_px", 32.0))
        head_px = int(cfg.get("head_px", 6))
        line_thick = int(cfg.get("line_thick", 2))
        _draw_flow_grid(img, fl, sx, sy,
                        target_stroke_px=target_stroke_px,
                        max_stroke_px=max_stroke_px,
                        alpha=alpha, head_px=head_px, line_thick=line_thick)
        if draw_legend:
            _flow_grid_legend(img)
        return

    # ----- legacy styles ("hsv" / "mag") on goodFeaturesToTrack points -----
    pts, nxt = fl["pts"], fl["nxt"]
    if pts is None or len(pts) == 0:
        return
    vec = (nxt - pts) * np.array([sx, sy], dtype=np.float32)
    mag = np.linalg.norm(vec, axis=1)
    pos_x = (pts[:, 0] * sx).astype(np.int32)
    pos_y = (pts[:, 1] * sy).astype(np.int32)
    nz = mag[mag > 1e-3]
    med = float(np.median(nz)) if nz.size else 0.0
    amp = float(np.clip(target_stroke_px / max(med, 1e-3), 1.0, 12.0)) if med > 0 else 4.0
    mag_ref = max(1.0, float(np.percentile(mag, 95))) if mag.size else 1.0
    if style == "hsv":
        ang = np.arctan2(vec[:, 1], vec[:, 0])
        hue = ((ang + np.pi) / (2.0 * np.pi)) * 180.0
        sat = np.clip(np.clip(mag / mag_ref, 0, 1) * 255.0, 70.0, 255.0)
        val = np.full_like(hue, 255.0)
        hsv = np.stack([hue, sat, val], axis=-1)[None, ...].astype(np.uint8)
        colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]
    else:  # "mag"
        t = np.clip(mag / mag_ref, 0.0, 1.0)
        colors = np.stack([
            (80 * (1 - t)).astype(np.uint8),
            (220 * (1 - t) + 40 * t).astype(np.uint8),
            (60 + 195 * t).astype(np.uint8),
        ], axis=-1)
    overlay = img.copy()
    for i in range(len(pts)):
        a = (int(pos_x[i]), int(pos_y[i]))
        b = (int(pos_x[i] + vec[i, 0] * amp), int(pos_y[i] + vec[i, 1] * amp))
        col = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
        if mag[i] < 0.5:
            cv2.circle(overlay, a, 2, col, -1, cv2.LINE_AA)
        else:
            cv2.arrowedLine(overlay, a, b, col, 1, cv2.LINE_AA, tipLength=0.28)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    if draw_legend and style == "hsv":
        _flow_wheel_legend(img)


def _draw_flow_grid(img, fl, sx, sy, *, target_stroke_px=24,
                    max_stroke_px=80, alpha=0.92, head_px=6, line_thick=2):
    """Paint the 4x7 grid LK as either a tapered comet trail (preferred)
    or a single colored stroke (fallback when no trail is available).

    The trail style follows what CoTracker / PIPs use for short tracklets:
    a polyline whose colour brightens and whose thickness grows from tail
    to head, ending in a colored dot with a tiny white highlight. Invalid
    tracks still draw a small hollow ring at their seed so the 4x7 grid
    structure is always visible.
    """
    g_traj = fl.get("grid_traj")
    g_traj_first = fl.get("grid_traj_first")
    g_shape = fl.get("grid_shape")
    if g_shape is None:
        return
    rows, cols = g_shape
    palette = _grid_color_field(rows, cols)

    # ----- quiver-style arrows (preferred) -----
    if g_traj is not None and g_traj.shape[0] >= 2:
        _paint_arrows(img, g_traj, g_traj_first, sx, sy, palette,
                      max_arrow_px=max_stroke_px, alpha=alpha,
                      head_px=int(head_px), line_thick=int(line_thick))
        return

    # ----- fallback: single-step LK line -----
    g_pts = fl.get("grid_pts")
    g_nxt = fl.get("grid_nxt")
    g_valid = fl.get("grid_valid")
    if g_pts is None or g_nxt is None:
        return
    vec_out = (g_nxt - g_pts) * np.array([sx, sy], dtype=np.float32)
    mag_out = np.linalg.norm(vec_out, axis=1)
    valid_mask = g_valid if g_valid is not None else np.ones_like(mag_out, bool)
    nz_mag = mag_out[valid_mask & (mag_out > 1e-3)]
    med = float(np.median(nz_mag)) if nz_mag.size else 0.0
    amp = float(np.clip(target_stroke_px / max(med, 1e-3), 1.0, 12.0)) if med > 0 else 4.0
    displayed = vec_out * amp
    disp_mag = np.linalg.norm(displayed, axis=1)
    over = disp_mag > max_stroke_px
    if over.any():
        scale = np.ones_like(disp_mag)
        scale[over] = max_stroke_px / np.maximum(disp_mag[over], 1e-6)
        displayed = displayed * scale[:, None]

    overlay = img.copy()
    for k in range(rows * cols):
        i, j = divmod(k, cols)
        col = palette[i, j]
        col_t = (int(col[0]), int(col[1]), int(col[2]))
        ax = int(g_pts[k, 0] * sx)
        ay = int(g_pts[k, 1] * sy)
        if g_valid is not None and not bool(g_valid[k]):
            cv2.circle(overlay, (ax, ay), 3, col_t, -1, cv2.LINE_AA)
            continue
        bx = int(ax + displayed[k, 0])
        by = int(ay + displayed[k, 1])
        cv2.line(overlay, (ax, ay), (bx, by), col_t, 3, cv2.LINE_AA)
        cv2.circle(overlay, (ax, ay), 3, col_t, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _thin_arrow(img, p0, p1, color, *, head_px=4, line_thick=1, head_angle_deg=22):
    """Thin antialiased arrow with a small *fixed-size* arrowhead.

    cv2.arrowedLine sizes the head as a fraction of the line, so long arrows
    look top-heavy. matplotlib's quiver uses an absolute head size in display
    units; this matches that convention so arrows of any length share the
    same delicate look.
    """
    cv2.line(img, p0, p1, color, line_thick, cv2.LINE_AA)
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    L = (dx * dx + dy * dy) ** 0.5
    if L < 1e-3:
        return
    ux, uy = -dx / L, -dy / L                       # unit vector back from head
    a = head_angle_deg * np.pi / 180.0
    ca, sa = float(np.cos(a)), float(np.sin(a))
    h1 = (int(p1[0] + head_px * (ux * ca - uy * sa)),
          int(p1[1] + head_px * (ux * sa + uy * ca)))
    h2 = (int(p1[0] + head_px * (ux * ca + uy * sa)),
          int(p1[1] + head_px * (-ux * sa + uy * ca)))
    cv2.line(img, p1, h1, color, line_thick, cv2.LINE_AA)
    cv2.line(img, p1, h2, color, line_thick, cv2.LINE_AA)


def _paint_arrows(img, traj, first_idx, sx, sy, palette, *,
                  max_arrow_px=40, target_arrow_px=18, alpha=0.95,
                  head_px=6, line_thick=2):
    """matplotlib-quiver-style arrows on the 4x7 grid.

    Each grid point at the current frame (positions[T, n]) gets a small
    coloured dot. If single-step backward LK succeeded, it ALSO gets a
    thin antialiased arrow from the grid pointing in the direction of the
    motion that just brought the point to its current spot. Magnitude is
    auto-amped to a median visible length of ~target_arrow_px and
    hard-capped at max_arrow_px per arrow (direction preserved).

    Inspired by /root/optical-flow/flow_display.py's `sparse_flow` (which
    uses plt.quiver); we keep the per-point bilinear palette colour from
    earlier so each arrow's origin in the grid is still readable.
    """
    T = traj.shape[0] - 1
    N = traj.shape[1]
    cols = palette.shape[1]

    out = traj * np.array([sx, sy], dtype=np.float32)  # (T+1, N, 2)
    base_pts = out[T]                                   # current grid
    prev_pts = out[T - 1] if T >= 1 else base_pts       # one step back
    motion = base_pts - prev_pts                        # output px
    motion_mag = np.linalg.norm(motion, axis=1)
    valid = first_idx < T if first_idx is not None else np.ones(N, dtype=bool)

    nz = motion_mag[valid & (motion_mag > 1e-3)]
    med = float(np.median(nz)) if nz.size else 0.0
    amp = float(np.clip(target_arrow_px / max(med, 1e-3), 1.0, 8.0)) if med > 0 else 2.0

    disp = motion * amp
    disp_mag = np.linalg.norm(disp, axis=1)
    over = disp_mag > max_arrow_px
    if over.any():
        scale = np.ones_like(disp_mag)
        scale[over] = max_arrow_px / np.maximum(disp_mag[over], 1e-6)
        disp = disp * scale[:, None]

    overlay = img.copy()
    for n in range(N):
        i, j = divmod(n, cols)
        base = palette[i, j].astype(np.float32)
        col = (int(base[0]), int(base[1]), int(base[2]))
        ax = int(base_pts[n, 0])
        ay = int(base_pts[n, 1])
        cv2.circle(overlay, (ax, ay), 3, col, -1, cv2.LINE_AA)   # grid dot
        if not bool(valid[n]) or motion_mag[n] < 0.3:
            continue
        bx = int(ax + disp[n, 0])
        by = int(ay + disp[n, 1])
        _thin_arrow(overlay, (ax, ay), (bx, by), col,
                    head_px=head_px, line_thick=line_thick)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _flow_grid_legend(img, *, rows=4, cols=7, cell=10, margin=18):
    """Small thumbnail of the 4x7 colour grid in the bottom-right so the
    reader knows colour = grid origin (TL red, TR yellow, BL blue, BR cyan)."""
    H, W = img.shape[:2]
    pal = _grid_color_field(rows, cols)
    panel_w = cols * cell
    panel_h = rows * cell
    x0 = W - panel_w - margin
    y0 = H - panel_h - margin - 14
    if x0 < 0 or y0 < 0:
        return
    # solid background for legibility
    cv2.rectangle(img, (x0 - 4, y0 - 4), (x0 + panel_w + 4, y0 + panel_h + 4), (24, 24, 24), -1)
    for i in range(rows):
        for j in range(cols):
            c = pal[i, j]
            col = (int(c[0]), int(c[1]), int(c[2]))
            cv2.rectangle(img,
                          (x0 + j * cell, y0 + i * cell),
                          (x0 + (j + 1) * cell, y0 + (i + 1) * cell),
                          col, -1)
    cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (240, 240, 240), 1, cv2.LINE_AA)
    label = "color = grid origin"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    tx = max(2, x0 + panel_w // 2 - tw // 2)
    cv2.putText(img, label, (tx, y0 + panel_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)


def gate_lines(sample, meta):
    fl = sample["flow"]
    ga = meta["gate_a"]
    gc = meta["gate_c"]
    a_col = C_PASS if sample["gateA_pass"] else C_REJECT
    b_col = C_PASS if (not fl["have_flow"] or fl["stable_camera"]) else C_REJECT
    c_col = C_PASS if fl["hand_motion"] else C_REJECT
    o_col = C_PASS if sample["overall"] else C_REJECT
    lines = [
        (f"frame {sample['frame_idx']}  |  OVERALL: {'PASS' if sample['overall'] else 'REJECT'}", o_col),
        (f"A detect: {sample['qualified']}/{int(ga.get('min_hands', 2))} hands in ROI "
         f"[area {ga.get('min_area_ratio')}-{ga.get('max_area_ratio')}] -> "
         f"{'pass' if sample['gateA_pass'] else 'FAIL'}", a_col),
    ]
    if fl["have_flow"]:
        lines.append((
            f"B camera: median flow {fl['median_disp']:.1f}px "
            f"(thr {fl['thresh_px']:.1f}px) -> {'pass' if fl['stable_camera'] else 'FAIL too much motion'}",
            b_col))
    else:
        lines.append(("B camera: insufficient tracked points", b_col))
    lines.append((
        f"C hand motion: diff {fl['diff_score']:.3f} (thr {gc.get('hand_motion_thresh')}) -> "
        f"{'pass' if fl['hand_motion'] else 'fail'}", c_col))
    return lines


def _ensure_grid_trail(video_path, sample, meta, n_trail):
    """If the sample doesn't already carry grid_traj, compute it on demand by
    decoding n_trail+1 frames spaced by meta['skip'] ending at frame_idx."""
    fl = sample.get("flow") or {}
    if "grid_traj" in fl and fl["grid_traj"] is not None:
        return
    if n_trail <= 0:
        return
    skip = max(1, int(meta.get("skip", 15)))
    dw, dh = meta["decode"]
    roi_px = meta["roi_px_small"]
    target = int(sample["frame_idx"])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    grays = []
    for k in range(n_trail, -1, -1):
        fi = max(0, target - k * skip)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            break
        grays.append(cv2.cvtColor(cv2.resize(fr, (dw, dh)), cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(grays) < 2:
        return
    traj, first_idx = _grid_lk_trail(grays)
    if traj is None:
        return
    fl["grid_traj"] = traj
    fl["grid_traj_first"] = first_idx
    fl["grid_shape"] = (4, 7)
    sample["flow"] = fl


def render_frame(video_path, sample, meta, case):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample["frame_idx"]))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    flow_cfg = meta.get("_flow_viz", {}) if isinstance(meta, dict) else {}
    n_trail = int(flow_cfg.get("n_trail", 4))
    show_banner = bool(flow_cfg.get("banner", True))
    show_roi = bool(flow_cfg.get("roi", True))
    if case in ("caseB_flow", "caseC_pass") and n_trail > 0:
        _ensure_grid_trail(video_path, sample, meta, n_trail)
    img = frame.copy()
    if case in ("caseA_size", "caseC_pass"):
        if show_roi:
            _draw_roi(img, meta)
        _draw_boxes(img, sample, meta)
    if case in ("caseB_flow", "caseC_pass"):
        _draw_flow(img, sample, meta)
    if show_banner:
        _banner(img, gate_lines(sample, meta))
    return img


def render_clip(video_path, sample, meta, case, out_path, pre=12, post=12):
    """Short annotated clip centred on the candidate frame (overlays recomputed
    per displayed frame using the same gate logic)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    Wf, Hf = meta["full"]
    dw, dh = meta["decode"]
    skip = meta["skip"]
    f0 = max(0, int(sample["frame_idx"]) - pre * skip)
    f1 = int(sample["frame_idx"]) + post * skip
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (Wf, Hf))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    gate_a, gate_b, gate_c = meta["gate_a"], meta["gate_b"], meta["gate_c"]
    roi_px = meta["roi_px_small"]
    prev_gray = None
    for fi in range(f0, f1):
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (dw, dh))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        boxes, qcount, gateA = _detect_boxes(None, small, gate_a)  # boxes optional in clip
        fl = _flow(prev_gray, gray, gate_b, gate_c, roi_px)
        s = {"frame_idx": fi, "boxes": boxes, "qualified": qcount,
             "gateA_pass": gateA, "flow": fl, "overall": bool(gateA and fl["gateB_pass"])}
        img = frame.copy()
        if case in ("caseA_size", "caseC_pass"):
            _draw_roi(img, meta)
        if case in ("caseB_flow", "caseC_pass"):
            _draw_flow(img, s, meta)
        writer.write(img)
        prev_gray = gray
    writer.release()
    cap.release()


# --------------------------------------------------------------------------- #
# one-off analysis at an exact frame (used to force caseC to a specific frame).
# Flow is computed at the pipeline's sampling cadence (prev = frame_idx - skip),
# so the gate verdict here matches what the real clipper would have decided.
# --------------------------------------------------------------------------- #
def analyze_at_frame(video_path, cfg, model, frame_idx):
    heuristic = _heuristic_section(cfg)
    gate_a = heuristic.get("gate_a") or {}
    gate_b = heuristic.get("gate_b") or {}
    gate_c = heuristic.get("gate_c") or {}
    skip = max(1, int(heuristic.get("skip_frames", 15)))
    dw = int(heuristic.get("decode_width", 448))
    dh = int(heuristic.get("decode_height", 256))
    roi_px = _roi_bounds(dw, dh, gate_a.get("roi", [0.0, 0.0, 1.0, 1.0]))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    Wf = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or dw)
    Hf = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or dh)

    prev_gray = None
    prev_idx = frame_idx - skip
    if prev_idx >= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, prev_idx)
        ok, prev = cap.read()
        if ok:
            prev_gray = cv2.cvtColor(cv2.resize(prev, (dw, dh)), cv2.COLOR_BGR2GRAY)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_idx} from {video_path}")
    small = cv2.resize(frame, (dw, dh))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    boxes, qcount, gateA = _detect_boxes(model, small, gate_a)
    fl = _flow(prev_gray, gray, gate_b, gate_c, roi_px)
    sample = {
        "frame_idx": frame_idx, "boxes": boxes, "qualified": qcount,
        "gateA_pass": gateA, "flow": fl,
        "overall": bool(gateA and fl["gateB_pass"]),
    }
    meta = {
        "video": str(video_path), "fps": fps, "skip": skip,
        "decode": (dw, dh), "full": (Wf, Hf), "roi_px_small": roi_px,
        "gate_a": gate_a, "gate_b": gate_b, "gate_c": gate_c,
    }
    return sample, meta


# --------------------------------------------------------------------------- #
# range mode: dump every frame in [start, end) as an annotated jpg
# --------------------------------------------------------------------------- #
def render_range(video_path, cfg, model, start, end, out_dir, stride=1, jpg_quality=92,
                 flow_viz=None):
    """Walk [start, end) and write one annotated jpg per frame.

    Overlays match caseC (ROI + qualified/rejected YOLO boxes + LK flow arrows +
    gate-status banner), but every frame is exported so the user can pick.
    """
    heuristic = _heuristic_section(cfg)
    gate_a = heuristic.get("gate_a") or {}
    gate_b = heuristic.get("gate_b") or {}
    gate_c = heuristic.get("gate_c") or {}
    dw = int(heuristic.get("decode_width", 448))
    dh = int(heuristic.get("decode_height", 256))
    roi_px = _roi_bounds(dw, dh, gate_a.get("roi", [0.0, 0.0, 1.0, 1.0]))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    Wf = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or dw)
    Hf = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or dh)
    if nframes:
        end = min(end, nframes)
    if end <= start:
        cap.release()
        raise RuntimeError(f"empty range [{start},{end}); total={nframes}")

    meta = {
        "video": str(video_path), "fps": fps, "skip": 1,
        "decode": (dw, dh), "full": (Wf, Hf), "roi_px_small": roi_px,
        "gate_a": gate_a, "gate_b": gate_b, "gate_c": gate_c,
        "_flow_viz": flow_viz or {
            "style": "grid", "legend": False, "banner": False, "roi": False,
            "n_trail": 1, "max_stroke_px": 40.0, "head_px": 6, "line_thick": 2,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats.jsonl"
    stats_fh = stats_path.open("w", encoding="utf-8")

    # Maintain a sliding window of decoded grayscales so we can paint a short
    # per-frame trail (CoTracker-style) at every output instead of a single
    # stiff line. n_trail comes from --n_trail (default 4).
    from collections import deque
    n_trail = int((flow_viz or {}).get("n_trail", 4))
    gray_window = deque(maxlen=max(2, n_trail + 1))
    show_banner = bool((flow_viz or {}).get("banner", True))
    show_roi = bool((flow_viz or {}).get("roi", True))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    prev_gray = None
    written = 0
    for fi in range(start, end):
        ok, frame = cap.read()
        if not ok:
            break
        if (fi - start) % stride != 0:
            g = cv2.cvtColor(cv2.resize(frame, (dw, dh)), cv2.COLOR_BGR2GRAY)
            gray_window.append(g)
            prev_gray = g
            continue
        small = cv2.resize(frame, (dw, dh))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_window.append(gray)
        boxes, qcount, gateA = _detect_boxes(model, small, gate_a)
        fl = _flow(prev_gray, gray, gate_b, gate_c, roi_px)
        if len(gray_window) >= 2 and n_trail > 0:
            traj, traj_first = _grid_lk_trail(list(gray_window))
            if traj is not None:
                fl["grid_traj"] = traj
                fl["grid_traj_first"] = traj_first
                fl["grid_shape"] = (4, 7)
        sample = {
            "frame_idx": fi, "boxes": boxes, "qualified": qcount,
            "gateA_pass": gateA, "flow": fl,
            "overall": bool(gateA and fl["gateB_pass"]),
        }
        img = frame.copy()
        if show_roi:
            _draw_roi(img, meta)
        _draw_boxes(img, sample, meta)
        _draw_flow(img, sample, meta)
        if show_banner:
            _banner(img, gate_lines(sample, meta))
        out_path = out_dir / f"frame_{fi:06d}.jpg"
        cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
        stats_fh.write(json.dumps({
            "frame_idx": fi,
            "qualified": qcount,
            "gateA_pass": bool(gateA),
            "stable_camera": bool(fl["stable_camera"]),
            "hand_motion": bool(fl["hand_motion"]),
            "median_disp": fl["median_disp"],
            "thresh_px": fl["thresh_px"],
            "diff_score": fl["diff_score"],
            "have_flow": bool(fl["have_flow"]),
            "overall": sample["overall"],
            "n_boxes": len(boxes),
        }) + "\n")
        prev_gray = gray
        written += 1
        if written % 50 == 0:
            print(f"  [range] {fi}/{end} ({written} jpgs)")
    stats_fh.close()
    cap.release()
    print(f"  [range] wrote {written} jpgs + stats.jsonl -> {out_dir}")
    return written


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=None)
    ap.add_argument("--video_root", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--override_config", default=None)
    ap.add_argument("--model_path", default=str(DEFAULT_MODEL),
                    help="YOLO hand detector (default weights/external/detector.pt)")
    ap.add_argument("--out_dir", default="docs/figure_assets/filter_stage")
    ap.add_argument("--topk", type=int, default=3, help="candidates rendered per case")
    ap.add_argument("--max_videos", type=int, default=0, help="0 = all")
    ap.add_argument("--clip", action="store_true", help="also write a short annotated mp4 per case")
    ap.add_argument("--start", type=int, default=None,
                    help="range mode: first frame index (inclusive). Requires --video and --end.")
    ap.add_argument("--end", type=int, default=None,
                    help="range mode: last frame index (exclusive).")
    ap.add_argument("--stride", type=int, default=1,
                    help="range mode: export every Nth frame (default 1 = every frame).")
    ap.add_argument("--jpg_quality", type=int, default=92, help="range mode jpg quality (1-100).")
    ap.add_argument("--pass_frame", type=int, default=None,
                    help="force caseC_pass to render exactly this frame index "
                         "(requires --video). caseA/caseB still auto-picked.")
    ap.add_argument("--flow_style", choices=["grid", "hsv", "mag"], default="grid",
                    help="flow viz style. grid (default) = 4x7 fixed-grid LK trails, "
                         "each cell coloured by its position (bilinear corner blend); "
                         "hsv = Middlebury direction colormap on goodFeatures pts; "
                         "mag = legacy green->red by magnitude.")
    ap.add_argument("--flow_legend", action="store_true",
                    help="(debug) show the small flow style legend in the corner. "
                         "OFF by default for clean figures.")
    ap.add_argument("--max_stroke_px", type=float, default=40.0,
                    help="grid style: hard cap per arrow length in output px "
                         "(direction preserved). Default 40. Larger => arrows "
                         "convey more motion intensity but clutter the figure.")
    ap.add_argument("--arrow_head_px", type=int, default=6,
                    help="arrowhead size in pixels (FIXED, not a fraction of "
                         "the arrow length). Default 6.")
    ap.add_argument("--arrow_line_thick", type=int, default=2,
                    help="arrow shaft thickness in pixels. Default 2 - keeps "
                         "the shaft close to the 4 px filled grid dot so the "
                         "arrow doesn't look like a hair next to the anchor.")
    ap.add_argument("--camera_disp_thresh", type=float, default=None,
                    help="override gate_b.camera_disp_thresh for this run (fraction "
                         "of decoded max-side). Default config is 0.2 (= ~89.6 px "
                         "on 448x256); lower it (e.g. 0.05) to make walking-camera "
                         "footage actually trip caseB. Banner shows the value in use.")
    ap.add_argument("--hand_motion_thresh", type=float, default=None,
                    help="override gate_c.hand_motion_thresh for this run.")
    ap.add_argument("--n_trail", type=int, default=1,
                    help="number of backward LK steps for the quiver direction "
                         "(default 1 = use immediate previous-frame motion). "
                         "Higher values average over a longer window but the "
                         "arrow itself is always single-step.")
    ap.add_argument("--banner", action="store_true",
                    help="(debug) show the gate-status A/B/C banner. OFF by default.")
    ap.add_argument("--roi", action="store_true",
                    help="(debug) show the amber central ROI rectangle. OFF by default.")
    args = ap.parse_args(argv)

    if not args.video and not args.video_root:
        raise SystemExit("provide --video or --video_root")
    range_mode = args.start is not None and args.end is not None
    if range_mode and not args.video:
        raise SystemExit("range mode (--start/--end) requires --video (single file)")
    if args.pass_frame is not None and not args.video:
        raise SystemExit("--pass_frame requires --video (single file)")

    override = {}
    if args.override_config:
        import yaml
        override = yaml.safe_load(Path(args.override_config).read_text()) or {}
    cfg = load_clip_config(args.config, override)
    # CLI threshold overrides (applied AFTER config merge so they always win)
    if args.camera_disp_thresh is not None or args.hand_motion_thresh is not None:
        heur = cfg.setdefault("heuristic", {})
        if args.camera_disp_thresh is not None:
            gb = heur.setdefault("gate_b", {})
            old = gb.get("camera_disp_thresh", 0.2)
            gb["camera_disp_thresh"] = float(args.camera_disp_thresh)
            print(f"[override] gate_b.camera_disp_thresh: {old} -> "
                  f"{gb['camera_disp_thresh']} (= "
                  f"{gb['camera_disp_thresh'] * max(int(heur.get('decode_width', 448)), int(heur.get('decode_height', 256))):.1f} "
                  f"px on decoded grid)")
        if args.hand_motion_thresh is not None:
            gc = heur.setdefault("gate_c", {})
            old = gc.get("hand_motion_thresh", 0.012)
            gc["hand_motion_thresh"] = float(args.hand_motion_thresh)
            print(f"[override] gate_c.hand_motion_thresh: {old} -> {gc['hand_motion_thresh']}")
    model = _load_yolo(args.model_path)
    if model is None:
        print(f"[warn] detector not loaded from {args.model_path}; "
              f"gate A will pass-through and caseA cannot be found.")

    if args.video:
        videos = [Path(args.video)]
    else:
        videos = discover_videos(args.video_root)
    if args.max_videos:
        videos = videos[: args.max_videos]

    flow_viz_cfg = {
        "style": args.flow_style,
        "legend": bool(args.flow_legend),
        "max_stroke_px": float(args.max_stroke_px),
        "n_trail": int(args.n_trail),
        "banner": bool(args.banner),
        "roi": bool(args.roi),
        "head_px": int(args.arrow_head_px),
        "line_thick": int(args.arrow_line_thick),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if range_mode:
        vp = videos[0]
        stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in vp.stem)
        sub = out_dir / f"{stem}_f{args.start:06d}-{args.end:06d}"
        print(f"[range] {vp} -> frames [{args.start},{args.end}) stride={args.stride}")
        render_range(vp, cfg, model, args.start, args.end, sub,
                     stride=max(1, args.stride), jpg_quality=int(args.jpg_quality),
                     flow_viz=flow_viz_cfg)
        return

    report = {"videos": [], "thresholds": _heuristic_section(cfg)}

    for vp in videos:
        print(f"[scan] {vp}")
        try:
            samples, meta = analyze(vp, cfg, model)
        except Exception as exc:
            print(f"  [skip] {exc}")
            continue
        meta["_flow_viz"] = flow_viz_cfg
        cands = select_candidates(samples, args.topk)
        if args.pass_frame is not None:
            try:
                forced, meta = analyze_at_frame(vp, cfg, model, int(args.pass_frame))
            except Exception as exc:
                print(f"  [warn] --pass_frame {args.pass_frame}: {exc}")
            else:
                meta["_flow_viz"] = flow_viz_cfg
                if not forced["overall"]:
                    fl = forced["flow"]
                    print(f"  [warn] forced frame {args.pass_frame} does NOT actually "
                          f"pass all gates (gateA={forced['gateA_pass']} "
                          f"stable_cam={fl['stable_camera']} hand_motion={fl['hand_motion']}); "
                          f"rendering it anyway as caseC_pass.")
                cands["caseC_pass"] = [forced]
        stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in vp.stem)
        vid_rec = {"video": str(vp), "cases": {}}
        for case, samp_list in cands.items():
            recs = []
            for rank, s in enumerate(samp_list):
                img = render_frame(vp, s, meta, case)
                if img is None:
                    continue
                name = f"{stem}_f{s['frame_idx']:06d}_{case}_r{rank}.png"
                cv2.imwrite(str(out_dir / name), img)
                rec = {
                    "png": name, "frame_idx": s["frame_idx"],
                    "qualified": s["qualified"], "gateA_pass": s["gateA_pass"],
                    "flow_median_disp": s["flow"]["median_disp"],
                    "flow_thresh_px": s["flow"]["thresh_px"],
                    "diff_score": s["flow"]["diff_score"],
                    "overall": s["overall"],
                }
                if args.clip and rank == 0:
                    clip_name = f"{stem}_f{s['frame_idx']:06d}_{case}_clip.mp4"
                    render_clip(vp, s, meta, case, out_dir / clip_name)
                    rec["clip"] = clip_name
                recs.append(rec)
                print(f"  [{case}] frame {s['frame_idx']} -> {name}")
            vid_rec["cases"][case] = recs
        report["videos"].append(vid_rec)

    (out_dir / "_filter_viz_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()
