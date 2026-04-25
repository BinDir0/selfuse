#!/usr/bin/env python3
"""Rank WebDataset episodes for demos and export top candidates as videos."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.clip_manifest import load_clip_manifest
from lib.pipeline.quality_metrics import finalize_clip_quality_metrics, new_clip_quality_stats, update_clip_quality_stats
from lib.pipeline.viewer_backend import EpisodeViewerBackend, scan_sample_summaries, summarize_episode_candidates
from tools.ops import webdataset_visualizer as wv


@dataclass
class EpisodeDemoResult:
    episode_key: str
    clip_id: Optional[str]
    instruction_preview: str
    frame_count: int
    score: float
    score_breakdown: dict
    quality_metrics: dict
    demo_metrics: dict
    exportable: dict


@dataclass
class EpisodeDemoSkip:
    episode_key: str
    clip_id: Optional[str]
    reason: str


def build_parser():
    parser = argparse.ArgumentParser(description="Rank and export pipeline demo candidates from WebDataset shards")
    parser.add_argument("--input", required=True, help="Path to a .tar shard or directory containing .tar shards")
    parser.add_argument("--descriptor_manifest", default=None, help="Optional clip manifest; enables seq_folder lookup and depth demo checks")
    parser.add_argument(
        "--buildai_processed_root",
        default=None,
        help="Optional BuildAI processed root; used to auto-resolve seq_folder from clip_id when no manifest is provided",
    )
    parser.add_argument("--sample-limit", type=int, default=None, help="Only index the first N matched samples")
    parser.add_argument("--episode-limit", type=int, default=None, help="Only evaluate the first N matched episodes")
    parser.add_argument("--filter-key", default="", help="Substring filter on key / clip_id / instruction")
    parser.add_argument("--filter-presence", type=int, default=None, choices=[0, 1, 2, 3], help="Only scan samples with the chosen presence flag")
    parser.add_argument("--top-k", type=int, default=5, help="How many top-ranked episodes to export / report")
    parser.add_argument("--min-frames", type=int, default=60, help="Minimum episode length to qualify as a demo candidate")
    parser.add_argument("--render-modes", default="keypoint,mano", help="Comma-separated render modes to export: keypoint, mano, depth")
    parser.add_argument("--video-fps", type=int, default=30, help="FPS for exported demo videos")
    parser.add_argument("--export-dir", default=None, help="Optional output directory for exported demo videos and posters")
    parser.add_argument("--report-out", default=None, help="Optional JSON report path")
    parser.add_argument("--mano-dir", default=None, help="Optional MANO model directory override")
    parser.add_argument("--mano-device", default="cpu", help="Device for MANO rendering, e.g. cpu or cuda:0")
    return parser


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def peak_score(value: float, *, low: float, peak: float, high: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if value <= low or value >= high:
        return 0.0
    if value == peak:
        return 1.0
    if value < peak:
        return clamp01((value - low) / max(peak - low, 1e-8))
    return clamp01((high - value) / max(high - peak, 1e-8))


def stable_inverse_score(value: float, scale: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return 1.0 / (1.0 + max(0.0, float(value)) / max(scale, 1e-8))


def parse_render_modes(raw: str) -> list[str]:
    modes = []
    for item in str(raw).split(","):
        mode = item.strip().lower()
        if not mode:
            continue
        if mode not in {"keypoint", "mano", "depth"}:
            raise ValueError(f"Unsupported render mode: {mode}")
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("Expected at least one render mode")
    return modes


def build_seq_folder_lookup(descriptor_manifest: str | None) -> dict[str, str]:
    if not descriptor_manifest:
        return {}
    records = load_clip_manifest(descriptor_manifest)
    return {record.clip_id: record.descriptor.seq_folder for record in records}


def _resolve_buildai_seq_folder(processed_root: Path, clip_id: str) -> Optional[Path]:
    parts = clip_id.split("_")
    if len(parts) < 2 or not parts[0].startswith("f") or not parts[1].startswith("w"):
        return None
    try:
        factory_id = int(parts[0][1:])
        worker_id = int(parts[1][1:])
    except ValueError:
        return None

    candidates = [
        processed_root / f"factory_{factory_id:03d}" / f"worker_{worker_id:03d}" / "processed" / clip_id,
        processed_root / f"factory{factory_id:03d}" / "outputs" / clip_id,
        processed_root / f"factory_{factory_id:03d}" / "outputs" / clip_id,
    ]
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    return None


def resolve_seq_folder(
    clip_id: Optional[str],
    *,
    seq_folder_lookup: dict[str, str],
    buildai_processed_root: Optional[Path],
) -> Optional[str]:
    if not clip_id:
        return None
    seq_folder = seq_folder_lookup.get(clip_id)
    if seq_folder:
        return seq_folder
    if buildai_processed_root is None:
        return None
    resolved = _resolve_buildai_seq_folder(buildai_processed_root, clip_id)
    return None if resolved is None else str(resolved)


def _visible_bbox_metrics(points_world: np.ndarray, c2w: np.ndarray, intrinsic: np.ndarray, image_shape) -> dict:
    uv, valid = wv._project_points(points_world, c2w, intrinsic)
    mask = wv._clip_uv_mask(uv, valid, image_shape)
    visible = uv[mask]
    if visible.shape[0] < 2:
        return {
            "point_visibility_ratio": float(mask.mean()) if mask.size else 0.0,
            "full_visibility": bool(mask.all()) if mask.size else False,
            "bbox_area_ratio": 0.0,
            "center_offset_ratio": 1.0,
        }

    min_xy = visible.min(axis=0)
    max_xy = visible.max(axis=0)
    width = float(max_xy[0] - min_xy[0])
    height = float(max_xy[1] - min_xy[1])
    image_h, image_w = image_shape[:2]
    bbox_area_ratio = (width * height) / max(float(image_h * image_w), 1.0)
    bbox_center = (min_xy + max_xy) * 0.5
    image_center = np.array([image_w * 0.5, image_h * 0.5], dtype=np.float32)
    diag = math.sqrt(float(image_w * image_w + image_h * image_h))
    center_offset_ratio = float(np.linalg.norm(bbox_center - image_center) / max(diag * 0.5, 1.0))
    return {
        "point_visibility_ratio": float(mask.mean()),
        "full_visibility": bool(mask.all()),
        "bbox_area_ratio": float(bbox_area_ratio),
        "center_offset_ratio": center_offset_ratio,
    }


def _find_depth_cache(seq_folder: str) -> Optional[Path]:
    slam_dir = Path(seq_folder) / "SLAM"
    if not slam_dir.is_dir():
        return None
    patterns = (
        "dense_depth_any4d_*.npz",
        "dense_depth_any4d_all_*.npz",
        "dense_depth_any4d_keyframes_*.npz",
    )
    for pattern in patterns:
        matches = sorted(slam_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _load_depth_cache(cache_path: Optional[Path]) -> Optional[dict]:
    if cache_path is None:
        return None
    try:
        with np.load(str(cache_path), allow_pickle=False) as payload:
            frame_indices = payload["frame_indices"].astype(np.int64).reshape(-1) if "frame_indices" in payload.files else None
            if "depths_uint16" in payload.files:
                depths = payload["depths_uint16"].astype(np.float32) * 1e-3
            elif "pred_depths" in payload.files:
                depths = payload["pred_depths"].astype(np.float32)
            elif "depths" in payload.files:
                depths = payload["depths"].astype(np.float32)
            else:
                return None
    except Exception:
        return None
    if frame_indices is None or depths.shape[0] != frame_indices.shape[0]:
        return None
    return {
        "frame_indices": frame_indices,
        "depths": depths,
        "cache_path": str(cache_path),
    }


def compute_episode_demo_result(
    episode_key: str,
    frames,
    *,
    seq_folder_lookup: dict[str, str],
    buildai_processed_root: Optional[Path],
    min_frames: int,
) -> EpisodeDemoResult:
    clip_id = frames[0].summary.clip_id if frames else None
    instruction_preview = frames[0].summary.instruction_preview if frames else ""
    stats = new_clip_quality_stats(clip_id or episode_key)

    both_hands_frames = 0
    visible_ratio_values = []
    full_visibility_values = []
    bbox_area_values = []
    center_offset_values = []
    mean_wrist_step_values = []
    depth_match_frame_count = 0
    depth_total_frames = 0

    prev_left = None
    prev_right = None

    seq_folder = resolve_seq_folder(
        clip_id,
        seq_folder_lookup=seq_folder_lookup,
        buildai_processed_root=buildai_processed_root,
    )
    depth_cache = _load_depth_cache(_find_depth_cache(seq_folder)) if seq_folder else None
    depth_index_set = set(depth_cache["frame_indices"].tolist()) if depth_cache is not None else set()

    for frame in frames:
        if depth_cache is not None:
            depth_total_frames += 1
            if int(frame.frame_idx) in depth_index_set:
                depth_match_frame_count += 1

        update_clip_quality_stats(
            stats,
            frame.frame_idx,
            int(frame.summary.instruction_num or 0),
            int(frame.presence or 0),
            frame.lowdim_array,
        )

        if not np.isfinite(frame.lowdim_array).all():
            prev_left = None
            prev_right = None
            continue

        fields = wv._decode_lowdim_fields(frame.lowdim_array)
        c2w, _ = wv._resolve_camera_c2w(fields["camera_w2c"])
        image_shape = wv._decode_image_bgr(frame.sample["image_bytes"]).shape
        left_present, right_present = wv._presence_flags(frame.presence)
        if left_present and right_present:
            both_hands_frames += 1

        for side, present in (("left", left_present), ("right", right_present)):
            if not present:
                continue
            points = np.concatenate(
                [
                    fields[f"{side}_wrist_world"][None, :],
                    fields[f"{side}_fingertips_world"],
                ],
                axis=0,
            ).astype(np.float32)
            metrics = _visible_bbox_metrics(points, c2w, fields["camera_intrinsic"], image_shape)
            visible_ratio_values.append(metrics["point_visibility_ratio"])
            full_visibility_values.append(1.0 if metrics["full_visibility"] else 0.0)
            bbox_area_values.append(metrics["bbox_area_ratio"])
            center_offset_values.append(metrics["center_offset_ratio"])

        left = np.asarray(fields["left_wrist_world"], dtype=np.float32)
        right = np.asarray(fields["right_wrist_world"], dtype=np.float32)
        if prev_left is not None and prev_right is not None:
            left_step = float(np.linalg.norm(left - prev_left))
            right_step = float(np.linalg.norm(right - prev_right))
            mean_wrist_step_values.append(0.5 * (left_step + right_step))
        prev_left = left
        prev_right = right

    quality_metrics = finalize_clip_quality_metrics(stats)
    frame_count = int(len(frames))
    both_hands_ratio = float(both_hands_frames / frame_count) if frame_count > 0 else 0.0
    point_visibility_mean = float(np.mean(visible_ratio_values)) if visible_ratio_values else 0.0
    full_visibility_ratio = float(np.mean(full_visibility_values)) if full_visibility_values else 0.0
    bbox_area_median = float(np.median(bbox_area_values)) if bbox_area_values else 0.0
    center_offset_mean = float(np.mean(center_offset_values)) if center_offset_values else 1.0
    mean_wrist_step = float(np.mean(mean_wrist_step_values)) if mean_wrist_step_values else 0.0
    depth_match_ratio = float(depth_match_frame_count / depth_total_frames) if depth_total_frames > 0 else 0.0

    clean_score = 1.0 if (
        quality_metrics["invalid_lowdim_frames"] == 0
        and quality_metrics["nonfinite_lowdim_frames"] == 0
        and quality_metrics["frames_kept_candidate"] == frame_count
    ) else 0.0
    length_score = clamp01((frame_count - float(min_frames)) / max(120.0 - float(min_frames), 1.0))
    presence_score = clamp01(0.45 * quality_metrics["presence_ratio"] + 0.55 * both_hands_ratio)
    visibility_score = clamp01(0.55 * point_visibility_mean + 0.45 * full_visibility_ratio)
    size_score = peak_score(bbox_area_median, low=0.005, peak=0.04, high=0.20)
    center_score = clamp01(1.0 - center_offset_mean / 0.6)
    motion_score = peak_score(mean_wrist_step, low=0.002, peak=0.02, high=0.12)
    camera_stability_score = clamp01(
        0.6 * stable_inverse_score(quality_metrics["max_camera_translation_step"], 0.03)
        + 0.4 * stable_inverse_score(quality_metrics["max_camera_rotation_step"], 0.6)
    )
    instruction_score = 1.0 if int(quality_metrics["instruction_num_max"]) > 0 else 0.0
    depth_bonus = 0.05 if depth_match_ratio >= 1.0 else 0.0

    score_breakdown = {
        "clean_score": clean_score,
        "length_score": length_score,
        "presence_score": presence_score,
        "visibility_score": visibility_score,
        "size_score": size_score,
        "center_score": center_score,
        "motion_score": motion_score,
        "camera_stability_score": camera_stability_score,
        "instruction_score": instruction_score,
        "depth_bonus": depth_bonus,
    }
    score = (
        0.20 * clean_score
        + 0.14 * length_score
        + 0.18 * presence_score
        + 0.18 * visibility_score
        + 0.10 * size_score
        + 0.06 * center_score
        + 0.06 * motion_score
        + 0.03 * camera_stability_score
        + 0.05 * instruction_score
        + depth_bonus
    )

    demo_metrics = {
        "both_hands_ratio": both_hands_ratio,
        "point_visibility_mean": point_visibility_mean,
        "full_visibility_ratio": full_visibility_ratio,
        "bbox_area_median": bbox_area_median,
        "center_offset_mean": center_offset_mean,
        "mean_wrist_step": mean_wrist_step,
        "depth_match_ratio": depth_match_ratio,
    }
    exportable = {
        "keypoint": frame_count >= min_frames,
        "mano": frame_count >= min_frames and all(frame.mano_array is not None for frame in frames),
        "depth": frame_count >= min_frames and depth_match_ratio >= 1.0 and depth_cache is not None,
        "depth_cache_path": None if depth_cache is None else depth_cache["cache_path"],
    }

    return EpisodeDemoResult(
        episode_key=episode_key,
        clip_id=clip_id,
        instruction_preview=instruction_preview,
        frame_count=frame_count,
        score=float(score),
        score_breakdown=score_breakdown,
        quality_metrics=quality_metrics,
        demo_metrics=demo_metrics,
        exportable=exportable,
    )


def make_banner_lines(result: EpisodeDemoResult, *, render_mode: str) -> list[str]:
    return [
        f"{result.clip_id or result.episode_key} | mode={render_mode} | score={result.score:.3f} | frames={result.frame_count}",
        result.instruction_preview or "(no instruction)",
    ]


def draw_banner(image_bgr: np.ndarray, lines: list[str], *, footer: Optional[str] = None) -> np.ndarray:
    frame = image_bgr.copy()
    banner_height = 72 if footer is None else 94
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], banner_height), (18, 24, 36), -1)
    frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0.0)

    y = 24
    for idx, line in enumerate(lines[:2]):
        scale = 0.58 if idx == 0 else 0.50
        thickness = 1 if idx else 2
        cv2.putText(frame, line[:160], (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 244, 250), thickness, cv2.LINE_AA)
        y += 24
    if footer:
        cv2.putText(frame, footer[:180], (18, banner_height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 190, 210), 1, cv2.LINE_AA)
    return frame


def colorize_depth(depth: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    output = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if not valid.any():
        return output
    clipped = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    colored = cv2.applyColorMap((clipped * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    output[valid] = colored[valid]
    return output


def export_episode_video(
    backend: EpisodeViewerBackend,
    result: EpisodeDemoResult,
    frames,
    *,
    render_mode: str,
    output_path: Path,
    fps: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_frame = None
    rendered_frames = []
    banner_lines = make_banner_lines(result, render_mode=render_mode)

    for idx, frame in enumerate(frames):
        if render_mode == "keypoint":
            keypoint_frame = backend.build_keypoint_frame(frame.summary, frame.lowdim_array, frame.mano_array)
            rendered = wv._render_keypoint_overlay(frame.sample["image_bytes"], keypoint_frame, frame.presence)
        elif render_mode == "mano":
            if frame.mano_array is None:
                raise ValueError(f"Frame {frame.summary.key} is missing mano.npy")
            mano_frame = backend.build_mano_frame(frame.summary, frame.lowdim_array, frame.mano_array)
            rendered = wv._render_mano_overlay(
                frame.sample["image_bytes"],
                mano_frame["c2w"],
                mano_frame["intrinsic"],
                mano_frame,
                frame.presence,
            )
        else:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        rendered = draw_banner(
            rendered,
            banner_lines,
            footer=f"frame {idx + 1}/{len(frames)} | presence={frame.presence}",
        )
        if first_frame is None:
            first_frame = rendered
        rendered_frames.append(rendered)

    assert first_frame is not None
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    try:
        for image in rendered_frames:
            writer.write(image)
    finally:
        writer.release()

    poster_path = output_path.with_suffix(".poster.jpg")
    cv2.imwrite(str(poster_path), rendered_frames[len(rendered_frames) // 2])


def export_depth_video(
    result: EpisodeDemoResult,
    frames,
    *,
    seq_folder_lookup: dict[str, str],
    buildai_processed_root: Optional[Path],
    output_path: Path,
    fps: int,
) -> None:
    clip_id = result.clip_id or ""
    seq_folder = resolve_seq_folder(
        clip_id,
        seq_folder_lookup=seq_folder_lookup,
        buildai_processed_root=buildai_processed_root,
    )
    if not seq_folder:
        raise ValueError(f"No seq_folder mapping available for clip_id={clip_id}")
    depth_cache_path = _find_depth_cache(seq_folder)
    depth_cache = _load_depth_cache(depth_cache_path) if depth_cache_path is not None else None
    if depth_cache is None:
        raise ValueError(f"No usable depth cache found for clip_id={clip_id}")

    depth_by_index = {
        int(frame_idx): depth_cache["depths"][idx]
        for idx, frame_idx in enumerate(depth_cache["frame_indices"].tolist())
    }
    selected_depths = [depth_by_index[int(frame.frame_idx)] for frame in frames]
    positive_values = np.concatenate([depth[depth > 0] for depth in selected_depths if np.any(depth > 0)], axis=0)
    if positive_values.size == 0:
        raise ValueError(f"Depth cache has no positive values for clip_id={clip_id}")
    vmin = float(np.percentile(positive_values, 5))
    vmax = float(np.percentile(positive_values, 95))
    banner_lines = make_banner_lines(result, render_mode="depth")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_frame = None
    rendered_frames = []
    for idx, frame in enumerate(frames):
        rgb = wv._decode_image_bgr(frame.sample["image_bytes"])
        depth_vis = colorize_depth(depth_by_index[int(frame.frame_idx)], vmin=vmin, vmax=vmax)
        depth_vis = cv2.resize(depth_vis, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        combined = np.concatenate([rgb, depth_vis], axis=1)
        combined = draw_banner(
            combined,
            banner_lines,
            footer=f"frame {idx + 1}/{len(frames)} | depth cache={Path(depth_cache['cache_path']).name}",
        )
        if first_frame is None:
            first_frame = combined
        rendered_frames.append(combined)

    assert first_frame is not None
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    try:
        for image in rendered_frames:
            writer.write(image)
    finally:
        writer.release()

    poster_path = output_path.with_suffix(".poster.jpg")
    cv2.imwrite(str(poster_path), rendered_frames[len(rendered_frames) // 2])


def main():
    args = build_parser().parse_args()
    render_modes = parse_render_modes(args.render_modes)
    seq_folder_lookup = build_seq_folder_lookup(args.descriptor_manifest)
    buildai_processed_root = None if not args.buildai_processed_root else Path(args.buildai_processed_root).expanduser().resolve()
    tar_paths = wv.resolve_tar_paths(args.input)
    summaries = scan_sample_summaries(
        tar_paths,
        sample_limit=args.sample_limit,
        episode_limit=args.episode_limit,
        filter_key=args.filter_key,
        filter_presence=args.filter_presence,
    )
    if not summaries:
        raise SystemExit("No samples matched the current filters.")

    backend = EpisodeViewerBackend(
        summaries,
        descriptor_manifest=args.descriptor_manifest,
        mano_dir=args.mano_dir,
        mano_device=args.mano_device,
    )

    results: list[EpisodeDemoResult] = []
    skips: list[EpisodeDemoSkip] = []
    episode_keys = [str(item["episode_key"]) for item in summarize_episode_candidates(backend.summaries)]

    for idx, episode_key in enumerate(episode_keys, start=1):
        print(f"[demo-rank] {idx}/{len(episode_keys)} episode={episode_key}", flush=True)
        try:
            frames = backend.load_episode_frames(episode_key)
            result = compute_episode_demo_result(
                episode_key,
                frames,
                seq_folder_lookup=seq_folder_lookup,
                buildai_processed_root=buildai_processed_root,
                min_frames=args.min_frames,
            )
        except Exception as error:
            clip_id = next((summary.clip_id for summary in backend.summaries if summary.episode_key == episode_key), None)
            print(f"[demo-rank] skip episode={episode_key}: {error}", flush=True)
            skips.append(EpisodeDemoSkip(episode_key=episode_key, clip_id=clip_id, reason=str(error)))
            continue
        results.append(result)

    results.sort(key=lambda item: (-item.score, -item.frame_count, item.episode_key))
    top_results = results[: max(0, args.top_k)]

    payload = {
        "input": str(Path(args.input).expanduser().resolve()),
        "descriptor_manifest": None if args.descriptor_manifest is None else str(Path(args.descriptor_manifest).expanduser().resolve()),
        "buildai_processed_root": None if buildai_processed_root is None else str(buildai_processed_root),
        "render_modes": render_modes,
        "video_fps": int(args.video_fps),
        "episodes_scored": len(results),
        "episodes_skipped": len(skips),
        "top_k": int(args.top_k),
        "skipped": [asdict(item) for item in skips],
        "results": [asdict(item) for item in results],
        "top_results": [asdict(item) for item in top_results],
    }

    if args.report_out:
        report_path = Path(args.report_out).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    export_dir = Path(args.export_dir).expanduser().resolve() if args.export_dir else None
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)
        for rank, result in enumerate(top_results, start=1):
            try:
                frames = backend.load_episode_frames(result.episode_key)
            except Exception as error:
                print(f"[demo-export] skip {result.episode_key}: failed to reload frames: {error}", flush=True)
                continue
            prefix = f"{rank:02d}_{wv.sanitize_filename(result.clip_id or result.episode_key)}"
            for render_mode in render_modes:
                try:
                    if render_mode == "depth":
                        if not result.exportable.get("depth"):
                            print(f"[demo-export] skip depth {result.episode_key}: depth demo is not exportable", flush=True)
                            continue
                        output_path = export_dir / f"{prefix}.depth.mp4"
                        export_depth_video(
                            result,
                            frames,
                            seq_folder_lookup=seq_folder_lookup,
                            buildai_processed_root=buildai_processed_root,
                            output_path=output_path,
                            fps=args.video_fps,
                        )
                        print(f"[demo-export] wrote {output_path}", flush=True)
                        continue

                    if not result.exportable.get(render_mode):
                        print(f"[demo-export] skip {render_mode} {result.episode_key}: not exportable", flush=True)
                        continue
                    output_path = export_dir / f"{prefix}.{render_mode}.mp4"
                    export_episode_video(
                        backend,
                        result,
                        frames,
                        render_mode=render_mode,
                        output_path=output_path,
                        fps=args.video_fps,
                    )
                    print(f"[demo-export] wrote {output_path}", flush=True)
                except Exception as error:
                    print(f"[demo-export] skip {render_mode} {result.episode_key}: {error}", flush=True)

    print(json.dumps({"top_results": [asdict(item) for item in top_results]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
