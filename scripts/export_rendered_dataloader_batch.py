"""
Render one batch with MANO and export the generated video.
Usage: python3 scripts/export_rendered_dataloader_batch.py \
  --dataset /share_data/zhangtingrui/datasets/taco_v3 \
  --window-size 30 \
  --stride 15 \
  --batch-size 4 \
  --fps 10 \
  --output-dir outputs/rendered_batch_v3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader import EpisodeWindowDataLoader
from dataloader.utils import sanitize_key
from vis.mano_render import get_cached_mano_layers, render_hand_on_frame

DEFAULT_DATASET_PATH = "/share_data/zhangtingrui/datasets/taco_v3"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one dataloader batch with MANO and export videos.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Dataset directory or shard path.")
    parser.add_argument("--window-size", type=int, default=30, help="Number of frames in each episode window.")
    parser.add_argument("--stride", type=int, default=15, help="Window stride.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of windows per batch.")
    parser.add_argument("--workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--episode", type=str, default=None, help="Optional episode name filter.")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video fps.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/rendered_batch",
        help="Directory where rendered sample videos will be written.",
    )
    return parser.parse_args()


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _frame_to_rgb_uint8(frame_chw: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame_chw)
    if frame.ndim != 3:
        raise ValueError(f"Expected frame with shape (C,H,W), got {frame.shape}")
    frame = np.transpose(frame, (1, 2, 0))
    if np.issubdtype(frame.dtype, np.floating):
        max_value = float(frame.max()) if frame.size else 0.0
        if max_value <= 1.0:
            frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def _existence_to_presence(existence_row: np.ndarray) -> int:
    existence_row = np.asarray(existence_row).reshape(-1)
    if existence_row.size != 2:
        raise ValueError(f"Expected existence row with shape (2,), got {existence_row.shape}")
    left = int(bool(existence_row[0]))
    right = int(bool(existence_row[1]))
    return left + 2 * right


def _select_row(batch: dict[str, Any], key: str, batch_idx: int) -> np.ndarray:
    value = _to_numpy(batch[key])
    return value[batch_idx]


def render_sample_frames(
    batch: dict[str, Any],
    *,
    batch_idx: int,
    mano_layers: Any,
) -> list[np.ndarray]:
    video = _select_row(batch, "video", batch_idx)
    existence = _select_row(batch, "existence", batch_idx)
    left_translation = _select_row(batch, "left_translation", batch_idx)
    right_translation = _select_row(batch, "right_translation", batch_idx)
    left_rot6 = _select_row(batch, "left_rot6", batch_idx)
    right_rot6 = _select_row(batch, "right_rot6", batch_idx)
    left_hand_pose45 = _select_row(batch, "left_hand_pose45", batch_idx)
    right_hand_pose45 = _select_row(batch, "right_hand_pose45", batch_idx)
    left_shape = _select_row(batch, "left_shape", batch_idx)
    right_shape = _select_row(batch, "right_shape", batch_idx)
    extrinsic = _select_row(batch, "extrinsic_4x4", batch_idx)
    intrinsic = _select_row(batch, "intrinsic", batch_idx)

    rendered_frames: list[np.ndarray] = []
    for frame_idx in range(video.shape[0]):
        frame_rgb = _frame_to_rgb_uint8(video[frame_idx])
        rendered_rgb = render_hand_on_frame(
            frame_rgb,
            mano_params={
                "left": left_hand_pose45[frame_idx],
                "right": right_hand_pose45[frame_idx],
            },
            shape_params={
                "left": left_shape[frame_idx],
                "right": right_shape[frame_idx],
            },
            wrist_params={
                "left_translation": left_translation[frame_idx],
                "right_translation": right_translation[frame_idx],
                "left_rotation": left_rot6[frame_idx],
                "right_rotation": right_rot6[frame_idx],
            },
            extrinsic=extrinsic[frame_idx],
            intrinsic=intrinsic[frame_idx],
            presence=_existence_to_presence(existence[frame_idx]),
            mano_layers=mano_layers,
        )
        rendered_frames.append(np.asarray(rendered_rgb, dtype=np.uint8))
    return rendered_frames


def write_video(frames_rgb: list[np.ndarray], output_path: Path, fps: float) -> None:
    if not frames_rgb:
        raise ValueError("No frames to write.")

    first_frame = np.asarray(frames_rgb[0])
    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        for frame_rgb in frames_rgb:
            frame_bgr = cv2.cvtColor(np.asarray(frame_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()


def main() -> None:
    args = _parse_args()

    dataloader = EpisodeWindowDataLoader(
        args.dataset,
        window_size=args.window_size,
        stride=args.stride,
        episode_filter=args.episode,
        shuffle_windows=False,
        shuffle_buffer_size=0,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
    )

    try:
        batch = next(iter(dataloader))
    except StopIteration as exc:
        raise RuntimeError("No batch could be loaded from the dataset with the current arguments.") from exc

    videos = _to_numpy(batch["video"])
    episode_names = list(batch["episode_name"])
    frame_indices = _to_numpy(batch["frame_indices"])
    output_dir = Path(args.output_dir).resolve()
    mano_layers = get_cached_mano_layers()

    saved_paths: list[Path] = []
    for batch_idx in range(videos.shape[0]):
        rendered_frames = render_sample_frames(batch, batch_idx=batch_idx, mano_layers=mano_layers)
        episode_name = sanitize_key(episode_names[batch_idx])
        frame_start = int(frame_indices[batch_idx, 0])
        frame_end = int(frame_indices[batch_idx, -1])
        output_path = output_dir / f"{batch_idx:02d}_{episode_name}_{frame_start:06d}_{frame_end:06d}.mp4"
        write_video(rendered_frames, output_path, fps=args.fps)
        saved_paths.append(output_path)

    print(f"rendered {len(saved_paths)} videos to {output_dir}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
