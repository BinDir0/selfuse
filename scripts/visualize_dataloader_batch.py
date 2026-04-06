"""
Export one batch and create a mosaic image for visualization.
Usage: python3 scripts/visualize_dataloader_batch.py \
  --dataset /share_data/zhangtingrui/datasets/taco_v2 \
  --window-size 30 \
  --stride 15 \
  --batch-size 4 \
  --output outputs/dataloader_batch_mosaic.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader import EpisodeWindowDataLoader

DEFAULT_DATASET_PATH = "/share_data/zhangtingrui/datasets/taco_v3"
RESAMPLING_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a dataloader batch as a mosaic image.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Dataset directory or shard path.")
    parser.add_argument("--window-size", type=int, default=30, help="Number of frames in each episode window.")
    parser.add_argument("--stride", type=int, default=15, help="Window stride.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of windows per batch.")
    parser.add_argument("--workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--episode", type=str, default=None, help="Optional episode name filter.")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/dataloader_batch_mosaic.jpg",
        help="Output mosaic image path.",
    )
    parser.add_argument(
        "--max-frame-width",
        type=int,
        default=256,
        help="Resize each frame tile to at most this width before composing the mosaic.",
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


def _frame_to_uint8(frame_chw: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame_chw)
    if frame.ndim != 3:
        raise ValueError(f"Expected frame with shape (C,H,W), got {frame.shape}")
    frame = np.transpose(frame, (1, 2, 0))
    if np.issubdtype(frame.dtype, np.floating):
        max_value = float(frame.max()) if frame.size else 0.0
        if max_value <= 1.0:
            frame = frame * 255.0
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _resize_frame(frame_hwc: np.ndarray, max_frame_width: int) -> np.ndarray:
    image = Image.fromarray(frame_hwc)
    if image.width > max_frame_width:
        target_height = max(1, round(image.height * max_frame_width / image.width))
        image = image.resize((max_frame_width, target_height), RESAMPLING_BILINEAR)
    return np.asarray(image)


def _presence_text(existence: np.ndarray) -> str:
    left = int(bool(existence[0]))
    right = int(bool(existence[1]))
    return f"L{left}R{right}"


def create_batch_mosaic(
    batch: dict[str, Any],
    *,
    max_frame_width: int,
    tile_padding: int = 6,
    header_height: int = 20,
) -> Image.Image:
    videos = _to_numpy(batch["video"])
    existence = _to_numpy(batch["existence"])
    frame_indices = _to_numpy(batch["frame_indices"])
    episode_names = list(batch["episode_name"])

    if videos.ndim != 5:
        raise ValueError(f"Expected batch['video'] with shape (B,T,3,H,W), got {videos.shape}")

    batch_size, num_frames = videos.shape[:2]

    processed_frames: list[list[np.ndarray]] = []
    tile_width = 0
    tile_height = 0

    for batch_idx in range(batch_size):
        row_frames: list[np.ndarray] = []
        for frame_idx in range(num_frames):
            frame_hwc = _frame_to_uint8(videos[batch_idx, frame_idx])
            frame_hwc = _resize_frame(frame_hwc, max_frame_width=max_frame_width)
            row_frames.append(frame_hwc)
            tile_height = max(tile_height, frame_hwc.shape[0])
            tile_width = max(tile_width, frame_hwc.shape[1])
        processed_frames.append(row_frames)

    mosaic_width = num_frames * tile_width + (num_frames + 1) * tile_padding
    mosaic_height = batch_size * (tile_height + header_height) + (batch_size + 1) * tile_padding
    canvas = Image.new("RGB", (mosaic_width, mosaic_height), color=(24, 28, 32))
    draw = ImageDraw.Draw(canvas)

    for batch_idx in range(batch_size):
        row_top = tile_padding + batch_idx * (tile_height + header_height + tile_padding)
        header_text = f"sample {batch_idx} | episode={episode_names[batch_idx]}"
        draw.text((tile_padding, row_top), header_text, fill=(230, 230, 230))

        for frame_idx in range(num_frames):
            frame = processed_frames[batch_idx][frame_idx]
            cell_left = tile_padding + frame_idx * (tile_width + tile_padding)
            cell_top = row_top + header_height
            frame_image = Image.fromarray(frame)
            canvas.paste(frame_image, (cell_left, cell_top))

            frame_number = int(frame_indices[batch_idx, frame_idx])
            presence_text = _presence_text(existence[batch_idx, frame_idx])
            text = f"t={frame_number} {presence_text}"

            text_bg_top = cell_top + max(0, frame.shape[0] - 16)
            draw.rectangle(
                [(cell_left, text_bg_top), (cell_left + frame.shape[1], text_bg_top + 16)],
                fill=(0, 0, 0),
            )
            draw.text((cell_left + 3, text_bg_top + 2), text, fill=(255, 255, 255))

    return canvas


def main() -> None:
    args = _parse_args()

    dataloader = EpisodeWindowDataLoader(
        args.dataset,
        window_size=args.window_size,
        stride=args.stride,
        episode_filter=args.episode,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
    )

    try:
        batch = next(iter(dataloader))
    except StopIteration as exc:
        raise RuntimeError("No batch could be loaded from the dataset with the current arguments.") from exc

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mosaic = create_batch_mosaic(batch, max_frame_width=args.max_frame_width)
    mosaic.save(output_path)

    videos = _to_numpy(batch["video"])
    print(f"saved mosaic: {output_path}")
    print(f"batch video shape: {tuple(videos.shape)}")


if __name__ == "__main__":
    main()
