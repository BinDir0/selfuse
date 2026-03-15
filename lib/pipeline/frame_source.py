import cv2
import numpy as np
import os

import torch
import torch.utils.data

# Try to import turbojpeg for faster JPEG decoding
try:
    from turbojpeg import TurboJPEG
    TURBOJPEG_AVAILABLE = True
except ImportError:
    TURBOJPEG_AVAILABLE = False

# Check if we should suppress verbose output
QUIET_MODE = os.environ.get("HAWOR_QUIET", "0") == "1"


class BaseFrameSource:
    def __len__(self):
        raise NotImplementedError

    def get_frame(self, index: int, rgb: bool = False):
        raise NotImplementedError

    def iter_frames(self, rgb: bool = False):
        for idx in range(len(self)):
            yield idx, self.get_frame(idx, rgb=rgb)

    def get_size(self):
        frame = self.get_frame(0, rgb=False)
        h, w = frame.shape[:2]
        return h, w


class ImageFolderFrameSource(BaseFrameSource):
    def __init__(self, image_paths, use_turbojpeg=True):
        self.image_paths = list(image_paths)

        if not QUIET_MODE:
            print(f"ImageFolderFrameSource: {len(self.image_paths)} frames")

        if len(self.image_paths) == 0:
            raise RuntimeError("ImageFolderFrameSource requires non-empty image_paths")

        self.use_turbojpeg = use_turbojpeg and TURBOJPEG_AVAILABLE
        if self.use_turbojpeg:
            self.jpeg_decoder = TurboJPEG()
        else:
            self.jpeg_decoder = None

    def __len__(self):
        return len(self.image_paths)

    def get_frame(self, index: int, rgb: bool = False):
        if index < 0 or index >= len(self.image_paths):
            raise IndexError(
                f"Frame index {index} out of range [0, {len(self.image_paths)}). "
                f"Total frames available: {len(self.image_paths)}"
            )

        path = self.image_paths[index]

        # Use turbojpeg for JPEG files if available (2-3x faster than cv2.imread)
        if self.use_turbojpeg and path.lower().endswith(('.jpg', '.jpeg')):
            try:
                with open(path, 'rb') as f:
                    jpeg_data = f.read()
                if rgb:
                    frame = self.jpeg_decoder.decode(jpeg_data, pixel_format=0)  # RGB
                else:
                    frame = self.jpeg_decoder.decode(jpeg_data, pixel_format=1)  # BGR
                return frame
            except Exception:
                pass

        frame = cv2.imread(path)
        if frame is None:
            raise RuntimeError(f"Failed to read image: {path}")
        if rgb:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame


class FrameDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for parallel frame loading via DataLoader."""

    def __init__(self, frame_source: ImageFolderFrameSource):
        self.image_paths = frame_source.image_paths
        self.use_turbojpeg = frame_source.use_turbojpeg
        if self.use_turbojpeg:
            self.jpeg_decoder = TurboJPEG()
        else:
            self.jpeg_decoder = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        if self.use_turbojpeg and path.lower().endswith(('.jpg', '.jpeg')):
            try:
                with open(path, 'rb') as f:
                    jpeg_data = f.read()
                frame = self.jpeg_decoder.decode(jpeg_data, pixel_format=1)  # BGR
                return idx, frame
            except Exception:
                pass

        frame = cv2.imread(path)
        if frame is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return idx, frame


def _numpy_collate(batch):
    """Collate (idx, np_array) pairs without stacking (frames may vary in content)."""
    indices = [b[0] for b in batch]
    frames = [b[1] for b in batch]
    return indices, frames


def _frame_dataset_worker_init(worker_id):
    """Each DataLoader worker needs its own TurboJPEG instance (C library not fork-safe)."""
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if dataset.use_turbojpeg and TURBOJPEG_AVAILABLE:
        dataset.jpeg_decoder = TurboJPEG()


def build_frame_source(video_path: str):
    """Build an ImageFolderFrameSource from pre-extracted frames. Raises if not found."""
    from pathlib import Path
    import glob
    from natsort import natsorted

    video_path_obj = Path(video_path)
    video_dir = video_path_obj.parent
    video_stem = video_path_obj.stem

    extracted_dir = video_dir / video_stem / "extracted_images"

    if not extracted_dir.exists() or not extracted_dir.is_dir():
        raise FileNotFoundError(
            f"Pre-extracted frames not found at: {extracted_dir}\n"
            f"Run frame extraction first: python scripts/extract_frames.py --video_path {video_path}"
        )

    image_files = natsorted(glob.glob(str(extracted_dir / "*.jpg")))
    if not image_files:
        image_files = natsorted(glob.glob(str(extracted_dir / "*.png")))

    if not image_files:
        raise FileNotFoundError(
            f"No image files (jpg/png) found in: {extracted_dir}\n"
            f"Run frame extraction first: python scripts/extract_frames.py --video_path {video_path}"
        )

    if not QUIET_MODE:
        print(f"Using extracted frames from: {extracted_dir} ({len(image_files)} frames)")

    return ImageFolderFrameSource(image_files)
