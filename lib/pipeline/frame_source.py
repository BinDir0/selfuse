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


class ShardVideoFrameSource(BaseFrameSource):
    """Frame source that reads a single video's frames from a WebDataset tar shard.

    Uses pre-computed byte offsets to read frames via direct seek+read,
    bypassing tarfile's expensive member index scanning entirely.
    """

    def __init__(self, tar_path, frame_names, frame_offsets=None, use_turbojpeg=True):
        """
        Args:
            tar_path: Path to the tar shard containing this video's frames.
            frame_names: Sorted list of JPEG filenames within the tar for this video.
            frame_offsets: List of [offset, size] pairs parallel to frame_names.
                          If provided, uses direct seek+read (fast path).
                          If None, falls back to tarfile (legacy path).
            use_turbojpeg: Use TurboJPEG for faster decoding if available.
        """
        self.tar_path = tar_path
        self.frame_names = list(frame_names)
        self.frame_offsets = frame_offsets  # [[offset, size], ...]

        if len(self.frame_names) == 0:
            raise RuntimeError(f"ShardVideoFrameSource requires non-empty frame_names for {tar_path}")

        if not QUIET_MODE:
            mode = "direct-seek" if frame_offsets else "tarfile"
            print(f"ShardVideoFrameSource: {len(self.frame_names)} frames from {os.path.basename(tar_path)} ({mode})")

        self.use_turbojpeg = use_turbojpeg and TURBOJPEG_AVAILABLE
        if self.use_turbojpeg:
            self.jpeg_decoder = TurboJPEG()

        # Lazy-opened file handle for direct seek mode
        self._fh = None
        # Lazy-opened tar handle for legacy fallback
        self._tar = None

    def _get_fh(self):
        if self._fh is None:
            self._fh = open(self.tar_path, 'rb')
        return self._fh

    def _get_tar(self):
        if self._tar is None:
            import tarfile
            self._tar = tarfile.open(self.tar_path, 'r')
        return self._tar

    def __len__(self):
        return len(self.frame_names)

    def get_frame(self, index: int, rgb: bool = False):
        if index < 0 or index >= len(self.frame_names):
            raise IndexError(
                f"Frame index {index} out of range [0, {len(self.frame_names)}). "
                f"Total frames available: {len(self.frame_names)}"
            )

        member_name = self.frame_names[index]

        # Fast path: direct seek+read using pre-computed offsets
        if self.frame_offsets is not None:
            offset, size = self.frame_offsets[index]
            fh = self._get_fh()
            fh.seek(offset)
            jpeg_data = fh.read(size)
        else:
            # Legacy fallback: tarfile
            tar = self._get_tar()
            member = tar.getmember(member_name)
            jpeg_data = tar.extractfile(member).read()

        if self.use_turbojpeg and member_name.lower().endswith(('.jpg', '.jpeg')):
            try:
                pixel_format = 0 if rgb else 1  # RGB=0, BGR=1
                frame = self.jpeg_decoder.decode(jpeg_data, pixel_format=pixel_format)
                return frame
            except Exception:
                pass

        frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to decode image from tar: {self.tar_path}/{member_name}")
        if rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __del__(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
        if self._tar is not None:
            try:
                self._tar.close()
            except Exception:
                pass


class FrameDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for parallel frame loading via DataLoader.

    Works with any BaseFrameSource (ImageFolderFrameSource, ShardVideoFrameSource, etc.).
    """

    def __init__(self, frame_source: BaseFrameSource):
        self.frame_source = frame_source
        self.use_turbojpeg = getattr(frame_source, 'use_turbojpeg', False)
        if self.use_turbojpeg:
            self.jpeg_decoder = TurboJPEG()
        else:
            self.jpeg_decoder = None
        # Cache image_paths for ImageFolderFrameSource fast path
        self._image_paths = getattr(frame_source, 'image_paths', None)

    def __len__(self):
        return len(self.frame_source)

    def __getitem__(self, idx):
        # Fast path: ImageFolderFrameSource with TurboJPEG (avoids get_frame overhead)
        if self._image_paths is not None:
            path = self._image_paths[idx]
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

        # Generic path: any BaseFrameSource (ShardVideoFrameSource etc.)
        frame = self.frame_source.get_frame(idx, rgb=False)
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
    # Re-open file/tar handles for ShardVideoFrameSource (not fork-safe)
    fs = dataset.frame_source
    if hasattr(fs, '_fh') and fs._fh is not None:
        fs._fh.close()
        fs._fh = None
    if hasattr(fs, '_tar') and fs._tar is not None:
        fs._tar.close()
        fs._tar = None


class MultiVideoFrameDataset(torch.utils.data.Dataset):
    """Dataset that spans multiple videos for cross-video batched detection.

    Builds a flat index mapping global_idx -> (video_idx, local_frame_idx).
    Workers load frames from any video in parallel.
    """

    def __init__(self, frame_sources):
        """
        Args:
            frame_sources: List of BaseFrameSource, one per video.
        """
        self.frame_sources = frame_sources
        self.use_turbojpeg = TURBOJPEG_AVAILABLE
        if self.use_turbojpeg:
            self.jpeg_decoder = TurboJPEG()
        else:
            self.jpeg_decoder = None

        # Build flat index: [(video_idx, local_frame_idx), ...]
        self._index = []
        for vid_idx, fs in enumerate(frame_sources):
            for local_idx in range(len(fs)):
                self._index.append((vid_idx, local_idx))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, global_idx):
        vid_idx, local_idx = self._index[global_idx]
        frame = self.frame_sources[vid_idx].get_frame(local_idx, rgb=False)
        return vid_idx, local_idx, frame


def _multi_video_collate(batch):
    """Collate for MultiVideoFrameDataset."""
    vid_indices = [b[0] for b in batch]
    local_indices = [b[1] for b in batch]
    frames = [b[2] for b in batch]
    return vid_indices, local_indices, frames


def _multi_video_worker_init(worker_id):
    """Re-init TurboJPEG + file handles for each DataLoader worker."""
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if dataset.use_turbojpeg and TURBOJPEG_AVAILABLE:
        dataset.jpeg_decoder = TurboJPEG()
    # Re-open file/tar handles for all ShardVideoFrameSources (not fork-safe)
    for fs in dataset.frame_sources:
        if hasattr(fs, '_fh') and fs._fh is not None:
            fs._fh.close()
            fs._fh = None
        if hasattr(fs, '_tar') and fs._tar is not None:
            fs._tar.close()
            fs._tar = None
        # Re-init turbojpeg decoder per worker
        if hasattr(fs, 'use_turbojpeg') and fs.use_turbojpeg:
            fs.jpeg_decoder = TurboJPEG()


def build_frame_source(video_path: str):
    """Build an ImageFolderFrameSource from pre-extracted frames.

    For WebDataset format, use ShardVideoFrameSource directly instead.
    """
    from pathlib import Path
    import glob
    from natsort import natsorted

    video_path_obj = Path(video_path)
    video_dir = video_path_obj.parent
    video_stem = video_path_obj.stem

    extracted_dir = video_dir / video_stem / "extracted_images"
    if extracted_dir.exists():
        image_files = natsorted(glob.glob(str(extracted_dir / "*.jpg")))
        if not image_files:
            image_files = natsorted(glob.glob(str(extracted_dir / "*.png")))

        if image_files:
            if not QUIET_MODE:
                print(f"Using extracted frames: {extracted_dir} ({len(image_files)} frames)")
            return ImageFolderFrameSource(image_files)

    raise FileNotFoundError(
        f"No frames found for {video_path}. Expected:\n"
        f"  - JPEG folder: {extracted_dir}/*.jpg\n"
        f"  - Or use ShardVideoFrameSource for WebDataset format"
    )
