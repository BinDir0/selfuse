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


class WebDatasetFrameSource(BaseFrameSource):
    """Frame source that reads from WebDataset tar archives."""

    def __init__(self, tar_pattern, use_turbojpeg=True, cache_shards=True):
        """
        Args:
            tar_pattern: Glob pattern for tar files, e.g.,
                        "/path/to/video/frames_*.tar"
            use_turbojpeg: Use TurboJPEG for faster decoding if available
            cache_shards: Keep tar file handles open for faster sequential access
        """
        import glob

        self.tar_files = sorted(glob.glob(tar_pattern))
        if not self.tar_files:
            raise FileNotFoundError(f"No tar files found: {tar_pattern}")

        if not QUIET_MODE:
            print(f"WebDatasetFrameSource: {len(self.tar_files)} tar shards")

        # Build index: frame_idx -> (tar_file, member_name)
        self._build_index()

        self.use_turbojpeg = use_turbojpeg and TURBOJPEG_AVAILABLE
        if self.use_turbojpeg:
            self.jpeg_decoder = TurboJPEG()

        # Shard-level caching for performance
        self.cache_shards = cache_shards
        self._tar_cache = {}  # {tar_path: tarfile.TarFile}
        self._cache_size_limit = 2  # Keep 2 shards open

    def _build_index(self):
        """Scan tar files to build frame index."""
        import tarfile
        self.frame_index = []  # [(tar_path, member_name), ...]

        for tar_path in self.tar_files:
            with tarfile.open(tar_path, 'r') as tar:
                members = sorted([m for m in tar.getmembers() if m.isfile()],
                               key=lambda m: m.name)
                for member in members:
                    if member.name.endswith(('.jpg', '.jpeg', '.png')):
                        self.frame_index.append((tar_path, member.name))

        if not QUIET_MODE:
            print(f"WebDatasetFrameSource: indexed {len(self.frame_index)} frames")

    def __len__(self):
        return len(self.frame_index)

    def get_frame(self, index: int, rgb: bool = False):
        import tarfile

        if index < 0 or index >= len(self.frame_index):
            raise IndexError(
                f"Frame index {index} out of range [0, {len(self.frame_index)}). "
                f"Total frames available: {len(self.frame_index)}"
            )

        tar_path, member_name = self.frame_index[index]

        # Use cached tar file if available
        if self.cache_shards and tar_path in self._tar_cache:
            tar = self._tar_cache[tar_path]
        else:
            tar = tarfile.open(tar_path, 'r')
            if self.cache_shards:
                self._tar_cache[tar_path] = tar
                # LRU eviction if cache full
                if len(self._tar_cache) > self._cache_size_limit:
                    oldest = next(iter(self._tar_cache))
                    self._tar_cache[oldest].close()
                    del self._tar_cache[oldest]

        member = tar.getmember(member_name)
        f = tar.extractfile(member)
        jpeg_data = f.read()

        if self.use_turbojpeg and member_name.lower().endswith(('.jpg', '.jpeg')):
            try:
                pixel_format = 0 if rgb else 1  # RGB=0, BGR=1
                frame = self.jpeg_decoder.decode(jpeg_data, pixel_format=pixel_format)
                return frame
            except Exception:
                pass

        frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to decode image from tar: {tar_path}/{member_name}")
        if rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __del__(self):
        """Close all cached tar files on cleanup."""
        for tar in self._tar_cache.values():
            try:
                tar.close()
            except Exception:
                pass


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
    """Build frame source from pre-extracted frames (JPEG or WebDataset).

    Auto-detects format:
    - If frames_*.tar exists → WebDatasetFrameSource (new format)
    - If extracted_images/ exists with JPEGs → ImageFolderFrameSource (old format)
    """
    from pathlib import Path
    import glob
    from natsort import natsorted

    video_path_obj = Path(video_path)
    video_dir = video_path_obj.parent
    video_stem = video_path_obj.stem

    base_dir = video_dir / video_stem

    # Check for WebDataset tar files first (new format)
    tar_pattern = str(base_dir / "frames_*.tar")
    tar_files = glob.glob(tar_pattern)
    if tar_files:
        if not QUIET_MODE:
            print(f"Using WebDataset frames: {tar_pattern} ({len(tar_files)} shards)")
        return WebDatasetFrameSource(tar_pattern)

    # Fall back to JPEG folder (old format)
    extracted_dir = base_dir / "extracted_images"
    if extracted_dir.exists():
        image_files = natsorted(glob.glob(str(extracted_dir / "*.jpg")))
        if not image_files:
            image_files = natsorted(glob.glob(str(extracted_dir / "*.png")))

        if image_files:
            if not QUIET_MODE:
                print(f"Using extracted frames: {extracted_dir} ({len(image_files)} frames)")
            return ImageFolderFrameSource(image_files)

    raise FileNotFoundError(
        f"No frames found for {video_path}. Expected either:\n"
        f"  - WebDataset: {tar_pattern}\n"
        f"  - JPEG folder: {extracted_dir}/*.jpg"
    )
