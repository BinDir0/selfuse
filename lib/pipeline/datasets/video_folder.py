"""Dataset adapter for video folders with extracted frame directories."""

from __future__ import annotations

from pathlib import Path

from lib.pipeline.datasets.base import BaseDatasetAdapter, register_dataset_adapter
from lib.pipeline.datasets.descriptors import ClipDescriptor
from lib.pipeline.datasets.image_sequence import IMAGE_EXTENSIONS


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def _collect_videos(video_root: Path) -> list[Path]:
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(video_root.rglob(f"*{ext}"))
    return sorted(videos)


def _list_image_names(frame_dir: Path) -> list[str]:
    return sorted(path.name for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


@register_dataset_adapter
class VideoFolderDatasetAdapter(BaseDatasetAdapter):
    name = "video_folder"

    def build_descriptors(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        context=None,
        prepared=None,
    ):
        video_root = Path(adapter_cfg.get("video_root") or paths_cfg.get("video_root", ""))
        if not video_root.is_dir():
            raise FileNotFoundError(f"video_root not found: {video_root}")

        frames_root = Path(adapter_cfg.get("frames_root") or paths_cfg.get("frames_root", video_root))
        seq_folder_root = Path(adapter_cfg.get("seq_folder_root") or (frames_root / "outputs"))
        frame_subdir = adapter_cfg.get("frame_subdir", "extracted_images")

        descriptors = []
        for video_path in _collect_videos(video_root):
            relative_video = video_path.relative_to(video_root)
            relative_parent = relative_video.parent
            relative_stem = relative_video.with_suffix("")
            clip_id = "__".join(relative_stem.parts)
            clip_name = relative_stem.as_posix()
            frame_dir = frames_root / relative_parent / video_path.stem / frame_subdir
            if not frame_dir.is_dir():
                continue
            frame_names = _list_image_names(frame_dir)
            if not frame_names:
                continue
            descriptors.append(
                ClipDescriptor.from_image_sequence(
                    clip_id=clip_id,
                    clip_name=clip_name,
                    root_dir=str(video_root.resolve()),
                    seq_folder=str((seq_folder_root / relative_parent / clip_id).resolve()),
                    frame_dir=str(frame_dir.resolve()),
                    frame_names=frame_names,
                    media_path=str(video_path.resolve()),
                    extra={"adapter": self.name},
                )
            )
        return descriptors
