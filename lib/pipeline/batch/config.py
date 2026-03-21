from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from lib.pipeline.video_index import VideoDescriptor


STAGE_ALIASES = {
    "detect": "detect_track",
}

VALID_BATCH_STAGES = ["detect_track", "motion", "slam", "infiller"]


@dataclass(frozen=True)
class BatchRunConfig:
    video_paths: List[str]
    descriptors: Optional[List["VideoDescriptor"]]
    gpus: List[int]
    stages: List[str]
    resume: bool
    run_dir: Path
    checkpoint: str
    infiller_weight: str
    img_focal: Optional[float]
    chunk_batch_size: int
    num_workers: int
    metric3d_batch_size: int
    render_batch_size: int
    infiller_window_batch_size: int
    detect_batch_size: int
    detect_device: str
    detect_half_precision: bool
    detect_io_workers: int
    rebuild_cam_space_cache: bool
    max_stage_retries: int
    workers_per_gpu: int
    detect_track_workers_per_gpu: Optional[int]
    motion_workers_per_gpu: Optional[int]
    slam_workers_per_gpu: Optional[int]
    infiller_workers_per_gpu: Optional[int]
    enable_profiler: bool = False

    @classmethod
    def from_args(cls, args, *, video_paths: List[str], descriptors: Optional[List["VideoDescriptor"]], run_dir: Path):
        raw_stages = [part.strip() for part in args.stages.split(",") if part.strip()]
        stages = [STAGE_ALIASES.get(stage, stage) for stage in raw_stages]
        invalid = [stage for stage in stages if stage not in VALID_BATCH_STAGES]
        if invalid:
            raise ValueError(f"Unknown stages: {invalid}. Valid stages: {VALID_BATCH_STAGES}")

        gpus = [int(gpu.strip()) for gpu in args.gpus.split(",") if gpu.strip()]
        if not gpus:
            raise ValueError("At least one GPU must be specified via --gpus")

        worker_counts = {
            "workers_per_gpu": args.workers_per_gpu,
            "detect_track_workers_per_gpu": args.detect_track_workers_per_gpu,
            "motion_workers_per_gpu": args.motion_workers_per_gpu,
            "slam_workers_per_gpu": args.slam_workers_per_gpu,
            "infiller_workers_per_gpu": args.infiller_workers_per_gpu,
        }
        invalid_counts = {name: value for name, value in worker_counts.items() if value is not None and value < 1}
        if invalid_counts:
            raise ValueError(f"Worker counts must be >= 1: {invalid_counts}")

        return cls(
            video_paths=video_paths,
            descriptors=descriptors,
            gpus=gpus,
            stages=stages,
            resume=args.resume,
            run_dir=run_dir,
            checkpoint=args.checkpoint,
            infiller_weight=args.infiller_weight,
            img_focal=args.img_focal,
            chunk_batch_size=args.chunk_batch_size,
            num_workers=args.num_workers,
            metric3d_batch_size=args.metric3d_batch_size,
            render_batch_size=args.render_batch_size,
            infiller_window_batch_size=args.infiller_window_batch_size,
            detect_batch_size=args.detect_batch_size,
            detect_device=args.detect_device,
            detect_half_precision=args.detect_half_precision,
            detect_io_workers=args.detect_io_workers,
            rebuild_cam_space_cache=args.rebuild_cam_space_cache,
            max_stage_retries=args.max_stage_retries,
            workers_per_gpu=args.workers_per_gpu,
            detect_track_workers_per_gpu=args.detect_track_workers_per_gpu,
            motion_workers_per_gpu=args.motion_workers_per_gpu,
            slam_workers_per_gpu=args.slam_workers_per_gpu,
            infiller_workers_per_gpu=args.infiller_workers_per_gpu,
            enable_profiler=args.enable_profiler,
        )

    @property
    def descriptor_map(self):
        if not self.descriptors:
            return {}
        return {descriptor.video_key: descriptor for descriptor in self.descriptors}

    def worker_count_for_stage(self, stage: str) -> int:
        overrides = {
            "detect_track": self.detect_track_workers_per_gpu,
            "motion": self.motion_workers_per_gpu,
            "slam": self.slam_workers_per_gpu,
            "infiller": self.infiller_workers_per_gpu,
        }
        return overrides.get(stage) or self.workers_per_gpu
