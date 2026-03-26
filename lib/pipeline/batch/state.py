import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from lib.pipeline.batch.config import BatchRunConfig
from lib.pipeline.stage_api import (
    PipelineVideoTask,
    get_stage_done_marker,
)
from lib.pipeline.video_index import VideoDescriptor


@dataclass
class VideoTaskState:
    video_path: str
    video_name: str
    run_id: str
    log_dir: Path
    descriptor: Optional[VideoDescriptor] = None
    stage_status: Dict[str, str] = field(default_factory=dict)
    retry_count: Dict[str, int] = field(default_factory=dict)
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    @classmethod
    def create(cls, video_path: str, stages, run_id: str, log_dir: Path, descriptor: Optional[VideoDescriptor] = None):
        video_name = Path(video_path).stem if descriptor is None else descriptor.video_key
        return cls(
            video_path=video_path,
            video_name=video_name,
            run_id=run_id,
            log_dir=log_dir,
            descriptor=descriptor,
            stage_status={stage: "pending" for stage in stages},
            retry_count={stage: 0 for stage in stages},
        )

    def to_dict(self):
        return {
            "video_path": self.video_path,
            "video_name": self.video_name,
            "stage_status": self.stage_status,
            "retry_count": self.retry_count,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class BatchRunState:
    def __init__(self, config: BatchRunConfig):
        self.config = config
        self.log_dir = config.run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.status_file = config.run_dir / "status.json"
        descriptor_map = config.descriptor_map
        self.tasks = {
            video_path: VideoTaskState.create(
                video_path=video_path,
                stages=config.stages,
                run_id=config.run_dir.name,
                log_dir=self.log_dir,
                descriptor=descriptor_map.get(video_path),
            )
            for video_path in config.video_paths
        }
        self._done_marker_cache: Dict[str, Dict[str, bool]] = defaultdict(dict)

    def _task_seq_folder(self, task: VideoTaskState) -> Path:
        if task.descriptor is not None:
            return Path(task.descriptor.seq_folder)
        return Path(task.video_path).parent / Path(task.video_path).stem

    def _first_incomplete_stage(self, task: VideoTaskState) -> str:
        return next(
            (stage for stage in self.config.stages if task.stage_status.get(stage) != "completed"),
            "unknown",
        )

    def build_pipeline_task(self, video_path: str) -> PipelineVideoTask:
        task = self.tasks[video_path]
        return PipelineVideoTask.from_inputs(video_path=video_path, descriptor=task.descriptor)

    def get_seq_folder(self, video_path: str) -> Path:
        return self._task_seq_folder(self.tasks[video_path])

    def save(self):
        status_data = {
            "run_dir": str(self.config.run_dir),
            "gpus": self.config.gpus,
            "stages": self.config.stages,
            "tasks": {video_path: task.to_dict() for video_path, task in self.tasks.items()},
        }
        with open(self.status_file, "w") as handle:
            json.dump(status_data, handle, indent=2, ensure_ascii=False)

    def load(self):
        if not self.status_file.exists():
            return
        with open(self.status_file) as handle:
            data = json.load(handle)
        for video_path, task_data in data.get("tasks", {}).items():
            if video_path not in self.tasks:
                continue
            task = self.tasks[video_path]
            task.stage_status = task_data.get("stage_status", task.stage_status)
            task.retry_count = task_data.get("retry_count", task.retry_count)
            task.start_time = task_data.get("start_time")
            task.end_time = task_data.get("end_time")

    def prepare_for_resume(self):
        if not self.config.resume:
            return
        self.load()
        self._normalize_resume_states()
        self._print_resume_distribution()

    def _stage_done_marker_exists(self, task: VideoTaskState, stage: str) -> bool:
        cached = self._done_marker_cache[task.video_path]
        if stage not in cached:
            cached[stage] = get_stage_done_marker(self._task_seq_folder(task), stage).exists()
        return cached[stage]

    def _mark_stage_completed_from_done_marker(self, task: VideoTaskState, stage: str) -> bool:
        if not self._stage_done_marker_exists(task, stage):
            return False
        task.stage_status[stage] = "completed"
        return True

    def _normalize_resume_states(self):
        reconciled_done = defaultdict(int)
        normalized_running = defaultdict(int)

        for task in self.tasks.values():
            for stage in self.config.stages:
                current_status = task.stage_status.get(stage, "pending")

                if self._mark_stage_completed_from_done_marker(task, stage):
                    if current_status != "completed":
                        reconciled_done[stage] += 1
                elif current_status == "running":
                    normalized_running[stage] += 1
                    task.stage_status[stage] = "pending"

        if reconciled_done or normalized_running:
            print("\n[Resume Normalization]")
            for stage in self.config.stages:
                if reconciled_done[stage] > 0:
                    print(f"  {stage}: reconciled {reconciled_done[stage]} from .done markers")
                if normalized_running[stage] > 0:
                    print(f"  {stage}: normalized {normalized_running[stage]} stale 'running' -> 'pending'")
            print()

    def _print_resume_distribution(self):
        print("\n[Resume Status Distribution]")
        for stage in self.config.stages:
            status_counts = defaultdict(int)
            for task in self.tasks.values():
                status = task.stage_status.get(stage, "pending")
                status_counts[status] += 1
            print(f"  {stage}: " + ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items())))
        print()

    def mark_batch_started(self):
        timestamp = datetime.now(timezone.utc).isoformat()
        for task in self.tasks.values():
            if task.start_time is None:
                task.start_time = timestamp

    def mark_stage_running(self, video_path: str, stage: str):
        self.tasks[video_path].stage_status[stage] = "running"

    def mark_stage_result(self, video_path: str, stage: str, success: bool):
        self.tasks[video_path].stage_status[stage] = "completed" if success else "failed"
        if success:
            self._done_marker_cache[video_path][stage] = True

    def record_retry(self, video_path: str, stage: str, attempt: int):
        self.tasks[video_path].retry_count[stage] = attempt

    def finalize_videos(self):
        success_count = 0
        fail_count = 0
        completed = []
        failed = []

        for video_path, task in self.tasks.items():
            if all(task.stage_status.get(stage) == "completed" for stage in self.config.stages):
                success_count += 1
                if task.end_time is None:
                    task.end_time = datetime.now(timezone.utc).isoformat()
                completed.append(video_path)
            else:
                fail_count += 1
                failed.append((video_path, self._first_incomplete_stage(task)))

        return success_count, fail_count, completed, failed

    def get_stage_pending_videos(self, stage: str):
        stage_idx = self.config.stages.index(stage)
        prev_stage = self.config.stages[stage_idx - 1] if stage_idx > 0 else None
        excluded_completed = 0
        excluded_running = 0
        excluded_other = 0
        excluded_prev_stage = 0
        excluded_done_marker = 0

        candidates = []
        for video_path in self.config.video_paths:
            task = self.tasks[video_path]

            if prev_stage is not None:
                prev_status = task.stage_status.get(prev_stage, "pending")
                if self.config.resume and prev_status != "completed":
                    self._mark_stage_completed_from_done_marker(task, prev_stage)
                    prev_status = task.stage_status.get(prev_stage, "pending")
                if prev_status != "completed":
                    excluded_prev_stage += 1
                    continue

            current_status = task.stage_status.get(stage, "pending")
            if self.config.resume and current_status != "completed":
                if self._mark_stage_completed_from_done_marker(task, stage):
                    excluded_done_marker += 1
                    continue
                current_status = task.stage_status.get(stage, "pending")

            if current_status not in ("pending", "failed"):
                if current_status == "completed":
                    excluded_completed += 1
                elif current_status == "running":
                    excluded_running += 1
                else:
                    excluded_other += 1
                continue

            candidates.append(video_path)

        if not self.config.resume:
            self._print_stage_eligibility(
                stage,
                scheduled=len(candidates),
                excluded_completed=excluded_completed,
                excluded_running=excluded_running,
                excluded_prev_stage=excluded_prev_stage,
                excluded_done_marker=0,
                excluded_other=excluded_other,
            )
            return candidates

        self._print_stage_eligibility(
            stage,
            scheduled=len(candidates),
            excluded_completed=excluded_completed,
            excluded_running=excluded_running,
            excluded_prev_stage=excluded_prev_stage,
            excluded_done_marker=excluded_done_marker,
            excluded_other=excluded_other,
        )
        return candidates

    def _print_stage_eligibility(
        self,
        stage: str,
        *,
        scheduled: int,
        excluded_completed: int,
        excluded_running: int,
        excluded_prev_stage: int,
        excluded_done_marker: int,
        excluded_other: int,
    ):
        total = len(self.config.video_paths)
        print(
            f"  [{stage}] Eligibility: total={total} scheduled={scheduled} "
            f"excluded_completed={excluded_completed} excluded_running={excluded_running} "
            f"excluded_prev_stage={excluded_prev_stage} excluded_done={excluded_done_marker} "
            f"excluded_other={excluded_other}"
        )
