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
    is_stage_complete,
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
        self.tasks = {}
        descriptor_map = config.descriptor_map
        for video_path in config.video_paths:
            descriptor = descriptor_map.get(video_path)
            self.tasks[video_path] = VideoTaskState.create(
                video_path=video_path,
                stages=config.stages,
                run_id=config.run_dir.name,
                log_dir=self.log_dir,
                descriptor=descriptor,
            )

    def build_pipeline_task(self, video_path: str) -> PipelineVideoTask:
        task = self.tasks[video_path]
        return PipelineVideoTask.from_inputs(video_path=video_path, descriptor=task.descriptor)

    def get_seq_folder(self, video_path: str) -> Path:
        return self.build_pipeline_task(video_path).seq_folder

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
        self._initialize_completed_from_disk()
        self._print_resume_distribution()

    def _normalize_resume_states(self):
        reconciled_done = defaultdict(int)
        normalized_running = defaultdict(int)

        for video_path, task in self.tasks.items():
            seq_folder = self.get_seq_folder(video_path)
            for stage in self.config.stages:
                done_marker = get_stage_done_marker(seq_folder, stage)
                current_status = task.stage_status.get(stage, "pending")

                if done_marker.exists():
                    if current_status != "completed":
                        reconciled_done[stage] += 1
                        task.stage_status[stage] = "completed"
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

    def _initialize_completed_from_disk(self):
        for video_path, task in self.tasks.items():
            if not all(task.stage_status.get(stage) == "pending" for stage in self.config.stages):
                continue
            seq_folder = self.get_seq_folder(video_path)
            for stage in self.config.stages:
                if is_stage_complete(stage, seq_folder, fast_check=True):
                    task.stage_status[stage] = "completed"

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
                failed_stage = next(
                    (stage for stage in self.config.stages if task.stage_status.get(stage) != "completed"),
                    "unknown",
                )
                failed.append((video_path, failed_stage))

        return success_count, fail_count, completed, failed

    def get_stage_pending_videos(self, stage: str):
        stage_idx = self.config.stages.index(stage)
        excluded_completed = 0
        excluded_running = 0
        excluded_other = 0
        excluded_prev_stage = 0
        excluded_done_marker = 0

        candidates = []
        for video_path in self.config.video_paths:
            task = self.tasks[video_path]
            current_status = task.stage_status.get(stage, "pending")

            if stage_idx > 0:
                prev_stage = self.config.stages[stage_idx - 1]
                prev_status = task.stage_status.get(prev_stage, "pending")
                if prev_status != "completed":
                    excluded_prev_stage += 1
                    continue

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

        pending = []
        for video_path in candidates:
            seq_folder = self.get_seq_folder(video_path)
            if is_stage_complete(stage, seq_folder, fast_check=True):
                excluded_done_marker += 1
                continue
            pending.append(video_path)

        self._print_stage_eligibility(
            stage,
            scheduled=len(pending),
            excluded_completed=excluded_completed,
            excluded_running=excluded_running,
            excluded_prev_stage=excluded_prev_stage,
            excluded_done_marker=excluded_done_marker,
            excluded_other=excluded_other,
        )
        return pending

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
