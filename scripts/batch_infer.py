#!/usr/bin/env python3
"""
Multi-GPU batch inference scheduler for HaWoR.

Schedules multiple videos across GPU workers, executing stages sequentially
per video with retry logic, resume/skip support, and structured logging.
"""
import argparse
import json
import multiprocessing as mp
import numpy as np
import os
import subprocess
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# Suppress common warnings to reduce output noise
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', message='.*timm.models.layers.*')

# Set temporary directory to shared storage instead of local /tmp
SHARED_TMP_DIR = Path("/share_data/guantianrui/tmp")
SHARED_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(SHARED_TMP_DIR)
os.environ["TEMP"] = str(SHARED_TMP_DIR)
os.environ["TMP"] = str(SHARED_TMP_DIR)
tempfile.tempdir = str(SHARED_TMP_DIR)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add project root to path for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.frame_source import build_frame_source
from lib.pipeline.runtime import WorkerRuntime, set_determinism
from lib.pipeline.stage_api import (
    STAGES,
    PipelineVideoTask,
    get_stage_done_marker,
    get_track_range,
    get_tracks_dir,
    is_stage_complete,
    run_pipeline_stage,
)
from lib.pipeline.video_index import VideoDescriptor, collect_videos_from_factory, collect_videos_from_factories


class VideoTask:
    def __init__(self, video_path: str, run_id: str, log_dir: Path, descriptor: VideoDescriptor = None):
        self.video_path = video_path
        self.video_name = Path(video_path).stem if not descriptor else descriptor.video_key
        self.run_id = run_id
        self.log_dir = log_dir
        self.descriptor = descriptor
        self.stage_status = {stage: "pending" for stage in STAGES}
        self.retry_count = {stage: 0 for stage in STAGES}
        self.start_time = None
        self.end_time = None

    def to_dict(self):
        return {
            "video_path": self.video_path,
            "video_name": self.video_name,
            "stage_status": self.stage_status,
            "retry_count": self.retry_count,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class BatchScheduler:
    def __init__(
        self,
        video_paths: List[str],
        gpus: List[int],
        stages: List[str],
        max_retries: int,
        resume: bool,
        run_dir: Path,
        checkpoint: str,
        infiller_weight: str,
        img_focal: Optional[float],
        chunk_batch_size: int,
        num_workers: int,
        metric3d_batch_size: int,
        render_batch_size: int,
        infiller_window_batch_size: int,
        detect_batch_size: int,
        detect_device: str,
        detect_half_precision: bool,
        detect_io_workers: int,
        rebuild_cam_space_cache: bool,
        scheduler_mode: str,
        persistent_worker: bool,
        max_stage_retries: int,
        enable_profiler: bool = False,
        descriptors: List[VideoDescriptor] = None,
    ):
        self.video_paths = video_paths
        self.gpus = gpus
        self.stages = stages
        self.max_retries = max_retries
        self.resume = resume
        self.run_dir = run_dir
        self.checkpoint = checkpoint
        self.infiller_weight = infiller_weight
        self.img_focal = img_focal
        self.chunk_batch_size = chunk_batch_size
        self.num_workers = num_workers
        self.metric3d_batch_size = metric3d_batch_size
        self.render_batch_size = render_batch_size
        self.infiller_window_batch_size = infiller_window_batch_size
        self.detect_batch_size = detect_batch_size
        self.detect_device = detect_device
        self.detect_half_precision = detect_half_precision
        self.detect_io_workers = detect_io_workers
        self.rebuild_cam_space_cache = rebuild_cam_space_cache
        self.scheduler_mode = scheduler_mode
        self.persistent_worker = persistent_worker
        self.max_stage_retries = max_stage_retries
        self.enable_profiler = enable_profiler
        self.descriptors = descriptors  # Optional list of VideoDescriptors (same length as video_paths)

        self.log_dir = run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.status_file = run_dir / "status.json"
        self.events_file = run_dir / "events.jsonl"

        self.tasks: Dict[str, VideoTask] = {}
        self.lock = mp.Lock()

        for i, vp in enumerate(video_paths):
            desc = descriptors[i] if descriptors else None
            task = VideoTask(vp, run_dir.name, self.log_dir, descriptor=desc)
            self.tasks[vp] = task

    def _build_pipeline_task(self, video_path: str) -> PipelineVideoTask:
        task = self.tasks.get(video_path)
        descriptor = task.descriptor if task else None
        return PipelineVideoTask.from_inputs(video_path=video_path, descriptor=descriptor)

    def _get_seq_folder(self, video_path: str) -> Path:
        return self._build_pipeline_task(video_path).seq_folder

    def emit_event(self, event: str, **kwargs):
        payload = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **kwargs,
        }
        with open(self.events_file, "a") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _update_progress_bar(pbar, stage_results: Dict[str, bool]):
        if not pbar:
            return
        pbar.update(1)
        pbar.set_postfix({
            "success": sum(1 for ok in stage_results.values() if ok),
            "failed": sum(1 for ok in stage_results.values() if not ok),
        })

    def save_status(self):
        with self.lock:
            status_data = {
                "run_dir": str(self.run_dir),
                "gpus": self.gpus,
                "stages": self.stages,
                "tasks": {vp: task.to_dict() for vp, task in self.tasks.items()},
            }
            with open(self.status_file, "w") as f:
                json.dump(status_data, f, indent=2, ensure_ascii=False)

    def load_status(self):
        if not self.status_file.exists():
            return
        with open(self.status_file) as f:
            data = json.load(f)
        for vp, task_data in data.get("tasks", {}).items():
            if vp in self.tasks:
                self.tasks[vp].stage_status = task_data.get("stage_status", {})
                self.tasks[vp].retry_count = task_data.get("retry_count", {})
                self.tasks[vp].start_time = task_data.get("start_time")
                self.tasks[vp].end_time = task_data.get("end_time")

    def run_stage_subprocess(
        self, video_path: str, stage: str, gpu: int
    ) -> Tuple[int, str, str]:
        task = self.tasks[video_path]
        log_file = self.log_dir / f"{task.video_name}_{stage}.log"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "batch_worker.py"),
            "--stage", stage,
            "--video_path", video_path,
            "--gpu", str(gpu),
            "--checkpoint", self.checkpoint,
            "--infiller_weight", self.infiller_weight,
        ]
        # Pass descriptor if available
        if task.descriptor is not None:
            cmd.extend(["--video_descriptor", task.descriptor.to_json()])
        if self.img_focal is not None:
            cmd.extend(["--img_focal", str(self.img_focal)])
        cmd.extend(["--run_dir", str(self.run_dir)])
        cmd.extend(["--chunk_batch_size", str(self.chunk_batch_size)])
        cmd.extend(["--num_workers", str(self.num_workers)])
        cmd.extend(["--metric3d_batch_size", str(self.metric3d_batch_size)])
        cmd.extend(["--render_batch_size", str(self.render_batch_size)])
        cmd.extend(["--infiller_window_batch_size", str(self.infiller_window_batch_size)])
        cmd.extend(["--detect_batch_size", str(self.detect_batch_size)])
        cmd.extend(["--detect_io_workers", str(self.detect_io_workers)])
        cmd.extend(["--detect_device", self.detect_device])
        if self.rebuild_cam_space_cache:
            cmd.append("--rebuild_cam_space_cache")
        if self.detect_half_precision:
            cmd.append("--detect_half_precision")
        else:
            cmd.append("--no-detect_half_precision")
        if self.enable_profiler:
            cmd.append("--enable_profiler")
        if self.resume:
            cmd.append("--resume")
        else:
            cmd.append("--force")

        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
            )
            proc.wait()

        with open(log_file) as f:
            log_content = f.read()

        return proc.returncode, str(log_file), log_content

    def run_stage_persistent_subprocess(
        self, video_paths: List[str], stage: str, gpu: int
    ) -> Tuple[Dict[str, bool], str]:
        if not video_paths:
            return {}, ""

        list_file = self.run_dir / f"stage_{stage}_gpu{gpu}_videos.txt"
        with open(list_file, "w") as f:
            for vp in video_paths:
                task = self.tasks[vp]
                if task.descriptor is not None:
                    # Write as JSON Lines for WebDataset mode
                    f.write(task.descriptor.to_json() + "\n")
                else:
                    f.write(vp + "\n")

        log_file = self.log_dir / f"stage_wave_{stage}_gpu{gpu}.log"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "batch_worker.py"),
            "--persistent_worker",
            "--stage", stage,
            "--video_list", str(list_file),
            "--gpu", str(gpu),
            "--checkpoint", self.checkpoint,
            "--infiller_weight", self.infiller_weight,
            "--chunk_batch_size", str(self.chunk_batch_size),
            "--num_workers", str(self.num_workers),
            "--render_batch_size", str(self.render_batch_size),
            "--metric3d_batch_size", str(self.metric3d_batch_size),
            "--infiller_window_batch_size", str(self.infiller_window_batch_size),
            "--detect_batch_size", str(self.detect_batch_size),
            "--detect_io_workers", str(self.detect_io_workers),
            "--detect_device", self.detect_device,
        ]
        if self.img_focal is not None:
            cmd.extend(["--img_focal", str(self.img_focal)])
        cmd.extend(["--run_dir", str(self.run_dir)])
        if self.rebuild_cam_space_cache:
            cmd.append("--rebuild_cam_space_cache")
        if self.detect_half_precision:
            cmd.append("--detect_half_precision")
        else:
            cmd.append("--no-detect_half_precision")
        if self.enable_profiler:
            cmd.append("--enable_profiler")
        if self.resume:
            cmd.append("--resume")
        else:
            cmd.append("--force")

        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
            )
            proc.wait()

        status_by_video = {vp: False for vp in video_paths}
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("event") != "stage_end":
                    continue
                video = obj.get("video")
                if video not in status_by_video:
                    continue
                status = obj.get("status")
                status_by_video[video] = status in ("success", "skipped")

        return status_by_video, str(log_file)

    def verify_stage_complete(self, video_path: str, stage: str) -> bool:
        """Verify that stage output actually exists on disk (fast check)."""
        try:
            seq_folder = self._get_seq_folder(video_path)
            return is_stage_complete(stage, seq_folder, fast_check=True)
        except Exception:
            return False

    def get_stage_pending_videos(self, stage: str) -> List[str]:
        """Get list of videos that need processing for this stage."""
        stage_idx = self.stages.index(stage)

        # Track exclusion reasons for diagnostics
        excluded_completed = 0
        excluded_running = 0
        excluded_other = 0
        excluded_prev_stage = 0
        excluded_done_marker = 0

        # First filter by status.json
        candidates = []
        for vp in self.video_paths:
            task = self.tasks[vp]
            current_status = task.stage_status.get(stage, "pending")

            # First stage: only pending/failed are schedulable
            if stage_idx == 0:
                if current_status not in ("pending", "failed"):
                    if current_status == "completed":
                        excluded_completed += 1
                    elif current_status == "running":
                        excluded_running += 1
                    else:
                        excluded_other += 1
                    continue
            else:
                # Later stages: previous must be completed
                prev_stage = self.stages[stage_idx - 1]
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

            candidates.append(vp)

        if not self.resume:
            total = len(self.video_paths)
            print(f"  [{stage}] Eligibility: total={total} scheduled={len(candidates)} "
                  f"excluded_completed={excluded_completed} excluded_running={excluded_running} "
                  f"excluded_prev_stage={excluded_prev_stage} excluded_done=0 "
                  f"excluded_other={excluded_other}")
            return candidates

        # Then filter by on-disk outputs using the shared stage API.
        pending = []
        for vp in candidates:
            seq_folder = self._get_seq_folder(vp)
            if is_stage_complete(stage, seq_folder, fast_check=True):
                excluded_done_marker += 1
                continue
            pending.append(vp)

        # Log exclusion breakdown
        total = len(self.video_paths)
        print(f"  [{stage}] Eligibility: total={total} scheduled={len(pending)} "
              f"excluded_completed={excluded_completed} excluded_running={excluded_running} "
              f"excluded_prev_stage={excluded_prev_stage} excluded_done={excluded_done_marker} "
              f"excluded_other={excluded_other}")

        return pending

    def run_stage_wave_dynamic(self, stage: str, pbar=None) -> Dict[str, bool]:
        """Run stage wave with dynamic load balancing across GPUs.

        Args:
            stage: Stage name to run
            pbar: Optional tqdm progress bar to update
        """
        pending_videos = self.get_stage_pending_videos(stage)
        if not pending_videos:
            return {}

        self.emit_event("wave_start", stage=stage, total=len(pending_videos))

        # Create shared queues for dynamic load balancing
        video_queue = mp.Queue()
        result_queue = mp.Queue()

        # Populate video queue
        for vp in pending_videos:
            video_queue.put(vp)

        # Add sentinel values to signal workers to stop
        for _ in self.gpus:
            video_queue.put(None)

        # Launch worker processes for each GPU
        workers = []
        for gpu in self.gpus:
            p = mp.Process(
                target=self.stage_wave_worker_dynamic,
                args=(gpu, stage, video_queue, result_queue)
            )
            p.start()
            workers.append(p)

        # Collect results
        stage_results = {}
        completed = 0
        total = len(pending_videos)
        last_save = 0

        while completed < total:
            try:
                result = result_queue.get(timeout=1)
            except:
                # Check if all workers are done
                if all(not w.is_alive() for w in workers):
                    break
                continue

            video_path = result["video"]
            success = result["success"]
            gpu = result["gpu"]

            task = self.tasks[video_path]
            if success:
                task.stage_status[stage] = "completed"
                self.emit_event("stage_success", video=video_path, stage=stage, gpu=gpu)
            else:
                task.stage_status[stage] = "failed"
                self.emit_event("stage_failure", video=video_path, stage=stage, gpu=gpu)

            stage_results[video_path] = success
            completed += 1

            self._update_progress_bar(pbar, stage_results)

            # Drain any additional results that are already in the queue (no blocking)
            while not result_queue.empty():
                try:
                    result = result_queue.get_nowait()
                except:
                    break
                video_path = result["video"]
                success = result["success"]
                gpu = result["gpu"]
                task = self.tasks[video_path]
                if success:
                    task.stage_status[stage] = "completed"
                    self.emit_event("stage_success", video=video_path, stage=stage, gpu=gpu)
                else:
                    task.stage_status[stage] = "failed"
                    self.emit_event("stage_failure", video=video_path, stage=stage, gpu=gpu)
                stage_results[video_path] = success
                completed += 1
                self._update_progress_bar(pbar, stage_results)

            # Save status periodically (not every single video — reduces I/O)
            if completed - last_save >= 10 or completed >= total:
                self.save_status()
                last_save = completed

        # Wait for all workers to finish
        for w in workers:
            w.join()

        self.emit_event(
            "wave_end",
            stage=stage,
            success=sum(1 for ok in stage_results.values() if ok),
            failed=sum(1 for ok in stage_results.values() if not ok),
        )
        return stage_results

    def _prefetch_video_data(self, video_path: str, stage: str, runtime):
        """Prefetch video data (frame source, tracks) in background thread.

        Only useful for 'motion' stage where data loading is significant.
        Returns a dict with prefetched data, or None if prefetch not applicable.
        """
        if stage != "motion":
            return None

        try:
            pipeline_task = self._build_pipeline_task(video_path)
            seq_folder = pipeline_task.seq_folder

            # Skip if already complete (no need to prefetch)
            if self.resume and is_stage_complete(stage, seq_folder, fast_check=True):
                return None

            start_idx, end_idx = get_track_range(seq_folder)
            tracks_dir = get_tracks_dir(seq_folder, start_idx, end_idx)

            # Check if output already exists (skip check)
            frame_chunks_file = tracks_dir / "frame_chunks_all.npy"
            model_masks_file = tracks_dir / "model_masks.npy"
            if self.resume and frame_chunks_file.exists() and model_masks_file.exists():
                return None

            # Prefetch frame source and tracks (IO-bound operations)
            frame_source = pipeline_task.build_frame_source() or build_frame_source(video_path)
            tracks = np.load(tracks_dir / "model_tracks.npy", allow_pickle=True).item()

            return {
                'frame_source': frame_source,
                'tracks': tracks,
            }
        except Exception:
            # Prefetch failure is non-fatal; data will be loaded normally
            return None

    def stage_wave_worker_dynamic(self, gpu: int, stage: str, video_queue: mp.Queue, result_queue: mp.Queue):
        """Worker process that pulls videos from queue and processes them with model reuse."""
        # Set GPU for this worker
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        # Initialize runtime once for this worker
        set_determinism(42)
        runtime = WorkerRuntime(
            gpu=str(gpu),
            checkpoint=self.checkpoint,
            infiller_weight=self.infiller_weight,
            img_focal=self.img_focal,
            input_type="file",
            chunk_batch_size=self.chunk_batch_size,
            num_workers=self.num_workers,
            render_batch_size=self.render_batch_size,
            metric3d_batch_size=self.metric3d_batch_size,
            detect_batch_size=self.detect_batch_size,
            detect_io_workers=self.detect_io_workers,
            detect_device=self.detect_device,
            detect_half_precision=self.detect_half_precision,
            infiller_window_batch_size=self.infiller_window_batch_size,
            rebuild_cam_space_cache=self.rebuild_cam_space_cache,
        )

        # Ensure stage models are loaded
        runtime.ensure_runner(stage)

        # Standard single-video processing with prefetching
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as prefetcher:
            # Get first video
            try:
                video_path = video_queue.get(timeout=1)
            except:
                video_path = None

            prefetch_future = None

            while video_path is not None:
                # Get prefetched data for current video (if available)
                prefetched_data = None
                if prefetch_future is not None:
                    try:
                        prefetched_data = prefetch_future.result()
                    except Exception:
                        prefetched_data = None

                # Peek at next video and start prefetching
                try:
                    next_video = video_queue.get(timeout=0)
                except:
                    next_video = None

                next_future = None
                if next_video is not None:
                    next_future = prefetcher.submit(
                        self._prefetch_video_data, next_video, stage, runtime
                    )

                # Process current video (GPU busy here)
                success = self.run_single_video_with_runtime(
                    video_path, stage, gpu, runtime,
                    prefetched_data=prefetched_data,
                )

                # Report result
                result_queue.put({
                    "video": video_path,
                    "success": success,
                    "gpu": gpu,
                })

                # Move to next
                video_path = next_video
                prefetch_future = next_future

    def run_single_video_with_runtime(self, video_path: str, stage: str, gpu: int, runtime, prefetched_data=None) -> bool:
        """Run a single stage for a single video using existing runtime."""
        try:
            pipeline_task = self._build_pipeline_task(video_path)
            result = run_pipeline_stage(
                stage,
                pipeline_task,
                runtime.stage_config,
                runtime=runtime,
                prefetched_data=prefetched_data,
                resume=self.resume,
                force=not self.resume,
            )
            success = result.get("status") in ("success", "skipped")
            if not success:
                print(f"WARNING: run_pipeline_stage returned unexpected status: {result.get('status')}")
            return success
        except Exception as e:
            print(f"Error processing {video_path} on GPU {gpu}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_stage_wave(self, stage: str, pbar=None) -> Dict[str, bool]:
        """Run stage wave - delegates to dynamic load balancing implementation."""
        return self.run_stage_wave_dynamic(stage, pbar=pbar)

    def run_stage_wave_static(self, stage: str) -> Dict[str, bool]:
        """Original static load balancing (kept for reference/fallback)."""

        pending_videos = self.get_stage_pending_videos(stage)
        if not pending_videos:
            return {}

        self.emit_event("wave_start", stage=stage, total=len(pending_videos))

        groups = defaultdict(list)
        for i, vp in enumerate(pending_videos):
            gpu = self.gpus[i % len(self.gpus)]
            groups[gpu].append(vp)

        stage_results: Dict[str, bool] = {}
        for gpu, videos in groups.items():
            for vp in videos:
                task = self.tasks[vp]
                task.stage_status[stage] = "running"
            self.save_status()

            result_map, log_file = self.run_stage_persistent_subprocess(videos, stage, gpu)
            for vp in videos:
                ok = result_map.get(vp, False)
                task = self.tasks[vp]
                if ok:
                    task.stage_status[stage] = "completed"
                    self.emit_event("stage_success", video=vp, stage=stage, gpu=gpu, log_file=log_file)
                else:
                    task.stage_status[stage] = "failed"
                    self.emit_event("stage_failure", video=vp, stage=stage, gpu=gpu, log_file=log_file)
                stage_results[vp] = ok
            self.save_status()

        self.emit_event(
            "wave_end",
            stage=stage,
            success=sum(1 for ok in stage_results.values() if ok),
            failed=sum(1 for ok in stage_results.values() if not ok),
        )
        return stage_results

    def run_stage_wave_with_retries(self, stage: str, pbar=None) -> Dict[str, bool]:
        final_results: Dict[str, bool] = {}
        for attempt in range(self.max_stage_retries + 1):
            stage_results = self.run_stage_wave(stage, pbar=pbar)
            if not stage_results:
                break

            for vp, ok in stage_results.items():
                final_results[vp] = ok
                self.tasks[vp].retry_count[stage] = attempt

            failed = [vp for vp, ok in stage_results.items() if not ok]
            if not failed:
                break
            if attempt < self.max_stage_retries:
                self.emit_event("wave_retry", stage=stage, attempt=attempt + 1, failed=len(failed))

        return final_results

    def _normalize_resume_states(self):
        """
        Normalize stale or inconsistent stage states after resume.

        Fixes two common issues:
        1. Stale 'running' states from interrupted processes
        2. Mismatch between .done markers and in-memory status
        """
        # Track normalization counts for logging
        reconciled_done = defaultdict(int)
        normalized_running = defaultdict(int)

        for vp in self.video_paths:
            task = self.tasks[vp]
            seq_folder = self._get_seq_folder(vp)

            for stage in self.stages:
                done_marker = get_stage_done_marker(seq_folder, stage)
                current_status = task.stage_status.get(stage, "pending")

                # Priority 1: .done marker exists → force to completed
                if done_marker.exists():
                    if current_status != "completed":
                        reconciled_done[stage] += 1
                        task.stage_status[stage] = "completed"

                # Priority 2: stale 'running' without .done → reset to pending
                elif current_status == "running":
                    normalized_running[stage] += 1
                    task.stage_status[stage] = "pending"

        # Log normalization summary
        if reconciled_done or normalized_running:
            print("\n[Resume Normalization]")
            for stage in self.stages:
                if reconciled_done[stage] > 0:
                    print(f"  {stage}: reconciled {reconciled_done[stage]} from .done markers")
                if normalized_running[stage] > 0:
                    print(f"  {stage}: normalized {normalized_running[stage]} stale 'running' → 'pending'")
            print()

    def run_wave(self):
        if self.resume:
            self.load_status()

            # Normalize stale/inconsistent states on resume
            self._normalize_resume_states()

            # Log status distribution for diagnostics
            print("\n[Resume Status Distribution]")
            for stage in self.stages:
                status_counts = defaultdict(int)
                for vp in self.video_paths:
                    task = self.tasks[vp]
                    status = task.stage_status.get(stage, "pending")
                    status_counts[status] += 1

                print(f"  {stage}: ", end="")
                print(", ".join(f"{status}={count}" for status, count in sorted(status_counts.items())))
            print()

        # Initialize tasks that still carry the default all-pending state from on-disk outputs.
        if self.resume:
            for vp in self.video_paths:
                task = self.tasks[vp]

                if all(task.stage_status.get(stage) == "pending" for stage in self.stages):
                    seq_folder = self._get_seq_folder(vp)
                    for stage in self.stages:
                        if is_stage_complete(stage, seq_folder, fast_check=True):
                            task.stage_status[stage] = "completed"

        self.emit_event("batch_start", total_videos=len(self.video_paths), gpus=self.gpus, mode="wave")

        for vp in self.video_paths:
            task = self.tasks[vp]
            if task.start_time is None:
                task.start_time = datetime.now(timezone.utc).isoformat()

        print(f"\n{'='*60}")
        print(f"Stage-Wave Scheduling: {len(self.video_paths)} videos, {len(self.gpus)} GPUs")
        print(f"{'='*60}\n")

        for stage_idx, stage in enumerate(self.stages, 1):
            pending = self.get_stage_pending_videos(stage)
            total_for_stage = len(pending)

            print(f"Stage {stage_idx}/{len(self.stages)}: {stage} ({total_for_stage} videos)")

            with tqdm(total=total_for_stage, desc=f"  {stage}", unit="video", leave=True) as pbar:
                # Run stage wave with progress bar
                original_results = self.run_stage_wave_with_retries(stage, pbar=pbar)

        success_count = 0
        fail_count = 0
        for vp in self.video_paths:
            task = self.tasks[vp]
            if all(task.stage_status.get(s) == "completed" for s in self.stages):
                success_count += 1
                if task.end_time is None:
                    task.end_time = datetime.now(timezone.utc).isoformat()
                self.emit_event("video_completed", video=vp)
            else:
                fail_count += 1
                failed_stage = next((s for s in self.stages if task.stage_status.get(s) != "completed"), "unknown")
                self.emit_event("video_failed", video=vp, stage=failed_stage)

        self.save_status()
        self.emit_event("batch_end", total=len(self.video_paths), success=success_count, failed=fail_count)

        print(f"\n{'='*60}")
        print(f"Batch Inference Complete")
        print(f"{'='*60}")
        print(f"Total videos: {len(self.video_paths)}")
        print(f"Success: {success_count}")
        print(f"Failed: {fail_count}")
        print(f"Run directory: {self.run_dir}")
        print(f"Status file: {self.status_file}")
        print(f"Events log: {self.events_file}")
        print(f"{'='*60}\n")

        return fail_count == 0

    def process_video(self, video_path: str, gpu: int):
        task = self.tasks[video_path]
        task.start_time = datetime.now(timezone.utc).isoformat()
        self.emit_event("video_start", video=video_path, gpu=gpu)

        for stage in self.stages:
            # Check both status.json AND actual disk artifacts
            if task.stage_status[stage] == "completed":
                if self.verify_stage_complete(video_path, stage):
                    self.emit_event("stage_skip", video=video_path, stage=stage, gpu=gpu)
                    continue
                else:
                    # Status says completed but output missing - need to rerun
                    self.emit_event(
                        "stage_revalidate",
                        video=video_path,
                        stage=stage,
                        gpu=gpu,
                        reason="output_missing"
                    )
                    task.stage_status[stage] = "pending"

            success = False
            for attempt in range(self.max_retries + 1):
                task.retry_count[stage] = attempt
                task.stage_status[stage] = "running"
                self.save_status()

                self.emit_event(
                    "stage_attempt",
                    video=video_path,
                    stage=stage,
                    gpu=gpu,
                    attempt=attempt,
                )

                returncode, log_file, log_content = self.run_stage_subprocess(
                    video_path, stage, gpu
                )

                if returncode == 0:
                    task.stage_status[stage] = "completed"
                    self.save_status()
                    self.emit_event(
                        "stage_success",
                        video=video_path,
                        stage=stage,
                        gpu=gpu,
                        attempt=attempt,
                        log_file=log_file,
                    )
                    success = True
                    break
                else:
                    self.emit_event(
                        "stage_failure",
                        video=video_path,
                        stage=stage,
                        gpu=gpu,
                        attempt=attempt,
                        returncode=returncode,
                        log_file=log_file,
                    )

            if not success:
                task.stage_status[stage] = "failed"
                self.save_status()
                self.emit_event("video_failed", video=video_path, stage=stage, gpu=gpu)
                return False

        task.end_time = datetime.now(timezone.utc).isoformat()
        self.save_status()
        self.emit_event("video_completed", video=video_path, gpu=gpu)
        return True

    def worker_loop(self, gpu: int, video_queue: mp.Queue, result_queue: mp.Queue, progress_queue: mp.Queue):
        while True:
            try:
                video_path = video_queue.get(timeout=1)
            except:
                break

            if video_path is None:
                break

            success = self.process_video(video_path, gpu)
            result_queue.put((video_path, success))
            progress_queue.put(1)  # Signal completion

    def run(self):
        if self.scheduler_mode == "wave":
            return self.run_wave()

        if self.resume:
            self.load_status()

        self.emit_event("batch_start", total_videos=len(self.video_paths), gpus=self.gpus, mode="legacy")

        video_queue = mp.Queue()
        result_queue = mp.Queue()
        progress_queue = mp.Queue()

        for vp in self.video_paths:
            video_queue.put(vp)

        for _ in self.gpus:
            video_queue.put(None)

        workers = []
        for gpu in self.gpus:
            p = mp.Process(target=self.worker_loop, args=(gpu, video_queue, result_queue, progress_queue))
            p.start()
            workers.append(p)

        # Progress bar
        with tqdm(total=len(self.video_paths), desc="Processing videos", unit="video") as pbar:
            completed = 0
            while completed < len(self.video_paths):
                try:
                    progress_queue.get(timeout=0.1)
                    completed += 1
                    pbar.update(1)
                except:
                    # Check if all workers are done
                    if all(not w.is_alive() for w in workers):
                        break

        for w in workers:
            w.join()

        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        success_count = sum(1 for _, success in results if success)
        fail_count = len(results) - success_count

        self.emit_event(
            "batch_end",
            total=len(results),
            success=success_count,
            failed=fail_count,
        )

        print(f"\n=== Batch Inference Complete ===")
        print(f"Total videos: {len(results)}")
        print(f"Success: {success_count}")
        print(f"Failed: {fail_count}")
        print(f"Run directory: {self.run_dir}")
        print(f"Status file: {self.status_file}")
        print(f"Events log: {self.events_file}")

        return fail_count == 0


def collect_videos(video_dir: Path, extensions=(".mp4", ".avi", ".mov")) -> List[str]:
    videos = []
    for ext in extensions:
        videos.extend(str(p) for p in video_dir.rglob(f"*{ext}"))
    return sorted(videos)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Multi-GPU batch inference scheduler for HaWoR"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video_list",
        type=str,
        help="Path to text file with one video path per line",
    )
    input_group.add_argument(
        "--video_dir",
        type=str,
        help="Directory to recursively search for video files",
    )
    input_group.add_argument(
        "--factory_dir",
        type=str,
        help="WebDataset factory directory containing tar shards",
    )
    input_group.add_argument(
        "--factory_list",
        type=str,
        help="Path to text file with one factory directory per line",
    )
    input_group.add_argument(
        "--factory_range",
        type=str,
        nargs=2,
        metavar=("START", "END"),
        help="Factory ID range (inclusive), e.g. --factory_range 1 10",
    )

    parser.add_argument(
        "--factory_base",
        type=str,
        default="/share_data/guantianrui/datasets/Egocentric-100K/processed_v9_test_jpg",
        help="Base directory for --factory_range (default: processed_v9_test_jpg)",
    )

    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU IDs (e.g., '0,1,2,3')",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="detect_track,motion,slam,infiller",
        help="Comma-separated stage names",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Max retries per stage (default: 2)",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from existing outputs (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing outputs and rerun all stages",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        help="Custom run directory (default: batch_runs/<timestamp>)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./weights/hawor/checkpoints/hawor.ckpt",
        help="Path to HaWoR checkpoint",
    )
    parser.add_argument(
        "--infiller_weight",
        type=str,
        default="./weights/hawor/checkpoints/infiller.pt",
        help="Path to infiller weights",
    )
    parser.add_argument(
        "--img_focal",
        type=float,
        help="Image focal length (optional)",
    )
    parser.add_argument(
        "--chunk_batch_size",
        type=int,
        default=64,
        help="Number of 16-frame chunks processed per forward in HAWOR motion stage",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of DataLoader workers for parallel frame loading in motion stage",
    )
    parser.add_argument(
        "--metric3d_batch_size",
        type=int,
        default=32,
        help="Batch size for Metric3D depth estimation in SLAM stage",
    )
    parser.add_argument(
        "--render_batch_size",
        type=int,
        default=8,
        help="Batch size for rendering phase in motion stage (Phase 3). Higher = faster but more GPU memory. Default: 8",
    )
    parser.add_argument(
        "--infiller_window_batch_size",
        type=int,
        default=64,
        help="Number of infiller windows to batch per forward pass",
    )
    parser.add_argument(
        "--detect_batch_size",
        type=int,
        default=128,
        help="Batch size for YOLO detection in detect_track stage (default: 128)",
    )
    parser.add_argument(
        "--detect_io_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers for parallel frame loading in detect_track stage",
    )
    parser.add_argument(
        "--detect_device",
        type=str,
        default="cuda:0",
        help="Device for YOLO detector in detect_track stage (e.g., cuda:0)",
    )
    parser.add_argument(
        "--detect_half_precision",
        action="store_true",
        default=True,
        help="Use FP16 for YOLO detector (2x faster, default: enabled)",
    )
    parser.add_argument(
        "--no-detect_half_precision",
        dest="detect_half_precision",
        action="store_false",
        help="Disable FP16 for YOLO detector",
    )
    parser.add_argument(
        "--enable_profiler",
        action="store_true",
        help="Enable torch profiler to diagnose performance bottlenecks (generates trace files)",
    )
    parser.add_argument(
        "--rebuild_cam_space_cache",
        action="store_true",
        help="Rebuild cached cam_space tensors before running infiller",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index of video list (inclusive, 0-based)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index of video list (exclusive, None means process all)",
    )
    parser.add_argument(
        "--scheduler_mode",
        type=str,
        default="legacy",
        choices=["legacy", "wave"],
        help="Scheduling mode: legacy (per-video stages) or wave (global stage waves)",
    )
    parser.add_argument(
        "--persistent_worker",
        action="store_true",
        help="Use persistent workers with model reuse (only for wave mode)",
    )
    parser.add_argument(
        "--max_stage_retries",
        type=int,
        default=1,
        help="Max retries per stage wave (wave mode only, default: 1)",
    )

    return parser


def main():
    args = get_parser().parse_args()

    descriptors = None  # Will be set for factory mode

    if args.video_list:
        with open(args.video_list) as f:
            video_paths = [line.strip() for line in f if line.strip()]
    elif args.video_dir:
        video_paths = collect_videos(Path(args.video_dir))
    elif args.factory_dir:
        descs = collect_videos_from_factory(args.factory_dir)
        descriptors = descs
        video_paths = [d.video_key for d in descs]
        print(f"Factory mode: {args.factory_dir}")
        print(f"Discovered {len(descs)} videos from factory")
    elif args.factory_list:
        with open(args.factory_list) as f:
            factory_dirs = [line.strip() for line in f if line.strip()]
        descs = collect_videos_from_factories(factory_dirs)
        descriptors = descs
        video_paths = [d.video_key for d in descs]
        print(f"Factory mode: {len(factory_dirs)} factories")
        print(f"Discovered {len(descs)} videos total")
    elif args.factory_range:
        fstart, fend = int(args.factory_range[0]), int(args.factory_range[1])
        factory_dirs = [
            os.path.join(args.factory_base, f"factory{i:03d}")
            for i in range(fstart, fend + 1)
        ]
        # Filter to only existing directories
        missing = [d for d in factory_dirs if not os.path.isdir(d)]
        if missing:
            print(f"Warning: {len(missing)} factory dirs not found, skipping", file=sys.stderr)
            factory_dirs = [d for d in factory_dirs if os.path.isdir(d)]
        if not factory_dirs:
            print("Error: No valid factory directories found", file=sys.stderr)
            sys.exit(1)
        descs = collect_videos_from_factories(factory_dirs)
        descriptors = descs
        video_paths = [d.video_key for d in descs]
        print(f"Factory range: factory{fstart:03d} ~ factory{fend:03d} ({len(factory_dirs)} factories)")
        print(f"Discovered {len(descs)} videos total")

    if not video_paths:
        print("Error: No videos found", file=sys.stderr)
        sys.exit(1)

    # Apply start-end slicing
    total_videos = len(video_paths)
    start_idx = args.start
    end_idx = args.end if args.end is not None else total_videos

    # Validate indices
    if start_idx < 0 or start_idx >= total_videos:
        print(f"Error: --start {start_idx} is out of range [0, {total_videos})", file=sys.stderr)
        sys.exit(1)
    if end_idx < start_idx or end_idx > total_videos:
        print(f"Error: --end {end_idx} is out of range [{start_idx}, {total_videos}]", file=sys.stderr)
        sys.exit(1)

    video_paths = video_paths[start_idx:end_idx]
    if descriptors is not None:
        descriptors = descriptors[start_idx:end_idx]

    if not video_paths:
        print(f"Error: No videos in range [{start_idx}, {end_idx})", file=sys.stderr)
        sys.exit(1)

    gpus = [int(g.strip()) for g in args.gpus.split(",")]
    # Normalize stage names (support short aliases)
    STAGE_ALIASES = {"detect": "detect_track"}
    stages = [STAGE_ALIASES.get(s.strip(), s.strip()) for s in args.stages.split(",")]

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = PROJECT_ROOT / "batch_runs" / timestamp

    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Batch Inference Configuration ===")
    print(f"Total videos in list: {total_videos}")
    print(f"Processing range: [{start_idx}, {end_idx})")
    print(f"Videos to process: {len(video_paths)}")
    print(f"GPUs: {gpus}")
    print(f"Stages: {stages}")
    print(f"Scheduler mode: {args.scheduler_mode}")
    print(f"Persistent worker: {args.persistent_worker}")
    print(f"Max retries: {args.retries}")
    print(f"Max stage retries (wave): {args.max_stage_retries}")
    print(f"Detect batch size (detect_track): {args.detect_batch_size}")
    print(f"Detect I/O workers: {args.detect_io_workers}")
    print(f"Chunk batch size (motion): {args.chunk_batch_size}")
    print(f"Metric3D batch size (slam): {args.metric3d_batch_size}")
    print(f"Infiller window batch size: {args.infiller_window_batch_size}")
    print(f"Resume: {args.resume}")
    print(f"Run directory: {run_dir}")
    print()

    scheduler = BatchScheduler(
        video_paths=video_paths,
        gpus=gpus,
        stages=stages,
        max_retries=args.retries,
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
        scheduler_mode=args.scheduler_mode,
        persistent_worker=args.persistent_worker,
        max_stage_retries=args.max_stage_retries,
        enable_profiler=args.enable_profiler,
        descriptors=descriptors,
    )

    success = scheduler.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
