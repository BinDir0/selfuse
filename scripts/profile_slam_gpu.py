#!/usr/bin/env python3
"""Profile SLAM stage GPU utilization with fine-grained timing.

Usage:
    python scripts/profile_slam_gpu.py --video_path /path/to/video.mp4

Runs the full SLAM stage on a single video and outputs:
  - Per-substep wall-clock timing
  - Per-substep GPU kernel timing (via CUDA events)
  - Background nvidia-smi GPU utilization sampling (every 0.5s)
  - Summary statistics

Requires: a video that has already completed detect_track stage
(i.e., tracks_*/ and model_masks.npy exist).
"""
import argparse
import os
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch


# ---- GPU Utilization Sampler (background thread) ----

class GPUSampler:
    """Sample GPU utilization via nvidia-smi in a background thread."""

    def __init__(self, gpu_id=0, interval=0.5):
        self.gpu_id = gpu_id
        self.interval = interval
        self.samples = []  # list of (timestamp, gpu_util%, mem_util%, mem_used_mb)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    ['nvidia-smi',
                     f'--id={self.gpu_id}',
                     '--query-gpu=utilization.gpu,utilization.memory,memory.used',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(',')
                    gpu_util = float(parts[0].strip())
                    mem_util = float(parts[1].strip())
                    mem_used = float(parts[2].strip())
                    self.samples.append((time.time(), gpu_util, mem_util, mem_used))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def summary(self):
        if not self.samples:
            return {}
        utils = [s[1] for s in self.samples]
        mem_utils = [s[2] for s in self.samples]
        mem_used = [s[3] for s in self.samples]
        return {
            'n_samples': len(self.samples),
            'gpu_util_mean': np.mean(utils),
            'gpu_util_median': np.median(utils),
            'gpu_util_min': np.min(utils),
            'gpu_util_max': np.max(utils),
            'mem_util_mean': np.mean(mem_utils),
            'mem_used_mean_mb': np.mean(mem_used),
            'mem_used_max_mb': np.max(mem_used),
        }

    def per_phase_summary(self, phase_times):
        """Compute GPU util for each phase given {name: (start_time, end_time)}."""
        results = {}
        for name, (t_start, t_end) in phase_times.items():
            phase_samples = [s[1] for s in self.samples if t_start <= s[0] <= t_end]
            if phase_samples:
                results[name] = {
                    'gpu_util_mean': np.mean(phase_samples),
                    'n_samples': len(phase_samples),
                }
            else:
                results[name] = {'gpu_util_mean': float('nan'), 'n_samples': 0}
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--img_focal", type=float, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--metric3d_batch_size", type=int, default=32)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda')

    # Import SLAM components
    from lib.pipeline.masked_droid_slam import run_slam, est_calib
    from lib.pipeline.est_scale import est_scale_hybrid, est_scale_hybrid_batch
    from lib.pipeline.frame_source import build_frame_source

    sys.path.insert(0, str(PROJECT_ROOT / 'thirdparty' / 'Metric3D'))
    from metric import Metric3D

    # Setup paths
    file = args.video_path
    video_root = os.path.dirname(file)
    video = os.path.basename(file).split('.')[0]
    seq_folder = os.path.join(video_root, video)
    frame_source = build_frame_source(file)

    # Detect track range
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    start_idx, end_idx, _, _ = detect_track_video(args)

    # Load masks
    masks = np.load(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_masks.npy', allow_pickle=True)
    masks = torch.from_numpy(masks)

    # Camera calibration
    focal = args.img_focal
    if focal is None:
        try:
            with open(os.path.join(seq_folder, 'est_focal.txt'), 'r') as f:
                focal = float(f.read())
        except Exception:
            focal = 600.0
    calib = np.array(est_calib(frame_source))
    calib[:2] = focal

    # Start GPU sampler
    sampler = GPUSampler(gpu_id=args.gpu_id)
    sampler.start()
    phase_times = {}

    print(f"\n{'='*70}")
    print(f"GPU PROFILING: {os.path.basename(args.video_path)}")
    print(f"{'='*70}\n")

    # ---- Phase 1: DROID-SLAM tracking + backend ----
    torch.cuda.synchronize()
    t0 = time.time()
    droid, traj = run_slam(frame_source, masks=masks, calib=calib)
    torch.cuda.synchronize()
    t1 = time.time()
    phase_times['droid_slam'] = (t0, t1)

    n = droid.video.counter.value
    tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
    disps = droid.video.disps_up.cpu().numpy()[:n]
    del droid
    torch.cuda.empty_cache()

    print(f"  DROID-SLAM:      {t1-t0:7.2f}s  ({n} keyframes)")

    # ---- Phase 2: Metric3D depth prediction ----
    from scripts.scripts_test_video.hawor_slam import get_dimention
    from concurrent.futures import ThreadPoolExecutor
    import cv2

    torch.cuda.synchronize()
    t0 = time.time()
    metric3d_weight = str(PROJECT_ROOT / 'thirdparty' / 'Metric3D' / 'weights' / 'metric_depth_vit_large_800k.pth')
    metric = Metric3D(metric3d_weight)
    H, W = get_dimention(frame_source)
    pred_depths = []
    num_frames = len(tstamp)
    bs = args.metric3d_batch_size

    with ThreadPoolExecutor(max_workers=8) as loader:
        for batch_start in range(0, num_frames, bs):
            batch_end = min(batch_start + bs, num_frames)
            batch_indices = tstamp[batch_start:batch_end]
            batch_frames = list(loader.map(
                lambda t: frame_source.get_frame(int(t), rgb=True), batch_indices
            ))
            batch_depths = metric.batch_inference(batch_frames, calib)
            for d in batch_depths:
                pred_depths.append(cv2.resize(d, (W, H)))

    torch.cuda.synchronize()
    t1 = time.time()
    phase_times['metric3d'] = (t0, t1)
    print(f"  Metric3D:        {t1-t0:7.2f}s  ({num_frames} frames, bs={bs})")

    # ---- Phase 3: Scale estimation ----
    import math

    torch.cuda.synchronize()
    t0 = time.time()
    slam_depth_list = [1.0 / disps[i] for i in range(num_frames)]
    mask_list = [masks[tstamp[i]].numpy().astype(np.uint8) for i in range(num_frames)]

    scales = est_scale_hybrid_batch(
        slam_depth_list, pred_depths, sigma=0.5,
        masks=mask_list, near_thresh=0.4, far_thresh=0.7)

    for i in range(num_frames):
        if math.isnan(scales[i]):
            nt, ft = 0.4, 0.7
            while math.isnan(scales[i]):
                nt -= 0.1; ft += 0.1
                scales[i] = est_scale_hybrid(
                    slam_depth_list[i], pred_depths[i], sigma=0.5,
                    msk=mask_list[i], near_thresh=nt, far_thresh=ft)

    torch.cuda.synchronize()
    t1 = time.time()
    phase_times['scale_est'] = (t0, t1)
    print(f"  Scale estimation:{t1-t0:7.2f}s  ({num_frames} keyframes)")

    # ---- Stop sampler and report ----
    sampler.stop()

    total = sum(t[1] - t[0] for t in phase_times.values())
    print(f"\n  TOTAL:           {total:7.2f}s")

    print(f"\n{'='*70}")
    print("GPU Utilization (nvidia-smi sampling)")
    print(f"{'='*70}")

    overall = sampler.summary()
    print(f"  Overall:  mean={overall.get('gpu_util_mean', 0):.1f}%  "
          f"median={overall.get('gpu_util_median', 0):.1f}%  "
          f"min={overall.get('gpu_util_min', 0):.1f}%  "
          f"max={overall.get('gpu_util_max', 0):.1f}%  "
          f"({overall.get('n_samples', 0)} samples)")
    print(f"  Memory:   mean={overall.get('mem_used_mean_mb', 0):.0f}MB  "
          f"max={overall.get('mem_used_max_mb', 0):.0f}MB")

    per_phase = sampler.per_phase_summary(phase_times)
    print(f"\n  Per-phase GPU util:")
    for name in ['droid_slam', 'metric3d', 'scale_est']:
        info = per_phase.get(name, {})
        wall = phase_times[name][1] - phase_times[name][0]
        pct = wall / total * 100
        print(f"    {name:20s}: {info.get('gpu_util_mean', 0):5.1f}% avg  "
              f"({wall:.1f}s = {pct:.0f}% of total, {info.get('n_samples', 0)} samples)")

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
