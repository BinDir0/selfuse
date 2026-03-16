# HaWoR Batch Inference System

## Overview

The batch inference system enables parallel processing of multiple videos across multiple GPUs with automatic resume, retry logic, and progress tracking.

## Architecture

### Design Principles
- **Video-level parallelism**: Each GPU processes one complete video through all stages
- **Stage-sequential execution**: Within each video, stages run sequentially (detect_track → motion → slam → infiller)
- **Artifact-based resume**: Uses output files to determine completion status
- **Isolated workers**: Each GPU worker operates independently

### Components

1. **`scripts/batch_worker.py`**: Single-stage executor
   - Executes one stage for one video on one GPU
   - Validates stage outputs
   - Handles resume/force semantics
   - Emits structured JSON logs

2. **`scripts/batch_infer.py`**: Multi-GPU scheduler
   - Manages video queue and GPU worker pool
   - Orchestrates stage execution per video
   - Handles retries and failure tracking
   - Persists status and events

## Usage

### Basic Usage

Process videos from a directory:
```bash
python scripts/batch_infer.py \
  --video_dir /path/to/videos \
  --gpus 0,1,2,3,4,5,6,7
```

Process videos from a list file:
```bash
python scripts/batch_infer.py \
  --video_list videos.txt \
  --gpus 0,1,2,3,4,5,6,7
```

### Advanced Options

```bash
python scripts/batch_infer.py \
  --video_list videos.txt \
  --gpus 0,1,2,3 \
  --stages detect_track,motion,slam,infiller \
  --retries 3 \
  --checkpoint ./weights/hawor/checkpoints/hawor.ckpt \
  --infiller_weight ./weights/hawor/checkpoints/infiller.pt \
  --img_focal 500.0
```

### Resume Interrupted Batch

```bash
python scripts/batch_infer.py \
  --video_list videos.txt \
  --gpus 0,1,2,3,4,5,6,7 \
  --run_dir batch_runs/20260301_120000
```

### Force Rerun (Ignore Existing Outputs)

```bash
python scripts/batch_infer.py \
  --video_list videos.txt \
  --gpus 0,1,2,3 \
  --no-resume
```

## Output Structure

```
batch_runs/<timestamp>/
├── status.json          # Current status of all videos and stages
├── events.jsonl         # Event stream (start/success/fail/retry/skip)
└── logs/
    ├── video1_detect_track.log
    ├── video1_motion.log
    ├── video1_slam.log
    ├── video1_infiller.log
    ├── video2_detect_track.log
    └── ...
```

### Status File Format

```json
{
  "run_dir": "batch_runs/20260301_120000",
  "gpus": [0, 1, 2, 3, 4, 5, 6, 7],
  "stages": ["detect_track", "motion", "slam", "infiller"],
  "tasks": {
    "/path/to/video1.mp4": {
      "video_path": "/path/to/video1.mp4",
      "video_name": "video1",
      "stage_status": {
        "detect_track": "completed",
        "motion": "completed",
        "slam": "running",
        "infiller": "pending"
      },
      "retry_count": {
        "detect_track": 0,
        "motion": 0,
        "slam": 1,
        "infiller": 0
      },
      "start_time": "2026-03-01T12:00:00Z",
      "end_time": null
    }
  }
}
```

### Events Log Format

Each line is a JSON object:
```json
{"time": "2026-03-01T12:00:00Z", "event": "batch_start", "total_videos": 20, "gpus": [0,1,2,3,4,5,6,7]}
{"time": "2026-03-01T12:00:01Z", "event": "video_start", "video": "/path/to/video1.mp4", "gpu": 0}
{"time": "2026-03-01T12:00:02Z", "event": "stage_attempt", "video": "/path/to/video1.mp4", "stage": "detect_track", "gpu": 0, "attempt": 0}
{"time": "2026-03-01T12:05:30Z", "event": "stage_success", "video": "/path/to/video1.mp4", "stage": "detect_track", "gpu": 0, "attempt": 0}
{"time": "2026-03-01T12:05:31Z", "event": "stage_skip", "video": "/path/to/video2.mp4", "stage": "detect_track", "gpu": 1}
{"time": "2026-03-01T12:10:00Z", "event": "stage_failure", "video": "/path/to/video3.mp4", "stage": "motion", "gpu": 2, "attempt": 0, "returncode": 1}
{"time": "2026-03-01T12:30:00Z", "event": "video_completed", "video": "/path/to/video1.mp4", "gpu": 0}
{"time": "2026-03-01T12:35:00Z", "event": "video_failed", "video": "/path/to/video3.mp4", "stage": "motion", "gpu": 2}
{"time": "2026-03-01T13:00:00Z", "event": "batch_end", "total": 20, "success": 18, "failed": 2}
```

## Stage Output Validation

Each stage produces specific outputs that are validated before marking as complete:

### detect_track
- `<video_stem>/tracks_<start>_<end>/model_boxes.npy`
- `<video_stem>/tracks_<start>_<end>/model_tracks.npy`
- `<video_stem>/extracted_images/*.jpg`

### motion
- `<video_stem>/tracks_<start>_<end>/frame_chunks_all.npy`
- `<video_stem>/tracks_<start>_<end>/model_masks.npy`

### slam
- `<video_stem>/SLAM/hawor_slam_w_scale_<start>_<end>.npz`
  - Must contain keys: `traj`, `scale`

### infiller
- `<video_stem>/world_space_res.pth`
  - Must contain valid hand pose tensors with correct shapes

## Testing

Run validation tests:
```bash
python scripts/test_batch_inference.py \
  --video_list test_videos.txt \
  --gpus 0,1
```

Tests include:
1. **Smoke test**: Full pipeline on 2-3 short videos
2. **Resume test**: Verify skip behavior on re-run
3. **Partial recovery**: Delete one stage output and verify recovery

## Performance Considerations

### Current Implementation (Phase 1)
- **Throughput**: ~8x speedup with 8 GPUs (video-level parallelism)
- **Model loading**: Models are loaded per stage (subprocess overhead)
- **GPU utilization**: May vary by stage (detect_track is I/O-bound, motion is compute-heavy)

### Expected Performance
- **detect_track**: ~5-10% of total time (I/O-bound)
- **motion**: ~40-50% of total time (compute-heavy, bottleneck)
- **slam**: ~30-40% of total time
- **infiller**: ~10-15% of total time

### Future Optimizations (Phase 2)
- Persistent worker processes with model caching
- Reduced model reload overhead
- Dynamic load balancing
- Stage-aware scheduling

## Troubleshooting

### Check batch status
```bash
cat batch_runs/<timestamp>/status.json | jq
```

### Monitor progress
```bash
tail -f batch_runs/<timestamp>/events.jsonl
```

### Check failed videos
```bash
cat batch_runs/<timestamp>/status.json | jq '.tasks | to_entries[] | select(.value.stage_status | to_entries[] | select(.value == "failed"))'
```

### View stage logs
```bash
cat batch_runs/<timestamp>/logs/video1_motion.log
```

### Common Issues

1. **CUDA out of memory**: Reduce number of concurrent workers or use smaller batch sizes
2. **Stage hangs**: Check individual stage logs for errors
3. **Resume not working**: Verify output files exist and pass validation
4. **GPU not utilized**: Check CUDA_VISIBLE_DEVICES in stage logs

## Architecture Decision: Why Video-Level Parallelism?

### Alternatives Considered

**Stage-level pipeline** (rejected):
- Assign each stage to different GPUs
- Form a pipeline: GPU0→detect_track, GPU1→motion, GPU2→slam, GPU3→infiller
- Theoretical 1.5-1.8x speedup (limited by slowest stage)

**Why video-level parallelism is better**:
1. **Better scaling**: 8 GPUs = 8x throughput (vs 1.5x for pipeline)
2. **Simpler implementation**: No cross-stage synchronization needed
3. **Lower risk**: No modifications to third-party code (DROID-SLAM, Metric3D)
4. **Easier debugging**: Each video is independent
5. **Better load balancing**: No pipeline stalls from stage imbalance

### Scalability

- **4 GPUs**: ~4x speedup
- **8 GPUs**: ~8x speedup
- **16 GPUs**: ~16x speedup (linear scaling)

The system scales linearly with GPU count, making it ideal for large-scale batch processing.
