# Deployment Checklist

Use this checklist when deploying the batch inference system on your production machine.

## Pre-Deployment

- [ ] Machine has NVIDIA GPUs available
- [ ] CUDA drivers installed (`nvidia-smi` works)
- [ ] Conda environment `hawor` created and configured
- [ ] All model weights downloaded:
  - [ ] `weights/external/droid.pth`
  - [ ] `thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth`
  - [ ] `weights/external/detector.pt`
  - [ ] `weights/hawor/checkpoints/hawor.ckpt`
  - [ ] `weights/hawor/checkpoints/infiller.pt`
  - [ ] `weights/hawor/model_config.yaml`
  - [ ] MANO models in `_DATA/data/mano/`

## Testing Phase

- [ ] Copy HaWoR repository to production machine
- [ ] Place 2-3 short test videos in `example/` directory
- [ ] Run one-click test: `bash scripts/setup_and_test_batch.sh`
- [ ] Verify test outputs:
  - [ ] All test videos completed successfully
  - [ ] `world_space_res.pth` exists for each video
  - [ ] Resume test showed skip events
  - [ ] No errors in log files

## Production Deployment

### 1. Prepare Video List

Choose one:

**Option A: Directory of videos**
```bash
# System will recursively find all .mp4/.avi/.mov files
VIDEO_SOURCE="--video_dir /path/to/videos"
```

**Option B: Explicit list**
```bash
# Create list file
find /path/to/videos -name "*.mp4" > video_list.txt
VIDEO_SOURCE="--video_list video_list.txt"
```

- [ ] Video source prepared

### 2. Configure GPUs

```bash
# Check available GPUs
nvidia-smi

# Set GPU list (example for 8 GPUs)
GPUS="0,1,2,3,4,5,6,7"
```

- [ ] GPU configuration determined

### 3. Run Batch Inference

```bash
conda activate hawor

python scripts/batch_infer.py \
  ${VIDEO_SOURCE} \
  --gpus ${GPUS} \
  --retries 2
```

- [ ] Batch inference started
- [ ] Run directory noted: `batch_runs/<timestamp>`

### 4. Monitor Progress

In separate terminal:

```bash
# Watch events in real-time
tail -f batch_runs/<timestamp>/events.jsonl

# Count completed videos
grep "video_completed" batch_runs/<timestamp>/events.jsonl | wc -l

# Check for failures
grep "video_failed" batch_runs/<timestamp>/events.jsonl
```

- [ ] Monitoring set up

## Post-Processing

### 5. Verify Results

```bash
# Check final status
cat batch_runs/<timestamp>/status.json | jq

# Count successes and failures
cat batch_runs/<timestamp>/status.json | jq '.tasks | to_entries | map(.value.stage_status.infiller) | group_by(.) | map({status: .[0], count: length})'

# List failed videos
cat batch_runs/<timestamp>/status.json | jq -r '.tasks | to_entries[] | select(.value.stage_status | to_entries[] | select(.value == "failed")) | .key'
```

- [ ] Success/failure counts verified
- [ ] Failed videos identified (if any)

### 6. Handle Failures (if any)

For failed videos:

```bash
# Check specific video logs
cat batch_runs/<timestamp>/logs/<video_name>_<stage>.log

# Retry failed videos only
# Create list of failed videos
cat batch_runs/<timestamp>/status.json | jq -r '.tasks | to_entries[] | select(.value.stage_status | to_entries[] | select(.value == "failed")) | .key' > failed_videos.txt

# Re-run with increased retries
python scripts/batch_infer.py \
  --video_list failed_videos.txt \
  --gpus ${GPUS} \
  --retries 5
```

- [ ] Failures investigated
- [ ] Retry strategy executed (if needed)

### 7. Validate Outputs

```bash
# Check output files exist
for video in $(cat video_list.txt); do
  video_stem=$(basename "$video" .mp4)
  video_dir=$(dirname "$video")
  output="${video_dir}/${video_stem}/world_space_res.pth"
  if [ -f "$output" ]; then
    echo "✓ $video_stem"
  else
    echo "✗ $video_stem - MISSING OUTPUT"
  fi
done
```

- [ ] All expected outputs exist
- [ ] Output file sizes reasonable (not 0 bytes)

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce number of GPUs: `--gpus 0,1,2,3` instead of all 8
- Check GPU memory: `nvidia-smi`
- Kill other processes using GPUs

**Stage Hangs**
- Check stage logs: `batch_runs/<timestamp>/logs/`
- Look for DROID-SLAM or Metric3D errors
- Verify model weights are correct

**Resume Not Working**
- Check if output files exist and are valid
- Try `--no-resume` to force rerun
- Check disk space

**Slow Progress**
- Verify all GPUs are being used: `watch -n 1 nvidia-smi`
- Check if videos are very long (will take proportionally longer)
- Monitor CPU usage (I/O bottleneck in detect_track stage)

## Performance Benchmarking

Record performance metrics:

```bash
# Total videos
TOTAL=$(cat video_list.txt | wc -l)

# Start time (from events.jsonl)
START=$(head -1 batch_runs/<timestamp>/events.jsonl | jq -r .time)

# End time
END=$(tail -1 batch_runs/<timestamp>/events.jsonl | jq -r .time)

# Calculate throughput
# (videos per hour)
```

- [ ] Start time: _______________
- [ ] End time: _______________
- [ ] Total videos: _______________
- [ ] Success count: _______________
- [ ] Throughput: _______________ videos/hour

## Cleanup (Optional)

```bash
# Archive batch run logs
tar -czf batch_runs_<timestamp>.tar.gz batch_runs/<timestamp>

# Remove intermediate files (keep only final outputs)
# WARNING: Only do this if you're sure you don't need to resume
# rm -rf batch_runs/<timestamp>
```

- [ ] Logs archived
- [ ] Cleanup completed (if desired)

## Sign-Off

- [ ] All videos processed successfully
- [ ] Outputs validated
- [ ] Performance metrics recorded
- [ ] Documentation updated with any issues/solutions
- [ ] System ready for next batch

**Deployment Date**: _______________
**Operator**: _______________
**Notes**: _______________
