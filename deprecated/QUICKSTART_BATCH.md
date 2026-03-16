# Quick Start: Batch Inference Testing

## Prerequisites

1. Machine with NVIDIA GPUs and CUDA installed
2. HaWoR conda environment set up (see main README.md)
3. Model weights downloaded
4. 2-3 short test videos in `example/` directory

## One-Click Test

Run the automated setup and test script:

```bash
cd /path/to/HaWoR
bash scripts/setup_and_test_batch.sh
```

This script will:
1. ✓ Verify conda environment exists
2. ✓ Activate environment and check dependencies
3. ✓ Find test videos in `example/` directory
4. ✓ Run full pipeline on test videos (smoke test)
5. ✓ Test resume functionality
6. ✓ Verify all outputs are generated

## Manual Testing

If you prefer to test manually:

### 1. Prepare test video list

```bash
# Create a list of test videos
ls -1 $PWD/example/*.mp4 | head -3 > test_videos.txt
```

### 2. Run batch inference

```bash
# Activate environment
conda activate hawor

# Run on 2 GPUs
python scripts/batch_infer.py \
  --video_list test_videos.txt \
  --gpus 0,1 \
  --retries 1
```

### 3. Check results

```bash
# View status
cat batch_runs/*/status.json | jq

# View events
cat batch_runs/*/events.jsonl | tail -20

# Check outputs exist
for video in $(cat test_videos.txt); do
  video_stem=$(basename "$video" .mp4)
  video_dir=$(dirname "$video")
  ls -lh "${video_dir}/${video_stem}/world_space_res.pth"
done
```

### 4. Test resume

```bash
# Re-run with same run_dir (should skip completed stages)
python scripts/batch_infer.py \
  --video_list test_videos.txt \
  --gpus 0,1 \
  --run_dir batch_runs/<timestamp> \
  --resume

# Check for skip events
grep "stage_skip" batch_runs/*/events.jsonl
```

## Production Usage

Once testing is successful, run on your full dataset:

```bash
# Process all videos in a directory using 8 GPUs
python scripts/batch_infer.py \
  --video_dir /path/to/your/videos \
  --gpus 0,1,2,3,4,5,6,7 \
  --retries 2
```

## Troubleshooting

### Check GPU availability
```bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

### View failed video logs
```bash
# Find failed videos
cat batch_runs/*/status.json | jq '.tasks | to_entries[] | select(.value.stage_status | to_entries[] | select(.value == "failed"))'

# View specific stage log
cat batch_runs/*/logs/video_name_stage.log
```

### Monitor progress in real-time
```bash
# Watch events
tail -f batch_runs/*/events.jsonl

# Count completed videos
grep "video_completed" batch_runs/*/events.jsonl | wc -l
```

## Expected Output Structure

After successful completion, each video will have:

```
<video_dir>/<video_stem>/
├── extracted_images/           # detect_track stage
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── tracks_<start>_<end>/       # detect_track + motion stages
│   ├── model_boxes.npy
│   ├── model_tracks.npy
│   ├── frame_chunks_all.npy
│   └── model_masks.npy
├── SLAM/                       # slam stage
│   └── hawor_slam_w_scale_<start>_<end>.npz
└── world_space_res.pth         # infiller stage (final output)
```

## Performance Expectations

With 8× A100 GPUs processing typical egocentric hand videos:

- **Single video**: ~5-10 minutes (varies by length)
- **8 videos in parallel**: ~5-10 minutes total (near-linear scaling)
- **100 videos**: ~60-120 minutes (depending on video lengths)

Stage breakdown (approximate):
- detect_track: 5-10% of time
- motion: 40-50% of time (bottleneck)
- slam: 30-40% of time
- infiller: 10-15% of time

## Next Steps

See `docs/BATCH_INFERENCE.md` for detailed documentation on:
- Architecture and design decisions
- Advanced configuration options
- Output format specifications
- Performance optimization tips
