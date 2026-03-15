# Batch Inference System - Quick Reference

## 🚀 One-Click Test (Recommended)

On a machine with GPUs and conda environment:

```bash
bash scripts/setup_and_test_batch.sh
```

This will automatically verify everything and run tests.

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **[QUICKSTART_BATCH.md](docs/QUICKSTART_BATCH.md)** | Quick start guide for testing |
| **[BATCH_INFERENCE.md](docs/BATCH_INFERENCE.md)** | Complete technical documentation |
| **[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)** | Step-by-step deployment guide |
| **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** | What was implemented and why |

## 🎯 Quick Commands

### Test with sample videos
```bash
bash scripts/setup_and_test_batch.sh
```

### Process videos from directory (8 GPUs)
```bash
python scripts/batch_infer.py \
  --video_dir /path/to/videos \
  --gpus 0,1,2,3,4,5,6,7
```

### Process videos from list file
```bash
python scripts/batch_infer.py \
  --video_list videos.txt \
  --gpus 0,1,2,3,4,5,6,7
```

### Resume interrupted batch
```bash
python scripts/batch_infer.py \
  --video_list videos.txt \
  --gpus 0,1,2,3,4,5,6,7 \
  --run_dir batch_runs/20260301_120000
```

## 📁 Key Files

### Scripts
- `scripts/batch_infer.py` - Main batch scheduler
- `scripts/batch_worker.py` - Single-stage executor
- `scripts/setup_and_test_batch.sh` - One-click test script
- `scripts/test_batch_inference.py` - Validation test suite

### Documentation
- `docs/QUICKSTART_BATCH.md` - Start here for testing
- `docs/BATCH_INFERENCE.md` - Full technical docs
- `DEPLOYMENT_CHECKLIST.md` - Production deployment guide
- `IMPLEMENTATION_SUMMARY.md` - Implementation details

## ✅ Features

- ✓ Multi-GPU parallel processing (1 video per GPU)
- ✓ Automatic resume from checkpoints
- ✓ Per-stage retry logic
- ✓ Structured logging and progress tracking
- ✓ Failure isolation (one video failure doesn't block others)
- ✓ Output validation for each stage
- ✓ Support for video list or directory input

## 📊 Expected Performance

With 8× A100 GPUs:
- **Throughput**: ~8x speedup vs single GPU
- **Single video**: ~5-10 minutes
- **100 videos**: ~60-120 minutes

## 🔍 Monitor Progress

```bash
# Watch events in real-time
tail -f batch_runs/<timestamp>/events.jsonl

# Count completed videos
grep "video_completed" batch_runs/<timestamp>/events.jsonl | wc -l

# Check status
cat batch_runs/<timestamp>/status.json | jq
```

## 🐛 Troubleshooting

See [BATCH_INFERENCE.md](docs/BATCH_INFERENCE.md#troubleshooting) for detailed troubleshooting guide.

Quick checks:
```bash
# Check GPU availability
nvidia-smi

# View failed video logs
cat batch_runs/<timestamp>/logs/<video_name>_<stage>.log

# List failed videos
cat batch_runs/<timestamp>/status.json | jq -r '.tasks | to_entries[] | select(.value.stage_status | to_entries[] | select(.value == "failed")) | .key'
```

## 📞 Support

For issues or questions:
1. Check [BATCH_INFERENCE.md](docs/BATCH_INFERENCE.md) troubleshooting section
2. Review log files in `batch_runs/<timestamp>/logs/`
3. Check GitHub issues

## 🎓 Learn More

- **Architecture**: See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for design decisions
- **API Reference**: See [BATCH_INFERENCE.md](docs/BATCH_INFERENCE.md) for all options
- **Deployment**: Follow [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) step-by-step
