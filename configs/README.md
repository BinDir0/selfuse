# Configs

This directory contains active recipe-style configs and reusable examples for the official dataset pipeline schema.

## Naming Convention

Recommended interpretation:

- `*.example.yaml`: templates to copy from for new runs
- recipe-like files such as `dataset_pipeline_buildai_v9_30fps_part1.yaml`: concrete configs used for specific datasets or launches
- `legacy_*`: configs for legacy dataset adapters, still expressed in the official nested pipeline schema

## Official Config Shape

All configs consumed by `scripts/run_dataset_pipeline.py` must use the nested shape below:

```yaml
dataset:
  adapter: buildai

paths:
  shard_root: /path/to/source_shards
  annotation_root: /path/to/annotations
  final_dataset_root: /path/to/output_dataset
  log_root: /path/to/pipeline_runs

runtimes:
  hawor_python: /path/to/hawor/bin/python
  slam_python: /path/to/slam/bin/python

infer:
  common: {}
  detect_motion: {}
  slam: {}
  infiller: {}

filter: {}
build: {}
validation: {}
```

## Current Examples

- `dataset_pipeline_buildai.example.yaml`: standard BuildAI-style config
- `dataset_pipeline_fpha.yaml`: ready-to-run FPHA config
- `dataset_pipeline_flat_shard.example.yaml`: flat shard adapter example
- `dataset_pipeline_fpha_tar.example.yaml`: FPHA sequence-per-tar adapter example
- `dataset_pipeline_image_sequence.example.yaml`: image-sequence adapter example
- `dataset_pipeline_video_folder.example.yaml`: video-folder adapter example
- `dataset_pipeline_legacy_buildai.example.yaml`: old processed BuildAI layout

## Stability

Compact top-level pipeline configs are no longer supported. Breaking changes to the pipeline schema should be made by updating the official nested shape and the first-party callers together.

## Throughput Knobs

Useful safe-throughput keys now supported under `infer.common`:

- `infer_profile: throughput_80gb`
- `local_cache_root: /DATA/.../hawor_local_cache`
- `local_cache_quota_gb: 2000`
- `local_cache_mode: all`
- `local_cache_min_frames: 96`

These only change scheduling/cache defaults and do not change model or SLAM algorithm settings.
