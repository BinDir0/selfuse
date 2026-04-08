# Configs

This directory currently contains both examples and active recipe-style configs for the dataset pipeline.

## Naming Convention

Recommended interpretation:

- `*.example.yaml`: templates to copy from for new runs
- recipe-like files such as `dataset_pipeline_buildai_v9_30fps_part1.yaml`: concrete configs used for specific datasets or launches
- `legacy_*`: configs for older layouts that are still supported through compatibility adapters

## Official Config Shape

New configs should prefer the normalized nested shape used by `scripts/run_dataset_pipeline.py`:

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
- `dataset_pipeline_buildai.compact.example.yaml`: compact shorthand form
- `dataset_pipeline_flat_shard.example.yaml`: flat shard adapter example
- `dataset_pipeline_image_sequence.example.yaml`: image-sequence adapter example
- `dataset_pipeline_video_folder.example.yaml`: video-folder adapter example
- `dataset_pipeline_legacy_buildai.example.yaml`: old processed BuildAI layout

## Stability

Existing config paths are intentionally not being renamed yet to avoid breaking launch scripts. The cleanup here is documentation-first: make the intended structure obvious without forcing a path migration immediately.
