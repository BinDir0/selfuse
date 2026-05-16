# Dataset Pipeline

This repository has one official dataset-production entrypoint:

```bash
python scripts/run_dataset_pipeline.py \
  --config <config.yaml>
```

The preferred config is now a single-video config. The first release treats one input video as one clip, keeps the native FPS, skips language annotation unless `annotation.command` is configured, and exports trainable WebDataset samples with image, lowdim, MANO, meta, and depth payloads.

Minimal config:

```yaml
video: /path/to/input.mp4
```

Optional output root:

```yaml
video: /path/to/input.mp4
output_root: /path/to/input.hawor_pipeline
```

If `output_root` is omitted, outputs go under `<video_dir>/<video_stem>.hawor_pipeline/`:

- `frames/`: extracted native-FPS RGB frames
- `stage_outputs/`: HaWoR/SLAM/infiller outputs
- `runs/run/`: logs, run state, reports
- `webdataset/`: final trainable WebDataset shards

The pipeline remains adapter-driven internally. Different source datasets are normalized into the same prepared clip state, and then the same annotation, inference, filter, build, and validation logic runs on top of that shared boundary.

## Official Stages

- `prepare`: source-specific preprocessing plus prepared clip state creation
- `annotate`: clip-level language sidecars
- `infer`: `detect_track`, `motion`, `slam`, and `infiller`
- `filter`: build-equivalent clip filtering before export
- `build`: final WebDataset export
- `validate`: source and dataset checks

Recommended full run:

```bash
python scripts/run_dataset_pipeline.py \
  --config configs/dataset_pipeline_single_video.example.yaml
```

Default stages are `prepare,infer,filter,build,validate`. `annotate` is inserted automatically only when `annotation.command` is configured. `--stages` can still be used for debugging or resume runs.

Useful partial runs:

```bash
python scripts/run_dataset_pipeline.py --config configs/dataset_pipeline_buildai.example.yaml --stages prepare
python scripts/run_dataset_pipeline.py --config configs/dataset_pipeline_buildai.example.yaml --stages annotate
python scripts/run_dataset_pipeline.py --config configs/dataset_pipeline_buildai.example.yaml --stages infer
python scripts/run_dataset_pipeline.py --config configs/dataset_pipeline_buildai.example.yaml --stages filter,build,validate
```

Legacy stage names such as `preprocess`, `manifest`, `detect_motion`, `slam`, and `infiller` are still accepted for compatibility, but they are deprecated. Prefer the official stage names above.

## Config Shape

Preferred first-party configs use the simplified single-video layout:

```yaml
video: /path/to/input.mp4
output_root: /optional/output_root

# Optional advanced annotation hook. Without this, instruction/language fields
# are allowed to be empty.
annotation:
  command: >
    echo "Read {prepared_state} and write annotations to {annotation_root}"
```

Nested adapter configs are still supported as a migration path for BuildAI, HOT3D, FPHA, and other existing datasets:

```yaml
dataset:
  adapter: buildai
  source_id: buildai
  split: train

paths:
  shard_root: /path/to/source_shards
  annotation_root: /path/to/annotations
  final_dataset_root: /path/to/final_dataset
  log_root: /path/to/pipeline_runs

infer:
  common: {}
  detect_motion: {}
  slam: {}
  infiller: {}

annotation:
  command: >
    echo "Read {prepared_state} and write annotations to {annotation_root}"

filter: {}
build: {}
validation: {}
```

Notes:

- `infer:` is the preferred block for HaWoR stage execution.
- simplified single-video configs do not take runtime paths; the orchestrator resolves conda envs named `hawor` and `any4d`.
- nested configs with explicit `runtimes.hawor_python` and `runtimes.slam_python` remain supported with a migration warning.
- `annotation.command` receives `{prepared_state}`, `{active_prepared_state}`, `{annotation_root}`, `{run_dir}`, `{hawor_python}`, `{slam_python}`, and `{project_root}`. `{manifest}` and `{active_manifest}` remain available for older annotation commands.
- for BuildAI-like layouts, `paths.shard_root` and `paths.seq_folder_root` may refer to different trees.

See [configs/README.md](/root/.openclaw/workspace/projects/hawor_original/HaWoR/configs/README.md) for the current config inventory.

## Runtime Separation

The simplified config resolves two repo-level conda environments automatically:

- `hawor`: `detect_track`, `motion`, `infiller`, filter, build, and validation
- `any4d`: `slam`

Resolution checks `conda`, `mamba`, and `micromamba` env lists and fails fast if either env is missing. Nested configs may still provide explicit runtime paths.

## Internal Prepared State

The prepared clip state is an intentional internal boundary:

- it freezes the clip list before expensive stage work
- it decouples source layout from downstream logic
- it lets annotation, filtering, and rebuilding operate on the same stable clip set
- it enables advanced reruns without rescanning the source

Normal users should not need to operate on these files manually; the orchestrator owns that path.

## Source Adapters

Built-in adapters:

- `buildai`
- `flat_shard`
- `fpha_tar`
- `image_sequence`
- `legacy_buildai`
- `single_video`
- `video_folder`

For a new dataset:

1. Add or extend an adapter under `lib/pipeline/datasets/`.
2. Make it emit `ClipDescriptor` objects.
3. Point a standard config at that adapter.
4. Use `scripts/run_dataset_pipeline.py` as the entrypoint.

## Legacy BuildAI Layout

The old processed layout is supported through the `legacy_buildai` adapter.

Use:

```bash
python scripts/run_dataset_pipeline.py \
  --config configs/dataset_pipeline_legacy_buildai.example.yaml \
  --stages prepare,build,validate
```

Typical settings for the legacy BuildAI path:

- `build.source_fps: 5.0`
- `build.target_fps: 30.0`
- `build.interpolate_labels: true`

This uses 30 FPS RGB descriptors and resamples 5 FPS stage outputs onto that timeline during final build.

## Advanced Manual Workflow

These manifest-level scripts remain supported for explicit split workflows, debugging, and reruns. They are not the normal single-video path.

Build a frozen manifest:

```bash
python scripts/build_clip_manifest.py \
  --config configs/dataset_pipeline_buildai.example.yaml \
  --manifest_out /path/to/run/clip_manifest.jsonl \
  --shard_dirs_out /path/to/run/shard_dirs.txt
```

Run inference from an existing manifest:

```bash
python scripts/batch_infer.py \
  --descriptor_manifest /path/to/run/clip_manifest.jsonl \
  --stages detect_track,motion,slam,infiller \
  --gpus 0,1,2,3
```

Filter a manifest with build-equivalent rules:

```bash
python scripts/filter_manifest_by_quality.py \
  --input_manifest /path/to/run/clip_manifest.jsonl \
  --output_manifest /path/to/run/clip_manifest.filtered.jsonl \
  --report_out /path/to/run/filter_report.json
```

Filter an already-built WebDataset with the same hard quality rules:

```bash
python scripts/filter_webdataset.py \
  --source_shard_dir /path/to/wds \
  --output_dir /path/to/wds.filtered \
  --report_out /path/to/wds_filter_report.json
```

Analyze an already-built WebDataset:

```bash
python scripts/sanity_check_webdataset.py \
  --source_shard_dir /path/to/wds \
  --report_out /path/to/wds_sanity_report.json
```

Hard rules are always enabled in both filter paths: any `NaN/Inf` lowdim frame or any missing/empty/mismatched instruction frame drops the whole episode. Optional outlier checks can be disabled with `--no-outlier_checks`.

Build the final dataset directly from a manifest:

```bash
python scripts/build_vla_from_manifest.py \
  --descriptor_manifest /path/to/run/clip_manifest.filtered.jsonl \
  --annotation_root /path/to/annotations \
  --output_dir /path/to/final_dataset \
  --require_annotation \
  --frames_per_shard 10000 \
  --source_fps 5.0 \
  --target_fps 30.0 \
  --interpolate_labels \
  --export_depth
```

## Final Dataset Schema

Each final WebDataset sample contains:

- `*.image.jpg`: RGB frame bytes
- `*.lowdim.npy`: `float32[116]`
- `*.mano.npy`: MANO pose/shape payload used by the viewer
- `*.meta.json`: frame metadata such as `clip_id`, `instruction`, `instruction_num`, `language`, and `presence`
- `*.depth.npy`: optional metric depth payload, exported by default in the single-video pipeline

`lowdim[116]` layout:

- `0:3` left wrist position in world coordinates
- `3:6` right wrist position in world coordinates
- `6:12` left root orientation as rot6d
- `12:18` right root orientation as rot6d
- `18:33` left fingertip positions
- `33:48` right fingertip positions
- `48:66` next-frame wrist state
- `66:96` next-frame fingertip state
- `96:112` flattened `w2c` camera extrinsic
- `112:116` camera intrinsic `[fx, fy, cx, cy]`

Important conventions:

- wrist translation uses MANO wrist joint world coordinates, not raw `pred_trans`
- fingertip coordinates are stored in the HaWoR/SLAM world frame
- camera extrinsic is `World2Cam`

If stage outputs were generated at a lower FPS than the descriptor frames, set:

- `build.source_fps`
- `build.target_fps`
- `build.interpolate_labels`

This resamples the stage outputs onto the descriptor frame timeline during final build.

## Validation

Validate a completed run:

```bash
python scripts/run_dataset_pipeline.py \
  --config configs/dataset_pipeline_buildai.example.yaml \
  --stages validate
```

Or directly:

```bash
python scripts/validate_pipeline_run.py \
  --descriptor_manifest /path/to/run/clip_manifest.filtered.jsonl \
  --annotation_root /path/to/annotations \
  --dataset_dir /path/to/final_dataset \
  --max_clips 200 \
  --dataset_sample_checks 20
```

Recommended smoke pass before a large production launch:

1. Run a small representative subset.
2. Complete `prepare` through `validate`.
3. Inspect multiple output samples across different shards.
4. Confirm image, lowdim, MANO, camera, and depth stay aligned.

## Inspecting Built Samples

For frame-level inspection of final WebDataset shards, prefer the Rerun viewer:

```bash
python tools/ops/rerun_webdataset_visualizer.py \
  --input /path/to/final_dataset/train \
  --render-mode keypoint
```

Offline `.rrd` export:

```bash
python tools/ops/rerun_webdataset_visualizer.py \
  --input /path/to/final_dataset/train \
  --render-mode mesh \
  --output-mode offline \
  --rrd-out /path/to/inspect_mesh.rrd
```

The older `tools/ops/webdataset_visualizer.py` viewer is still available as a fallback, but it is no longer the preferred inspection path.

## Related Docs

- [README.md](/root/.openclaw/workspace/projects/hawor_original/HaWoR/README.md)
- [docs/repo_map.md](/root/.openclaw/workspace/projects/hawor_original/HaWoR/docs/repo_map.md)
- [docs/legacy_tools.md](/root/.openclaw/workspace/projects/hawor_original/HaWoR/docs/legacy_tools.md)
