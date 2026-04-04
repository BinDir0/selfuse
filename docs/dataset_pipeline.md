# Dataset Pipeline

This repository has one official dataset-production entrypoint:

```bash
python scripts/run_dataset_pipeline.py --config <config.yaml> --stages prepare,annotate,infer,filter,build,validate
```

The pipeline is adapter-driven. Different source datasets are normalized into a frozen clip manifest internally, then the same HaWoR stages, filtering, build, and validation logic run on top of that shared boundary.

## Official Workflow

Use these official stages:

- `prepare`: source-specific preprocess plus frozen manifest creation
- `annotate`: clip-level language sidecars
- `infer`: `detect_track`, `motion`, `slam`, and `infiller`
- `filter`: clip-level quality filtering before final build
- `build`: final WebDataset export
- `validate`: manifest outputs and final dataset checks

Recommended full run:

```bash
python scripts/run_dataset_pipeline.py \
  --config configs/dataset_pipeline_buildai.example.yaml \
  --stages prepare,annotate,infer,filter,build,validate
```

Useful partial runs:

```bash
# Prepare only
python scripts/run_dataset_pipeline.py \
  --config configs/dataset_pipeline_buildai.example.yaml \
  --stages prepare

# Annotation can run as soon as prepare finishes
python scripts/run_dataset_pipeline.py \
  --config configs/dataset_pipeline_buildai.example.yaml \
  --stages annotate

# Run HaWoR inference only
python scripts/run_dataset_pipeline.py \
  --config configs/dataset_pipeline_buildai.example.yaml \
  --stages infer

# Rebuild from existing stage outputs
python scripts/run_dataset_pipeline.py \
  --config configs/dataset_pipeline_buildai.example.yaml \
  --stages build,validate
```

Legacy stage names such as `preprocess`, `manifest`, `detect_motion`, `slam`, and `infiller` are still accepted for compatibility, but they are deprecated. Prefer `prepare` and `infer`.

## Standard Config Shape

For new configs, prefer the standard nested shape:

```yaml
dataset:
  adapter: buildai
  source_id: buildai
  split: train
  start_factory_id: 1
  end_factory_id: 2

paths:
  shard_root: /path/to/shards
  annotation_root: /path/to/annotations
  final_dataset_root: /path/to/final_dataset
  log_root: /path/to/pipeline_runs

runtimes:
  hawor_python: /path/to/hawor/bin/python
  slam_python: /path/to/any4/bin/python

infer:
  common:
    gpus: 0,1,2,3
    workers_per_gpu: 1
    resume: true
    checkpoint: /path/to/hawor.ckpt
    infiller_weight: /path/to/infiller.pt
  detect_motion:
    chunk_batch_size: 64
  slam:
    any4d_batch_size: 32
  infiller:
    infiller_window_batch_size: 64

annotation:
  command: >
    echo "Read {manifest} and write {annotation_root}/<clip_id>.annotation.json"

filter:
  stages: detect_track,motion,slam,infiller
  workers: 8
  drop_nonfinite_world_res: true
  drop_nonfinite_slam: true

build:
  require_annotation: true
  preprocess_workers: 8
  writer_workers: 4
  frames_per_shard: 10000
  repeat_episodes: 1
  mano_device: cuda:0
  source_fps: 5.0
  target_fps: 30.0
  interpolate_labels: true

validation:
  max_clips: 200
  dataset_sample_checks: 20
```

Notes:

- `infer:` is the preferred config block for HaWoR stage execution.
- Old `batch_infer:` configs and compact top-level shorthands are still supported.
- `annotation.command` receives `{manifest}`, `{active_manifest}`, `{annotation_root}`, `{run_dir}`, `{hawor_python}`, `{slam_python}`, and `{project_root}`.
- For BuildAI-like datasets, `paths.shard_root` and `paths.seq_folder_root` may point to different roots. This supports cases where 30 FPS RGB shards and 5 FPS stage outputs live in separate directory trees.

## Runtime Separation

The pipeline assumes two Python runtimes:

- `hawor_python` for `detect_track`, `motion`, `infiller`, build, and validation
- `slam_python` for `slam`

Do not merge them unless you have already unified the environments yourself. The orchestrator dispatches each stage with the interpreter configured under `runtimes`.

## Annotation Sidecars

Annotation is stored as:

```text
<annotation_root>/<clip_id>.annotation.json
```

Minimum recommended fields:

```json
{
  "clip_id": "factory023_worker001_00032_crop004",
  "status": "Valid",
  "language": "en",
  "instruction": [
    "Assemble plastic model pieces."
  ]
}
```

Rules:

- `status` should be `"Valid"` when `build.require_annotation` is enabled
- `instruction` is the preferred normalized field
- if `instruction` is absent, build falls back to hierarchy-style fields when available

## Supported Source Types

Built-in adapters:

- `buildai`
- `flat_shard`
- `image_sequence`
- `legacy_buildai`
- `video_folder`

Adapter responsibility:

- source-specific preprocess if needed
- produce clip descriptors with frame references
- resolve each clip's `seq_folder`
- expose optional annotation context

For a new dataset:

1. Add or extend one dataset adapter under `lib/pipeline/datasets/`.
2. Make that adapter produce `ClipDescriptor` objects.
3. Point a standard config at that adapter.
4. Use `run_dataset_pipeline.py` as the official entrypoint.

Do not add a second full build stack per dataset unless the data shape truly cannot be normalized through descriptors and adapters.

### Legacy BuildAI 100K

The old processed layout is now supported through the `legacy_buildai` adapter.

Expected structure:

```text
<processed_root>/
  factory_001/
    worker_001/
      processed/
        factory001_worker001_00010_crop018/
          extracted_images/
          world_space_res.pth
          SLAM/
```

Use the example config:

```bash
python scripts/run_dataset_pipeline.py \
  --config configs/dataset_pipeline_legacy_buildai.example.yaml \
  --stages prepare,build,validate
```

For old BuildAI, the typical build settings are:

- `build.source_fps: 5.0`
- `build.target_fps: 30.0`
- `build.interpolate_labels: true`

This uses `extracted_images/` as the 30 FPS descriptor frame source and resamples 5 FPS stage outputs onto that timeline during final build.

## Internal Manifest Boundary

The pipeline still uses a frozen clip manifest internally. That is intentional.

Why it stays:

- it freezes the clip list before expensive stage runs
- it decouples source layout from downstream HaWoR stages
- it lets annotation, filtering, and build operate on the same stable clip set
- it allows rebuilds without rescanning the source

Why it should feel less complex now:

- normal users should not need to manually operate on manifests
- the orchestrator creates and consumes the manifest for the official path
- direct manifest scripts are now considered advanced tools, not the default workflow

## Advanced Workflow

Use these scripts only when you explicitly want to split the workflow by hand.

Build a frozen manifest:

```bash
python scripts/build_clip_manifest.py \
  --config configs/dataset_pipeline_buildai.example.yaml \
  --manifest_out /path/to/run/clip_manifest.jsonl \
  --shard_dirs_out /path/to/run/shard_dirs.txt
```

Run stage inference from a manifest:

```bash
python scripts/batch_infer.py \
  --descriptor_manifest /path/to/run/clip_manifest.jsonl \
  --stages detect_track,motion \
  --gpus 0,1,2,3
```

Build the final dataset directly from a manifest:

```bash
python scripts/build_vla_from_manifest.py \
  --descriptor_manifest /path/to/run/clip_manifest.jsonl \
  --annotation_root /path/to/annotations \
  --output_dir /path/to/final_dataset \
  --require_annotation \
  --frames_per_shard 10000 \
  --source_fps 5.0 \
  --target_fps 30.0 \
  --interpolate_labels
```

This direct manifest path is the build kernel behind the orchestrator. Use it for advanced split workflows, reruns, or debugging.

## Dataset Schema

Each final WebDataset sample contains:

- `*.image.jpg`: RGB frame bytes
- `*.lowdim.npy`: `float32[116]`
- `*.meta.json`: frame metadata such as `clip_id`, `instruction`, `instruction_num`, `language`, and `presence`

`lowdim[116]` layout:

- `0:3` left wrist joint position in world coordinates
- `3:6` right wrist joint position in world coordinates
- `6:12` left root orientation as rot6d
- `12:18` right root orientation as rot6d
- `18:33` left fingertip positions `(5, 3)` in world coordinates
- `33:48` right fingertip positions `(5, 3)` in world coordinates
- `48:66` next-frame wrist state
- `66:96` next-frame fingertip state
- `96:112` camera extrinsic as flattened `w2c` `4x4`
- `112:116` camera intrinsic `[fx, fy, cx, cy]`

Important conventions:

- wrist translation uses MANO wrist joint world coordinates, not raw `pred_trans`
- fingertip coordinates are in the HaWoR/SLAM world frame
- camera extrinsic is `World2Cam`

If stage outputs were generated at a lower FPS than the descriptor frames, use:

- `build.source_fps`
- `build.target_fps`
- `build.interpolate_labels`

This resamples wrist pose, fingertip positions, camera extrinsics, and presence flags onto the descriptor frame timeline during final build.

Typical choices:

- BuildAI-style 5 FPS stage outputs over 30 FPS RGB descriptors: `5.0 / 30.0 / true`
- video or image datasets where stages and descriptors use the same frames: matching FPS values and `interpolate_labels: false`
- If your annotations are named like `<clip_id>_qwen-annotation.json`, set `build.annotation_suffix: _qwen-annotation.json`

## Validation

Validate a completed run with:

```bash
python scripts/run_dataset_pipeline.py \
  --config configs/dataset_pipeline_buildai.example.yaml \
  --stages validate
```

Or directly:

```bash
python scripts/validate_pipeline_run.py \
  --descriptor_manifest /path/to/run/clip_manifest.jsonl \
  --annotation_root /path/to/annotations \
  --dataset_dir /path/to/final_dataset \
  --max_clips 200 \
  --dataset_sample_checks 20
```

Recommended smoke test before full production:

1. Run on 2 to 5 representative source groups.
2. Finish `prepare`, `annotate`, `infer`, `filter`, `build`, and `validate`.
3. Inspect at least 20 samples across multiple shards.
4. Confirm images, lowdim, and language all refer to the same clip and frame.

## Inspecting Built Samples

For frame-level inspection of final WebDataset shards, prefer the Rerun viewer:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-rerun-viewer.txt

python tools/ops/rerun_webdataset_visualizer.py \
  --input /path/to/final_dataset/train \
  --render-mode keypoint
```

`requirements-rerun-viewer.txt` is intentionally viewer-only. It does not pull in the full RoWaH training/inference stack, so you can install it in a separate environment as long as that environment already has a compatible `torch` build.

Offline `.rrd` export uses the same recording path as the live viewer:

```bash
python tools/ops/rerun_webdataset_visualizer.py \
  --input /path/to/final_dataset/train \
  --render-mode mesh \
  --output-mode offline \
  --rrd-out /path/to/inspect_mesh.rrd
```

Use `--output-mode both` to save the `.rrd` and open the live viewer from the same run.

If the filters match multiple episodes, the script will list candidates in the terminal and ask you to choose one. You can also select directly with:

- `--clip-id <clip_id>`
- `--episode-key <episode_key>`
- `--episode-index <1-based-index>`

Render modes:

- `keypoint`: wrist plus five fingertips
- `skeleton`: full MANO 21-joint skeleton
- `mesh`: full MANO mesh with 2D joint overlay

`skeleton` and `mesh` now replay MANO directly from each sample's `mano.npy`, so no descriptor manifest is required:

```bash
python tools/ops/rerun_webdataset_visualizer.py \
  --input /path/to/final_dataset/train \
  --render-mode mesh
```

The older `tools/ops/webdataset_visualizer.py` HTTP viewer is still available as a fallback, but it is no longer the preferred inspection path.

## Maintenance And Legacy Tools

These are useful, but they are not part of the official mainline workflow.

- `scripts/filter_webdataset.py`
  - analyze and filter already-built WDS shards
- `scripts/repack_webdataset.py`
  - repack kept samples into new shard sizes after WDS filtering
- `scripts/rewrite_webdataset_lowdim.py`
  - repair old BuildAI 10K WebDataset lowdim semantics
  - this is a BuildAI-specific legacy repair tool, not a normal pipeline step
- `scripts/build_vla_dataset.py`
  - old BuildAI-oriented builder wrapper
  - kept only as a legacy fallback, not the recommended build path

## Example Configs

Standard examples:

- `configs/dataset_pipeline_buildai.example.yaml`
- `configs/dataset_pipeline_flat_shard.example.yaml`
- `configs/dataset_pipeline_image_sequence.example.yaml`
- `configs/dataset_pipeline_legacy_buildai.example.yaml`
- `configs/dataset_pipeline_video_folder.example.yaml`

Compact shorthand example:

- `configs/dataset_pipeline_buildai.compact.example.yaml`
