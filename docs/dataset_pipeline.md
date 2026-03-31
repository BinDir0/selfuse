# Dataset Pipeline

This repository now supports a production dataset pipeline that connects raw-source preprocessing, clip-level manifest freezing, annotation sidecars, HaWoR stage inference, quality filtering, final WebDataset build, and validation.

The pipeline is adapter-driven. A new dataset should normally require only:

- one dataset adapter
- one YAML config
- optional preprocess / annotation bridge scripts

## Goal

The pipeline is designed for production use, not just one-off BuildAI exports.

The stable contract is:

1. Raw videos enter an external source-specific preprocess pipeline.
2. Preprocess emits JPEG tar shard groups plus per-clip `seq_folder` directories.
3. HaWoR stages run against a frozen clip manifest snapshot.
4. Annotation is written as one sidecar JSON per clip.
5. HaWoR stages run and produce clip-local outputs under each `seq_folder`.
6. Optional quality filtering drops bad clips by rewriting the manifest.
7. Final build reads the filtered manifest, stage outputs, and sidecars to produce the trainable WebDataset.

BuildAI is treated as one source adapter. The intermediate and final contracts are source-agnostic.

## Adapter Model

Each dataset plugs into the pipeline through a small adapter interface:

- `prepare(...)`
- `build_descriptors(...)`
- `resolve_annotation_context(...)`
- `validate_source(...)`

The canonical handoff is the frozen clip manifest. Downstream stages do not need to know whether the source came from BuildAI tar shards, image-sequence clips, or video folders with extracted frames.

## Storage Contract

### 1. Preprocess Output

The preprocess stage should emit a shard root like:

```text
<shard_root>/
  group001/
    shard-000000.tar
    shard-000001.tar
    ...
  group002/
    shard-000000.tar
    ...
```

Each clip must also have a `seq_folder` resolved by the descriptor. HaWoR stage outputs continue to live under that clip-local directory, for example:

```text
<seq_folder>/
  img/
  tracking_result.pth
  cam_space_res.pth
  SLAM/
  world_space_res.pth
```

### 2. Frozen Clip Manifest

`scripts/build_clip_manifest.py` writes a JSONL manifest. Each row freezes:

- `clip_id`
- `source_id`
- `split`
- `group_id`
- the full `ClipDescriptor`

This is the critical orchestration boundary. All downstream stages consume the manifest instead of rescanning live shard directories.

The built-in descriptors currently support:

- `tar_shard`
- `image_sequence`

### 3. Annotation Sidecars

Annotation is stored as:

```text
<annotation_root>/<clip_id>.annotation.json
```

Recommended schema:

```json
{
  "clip_id": "factory023_worker001_00032_crop004",
  "status": "Valid",
  "language": "en",
  "instruction": [
    "Assemble plastic model pieces.",
    "Detach and join small white plastic components from a sprue on a dark work surface."
  ],
  "hierarchy": {
    "level1": "Assemble plastic model pieces.",
    "level2": "Detach and join small white plastic components from a sprue on a dark work surface.",
    "level3": "Manipulate a white plastic model sprue containing multiple ribbed, rectangular parts with tabs and slots; carefully break off individual pieces and align them for assembly.",
    "level4": "Hold the main sprue steady with the left hand while the right hand pinches and snaps off a small ribbed component near the center of the sprue; then reposition the detached piece to interlock it with another nearby component.",
    "level5": "Grip the central section of the white plastic sprue with the left hand. Use the right thumb and index finger to press and detach a rectangular ribbed piece with a tab. Place the detached piece aside temporarily. Pick up a second nearby piece with the right hand. Align its slot with the tab of the first piece. Press them together until they click into place. Repeat with adjacent parts."
  }
}
```

Rules:

- `status` must be `"Valid"` for the clip to be accepted when annotation is required.
- `language` is optional but should be filled for multilingual training.
- `instruction` is the preferred normalized field.
- If `instruction` is absent, build falls back to `hierarchy` and then `global_analysis.level1..level5`.

## Runtime Separation

The pipeline assumes two Python runtimes:

- `hawor_python`: `detect_track`, `motion`, `infiller`, final build, validation
- `slam_python`: `slam`

Do not merge these environments. The orchestrator dispatches each stage with the correct interpreter from config.

## Entry Points

### Whole Pipeline

Use:

```bash
python scripts/run_dataset_pipeline.py \
  --config configs/dataset_pipeline_buildai.example.yaml \
  --stages preprocess,manifest,annotate,detect_motion,slam,infiller,filter,build,validate
```

The config controls:

- adapter selection
- preprocess repo path and config
- runtime Python executables
- batch inference arguments
- annotation command hook
- final build arguments
- validation limits

### Build Frozen Manifest

```bash
python scripts/build_clip_manifest.py \
  --config configs/dataset_pipeline_buildai.example.yaml \
  --manifest_out /path/to/run/clip_manifest.jsonl \
  --shard_dirs_out /path/to/run/shard_dirs.txt
```

Legacy shard scanning mode is still supported for existing BuildAI-style roots.

### Run Annotation From Manifest

Annotation consumes the frozen manifest only. It does not depend on `detect_motion`, `slam`, `infiller`, or `filter`, so it can be run as soon as `manifest` completes.

The orchestrator passes the base manifest as `{manifest}` to the annotation command template even if later stages switch to a filtered manifest.

### Run HaWoR Stages From Manifest

```bash
python scripts/batch_infer.py \
  --descriptor_manifest /path/to/run/clip_manifest.jsonl \
  --stages detect_track,motion \
  --gpus 0,1,2,3
```

Run `slam` separately with the `any4` environment, then `infiller` back in the `hawor` environment.

### Build Final Dataset

```bash
python scripts/build_vla_from_manifest.py \
  --descriptor_manifest /path/to/run/clip_manifest.jsonl \
  --annotation_root /path/to/annotations \
  --output_dir /path/to/final_dataset \
  --require_annotation \
  --frames_per_shard 10000
```

Final WebDataset samples contain:

- `*.image.jpg`: the frame bytes
- `*.lowdim.npy`: one `float32[116]` vector per frame
- `*.meta.json`: per-frame metadata such as `clip_id`, `instruction`, `instruction_num`, `language`, and `presence`

`lowdim[116]` is laid out as:

- `0:3` left wrist joint position in world coordinates
- `3:6` right wrist joint position in world coordinates
- `6:12` left root orientation as rot6d
- `12:18` right root orientation as rot6d
- `18:33` left fingertip positions `(5, 3)` in world coordinates
- `33:48` right fingertip positions `(5, 3)` in world coordinates
- `48:66` next-frame wrist position + rot6d
- `66:96` next-frame fingertip positions
- `96:112` camera `w2c` extrinsic flattened as `4x4`
- `112:116` camera intrinsic `[fx, fy, cx, cy]`

The coordinates above are in the HaWoR/SLAM world frame, while the camera extrinsic is stored as a `World2Cam` homogeneous transform.

### Validate a Run

```bash
python scripts/validate_pipeline_run.py \
  --descriptor_manifest /path/to/run/clip_manifest.jsonl \
  --annotation_root /path/to/annotations \
  --dataset_dir /path/to/final_dataset \
  --max_clips 200 \
  --dataset_sample_checks 20
```

## Production Rollout Test

Before full-scale production, run a smoke test with a small but representative slice.

Recommended test:

1. Select 2 to 5 source groups with real diversity.
2. Run the full pipeline end-to-end, including annotation sidecar generation.
3. Limit validation to the first 100 to 200 clips for a fast first pass.
4. Manually inspect at least 20 final dataset samples across multiple shards.
5. Confirm dataset metadata contains `clip_id`, `instruction`, `instruction_num`, `language`, and `presence`.

Acceptance criteria:

- manifest clip count matches expected preprocess output
- `detect_track`, `motion`, `slam`, and `infiller` all validate cleanly
- missing annotation count is zero when `require_annotation=true`
- final dataset shards are readable and contain expected schema
- at least one repeated spot-check verifies image bytes, lowdim features, and instruction text refer to the same clip

## Notes

- The final dataset builder shards by approximate frame budget, but each shard contains whole episodes only.
- The manifest-based builder does not rely on old BuildAI directory rescans.
- If annotation is delayed, you can generate the final dataset later from the frozen manifest without rerunning preprocess or HaWoR stages.
- `tools/ops/webdataset_visualizer.py` supports two modes:
  - `keypoint`: lightweight overlay from the stored lowdim wrist/fingertip world coordinates; with `--descriptor_manifest` it can cross-check against MANO cache and unlock better diagnostics
  - `mano`: projects MANO mesh/joints reconstructed from `world_space_res.pth`; this is the more reliable mode for checking geometric accuracy
- Built-in adapters now include `buildai`, `flat_shard`, `image_sequence`, and `video_folder`.
- Example configs live under:
  - `configs/dataset_pipeline_buildai.example.yaml`
  - `configs/dataset_pipeline_flat_shard.example.yaml`
  - `configs/dataset_pipeline_image_sequence.example.yaml`
  - `configs/dataset_pipeline_video_folder.example.yaml`
