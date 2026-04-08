# Repo Map

This repository has a few clearly different layers. Use this map to avoid mixing official pipeline code, batch-inference utilities, and legacy material.

## Top-Level Layout

- `lib/`: maintained library code
- `scripts/`: public entrypoints and compatibility wrappers
- `tools/`: operational helpers and analysis utilities
- `configs/`: example and recipe configs for the official dataset pipeline
- `docs/`: maintained documentation for official and legacy workflows
- `deprecated/`: old experiments, notes, and scripts kept only for reference

## Official Paths

Dataset pipeline:

- Entry: `scripts/run_dataset_pipeline.py`
- Config normalization: `lib/pipeline/pipeline_config.py`
- Orchestration: `lib/pipeline/orchestrator/`
- Dataset adapters: `lib/pipeline/datasets/`
- Clip manifest boundary: `lib/pipeline/clip_manifest.py`
- Filtering: `lib/pipeline/filtering/`
- Final build/export: `lib/pipeline/exporters/manifest_vla.py`

Batch inference:

- Entry: `scripts/batch_infer.py`
- CLI/runtime state: `lib/pipeline/batch/`
- Stage validation helpers: `lib/pipeline/stage_api.py`
- Stage implementations: `lib/pipeline/stages/`

Visualization:

- Viewer/runtime code: `lib/vis/`
- Ops/debug helpers: `tools/ops/`

## Public Scripts

Use these as the maintained public scripts:

- `scripts/run_dataset_pipeline.py`
- `scripts/build_clip_manifest.py`
- `scripts/build_vla_from_manifest.py`
- `scripts/filter_manifest_by_quality.py`
- `scripts/batch_infer.py`
- `scripts/validate_pipeline_run.py`

The following compatibility scripts are still callable but are wrappers around `tools/`:

- `scripts/analyze_run.py`
- `scripts/create_done_markers.py`
- `scripts/generate_video_list.py`
- `scripts/validate_setup.sh`

## Internal Boundaries

The current intended module boundaries are:

- `lib/pipeline/orchestrator/`: config validation, stage selection, command construction, run execution
- `lib/pipeline/filtering/`: manifest filter orchestration and report generation
- `lib/pipeline/exporters/manifest_build/`: manifest preparation, feature loading/resampling, shard writing, final build runner

Compatibility facades stay in place for user-facing imports and scripts.

## Legacy Areas

These still exist, but they are not the main path:

- `scripts/build_vla_dataset.py`
- `lib/pipeline/exporters/webdataset.py`
- most files under `deprecated/`

See [docs/legacy_tools.md](/root/.openclaw/workspace/projects/hawor_original/HaWoR/docs/legacy_tools.md) for the current status of legacy tooling.
