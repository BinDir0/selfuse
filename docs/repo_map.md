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

Diagnostics:

- Read-only inspection / analysis tools: `tools/diagnostics/` (e.g. `diagnose_*`,
  `audit_*`, `check_fps_alignment`, `dataset_stats`, `validate_hand_metric_size`).
  These are developer tools, not part of the pipeline; run them directly, e.g.
  `python tools/diagnostics/diagnose_camera_drift_reproj.py --seq_folder <clip>`.

Dataset / export tools:

- Frame/mesh/shard utilities and demo packaging: `tools/dataset/` (e.g. `make_demo`,
  `export_cam_space_meshes`, `repack_webdataset`, `prepare_partial_infer_manifest`,
  `generate_video_list`).

End-to-end regression:

- `tools/ops/e2e_smoke.sh` runs the REAL pipeline (real worker fork + CUDA + manifest
  flow) and asserts invariants the CPU unit tests can't cover (CUDA-in-fork, input
  wiring, preflight fail-fast, result.npz health, cleanup, resume). Run on a GPU box.

## Public Scripts

Use these as the maintained public scripts:

- `scripts/run_dataset_pipeline.py`
- `scripts/build_clip_manifest.py`
- `scripts/build_vla_from_manifest.py`
- `scripts/filter_manifest_by_quality.py`
- `scripts/filter_webdataset.py`
- `scripts/sanity_check_webdataset.py`
- `scripts/batch_infer.py`
- `scripts/validate_pipeline_run.py`

The following compatibility wrappers have been moved to `deprecated/scripts_wrappers/` (use `tools/` directly instead):

- `analyze_run.py` -> `tools/ops/analyze_run.py`
- `create_done_markers.py` -> `tools/ops/create_done_markers.py`
- `generate_video_list.py` -> `tools/dataset/generate_video_list.py`

The following compatibility script is still in place:

- `scripts/validate_setup.sh` -> `tools/ops/validate_setup.sh`

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
