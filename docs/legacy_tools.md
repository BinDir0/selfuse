# Legacy Tools

This repository keeps a number of older scripts for compatibility and debugging. They are not the preferred path for new work.

## Legacy But Still Kept

Legacy WebDataset builder:

- `scripts/build_vla_dataset.py`
- `lib/pipeline/exporters/webdataset.py`

These remain available because some older BuildAI layouts and debugging workflows still refer to them. New dataset production should use:

```bash
python scripts/run_dataset_pipeline.py --config <config.yaml> --stages prepare,annotate,infer,filter,build,validate
```

For the current single-video path, the shorter default is preferred:

```bash
python scripts/run_dataset_pipeline.py --config configs/dataset_pipeline_single_video.example.yaml
```

or, for advanced manual runs:

```bash
python scripts/build_clip_manifest.py ...
python scripts/filter_manifest_by_quality.py ...
python scripts/build_vla_from_manifest.py ...
```

For already-built WebDataset outputs, use the maintained pair only:

```bash
python scripts/filter_webdataset.py ...
python scripts/sanity_check_webdataset.py ...
```

## Compatibility Wrappers

The former compatibility wrappers in `scripts/` have been moved to `deprecated/scripts_wrappers/`. Use the canonical `tools/` paths directly:

- `tools/ops/analyze_run.py`
- `tools/ops/create_done_markers.py`
- `tools/dataset/generate_video_list.py`
- `tools/ops/validate_setup.sh`

## Deprecated Area

Everything under `deprecated/` should be treated as reference material only unless you have a concrete reason to revive it. That directory contains:

- older batch schedulers
- old smoke/perf scripts
- implementation notes from previous iterations
- scripts tied to superseded workflows
- unrefactored one-off tools preserved under `deprecated/unrefactored_tools/`

Do not build new production paths on top of `deprecated/`.

## Quarantine Boundary

`deprecated/unrefactored_tools/manifest.json` is the source of truth for which
scripts were moved out of `scripts/` during the single-video pipeline refactor.
`tests/test_quarantine_unrefactored_tools.py` enforces two invariants:

1. Every manifest entry's `original` path is gone and its `deprecated` path
   exists.
2. No active (non-`deprecated/`) file references a moved `original` path; the
   only allowed reference is the new `deprecated/unrefactored_tools/...` path.

Quarantined tools keep their original relative layout, so their internal
`PROJECT_ROOT / "scripts" / ...` references resolve **within** the quarantine.
Some of these tools also reference scripts that were *not* moved (e.g.
`scripts/rewrite_buildai_interpolated_wds.py`,
`scripts/check_motion_stage_outputs.py`) — those cross-references are stale by
design and are not fixed, because everything under `deprecated/` is
reference-only and must not be wired into production.
