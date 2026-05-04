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

Do not build new production paths on top of `deprecated/`.
