# Pipeline Structure

This repository now separates pipeline code into three layers so later integration work can target stable module paths without breaking existing scripts.

## Layout

- `lib/pipeline/stages/`
  - Primary implementations for the runtime stages:
  - `detect_track.py`
  - `hawor_video.py`
  - `motion.py`
  - `infiller.py`
  - `slam.py`
- `lib/pipeline/exporters/`
  - Dataset/export logic.
  - `webdataset.py` is the canonical WebDataset builder implementation.
- `lib/pipeline/batch/`
  - Batch inference orchestration.
  - Holds the stage-wave scheduler, persisted run state, event logging, and worker pool logic.
- `lib/pipeline/stage_api.py`
  - Shared batch/pipeline orchestration surface.
  - Holds common stage config, task resolution, output validation, and unified stage dispatch.
- `scripts/`
  - User-facing entrypoints and batch orchestration.
  - `batch_infer.py`, `batch_worker.py`, and `build_vla_dataset.py` should remain valid stable CLI entrypoints.
- `deprecated/scripts_test_video/`
  - Compatibility wrappers for older paths.
  - They re-export the new implementations so old automation does not break immediately.
- `deprecated/`
  - Historical experiments and older pipeline variants.
  - These are not the preferred integration surface.

## Recommended Imports

Prefer importing from `lib.pipeline` in new code:

```python
from lib.pipeline.stage_api import PipelineVideoTask, StageExecutionConfig, run_pipeline_stage
from lib.pipeline.batch.scheduler import BatchScheduler
from lib.pipeline.stages.detect_track import detect_track_video
from lib.pipeline.stages.hawor_video import hawor_motion_estimation, hawor_infiller
from lib.pipeline.stages.slam import hawor_slam
from lib.pipeline.exporters.webdataset import main
```

Avoid introducing new dependencies on `deprecated/scripts_test_video/*` unless the goal is temporary backward compatibility.

## Entry Point Rules

- Keep `scripts/*.py` as CLI entrypoints.
- Keep reusable logic in `lib/pipeline/...`.
- Keep orchestration data structures and shared stage dispatch in `lib/pipeline/stage_api.py`.
- Keep compatibility wrappers thin and non-authoritative.
- Move future refactors toward `lib/pipeline/...` first, then update wrappers if needed.

## Practical Impact

- Existing commands continue to work:
  - `python scripts/batch_infer.py ...`
  - `python scripts/build_vla_dataset.py ...`
  - `python demo.py ...`
- New pipeline integration can depend on `lib/pipeline/...` without coupling to legacy script locations.
