#!/usr/bin/env python3
"""Unified batch inference entrypoint for HaWoR."""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.batch.cli import (
    build_batch_infer_parser,
    load_batch_inputs,
    normalize_batch_infer_args,
)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")

_env_tmp = os.environ.get("HAWOR_BATCH_TMPDIR")
if _env_tmp:
    SHARED_TMP_DIR = Path(_env_tmp).expanduser().resolve()
else:
    SHARED_TMP_DIR = (PROJECT_ROOT / ".tmp").resolve()
SHARED_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(SHARED_TMP_DIR)
os.environ["TEMP"] = str(SHARED_TMP_DIR)
os.environ["TMP"] = str(SHARED_TMP_DIR)
tempfile.tempdir = str(SHARED_TMP_DIR)


def get_parser():
    return build_batch_infer_parser()


def _resolve_run_dir(run_dir: str | None) -> Path:
    if run_dir:
        return Path(run_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "batch_runs" / timestamp


def main(argv: list[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = get_parser()
    args = parser.parse_args(raw_argv)
    compatibility_notes = normalize_batch_infer_args(args, raw_argv=raw_argv)
    inputs = load_batch_inputs(args)

    run_dir = _resolve_run_dir(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=== Batch Inference Configuration ===")
    print(f"Input mode: {inputs.input_mode}")
    print(f"Input path: {inputs.input_path}")
    print(f"Total videos in source: {inputs.total_items}")
    print(f"Processing range: [{inputs.start_idx}, {inputs.end_idx})")
    print(f"Videos to process: {len(inputs.video_paths)}")
    print(f"GPUs: {args.gpus}")
    print(f"Stages: {args.stages}")
    print("Scheduler mode: wave")
    print(f"Worker slots per GPU: {args.workers_per_gpu}")
    print(f"Max stage retries: {args.max_stage_retries}")
    print(f"Chunk batch size (motion): {args.chunk_batch_size}")
    print(f"Render batch size (motion): {args.render_batch_size}")
    print(f"Detect batch size (detect_track): {args.detect_batch_size}")
    print(f"Detect I/O workers: {args.detect_io_workers}")
    print(f"Any4D batch size (slam): {args.any4d_batch_size}")
    print(f"SLAM backend: {args.slam_backend}")
    print(f"Depth backend (slam): {args.depth_backend or '(env HAWOR_DEPTH_BACKEND, default metric3d)'}")
    print(f"Dense depth all frames (slam): {args.depth_predict_all_frames}")
    print(f"Resume: {args.resume}")
    print(f"Run directory: {run_dir}")
    if inputs.input_mode != "descriptor_manifest":
        print("Compatibility input mode is in use. `--descriptor_manifest` is the preferred infer input.")
    for note in compatibility_notes:
        print(f"Compatibility note: {note}")
    print()

    from lib.pipeline.batch import BatchRunConfig, BatchScheduler

    config = BatchRunConfig.from_args(
        args,
        video_paths=inputs.video_paths,
        descriptors=inputs.descriptors,
        run_dir=run_dir,
    )
    success = BatchScheduler(config).run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
