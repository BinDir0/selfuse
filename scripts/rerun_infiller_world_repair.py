#!/usr/bin/env python3
"""Rerun the maintained infiller/world stage for selected seq_folder outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.datasets.descriptors import ClipDescriptor
from lib.pipeline.runtime import WorkerRuntime
from lib.pipeline.stage_api import PipelineVideoTask, run_pipeline_stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rerun the maintained infiller/world stage for selected seq_folder outputs."
    )
    parser.add_argument(
        "--seq_folder",
        action="append",
        default=[],
        help="Sequence folder to repair. Can be passed multiple times.",
    )
    parser.add_argument(
        "--seq_folder_list",
        default=None,
        help="Optional text file with one seq_folder per line.",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="CUDA_VISIBLE_DEVICES value for the infiller runtime. Use empty string for CPU-only.",
    )
    parser.add_argument(
        "--checkpoint",
        default="./weights/hawor/checkpoints/hawor.ckpt",
        help="Motion checkpoint path kept for runtime config compatibility.",
    )
    parser.add_argument(
        "--infiller_weight",
        default="./weights/hawor/checkpoints/infiller.pt",
        help="Infiller checkpoint path.",
    )
    parser.add_argument(
        "--infiller_window_batch_size",
        type=int,
        default=64,
        help="Batch size for infiller windows.",
    )
    parser.add_argument(
        "--rebuild_cam_space_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rebuild cam_space cache before rerunning infiller.",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force rerun even if .infiller.done exists.",
    )
    parser.add_argument(
        "--backup_dir",
        default=None,
        help="Optional directory where existing world_space_res.pth files are copied before rerun.",
    )
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Report planned reruns without executing them.",
    )
    parser.add_argument(
        "--report_out",
        default=None,
        help="Optional JSON report path.",
    )
    return parser


def _collect_seq_folders(args: argparse.Namespace) -> list[Path]:
    seq_folders: list[Path] = []
    seen: set[Path] = set()

    def _add(path_str: str) -> None:
        path = Path(path_str).expanduser().resolve()
        if path in seen:
            return
        seen.add(path)
        seq_folders.append(path)

    for item in args.seq_folder:
        if item:
            _add(item)

    if args.seq_folder_list:
        for line in Path(args.seq_folder_list).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                _add(line)

    if not seq_folders:
        raise SystemExit("No seq_folder provided. Use --seq_folder or --seq_folder_list.")
    return seq_folders


def _infer_frame_count(seq_folder: Path) -> int:
    world_path = seq_folder / "world_space_res.pth"
    if world_path.is_file():
        pred_trans, *_rest = joblib.load(world_path)
        return int(np.asarray(pred_trans).shape[1])

    frame_dir = seq_folder / "extracted_images"
    if frame_dir.is_dir():
        image_count = len(sorted(frame_dir.glob("*.jpg"))) or len(sorted(frame_dir.glob("*.png")))
        if image_count > 0:
            return int(image_count)

    tracks = sorted(seq_folder.glob("tracks_*_*"))
    for track_dir in reversed(tracks):
        frame_chunks_path = track_dir / "frame_chunks_all.npy"
        if not frame_chunks_path.is_file():
            continue
        frame_chunks_all = joblib.load(frame_chunks_path)
        max_frame = -1
        for hand_idx in (0, 1):
            for chunk in frame_chunks_all.get(hand_idx, []):
                chunk_array = np.asarray(chunk, dtype=np.int64).reshape(-1)
                if chunk_array.size > 0:
                    max_frame = max(max_frame, int(chunk_array.max()))
        if max_frame >= 0:
            return int(max_frame + 1)

    raise RuntimeError(f"Failed to infer frame count for {seq_folder}")


def _build_task(seq_folder: Path) -> PipelineVideoTask:
    clip_id = seq_folder.name
    frame_count = _infer_frame_count(seq_folder)
    descriptor = ClipDescriptor(
        clip_id=clip_id,
        clip_name=clip_id,
        storage_kind="image_sequence",
        root_dir=str(seq_folder.parent.resolve()),
        seq_folder=str(seq_folder.resolve()),
        frame_names=[],
        frame_dir=str((seq_folder / "extracted_images").resolve()),
        media_path=clip_id,
        frame_count_override=frame_count,
    )
    return PipelineVideoTask(
        video_path=clip_id,
        seq_folder=seq_folder.resolve(),
        descriptor=descriptor,
    )


def _backup_world(seq_folder: Path, backup_dir: Path | None) -> str | None:
    if backup_dir is None:
        return None
    world_path = seq_folder / "world_space_res.pth"
    if not world_path.is_file():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"{seq_folder.name}.world_space_res.before_rerun.pth"
    shutil.copy2(world_path, target)
    return str(target)


def rerun_seq_folder(
    seq_folder: Path,
    *,
    runtime: WorkerRuntime | None,
    force: bool,
    dry_run: bool,
    backup_dir: Path | None,
) -> dict:
    report: dict[str, object] = {
        "seq_folder": str(seq_folder),
        "clip_id": seq_folder.name,
    }
    if not seq_folder.is_dir():
        report["status"] = "missing_seq_folder"
        return report

    frame_count = _infer_frame_count(seq_folder)
    report["frame_count"] = int(frame_count)
    report["world_exists_before"] = bool((seq_folder / "world_space_res.pth").is_file())
    report["backup_path"] = _backup_world(seq_folder, backup_dir)

    if dry_run:
        report["status"] = "planned"
        report["force"] = bool(force)
        return report

    if runtime is None:
        raise RuntimeError("runtime is required when dry_run is False")

    task = _build_task(seq_folder)
    result = run_pipeline_stage(
        "infiller",
        task,
        runtime.stage_config,
        runtime=runtime,
        resume=not force,
        force=force,
    )
    report["status"] = result.get("status", "unknown")
    report["result"] = result
    report["world_exists_after"] = bool((seq_folder / "world_space_res.pth").is_file())
    report["done_marker_exists_after"] = bool((seq_folder / ".infiller.done").is_file())
    return report


def main() -> None:
    args = build_parser().parse_args()
    seq_folders = _collect_seq_folders(args)
    backup_dir = Path(args.backup_dir).expanduser().resolve() if args.backup_dir else None

    runtime = None
    if not args.dry_run:
        runtime = WorkerRuntime(
            gpu=args.gpu,
            checkpoint=args.checkpoint,
            infiller_weight=args.infiller_weight,
            infiller_window_batch_size=int(args.infiller_window_batch_size),
            rebuild_cam_space_cache=bool(args.rebuild_cam_space_cache),
        )

    report = [
        rerun_seq_folder(
            seq_folder,
            runtime=runtime,
            force=bool(args.force),
            dry_run=bool(args.dry_run),
            backup_dir=backup_dir,
        )
        for seq_folder in seq_folders
    ]

    if args.report_out:
        report_path = Path(args.report_out).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
