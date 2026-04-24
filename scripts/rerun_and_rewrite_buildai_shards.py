#!/usr/bin/env python3
"""Batch rerun BuildAI infiller/world outputs and rewrite matching WDS shards."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.exporters.webdataset_rewriter import (  # noqa: E402
    iter_shard_paths,
    iter_shard_samples,
    validate_sample_record,
)
from scripts.rewrite_webdataset_lowdim import _sample_clip_id  # noqa: E402


BUILDAI_CLIP_RE = re.compile(r"^factory(\d{3})_worker(\d{3})_")
BUILDAI_SHORT_CLIP_RE = re.compile(r"^f(\d{3})_w(\d{3})_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Directly rerun BuildAI world outputs for the selected shard range, then rewrite the same shards."
    )
    parser.add_argument("--source_shard_dir", required=True, help="Source WDS shard directory")
    parser.add_argument("--buildai_processed_root", required=True, help="Processed root containing seq_folder outputs")
    parser.add_argument("--output_dir", required=True, help="Root directory for rewritten batch outputs")
    parser.add_argument("--report_root", required=True, help="Root directory for batch reports")
    parser.add_argument("--shard_start", type=int, required=True, help="Inclusive shard index in sorted shard order")
    parser.add_argument("--shard_end", type=int, required=True, help="Exclusive shard index in sorted shard order")
    parser.add_argument("--python_bin", default=sys.executable, help="Python executable used to invoke helper scripts")
    parser.add_argument("--gpu", default="0", help="GPU id passed to rerun_infiller_world_repair.py")
    parser.add_argument("--checkpoint", default="./weights/hawor/checkpoints/hawor.ckpt", help="Optional motion checkpoint path")
    parser.add_argument("--infiller_weight", default="./weights/hawor/checkpoints/infiller.pt", help="Optional infiller checkpoint path")
    parser.add_argument("--infiller_window_batch_size", type=int, default=64, help="Infiller window batch size")
    parser.add_argument(
        "--repair_dense_slam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repair hawor_slam_w_scale exports using dense dpvo_raw trajectories before rerunning world",
    )
    parser.add_argument("--rewrite_workers", type=int, default=1, help="Worker count for shard rewrite")
    parser.add_argument("--rewrite_resume", action=argparse.BooleanOptionalAction, default=True, help="Resume rewrite if output shard already exists")
    parser.add_argument("--mano_device", default="cuda:0", help="MANO device for rewrite/check steps")
    parser.add_argument("--mano_gpus", default=None, help="Optional comma-separated GPU list for rewrite workers")
    parser.add_argument("--mano_dir", default=None, help="Optional MANO model directory")
    parser.add_argument("--source_fps", type=float, default=5.0, help="Source label fps used during export")
    parser.add_argument("--target_fps", type=float, default=30.0, help="Target label fps used during export")
    parser.add_argument("--interpolate_labels", action=argparse.BooleanOptionalAction, default=True, help="Use interpolated label path during rewrite/check")
    parser.add_argument("--backup_root", default=None, help="Optional root directory for world backups")
    parser.add_argument("--vis_root", default=None, help="Optional root directory for exported sample videos")
    parser.add_argument("--verify_sample_count", type=int, default=4, help="How many clips to run numeric WDS checks on after rewrite")
    parser.add_argument("--visualize_sample_count", type=int, default=4, help="How many clips to visualize after rewrite")
    parser.add_argument("--sample_clip_ids", default=None, help="Optional comma-separated clip ids for verification / visualization")
    parser.add_argument("--sample_clip_ids_file", default=None, help="Optional newline-delimited clip id file for verification / visualization")
    parser.add_argument("--verify_device", default="cuda:0", help="Device for check_motion_stage_outputs.py")
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=False, help="Print planned actions without executing helper scripts")
    return parser


def _append_bool_flag(cmd: list[str], flag: str, value: bool) -> None:
    cmd.append(flag if value else flag.replace("--", "--no-", 1))


def _load_meta(sample: dict) -> dict | None:
    try:
        return json.loads(sample["meta_bytes"].decode("utf-8"))
    except Exception:
        return None


def _resolve_buildai_seq_folder(processed_root: Path, clip_id: str) -> Path:
    match = BUILDAI_CLIP_RE.match(clip_id)
    short_match = BUILDAI_SHORT_CLIP_RE.match(clip_id)
    if match is not None:
        factory_id = int(match.group(1))
        worker_id = int(match.group(2))
    elif short_match is not None:
        factory_id = int(short_match.group(1))
        worker_id = int(short_match.group(2))
    else:
        raise ValueError(f"Unsupported BuildAI clip id format: {clip_id}")

    candidates = [
        processed_root / f"factory_{factory_id:03d}" / f"worker_{worker_id:03d}" / "processed" / clip_id,
        processed_root / f"factory{factory_id:03d}" / "outputs" / clip_id,
        processed_root / f"factory_{factory_id:03d}" / "outputs" / clip_id,
    ]
    seq_folder = next((path for path in candidates if path.is_dir() and (path / "world_space_res.pth").is_file()), None)
    if seq_folder is None:
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Failed to resolve seq_folder for {clip_id}. Checked: {checked}")
    return seq_folder.resolve()


def _select_shards(source_dir: Path, shard_start: int, shard_end: int) -> list[Path]:
    shard_paths = [Path(path) for path in iter_shard_paths(str(source_dir))]
    total = len(shard_paths)
    if total <= 0:
        raise RuntimeError(f"No shard tar files found in {source_dir}")
    if shard_start < 0 or shard_end < 0 or shard_end < shard_start:
        raise ValueError("Invalid shard range")
    if shard_start > total:
        raise ValueError(f"--shard_start ({shard_start}) exceeds shard count ({total})")
    return shard_paths[shard_start:min(shard_end, total)]


def _scan_selected_shards(selected_shards: list[Path], processed_root: Path) -> tuple[list[str], dict[str, str], dict[str, str]]:
    clip_to_seq: dict[str, str] = {}
    clip_to_shard: dict[str, str] = {}
    for shard_path in selected_shards:
        for sample in iter_shard_samples(str(shard_path)):
            validate_sample_record(sample)
            clip_id = _sample_clip_id(sample, _load_meta(sample))
            if clip_id not in clip_to_seq:
                clip_to_seq[clip_id] = str(_resolve_buildai_seq_folder(processed_root, clip_id))
            clip_to_shard.setdefault(clip_id, shard_path.name)
    ordered_clips = sorted(clip_to_seq.keys())
    if not ordered_clips:
        raise RuntimeError("Selected shard range contains no BuildAI clips")
    return ordered_clips, clip_to_seq, clip_to_shard


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _parse_sample_clip_ids(args: argparse.Namespace, available_clip_ids: list[str], limit: int) -> list[str]:
    selected: list[str] = []
    if args.sample_clip_ids:
        selected.extend(part.strip() for part in str(args.sample_clip_ids).split(","))
    if args.sample_clip_ids_file:
        sample_file = Path(args.sample_clip_ids_file).expanduser().resolve()
        selected.extend(line.strip() for line in sample_file.read_text(encoding="utf-8").splitlines())

    deduped = []
    seen = set()
    available = set(available_clip_ids)
    for clip_id in selected:
        if not clip_id or clip_id in seen or clip_id not in available:
            continue
        seen.add(clip_id)
        deduped.append(clip_id)

    if deduped:
        return deduped[: max(0, int(limit))]
    return available_clip_ids[: max(0, int(limit))]


def _run_command(cmd: list[str], *, dry_run: bool) -> None:
    print("[run]", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    args = build_parser().parse_args()
    source_dir = Path(args.source_shard_dir).expanduser().resolve()
    processed_root = Path(args.buildai_processed_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    report_root = Path(args.report_root).expanduser().resolve()
    vis_root = Path(args.vis_root).expanduser().resolve() if args.vis_root else None
    backup_root = Path(args.backup_root).expanduser().resolve() if args.backup_root else report_root / "world_backups"

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source shard dir not found: {source_dir}")
    if not processed_root.is_dir():
        raise FileNotFoundError(f"Processed root not found: {processed_root}")

    selected_shards = _select_shards(source_dir, int(args.shard_start), int(args.shard_end))
    selected_end = int(args.shard_start) + len(selected_shards)
    batch_id = f"shards_{int(args.shard_start):06d}_{int(selected_end):06d}"
    batch_output_dir = output_root / batch_id
    report_dir = report_root / batch_id
    feature_cache_dir = batch_output_dir / "_feature_cache"
    vis_dir = vis_root / batch_id if vis_root is not None else None
    backup_dir = backup_root / batch_id

    report_dir.mkdir(parents=True, exist_ok=True)
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    ordered_clips, clip_to_seq, clip_to_shard = _scan_selected_shards(selected_shards, processed_root)
    seq_folders = sorted(set(clip_to_seq.values()))
    seq_folder_list = report_dir / "seq_folders.txt"
    clip_id_list = report_dir / "clip_ids.txt"
    _write_lines(seq_folder_list, seq_folders)
    _write_lines(clip_id_list, ordered_clips)

    scan_report = {
        "batch_id": batch_id,
        "source_shard_dir": str(source_dir),
        "buildai_processed_root": str(processed_root),
        "selected_shards": [path.name for path in selected_shards],
        "clip_count": int(len(ordered_clips)),
        "seq_folder_count": int(len(seq_folders)),
        "seq_folder_list": str(seq_folder_list),
        "clip_id_list": str(clip_id_list),
        "batch_output_dir": str(batch_output_dir),
        "report_dir": str(report_dir),
    }
    (report_dir / "scan_report.json").write_text(json.dumps(scan_report, ensure_ascii=False, indent=2), encoding="utf-8")

    if bool(args.repair_dense_slam):
        repair_slam_cmd = [
            args.python_bin,
            "-u",
            str(PROJECT_ROOT / "scripts" / "repair_dpvo_dense_slam_exports.py"),
            "--seq_folder_list",
            str(seq_folder_list),
            "--report_out",
            str(report_dir / "repair_dense_slam_report.json"),
        ]
        _run_command(repair_slam_cmd, dry_run=bool(args.dry_run))

    rerun_cmd = [
        args.python_bin,
        "-u",
        str(PROJECT_ROOT / "scripts" / "rerun_infiller_world_repair.py"),
        "--seq_folder_list",
        str(seq_folder_list),
        "--gpu",
        str(args.gpu),
        "--checkpoint",
        str(args.checkpoint),
        "--infiller_weight",
        str(args.infiller_weight),
        "--infiller_window_batch_size",
        str(int(args.infiller_window_batch_size)),
        "--rebuild_cam_space_cache",
        "--backup_dir",
        str(backup_dir),
        "--report_out",
        str(report_dir / "rerun_report.json"),
    ]
    _run_command(rerun_cmd, dry_run=bool(args.dry_run))

    rewrite_cmd = [
        args.python_bin,
        "-u",
        str(PROJECT_ROOT / "scripts" / "rewrite_buildai_interpolated_wds.py"),
        "--source_shard_dir",
        str(source_dir),
        "--output_dir",
        str(batch_output_dir),
        "--buildai_processed_root",
        str(processed_root),
        "--shard_start",
        str(int(args.shard_start)),
        "--shard_end",
        str(int(selected_end)),
        "--workers",
        str(int(args.rewrite_workers)),
        "--mano_device",
        str(args.mano_device),
        "--feature_cache_dir",
        str(feature_cache_dir),
        "--source_fps",
        str(float(args.source_fps)),
        "--target_fps",
        str(float(args.target_fps)),
        "--report_out",
        str(report_dir / "rewrite_report.json"),
    ]
    if args.mano_gpus:
        rewrite_cmd.extend(["--mano_gpus", str(args.mano_gpus)])
    if args.mano_dir:
        rewrite_cmd.extend(["--mano_dir", str(args.mano_dir)])
    _append_bool_flag(rewrite_cmd, "--resume", bool(args.rewrite_resume))
    _append_bool_flag(rewrite_cmd, "--interpolate_labels", bool(args.interpolate_labels))
    _run_command(rewrite_cmd, dry_run=bool(args.dry_run))

    verify_sample_clips = _parse_sample_clip_ids(args, ordered_clips, int(args.verify_sample_count))
    for clip_id in verify_sample_clips:
        shard_name = clip_to_shard[clip_id]
        check_cmd = [
            args.python_bin,
            str(PROJECT_ROOT / "scripts" / "check_motion_stage_outputs.py"),
            "--seq_folder",
            clip_to_seq[clip_id],
            "--wds-shard",
            str(batch_output_dir / shard_name),
            "--device",
            str(args.verify_device),
            "--source-fps",
            str(float(args.source_fps)),
            "--target-fps",
            str(float(args.target_fps)),
            "--report_out",
            str(report_dir / f"check_{clip_id}.json"),
        ]
        if args.mano_dir:
            check_cmd.extend(["--mano-dir", str(args.mano_dir)])
        _append_bool_flag(check_cmd, "--interpolate-labels", bool(args.interpolate_labels))
        _run_command(check_cmd, dry_run=bool(args.dry_run))

    visualize_sample_clips = _parse_sample_clip_ids(args, ordered_clips, int(args.visualize_sample_count))
    if vis_dir is not None:
        for clip_id in visualize_sample_clips:
            shard_name = clip_to_shard[clip_id]
            vis_cmd = [
                args.python_bin,
                str(PROJECT_ROOT / "tools" / "ops" / "webdataset_visualizer.py"),
                "--input",
                str(batch_output_dir / shard_name),
                "--output-mode",
                "video",
                "--filter-key",
                clip_id,
                "--episode-limit",
                "1",
                "--render-mode",
                "keypoint",
                "--keypoint-source",
                "lowdim",
                "--video-out",
                str(vis_dir / f"{clip_id}.lowdim.mp4"),
            ]
            _run_command(vis_cmd, dry_run=bool(args.dry_run))

    summary = {
        "batch_id": batch_id,
        "selected_shards": [path.name for path in selected_shards],
        "clip_count": int(len(ordered_clips)),
        "seq_folder_count": int(len(seq_folders)),
        "repair_dense_slam": bool(args.repair_dense_slam),
        "batch_output_dir": str(batch_output_dir),
        "report_dir": str(report_dir),
        "backup_dir": str(backup_dir),
        "verify_sample_clips": verify_sample_clips,
        "visualize_sample_clips": visualize_sample_clips if vis_dir is not None else [],
    }
    (report_dir / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
