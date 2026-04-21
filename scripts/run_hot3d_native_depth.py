#!/usr/bin/env python3
"""Run HOT3D native Any4D depth-only inference from a descriptor manifest."""

from __future__ import annotations

import argparse
import json
import os
import queue as queue_module
import sys
import tarfile
from io import BytesIO
from multiprocessing import get_context
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    class _TqdmFallback:
        def __init__(self, iterable=None, *, total=None, desc=None):
            self.iterable = iterable
            self.total = total
            self.desc = desc

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, _n=1):
            return None

        def set_postfix(self, **_kwargs):
            return None

    def tqdm(iterable=None, *args, total=None, desc=None, **kwargs):
        del args, kwargs
        return _TqdmFallback(iterable=iterable, total=total, desc=desc)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.clip_manifest import load_clip_manifest  # noqa: E402
from lib.pipeline.native_depth import (  # noqa: E402
    NATIVE_DEPTH_DIRNAME,
    NATIVE_DEPTH_FILENAME,
    NATIVE_DEPTH_STAGE_NAME,
    get_native_depth_output_path,
    validate_native_depth_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HOT3D native Any4D depth-only inference")
    parser.add_argument("--descriptor_manifest", required=True, help="HOT3D descriptor manifest JSONL")
    parser.add_argument("--run_dir", default=None, help="Optional pipeline run directory for logs/summary")
    parser.add_argument("--start", type=int, default=0, help="Start clip index in manifest")
    parser.add_argument("--end", type=int, default=None, help="Exclusive end clip index in manifest")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated CUDA device ids; use cpu for CPU mode")
    parser.add_argument("--resume", action="store_true", help="Skip clips whose native depth artifact already validates")
    parser.add_argument(
        "--infer_profile",
        type=str,
        default="standard",
        choices=("standard", "throughput_80gb"),
        help="Compatibility throughput profile. Currently accepted for config parity with batch infer.",
    )
    parser.add_argument(
        "--local_cache_root",
        type=str,
        default=None,
        help="Compatibility local cache root. Accepted for config parity with batch infer.",
    )
    parser.add_argument(
        "--local_cache_quota_gb",
        type=float,
        default=None,
        help="Compatibility local cache quota in GB. Accepted for config parity with batch infer.",
    )
    parser.add_argument(
        "--local_cache_mode",
        type=str,
        default="off",
        choices=("off", "tar", "image_sequence", "all"),
        help="Compatibility local cache mode. Accepted for config parity with batch infer.",
    )
    parser.add_argument(
        "--local_cache_min_frames",
        type=int,
        default=1,
        help="Compatibility local cache minimum frames. Accepted for config parity with batch infer.",
    )
    parser.add_argument("--any4d_batch_size", type=int, default=32, help="Reserved compatibility arg; currently clip-level inference uses all frames per clip")
    parser.add_argument("--any4d_repo_root", type=str, default=None, help="Optional Any4D repo root")
    parser.add_argument("--any4d_checkpoint_path", type=str, default=None, help="Optional Any4D checkpoint path")
    parser.add_argument("--any4d_resolution_set", type=int, default=None, help="Any4D fixed resolution set, e.g. 518")
    parser.add_argument(
        "--any4d_use_amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable AMP during Any4D inference",
    )
    parser.add_argument(
        "--save_dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Depth dtype stored on disk",
    )
    parser.add_argument("--report_out", type=str, default=None, help="Optional JSON summary path")
    return parser


def _parse_devices(raw: str) -> list[str]:
    if isinstance(raw, (list, tuple)):
        devices = [str(item).strip() for item in raw if str(item).strip()]
    else:
        devices = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not devices:
        return ["cuda:0"]
    normalized = []
    for value in devices:
        if value.lower() == "cpu":
            normalized.append("cpu")
        elif value.startswith("cuda:"):
            normalized.append(value)
        else:
            normalized.append(f"cuda:{value}")
    return normalized


def _sample_key_from_image_member(member_name: str) -> str:
    suffix = ".image.jpg"
    if not member_name.endswith(suffix):
        raise ValueError(f"Unexpected HOT3D image member name: {member_name}")
    return member_name[: -len(suffix)]


def _load_npy_from_tar(tar_reader: tarfile.TarFile, member_name: str) -> np.ndarray:
    extracted = tar_reader.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(f"Missing tar member: {member_name}")
    payload = extracted.read()
    return np.load(BytesIO(payload), allow_pickle=False)


def _hot3d_camera_inputs_from_lowdim(lowdim: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lowdim = np.asarray(lowdim, dtype=np.float32).reshape(-1)
    if lowdim.shape != (116,):
        raise ValueError(f"HOT3D lowdim must have shape (116,), got {lowdim.shape}")
    if not np.isfinite(lowdim).all():
        raise ValueError("HOT3D lowdim contains non-finite values")

    world2cam = lowdim[96:112].reshape(4, 4).astype(np.float32)
    intrinsics_4 = lowdim[112:116].astype(np.float32)
    fx, fy = float(intrinsics_4[0]), float(intrinsics_4[1])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Invalid HOT3D intrinsics fx/fy: {intrinsics_4.tolist()}")
    if abs(float(np.linalg.det(world2cam[:3, :3]))) < 1e-8:
        raise ValueError("HOT3D world2cam rotation is singular")
    cam2world = np.linalg.inv(world2cam).astype(np.float32)
    intrinsics_3 = np.array(
        [
            [intrinsics_4[0], 0.0, intrinsics_4[2]],
            [0.0, intrinsics_4[1], intrinsics_4[3]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return intrinsics_4, intrinsics_3, cam2world


def _save_depth_artifact(
    output_path: Path,
    *,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    camera_poses: np.ndarray,
    source_intrinsics: np.ndarray,
    source_world2cam: np.ndarray,
    save_dtype: str,
    resolution_set: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if save_dtype == "float16":
        stored_depths = depths.astype(np.float16)
    else:
        stored_depths = depths.astype(np.float32)
    np.savez(
        output_path,
        depths=stored_depths,
        frame_indices=np.arange(depths.shape[0], dtype=np.int64),
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
        camera_poses=np.asarray(camera_poses, dtype=np.float32),
        source_intrinsics=np.asarray(source_intrinsics, dtype=np.float32),
        source_world2cam=np.asarray(source_world2cam, dtype=np.float32),
        source_extrinsic_convention=np.array("hot3d_lowdim_world2cam"),
        any4d_pose_convention=np.array("opencv_rdf_cam2world"),
        resolution_set=np.array(int(resolution_set), dtype=np.int32),
    )


def _get_done_marker(seq_folder: Path) -> Path:
    return seq_folder / f".{NATIVE_DEPTH_STAGE_NAME}.done"


def _process_record(record, args, runner: dict) -> dict:
    from lib.pipeline.any4d_depth import (
        build_any4d_camera_views_from_image_bytes,
        predict_any4d_depths_from_views,
    )
    from lib.pipeline.frame_sources import build_frame_bytes_reader

    descriptor = record.descriptor
    seq_folder = Path(descriptor.seq_folder)
    done_marker = _get_done_marker(seq_folder)
    output_path = get_native_depth_output_path(seq_folder)

    if args.resume and done_marker.is_file() and output_path.is_file():
        summary = validate_native_depth_output(seq_folder, expected_frame_count=descriptor.frame_count)
        return {
            "clip_id": record.clip_id,
            "seq_folder": str(seq_folder),
            "status": "skipped",
            "summary": summary,
        }

    if descriptor.storage_kind != "tar_shard" or not descriptor.shard_path:
        raise ValueError(f"HOT3D native depth only supports tar_shard descriptors, got {descriptor.storage_kind}")

    reader = build_frame_bytes_reader(descriptor)
    frame_names = list(descriptor.frame_names)
    image_payloads: list[bytes] = []
    source_intrinsics_4 = []
    intrinsics_3 = []
    cam2world_poses = []
    world2cam_poses = []

    with tarfile.open(descriptor.shard_path, "r") as tar_reader:
        for frame_idx, frame_name in enumerate(frame_names):
            sample_key = _sample_key_from_image_member(frame_name)
            lowdim = _load_npy_from_tar(tar_reader, f"{sample_key}.lowdim.npy")
            intrinsics_4, intrinsics_3_matrix, cam2world = _hot3d_camera_inputs_from_lowdim(lowdim)

            image_payloads.append(reader(frame_idx))
            source_intrinsics_4.append(intrinsics_4)
            intrinsics_3.append(intrinsics_3_matrix)
            cam2world_poses.append(cam2world)
            world2cam_poses.append(lowdim[96:112].reshape(4, 4).astype(np.float32))

    views, resized_intrinsics = build_any4d_camera_views_from_image_bytes(
        image_payloads,
        intrinsics_3,
        cam2world_poses,
        runner=runner,
        any4d_repo_root=args.any4d_repo_root,
        checkpoint_path=args.any4d_checkpoint_path,
        resolution_set=args.any4d_resolution_set,
        use_amp=args.any4d_use_amp,
        task="mvs",
    )
    frame_indices = list(range(len(frame_names)))
    depths = predict_any4d_depths_from_views(
        frame_indices,
        views,
        runner=runner,
        any4d_repo_root=args.any4d_repo_root,
        checkpoint_path=args.any4d_checkpoint_path,
        resolution_set=args.any4d_resolution_set,
        use_amp=args.any4d_use_amp,
    )

    try:
        if output_path.exists():
            output_path.unlink()
        _save_depth_artifact(
            output_path,
            depths=np.asarray(depths, dtype=np.float32),
            intrinsics=resized_intrinsics,
            camera_poses=np.stack(cam2world_poses, axis=0),
            source_intrinsics=np.stack(source_intrinsics_4, axis=0),
            source_world2cam=np.stack(world2cam_poses, axis=0),
            save_dtype=args.save_dtype,
            resolution_set=runner["resolution_set"],
        )
        done_marker.parent.mkdir(parents=True, exist_ok=True)
        done_marker.touch()
        summary = validate_native_depth_output(seq_folder, expected_frame_count=descriptor.frame_count)
    except Exception:
        if output_path.exists():
            output_path.unlink()
        if done_marker.exists():
            done_marker.unlink()
        raise

    return {
        "clip_id": record.clip_id,
        "seq_folder": str(seq_folder),
        "status": "success",
        "summary": summary,
    }


def _worker_main(records, args_dict: dict, device: str, queue) -> None:
    from lib.pipeline.any4d_depth import build_any4d_runner

    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        args = argparse.Namespace(**args_dict)
        runner = build_any4d_runner(
            any4d_repo_root=args.any4d_repo_root,
            checkpoint_path=args.any4d_checkpoint_path,
            resolution_set=args.any4d_resolution_set,
            use_amp=args.any4d_use_amp,
            task="mvs",
            device=device,
        )
        for record in records:
            try:
                result = _process_record(record, args, runner)
            except Exception as error:
                result = {
                    "clip_id": record.clip_id,
                    "seq_folder": str(record.descriptor.seq_folder),
                    "status": "failed",
                    "error": str(error),
                }
            queue.put(result)
    finally:
        queue.put({"status": "_worker_done"})


def _build_summary(results: list[dict], manifest_path: Path, devices: list[str], start: int, end: int | None) -> dict:
    success = [item for item in results if item.get("status") == "success"]
    skipped = [item for item in results if item.get("status") == "skipped"]
    failed = [item for item in results if item.get("status") == "failed"]
    return {
        "descriptor_manifest": str(manifest_path.resolve()),
        "devices": devices,
        "start": int(start),
        "end": None if end is None else int(end),
        "total": len(results),
        "success": len(success),
        "skipped": len(skipped),
        "failed": len(failed),
        "failed_examples": failed[:64],
        "output_dirname": NATIVE_DEPTH_DIRNAME,
        "output_filename": NATIVE_DEPTH_FILENAME,
    }


def main() -> None:
    args, unknown = build_parser().parse_known_args()
    if unknown:
        print(f"[native_depth] ignoring unknown args: {unknown}", file=sys.stderr, flush=True)

    manifest_path = Path(args.descriptor_manifest)
    records = load_clip_manifest(manifest_path)
    start = max(0, int(args.start or 0))
    end = len(records) if args.end is None else min(len(records), int(args.end))
    records = records[start:end]
    if not records:
        raise RuntimeError("No HOT3D clips selected for native depth inference")

    devices = _parse_devices(args.gpus)
    if len(devices) == 1:
        assignments = [records]
    else:
        assignments = [[] for _ in devices]
        for index, record in enumerate(records):
            assignments[index % len(devices)].append(record)

    results = []
    if len(devices) == 1:
        from lib.pipeline.any4d_depth import build_any4d_runner

        runner = build_any4d_runner(
            any4d_repo_root=args.any4d_repo_root,
            checkpoint_path=args.any4d_checkpoint_path,
            resolution_set=args.any4d_resolution_set,
            use_amp=args.any4d_use_amp,
            task="mvs",
            device=devices[0],
        )
        for record in tqdm(assignments[0], desc="HOT3D native depth"):
            try:
                results.append(_process_record(record, args, runner))
            except Exception as error:
                results.append(
                    {
                        "clip_id": record.clip_id,
                        "seq_folder": str(record.descriptor.seq_folder),
                        "status": "failed",
                        "error": str(error),
                    }
                )
    else:
        ctx = get_context("spawn")
        queue = ctx.Queue()
        processes = []
        args_dict = vars(args).copy()
        active_workers = 0
        for device, worker_records in zip(devices, assignments):
            if not worker_records:
                continue
            process = ctx.Process(
                target=_worker_main,
                args=(worker_records, args_dict, device, queue),
            )
            process.start()
            processes.append(process)
            active_workers += 1

        with tqdm(total=len(records), desc="HOT3D native depth") as progress:
            finished_workers = 0
            success_count = 0
            skipped_count = 0
            failed_count = 0
            while finished_workers < active_workers:
                try:
                    item = queue.get(timeout=5.0)
                except queue_module.Empty:
                    dead_workers = [process for process in processes if not process.is_alive()]
                    if len(dead_workers) == active_workers:
                        break
                    continue
                if item.get("status") == "_worker_done":
                    finished_workers += 1
                    continue
                results.append(item)
                progress.update(1)
                if item.get("status") == "success":
                    success_count += 1
                elif item.get("status") == "skipped":
                    skipped_count += 1
                elif item.get("status") == "failed":
                    failed_count += 1
                progress.set_postfix(
                    success=success_count,
                    skipped=skipped_count,
                    failed=failed_count,
                )

        for process in processes:
            process.join()
            if process.exitcode not in (0, None):
                results.append(
                    {
                        "clip_id": None,
                        "seq_folder": None,
                        "status": "failed",
                        "error": f"worker exited with code {process.exitcode}",
                    }
                )

    summary = _build_summary(results, manifest_path, devices, start, end)
    report_out = Path(args.report_out) if args.report_out else None
    if report_out is None and args.run_dir:
        report_out = Path(args.run_dir) / "native_depth_summary.json"
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if summary["failed"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
