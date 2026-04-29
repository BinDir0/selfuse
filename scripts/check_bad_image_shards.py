#!/usr/bin/env python3
"""Find tar shards containing images that cv2.imdecode cannot decode."""

from __future__ import annotations

import argparse
import itertools
import mmap
import os
import random
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


TAR_BLOCK = 512
ZERO_BLOCK = b"\x00" * TAR_BLOCK
IMAGE_SUFFIXES = (".image.jpg", ".image.jpeg", ".image.png", ".jpg", ".jpeg", ".png")


def _payload_padding(size: int) -> int:
    return (TAR_BLOCK - (size % TAR_BLOCK)) % TAR_BLOCK


def _skip_payload(handle, nbytes: int, file_size: int | None) -> None:
    if nbytes <= 0:
        return
    current = handle.tell()
    target = current + int(nbytes)
    if file_size is not None and target > file_size:
        raise OSError(f"unexpected EOF while skipping payload: need {nbytes} bytes")
    handle.seek(nbytes, os.SEEK_CUR)


def _parse_header(header: bytes) -> tuple[str, int, str]:
    if len(header) != TAR_BLOCK:
        raise ValueError("bad header length")
    size_field = header[124:136].strip(b"\x00 \n")
    size = int(size_field, 8) if size_field else 0
    typeflag = chr(header[156])
    name = header[0:100].split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    if header[257:262] == b"ustar":
        prefix = header[345:500].split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        if prefix:
            name = f"{prefix.rstrip('/')}/{name}".replace("//", "/")
    return name, size, typeflag


def _read_padding(handle, size: int, file_size: int | None) -> None:
    padding = _payload_padding(size)
    if padding:
        _skip_payload(handle, padding, file_size)


def _is_image_member(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(IMAGE_SUFFIXES)


def _decode_image(payload: bytes):
    if not payload:
        return None
    array = np.frombuffer(payload, dtype=np.uint8)
    if array.size == 0:
        return None
    return cv2.imdecode(array, cv2.IMREAD_UNCHANGED)


def check_one_sequential(
    tar_path: str,
    limit_images: int | None = None,
    stop_after_first_bad: bool = True,
) -> tuple[str, list[tuple[str, str]], int]:
    bad: list[tuple[str, str]] = []
    images_checked = 0
    pending_long_name: str | None = None

    try:
        file_size = os.path.getsize(tar_path)
        with open(tar_path, "rb") as handle:
            while True:
                header = handle.read(TAR_BLOCK)
                if len(header) == 0:
                    break
                if len(header) < TAR_BLOCK:
                    return tar_path, [("__tar__", "truncated_header")], images_checked
                if header == ZERO_BLOCK:
                    break

                name, size, typeflag = _parse_header(header)

                if typeflag == "L":
                    payload = handle.read(size)
                    if len(payload) < size:
                        return tar_path, [("__tar__", "truncated_longname")], images_checked
                    pending_long_name = payload.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
                    _read_padding(handle, size, file_size)
                    continue

                if typeflag in ("K", "x", "g"):
                    _skip_payload(handle, size, file_size)
                    _read_padding(handle, size, file_size)
                    continue

                if typeflag in ("0", "\0"):
                    effective_name = pending_long_name if pending_long_name is not None else name
                    pending_long_name = None
                    if _is_image_member(effective_name):
                        payload = handle.read(size)
                        if len(payload) < size:
                            bad.append((effective_name, f"short_read_{len(payload)}_of_{size}"))
                        else:
                            image = _decode_image(payload)
                            if image is None:
                                bad.append((effective_name, "cv2_imdecode_none"))
                        images_checked += 1
                        _read_padding(handle, size, file_size)
                        if bad and stop_after_first_bad:
                            return tar_path, bad, images_checked
                        if limit_images is not None and images_checked >= limit_images:
                            break
                    else:
                        _skip_payload(handle, size, file_size)
                        _read_padding(handle, size, file_size)
                    continue

                _skip_payload(handle, size, file_size)
                _read_padding(handle, size, file_size)
                pending_long_name = None
    except Exception as error:  # noqa: BLE001
        return tar_path, [("__tar__", str(error))], images_checked

    return tar_path, bad, images_checked


def check_one_mmap(
    tar_path: str,
    limit_images: int | None = None,
    stop_after_first_bad: bool = True,
) -> tuple[str, list[tuple[str, str]], int]:
    bad: list[tuple[str, str]] = []
    images_checked = 0
    pending_long_name: str | None = None

    try:
        file_size = os.path.getsize(tar_path)
        if file_size == 0:
            return tar_path, [("__tar__", "empty_file")], images_checked

        with open(tar_path, "rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                offset = 0
                while True:
                    if offset == file_size:
                        break
                    if offset + TAR_BLOCK > file_size:
                        return tar_path, [("__tar__", "truncated_header")], images_checked

                    header = mapped[offset : offset + TAR_BLOCK]
                    offset += TAR_BLOCK
                    if header == ZERO_BLOCK:
                        break

                    name, size, typeflag = _parse_header(header)
                    payload_start = offset
                    payload_end = payload_start + size
                    next_offset = payload_end + _payload_padding(size)
                    if payload_end > file_size:
                        return tar_path, [("__tar__", f"truncated_payload_{name}")], images_checked
                    if next_offset > file_size:
                        return tar_path, [("__tar__", f"truncated_padding_{name}")], images_checked

                    if typeflag == "L":
                        payload = mapped[payload_start:payload_end]
                        pending_long_name = payload.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
                        offset = next_offset
                        continue

                    if typeflag in ("K", "x", "g"):
                        offset = next_offset
                        continue

                    if typeflag in ("0", "\0"):
                        effective_name = pending_long_name if pending_long_name is not None else name
                        pending_long_name = None
                        if _is_image_member(effective_name):
                            if size <= 0:
                                bad.append((effective_name, "empty_payload"))
                            else:
                                payload_view = memoryview(mapped)[payload_start:payload_end]
                                image = None
                                try:
                                    image = _decode_image(payload_view)
                                    if image is None:
                                        bad.append((effective_name, "cv2_imdecode_none"))
                                except Exception as error:  # noqa: BLE001
                                    bad.append((effective_name, f"decode_err:{error}"))
                                finally:
                                    del image
                                    del payload_view
                            images_checked += 1
                            offset = next_offset
                            if bad and stop_after_first_bad:
                                return tar_path, bad, images_checked
                            if limit_images is not None and images_checked >= limit_images:
                                break
                        else:
                            offset = next_offset
                        continue

                    offset = next_offset
                    pending_long_name = None
    except Exception as error:  # noqa: BLE001
        return tar_path, [("__tar__", str(error))], images_checked

    return tar_path, bad, images_checked


def check_one_tarfile(
    tar_path: str,
    limit_images: int | None = None,
    stop_after_first_bad: bool = True,
) -> tuple[str, list[tuple[str, str]], int]:
    bad: list[tuple[str, str]] = []
    images_checked = 0
    try:
        with tarfile.open(tar_path, "r:*") as tar_reader:
            for member in tar_reader:
                if not member.isfile() or not _is_image_member(member.name):
                    continue
                try:
                    member_file = tar_reader.extractfile(member)
                    if member_file is None:
                        bad.append((member.name, "extractfile_none"))
                    else:
                        payload = member_file.read()
                        image = _decode_image(payload)
                        if image is None:
                            bad.append((member.name, "cv2_imdecode_none"))
                except Exception as error:  # noqa: BLE001
                    bad.append((member.name, f"read_err:{error}"))
                images_checked += 1
                if bad and stop_after_first_bad:
                    break
                if limit_images is not None and images_checked >= limit_images:
                    break
    except Exception as error:  # noqa: BLE001
        return tar_path, [("__tar__", str(error))], images_checked
    return tar_path, bad, images_checked


def _default_jobs(executor: str) -> int:
    cpu = os.cpu_count() or 4
    if executor == "thread":
        return min(64, max(16, cpu * 4))
    return min(32, max(4, cpu))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan shard-*.tar files and report shards with images that cv2.imdecode cannot decode."
    )
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("-j", "--jobs", type=int, default=0, help="parallel workers; 0 selects a conservative auto value")
    parser.add_argument(
        "--executor",
        choices=("thread", "process"),
        default="thread",
        help="thread is usually best for I/O plus OpenCV decode",
    )
    parser.add_argument(
        "--engine",
        choices=("mmap", "sequential", "tarfile"),
        default="mmap",
        help="mmap avoids per-image payload read copies; sequential is the seek-based fallback",
    )
    parser.add_argument(
        "--limit-images-per-shard",
        type=int,
        default=None,
        metavar="N",
        help="only verify first N images per shard; this is not a full audit",
    )
    parser.add_argument(
        "--full-detail",
        action="store_true",
        help="scan all image members in a bad shard and record up to --max-issues-per-shard details",
    )
    parser.add_argument(
        "--max-issues-per-shard",
        type=int,
        default=5,
        help="maximum issue detail rows to write per bad shard",
    )
    parser.add_argument("-o", "--out", type=Path, default=Path("bad_image_shards.txt"))
    parser.add_argument("--sample", type=int, default=None, metavar="N", help="only check N random shards")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for --sample")
    parser.add_argument(
        "--start-shard",
        type=int,
        default=0,
        help="start index in sorted shard list, inclusive; useful for multi-machine splits",
    )
    parser.add_argument(
        "--end-shard",
        type=int,
        default=None,
        help="end index in sorted shard list, exclusive; useful for multi-machine splits",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="print progress every N completed shards; 0 selects about 100 progress lines",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=None,
        help="optional file refreshed at each progress print with bad shard details so far",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="chunksize for executor.map; larger values reduce scheduling overhead for many shards",
    )
    parser.add_argument(
        "--opencv-threads",
        type=int,
        default=1,
        help="OpenCV internal thread count per process; outer -j already provides parallelism",
    )
    return parser


def _write_rows(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    os.replace(tmp_path, path)


def main() -> None:
    args = _build_parser().parse_args()
    root: Path = args.root
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr, flush=True)
        sys.exit(1)

    if args.jobs <= 0:
        args.jobs = _default_jobs(args.executor)
    if args.jobs < 1:
        print("--jobs must be >= 1", file=sys.stderr, flush=True)
        sys.exit(1)
    if args.opencv_threads < 0:
        print("--opencv-threads must be >= 0", file=sys.stderr, flush=True)
        sys.exit(1)
    cv2.setNumThreads(int(args.opencv_threads))

    shards = sorted(root.glob("shard-*.tar"))
    total_glob = len(shards)
    if not shards:
        print(f"no shard-*.tar under {root}", file=sys.stderr, flush=True)
        sys.exit(1)

    if args.start_shard < 0:
        print("--start-shard must be >= 0", file=sys.stderr, flush=True)
        sys.exit(1)
    if args.end_shard is not None and args.end_shard < args.start_shard:
        print("--end-shard must be >= --start-shard", file=sys.stderr, flush=True)
        sys.exit(1)
    start = min(int(args.start_shard), total_glob)
    end = total_glob if args.end_shard is None else min(int(args.end_shard), total_glob)
    shards = shards[start:end]
    total_slice = len(shards)
    if not shards:
        print(f"empty shard slice: start={start} end={end} total={total_glob}", file=sys.stderr, flush=True)
        sys.exit(1)

    if args.sample is not None:
        if args.sample <= 0:
            print("--sample N must be positive", file=sys.stderr, flush=True)
            sys.exit(1)
        rng = random.Random(args.seed) if args.seed is not None else random
        shards = rng.sample(shards, k=min(args.sample, total_slice))

    progress_every = args.progress_every
    if progress_every <= 0:
        progress_every = max(1, len(shards) // 100)

    stop_after_first_bad = not bool(args.full_detail)
    if args.engine == "mmap":
        checker = check_one_mmap
    elif args.engine == "sequential":
        checker = check_one_sequential
    else:
        checker = check_one_tarfile
    executor_cls = ThreadPoolExecutor if args.executor == "thread" else ProcessPoolExecutor

    limit_text = f" limit_images/shard={args.limit_images_per_shard}" if args.limit_images_per_shard else ""
    detail_text = "full_detail" if args.full_detail else "stop_after_first_bad"
    print(
        f"glob={total_glob} slice=[{start},{end}) checking={len(shards)} "
        f"executor={args.executor} jobs={args.jobs} engine={args.engine} "
        f"opencv_threads={args.opencv_threads} {detail_text}{limit_text}",
        flush=True,
    )

    bad_rows: list[str] = []
    bad_shard_count = 0
    images_checked_total = 0
    done = 0
    started_at = time.monotonic()

    shard_path_strings = [str(shard_path) for shard_path in shards]
    limit_iter = itertools.repeat(args.limit_images_per_shard)
    stop_iter = itertools.repeat(stop_after_first_bad)
    with executor_cls(max_workers=args.jobs) as executor:
        result_iter = executor.map(
            checker,
            shard_path_strings,
            limit_iter,
            stop_iter,
            chunksize=max(1, int(args.chunksize)),
        )
        for path, issues, images_checked in result_iter:
            done += 1
            images_checked_total += int(images_checked)
            if issues:
                bad_shard_count += 1
                max_issues = max(1, int(args.max_issues_per_shard))
                for name, reason in issues[:max_issues]:
                    bad_rows.append(f"{path}\t{name}\t{reason}")
                if len(issues) > max_issues:
                    bad_rows.append(f"{path}\t...\t(+{len(issues) - max_issues} more)")

            if done % progress_every == 0 or done == len(shards):
                elapsed = max(1e-6, time.monotonic() - started_at)
                rate = done / elapsed
                image_rate = images_checked_total / elapsed
                print(
                    f"progress {done}/{len(shards)} bad_shards={bad_shard_count} "
                    f"images_checked={images_checked_total} rate={rate:.2f} shard/s "
                    f"image_rate={image_rate:.1f} image/s",
                    flush=True,
                )
                if args.checkpoint_out is not None:
                    _write_rows(args.checkpoint_out, bad_rows)

    _write_rows(args.out, bad_rows)
    bad_paths = list(dict.fromkeys(row.split("\t", 1)[0] for row in bad_rows))
    print(
        f"done. bad_shard_count={bad_shard_count} unique_paths={len(bad_paths)} "
        f"images_checked={images_checked_total} detail={args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
