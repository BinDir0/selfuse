#!/usr/bin/env python3
"""Upsample dirty samples in a WebDataset dir by duplicating them into new shards.

Intended use in the dirty-data ablation pipeline:

  1. Build ``<DST_DIRTY>`` via ``drop_bad_wds_frames.py`` so only unloadable
     samples are removed (dirty samples remain).
  2. Run this script with the dirty-only JSONL produced by
     ``split_bad_keys_by_reason.py``. It appends new ``shard-dup-*.tar`` files
     to ``<DST_DIRTY>`` that contain ``multiplier - 1`` additional copies of
     every dirty sample. Tar member names are suffixed with ``__dupK`` so the
     copies are distinct WebDataset samples.

Original shards are not modified. If --multiplier=1, the script is a no-op
besides writing the report.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else ()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drop_bad_wds_frames import load_bad_keys


DUP_SHARD_PREFIX = "shard-dup-"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Duplicate dirty samples into new shards inside an existing WDS directory.",
    )
    parser.add_argument("--dst-dir", required=True, help="WDS directory holding the error-dropped shards.")
    parser.add_argument("--dirty-keys", required=True, help="dirty_only JSONL from split_bad_keys_by_reason.py.")
    parser.add_argument("--multiplier", type=int, default=2, help="Replication factor (N=1 means no-op).")
    parser.add_argument(
        "--allow-large-multiplier",
        action="store_true",
        help="Required if --multiplier is greater than 3.",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=1000,
        help="Target samples per output dup shard (one 'sample' = one duplicated copy).",
    )
    parser.add_argument("--report", default=None, help="Optional JSON report output path.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing shard-dup-*.tar before writing.")
    return parser.parse_args()


def resolve_source_shard(recorded_shard: str, dst_dir: Path) -> Path:
    # Prefer the shard copy inside dst_dir; it is the WDS we are extending. Fall
    # back to the original path the scan recorded, since drop_bad_wds_frames only
    # rewrites shards that had error samples.
    candidate = dst_dir / Path(recorded_shard).name
    if candidate.exists():
        return candidate
    original = Path(recorded_shard)
    if original.exists():
        return original
    raise FileNotFoundError(
        f"shard not found in dst-dir or at original path: {recorded_shard}"
    )


def dup_member_names(sample_key: str, dup_index: int) -> tuple[str, str]:
    return f"{sample_key}.", f"{sample_key}__dup{dup_index}."


def collect_dirty_members(src_shard: Path, wanted_keys: set[str]) -> list[tuple[tarfile.TarInfo, bytes, str]]:
    out: list[tuple[tarfile.TarInfo, bytes, str]] = []
    prefix_to_key = {key + ".": key for key in wanted_keys}
    with tarfile.open(src_shard, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            owning_key: str | None = None
            if name in wanted_keys:
                owning_key = name
            else:
                dot = name.find(".")
                if dot > 0:
                    owning_key = prefix_to_key.get(name[: dot + 1])
            if owning_key is None:
                continue
            file_obj = tar.extractfile(member)
            if file_obj is None:
                raise ValueError(f"Failed to extract tar member: {src_shard}:{name}")
            out.append((member, file_obj.read(), owning_key))
    return out


def write_dup_member(tar_writer: tarfile.TarFile, member: tarfile.TarInfo, payload: bytes, new_name: str) -> None:
    info = tarfile.TarInfo(name=new_name)
    info.size = len(payload)
    info.mode = member.mode
    info.mtime = member.mtime
    info.uid = member.uid
    info.gid = member.gid
    info.uname = member.uname
    info.gname = member.gname
    tar_writer.addfile(info, io.BytesIO(payload))


def next_dup_shard(dst_dir: Path, shard_idx: int) -> Path:
    return dst_dir / f"{DUP_SHARD_PREFIX}{shard_idx:05d}.tar"


def main() -> None:
    args = parse_args()
    multiplier = int(args.multiplier)
    if multiplier < 1:
        raise SystemExit("--multiplier must be >= 1")
    if multiplier > 3 and not args.allow_large_multiplier:
        raise SystemExit(
            f"--multiplier={multiplier} exceeds the safe default of 3. "
            "Re-run with --allow-large-multiplier if you really mean to inflate "
            "dirty samples this aggressively."
        )
    if multiplier > 3:
        print(
            f"WARNING: --multiplier={multiplier} will {multiplier}x dirty samples; "
            "this may overwhelm clean signal and compromise experiment honesty.",
            file=sys.stderr,
        )

    dst_dir = Path(args.dst_dir)
    if not dst_dir.is_dir():
        raise SystemExit(f"--dst-dir not found: {dst_dir}")

    existing_dup = sorted(dst_dir.glob(f"{DUP_SHARD_PREFIX}*.tar"))
    if existing_dup and not args.overwrite:
        raise SystemExit(
            f"{len(existing_dup)} existing shard-dup-*.tar in {dst_dir}. "
            "Pass --overwrite to replace them."
        )
    for path in existing_dup:
        path.unlink()

    keys_by_shard = load_bad_keys(args.dirty_keys)
    total_dirty_keys = sum(len(keys) for keys in keys_by_shard.values())
    if multiplier == 1 or total_dirty_keys == 0:
        print(
            f"No-op: multiplier={multiplier}, dirty_keys={total_dirty_keys}. "
            "Nothing to duplicate.",
        )
        _write_report(args, multiplier, keys_by_shard, 0, 0, [], time.time())
        return

    shard_idx = 0
    current_tar: tarfile.TarFile | None = None
    current_path: Path | None = None
    current_samples = 0
    produced_shards: list[dict] = []
    total_samples_written = 0
    total_members_written = 0
    started = time.time()

    def close_current() -> None:
        nonlocal current_tar, current_path, current_samples
        if current_tar is None:
            return
        current_tar.close()
        produced_shards.append(
            {
                "path": str(current_path),
                "samples": current_samples,
            }
        )
        current_tar = None
        current_path = None
        current_samples = 0

    def open_next() -> None:
        nonlocal current_tar, current_path, shard_idx
        current_path = next_dup_shard(dst_dir, shard_idx)
        shard_idx += 1
        current_tar = tarfile.open(current_path, "w")

    open_next()
    shard_iter = tqdm(
        sorted(keys_by_shard.keys()),
        desc="Duplicating dirty",
        unit="shard",
    )
    for recorded_shard in shard_iter:
        wanted_keys = keys_by_shard[recorded_shard]
        src_shard = resolve_source_shard(recorded_shard, dst_dir)
        members = collect_dirty_members(src_shard, wanted_keys)
        if not members:
            tqdm.write(
                f"WARN: no dirty members found in {src_shard} "
                f"(expected {len(wanted_keys)} keys)",
            )
            continue
        members_by_key: dict[str, list[tuple[tarfile.TarInfo, bytes]]] = defaultdict(list)
        for member, payload, owning_key in members:
            members_by_key[owning_key].append((member, payload))

        for sample_key, member_list in members_by_key.items():
            for dup_index in range(1, multiplier):
                old_prefix, new_prefix = dup_member_names(sample_key, dup_index)
                if current_samples >= args.samples_per_shard:
                    close_current()
                    open_next()
                for member, payload in member_list:
                    new_name = new_prefix + member.name[len(old_prefix):]
                    write_dup_member(current_tar, member, payload, new_name)
                    total_members_written += 1
                current_samples += 1
                total_samples_written += 1
        if hasattr(shard_iter, "set_postfix"):
            shard_iter.set_postfix(
                samples=total_samples_written,
                dup_shards=len(produced_shards) + (1 if current_tar is not None else 0),
            )

    close_current()

    elapsed = time.time() - started
    print(
        f"Done. shards_written={len(produced_shards)} "
        f"samples_written={total_samples_written} "
        f"members_written={total_members_written} "
        f"elapsed={elapsed:.1f}s",
    )
    _write_report(args, multiplier, keys_by_shard, total_samples_written, total_members_written, produced_shards, started)


def _write_report(
    args: argparse.Namespace,
    multiplier: int,
    keys_by_shard: dict[str, set[str]],
    total_samples_written: int,
    total_members_written: int,
    produced_shards: list[dict],
    started: float,
) -> None:
    if not args.report:
        return
    report = {
        "dst_dir": str(Path(args.dst_dir)),
        "dirty_keys": str(Path(args.dirty_keys)),
        "multiplier": multiplier,
        "samples_per_shard": int(args.samples_per_shard),
        "dirty_source_shards": len(keys_by_shard),
        "dirty_keys_total": sum(len(keys) for keys in keys_by_shard.values()),
        "samples_written": total_samples_written,
        "members_written": total_members_written,
        "produced_shards": produced_shards,
        "elapsed_sec": time.time() - started,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
