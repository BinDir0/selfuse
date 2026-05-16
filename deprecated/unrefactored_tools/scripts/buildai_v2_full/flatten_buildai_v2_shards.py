#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flatten nested BuildAI rewritten shard outputs into a single directory."
    )
    parser.add_argument(
        "--input_root",
        required=True,
        help="Nested root containing shard-*.tar under shards_xxxxxx_xxxxxx subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Flat destination dir that will contain shard-XXXXXX.tar only",
    )
    parser.add_argument(
        "--link_mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to materialize the flat shard files",
    )
    parser.add_argument(
        "--prune",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove files in output_dir that are not part of the current flattened result",
    )
    parser.add_argument(
        "--report_out",
        default=None,
        help="Optional JSON report path; defaults to <output_dir>/flatten_summary.json",
    )
    return parser


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _materialize(src: Path, dst: Path, mode: str) -> str:
    if dst.exists():
        if dst.is_file():
            try:
                if src.samefile(dst):
                    return "reused"
            except Exception:
                pass
            dst.unlink()
        else:
            raise RuntimeError(f"Destination path exists and is not a file: {dst}")

    if mode == "hardlink":
        os.link(src, dst)
        return "linked"
    if mode == "symlink":
        dst.symlink_to(src)
        return "symlinked"

    shutil.copy2(src, dst)
    return "copied"


def _choose_unique_candidate(name: str, candidates: list[Path]) -> tuple[Path, dict]:
    if len(candidates) == 1:
        return candidates[0], {
            "shard_name": name,
            "candidate_count": 1,
            "selected": str(candidates[0]),
            "verified_identical": True,
            "duplicates": [str(candidates[0])],
        }

    size_map: dict[int, list[Path]] = {}
    for path in candidates:
        size_map.setdefault(path.stat().st_size, []).append(path)
    if len(size_map) != 1:
        detail = {str(size): [str(path) for path in paths] for size, paths in size_map.items()}
        raise RuntimeError(
            f"Duplicate shard {name} has differing file sizes; refuse to flatten. detail={detail}"
        )

    hash_map: dict[str, list[Path]] = {}
    for path in candidates:
        hash_map.setdefault(_sha256(path), []).append(path)
    if len(hash_map) != 1:
        detail = {digest: [str(path) for path in paths] for digest, paths in hash_map.items()}
        raise RuntimeError(
            f"Duplicate shard {name} has differing file content; refuse to flatten. detail={detail}"
        )

    selected = sorted(candidates, key=lambda path: str(path))[0]
    return selected, {
        "shard_name": name,
        "candidate_count": len(candidates),
        "selected": str(selected),
        "verified_identical": True,
        "duplicates": [str(path) for path in sorted(candidates, key=lambda path: str(path))],
    }


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    grouped: dict[str, list[Path]] = {}
    for path in sorted(input_root.rglob("shard-*.tar")):
        if not path.is_file():
            continue
        grouped.setdefault(path.name, []).append(path.resolve())

    if not grouped:
        raise RuntimeError(f"No shard-*.tar found under {input_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    copied = 0
    symlinked = 0
    reused = 0
    duplicate_groups = []
    staged_names = set()

    for shard_name in sorted(grouped.keys()):
        selected, duplicate_info = _choose_unique_candidate(shard_name, grouped[shard_name])
        if duplicate_info["candidate_count"] > 1:
            duplicate_groups.append(duplicate_info)

        dst = output_dir / shard_name
        result = _materialize(selected, dst, args.link_mode)
        staged_names.add(shard_name)
        if result == "linked":
            linked += 1
        elif result == "copied":
            copied += 1
        elif result == "symlinked":
            symlinked += 1
        else:
            reused += 1

    pruned = 0
    if bool(args.prune):
        for path in output_dir.glob("shard-*.tar"):
            if path.name not in staged_names:
                path.unlink()
                pruned += 1

    report = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "link_mode": args.link_mode,
        "unique_shard_count": len(grouped),
        "duplicate_group_count": len(duplicate_groups),
        "linked": linked,
        "copied": copied,
        "symlinked": symlinked,
        "reused": reused,
        "pruned": pruned,
        "duplicate_groups": duplicate_groups,
    }
    report_path = (
        Path(args.report_out).expanduser().resolve()
        if args.report_out
        else output_dir / "flatten_summary.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
