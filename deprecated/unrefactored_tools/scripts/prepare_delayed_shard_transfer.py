#!/usr/bin/env python3
"""Split shard transfer into contiguous batches and emit delayed launcher scripts."""

from __future__ import annotations

import argparse
import math
import os
import shlex
import subprocess
from pathlib import Path


def _iter_shards(source_dir: Path, start_shard: int | None, end_shard: int | None) -> list[Path]:
    shard_paths = sorted(path for path in source_dir.iterdir() if path.is_file() and path.name.endswith(".tar"))
    if start_shard is None and end_shard is None:
        return shard_paths
    selected: list[Path] = []
    for path in shard_paths:
        stem = path.name[:-4]
        if not stem.startswith("shard-"):
            continue
        shard_idx = int(stem.split("-", 1)[1])
        if start_shard is not None and shard_idx < int(start_shard):
            continue
        if end_shard is not None and shard_idx >= int(end_shard):
            continue
        selected.append(path)
    return selected


def _chunk_contiguous(items: list[Path], chunks: int) -> list[list[Path]]:
    if chunks < 1:
        raise ValueError("chunks must be >= 1")
    if not items:
        return [[] for _ in range(chunks)]
    base = len(items) // chunks
    rem = len(items) % chunks
    out: list[list[Path]] = []
    cursor = 0
    for chunk_idx in range(chunks):
        size = base + (1 if chunk_idx < rem else 0)
        out.append(items[cursor:cursor + size])
        cursor += size
    return out


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _format_script(
    *,
    source_dir: Path,
    dest: str,
    list_file: Path,
    log_file: Path,
    rsync_bin: str,
    extra_rsync_args: list[str],
) -> str:
    source_q = shlex.quote(str(source_dir))
    dest_q = shlex.quote(dest)
    list_q = shlex.quote(str(list_file))
    log_q = shlex.quote(str(log_file))
    rsync_q = shlex.quote(rsync_bin)
    extra = " ".join(shlex.quote(arg) for arg in extra_rsync_args)
    return f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$(dirname {log_q})"
echo "[start] $(date '+%F %T')" | tee -a {log_q}
{rsync_q} {extra} --files-from={list_q} {source_q}/ {dest_q}/ | tee -a {log_q}
echo "[done] $(date '+%F %T')" | tee -a {log_q}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate six contiguous shard-transfer scripts plus a launcher that starts "
            "them after a delay."
        )
    )
    parser.add_argument("--source-dir", required=True, help="Directory containing shard-*.tar files")
    parser.add_argument("--dest", required=True, help="rsync destination, e.g. user@host:/path/to/dir")
    parser.add_argument("--script-dir", required=True, help="Directory where transfer scripts, lists, and logs are written")
    parser.add_argument("--delay-hours", type=float, default=10.0, help="Delay before the launcher starts the transfer scripts")
    parser.add_argument("--chunks", type=int, default=6, help="Number of contiguous transfer chunks to generate")
    parser.add_argument("--start-shard", type=int, default=None, help="Optional inclusive shard index filter")
    parser.add_argument("--end-shard", type=int, default=None, help="Optional exclusive shard index filter")
    parser.add_argument("--rsync-bin", default="rsync", help="rsync executable")
    parser.add_argument(
        "--rsync-arg",
        action="append",
        default=["-av", "--partial", "--append-verify", "--info=progress2"],
        help="Repeatable extra rsync arg. Defaults to a resumable verbose transfer setup.",
    )
    parser.add_argument("--schedule-now", action="store_true", help="Immediately background the delayed launcher")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.delay_hours < 0:
        raise ValueError("--delay-hours must be >= 0")
    if args.chunks < 1:
        raise ValueError("--chunks must be >= 1")

    source_dir = Path(args.source_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source dir not found: {source_dir}")
    script_dir = Path(args.script_dir).resolve()
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir = script_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    shards = _iter_shards(source_dir, args.start_shard, args.end_shard)
    if not shards:
        raise RuntimeError("No shard tar files matched the requested range")

    grouped = _chunk_contiguous(shards, int(args.chunks))
    worker_scripts: list[Path] = []
    summary_lines = []

    for chunk_idx, chunk_paths in enumerate(grouped):
        list_path = script_dir / f"transfer_chunk_{chunk_idx:02d}.list"
        script_path = script_dir / f"transfer_chunk_{chunk_idx:02d}.sh"
        log_path = log_dir / f"transfer_chunk_{chunk_idx:02d}.log"
        relative_names = [path.name for path in chunk_paths]
        _write_text(list_path, "".join(f"{name}\n" for name in relative_names))
        script_text = _format_script(
            source_dir=source_dir,
            dest=str(args.dest),
            list_file=list_path,
            log_file=log_path,
            rsync_bin=str(args.rsync_bin),
            extra_rsync_args=list(args.rsync_arg),
        )
        _write_text(script_path, script_text)
        os.chmod(script_path, 0o755)
        worker_scripts.append(script_path)

        first_name = relative_names[0] if relative_names else None
        last_name = relative_names[-1] if relative_names else None
        summary_lines.append(
            f"chunk={chunk_idx:02d} count={len(relative_names)} first={first_name} last={last_name}"
        )

    delay_seconds = int(math.ceil(float(args.delay_hours) * 3600.0))
    launcher_path = script_dir / "launch_after_delay.sh"
    launcher_log = log_dir / "launch_after_delay.log"
    launcher_body = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(str(log_dir))}",
        f"echo \"[sleep] $(date '+%F %T') waiting {delay_seconds}s\" | tee -a {shlex.quote(str(launcher_log))}",
        f"sleep {delay_seconds}",
        f"echo \"[launch] $(date '+%F %T')\" | tee -a {shlex.quote(str(launcher_log))}",
    ]
    for script_path in worker_scripts:
        chunk_log = log_dir / f"{script_path.stem}.nohup.log"
        launcher_body.append(
            f"nohup bash {shlex.quote(str(script_path))} > {shlex.quote(str(chunk_log))} 2>&1 &"
        )
    launcher_body.append("echo \"[spawned] $(date '+%F %T')\" | tee -a " + shlex.quote(str(launcher_log)))
    _write_text(launcher_path, "\n".join(launcher_body) + "\n")
    os.chmod(launcher_path, 0o755)

    if args.schedule_now:
        subprocess.Popen(
            ["nohup", "bash", str(launcher_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    summary_path = script_dir / "transfer_plan.txt"
    _write_text(summary_path, "\n".join(summary_lines) + "\n")

    print(f"source_dir={source_dir}")
    print(f"dest={args.dest}")
    print(f"selected_shards={len(shards)}")
    print(f"chunks={int(args.chunks)}")
    print(f"launcher={launcher_path}")
    print(f"summary={summary_path}")
    if args.schedule_now:
        print("scheduled_now=true")
    else:
        print(f"run_later=nohup bash {launcher_path} >/dev/null 2>&1 &")


if __name__ == "__main__":
    main()
