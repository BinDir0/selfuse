#!/usr/bin/env python3
"""Plan and run the BuildAI dirty-ablation WDS pipeline.

The pipeline is intentionally stage-based so two machines can split the source
rewrite and trainable build safely:

  1. source dirty injection
  2. trainable build + dirty/error scan
  3. NonFiniteDataError lowdim repair
  4. final validation scan

By default this script prints the exact commands to run. Pass a run-* stage with
--yes to execute a single stage on the current machine.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and optionally run dirty-ablation WDS commands for one dataset.",
    )
    parser.add_argument("--src", required=True, help="Input WDS directory containing shard-*.tar.")
    parser.add_argument("--out-root", default=None, help="Output root. Default: parent directory of --src.")
    parser.add_argument("--name", default=None, help="Stable output name. Default: basename of --src.")
    parser.add_argument(
        "--stage",
        choices=("plan", "run-source", "run-build", "run-nonfinite-repair", "run-final-check"),
        default="plan",
        help="plan prints commands; run-* executes one stage.",
    )
    parser.add_argument("--role", choices=("a", "b"), default=None, help="Shard half for two-machine stages.")
    parser.add_argument("--yes", action="store_true", help="Actually run a run-* stage.")
    parser.add_argument("--python", default=os.environ.get("PYTHON", "python3"), help="Python executable.")
    parser.add_argument("--source-workers", type=int, default=16)
    parser.add_argument("--scan-workers", type=int, default=24)
    parser.add_argument("--repair-workers", type=int, default=8)
    parser.add_argument("--executor", choices=("process", "thread"), default="process")
    parser.add_argument("--dirty-seed", default=None, help="Default: <name>-dirty-v1.")
    parser.add_argument("--instruction-fraction", type=float, default=0.10)
    parser.add_argument("--instruction", default="do something useful")
    parser.add_argument("--scale-fraction", type=float, default=0.10)
    parser.add_argument("--scale-min", type=float, default=0.9)
    parser.add_argument("--scale-max", type=float, default=1.1)
    parser.add_argument("--nonfinite-min", type=float, default=-1.0)
    parser.add_argument("--nonfinite-max", type=float, default=1.0)
    parser.add_argument(
        "--rot6d-mode",
        choices=("repair", "skip", "keep-all"),
        default="skip",
        help=(
            "skip leaves rot6d untouched; repair repairs rot6d; keep-all keeps all rot6d legacy dirty "
            "using the older metadata behavior."
        ),
    )
    parser.add_argument(
        "--keep-legacy-rot6d-fraction",
        type=float,
        default=0.10,
        help="Used only when --rot6d-mode repair.",
    )
    parser.add_argument("--dirty-source-dir", default=None)
    parser.add_argument("--trainable-dir", default=None)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--scan-progress-interval", type=int, default=100)
    parser.add_argument("--skip-final-scan", action="store_true")
    return parser.parse_args()


def shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def run_command(args: list[str], *, cwd: Path = PROJECT_ROOT) -> None:
    print("+ " + shell_join(args), flush=True)
    subprocess.run(args, cwd=str(cwd), check=True)


def shard_count(path: Path) -> int:
    return len(sorted(path.glob("shard-*.tar")))


def role_range(total: int, role: str) -> tuple[int, int | None]:
    mid = (total + 1) // 2
    if role == "a":
        return 0, mid
    if role == "b":
        return mid, None
    raise ValueError(f"unknown role: {role}")


def require_role(args: argparse.Namespace) -> str:
    if args.role is None:
        raise SystemExit(f"--role a|b is required for --stage {args.stage}")
    return str(args.role)


def validate_common(args: argparse.Namespace, src: Path, dirty_source: Path, trainable: Path) -> None:
    if not src.is_dir():
        raise SystemExit(f"--src not found or not a directory: {src}")
    if shard_count(src) == 0:
        raise SystemExit(f"--src contains no shard-*.tar files: {src}")
    for output in (dirty_source, trainable):
        try:
            output.resolve().relative_to(src.resolve())
        except ValueError:
            continue
        raise SystemExit(f"output directory must not be nested inside --src: {output}")


def paths_from_args(args: argparse.Namespace) -> dict[str, Path | str]:
    src = Path(args.src).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else src.parent
    name = args.name or src.name
    dirty_source = Path(args.dirty_source_dir).resolve() if args.dirty_source_dir else out_root / f"{name}.dirty"
    trainable = Path(args.trainable_dir).resolve() if args.trainable_dir else out_root / f"{name}.mixed_trainable"
    work_root = Path(args.work_root).resolve() if args.work_root else out_root / f"{name}.dirty_pipeline.work"
    dirty_seed = args.dirty_seed or f"{name}-dirty-v1"
    validate_common(args, src, dirty_source, trainable)
    return {
        "src": src,
        "name": name,
        "dirty_source": dirty_source,
        "trainable": trainable,
        "work_root": work_root,
        "dirty_seed": dirty_seed,
    }


def work_dir(work_root: Path, role: str) -> Path:
    return work_root / f"work.{role}"


def source_command(args: argparse.Namespace, paths: dict[str, Path | str], role: str) -> list[str]:
    src = paths["src"]
    dirty_source = paths["dirty_source"]
    work = work_dir(paths["work_root"], role)
    total = shard_count(src)
    start, end = role_range(total, role)
    rot6d_mode = str(args.rot6d_mode)
    keep_fraction = 0.0
    extra_rot6d: list[str] = []
    if rot6d_mode == "repair":
        keep_fraction = float(args.keep_legacy_rot6d_fraction)
    elif rot6d_mode == "skip":
        extra_rot6d = ["--rot6d-mode", "skip"]
    elif rot6d_mode == "keep-all":
        keep_fraction = 1.0

    cmd = [
        args.python,
        str(SCRIPT_DIR / "repair_legacy_rot6d_wds.py"),
        "--source-shard-dir",
        str(src),
        "--output-dir",
        str(dirty_source),
        "--workers",
        str(args.source_workers),
        "--executor",
        str(args.executor),
        "--dirty-seed",
        str(paths["dirty_seed"]),
        "--keep-legacy-rot6d-episode-fraction",
        str(keep_fraction),
        "--dirty-instruction-episode-fraction",
        str(args.instruction_fraction),
        "--dirty-instruction-mode",
        "generic",
        "--generic-instruction",
        str(args.instruction),
        "--dirty-state-action-scale-episode-fraction",
        str(args.scale_fraction),
        "--dirty-state-action-scale-min",
        str(args.scale_min),
        "--dirty-state-action-scale-max",
        str(args.scale_max),
        "--progress-out",
        str(work / "source_dirty_progress.jsonl"),
        "--report-out",
        str(work / "source_dirty_report.json"),
        "--shard-start",
        str(start),
        "--no-resume",
    ]
    if end is not None:
        cmd.extend(["--shard-end", str(end)])
    cmd.extend(extra_rot6d)
    return cmd


def build_command(args: argparse.Namespace, paths: dict[str, Path | str], role: str) -> list[str]:
    dirty_source = paths["dirty_source"]
    trainable = paths["trainable"]
    work = work_dir(paths["work_root"], role)
    total = shard_count(dirty_source)
    if total == 0:
        raise SystemExit(f"dirty source has no shards yet: {dirty_source}")
    start, end = role_range(total, role)
    cmd = [
        "bash",
        str(SCRIPT_DIR / "build_mixed_dirty_trainable_wds.sh"),
        "--src",
        str(dirty_source),
        "--dst",
        str(trainable),
        "--work",
        str(work),
        "--workers",
        str(args.scan_workers),
        "--shard-start",
        str(start),
        "--scan-progress-interval",
        str(args.scan_progress_interval),
        "--no-check-media",
    ]
    if end is not None:
        cmd.extend(["--shard-end", str(end)])
    return cmd


def nonfinite_command(args: argparse.Namespace, paths: dict[str, Path | str], role: str) -> list[str]:
    work = work_dir(paths["work_root"], role)
    return [
        args.python,
        str(SCRIPT_DIR / "repair_nonfinite_lowdim_wds.py"),
        "--bad-keys",
        str(work / "bad_keys.jsonl"),
        "--dst-dir",
        str(paths["trainable"]),
        "--report",
        str(work / "repair_nonfinite_lowdim_report.json"),
        "--seed",
        str(paths["dirty_seed"]),
        "--replacement-min",
        str(args.nonfinite_min),
        "--replacement-max",
        str(args.nonfinite_max),
        "--overwrite",
        "--workers",
        str(args.repair_workers),
        "--executor",
        str(args.executor),
    ]


def final_scan_command(args: argparse.Namespace, paths: dict[str, Path | str]) -> list[str]:
    work = work_dir(paths["work_root"], "final")
    return [
        args.python,
        str(SCRIPT_DIR / "filter_and_check_datasets.py"),
        "wds",
        "--shards",
        str(paths["trainable"] / "shard-*.tar"),
        "--workers",
        str(args.scan_workers),
        "--bad-keys-output",
        str(work / "bad_keys.final_trainable_post_repair.jsonl"),
        "--report",
        str(work / "filter_summary.final_trainable_post_repair.json"),
    ]


def print_plan(args: argparse.Namespace, paths: dict[str, Path | str]) -> None:
    src = paths["src"]
    dirty_source = paths["dirty_source"]
    print("# Dirty ablation WDS pipeline plan")
    print(f"export DIRTY_PIPELINE_SRC={shlex.quote(str(src))}")
    print(f"export DIRTY_PIPELINE_DIRTY_SOURCE={shlex.quote(str(dirty_source))}")
    print(f"export DIRTY_PIPELINE_TRAINABLE={shlex.quote(str(paths['trainable']))}")
    print(f"export DIRTY_PIPELINE_WORK_ROOT={shlex.quote(str(paths['work_root']))}")
    print(f"# shards: source={shard_count(src)} dirty_source={shard_count(dirty_source) if dirty_source.exists() else 0}")
    print()
    for role in ("a", "b"):
        print(f"# Machine {role.upper()} source dirty injection")
        print(shell_join([str(args.python), str(PROJECT_ROOT / "scripts" / "run_dirty_ablation_wds_pipeline.py"), *base_invocation(args), "--stage", "run-source", "--role", role, "--yes"]))
        print()
    print("# After both source stages finish, run build on both machines")
    for role in ("a", "b"):
        print(f"# Machine {role.upper()} build trainable")
        print(shell_join([str(args.python), str(PROJECT_ROOT / "scripts" / "run_dirty_ablation_wds_pipeline.py"), *base_invocation(args), "--stage", "run-build", "--role", role, "--yes"]))
        print()
    print("# After both build stages finish, repair nonfinite on both machines")
    for role in ("a", "b"):
        print(f"# Machine {role.upper()} nonfinite repair")
        print(shell_join([str(args.python), str(PROJECT_ROOT / "scripts" / "run_dirty_ablation_wds_pipeline.py"), *base_invocation(args), "--stage", "run-nonfinite-repair", "--role", role, "--yes"]))
        print()
    if not args.skip_final_scan:
        print("# Final validation on one machine")
        print(shell_join([str(args.python), str(PROJECT_ROOT / "scripts" / "run_dirty_ablation_wds_pipeline.py"), *base_invocation(args), "--stage", "run-final-check", "--yes"]))


def base_invocation(args: argparse.Namespace) -> list[str]:
    items = [
        "--src",
        args.src,
        "--source-workers",
        str(args.source_workers),
        "--scan-workers",
        str(args.scan_workers),
        "--repair-workers",
        str(args.repair_workers),
        "--executor",
        str(args.executor),
        "--instruction-fraction",
        str(args.instruction_fraction),
        "--instruction",
        str(args.instruction),
        "--scale-fraction",
        str(args.scale_fraction),
        "--scale-min",
        str(args.scale_min),
        "--scale-max",
        str(args.scale_max),
        "--nonfinite-min",
        str(args.nonfinite_min),
        "--nonfinite-max",
        str(args.nonfinite_max),
        "--rot6d-mode",
        str(args.rot6d_mode),
        "--keep-legacy-rot6d-fraction",
        str(args.keep_legacy_rot6d_fraction),
        "--scan-progress-interval",
        str(args.scan_progress_interval),
        "--python",
        str(args.python),
    ]
    for flag, value in (
        ("--out-root", args.out_root),
        ("--name", args.name),
        ("--dirty-seed", args.dirty_seed),
        ("--dirty-source-dir", args.dirty_source_dir),
        ("--trainable-dir", args.trainable_dir),
        ("--work-root", args.work_root),
    ):
        if value is not None:
            items.extend([flag, str(value)])
    if args.skip_final_scan:
        items.append("--skip-final-scan")
    return items


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_source_done(paths: dict[str, Path | str]) -> None:
    src_count = shard_count(paths["src"])
    dirty_count = shard_count(paths["dirty_source"])
    if src_count != dirty_count:
        raise SystemExit(f"dirty source shard count mismatch: source={src_count} dirty={dirty_count}")


def assert_role_source_done(paths: dict[str, Path | str], role: str) -> None:
    src = paths["src"]
    dirty_source = paths["dirty_source"]
    source_shards = sorted(src.glob("shard-*.tar"))
    start, end = role_range(len(source_shards), role)
    missing = [dirty_source / shard.name for shard in source_shards[start:end] if not (dirty_source / shard.name).exists()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise SystemExit(f"dirty source is missing {len(missing)} shard(s) for role {role}:\n{preview}")


def assert_build_done(paths: dict[str, Path | str]) -> None:
    dirty_count = shard_count(paths["dirty_source"])
    trainable_count = shard_count(paths["trainable"])
    if dirty_count != trainable_count:
        raise SystemExit(f"trainable shard count mismatch: dirty={dirty_count} trainable={trainable_count}")


def assert_nonfinite_report(paths: dict[str, Path | str], role: str) -> None:
    report_path = work_dir(paths["work_root"], role) / "repair_nonfinite_lowdim_report.json"
    report = load_json(report_path)
    missing = int(report["sample_keys_missing_in_shard"])
    if missing:
        raise SystemExit(f"{report_path}: sample_keys_missing_in_shard={missing}")


def assert_final_report(paths: dict[str, Path | str]) -> None:
    report_path = work_dir(paths["work_root"], "final") / "filter_summary.final_trainable_post_repair.json"
    report = load_json(report_path)
    reasons = report.get("reason_counts", {})
    if "NonFiniteDataError" in reasons:
        raise SystemExit(f"{report_path}: NonFiniteDataError still present: {reasons['NonFiniteDataError']}")
    print(json.dumps({"samples_total": report["samples_total"], "reason_counts": reasons}, ensure_ascii=False, indent=2))


def execute_stage(args: argparse.Namespace, paths: dict[str, Path | str]) -> None:
    if args.stage == "plan":
        print_plan(args, paths)
        return
    if not args.yes:
        raise SystemExit(f"--stage {args.stage} requires --yes to execute. Omit --stage or use --stage plan to print commands.")
    if args.stage == "run-source":
        role = require_role(args)
        work_dir(paths["work_root"], role).mkdir(parents=True, exist_ok=True)
        run_command(source_command(args, paths, role))
        return
    if args.stage == "run-build":
        role = require_role(args)
        assert_role_source_done(paths, role)
        work_dir(paths["work_root"], role).mkdir(parents=True, exist_ok=True)
        run_command(build_command(args, paths, role))
        return
    if args.stage == "run-nonfinite-repair":
        assert_build_done(paths)
        role = require_role(args)
        run_command(nonfinite_command(args, paths, role))
        assert_nonfinite_report(paths, role)
        return
    if args.stage == "run-final-check":
        assert_build_done(paths)
        work_dir(paths["work_root"], "final").mkdir(parents=True, exist_ok=True)
        run_command(final_scan_command(args, paths))
        assert_final_report(paths)
        return
    raise AssertionError(args.stage)


def main() -> None:
    args = parse_args()
    paths = paths_from_args(args)
    execute_stage(args, paths)


if __name__ == "__main__":
    main()
