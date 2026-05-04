#!/usr/bin/env python3
"""Sample-check WebDataset shards.

This is a lightweight wrapper around ``data/filter_and_check_datasets.py wds``.
It samples a small number of shards from every configured VLA/VLM shard pattern
so checks are not biased toward the first files in a large dataset.

Two modes:
  --config  (default)  – read shard patterns from a Hydra experiment config.
  --shard-url          – check arbitrary shard glob patterns, no config needed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
FILTER_SCRIPT = Path(__file__).resolve().with_name("filter_and_check_datasets.py")
if not FILTER_SCRIPT.exists():
    FILTER_SCRIPT = REPO_ROOT / "data" / "filter_and_check_datasets.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample-check all configured VLA/VLM WebDataset paths."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Hydra experiment config path. Required unless --shard-url is given.",
    )
    parser.add_argument(
        "--shard-url",
        nargs="+",
        default=None,
        help="Arbitrary shard glob patterns or paths to check (bypasses --config).",
    )
    parser.add_argument(
        "--kind",
        choices=("vla", "vlm"),
        default="vla",
        help="Dataset kind for --shard-url mode (default: vla).",
    )
    parser.add_argument(
        "--dataset-name",
        default="manual",
        help="Label used in output filenames for --shard-url mode (default: manual).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: outputs/dataset_sample_check/<timestamp>.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("train-vla", "val-vla", "train-vlm", "val-vlm", "all"),
        default=["train-vla", "train-vlm"],
        help="Dataset groups to sample-check.",
    )
    parser.add_argument(
        "--max-shards-per-pattern",
        type=int,
        default=2,
        help="Random shards sampled from each configured shard pattern.",
    )
    parser.add_argument(
        "--max-samples-per-pattern",
        type=int,
        default=1000,
        help="Max samples scanned for each pattern. 0 scans sampled shards fully.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pattern-workers",
        type=int,
        default=1,
        help=(
            "Number of configured shard-pattern checks to run concurrently. "
            "This is the outer parallelism level."
        ),
    )
    parser.add_argument(
        "--shard-workers",
        type=int,
        default=1,
        help=(
            "Number of shard-level workers passed to filter_and_check_datasets.py. "
            "This is the inner parallelism level."
        ),
    )
    parser.add_argument(
        "--check-media",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode media and run training-compatible media checks.",
    )
    parser.add_argument(
        "--vlm-check-image-quality",
        action="store_true",
        help=(
            "Also apply brightness/contrast image-quality checks to VLM images. "
            "Training currently applies this quality gate to VLA images; VLM quality "
            "checking is stricter and may need manual review."
        ),
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run checks. Without this, only writes the sampled plan and commands.",
    )
    return parser.parse_args()


def load_config(config_path: str | Path):
    try:
        import hydra
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise SystemExit(
            "This script needs hydra-core and omegaconf in the training environment."
        ) from exc

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config_dir = config_path.parent.parent.resolve()
    config_name = f"{config_path.parent.name}/{config_path.stem}"

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "now", lambda fmt: datetime.now().strftime(fmt), replace=True,
    )
    OmegaConf.register_new_resolver("hydra", lambda _path: "", replace=True)

    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name=config_name)
    OmegaConf.set_struct(cfg, False)
    cfg.hydra = {"runtime": {"output_dir": "outputs", "choices": {}}, "job": {"num": 0, "name": "sample_check"}}

    # Hydra registers its own ``hydra:`` resolver that reads HydraConfig,
    # which is not populated when this utility composes configs outside
    # @hydra.main. Replace it after compose so logging/default.yaml fields like
    # ${hydra:runtime.choices.experiment} can resolve in this standalone script.
    def hydra_resolver(path: str):
        if path == "runtime.choices.experiment":
            return config_path.stem
        if path == "runtime.output_dir":
            return "outputs"
        if path == "job.num":
            return 0
        if path == "job.name":
            return "sample_check"
        return ""

    OmegaConf.register_new_resolver("hydra", hydra_resolver, replace=True)
    OmegaConf.resolve(cfg)
    return cfg


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [os.fspath(value)]
    return [os.fspath(item) for item in value]


def expand_pattern(pattern: str) -> list[str]:
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches
    if not any(ch in pattern for ch in "*?["):
        return [pattern]
    return []


def sanitize_name(value: str) -> str:
    keep = []
    for ch in value:
        keep.append(ch if ch.isalnum() or ch in "._-" else "_")
    return "".join(keep).strip("_") or "dataset"


def group_specs(cfg, selected_groups: set[str]) -> list[dict[str, Any]]:
    if "all" in selected_groups:
        selected_groups = {"train-vla", "val-vla", "train-vlm", "val-vlm"}

    specs = []
    mapping = [
        ("train-vla", "vla", "vla_wds_datasets"),
        ("val-vla", "vla", "val_vla_wds_datasets"),
        ("train-vlm", "vlm", "vlm_wds_datasets"),
        ("val-vlm", "vlm", "val_vlm_wds_datasets"),
    ]
    for group_name, kind, cfg_key in mapping:
        if group_name not in selected_groups:
            continue
        for entry in cfg.get(cfg_key, []):
            patterns = as_list(entry.shard_urls)
            for pattern_index, pattern in enumerate(patterns):
                specs.append(
                    {
                        "group": group_name,
                        "kind": kind,
                        "dataset_name": str(entry.get("name", "unknown")),
                        "pattern_index": pattern_index,
                        "pattern": pattern,
                    }
                )
    return specs


def build_check_command(
    *,
    spec: dict[str, Any],
    sampled_shards: list[str],
    output_dir: Path,
    max_samples: int,
    check_media: bool,
    vlm_check_image_quality: bool,
    target_image_size: list[int] | None,
    shard_workers: int,
) -> list[str]:
    stem = sanitize_name(
        f"{spec['group']}__{spec['dataset_name']}__p{spec['pattern_index']}"
    )
    cmd = [
        sys.executable,
        str(FILTER_SCRIPT),
        "wds",
        "--kind",
        spec["kind"],
        "--shards",
        *sampled_shards,
        "--report",
        str(output_dir / f"{stem}.report.json"),
        "--bad-keys-output",
        str(output_dir / f"{stem}.bad_keys.jsonl"),
        "--good-keys-output",
        str(output_dir / f"{stem}.good_keys.jsonl"),
    ]
    if shard_workers > 1:
        cmd.extend(["--workers", str(shard_workers)])
    if max_samples > 0:
        cmd.extend(["--max-samples", str(max_samples)])
    if check_media:
        cmd.append("--check-media")
    if spec["kind"] == "vlm" and vlm_check_image_quality:
        cmd.append("--check-image-quality")
    if spec["kind"] == "vlm" and target_image_size:
        cmd.extend(["--target-image-size", str(target_image_size[0]), str(target_image_size[1])])
    return cmd


def run_check_item(item: dict[str, Any]) -> dict[str, Any]:
    cmd = item.get("_cmd_list")
    if not cmd:
        item["returncode"] = None
        return item

    label = f"{item['group']} {item['dataset_name']} p{item['pattern_index']}"
    started = time.time()
    print(f"[RUN] {label}: {item['sampled_count']} shards", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    item["returncode"] = result.returncode
    item["elapsed_sec"] = time.time() - started
    status = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
    print(f"[DONE] {label}: {status} in {item['elapsed_sec']:.1f}s", flush=True)
    return item


def main() -> None:
    args = parse_args()
    if args.max_shards_per_pattern < 1:
        raise SystemExit("--max-shards-per-pattern must be >= 1")
    if args.max_samples_per_pattern < 0:
        raise SystemExit("--max-samples-per-pattern must be >= 0")
    if args.pattern_workers < 1:
        raise SystemExit("--pattern-workers must be >= 1")
    if args.shard_workers < 1:
        raise SystemExit("--shard-workers must be >= 1")
    if args.shard_workers > 1 and args.max_samples_per_pattern > 0:
        raise SystemExit(
            "--shard-workers > 1 requires --max-samples-per-pattern 0 because "
            "the underlying shard-level worker mode scans whole sampled shards."
        )
    if not args.shard_url and not args.config:
        raise SystemExit("Either --config or --shard-url is required.")

    cfg = None
    target_image_size = None
    if args.shard_url:
        # Direct mode: build specs from --shard-url, no config needed.
        specs = [
            {
                "group": "manual",
                "kind": args.kind,
                "dataset_name": args.dataset_name,
                "pattern_index": idx,
                "pattern": pattern,
            }
            for idx, pattern in enumerate(args.shard_url)
        ]
    else:
        cfg = load_config(args.config)
        if cfg.get("data") is not None and cfg.data.get("target_image_size") is not None:
            target_image_size = [int(x) for x in cfg.data.target_image_size]
        specs = group_specs(cfg, set(args.groups))

    rng = random.Random(args.seed)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "outputs" / "dataset_sample_check" / time.strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    plan: dict[str, Any] = {
        "config": str(args.config) if args.config else None,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "max_shards_per_pattern": args.max_shards_per_pattern,
        "max_samples_per_pattern": args.max_samples_per_pattern,
        "pattern_workers": args.pattern_workers,
        "shard_workers": args.shard_workers,
        "check_media": bool(args.check_media),
        "vlm_check_image_quality": bool(args.vlm_check_image_quality),
        "run": bool(args.run),
        "items": [],
    }

    for spec in specs:
        matches = expand_pattern(spec["pattern"])
        sampled = sorted(rng.sample(matches, min(len(matches), args.max_shards_per_pattern))) if matches else []
        item = dict(spec)
        item["available_shards"] = len(matches)
        item["sampled_shards"] = sampled
        item["sampled_count"] = len(sampled)
        if sampled:
            cmd = build_check_command(
                spec=spec,
                sampled_shards=sampled,
                output_dir=output_dir,
                max_samples=args.max_samples_per_pattern,
                check_media=args.check_media,
                vlm_check_image_quality=args.vlm_check_image_quality,
                target_image_size=target_image_size,
                shard_workers=args.shard_workers,
            )
            item["_cmd_list"] = cmd
            item["command"] = " ".join(shlex.quote(part) for part in cmd)
        else:
            item["_cmd_list"] = None
            item["command"] = None
            item["returncode"] = None
            print(f"[WARN] no shards matched: {spec['pattern']}", flush=True)
        plan["items"].append(item)

    if args.run:
        runnable = [item for item in plan["items"] if item.get("_cmd_list")]
        if args.pattern_workers == 1:
            completed = [run_check_item(item) for item in runnable]
        else:
            completed = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.pattern_workers) as executor:
                futures = [executor.submit(run_check_item, item) for item in runnable]
                for future in concurrent.futures.as_completed(futures):
                    completed.append(future.result())
            completed_by_command = {item["command"]: item for item in completed}
            for index, item in enumerate(plan["items"]):
                command = item.get("command")
                if command in completed_by_command:
                    plan["items"][index] = completed_by_command[command]

    for item in plan["items"]:
        item.pop("_cmd_list", None)

    plan_path = output_dir / "sample_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    commands_path = output_dir / "commands.sh"
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
    ]
    commands.extend(item["command"] for item in plan["items"] if item.get("command"))
    commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    commands_path.chmod(0o755)

    print(f"Plan: {plan_path}")
    print(f"Commands: {commands_path}")
    if not args.run:
        print("Dry plan only. Re-run with --run or execute commands.sh to perform checks.")

    failed = [item for item in plan["items"] if item.get("returncode")]
    if failed:
        raise SystemExit(f"{len(failed)} sampled checks failed; see {plan_path}")


if __name__ == "__main__":
    main()
