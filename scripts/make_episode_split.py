#!/usr/bin/env python3
"""
Scan tar shards, collect normalized episode_name values, write train.txt / test.txt.

Full split (train = all \\ random test holdout):
python scripts/make_episode_split.py \\
  --data-path /path/to/shards \\
  --holdout 10 \\
  --out-dir splits/full

Small train for overfit (first N episodes after sort; optional disjoint test from the rest):
python scripts/make_episode_split.py \\
  --data-path /path/to/oakink2_v3 \\
  --train-max 10 \\
  --holdout 5 \\
  --out-dir splits/overfit
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader.utils import discover_shards  # noqa: E402
from dataloader.webdataset import iter_normalized_episode_names_in_shard  # noqa: E402


def collect_episode_names(data_path: str, shard_glob: str) -> list[str]:
    shards = discover_shards(data_path, shard_glob)
    n = len(shards)
    seen: set[str] = set()
    for i, sp in enumerate(shards, start=1):
        print(f"scan [{i}/{n}] {sp}", flush=True)
        for name in iter_normalized_episode_names_in_shard(sp):
            seen.add(name)
    return sorted(seen)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--shard-glob", type=str, default="*.tar")
    p.add_argument(
        "--holdout",
        type=int,
        default=10,
        help="Test episode count: random from full list, or from (all \\ train) when --train-max is set; "
        "with --train-max, 0 copies train into test.",
    )
    p.add_argument(
        "--train-max",
        type=int,
        default=None,
        help="If set, train is only the first N episodes after sorting (overfit smoke); "
        "otherwise train is full list minus holdout test.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, required=True)
    args = p.parse_args()

    names = collect_episode_names(args.data_path, args.shard_glob)
    if not names:
        raise SystemExit("no episodes found")

    rng = random.Random(args.seed)
    names_sorted = sorted(names)

    if args.train_max is not None:
        if args.train_max < 1:
            raise SystemExit("--train-max must be >= 1")
        if args.train_max > len(names_sorted):
            raise SystemExit(f"--train-max={args.train_max} > episodes={len(names_sorted)}")
        train = names_sorted[: args.train_max]
        train_set = set(train)
        remaining = [n for n in names_sorted if n not in train_set]
        if args.holdout == 0:
            test = list(train)
        else:
            if len(remaining) < args.holdout:
                raise SystemExit(
                    f"holdout={args.holdout} but only {len(remaining)} episodes outside train"
                )
            test = sorted(rng.sample(remaining, args.holdout))
    else:
        if args.holdout < 1:
            raise SystemExit("without --train-max, --holdout must be >= 1")
        if len(names_sorted) <= args.holdout:
            raise SystemExit(f"episodes={len(names_sorted)} <= holdout={args.holdout}")
        test_set = set(rng.sample(names_sorted, args.holdout))
        train = [n for n in names_sorted if n not in test_set]
        test = sorted(test_set)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
    print(f"wrote {out/'train.txt'} ({len(train)})  {out/'test.txt'} ({len(test)})")


if __name__ == "__main__":
    main()
