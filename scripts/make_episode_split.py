#!/usr/bin/env python3
"""
扫描tar，列出 normalize 后的 episode_name，并划分 train.txt / test.txt。
Usage:
python scripts/make_episode_split.py \
  --data-path /share_data/zhangtingrui/datasets/taco_v2 \
  --holdout 10 \
  --out-dir /share_data/jixinhao/EgoTransformer/splits
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
    p.add_argument("--holdout", type=int, default=10, help="测试集 episode 数量")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, required=True)
    args = p.parse_args()

    names = collect_episode_names(args.data_path, args.shard_glob)
    if len(names) <= args.holdout:
        raise SystemExit(f"episodes={len(names)} <= holdout={args.holdout}")

    rng = random.Random(args.seed)
    test_set = set(rng.sample(names, args.holdout))
    train = [n for n in names if n not in test_set]
    test = sorted(test_set)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
    print(f"wrote {out/'train.txt'} ({len(train)})  {out/'test.txt'} ({len(test)})")


if __name__ == "__main__":
    main()
