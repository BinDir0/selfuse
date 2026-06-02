#!/usr/bin/env python3
"""Diagnose the FACTORY coverage of the parsed annotation records (and of a --sample_frac sample).

If the LLM task stats look like one domain only (e.g. all textile), this checks whether the input
records actually span all factories, or are concentrated -- i.e. whether the random clip sample is
really spread across factories. random.sample is uniform, so a concentrated sample means `records`
itself is concentrated (partial cache / rglob), not the sampling.

Usage:
  python scripts/stats/diagnose_factory_distribution.py \
      --annotation_root /efs-exp/guantianrui/buildai_6000_anno/ \
      --suffix _qwen-annotation.json --parse_workers 64 --sample_frac 0.1
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for _p in (str(PROJECT_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from language_annotation_stats import default_annotation_cache, load_annotations  # noqa: E402

_FAC = re.compile(r"(?:^f|^factory_?)(\d+)")


def factory_of(clip_id: str) -> str:
    m = _FAC.match(clip_id)
    return f"factory{int(m.group(1)):03d}" if m else f"??({clip_id[:12]})"


def _show(name, facs: Counter, n_clips: int):
    print(f"\n{name}: {n_clips} clips   distinct factories: {len(facs)}")
    top = facs.most_common()
    print("  top 15:")
    for f, c in top[:15]:
        print(f"    {f}: {c}  ({100*c/max(1,n_clips):.1f}%)")
    if len(top) > 15:
        print("  bottom 5:")
        for f, c in top[-5:]:
            print(f"    {f}: {c}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotation_root", required=True)
    ap.add_argument("--suffix", default="_qwen-annotation.json")
    ap.add_argument("--parse_workers", type=int, default=64)
    ap.add_argument("--sample_frac", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--rebuild_cache", action="store_true")
    args = ap.parse_args()

    root = Path(args.annotation_root).expanduser().resolve()
    cache = None if args.no_cache else (args.cache or default_annotation_cache(root, args.suffix))
    records, _cov = load_annotations(root, args.suffix, args.parse_workers,
                                     cache=cache, rebuild_cache=args.rebuild_cache)

    facs = Counter(factory_of(r["clip_id"]) for r in records)
    _show("ALL records", facs, len(records))
    print(f"\n  sample clip_id: {records[0]['clip_id']!r}")

    if args.sample_frac:
        k = max(1, int(round(len(records) * args.sample_frac)))
        sample = random.Random(args.seed).sample(records, k)
        _show(f"SAMPLE (frac={args.sample_frac}, seed={args.seed})",
              Counter(factory_of(r["clip_id"]) for r in sample), len(sample))


if __name__ == "__main__":
    main()
