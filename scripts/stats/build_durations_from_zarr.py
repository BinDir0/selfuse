#!/usr/bin/env python3
"""Build a clip_id -> frame_count table from a zarr root, for --weight duration in
l1_verb_noun_llm.py.

A zarr root stores meta/episode_names + meta/episode_ends (cumulative frame index), so the
per-episode frame count is just episode_ends[i] - episode_ends[i-1] -- no per-frame work, it
reads two small metadata arrays. frame_count is proportional to duration when fps is constant
(use it directly as the weight).

Join key: l1_verb_noun_llm.py keys clips by the annotation filename minus its suffix. That may
be the canonical buildai id (f001_w018_v00140_i000) OR the qwen-annotation spelling
(factory_001_worker_018_0140_cut000). To be safe we emit BOTH spellings per episode (same
count), so the join matches either way.

Usage:
    python scripts/stats/build_durations_from_zarr.py --zarr /path/to/root.zarr \
        --out ~/language_stat/output_llm/durations.csv --show 8

Then weight the stats with it:
    python scripts/stats/l1_verb_noun_llm.py ... --weight duration \
        --durations ~/language_stat/output_llm/durations.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.annotation_protocol import _build_buildai_qwen_annotation_name  # noqa: E402

_FACTORY_RE = re.compile(r"^factory_(\d+)_worker_(\d+)_(\d+)_cut(\d+)$")


def _zarr_open(path: str):
    try:
        import zarr
    except ImportError as error:
        raise SystemExit("zarr is not installed:  pip install zarr") from error
    return zarr.open(path, mode="r")


def _decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _group_get(root, path: str):
    node = root
    for part in path.split("/"):
        node = node[part]
    return node


def _alt_keys(name: str) -> set[str]:
    """Both id spellings for a buildai episode, so the join matches either filename form."""
    keys = {name}
    fact = _build_buildai_qwen_annotation_name(name, "")  # fXXX_wYYY_vZZZZ_iNNN -> factory_...
    if fact:
        keys.add(fact)
    m = _FACTORY_RE.match(name)                           # factory_... -> fXXX_wYYY_vZZZZ_iNNN
    if m:
        f, w, v, i = (int(x) for x in m.groups())
        keys.add(f"f{f:03d}_w{w:03d}_v{v:05d}_i{i:03d}")
    return keys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zarr", required=True, help="Path to the zarr root directory.")
    ap.add_argument("--out", required=True, help="Output CSV (clip_id,frame_count).")
    ap.add_argument("--show", type=int, default=8, help="Print this many sample rows for sanity.")
    args = ap.parse_args(argv)

    root = _zarr_open(args.zarr)
    names = [_decode_text(x) for x in _group_get(root, "meta/episode_names")[:].tolist()]
    ends = np.asarray(_group_get(root, "meta/episode_ends")[:], dtype=np.int64).reshape(-1)
    if len(names) != len(ends):
        raise SystemExit(f"episode_names ({len(names)}) != episode_ends ({len(ends)})")

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    total_frames = 0
    for i, name in enumerate(names):
        start = 0 if i == 0 else int(ends[i - 1])
        frames = int(ends[i]) - start
        if frames <= 0:
            continue
        total_frames += frames
        for key in _alt_keys(name):
            rows.append((key, frames))

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "frame_count"])
        w.writerows(rows)

    print(f"episodes: {len(names)}   total frames: {total_frames}   rows written (both spellings): {len(rows)}")
    print(f"-> {out_path}")
    print("samples (verify these clip_ids match your annotation filenames minus the suffix):")
    for key, frames in rows[: args.show]:
        print(f"  {key},{frames}")


if __name__ == "__main__":
    main()
