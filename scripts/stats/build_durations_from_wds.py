#!/usr/bin/env python3
"""Build a clip_id -> frame_count table from the Egocentric-100K WDS index files, for
--weight duration in l1_verb_noun_llm.py.

Each factoryNNN/ holds a _video_index.json:
  {"videos": {"<clip_id>": {"shard": "...tar",
                            "frames": [{"name","offset","size"}, ...]}, ...},
   "shards": [...], "num_videos": N, "num_shards": M}
so frame_count per clip = len(videos[clip_id]["frames"]) -- no tar reads, just the index JSONs.
The keys are the canonical buildai id (f001_w055_v00009_i000); we emit BOTH that and the
factory_001_worker_055_0009_cut000 spelling so the join matches whatever the annotation
filenames use.

Usage:
  python scripts/stats/build_durations_from_wds.py \
      --root /efs-exp/guantianrui/datasets/Egocentric-100K/processed_0324_jpg \
      --out ~/language_stat/output_llm/durations.csv --show 8
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.annotation_protocol import _build_buildai_qwen_annotation_name  # noqa: E402

_FACTORY_RE = re.compile(r"^factory_(\d+)_worker_(\d+)_(\d+)_cut(\d+)$")


def _alt_keys(name: str) -> set[str]:
    """Both id spellings for a buildai clip, so the durations join matches either filename form."""
    keys = {name}
    fact = _build_buildai_qwen_annotation_name(name, "")   # fXXX_wYYY_vZZZZ_iNNN -> factory_...
    if fact:
        keys.add(fact)
    m = _FACTORY_RE.match(name)                            # factory_... -> fXXX_wYYY_vZZZZ_iNNN
    if m:
        f, w, v, i = (int(x) for x in m.groups())
        keys.add(f"f{f:03d}_w{w:03d}_v{v:05d}_i{i:03d}")
    return keys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="processed_*_jpg dir holding factory*/_video_index.json")
    ap.add_argument("--out", required=True, help="Output CSV (clip_id,frame_count).")
    ap.add_argument("--index_glob", default="*/_video_index.json", help="Glob (under --root) for the index JSONs.")
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    index_files = sorted(glob.glob(str(root / args.index_glob)))
    if not index_files:
        raise SystemExit(f"no index files at {root}/{args.index_glob}")

    rows = []
    n_clips = n_ok = total_frames = 0
    for ip in index_files:
        try:
            with open(ip, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception as e:
            print(f"[warn] skip {ip}: {e}", file=sys.stderr)
            continue
        n_ok += 1
        for clip_id, entry in (d.get("videos") or {}).items():
            frames = entry.get("frames")
            fc = len(frames) if isinstance(frames, list) else int(entry.get("num_frames", 0) or 0)
            if fc <= 0:
                continue
            n_clips += 1
            total_frames += fc
            for key in _alt_keys(str(clip_id)):
                rows.append((key, fc))

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "frame_count"])
        w.writerows(rows)

    print(f"index files: {len(index_files)} ({n_ok} read)   clips: {n_clips}   "
          f"total frames: {total_frames}   rows (both spellings): {len(rows)}")
    print(f"-> {out_path}")
    print("samples (these clip_ids should match your annotation filenames minus the suffix):")
    for key, fc in rows[: args.show]:
        print(f"  {key},{fc}")


if __name__ == "__main__":
    main()
