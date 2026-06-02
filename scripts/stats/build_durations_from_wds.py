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
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for _p in (str(PROJECT_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.annotation_protocol import _build_buildai_qwen_annotation_name  # noqa: E402
from language_annotation_stats import _progress  # noqa: E402

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


def _process_index(ip: str):
    """Worker: parse ONE _video_index.json -> (ip, (n_clips, total_frames, rows), err). Each
    index JSON is large and json.load is CPU-bound, so this runs in a process pool; only the
    small (key, frame_count) rows cross back, not the parsed dict."""
    try:
        with open(ip, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception as e:  # noqa: BLE001
        return (ip, None, str(e))
    rows = []
    n_clips = total = 0
    for clip_id, entry in (d.get("videos") or {}).items():
        frames = entry.get("frames")
        fc = len(frames) if isinstance(frames, list) else int(entry.get("num_frames", 0) or 0)
        if fc <= 0:
            continue
        n_clips += 1
        total += fc
        for key in _alt_keys(str(clip_id)):
            rows.append((key, fc))
    return (ip, (n_clips, total, rows), None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="processed_*_jpg dir holding factory*/_video_index.json")
    ap.add_argument("--out", required=True, help="Output CSV (clip_id,frame_count).")
    ap.add_argument("--index_glob", default="*/_video_index.json", help="Glob (under --root) for the index JSONs.")
    ap.add_argument("--workers", type=int, default=0,
                    help="Process-pool workers for parsing the index JSONs (default min(16, cores)). "
                         "Each worker holds one parsed JSON, so lower it if memory is tight.")
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    index_files = sorted(glob.glob(str(root / args.index_glob)))
    if not index_files:
        raise SystemExit(f"no index files at {root}/{args.index_glob}")

    workers = args.workers if args.workers and args.workers > 0 else min(16, (os.cpu_count() or 4))
    rows = []
    n_clips = n_ok = total_frames = 0

    def _consume(ip, res, err):
        nonlocal n_clips, n_ok, total_frames
        if err:
            print(f"[warn] skip {ip}: {err}", file=sys.stderr)
            return
        nc, tf, rws = res
        n_ok += 1
        n_clips += nc
        total_frames += tf
        rows.extend(rws)

    if workers > 1 and len(index_files) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for ip, res, err in _progress(ex.map(_process_index, index_files),
                                          total=len(index_files), desc="index json"):
                _consume(ip, res, err)
    else:
        for ip in _progress(index_files, total=len(index_files), desc="index json"):
            _consume(*_process_index(ip))

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
