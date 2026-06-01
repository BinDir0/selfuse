#!/usr/bin/env python3
"""Dataset feature statistics over exported WebDataset shards (116-d lowdim).

One streaming pass over the WebDataset tar shards computes the data-derived
dataset numbers for the paper:

  #15 scale       : episodes (clips), frames, hours, presence coverage
  #17 hand-content: bimanual vs single-hand ratio, left/right presence, wrist
                    motion (world-frame step / path length / workspace span),
                    hand camera-depth range
  #18 camera      : camera ego-motion (translation / rotation step, trajectory
                    length), intrinsics (fx/fy) and approximate FOV

Reads the same WebDataset the trainer consumes; lowdim layout is taken from
lib/pipeline/quality_metrics (single source of truth). Per-dataset = one --input
root each; an aggregate over all roots is also reported. Scales to billions of
frames via reservoir sampling (exact count/mean/min/max + sampled quantiles).

Run on the PRODUCTION machine (pure CPU, reads tar shards; no GPU). Example:

    python scripts/stats/dataset_feature_stats.py \
        --input /path/buildai_wds /path/hot3d_wds ... \
        --out_dir ~/stats/feature_stats --fps 30

Assumptions (flagged; adjust if wrong):
  * fps is NOT stored in the shards -> --fps (default 30); needed for "hours".
  * wrist translation is WORLD-frame ("mano_joint_0_world" in the lowdim schema),
    so wrist motion is reported in world meters.
  * camera extrinsic is 4x4 world->camera (w2c); camera center = -R^T t.
  * FOV is approximated as 2*atan(cx/fx) (assumes principal point ~ image center;
    image width is not decoded).
  * frames of an episode are contiguous in shard order (the exporter writes them
    so); episodes are flushed when episode_index changes.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import tarfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.quality_metrics import (  # noqa: E402
    EXTRINSIC_SLICE,
    INTRINSIC_SLICE,
    LEFT_HAND_TRANSLATION_SLICE,
    LOWDIM_SIZE,
    RIGHT_HAND_TRANSLATION_SLICE,
)

METRIC_KEYS = (
    "wrist_step_m",        # per-frame world-frame wrist displacement (m)
    "wrist_depth_m",       # hand camera-frame depth z (m)
    "cam_trans_step_m",    # per-frame camera-center displacement (m)
    "cam_rot_step_frob",   # per-frame rotation-matrix Frobenius delta
    "fx",                  # focal length (px, decoded-frame units)
    "fov_x_deg",           # approx horizontal FOV (deg)
    "ep_path_length_m",    # per-episode wrist path length (m)
    "ep_hand_span_m",      # per-episode wrist workspace bbox diagonal (m)
    "ep_traj_length_m",    # per-episode camera trajectory length (m)
)


class Reservoir:
    """Streaming reservoir sample + exact count/mean/min/max for one scalar metric."""

    def __init__(self, k: int, seed: int):
        self.k = int(k)
        self.rng = np.random.default_rng(seed)
        self.count = 0
        self.sum = 0.0
        self.min = math.inf
        self.max = -math.inf
        self._keys = np.empty(0, dtype=np.float64)
        self._vals = np.empty(0, dtype=np.float64)

    def add_array(self, arr) -> None:
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        self.count += int(arr.size)
        self.sum += float(arr.sum())
        self.min = min(self.min, float(arr.min()))
        self.max = max(self.max, float(arr.max()))
        keys = self.rng.random(arr.size)
        keys = np.concatenate([self._keys, keys])
        vals = np.concatenate([self._vals, arr])
        if keys.size > self.k:  # keep the k items with smallest random keys (uniform sample)
            idx = np.argpartition(keys, self.k)[: self.k]
            keys, vals = keys[idx], vals[idx]
        self._keys, self._vals = keys, vals

    def summary(self) -> dict | None:
        if self.count == 0:
            return None
        s = np.sort(self._vals)

        def pct(p):
            return float(np.interp(p / 100.0 * (s.size - 1), np.arange(s.size), s)) if s.size else 0.0

        return {
            "count": self.count,
            "mean": self.sum / self.count,
            "min": self.min,
            "p25": pct(25), "p50": pct(50), "p75": pct(75), "p95": pct(95),
            "max": self.max,
            "sampled": int(s.size),
        }


class Stats:
    """Counts + per-metric reservoirs for one root (or the aggregate)."""

    def __init__(self, reservoir: int, seed: int):
        self.counts = dict(episodes=0, frames=0, present=0, both=0, left=0, right=0)
        self.metrics = {k: Reservoir(reservoir, seed + i) for i, k in enumerate(METRIC_KEYS)}

    def report(self, fps: float) -> dict:
        c = self.counts
        present = max(1, c["present"])
        return {
            "counts": {**c, "hours": c["frames"] / fps / 3600.0},
            "ratios": {
                "presence_coverage": c["present"] / max(1, c["frames"]),
                "bimanual": c["both"] / present,
                "left_only": (c["left"] - c["both"]) / present,
                "right_only": (c["right"] - c["both"]) / present,
            },
            "metrics": {k: self.metrics[k].summary() for k in METRIC_KEYS},
        }


_LOWDIM_SUFFIX = ".lowdim.npy"
_META_SUFFIX = ".meta.json"


def iter_frames(root: Path):
    """Yield (lowdim ndarray[116], meta dict) per frame, in shard order."""
    shards = sorted(root.rglob("shard-*.tar"))
    for shard in shards:
        with tarfile.open(shard, "r") as tf:
            pending: dict[str, dict] = {}
            for member in tf:
                name = member.name
                if name.endswith(_LOWDIM_SUFFIX):
                    key, field = name[: -len(_LOWDIM_SUFFIX)], "lowdim"
                elif name.endswith(_META_SUFFIX):
                    key, field = name[: -len(_META_SUFFIX)], "meta"
                else:
                    continue
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                slot = pending.setdefault(key, {})
                slot[field] = fobj.read()
                if "lowdim" in slot and "meta" in slot:
                    lo = np.load(io.BytesIO(slot["lowdim"]))
                    me = json.loads(slot["meta"])
                    del pending[key]
                    yield np.asarray(lo, dtype=np.float64).reshape(-1), me


def _flush_episode(buf: list[dict], stats_list: list[Stats], frame_step: int) -> None:
    """Compute episode-level + per-frame metrics for one episode; fold into all stats."""
    if not buf:
        return
    n_full = len(buf)
    pres_full = np.array([b["presence"] for b in buf], dtype=np.int64)
    left_full = np.isin(pres_full, (1, 3))
    right_full = np.isin(pres_full, (2, 3))

    for st in stats_list:
        st.counts["episodes"] += 1
        st.counts["frames"] += n_full
        st.counts["present"] += int((pres_full > 0).sum())
        st.counts["both"] += int((pres_full == 3).sum())
        st.counts["left"] += int(left_full.sum())
        st.counts["right"] += int(right_full.sum())

    # Subsample frames for the distribution metrics.
    sel = buf[:: max(1, frame_step)]
    if not sel:
        return
    wl = np.stack([b["wl"] for b in sel])          # (n,3) world left wrist
    wr = np.stack([b["wr"] for b in sel])          # (n,3) world right wrist
    R = np.stack([b["R"] for b in sel])            # (n,3,3) w2c rotation
    t = np.stack([b["t"] for b in sel])            # (n,3) w2c translation
    pres = np.array([b["presence"] for b in sel], dtype=np.int64)
    fx = np.array([b["fx"] for b in sel]); fy = np.array([b["fy"] for b in sel])
    cx = np.array([b["cx"] for b in sel])
    lp = np.isin(pres, (1, 3)); rp = np.isin(pres, (2, 3))

    # camera centers C = -R^T t  (w2c)
    C = -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t)

    def feed(key, arr):
        for st in stats_list:
            st.metrics[key].add_array(arr)

    # --- intrinsics / FOV (#18) ---
    feed("fx", fx)
    fov = 2.0 * np.degrees(np.arctan2(np.abs(cx), np.maximum(fx, 1e-6)))
    feed("fov_x_deg", fov)

    # --- hand camera depth (#17): z of present wrist in camera frame ---
    def cam_z(world):  # (n,3) world -> z in camera
        return np.einsum("nij,nj->ni", R, world)[:, 2] + t[:, 2]
    feed("wrist_depth_m", cam_z(wl)[lp])
    feed("wrist_depth_m", cam_z(wr)[rp])

    # --- camera ego-motion (#18) ---
    if C.shape[0] >= 2:
        feed("cam_trans_step_m", np.linalg.norm(np.diff(C, axis=0), axis=1))
        feed("cam_rot_step_frob", np.linalg.norm(np.diff(R, axis=0).reshape(R.shape[0] - 1, -1), axis=1))
        feed("ep_traj_length_m", np.array([np.linalg.norm(np.diff(C, axis=0), axis=1).sum()]))

    # --- wrist motion (#17): per-side, only between frames where that hand is present ---
    for world, present in ((wl, lp), (wr, rp)):
        if present.sum() < 2:
            continue
        pts = world[present]
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        feed("wrist_step_m", steps)
        feed("ep_path_length_m", np.array([steps.sum()]))
        span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        feed("ep_hand_span_m", np.array([span]))


def run_root(root: Path, stats_root: Stats, stats_all: Stats, frame_step: int, progress) -> None:
    cur_ep = None
    buf: list[dict] = []
    targets = [stats_root, stats_all]
    n = 0
    for lowdim, meta in iter_frames(root):
        if lowdim.shape[0] != LOWDIM_SIZE:
            continue
        ep = meta.get("episode_index")
        if ep != cur_ep and buf:
            _flush_episode(buf, targets, frame_step)
            buf = []
        cur_ep = ep
        mat = lowdim[EXTRINSIC_SLICE].reshape(4, 4)
        intr = lowdim[INTRINSIC_SLICE]
        buf.append({
            "wl": lowdim[LEFT_HAND_TRANSLATION_SLICE],
            "wr": lowdim[RIGHT_HAND_TRANSLATION_SLICE],
            "R": mat[:3, :3], "t": mat[:3, 3],
            "fx": float(intr[0]), "fy": float(intr[1]), "cx": float(intr[2]),
            "presence": int(meta.get("presence", 0)),
        })
        n += 1
        if progress is not None:
            progress(n)
    if buf:
        _flush_episode(buf, targets, frame_step)


def _make_progress(desc):
    try:
        from tqdm import tqdm  # type: ignore
        bar = tqdm(desc=desc, unit="frame", dynamic_ncols=True)
        return lambda n: bar.update(1), bar.close
    except Exception:
        import time
        t0 = time.time()
        def upd(n):
            if n % 20000 == 0:
                print(f"\r[{desc}] {n} frames ({n/(time.time()-t0+1e-9):.0f}/s)", end="", file=sys.stderr, flush=True)
        return upd, lambda: print("", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dataset feature statistics over WebDataset lowdim.")
    ap.add_argument("--input", nargs="+", required=True, help="One or more WebDataset roots (one per dataset).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels matching --input order (default: dir name).")
    ap.add_argument("--fps", type=float, default=30.0, help="Assumed fps for the 'hours' figure.")
    ap.add_argument("--frame_step", type=int, default=1, help="Subsample every Nth frame for distribution metrics.")
    ap.add_argument("--reservoir", type=int, default=200000, help="Reservoir sample size per metric.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    roots = [Path(p).expanduser().resolve() for p in args.input]
    labels = args.labels if args.labels and len(args.labels) == len(roots) else [r.name for r in roots]
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_all = Stats(args.reservoir, args.seed)
    per_root = {}
    for root, label in zip(roots, labels):
        if not root.is_dir():
            print(f"[warn] skip missing root: {root}", file=sys.stderr)
            continue
        print(f"[root] {label}: {root}", file=sys.stderr, flush=True)
        st = Stats(args.reservoir, args.seed)
        upd, done = _make_progress(f"{label}")
        run_root(root, st, stats_all, args.frame_step, upd)
        done()
        per_root[label] = st.report(args.fps)
        print(f"  episodes={st.counts['episodes']} frames={st.counts['frames']} "
              f"hours={per_root[label]['counts']['hours']:.1f}", file=sys.stderr)

    report = {
        "fps": args.fps,
        "frame_step": args.frame_step,
        "wrist_frame": "world (mano_joint_0_world)",
        "extrinsic_convention": "w2c; camera center = -R^T t",
        "fov_approx": "2*atan(cx/fx)",
        "per_root": per_root,
        "combined": stats_all.report(args.fps),
    }
    out_path = out_dir / "dataset_feature_stats.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    c = stats_all.counts
    print("=" * 64)
    print(f"datasets : {len(per_root)}   episodes: {c['episodes']}   frames: {c['frames']}   "
          f"hours: {report['combined']['counts']['hours']:.1f}")
    print(f"bimanual : {report['combined']['ratios']['bimanual']:.1%}   "
          f"presence-cov: {report['combined']['ratios']['presence_coverage']:.1%}")
    ws = report["combined"]["metrics"]["wrist_step_m"]
    if ws:
        print(f"wrist step (m/frame): mean {ws['mean']:.4f}  p50 {ws['p50']:.4f}  p95 {ws['p95']:.4f}")
    print(f"-> {out_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
