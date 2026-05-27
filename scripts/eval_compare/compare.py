"""Stage 3b: evaluate fork vs upstream across all prepared sequences and emit tables.

Walks <work_dir>/<dataset>/<seq>/runs/{fork,orig}/video, scores each against gt.npz,
and writes:
  results/comparison.csv   one row per (dataset, seq, system)
  results/comparison.md    per-sequence rows + per-dataset means + fork-vs-orig deltas

Run on the PRODUCTION machine after prepare_inputs + run_inference.

    python -m scripts.eval_compare.compare --config scripts/eval_compare/config.yaml
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

from scripts.eval_compare.common import load_config
from scripts.eval_compare.evaluate import evaluate

METRIC_COLS = ["ATE", "ATE_S", "RPE_trans", "RPE_rot_deg", "PA_MPJPE", "WA_MPJPE", "W_MPJPE"]


def _resolve_pred_folder(seq_dir: str, system: str) -> str | None:
    """Read the marker written by run_inference; fall back to globbing the run dir."""
    run_dir = os.path.join(seq_dir, "runs", system)
    marker = os.path.join(run_dir, "pred_path.txt")
    if os.path.exists(marker):
        p = open(marker).read().strip()
        if p and os.path.exists(os.path.join(p, "world_space_res.pth")):
            return p
    for wr in sorted(glob.glob(os.path.join(run_dir, "**", "world_space_res.pth"), recursive=True)):
        if glob.glob(os.path.join(os.path.dirname(wr), "SLAM", "hawor_slam_w_scale_*.npz")):
            return os.path.dirname(wr)
    return None


def collect(cfg: dict, use_cuda: bool) -> list[dict]:
    work_dir = cfg["work_dir"]
    rows: list[dict] = []
    for name in cfg["datasets"]:
        for seq_dir in sorted(glob.glob(os.path.join(work_dir, name, "*"))):
            gt_path = os.path.join(seq_dir, "gt.npz")
            if not os.path.exists(gt_path):
                continue
            for system in ("fork", "orig"):
                seq_folder = _resolve_pred_folder(seq_dir, system)
                if seq_folder is None:
                    continue
                try:
                    res = evaluate(gt_path, seq_folder, cfg.get("wa_segment", 100),
                                   cfg.get("rpe_delta", 1), use_cuda=use_cuda)
                except Exception as e:
                    print(f"  eval FAIL {name}/{os.path.basename(seq_dir)}/{system}: {e}")
                    continue
                res["system"] = system
                rows.append(res)
                print(f"  {name}/{os.path.basename(seq_dir)}/{system}: "
                      f"ATE={res['ATE']:.3f} PA={res['PA_MPJPE']:.1f} WA={res['WA_MPJPE']:.1f}")
    return rows


def _fmt(v) -> str:
    return "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def write_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["dataset", "seq_id", "system"] + METRIC_COLS + ["pred_scale", "n_cam_frames"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _mean_by(rows, dataset, system):
    sub = [r for r in rows if r["dataset"] == dataset and r["system"] == system]
    return {c: float(np.nanmean([r.get(c, np.nan) for r in sub])) if sub else float("nan") for c in METRIC_COLS}


def write_markdown(rows: list[dict], cfg: dict, path: str) -> None:
    datasets = list(cfg["datasets"])
    lines = ["# HaWoR fork vs upstream — comparison", "",
             f"Metrics: ATE/ATE-S/RPE_trans (m), RPE_rot (deg), PA/WA/W-MPJPE (mm). "
             f"WA segment = {cfg.get('wa_segment', 100)}, RPE delta = {cfg.get('rpe_delta', 1)}.", ""]

    # per-dataset means
    lines += ["## Per-dataset means", "",
              "| dataset | system | " + " | ".join(METRIC_COLS) + " |",
              "|" + "---|" * (len(METRIC_COLS) + 2)]
    for d in datasets:
        for sysname in ("fork", "orig"):
            m = _mean_by(rows, d, sysname)
            lines.append(f"| {d} | {sysname} | " + " | ".join(_fmt(m[c]) for c in METRIC_COLS) + " |")
        f, o = _mean_by(rows, d, "fork"), _mean_by(rows, d, "orig")
        delta = {c: f[c] - o[c] for c in METRIC_COLS}
        lines.append(f"| **{d}** | **fork-orig** | " + " | ".join(_fmt(delta[c]) for c in METRIC_COLS) + " |")
    lines.append("")

    # per-sequence detail
    lines += ["## Per-sequence", "",
              "| dataset | seq | system | " + " | ".join(METRIC_COLS) + " |",
              "|" + "---|" * (len(METRIC_COLS) + 3)]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['seq_id']} | {r['system']} | "
                     + " | ".join(_fmt(r.get(c)) for c in METRIC_COLS) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rows = collect(cfg, use_cuda=not args.cpu)
    if not rows:
        print("no predictions found — run prepare_inputs + run_inference first")
        return
    out_dir = os.path.join(cfg["work_dir"], "results")
    write_csv(rows, os.path.join(out_dir, "comparison.csv"))
    write_markdown(rows, cfg, os.path.join(out_dir, "comparison.md"))
    print(f"\nwrote {out_dir}/comparison.csv and comparison.md ({len(rows)} rows)")


if __name__ == "__main__":
    main()
