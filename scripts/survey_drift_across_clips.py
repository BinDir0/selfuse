#!/usr/bin/env python3
"""Survey DPVO whole-episode drift across MANY clips, to settle whether the user's DOMAIN
(static factory / small translation) actually has a camera-drift problem — on CORRECT-FOCAL
data, judged by the regime-appropriate metric per clip.

Why this and not single-clip diagnose: one clip (and especially the office focal-600 clip)
can't speak for the domain. This runs `diagnose_episode_consistency.py` on a batch, classifies
each clip's motion regime robustly (pure-rotation vs translation-present, via the small-Δ
homog/reproj ratio — independent of the flaky disps-based parallax), picks the VALID drift
metric per regime, and prints an aggregate table + a domain verdict.

It also flags clips that were run with a WRONG/default focal (600 or ~W/2), since those
reintroduce the focal confound and must not be trusted.

Usage (production):
  python scripts/survey_drift_across_clips.py --glob "/root/*.hawor_pipeline/stage_outputs/*"
  python scripts/survey_drift_across_clips.py --seq_folder <A> --seq_folder <B> ...
  # add --rerun to force re-running diagnose even if a summary json already exists
Reads/writes only; calls diagnose_episode_consistency.py as a subprocess per clip.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIAGNOSE = os.path.join(HERE, "diagnose_episode_consistency.py")
SUMMARY_NAME = "episode_consistency_summary.json"


def _has_slam(seq):
    return bool(glob.glob(os.path.join(seq, "SLAM", "hawor_slam_w_scale_*.npz")))


def _focal_is_suspect(focal, img_width):
    """600 default, or within 3% of the egocentric W/2 default => not a real calibration."""
    if abs(focal - 600.0) < 1.0:
        return "default-600"
    if img_width and abs(focal - img_width / 2.0) / max(focal, 1) < 0.03:
        return "W/2-default"
    return None


def run_one(seq, rerun, anchors):
    summ = os.path.join(seq, SUMMARY_NAME)
    if os.path.exists(summ) and not rerun:
        try:
            return json.load(open(summ))
        except Exception:
            pass
    if not _has_slam(seq):
        return {"seq": seq, "error": "no SLAM npz (slam stage not run)"}
    cmd = [sys.executable, DIAGNOSE, "--seq_folder", seq, "--anchors", str(anchors)]
    print(f"  running diagnose on {os.path.basename(seq.rstrip('/'))} ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(summ):
        return {"seq": seq, "error": f"diagnose produced no summary (rc={r.returncode}); tail:\n{r.stdout[-400:]}{r.stderr[-400:]}"}
    return json.load(open(summ))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", help="glob of seq folders, e.g. '/root/*.hawor_pipeline/stage_outputs/*'")
    ap.add_argument("--seq_folder", action="append", default=[], help="explicit seq folder (repeatable)")
    ap.add_argument("--rerun", action="store_true", help="re-run diagnose even if a summary json exists")
    ap.add_argument("--anchors", type=int, default=30)
    ap.add_argument("--width_thresh_pct", type=float, default=7.0,
                    help="domain verdict: a clip is 'drifts' if long-Δ valid-metric error exceeds this %% of image width")
    args = ap.parse_args()

    seqs = list(args.seq_folder)
    if args.glob:
        seqs += [p for p in glob.glob(args.glob) if os.path.isdir(p)]
    seqs = sorted(set(s.rstrip("/") for s in seqs))
    if not seqs:
        sys.exit("no seq folders (pass --glob or --seq_folder)")

    print(f"surveying {len(seqs)} clips ...\n")
    results = [run_one(s, args.rerun, args.anchors) for s in seqs]

    print(f"\n{'clip':<34}{'frames':>7}{'focal':>8}{'regime':>14}{'metric':>8}"
          f"{'short_px':>10}{'long_px':>9}{'%width':>8}{'rise':>7}  flags")
    print("-" * 120)
    ok, drift_clips, suspect_clips = [], [], []
    for r in results:
        name = os.path.basename(r["seq"].rstrip("/"))[:33]
        if "error" in r:
            print(f"{name:<34}  ERROR: {r['error'].splitlines()[0]}")
            continue
        regime = "translation" if r["translation_present"] else "rotation"
        rise = r["drift_long_px"] / max(r["drift_short_px"], 1e-6)
        flag = _focal_is_suspect(r["focal"], r.get("img_width", 0))
        flags = (f"FOCAL={flag}!" if flag else "")
        print(f"{name:<34}{r['n_frames']:>7}{r['focal']:>8.1f}{regime:>14}{r['valid_metric']:>8}"
              f"{r['drift_short_px']:>10.2f}{r['drift_long_px']:>9.2f}{r['drift_long_pct_width']:>8.1f}{rise:>6.1f}x  {flags}")
        if flag:
            suspect_clips.append(name); continue
        ok.append(r)
        if r["drift_long_pct_width"] > args.width_thresh_pct:
            drift_clips.append(name)

    print("\n=== DOMAIN VERDICT ===")
    if suspect_clips:
        print(f"  ⚠ {len(suspect_clips)} clip(s) ran with a WRONG/DEFAULT focal -> EXCLUDED (focal confound): {suspect_clips}")
        print("    Re-run them with the correct --img_focal before trusting their drift.")
    if not ok:
        print("  No correct-focal clips to judge. Provide clips run with the real focal."); return
    n_drift = len(drift_clips)
    print(f"  correct-focal clips judged: {len(ok)}  |  long-Δ valid-metric error > {args.width_thresh_pct}% width: {n_drift}")
    longs = [r["drift_long_pct_width"] for r in ok]
    print(f"  whole-episode drift (valid metric, % image width): "
          f"median={_median(longs):.1f}%  max={max(longs):.1f}%  min={min(longs):.1f}%")
    if n_drift == 0:
        print("  => DOMAIN OK: no correct-focal clip shows meaningful whole-episode drift.")
        print("     Goal C (camera rotation correction) stays SHELVED with real domain evidence.")
    elif n_drift <= len(ok) // 3:
        print(f"  => MOSTLY OK: {n_drift}/{len(ok)} clips drift; inspect those individually (scene/motion outliers?).")
    else:
        print(f"  => DOMAIN DRIFTS: {n_drift}/{len(ok)} clips exceed threshold -> camera drift is REAL for this domain.")
        print("     Reopen Goal C, but with a stronger method than C.0 (e.g. rotation-only bundle adjustment).")


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


if __name__ == "__main__":
    main()
