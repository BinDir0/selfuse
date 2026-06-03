#!/usr/bin/env python3
"""Nested donut (two-ring pie) of dataset size.

  inner ring  = FRAMES per category  (≈ duration share)
  outer ring  = EPISODES per category (clip-count share)

Same categories share a colour across both rings, so you can compare "share of duration" vs
"share of clips" at a glance (e.g. a category with many but short clips = big outer / small inner).

Fill the real numbers in DATA below, OR pass --csv with columns `label,frames,episodes`.

    python scripts/stats/plot_dataset_pie.py --out dataset_pie.png --fps 30
    python scripts/stats/plot_dataset_pie.py --csv my_counts.csv --out dataset_pie.png
"""

from __future__ import annotations

import argparse
import csv as csvmod
from pathlib import Path

# ----------------------------- FILL REAL DATA HERE -----------------------------
# (label, frames, episodes) -- placeholders; replace with the real counts.
DATA = [
    ("Textile / garment", 240_000_000, 700_000),
    ("Electronics",        90_000_000, 380_000),
    ("Packaging",          70_000_000, 300_000),
    ("Assembly",           60_000_000, 250_000),
    ("Soldering / wiring",  40_000_000, 180_000),
    ("Other",             140_000_000, 160_000),
]
# -------------------------------------------------------------------------------

# Clean, modern categorical palette (Tableau/Vega-style).
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
           "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"]


def _load_csv(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        r = csvmod.reader(fh)
        next(r, None)  # header
        for row in r:
            if len(row) >= 3:
                try:
                    rows.append((str(row[0]), float(row[1]), float(row[2])))
                except ValueError:
                    continue
    return rows


def _contrast(hex_color: str) -> str:
    """Black or white text for readability on a given wedge colour."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 145 else "#222222"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None, help="CSV `label,frames,episodes` (overrides inline DATA).")
    ap.add_argument("--out", default="dataset_pie.png")
    ap.add_argument("--fps", type=float, default=30.0, help="For the total-hours figure in the center.")
    ap.add_argument("--title", default="Dataset composition")
    ap.add_argument("--min_pct", type=float, default=3.0, help="Hide percentage labels below this %.")
    ap.add_argument("--width", type=float, default=0.34, help="Ring thickness.")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _load_csv(args.csv) if args.csv else DATA
    if not data:
        raise SystemExit("no data (check --csv or the inline DATA)")
    labels = [d[0] for d in data]
    frames = [float(d[1]) for d in data]
    eps = [float(d[2]) for d in data]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(data))]
    w = args.width

    def autopct(minp):
        return lambda p: f"{p:.0f}%" if p >= minp else ""

    fig, ax = plt.subplots(figsize=(9.5, 9))

    # OUTER ring = episodes
    _, _, a_out = ax.pie(
        eps, radius=1.0, colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(width=w, edgecolor="white", linewidth=2),
        autopct=autopct(args.min_pct), pctdistance=1 - w / 2,
        textprops=dict(fontsize=9.5, weight="bold"))
    # INNER ring = frames (duration)
    _, _, a_in = ax.pie(
        frames, radius=1.0 - w, colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(width=w, edgecolor="white", linewidth=2),
        autopct=autopct(args.min_pct), pctdistance=1 - w / (2 * (1.0 - w)),
        textprops=dict(fontsize=8.5))

    for t, c in zip(a_out, colors):
        t.set_color(_contrast(c))
    for t, c in zip(a_in, colors):
        t.set_color(_contrast(c))

    total_h = sum(frames) / args.fps / 3600.0
    ax.text(0, 0, f"{total_h:,.0f} h\n{int(sum(eps)):,}\nepisodes",
            ha="center", va="center", fontsize=13, weight="bold", linespacing=1.3)

    ax.set(aspect="equal")
    ax.set_title(args.title, fontsize=16, weight="bold", pad=16)
    ax.legend(labels, title="Category", loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=10.5, title_fontsize=11)
    ax.text(0, -1.16, "outer ring: episodes   ·   inner ring: frames (duration)",
            ha="center", fontsize=10, color="#555555")

    out = Path(args.out).expanduser()
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"-> {out.resolve()}")
    print(f"   categories: {len(data)}   total frames: {int(sum(frames)):,} (~{total_h:,.0f} h)   "
          f"total episodes: {int(sum(eps)):,}")


if __name__ == "__main__":
    main()
