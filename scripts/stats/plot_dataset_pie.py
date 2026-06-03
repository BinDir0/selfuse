#!/usr/bin/env python3
"""Nested donut (two-ring pie) of dataset size.

  inner ring  = HOURS per dataset  (duration share)
  outer ring  = EPISODES per dataset (clip-count share)

Same dataset shares a colour across both rings, so you can compare "share of duration" vs
"share of clips" at a glance (e.g. a dataset with many but short clips = big outer / small inner).

Fill the real numbers in DATA below, OR pass --csv with columns `label,hours,episodes`.

    python scripts/stats/plot_dataset_pie.py --out dataset_pie.png
    python scripts/stats/plot_dataset_pie.py --csv my_counts.csv --out dataset_pie.png
"""

from __future__ import annotations

import argparse
import csv as csvmod
from pathlib import Path

# ----------------------------- FILL REAL DATA HERE -----------------------------
# (label, hours, episodes) -- placeholders for 12 datasets; replace with real counts.
DATA = [
    ("Dataset 1",  2100, 700_000),
    ("Dataset 2",   980, 380_000),
    ("Dataset 3",   760, 300_000),
    ("Dataset 4",   640, 250_000),
    ("Dataset 5",   520, 180_000),
    ("Dataset 6",   430, 160_000),
    ("Dataset 7",   360, 140_000),
    ("Dataset 8",   300, 120_000),
    ("Dataset 9",   240,  95_000),
    ("Dataset 10",  180,  70_000),
    ("Dataset 11",  120,  45_000),
    ("Dataset 12",   70,  20_000),
]
# -------------------------------------------------------------------------------

# Clean, distinct categorical palette (12 hues).
PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
    "#B279A2", "#FF9DA6", "#9D755D", "#17BECF", "#BAB0AC", "#6B4C9A",
]


def _load_csv(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        r = csvmod.reader(fh)
        next(r, None)  # header: label,hours,episodes
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
    ap.add_argument("--csv", default=None, help="CSV `label,hours,episodes` (overrides inline DATA).")
    ap.add_argument("--out", default="dataset_pie.png")
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
    hours = [float(d[1]) for d in data]
    eps = [float(d[2]) for d in data]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(data))]
    w = args.width

    def autopct(minp):
        return lambda p: f"{p:.0f}%" if p >= minp else ""

    fig, ax = plt.subplots(figsize=(10, 9))

    # OUTER ring = episodes
    _, _, a_out = ax.pie(
        eps, radius=1.0, colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(width=w, edgecolor="white", linewidth=2),
        autopct=autopct(args.min_pct), pctdistance=1 - w / 2,
        textprops=dict(fontsize=9.5, weight="bold"))
    # INNER ring = hours (duration)
    _, _, a_in = ax.pie(
        hours, radius=1.0 - w, colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(width=w, edgecolor="white", linewidth=2),
        autopct=autopct(args.min_pct), pctdistance=1 - w / (2 * (1.0 - w)),
        textprops=dict(fontsize=8.5))

    for t, c in zip(a_out, colors):
        t.set_color(_contrast(c))
    for t, c in zip(a_in, colors):
        t.set_color(_contrast(c))

    ax.text(0, 0, f"{sum(hours):,.0f} h\n{int(sum(eps)):,}\nepisodes",
            ha="center", va="center", fontsize=13, weight="bold", linespacing=1.3)

    ax.set(aspect="equal")
    ax.set_title(args.title, fontsize=16, weight="bold", pad=16)
    ax.legend(labels, title="Dataset", loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=10.5, title_fontsize=11, ncol=1)
    ax.text(0, -1.16, "outer ring: episodes   ·   inner ring: hours (duration)",
            ha="center", fontsize=10, color="#555555")

    out = Path(args.out).expanduser()
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"-> {out.resolve()}")
    print(f"   datasets: {len(data)}   total hours: {sum(hours):,.0f}   total episodes: {int(sum(eps)):,}")


if __name__ == "__main__":
    main()
