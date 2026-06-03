#!/usr/bin/env python3
"""Nested donut (two-ring pie) of dataset size, with an optional "pie of pie" zoom.

  inner ring  = HOURS per dataset  (duration share)
  outer ring  = EPISODES per dataset (clip-count share)

When one dataset dominates and the smallest ones span several orders of magnitude, a single pie
buries the small datasets as invisible slivers. With --rest_below_hours, datasets below that many
hours are collapsed into a grey "Rest" wedge in the MAIN donut and broken out, zoomed, in a SECOND
donut on the right (connected by lines) -- so every dataset is readable.

Fill DATA below (label, hours, episodes), OR pass --csv with columns `label,hours,episodes`.

    python scripts/stats/plot_dataset_pie.py --out dataset_pie.png                     # single donut
    python scripts/stats/plot_dataset_pie.py --out dataset_pie.png --rest_below_hours 20  # pie of pie
"""

from __future__ import annotations

import argparse
import csv as csvmod
from pathlib import Path

# ----------------------------- FILL REAL DATA HERE -----------------------------
# (label, hours, episodes).  HOURS are real (sorted desc); EPISODES are PLACEHOLDERS -- replace.
# NOTE: BuildAI-100K = 3061 + 4988 = 8049 h (confirm whether to merge or split).
DATA = [
    ("BuildAI-100K", 8049, 1_795_731),
    ("EgoVerse",      690,    35_175),
    ("EgoDex",        370,   147_588),
    ("BuildAI-10K",   288,   194_915),
    ("Ego4D",         138,    74_505),
    ("Epic-Kitchen",   49,    26_454),
    ("HoloAssist",   11.5,    11_426),
    ("HOT3D",         4.5,     1_105),
    ("TACO",            3,     1_558),
    ("OakInk-v2",     1.7,       891),
    ("H2O",             1,       935),
    ("FPHA",          0.5,       578),
]
# -------------------------------------------------------------------------------

# Clean, distinct categorical palette (12 hues).
PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
    "#B279A2", "#FF9DA6", "#9D755D", "#17BECF", "#BAB0AC", "#6B4C9A",
]
REST_COLOR = "#9AA0A6"  # neutral grey for the collapsed "Rest" wedge


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
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 145 else "#222222"


def _fmt_eps(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{int(n)}"


def _legend_labels(items):
    """`name   <hours> h · <episodes> ep` for the side legend."""
    return [f"{name}   {h:g} h · {_fmt_eps(e)} ep" for name, h, e in items]


def _nested_donut(ax, hours, eps, colors, w, min_pct, *, fs_out=9.5, fs_in=8.5):
    """Two-ring donut on ax: outer=episodes, inner=hours. Returns the outer wedge list."""
    def autopct(minp):
        return lambda p: f"{p:.0f}%" if p >= minp else ""

    wout, _, aout = ax.pie(
        eps, radius=1.0, colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(width=w, edgecolor="white", linewidth=2),
        autopct=autopct(min_pct), pctdistance=1 - w / 2, textprops=dict(fontsize=fs_out, weight="bold"))
    _, _, ain = ax.pie(
        hours, radius=1.0 - w, colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(width=w, edgecolor="white", linewidth=2),
        autopct=autopct(min_pct), pctdistance=1 - w / (2 * (1.0 - w)), textprops=dict(fontsize=fs_in))
    for t, c in zip(aout, colors):
        t.set_color(_contrast(c))
    for t, c in zip(ain, colors):
        t.set_color(_contrast(c))
    ax.set(aspect="equal")
    return wout


def _center(ax, hours, eps, fs=12):
    ax.text(0, 0, f"{sum(hours):,.0f} h\n{int(sum(eps)):,}\nepisodes",
            ha="center", va="center", fontsize=fs, weight="bold", linespacing=1.3)


def plot_single(data, args, plt):
    labels = [d[0] for d in data]
    hours = [float(d[1]) for d in data]
    eps = [float(d[2]) for d in data]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(data))]
    fig, ax = plt.subplots(figsize=(10, 9))
    _nested_donut(ax, hours, eps, colors, args.width, args.min_pct)
    _center(ax, hours, eps)
    ax.set_title(args.title, fontsize=16, weight="bold", pad=16)
    ax.legend(_legend_labels(data), title="Dataset  (hours · episodes)", loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10.5, title_fontsize=11)
    ax.text(0, -1.16, "outer ring: episodes   ·   inner ring: hours (duration)",
            ha="center", fontsize=10, color="#555555")
    return fig


def plot_pie_of_pie(data, args, plt):
    big = [d for d in data if float(d[1]) >= args.rest_below_hours]
    small = [d for d in data if float(d[1]) < args.rest_below_hours]
    if not small:
        return plot_single(data, args, plt)   # nothing to zoom

    rest = ("Rest", sum(float(d[1]) for d in small), sum(float(d[2]) for d in small))
    main = big + [rest]
    main_hours = [float(d[1]) for d in main]
    main_eps = [float(d[2]) for d in main]
    main_colors = [PALETTE[i % len(PALETTE)] for i in range(len(big))] + [REST_COLOR]

    small_hours = [float(d[1]) for d in small]
    small_eps = [float(d[2]) for d in small]
    small_colors = [PALETTE[i % len(PALETTE)] for i in range(len(small))]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(17, 9), gridspec_kw=dict(wspace=0.05))
    _nested_donut(axL, main_hours, main_eps, main_colors, args.width, args.min_pct)
    _center(axL, main_hours, main_eps)
    axL.set_title(args.title, fontsize=16, weight="bold", pad=14)
    axL.legend(_legend_labels(main), title="Dataset  (hours · episodes)", loc="center left",
               bbox_to_anchor=(-0.45, 0.5), frameon=False, fontsize=10.5, title_fontsize=11)

    # smaller zoom donut on the right (the grey "Rest" broken out)
    _nested_donut(axR, small_hours, small_eps, small_colors, args.width, 0.0, fs_out=9, fs_in=8)
    axR.text(0, 0, f"Rest\n{sum(small_hours):,.1f} h\n{int(sum(small_eps)):,} ep",
             ha="center", va="center", fontsize=10, weight="bold", linespacing=1.3)
    axR.set_title("Rest — zoomed", fontsize=13, weight="bold", pad=14)
    axR.legend(_legend_labels(small), title="(hours · episodes)", loc="center left",
               bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10, title_fontsize=10)

    fig.text(0.5, 0.04, "outer ring: episodes   ·   inner ring: hours (duration)",
             ha="center", fontsize=10, color="#555555")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None, help="CSV `label,hours,episodes` (overrides inline DATA).")
    ap.add_argument("--out", default="dataset_pie.png")
    ap.add_argument("--title", default="Dataset composition")
    ap.add_argument("--min_pct", type=float, default=3.0, help="Hide percentage labels below this %.")
    ap.add_argument("--width", type=float, default=0.34, help="Ring thickness.")
    ap.add_argument("--rest_below_hours", type=float, default=0.0,
                    help="Collapse datasets below this many hours into a grey 'Rest' wedge + a zoomed "
                         "second donut (pie of pie). 0 = single donut.")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _load_csv(args.csv) if args.csv else DATA
    if not data:
        raise SystemExit("no data (check --csv or the inline DATA)")

    if args.rest_below_hours and args.rest_below_hours > 0:
        fig = plot_pie_of_pie(data, args, plt)
    else:
        fig = plot_single(data, args, plt)

    out = Path(args.out).expanduser()
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"-> {out.resolve()}")


if __name__ == "__main__":
    main()
