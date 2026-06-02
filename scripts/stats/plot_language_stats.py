#!/usr/bin/env python3
"""Render figures from a language_annotation_stats.py output directory.

Turns the data files (verb_freq.csv / noun_freq.csv / per-level CSVs /
language_stats.json) into paper-ready figures:

  * word clouds              verbs_wordcloud.png, nouns_wordcloud.png
  * top-K bar charts         verbs_topk.png, nouns_topk.png
  * rank-frequency (Zipf)    verbs_zipf.png, nouns_zipf.png   (log-log)
  * per-level unique counts  per_level_unique.png             (L1..L5 grouped bars)

The unique COUNTS themselves are already numbers in language_stats.json
("verbs.unique", "object_nouns.unique", "per_level_terms.<level>.unique_*");
this script only visualizes them.

Usage (run wherever the stats CSVs are; pure CPU, no GPU):

    pip install matplotlib            # required
    pip install wordcloud             # only for the word-cloud panels
    python scripts/stats/plot_language_stats.py \
        --stats_dir /path/to/lang_stats_out \
        [--top_k 30] [--max_words 150]

Figures are written to <stats_dir>/figures/.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_freq_csv(path: Path) -> list[tuple[str, int]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 2 and row[1].strip().isdigit():
                rows.append((row[0], int(row[1])))
    return rows


def _barh(ax, pairs, title, color, value_label="count"):
    pairs = pairs[::-1]  # largest on top
    labels = [p[0] for p in pairs]
    counts = [p[1] for p in pairs]
    ax.barh(range(len(labels)), counts, color=color)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(value_label)
    ax.set_title(title)


def _barv(ax, pairs, title, color, value_label="count"):
    """Vertical bars spread left-to-right (landscape), labels rotated under the x-axis."""
    labels = [p[0] for p in pairs]
    counts = [p[1] for p in pairs]
    ax.bar(range(len(labels)), counts, color=color)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(value_label)
    ax.set_title(title)
    ax.margins(x=0.005)


def _unit_scale_label(count_unit, fps):
    """How to scale the raw counts (frame-weighted) and what to label the axis."""
    if count_unit == "hours":
        return 1.0 / (fps * 3600.0), "hours"
    if count_unit == "minutes":
        return 1.0 / (fps * 60.0), "minutes"
    if count_unit == "seconds":
        return 1.0 / fps, "seconds"
    return 1.0, "count"


def _zipf(ax, pairs, title, color):
    counts = sorted((c for _, c in pairs), reverse=True)
    ranks = range(1, len(counts) + 1)
    ax.loglog(list(ranks), counts, marker=".", linestyle="none", color=color)
    ax.set_xlabel("rank (log)")
    ax.set_ylabel("frequency (log)")
    ax.set_title(title)


# Warm-dominant categorical palette tuned to match the dense, multi-color "OBJECTS/ACTIONS"
# style (coral/orange/gold lead, with green/teal/blue/gray accents).
_PALETTE = [
    "#D7574B", "#E0685E", "#E8823C", "#EE9A3A",   # reds / oranges (more entries -> warm-dominant)
    "#E3B23C", "#D9A23A",                          # golds
    "#7E9F3A",                                     # olive green
    "#3FA38B",                                     # teal
    "#3D78A8",                                     # blue
    "#8A8D93",                                     # slate gray
]


def _palette_color_func(palette, seed=0):
    import random
    rng = random.Random(seed)
    return lambda *a, **k: rng.choice(palette)


def _wordcloud(freq_pairs, out_path: Path, max_words: int, *, prefer_horizontal: float = 0.95,
               font_path=None, seed: int = 0, width: int = 2000, height: int = 900) -> bool:
    """Dense, mostly-horizontal, multi-color cloud (warm palette) -- the designed-figure look,
    not wordcloud's default colors. Multi-word keys (e.g. 'sewing machine') stay single units
    because we generate from frequencies, not raw text."""
    if not freq_pairs:
        return False
    try:
        from wordcloud import WordCloud  # type: ignore
    except Exception:
        return False
    wc = WordCloud(
        width=width, height=height, background_color="white",
        max_words=max_words, prefer_horizontal=prefer_horizontal,
        relative_scaling=0.5, min_font_size=8, margin=2,
        font_path=font_path, random_state=seed,
        color_func=_palette_color_func(_PALETTE, seed),
    )
    wc.generate_from_frequencies(dict(freq_pairs))
    wc.to_file(str(out_path))
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render language-stats figures.")
    ap.add_argument("--stats_dir", required=True, help="Output dir from language_annotation_stats.py.")
    ap.add_argument("--top_k", type=int, default=50, help="Bars to show in top-K charts (used when --min_count is 0).")
    ap.add_argument("--min_count", type=int, default=0,
                    help="If >0, bar charts (and word clouds) keep only terms with count >= this "
                         "frequency threshold instead of a fixed top-K. Bars are still capped at "
                         "--max_bars for readability.")
    ap.add_argument("--max_bars", type=int, default=60, help="Hard cap on bars when --min_count selects many terms.")
    ap.add_argument("--bar_orient", choices=["v", "h"], default="v",
                    help="v: vertical bars spread left-to-right (wide/landscape, default). "
                         "h: horizontal bars stacked top-to-bottom (tall/portrait).")
    ap.add_argument("--count_unit", choices=["count", "seconds", "minutes", "hours"], default="count",
                    help="Bar axis unit. For duration-weighted stats the counts are FRAMES; "
                         "'hours' divides by fps*3600 and labels the axis accordingly.")
    ap.add_argument("--fps", type=float, default=30.0, help="Frames per second, for --count_unit time conversion.")
    ap.add_argument("--count_scale", type=float, default=1.0,
                    help="Extra multiplier on the bar-axis values, e.g. 10 to extrapolate a 1/10 "
                         "sample (--sample_frac 0.1) back to full-dataset scale.")
    ap.add_argument("--bar_width", type=float, default=0.22,
                    help="Inches per bar (smaller = more compact figure). Default 0.22.")
    ap.add_argument("--max_words", type=int, default=400, help="Max words in each word cloud (dense look).")
    ap.add_argument("--prefer_horizontal", type=float, default=0.95, help="Fraction of words laid horizontally.")
    ap.add_argument("--font", default=None, help="Path to a .ttf for nicer cloud text (optional).")
    ap.add_argument("--wc_seed", type=int, default=0, help="Seed for word-cloud layout + colors (reproducible).")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args(argv)

    stats_dir = Path(args.stats_dir).expanduser().resolve()
    if not stats_dir.is_dir():
        raise SystemExit(f"stats_dir not found: {stats_dir}")
    fig_dir = stats_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib is required for everything except word clouds
        raise SystemExit(f"matplotlib is required: pip install matplotlib ({exc})")

    verbs = read_freq_csv(stats_dir / "verb_freq.csv")
    nouns = read_freq_csv(stats_dir / "noun_freq.csv")
    if not verbs and not nouns:
        raise SystemExit(f"no verb_freq.csv / noun_freq.csv under {stats_dir} (run language_annotation_stats.py first)")

    written = []
    min_count = args.min_count if args.min_count and args.min_count > 0 else 0

    def _threshold(pairs):
        """Keep terms with count >= min_count (pairs are pre-sorted desc); else all."""
        return [p for p in pairs if p[1] >= min_count] if min_count else pairs

    def _select_bars(pairs):
        """(bars, description) for one bar chart: frequency threshold or fixed top-K."""
        if min_count:
            sel = _threshold(pairs)
            note = f"count ≥ {min_count}  (n={len(sel)}"
            note += f", showing {args.max_bars})" if len(sel) > args.max_bars else ")"
            return sel[: args.max_bars], note
        return pairs[: args.top_k], f"top {args.top_k}"

    # ---- word clouds (also honor the frequency threshold when set) ----
    wc_ok = []
    wc_kw = dict(prefer_horizontal=args.prefer_horizontal, font_path=args.font, seed=args.wc_seed)
    if _wordcloud(_threshold(verbs), fig_dir / "verbs_wordcloud.png", args.max_words, **wc_kw):
        written.append("verbs_wordcloud.png"); wc_ok.append("verbs")
    if _wordcloud(_threshold(nouns), fig_dir / "nouns_wordcloud.png", args.max_words, **wc_kw):
        written.append("nouns_wordcloud.png"); wc_ok.append("nouns")

    # ---- bar charts: frequency threshold (--min_count) or fixed top-K ----
    for pairs, name, color, kind in (
        (verbs, "verbs_topk.png", "#3b78b0", "verbs"),
        (nouns, "nouns_topk.png", "#b0533b", "object nouns"),
    ):
        if not pairs:
            continue
        bars, note = _select_bars(pairs)
        if not bars:
            print(f"  (no {kind} with count >= {min_count}; lower --min_count)")
            continue
        title = f"{kind[:1].upper() + kind[1:]} ({note})"
        scale, vlabel = _unit_scale_label(args.count_unit, args.fps)
        bars_s = [(t, c * scale * args.count_scale) for t, c in bars]
        bw = args.bar_width
        if args.bar_orient == "h":
            fig, ax = plt.subplots(figsize=(8, max(3, bw * len(bars_s))))
            _barh(ax, bars_s, title, color, vlabel)
        else:
            fig, ax = plt.subplots(figsize=(max(6, bw * len(bars_s)), 4.5))
            _barv(ax, bars_s, title, color, vlabel)
        fig.tight_layout(); fig.savefig(fig_dir / name, dpi=args.dpi); plt.close(fig)
        written.append(name)

    # ---- Zipf rank-frequency (log-log) ----
    for pairs, name, color, title in (
        (verbs, "verbs_zipf.png", "#3b78b0", "Verb rank-frequency (Zipf)"),
        (nouns, "nouns_zipf.png", "#b0533b", "Object-noun rank-frequency (Zipf)"),
    ):
        if not pairs:
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        _zipf(ax, pairs, title, color)
        fig.tight_layout(); fig.savefig(fig_dir / name, dpi=args.dpi); plt.close(fig)
        written.append(name)

    # ---- per-level unique verb/object counts (from language_stats.json) ----
    stats_json = stats_dir / "language_stats.json"
    if stats_json.exists():
        summary = json.loads(stats_json.read_text(encoding="utf-8"))
        plt_terms = summary.get("per_level_terms") or {}
        levels = [k for k in ("level1", "level2", "level3", "level4", "level5") if k in plt_terms]
        if levels:
            uv = [plt_terms[k]["unique_verbs"] for k in levels]
            uo = [plt_terms[k]["unique_objects"] for k in levels]
            x = range(len(levels))
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar([i - 0.2 for i in x], uv, width=0.4, label="unique verbs", color="#3b78b0")
            ax.bar([i + 0.2 for i in x], uo, width=0.4, label="unique objects", color="#b0533b")
            ax.set_xticks(list(x))
            ax.set_xticklabels([l.replace("level", "L") for l in levels])
            ax.set_ylabel("unique count")
            ax.set_title("Per-level unique verbs / objects")
            ax.legend()
            fig.tight_layout(); fig.savefig(fig_dir / "per_level_unique.png", dpi=args.dpi); plt.close(fig)
            written.append("per_level_unique.png")

    print("=" * 60)
    print(f"stats_dir : {stats_dir}")
    print(f"verbs     : {len(verbs)} unique   nouns: {len(nouns)} unique")
    if "verbs" not in wc_ok or "nouns" not in wc_ok:
        print("word cloud: skipped some (pip install wordcloud to enable)")
    print(f"figures   : {len(written)} -> {fig_dir}")
    for name in written:
        print(f"  - {name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
