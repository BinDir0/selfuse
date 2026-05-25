#!/usr/bin/env python3
"""Track-A dataset statistics for the paper (scale table + language stats + figures).

Runs post-hoc over an already-built clip manifest (+ optional annotation root); no GPU,
no inference. Produces:
  * per-source scale table  (#clips, #frames, hours)  -> tab:dataset
  * totals                  (clips / frames / hours / language tokens)
  * language stats          (per-level avg tokens L1..L5, distinct level-1 verbs, verb h-index)
  * optional figures        (clip-length histogram, top-N verb frequency)  -> fig:dist (partial)

It intentionally does NOT touch lowdim features, so wrist trans/rot distributions are out of
scope here (those reuse the existing Stage-4 figures / a separate lowdim pass).

Examples
--------
  python scripts/dataset_stats.py \
      --manifest /path/to/clip_manifest.jsonl \
      --annotation_root /path/to/annotations \
      --tokenizer tiktoken \
      --out_md dataset_stats.md --out_json dataset_stats.json --fig_dir figs/

Notes
-----
* "samples": reported two ways since the term is ambiguous — #clips (episodes) and
  #frame-action pairs (sum of frame_count-1). Pick whichever the paper means.
* fps: uses descriptor.fps when present, else --default_fps (with a warning count).
* tokenizer: tiktoken (cl100k) > HF (--hf_model) > whitespace fallback; the choice is
  printed so the reported token count is reproducible.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# repo imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.pipeline.clip_manifest import load_clip_manifest
from lib.pipeline.annotation_protocol import HIERARCHY_KEYS, load_clip_annotation


def build_tokenizer(kind: str, hf_model: str):
    """Return (encode_fn, label). encode_fn(str) -> int token count."""
    if kind == "tiktoken":
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return (lambda s: len(enc.encode(s or "")), "tiktoken/cl100k_base")
        except Exception as e:
            print(f"  [tokenizer] tiktoken unavailable ({e}); falling back to whitespace")
    elif kind == "hf":
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(hf_model)
            return (lambda s: len(tok.encode(s or "", add_special_tokens=False)), f"hf/{hf_model}")
        except Exception as e:
            print(f"  [tokenizer] HF '{hf_model}' unavailable ({e}); falling back to whitespace")
    return (lambda s: len((s or "").split()), "whitespace")


def h_index(counts: Counter) -> int:
    """Largest h such that at least h items each occur >= h times (VITRA-style)."""
    freqs = sorted(counts.values(), reverse=True)
    h = 0
    for i, f in enumerate(freqs, start=1):
        if f >= i:
            h = i
        else:
            break
    return h


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="clip manifest .jsonl")
    ap.add_argument("--annotation_root", default=None, help="annotation dir (enables language stats)")
    ap.add_argument("--annotation_suffix", default=".annotation.json")
    ap.add_argument("--default_fps", type=float, default=30.0)
    ap.add_argument("--tokenizer", choices=["tiktoken", "hf", "whitespace"], default="tiktoken")
    ap.add_argument("--hf_model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--top_verbs", type=int, default=30)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--fig_dir", default=None)
    ap.add_argument("--max_clips", type=int, default=0, help="0 = all (debug cap otherwise)")
    args = ap.parse_args(argv)

    records = load_clip_manifest(args.manifest)
    if args.max_clips > 0:
        records = records[: args.max_clips]

    # --- scale per source ---
    per_source = defaultdict(lambda: {"clips": 0, "frames": 0, "seconds": 0.0})
    missing_fps = 0
    for r in records:
        d = r.descriptor
        fc = int(d.frame_count)
        fps = float(d.fps) if getattr(d, "fps", None) else args.default_fps
        if not getattr(d, "fps", None):
            missing_fps += 1
        s = per_source[r.source_id]
        s["clips"] += 1
        s["frames"] += fc
        s["seconds"] += fc / fps if fps > 0 else 0.0

    # --- language stats (optional) ---
    lang = None
    if args.annotation_root:
        encode, tok_label = build_tokenizer(args.tokenizer, args.hf_model)
        level_tokens = {k: 0 for k in HIERARCHY_KEYS}
        level_clips = {k: 0 for k in HIERARCHY_KEYS}
        verbs = Counter()
        total_tokens = 0
        annotated = 0
        for r in records:
            ann, err, _ = load_clip_annotation(args.annotation_root, r.clip_id, annotation_suffix=args.annotation_suffix)
            if ann is None:
                continue
            annotated += 1
            hier = ann.hierarchy or {}
            for k in HIERARCHY_KEYS:
                text = hier.get(k)
                if text:
                    n = encode(str(text))
                    level_tokens[k] += n
                    level_clips[k] += 1
                    total_tokens += n
            lvl1 = (hier.get("level1") or (ann.instruction[0] if ann.instruction else "") or "").strip()
            if lvl1:
                verbs[lvl1.split()[0].lower().strip(".,")] += 1
        lang = {
            "tokenizer": tok_label,
            "annotated_clips": annotated,
            "total_tokens": total_tokens,
            "avg_tokens_per_level": {k: (level_tokens[k] / level_clips[k] if level_clips[k] else 0.0) for k in HIERARCHY_KEYS},
            "distinct_level1_verbs": len(verbs),
            "verb_h_index": h_index(verbs),
            "top_verbs": verbs.most_common(args.top_verbs),
        }

    # --- assemble totals ---
    tot_clips = sum(s["clips"] for s in per_source.values())
    tot_frames = sum(s["frames"] for s in per_source.values())
    tot_hours = sum(s["seconds"] for s in per_source.values()) / 3600.0
    summary = {
        "totals": {
            "sources": len(per_source),
            "clips_episodes": tot_clips,
            "frames": tot_frames,
            "frame_action_pairs": tot_frames - tot_clips,  # sum(frame_count-1)
            "hours": round(tot_hours, 1),
            "language_tokens": (lang or {}).get("total_tokens"),
        },
        "per_source": {
            src: {
                "clips": s["clips"],
                "frames": s["frames"],
                "hours": round(s["seconds"] / 3600.0, 2),
            }
            for src, s in sorted(per_source.items(), key=lambda kv: -kv[1]["seconds"])
        },
        "language": lang,
        "missing_fps_clips": missing_fps,
        "default_fps_used": args.default_fps,
    }

    # --- markdown ---
    lines = ["| source | clips | frames | hours |", "|---|---:|---:|---:|"]
    for src, s in summary["per_source"].items():
        lines.append(f"| {src} | {s['clips']:,} | {s['frames']:,} | {s['hours']:,} |")
    t = summary["totals"]
    lines.append(f"| **total** | **{t['clips_episodes']:,}** | **{t['frames']:,}** | **{t['hours']:,}** |")
    md = "\n".join(lines)
    if lang:
        md += "\n\n**Language**: tokenizer=" + lang["tokenizer"]
        md += f"; total_tokens={lang['total_tokens']:,}; distinct L1 verbs={lang['distinct_level1_verbs']}; verb h-index={lang['verb_h_index']}\n"
        md += "avg tokens per level: " + ", ".join(f"{k}={lang['avg_tokens_per_level'][k]:.1f}" for k in HIERARCHY_KEYS)

    print(md)
    print("\n[totals]", json.dumps(summary["totals"], ensure_ascii=False))
    if missing_fps:
        print(f"[warn] {missing_fps} clips had no descriptor.fps; used --default_fps={args.default_fps}")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_md:
        Path(args.out_md).write_text(md + "\n", encoding="utf-8")

    # --- figures (optional) ---
    if args.fig_dir:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fd = Path(args.fig_dir)
            fd.mkdir(parents=True, exist_ok=True)
            # clip-length histogram (frames)
            lengths = [int(r.descriptor.frame_count) for r in records]
            plt.figure()
            plt.hist(lengths, bins=60)
            plt.xlabel("clip length (frames)")
            plt.ylabel("count")
            plt.title("Clip length distribution")
            plt.tight_layout()
            plt.savefig(fd / "clip_length_hist.png", dpi=150)
            plt.close()
            # top-N verb bar
            if lang and lang["top_verbs"]:
                vs, cs = zip(*lang["top_verbs"])
                plt.figure(figsize=(8, max(3, len(vs) * 0.25)))
                plt.barh(range(len(vs)), cs)
                plt.yticks(range(len(vs)), vs)
                plt.gca().invert_yaxis()
                plt.xlabel("frequency")
                plt.title(f"Top-{len(vs)} level-1 verbs")
                plt.tight_layout()
                plt.savefig(fd / "verb_frequency.png", dpi=150)
                plt.close()
            print(f"[figs] wrote to {fd}")
        except Exception as e:
            print(f"[figs] skipped ({e})")


if __name__ == "__main__":
    main()
