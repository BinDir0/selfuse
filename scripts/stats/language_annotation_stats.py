#!/usr/bin/env python3
"""Offline language-annotation statistics for the dataset/paper section.

Scans a directory of clip annotation sidecars (``*.annotation.json`` or a custom
suffix) and computes the language statistics we want to report:

  * valid-language coverage (status == "Valid", non-empty instruction)
  * unique verbs   : count + frequency distribution (+ word cloud)
  * unique object nouns : count + frequency distribution (+ word cloud)
  * per-level (L1-L5) token count + vocabulary size
  * diversity metrics: distinct-1/2, verb h-index, type-token ratio
  * instructions-per-clip + instruction-length distributions
  * representative episodes (all 5 levels present) for a qualitative L1-L5 table

Parsing reuses the pipeline's own normalization (lib/pipeline/annotation_protocol)
so these stats match exactly what the build consumes.

Run on the PRODUCTION machine (this only reads annotation JSON, no GPU):

    python scripts/stats/language_annotation_stats.py \
        --annotation_root /path/to/annotations \
        --out_dir /path/to/lang_stats_out \
        [--suffix .annotation.json] [--top_k 60] [--examples 5]

Optional extras (graceful fallback if missing):
    pip install spacy && python -m spacy download en_core_web_sm   # better verb/noun POS
    pip install wordcloud matplotlib                               # word-cloud PNGs

Outputs under --out_dir:
    language_stats.json      all numeric stats + provenance
    verb_freq.csv            verb, count (sorted desc)
    noun_freq.csv            noun, count (sorted desc)
    example_episodes.json    candidate representative clips with L1-L5
    verbs_wordcloud.png      (if wordcloud available)
    nouns_wordcloud.png      (if wordcloud available)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the pipeline's own annotation parsing so stats match the build exactly.
from lib.pipeline.annotation_protocol import (  # noqa: E402
    HIERARCHY_KEYS,
    _normalize_hierarchy,
    _normalize_instruction,
)

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")

# Small stopword / article set for the no-spaCy fallback object-noun heuristic.
_ARTICLES = {
    "a", "an", "the", "this", "that", "these", "those",
    "his", "her", "its", "their", "your", "my", "our", "some", "any",
}
_FALLBACK_STOP = _ARTICLES | {
    "of", "to", "with", "and", "or", "on", "in", "into", "onto", "from",
    "up", "down", "out", "off", "over", "under", "at", "by", "for", "then",
    "left", "right", "hand", "both", "hands",  # too generic to be the "object"
}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text or "")]


# --------------------------------------------------------------------------- #
# Optional spaCy backend for POS-based verb/noun extraction.
# --------------------------------------------------------------------------- #
class _Extractor:
    def __init__(self):
        self.nlp = None
        try:
            import spacy  # type: ignore

            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
            except Exception:
                self.nlp = None
        except Exception:
            self.nlp = None
        self.mode = "spacy" if self.nlp is not None else "heuristic"

    def head_verb(self, l1: str) -> str | None:
        """Main action verb of the L1 verb-object phrase."""
        if not l1:
            return None
        if self.nlp is not None:
            doc = self.nlp(l1)
            for tok in doc:
                if tok.pos_ == "VERB":
                    return tok.lemma_.lower()
            for tok in doc:  # imperative sometimes tagged ROOT/AUX
                if tok.dep_ == "ROOT" and tok.is_alpha:
                    return tok.lemma_.lower()
            return None
        toks = tokenize(l1)
        return toks[0] if toks else None  # imperative head ≈ first token

    def object_noun(self, l1: str) -> str | None:
        """Direct object of the L1 phrase (the thing being manipulated)."""
        if not l1:
            return None
        if self.nlp is not None:
            doc = self.nlp(l1)
            for tok in doc:  # prefer the grammatical direct object
                if tok.dep_ in ("dobj", "obj") and tok.pos_ in ("NOUN", "PROPN"):
                    return tok.lemma_.lower()
            for tok in doc:
                if tok.pos_ in ("NOUN", "PROPN"):
                    return tok.lemma_.lower()
            return None
        toks = tokenize(l1)[1:]  # drop head verb
        for t in toks:
            if t not in _FALLBACK_STOP:
                return t
        return None

    def all_verbs(self, text: str) -> list[str]:
        if self.nlp is None or not text:
            return []
        return [t.lemma_.lower() for t in self.nlp(text) if t.pos_ == "VERB"]

    def all_nouns(self, text: str) -> list[str]:
        if self.nlp is None or not text:
            return []
        return [t.lemma_.lower() for t in self.nlp(text) if t.pos_ in ("NOUN", "PROPN")]


# --------------------------------------------------------------------------- #
def h_index(freq: Counter) -> int:
    """Largest h such that h items each occur >= h times."""
    counts = sorted(freq.values(), reverse=True)
    h = 0
    for i, c in enumerate(counts, start=1):
        if c >= i:
            h = i
        else:
            break
    return h


def distinct_n(texts: list[str], n: int) -> dict:
    grams: Counter = Counter()
    total = 0
    for t in texts:
        toks = tokenize(t)
        for i in range(len(toks) - n + 1):
            grams[tuple(toks[i:i + n])] += 1
            total += 1
    return {
        "distinct": len(grams),
        "total": total,
        "ratio": (len(grams) / total) if total else 0.0,
    }


def dist_summary(values: list[float]) -> dict | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    vals_sorted = sorted(vals)

    def pct(p: float) -> float:
        if len(vals_sorted) == 1:
            return vals_sorted[0]
        k = (len(vals_sorted) - 1) * (p / 100.0)
        lo = int(k)
        hi = min(lo + 1, len(vals_sorted) - 1)
        return vals_sorted[lo] + (vals_sorted[hi] - vals_sorted[lo]) * (k - lo)

    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "min": vals_sorted[0],
        "p25": pct(25),
        "median": pct(50),
        "p75": pct(75),
        "p95": pct(95),
        "max": vals_sorted[-1],
    }


def load_annotations(root: Path, suffix: str) -> tuple[list[dict], dict]:
    """Return (valid_records, coverage_counts). Each record: clip_id, levels, instruction."""
    files = sorted(root.rglob(f"*{suffix}"))
    coverage = {
        "files_found": len(files),
        "valid": 0,
        "invalid_status": 0,
        "invalid_json": 0,
        "empty_instruction": 0,
    }
    records = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            coverage["invalid_json"] += 1
            continue
        if not isinstance(payload, dict):
            coverage["invalid_json"] += 1
            continue
        if str(payload.get("status", "Valid")).strip() != "Valid":
            coverage["invalid_status"] += 1
            continue
        hierarchy = _normalize_hierarchy(payload)
        instruction = _normalize_instruction(payload, hierarchy)
        if not instruction:
            coverage["empty_instruction"] += 1
            continue
        coverage["valid"] += 1
        records.append({
            "clip_id": path.stem,
            "levels": {k: hierarchy.get(k) for k in HIERARCHY_KEYS},
            "instruction": instruction,
        })
    return records, coverage


def aggregate_extraction(path: Path) -> dict:
    """Aggregate a Qwen extraction.jsonl into per-level + per-clip-union Counters."""
    level_verbs = {k: Counter() for k in HIERARCHY_KEYS}
    level_nouns = {k: Counter() for k in HIERARCHY_KEYS}
    union_verbs: Counter = Counter()
    union_nouns: Counter = Counter()
    clips = 0
    if not path.exists():
        raise SystemExit(f"extraction file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        clips += 1
        clip_verbs: set[str] = set()
        clip_nouns: set[str] = set()
        for lv, vo in (rec.get("levels") or {}).items():
            verbs = [str(x).lower() for x in (vo.get("verbs") or [])]
            nouns = [str(x).lower() for x in (vo.get("objects") or [])]
            if lv in level_verbs:
                level_verbs[lv].update(verbs)
                level_nouns[lv].update(nouns)
            clip_verbs.update(verbs)
            clip_nouns.update(nouns)
        union_verbs.update(clip_verbs)
        union_nouns.update(clip_nouns)
    return {
        "clips": clips,
        "level_verbs": level_verbs,
        "level_nouns": level_nouns,
        "union_verbs": union_verbs,
        "union_nouns": union_nouns,
    }


def maybe_wordcloud(freq: Counter, out_path: Path) -> bool:
    if not freq:
        return False
    try:
        from wordcloud import WordCloud  # type: ignore
    except Exception:
        return False
    try:
        wc = WordCloud(width=1600, height=900, background_color="white")
        wc.generate_from_frequencies(dict(freq))
        wc.to_file(str(out_path))
        return True
    except Exception:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description="Language-annotation statistics.")
    ap.add_argument("--annotation_root", required=True, help="Directory of annotation sidecars (searched recursively).")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--suffix", default=".annotation.json", help="Annotation file suffix to glob (default .annotation.json).")
    ap.add_argument("--top_k", type=int, default=60, help="How many top verbs/nouns to keep in the JSON summary.")
    ap.add_argument("--examples", type=int, default=5, help="How many representative episode candidates to dump.")
    ap.add_argument(
        "--extraction",
        default=None,
        help="Optional extraction.jsonl from extract_verbs_objects_qwen.py. When given, "
             "verb/object stats come from the LLM extraction (per-level + per-clip union) "
             "instead of the spaCy/heuristic L1-headline path.",
    )
    args = ap.parse_args(argv)

    root = Path(args.annotation_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise SystemExit(f"annotation_root not found: {root}")

    records, coverage = load_annotations(root, args.suffix)
    if not records:
        raise SystemExit(f"No valid annotations under {root} (coverage={coverage})")

    # ---- verbs / nouns ----
    # Two backends:
    #   (a) --extraction: per-level lists from the Qwen extraction; headline freq =
    #       per-clip UNION over levels; also keep per-level distributions.
    #   (b) default: spaCy/heuristic head-verb + object-noun from L1 only.
    verb_freq: Counter = Counter()
    noun_freq: Counter = Counter()
    all_verb_freq: Counter = Counter()
    all_noun_freq: Counter = Counter()
    per_level_terms = None
    if args.extraction:
        agg = aggregate_extraction(Path(args.extraction).expanduser().resolve())
        verb_freq = agg["union_verbs"]
        noun_freq = agg["union_nouns"]
        extractor_mode = "qwen_extraction"
        per_level_terms = {
            k: {
                "unique_verbs": len(agg["level_verbs"][k]),
                "unique_objects": len(agg["level_nouns"][k]),
                "verb_h_index": h_index(agg["level_verbs"][k]),
                "object_h_index": h_index(agg["level_nouns"][k]),
                "top_verbs": agg["level_verbs"][k].most_common(args.top_k),
                "top_objects": agg["level_nouns"][k].most_common(args.top_k),
            }
            for k in HIERARCHY_KEYS
        }
        print(f"[extraction] {agg['clips']} clips loaded from {args.extraction}")
    else:
        extractor = _Extractor()
        extractor_mode = extractor.mode
        for rec in records:
            l1 = rec["levels"].get("level1") or (rec["instruction"][0] if rec["instruction"] else "")
            v = extractor.head_verb(l1)
            if v:
                verb_freq[v] += 1
            n = extractor.object_noun(l1)
            if n:
                noun_freq[n] += 1
            if extractor.mode == "spacy":
                all_text = " ".join(t for t in rec["levels"].values() if t)
                all_verb_freq.update(extractor.all_verbs(all_text))
                all_noun_freq.update(extractor.all_nouns(all_text))

    # ---- per-level token / vocab ----
    per_level = {}
    for key in HIERARCHY_KEYS:
        texts = [rec["levels"][key] for rec in records if rec["levels"].get(key)]
        token_counts = [len(tokenize(t)) for t in texts]
        vocab: set[str] = set()
        for t in texts:
            vocab.update(tokenize(t))
        per_level[key] = {
            "present": len(texts),
            "avg_tokens": statistics.fmean(token_counts) if token_counts else 0.0,
            "token_dist": dist_summary(token_counts),
            "vocab_size": len(vocab),
        }

    # ---- diversity (over L1 set; the action layer) ----
    l1_texts = [rec["levels"].get("level1") or rec["instruction"][0] for rec in records]
    all_tokens = [tok for t in l1_texts for tok in tokenize(t)]
    ttr = (len(set(all_tokens)) / len(all_tokens)) if all_tokens else 0.0

    # ---- instructions-per-clip + length ----
    instr_counts = [len(rec["instruction"]) for rec in records]
    instr_lengths = [len(tokenize(s)) for rec in records for s in rec["instruction"]]

    # ---- representative episodes (all 5 levels present) ----
    full = [rec for rec in records if all(rec["levels"].get(k) for k in HIERARCHY_KEYS)]

    def richness(rec: dict) -> int:
        toks = set()
        for k in HIERARCHY_KEYS:
            toks.update(tokenize(rec["levels"][k]))
        return len(toks)

    examples = [
        {"clip_id": rec["clip_id"], "richness": richness(rec),
         "levels": {k: rec["levels"][k] for k in HIERARCHY_KEYS}}
        for rec in sorted(full, key=richness, reverse=True)[: args.examples]
    ]

    # ---- write distributions ----
    with (out_dir / "verb_freq.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["verb", "count"])
        w.writerows(verb_freq.most_common())
    with (out_dir / "noun_freq.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["noun", "count"])
        w.writerows(noun_freq.most_common())
    (out_dir / "example_episodes.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    # Per-level term distributions (qwen extraction mode only).
    if per_level_terms is not None:
        for key in HIERARCHY_KEYS:
            for kind, col in (("top_verbs", "verb"), ("top_objects", "object")):
                with (out_dir / f"{col}_freq_{key}.csv").open("w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow([col, "count"])
                    w.writerows(per_level_terms[key][kind])

    wc_verbs = maybe_wordcloud(verb_freq, out_dir / "verbs_wordcloud.png")
    wc_nouns = maybe_wordcloud(noun_freq, out_dir / "nouns_wordcloud.png")

    summary = {
        "annotation_root": str(root),
        "suffix": args.suffix,
        "extractor_mode": extractor_mode,  # "qwen_extraction" | "spacy" | "heuristic"
        "coverage": {
            **coverage,
            "valid_language_coverage": (
                coverage["valid"] / coverage["files_found"] if coverage["files_found"] else 0.0
            ),
        },
        "verbs": {  # in qwen_extraction mode these are the per-clip UNION over levels
            "unique": len(verb_freq),
            "total_occurrences": sum(verb_freq.values()),
            "h_index": h_index(verb_freq),
            "top": verb_freq.most_common(args.top_k),
        },
        "object_nouns": {
            "unique": len(noun_freq),
            "total_occurrences": sum(noun_freq.values()),
            "h_index": h_index(noun_freq),
            "top": noun_freq.most_common(args.top_k),
        },
        "per_level_terms": per_level_terms,  # per-level verb/object distributions (qwen mode)
        "all_levels_pos": {  # only populated by the spaCy fallback backend
            "unique_verbs": len(all_verb_freq),
            "unique_nouns": len(all_noun_freq),
            "top_verbs": all_verb_freq.most_common(args.top_k),
            "top_nouns": all_noun_freq.most_common(args.top_k),
        } if extractor_mode == "spacy" else None,
        "per_level": per_level,
        "diversity": {
            "type_token_ratio_L1": ttr,
            "distinct_1_L1": distinct_n(l1_texts, 1),
            "distinct_2_L1": distinct_n(l1_texts, 2),
            "verb_h_index": h_index(verb_freq),
        },
        "instructions_per_clip": dist_summary(instr_counts),
        "instruction_length_tokens": dist_summary(instr_lengths),
        "representative_episodes_count": len(full),
        "outputs": {
            "verb_freq_csv": "verb_freq.csv",
            "noun_freq_csv": "noun_freq.csv",
            "example_episodes_json": "example_episodes.json",
            "verbs_wordcloud_png": "verbs_wordcloud.png" if wc_verbs else None,
            "nouns_wordcloud_png": "nouns_wordcloud.png" if wc_nouns else None,
        },
    }
    (out_dir / "language_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- console digest ----
    print("=" * 64)
    print(f"annotation_root : {root}")
    _hint = "" if extractor_mode in ("spacy", "qwen_extraction") else "  (install spaCy+en_core_web_sm or pass --extraction)"
    print(f"extractor       : {extractor_mode}{_hint}")
    print(f"files found     : {coverage['files_found']}")
    print(f"valid           : {coverage['valid']}  "
          f"(coverage {summary['coverage']['valid_language_coverage']:.1%})")
    print(f"  invalid_status: {coverage['invalid_status']}  "
          f"invalid_json: {coverage['invalid_json']}  empty: {coverage['empty_instruction']}")
    print(f"unique verbs    : {summary['verbs']['unique']}  (h-index {summary['verbs']['h_index']})")
    print(f"unique nouns    : {summary['object_nouns']['unique']}  (h-index {summary['object_nouns']['h_index']})")
    print(f"L1 TTR          : {ttr:.4f}   distinct-2(L1): {summary['diversity']['distinct_2_L1']['ratio']:.4f}")
    print("per-level avg tokens / vocab:")
    for k in HIERARCHY_KEYS:
        pl = per_level[k]
        line = f"  {k}: present={pl['present']:>6}  avg_tokens={pl['avg_tokens']:6.1f}  vocab={pl['vocab_size']}"
        if per_level_terms is not None:
            t = per_level_terms[k]
            line += f"  verbs={t['unique_verbs']} objects={t['unique_objects']}"
        print(line)
    print(f"wordclouds      : verbs={'yes' if wc_verbs else 'no'} nouns={'yes' if wc_nouns else 'no'}"
          + ("" if (wc_verbs and wc_nouns) else "  (pip install wordcloud matplotlib)"))
    print(f"-> {out_dir / 'language_stats.json'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
