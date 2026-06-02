#!/usr/bin/env python3
"""Extract the L1 (verb, object) TASK list, with dependency parsing + optional synonym merge.

A "task" is a manipulation VERB paired with the OBJECT it acts on -- e.g. (assemble, gear).
language_annotation_stats.py lists verbs and nouns *separately* and disables the parser for
speed; pairing a verb with its object needs the dependency parse, so here we run the parser.
It only runs on L1 (one short imperative per clip), so it stays cheap (nlp.pipe + n_process).

What is handled automatically (no work from you):
  * morphology  -- we take token LEMMAS: "assembling the gears" -> (assemble, gear), so
                   "assemble A" and "assembling A" collapse to one task.
  * articles    -- a/an/the/this/your are DET, never the object, so they never appear.

What is NOT handled here (it is semantic): synonym merging (assemble~install~mount,
screw~bolt). Generate a canonical map with cluster_synonyms_qwen.py over verb_vocab.csv /
noun_vocab.csv, review/edit it, then pass it back via --verb_map / --noun_map to emit
tasks_canonical.csv with the synonyms collapsed.

Outputs under --out_dir:
  tasks_raw.csv         verb, object, count   (lemmatized + de-articled; sorted desc)
  verb_vocab.csv        verb, count           (unique task verbs  -> synonym-clustering input)
  noun_vocab.csv        object, count         (unique task objects -> synonym-clustering input)
  tasks_canonical.csv   verb, object, count   (ONLY when --verb_map and/or --noun_map given)

Run on the production machine (needs a spaCy model WITH a parser; en_core_web_sm is enough):

    python scripts/stats/extract_l1_tasks.py \
        --annotation_root /path/to/annotations --suffix _qwen-annotation.json \
        --out_dir /path/to/tasks_out --parse_workers 64 --n_process 16
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for _p in (str(PROJECT_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the parallel parser, progress bar, and the stop lists from the stats script so the
# task vocabulary matches the verb/noun distributions exactly.
from language_annotation_stats import (  # noqa: E402
    NOUN_STOP,
    VERB_STOP,
    _TERM_OK,
    _progress,
    add_cache_args,
    load_annotations,
    resolve_cache,
)

# en_core_web_* dependency labels for "the noun this verb acts on".
OBJ_DEPS = {"dobj", "dative", "attr", "oprd"}  # direct/indirect object, predicative
# nsubjpass handles passive voice ("the cup is lifted" -> object=cup).


def load_spacy_with_parser():
    """Load a spaCy English model that HAS a parser (needed to pair verb->object).
    ner is disabled (unused); parser + tagger + lemmatizer are kept."""
    try:
        import spacy  # type: ignore
    except Exception as e:
        raise SystemExit(f"spaCy not importable: {e}\nFix: pip install spacy")
    last = ""
    for name in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
        try:
            nlp = spacy.load(name, disable=["ner"])
        except Exception as e:
            last = f"{name}: {e}"
            continue
        if "parser" not in nlp.pipe_names:
            last = f"{name}: model has no parser pipe"
            continue
        return nlp, name
    raise SystemExit(
        "need a spaCy model WITH a parser for (verb, object) pairing; none loaded.\n"
        f"last error: {last}\nFix: python -m spacy download en_core_web_sm")


def tasks_from_doc(doc):
    """Return (pairs, verbs) for one parsed L1 doc.

    pairs : list of (verb_lemma, object_lemma) -- a manipulation verb and a noun it governs
            (direct/indirect object, predicative, or passive subject), plus conjoined objects
            ("pick up the screw and bolt" -> two pairs). Lemmas drop inflection + articles.
    verbs : set of manipulation-verb lemmas seen (kept even when a clean object is missing,
            so the synonym-clustering vocab still covers them).
    """
    pairs = []
    verbs = set()
    for tok in doc:
        if tok.pos_ != "VERB":
            continue
        vlem = tok.lemma_.lower().strip()
        if not _TERM_OK.match(vlem) or vlem in VERB_STOP:
            continue
        verbs.add(vlem)
        objs = []
        for ch in tok.children:
            if ch.dep_ in OBJ_DEPS or ch.dep_ == "nsubjpass":
                objs.append(ch)
                objs.extend(ch.conjuncts)  # "screw and bolt"
        for o in objs:
            if o.pos_ not in ("NOUN", "PROPN"):
                continue
            olem = o.lemma_.lower().strip()
            if not _TERM_OK.match(olem) or olem in NOUN_STOP:
                continue
            pairs.append((vlem, olem))
    return pairs, verbs


def _l1_text(rec) -> str:
    return rec["levels"].get("level1") or (rec["instruction"][0] if rec.get("instruction") else "")


def load_canon_map(path: Path) -> dict:
    """Read a {canonical: [variants...]} JSON map -> {word: canonical} (canonical maps to self)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    inv: dict[str, str] = {}
    for canon, members in (data or {}).items():
        c = str(canon).strip().lower()
        if not c:
            continue
        inv.setdefault(c, c)
        for m in (members or []):
            m = str(m).strip().lower()
            if m:
                inv[m] = c
    return inv


def write_pairs_csv(path: Path, counter: Counter, header):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for (a, b), c in counter.most_common():
            w.writerow([a, b, c])


def write_vocab_csv(path: Path, counter: Counter, header):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for term, c in counter.most_common():
            w.writerow([term, c])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotation_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--suffix", default=".annotation.json")
    ap.add_argument("--parse_workers", type=int, default=32, help="Threads for reading annotation JSON.")
    ap.add_argument("--n_process", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)),
                    help="spaCy nlp.pipe worker processes (parser is on, so this matters more here).")
    ap.add_argument("--spacy_batch_size", type=int, default=256)
    ap.add_argument("--verb_map", default=None, help="JSON {canonical:[variants]} for verbs -> tasks_canonical.csv")
    ap.add_argument("--noun_map", default=None, help="JSON {canonical:[variants]} for objects -> tasks_canonical.csv")
    add_cache_args(ap)
    args = ap.parse_args(argv)

    root = Path(args.annotation_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise SystemExit(f"annotation_root not found: {root}")

    records, coverage = load_annotations(root, args.suffix, args.parse_workers,
                                         cache=resolve_cache(args, root, args.suffix),
                                         rebuild_cache=args.rebuild_cache)
    if not records:
        raise SystemExit(f"No valid annotations under {root} (coverage={coverage})")

    nlp, model_name = load_spacy_with_parser()
    print(f"[tasks] spaCy model = {model_name} (parser on)  valid clips = {len(records)}", file=sys.stderr, flush=True)

    texts = [_l1_text(rec) for rec in records]
    keep = [t for t in texts if t]

    task_counts: Counter = Counter()
    verb_counts: Counter = Counter()
    noun_counts: Counter = Counter()
    docs = nlp.pipe(keep, batch_size=args.spacy_batch_size, n_process=args.n_process)
    for doc in _progress(docs, total=len(keep), desc="L1 verb->object"):
        pairs, verbs = tasks_from_doc(doc)
        task_counts.update(set(pairs))          # dedup within a clip
        verb_counts.update(verbs)
        noun_counts.update({o for _, o in pairs})

    write_pairs_csv(out_dir / "tasks_raw.csv", task_counts, ["verb", "object", "count"])
    write_vocab_csv(out_dir / "verb_vocab.csv", verb_counts, ["verb", "count"])
    write_vocab_csv(out_dir / "noun_vocab.csv", noun_counts, ["object", "count"])

    n_canon = None
    if args.verb_map or args.noun_map:
        vinv = load_canon_map(Path(args.verb_map)) if args.verb_map else {}
        ninv = load_canon_map(Path(args.noun_map)) if args.noun_map else {}
        canon: Counter = Counter()
        for (v, o), c in task_counts.items():
            canon[(vinv.get(v, v), ninv.get(o, o))] += c
        write_pairs_csv(out_dir / "tasks_canonical.csv", canon, ["verb", "object", "count"])
        n_canon = len(canon)

    print("=" * 60)
    print(f"valid clips      : {len(records)}  (L1 present: {len(keep)})")
    print(f"unique tasks (raw): {len(task_counts)}   verbs: {len(verb_counts)}   objects: {len(noun_counts)}")
    if n_canon is not None:
        print(f"unique tasks (canonical, after synonym merge): {n_canon}")
    else:
        print("no --verb_map/--noun_map -> tasks_canonical.csv not written")
        print("  next: cluster_synonyms_qwen.py over verb_vocab.csv / noun_vocab.csv, edit the")
        print("        JSON maps, then re-run with --verb_map/--noun_map to collapse synonyms.")
    print(f"-> {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
