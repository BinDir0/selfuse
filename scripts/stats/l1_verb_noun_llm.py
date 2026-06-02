#!/usr/bin/env python3
"""Verb + object vocabulary & frequency stats over the language levels via a small Qwen model
(vLLM), in ONE pass.

Replaces spaCy POS (which is grammatical and breaks on imperatives/gerunds/nominalizations:
"sewing"->NOUN, "inspection"->NOUN) with SEMANTIC extraction: the LLM classifies by meaning,
returns base forms ("sewing"->sew, "inspection"->inspect), uses context for words that are both
verb and noun ("drill a hole" vs "the drill"), keeps ONLY the manipulated object itself ("the
edge of the paper"->paper, not edge), and lists ALL verbs/objects when a sentence has several.

Levels: --levels (default level1..level5). L3-L5 are descriptive and routinely mention several
objects per sentence, so the stat is the per-clip UNION of verbs/objects over its levels, each
term counted once per clip ("#clips mentioning this term"). Identical level strings are inferred
ONCE and broadcast (templated annotations repeat a lot), so the GPU only sees UNIQUE texts.

It produces, in one run:
  * unique verbs   -> verb_freq.csv  (verb, count)   + verbs.unique in language_stats.json
  * unique objects -> noun_freq.csv  (noun, count)   + object_nouns.unique
  * the deduped per-clip-union frequency stats behind both.

A deterministic stop-list backstops the LLM (drops parts/features/directions/abstract words the
model may still emit); the raw extraction is saved to l1_extraction.json so you can re-tune the
stop-list offline with --from_extraction (no GPU).

Outputs (feed straight into plot_language_stats.py for the word clouds):
  verb_freq.csv, noun_freq.csv, language_stats.json, l1_extraction.json  (+ clouds if --wordcloud)

Speed: dedup + a 3B model makes this minutes on ONE A800. For more, --data_parallel N shards
the unique texts across N GPUs (independent tp=1 replicas -- the right pattern for a small
model; vLLM does continuous batching within each, so no manual batch size is needed).

Resumable: extractions are checkpointed to JSONL as they are produced (l1_extraction.jsonl for
single-process; _dp/shard_i.jsonl per data-parallel worker), flushed every --checkpoint_every.
If the run is killed, just rerun the SAME command -- it skips everything already done.

RUN ON THE PRODUCTION MACHINE (needs GPU + vllm). Example (8x A800):

    python scripts/stats/l1_verb_noun_llm.py \
        --annotation_root /efs-exp/guantianrui/buildai_6000_anno/ \
        --suffix _qwen-annotation.json \
        --out_dir ~/language_stat/output_llm \
        --model Qwen/Qwen2.5-3B-Instruct \
        --data_parallel 8 --gpu_memory_utilization 0.9 --wordcloud

Offline plumbing check (no GPU; validates parse/dedup/aggregate/CSV with empty extractions):
    python scripts/stats/l1_verb_noun_llm.py ... --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for _p in (str(PROJECT_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from language_annotation_stats import (  # noqa: E402
    NOUN_STOP,
    VERB_STOP,
    add_cache_args,
    h_index,
    load_annotations,
    maybe_wordcloud,
    resolve_cache,
)
from extract_verbs_objects_qwen import JSON_SCHEMA, parse_extraction  # noqa: E402

# Deterministic backstop for objects the LLM may still emit but that are NOT the manipulated
# thing itself: parts/features, spatial directions/adverbs, and abstract/nominal concepts.
# (NOUN_STOP already covers body parts + generic words like thing/item/area/side/part/top...)
_OBJECT_EXTRA_STOP = {
    "edge", "edges", "end", "ends", "corner", "corners", "surface", "surfaces",
    "face", "faces", "seam", "seams", "hole", "holes", "crack", "cracks",
    "scratch", "scratches", "gap", "gaps", "layer", "layers", "row", "rows",
    "column", "columns", "section", "sections", "region", "regions", "portion", "half",
    "rightward", "rightwards", "leftward", "leftwards", "upward", "upwards",
    "downward", "downwards", "forward", "forwards", "backward", "backwards",
    "inward", "inwards", "outward", "outwards", "clockwise", "counterclockwise",
    "alignment", "orientation", "angle", "angles", "distance", "length", "width",
    "height", "depth", "pressure", "force", "shape", "order", "manner", "amount",
}
_OBJECT_STOP = NOUN_STOP | _OBJECT_EXTRA_STOP

# Hardened, MEANING-based prompt: base-form verbs even from gerunds/nominalizations, and
# context for verb/noun-ambiguous tool words. Few-shot nails the exact failure modes.
SYSTEM_PROMPT = (
    "You read ONE short human-hand manipulation instruction and return the manipulation "
    "ACTIONS and the manipulated OBJECTS, by MEANING, not grammar. NOT every word is a verb or "
    "a noun -- ignore adverbs, spatial directions, and abstract words. Return ONLY a JSON "
    "object {\"verbs\": [...], \"objects\": [...]}.\n"
    "- verbs: each manipulation action as a base-form lowercase verb. Convert gerunds and "
    "nominalizations to the base verb: 'sewing'->sew, 'inspection'->inspect, 'assembly'->"
    "assemble. Exclude state/linking/perception verbs (be, remain, look, see, watch).\n"
    "- objects: ONLY the actual physical thing(s) the hand manipulates, lowercase singular, no "
    "article. Output the WHOLE object, NEVER a part/feature of it: 'the edge of the paper'->"
    "paper; 'the top of the box'->box; 'the panel surface'->panel.\n"
    "  NEVER put these in objects: parts/features (edge, end, side, corner, surface, top, "
    "bottom, hole, seam, crack, gap); directions/adverbs (left, right, up, rightward, "
    "clockwise); abstract concepts (alignment, orientation, position, angle, pressure); "
    "actions/verbs (insert, align, push); body parts; the person; generic words (thing, item, "
    "object, area). A word can be a tool object in one place and an action in another -- decide "
    "by context ('the drill' is an object; 'drill a hole' is the verb drill).\n"
    "One instruction may describe SEVERAL actions and SEVERAL objects (especially longer, "
    "descriptive sentences) -- list ALL of them, do not stop at one. Deduplicate within each "
    "array. If nothing qualifies, use an empty array."
)

_FEWSHOT = [
    ("Sewing the fabric edge along the seam.", {"verbs": ["sew"], "objects": ["fabric"]}),
    ("Inspection of the panel surface for scratches.", {"verbs": ["inspect"], "objects": ["panel"]}),
    ("Pick up the drill and drill a hole in the panel.",
     {"verbs": ["pick", "drill"], "objects": ["drill", "panel"]}),
    ("Slide the bracket rightward to adjust the alignment.",
     {"verbs": ["slide", "adjust"], "objects": ["bracket"]}),
    ("The left hand holds the bracket while the right hand inserts a screw and tightens it "
     "with a screwdriver.",
     {"verbs": ["hold", "insert", "tighten"], "objects": ["bracket", "screw", "screwdriver"]}),
]


def build_messages(text: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user, ans in _FEWSHOT:
        msgs.append({"role": "user", "content": f"Instruction:\n{user}\n\nJSON:"})
        msgs.append({"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)})
    msgs.append({"role": "user", "content": f"Instruction:\n{text}\n\nJSON:"})
    return msgs


def _chunked(seq, n):
    n = max(1, int(n))
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _load_done_jsonl(path) -> dict:
    """Load {text: {verbs, objects}} from a resumable extraction JSONL (one record per line)."""
    done: dict = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done[r["text"]] = {"verbs": r.get("verbs", []), "objects": r.get("objects", [])}
                except Exception:
                    continue
    return done


def run_single(texts: list[str], args, jsonl_path: str, keep_raw: bool = False) -> dict:
    """Extract {verbs, objects} per text, RESUMABLY: each result is appended to jsonl_path and
    flushed every --checkpoint_every, so a rerun (same args) skips what's already there. The model
    is loaded once and only the not-yet-done texts are inferred. Returns results for `texts`."""
    done = _load_done_jsonl(jsonl_path)
    todo = [t for t in texts if t not in done]
    print(f"[infer] texts={len(texts)}  resume_done={len(done)}  to_do={len(todo)}  -> {jsonl_path}",
          flush=True)
    if todo:
        from vllm import LLM, SamplingParams

        llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
                  gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len)
        tok = llm.get_tokenizer()
        sp_kwargs = dict(temperature=0.0, max_tokens=args.max_tokens)
        if not getattr(args, "no_guided", False):
            try:
                from vllm.sampling_params import GuidedDecodingParams
                sp_kwargs["guided_decoding"] = GuidedDecodingParams(json=JSON_SCHEMA)
            except Exception:
                pass
        try:
            sp = SamplingParams(**sp_kwargs)
        except TypeError:
            sp_kwargs.pop("guided_decoding", None)
            sp = SamplingParams(**sp_kwargs)

        os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
        n = len(done)
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            for chunk in _chunked(todo, args.checkpoint_every):
                prompts = [tok.apply_chat_template(build_messages(t), tokenize=False,
                                                   add_generation_prompt=True) for t in chunk]
                outputs = llm.generate(prompts, sp)
                for t, o in zip(chunk, outputs):
                    raw = o.outputs[0].text
                    p = parse_extraction(raw)
                    fh.write(json.dumps({"text": t, "verbs": p["verbs"], "objects": p["objects"]},
                                        ensure_ascii=False) + "\n")
                    entry = {"verbs": p["verbs"], "objects": p["objects"]}
                    if keep_raw:
                        entry["_raw"] = raw
                    done[t] = entry
                fh.flush()
                n += len(chunk)
                print(f"[ckpt] {n}/{len(texts)} done (+{len(chunk)}) -> {jsonl_path}", flush=True)
    return {t: done.get(t, {"verbs": [], "objects": []}) for t in texts}


def _shard(lst, n):
    k = (len(lst) + n - 1) // n
    return [lst[i * k:(i + 1) * k] for i in range(n) if i * k < len(lst)]


def run_data_parallel(texts: list[str], args, out_dir: Path) -> dict:
    """Shard unique texts across N GPUs as independent tp=1 worker processes; merge results.

    Sharding is DETERMINISTIC (contiguous split of the same frequency-sorted texts), and each
    worker checkpoints to its own resumable JSONL (_dp/shard_i.jsonl). So if the run is killed,
    just rerun the SAME command: every worker resumes its shard where it left off."""
    shards = _shard(texts, args.data_parallel)
    gpu_ids = ([s.strip() for s in args.gpu_ids.split(",") if s.strip()]
               if args.gpu_ids else [str(i) for i in range(args.data_parallel)])
    dp_dir = out_dir / "_dp"
    dp_dir.mkdir(parents=True, exist_ok=True)
    procs, jsonls = [], []
    for i, sh in enumerate(shards):
        fin, fjsonl = dp_dir / f"shard_{i}.json", dp_dir / f"shard_{i}.jsonl"
        fin.write_text(json.dumps(sh, ensure_ascii=False), encoding="utf-8")
        jsonls.append(fjsonl)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[i % len(gpu_ids)]
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--shard_in", str(fin), "--shard_jsonl", str(fjsonl),
               "--model", args.model, "--max_tokens", str(args.max_tokens),
               "--gpu_memory_utilization", str(args.gpu_memory_utilization),
               "--max_model_len", str(args.max_model_len), "--tensor_parallel_size", "1",
               "--checkpoint_every", str(args.checkpoint_every)]
        if args.no_guided:
            cmd.append("--no_guided")
        print(f"[dp] launch shard {i}/{len(shards)} on GPU {env['CUDA_VISIBLE_DEVICES']} "
              f"({len(sh)} unique texts)", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))
    rcs = [p.wait() for p in procs]
    if any(rc != 0 for rc in rcs):
        raise SystemExit(f"[dp] a worker failed (return codes={rcs}); FINISHED work is checkpointed "
                         f"in {dp_dir} -- rerun the SAME command to resume from where it stopped.")
    merged = {}
    for fj in jsonls:
        merged.update(_load_done_jsonl(str(fj)))
    return merged


def write_freq_csv(path: Path, counter: Counter, header):
    import csv
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        # integer counts so plot_language_stats.py (which does int()) reads them
        w.writerows((term, int(round(c))) for term, c in counter.most_common())


def load_durations(path: str) -> dict:
    """clip_id -> length weight (frames or seconds). Accepts JSON {id: num} or CSV 'id,num'.
    Keyed by the annotation file stem (== record clip_id)."""
    import csv
    p = Path(path).expanduser()
    if p.suffix.lower() == ".json":
        return {str(k): float(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    out: dict = {}
    with p.open("r", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                try:
                    out[str(row[0]).strip()] = float(row[1])
                except ValueError:
                    continue  # header / non-numeric
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotation_root")
    ap.add_argument("--out_dir")
    ap.add_argument("--suffix", default=".annotation.json")
    ap.add_argument("--parse_workers", type=int, default=32)
    add_cache_args(ap)
    ap.add_argument("--top_k", type=int, default=60)
    ap.add_argument("--levels", default="level1,level2,level3,level4,level5",
                    help="Comma list of hierarchy levels to extract from. L3-L5 are descriptive "
                         "and routinely mention SEVERAL objects per sentence; the per-clip union "
                         "over these levels is what feeds the verb/noun stats.")
    ap.add_argument("--wordcloud", action="store_true", help="Also emit verbs/nouns word clouds here.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Infer only the first N (most frequent) UNIQUE level-texts -- a fast trial.")
    ap.add_argument("--print_samples", type=int, default=0,
                    help="Print this many 'text -> verbs/objects' rows for eyeballing.")
    ap.add_argument("--from_extraction", default=None,
                    help="Skip the GPU: re-aggregate from a previously saved l1_extraction.json "
                         "(lets you re-tune the object stop-list / weighting offline).")
    ap.add_argument("--weight", choices=["clips", "duration"], default="clips",
                    help="clips: each clip counts once per term (task-instance diversity). "
                         "duration: weight each clip by its length from --durations (data volume "
                         "-- 'hours of footage involving the concept').")
    ap.add_argument("--durations", default=None,
                    help="clip_id -> length (frames or seconds) for --weight duration. JSON "
                         "{id:num} or CSV id,num, keyed by the annotation file stem (clip_id).")
    # model / vLLM
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--no_guided", action="store_true",
                    help="Disable JSON-schema guided decoding (use if outlines deps are broken, "
                         "e.g. missing pyairports). parse_extraction tolerates free-text JSON.")
    ap.add_argument("--data_parallel", type=int, default=1, help="Independent tp=1 replicas across N GPUs.")
    ap.add_argument("--gpu_ids", default=None, help="Comma list of GPU ids for --data_parallel (default 0..N-1).")
    ap.add_argument("--checkpoint_every", type=int, default=4000,
                    help="Append+flush extraction results to the resumable JSONL every N texts; a "
                         "rerun (same args) skips what's already done.")
    ap.add_argument("--dry_run", action="store_true", help="No model load; empty extractions (plumbing test).")
    # internal worker mode (used by --data_parallel)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--shard_in", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--shard_jsonl", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.worker:  # one data-parallel shard: resumably checkpoint to its shard JSONL
        texts = json.loads(Path(args.shard_in).read_text(encoding="utf-8"))
        run_single(texts, args, args.shard_jsonl)
        return

    if not args.annotation_root or not args.out_dir:
        raise SystemExit("--annotation_root and --out_dir are required")
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

    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    # Per clip, collect its chosen-level texts (L1 falls back to the first instruction). One
    # text can carry several objects, so we keep the per-clip list and union later. text_occ
    # counts identical strings across the corpus -> we infer each UNIQUE text once and broadcast.
    per_clip: list[tuple[str, list[str]]] = []   # (clip_id, level-texts)
    text_occ: Counter = Counter()
    for rec in records:
        ts = []
        for lv in levels:
            t = rec["levels"].get(lv)
            if not t and lv == "level1":
                t = rec["instruction"][0] if rec.get("instruction") else None
            if t:
                ts.append(t)
                text_occ[t] += 1
        per_clip.append((rec["clip_id"], ts))
    unique_texts = [t for t, _ in text_occ.most_common()]   # frequency-sorted (for --limit)
    n_unique, n_slots = len(unique_texts), sum(text_occ.values())
    if args.limit:
        unique_texts = unique_texts[: args.limit]
    print(f"[extract] clips: {len(records)}   levels: {','.join(levels)}   level-texts: {n_slots}   "
          f"unique: {n_unique}   dedup ratio: {n_unique / max(1, n_slots):.3f}   "
          f"{'inferring ' + str(len(unique_texts)) + ' (--limit)   ' if args.limit else ''}"
          f"model={args.model}", flush=True)

    ext_path = out_dir / "l1_extraction.json"
    ext_jsonl = out_dir / "l1_extraction.jsonl"   # resumable checkpoint store (single-process)
    if args.from_extraction:
        results = json.loads(Path(args.from_extraction).read_text(encoding="utf-8"))
        print(f"[extract] loaded {len(results)} cached extractions from {args.from_extraction} (no GPU)", flush=True)
    elif args.dry_run:
        results = {t: {"verbs": [], "objects": []} for t in unique_texts}
    elif args.data_parallel and args.data_parallel > 1:
        results = run_data_parallel(unique_texts, args, out_dir)
    else:
        results = run_single(unique_texts, args, str(ext_jsonl), keep_raw=bool(args.print_samples))

    if not args.dry_run and not args.from_extraction:
        slim = {t: {"verbs": r["verbs"], "objects": r["objects"]} for t, r in results.items()}
        ext_path.write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")
        print(f"[extract] saved {len(slim)} extractions -> {ext_path} "
              f"(re-tune stop-lists offline via --from_extraction)", flush=True)

    if args.print_samples:
        print("-" * 60)
        for t in unique_texts[: args.print_samples]:
            r = results.get(t) or {}
            print(f"[{text_occ[t]:>6}x] {t[:100]}\n         verbs={r.get('verbs')}  objects={r.get('objects')}")
            if not r.get("verbs") and not r.get("objects") and r.get("_raw") is not None:
                print(f"         RAW={r['_raw'][:240]!r}")   # why did it come back empty?
        print("-" * 60)

    # aggregate: per clip, UNION verbs/objects across ITS levels, then add the clip's WEIGHT
    # once per distinct term. weight = 1 (clips) or the clip's length (duration). Deterministic
    # stop-lists are a backstop to the LLM's classification.
    durations = None
    if args.weight == "duration":
        if not args.durations:
            raise SystemExit("--weight duration requires --durations <clip_id->length file>")
        durations = load_durations(args.durations)
        print(f"[weight] loaded {len(durations)} clip durations from {args.durations}", flush=True)
    verb_freq: Counter = Counter()
    noun_freq: Counter = Counter()
    n_missing_w = 0
    for clip_id, ts in per_clip:
        if durations is not None:
            w = durations.get(clip_id, 0.0)
            if w <= 0:
                n_missing_w += 1
                continue
        else:
            w = 1
        cv: set = set()
        co: set = set()
        for t in ts:
            r = results.get(t)
            if not r:
                continue
            cv.update(r.get("verbs", []))
            co.update(r.get("objects", []))
        for v in cv:
            if v and v not in VERB_STOP:
                verb_freq[v] += w
        for o in co:
            if o and o not in _OBJECT_STOP:
                noun_freq[o] += w
    if durations is not None:
        known = len(per_clip) - n_missing_w
        print(f"[weight] duration-weighted; clips with a known duration: {known}/{len(per_clip)} "
              f"(missing {n_missing_w} -- check the clip_id join key)", flush=True)

    write_freq_csv(out_dir / "verb_freq.csv", verb_freq, ["verb", "count"])
    write_freq_csv(out_dir / "noun_freq.csv", noun_freq, ["noun", "count"])

    wc = {"verbs": False, "nouns": False}
    if args.wordcloud:
        wc["verbs"] = maybe_wordcloud(verb_freq, out_dir / "verbs_wordcloud.png")
        wc["nouns"] = maybe_wordcloud(noun_freq, out_dir / "nouns_wordcloud.png")

    summary = {
        "source": "l1_verb_noun_llm",
        "model": args.model,
        "levels": levels,
        "weight": args.weight,
        "duration_weighted_clips": (len(per_clip) - n_missing_w) if durations is not None else None,
        "coverage": {**coverage,
                     "valid_language_coverage": (coverage["valid"] / coverage["files_found"]
                                                 if coverage["files_found"] else 0.0)},
        "dedup": {"clips": len(records), "level_texts": n_slots, "unique_texts": n_unique,
                  "dedup_ratio": n_unique / max(1, n_slots)},
        "verbs": {"unique": len(verb_freq), "total_occurrences": sum(verb_freq.values()),
                  "h_index": h_index(verb_freq), "top": verb_freq.most_common(args.top_k)},
        "object_nouns": {"unique": len(noun_freq), "total_occurrences": sum(noun_freq.values()),
                         "h_index": h_index(noun_freq), "top": noun_freq.most_common(args.top_k)},
        "outputs": {"verb_freq_csv": "verb_freq.csv", "noun_freq_csv": "noun_freq.csv",
                    "l1_extraction_json": "l1_extraction.json",
                    "verbs_wordcloud_png": "verbs_wordcloud.png" if wc["verbs"] else None,
                    "nouns_wordcloud_png": "nouns_wordcloud.png" if wc["nouns"] else None},
    }
    (out_dir / "language_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"unique verbs : {len(verb_freq)}  (h-index {summary['verbs']['h_index']})")
    print(f"unique nouns : {len(noun_freq)}  (h-index {summary['object_nouns']['h_index']})")
    print(f"top verbs    : {[v for v, _ in verb_freq.most_common(12)]}")
    print(f"top nouns    : {[n for n, _ in noun_freq.most_common(12)]}")
    if args.dry_run:
        print("DRY RUN: extractions empty (plumbing only). Re-run without --dry_run on a GPU box.")
    print(f"-> {out_dir}  (plot: python scripts/stats/plot_language_stats.py --stats_dir {out_dir} --min_count 50)")
    print("=" * 60)


if __name__ == "__main__":
    main()
