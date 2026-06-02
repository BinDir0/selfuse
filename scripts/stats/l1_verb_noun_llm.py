#!/usr/bin/env python3
"""L1 verb + object vocabulary & frequency stats via a small Qwen model (vLLM), in ONE pass.

Replaces spaCy POS (which is grammatical and breaks on imperatives/gerunds/nominalizations:
"sewing"->NOUN, "inspection"->NOUN) with SEMANTIC extraction: the LLM classifies by meaning
and returns base forms ("sewing"->sew, "inspection"->inspect), and uses sentence context for
words that are both verb and noun ("drill a hole" vs "the drill").

It produces, in one run, all three things you want:
  * unique verbs   -> verb_freq.csv  (verb, count)   + verbs.unique in language_stats.json
  * unique objects -> noun_freq.csv  (noun, count)   + object_nouns.unique
  * L1 deduped freq: each clip's L1 contributes its SET of verbs/objects once; counts are
    "#clips using this term". Identical L1 strings are inferred ONCE and broadcast (templated
    annotations repeat a lot), so the GPU only sees the UNIQUE L1 texts.

Outputs (feed straight into plot_language_stats.py for the word clouds):
  verb_freq.csv, noun_freq.csv, language_stats.json   (+ word clouds if --wordcloud)

Speed: dedup + a 3B model makes this minutes on ONE A800. For more, --data_parallel N shards
the unique texts across N GPUs (independent tp=1 replicas -- the right pattern for a small
model; vLLM does continuous batching within each, so no manual batch size is needed).

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
    add_cache_args,
    h_index,
    load_annotations,
    maybe_wordcloud,
    resolve_cache,
)
from extract_verbs_objects_qwen import JSON_SCHEMA, parse_extraction  # noqa: E402

# Hardened, MEANING-based prompt: base-form verbs even from gerunds/nominalizations, and
# context for verb/noun-ambiguous tool words. Few-shot nails the exact failure modes.
SYSTEM_PROMPT = (
    "You read ONE short human-hand manipulation instruction and return the manipulation "
    "ACTIONS and the manipulated OBJECTS, classified by MEANING, not grammar. Return ONLY a "
    "JSON object {\"verbs\": [...], \"objects\": [...]}.\n"
    "- verbs: every manipulation action as a base-form (infinitive) lowercase verb. Convert "
    "gerunds and nominalizations to the base verb: 'sewing'->sew, 'inspection'->inspect, "
    "'assembly'->assemble, 'installation'->install. Exclude state/linking/perception verbs "
    "(be, remain, look, see, watch).\n"
    "- objects: every physical TARGET object, lowercase singular, no article. Exclude body "
    "parts (hand, finger, arm, wrist), the person, and generic words (thing, item, area, "
    "side). A word can be a tool object in one place and an action in another -- decide by "
    "context: in 'the drill' it is an object; in 'drill a hole' the verb is drill.\n"
    "Deduplicate within each array. If none apply, use an empty array."
)

_FEWSHOT = [
    ("Sewing the fabric edge along the seam.", {"verbs": ["sew"], "objects": ["fabric", "seam"]}),
    ("Inspection of the welded joint for cracks.", {"verbs": ["inspect"], "objects": ["joint"]}),
    ("Pick up the drill and drill a hole in the panel.",
     {"verbs": ["pick", "drill"], "objects": ["drill", "hole", "panel"]}),
]


def build_messages(text: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user, ans in _FEWSHOT:
        msgs.append({"role": "user", "content": f"Instruction:\n{user}\n\nJSON:"})
        msgs.append({"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)})
    msgs.append({"role": "user", "content": f"Instruction:\n{text}\n\nJSON:"})
    return msgs


def _l1_text(rec) -> str:
    return rec["levels"].get("level1") or (rec["instruction"][0] if rec.get("instruction") else "")


def run_single(texts: list[str], args) -> dict:
    """Load vLLM (tensor_parallel_size as given) and extract {verbs, objects} per text."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len)
    tok = llm.get_tokenizer()
    sp_kwargs = dict(temperature=0.0, max_tokens=args.max_tokens)
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

    prompts = [tok.apply_chat_template(build_messages(t), tokenize=False, add_generation_prompt=True)
               for t in texts]
    outputs = llm.generate(prompts, sp)
    res = {}
    for t, o in zip(texts, outputs):
        p = parse_extraction(o.outputs[0].text)
        res[t] = {"verbs": p["verbs"], "objects": p["objects"]}
    return res


def _shard(lst, n):
    k = (len(lst) + n - 1) // n
    return [lst[i * k:(i + 1) * k] for i in range(n) if i * k < len(lst)]


def run_data_parallel(texts: list[str], args, out_dir: Path) -> dict:
    """Shard unique texts across N GPUs as independent tp=1 worker processes; merge results."""
    shards = _shard(texts, args.data_parallel)
    gpu_ids = ([s.strip() for s in args.gpu_ids.split(",") if s.strip()]
               if args.gpu_ids else [str(i) for i in range(args.data_parallel)])
    dp_dir = out_dir / "_dp"
    dp_dir.mkdir(parents=True, exist_ok=True)
    procs, out_files = [], []
    for i, sh in enumerate(shards):
        fin, fout = dp_dir / f"shard_{i}.json", dp_dir / f"shard_{i}.out.json"
        fin.write_text(json.dumps(sh, ensure_ascii=False), encoding="utf-8")
        out_files.append(fout)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[i % len(gpu_ids)]
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--shard_in", str(fin), "--shard_out", str(fout),
               "--model", args.model, "--max_tokens", str(args.max_tokens),
               "--gpu_memory_utilization", str(args.gpu_memory_utilization),
               "--max_model_len", str(args.max_model_len), "--tensor_parallel_size", "1"]
        print(f"[dp] launch shard {i}/{len(shards)} on GPU {env['CUDA_VISIBLE_DEVICES']} "
              f"({len(sh)} unique texts)", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))
    rcs = [p.wait() for p in procs]
    if any(rc != 0 for rc in rcs):
        raise SystemExit(f"[dp] a worker failed (return codes={rcs}); see logs above")
    merged = {}
    for fout in out_files:
        merged.update(json.loads(Path(fout).read_text(encoding="utf-8")))
    return merged


def write_freq_csv(path: Path, counter: Counter, header):
    import csv
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(counter.most_common())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotation_root")
    ap.add_argument("--out_dir")
    ap.add_argument("--suffix", default=".annotation.json")
    ap.add_argument("--parse_workers", type=int, default=32)
    add_cache_args(ap)
    ap.add_argument("--top_k", type=int, default=60)
    ap.add_argument("--wordcloud", action="store_true", help="Also emit verbs/nouns word clouds here.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Infer only the first N (most frequent) UNIQUE L1 texts -- a fast accuracy/speed trial.")
    ap.add_argument("--print_samples", type=int, default=0,
                    help="Print this many 'L1 text -> verbs/objects' rows for eyeballing.")
    # model / vLLM
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--data_parallel", type=int, default=1, help="Independent tp=1 replicas across N GPUs.")
    ap.add_argument("--gpu_ids", default=None, help="Comma list of GPU ids for --data_parallel (default 0..N-1).")
    ap.add_argument("--dry_run", action="store_true", help="No model load; empty extractions (plumbing test).")
    # internal worker mode (used by --data_parallel)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--shard_in", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--shard_out", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.worker:  # one data-parallel shard: run vLLM (tp=1) and dump {text: {verbs,objects}}
        texts = json.loads(Path(args.shard_in).read_text(encoding="utf-8"))
        Path(args.shard_out).write_text(json.dumps(run_single(texts, args), ensure_ascii=False),
                                        encoding="utf-8")
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

    l1_texts = [t for t in (_l1_text(rec) for rec in records) if t]
    text_count = Counter(l1_texts)          # identical L1 strings -> infer once, broadcast
    unique_texts = [t for t, _ in text_count.most_common()]   # frequency-sorted
    n_unique = len(unique_texts)
    if args.limit:
        unique_texts = unique_texts[: args.limit]
    print(f"[l1] clips with L1: {len(l1_texts)}   unique L1 texts: {n_unique}   "
          f"dedup ratio: {n_unique / max(1, len(l1_texts)):.3f}   "
          f"{'inferring ' + str(len(unique_texts)) + ' (--limit)   ' if args.limit else ''}"
          f"model={args.model}", flush=True)

    if args.dry_run:
        results = {t: {"verbs": [], "objects": []} for t in unique_texts}
    elif args.data_parallel and args.data_parallel > 1:
        results = run_data_parallel(unique_texts, args, out_dir)
    else:
        results = run_single(unique_texts, args)

    if args.print_samples:
        print("-" * 60)
        for t in unique_texts[: args.print_samples]:
            r = results.get(t) or {}
            print(f"[{text_count[t]:>6}x] {t[:90]}\n         verbs={r.get('verbs')}  objects={r.get('objects')}")
        print("-" * 60)

    # aggregate: each clip contributes its L1 SET once -> weight unique-text terms by clip count
    verb_freq: Counter = Counter()
    noun_freq: Counter = Counter()
    for text, n_clips in text_count.items():
        r = results.get(text) or {}
        for v in r.get("verbs", []):
            verb_freq[v] += n_clips
        for o in r.get("objects", []):
            noun_freq[o] += n_clips

    write_freq_csv(out_dir / "verb_freq.csv", verb_freq, ["verb", "count"])
    write_freq_csv(out_dir / "noun_freq.csv", noun_freq, ["noun", "count"])

    wc = {"verbs": False, "nouns": False}
    if args.wordcloud:
        wc["verbs"] = maybe_wordcloud(verb_freq, out_dir / "verbs_wordcloud.png")
        wc["nouns"] = maybe_wordcloud(noun_freq, out_dir / "nouns_wordcloud.png")

    summary = {
        "source": "l1_verb_noun_llm",
        "model": args.model,
        "level": "level1",
        "coverage": {**coverage,
                     "valid_language_coverage": (coverage["valid"] / coverage["files_found"]
                                                 if coverage["files_found"] else 0.0)},
        "dedup": {"clips_with_l1": len(l1_texts), "unique_l1_texts": len(unique_texts),
                  "dedup_ratio": len(unique_texts) / max(1, len(l1_texts))},
        "verbs": {"unique": len(verb_freq), "total_occurrences": sum(verb_freq.values()),
                  "h_index": h_index(verb_freq), "top": verb_freq.most_common(args.top_k)},
        "object_nouns": {"unique": len(noun_freq), "total_occurrences": sum(noun_freq.values()),
                         "h_index": h_index(noun_freq), "top": noun_freq.most_common(args.top_k)},
        "outputs": {"verb_freq_csv": "verb_freq.csv", "noun_freq_csv": "noun_freq.csv",
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
