#!/usr/bin/env python3
"""Extract action verbs + manipulated objects from 5-level language annotations
using an open-source Qwen model via vLLM offline batched inference.

Why an LLM instead of spaCy POS: L1 is an imperative verb-object phrase (easy),
but L2-L5 are descriptive sentences where every grammatical verb/noun is noise
("is/resting/wraps", "state/surface/side"). We want the *semantic* manipulation
action verbs and the *physical target objects*. We extract these PER LEVEL, and
each level may contain MULTIPLE verbs/objects (lists).

Pipeline (two stages — this is stage 1, the GPU step):
  1. THIS script -> per-clip extraction cache `extraction.jsonl`
  2. language_annotation_stats.py --extraction extraction.jsonl -> distributions

Output JSONL, one line per clip:
  {"clip_id": ..., "levels": {"level1": {"verbs": [...], "objects": [...]}, ...}}

Resumable: clips already present in the output file are skipped. Deterministic
(temperature 0). Reuses lib/pipeline/annotation_protocol for parsing so the input
matches the build exactly.

RUN ON THE PRODUCTION MACHINE (needs GPU + vllm). Example:

    python scripts/stats/extract_verbs_objects_qwen.py \
        --annotation_root /path/to/annotations \
        --out extraction.jsonl \
        --model Qwen/Qwen2.5-7B-Instruct \
        --tensor_parallel_size 1 --gpu_memory_utilization 0.9

Quick offline sanity check of the non-GPU plumbing (no model load):
    python scripts/stats/extract_verbs_objects_qwen.py ... --dry_run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.annotation_protocol import (  # noqa: E402
    HIERARCHY_KEYS,
    _normalize_hierarchy,
    _normalize_instruction,
)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verbs": {"type": "array", "items": {"type": "string"}},
        "objects": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verbs", "objects"],
}

SYSTEM_PROMPT = (
    "You extract structured information from a single human-hand manipulation "
    "instruction. Return ONLY a JSON object with two string arrays: \"verbs\" and "
    "\"objects\".\n"
    "- verbs: every concrete MANIPULATION ACTION in the text, as the base-form "
    "(infinitive) lowercase verb (e.g. open, pick, rotate, press). Include all of "
    "them if several actions are described; exclude state/linking verbs (is, are, "
    "remain, appear) and motion-of-camera/observation verbs (look, see, watch).\n"
    "- objects: every physical TARGET object being manipulated, as a lowercase "
    "singular noun (e.g. drawer, cup, handle, bottle cap). Include all of them; "
    "exclude body parts (hand, finger, arm, wrist), the person/agent, and generic "
    "words (object, thing, item, area, side).\n"
    "Deduplicate within each array. If none apply, use an empty array."
)


def build_messages(level_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Instruction:\n{level_text}\n\nJSON:"},
    ]


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_PUNCT_STRIP = re.compile(r"^[\s\W_]+|[\s\W_]+$")
_LEADING_ARTICLE = re.compile(r"^(?:a|an|the)\s+")


def normalize_terms(values) -> list[str]:
    """Lowercase, strip surrounding punctuation/space + leading article, drop empties, dedup (ordered)."""
    out: list[str] = []
    seen: set[str] = set()
    if not isinstance(values, list):
        return out
    for v in values:
        if not isinstance(v, str):
            continue  # skip None / non-string entries
        v = _PUNCT_STRIP.sub("", v.strip().lower())
        v = re.sub(r"\s+", " ", v)
        v = _LEADING_ARTICLE.sub("", v)
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def parse_extraction(text: str) -> dict:
    """Robustly pull {verbs, objects} from a model completion (tolerates fences/extra text)."""
    if not text:
        return {"verbs": [], "objects": [], "_parse_ok": False}
    match = _JSON_OBJ_RE.search(text)
    raw = match.group(0) if match else text
    try:
        data = json.loads(raw)
    except (ValueError, json.JSONDecodeError):
        return {"verbs": [], "objects": [], "_parse_ok": False}
    if not isinstance(data, dict):
        return {"verbs": [], "objects": [], "_parse_ok": False}
    return {
        "verbs": normalize_terms(data.get("verbs")),
        "objects": normalize_terms(data.get("objects")),
        "_parse_ok": True,
    }


def iter_valid_clips(root: Path, suffix: str):
    """Yield (clip_id, {level: text}) for valid annotations (status==Valid, non-empty)."""
    for path in sorted(root.rglob(f"*{suffix}")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("status", "Valid")).strip() != "Valid":
            continue
        hierarchy = _normalize_hierarchy(payload)
        if not _normalize_instruction(payload, hierarchy):
            continue
        levels = {k: hierarchy[k] for k in HIERARCHY_KEYS if hierarchy.get(k)}
        if levels:
            yield path.stem, levels


def load_done_ids(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["clip_id"])
        except (ValueError, KeyError):
            continue
    return done


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Qwen/vLLM verb+object extraction over 5-level annotations.")
    ap.add_argument("--annotation_root", required=True)
    ap.add_argument("--out", required=True, help="Output extraction JSONL (append/resume).")
    ap.add_argument("--suffix", default=".annotation.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--levels", default=",".join(HIERARCHY_KEYS), help="Comma list of levels to extract.")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N (un-done) clips.")
    ap.add_argument("--chunk_clips", type=int, default=2000, help="Clips per generate+flush chunk (resume granularity).")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--dry_run", action="store_true", help="Skip model load; emit empty extractions to validate plumbing.")
    args = ap.parse_args(argv)

    root = Path(args.annotation_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"annotation_root not found: {root}")
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    want_levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]

    done = load_done_ids(out_path)
    clips = [(cid, lv) for cid, lv in iter_valid_clips(root, args.suffix) if cid not in done]
    if args.limit is not None:
        clips = clips[: args.limit]
    print(f"[extract] root={root}  valid-to-do={len(clips)}  already-done={len(done)}  model={args.model}")
    if not clips:
        print("[extract] nothing to do.")
        return

    llm = None
    sampling_params = None
    tokenizer = None
    if not args.dry_run:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
        tokenizer = llm.get_tokenizer()
        sp_kwargs = dict(temperature=0.0, max_tokens=args.max_tokens)
        try:  # best-effort structured output; parser also tolerates free text
            from vllm.sampling_params import GuidedDecodingParams

            sp_kwargs["guided_decoding"] = GuidedDecodingParams(json=JSON_SCHEMA)
        except Exception:
            pass
        try:
            sampling_params = SamplingParams(**sp_kwargs)
        except TypeError:  # older vLLM without guided_decoding kwarg
            sp_kwargs.pop("guided_decoding", None)
            sampling_params = SamplingParams(**sp_kwargs)

    n_clips = 0
    n_calls = 0
    n_parse_fail = 0
    with out_path.open("a", encoding="utf-8") as out_fh:
        for chunk in chunked(clips, args.chunk_clips):
            # Flatten to one prompt per (clip, level); remember the mapping.
            prompts: list[str] = []
            index: list[tuple[int, str]] = []  # (clip_idx_in_chunk, level)
            for ci, (_cid, levels) in enumerate(chunk):
                for lv in want_levels:
                    text = levels.get(lv)
                    if not text:
                        continue
                    messages = build_messages(text)
                    if args.dry_run:
                        prompts.append(text)
                    else:
                        prompts.append(
                            tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            )
                        )
                    index.append((ci, lv))

            if args.dry_run:
                texts = ["" for _ in prompts]
            else:
                outputs = llm.generate(prompts, sampling_params)
                texts = [o.outputs[0].text for o in outputs]
            n_calls += len(prompts)

            # Reassemble per clip.
            per_clip: dict[int, dict] = {ci: {} for ci in range(len(chunk))}
            for (ci, lv), text in zip(index, texts):
                parsed = parse_extraction(text)
                if not parsed.get("_parse_ok", False):
                    n_parse_fail += 1
                per_clip[ci][lv] = {"verbs": parsed["verbs"], "objects": parsed["objects"]}

            for ci, (cid, _levels) in enumerate(chunk):
                out_fh.write(json.dumps(
                    {"clip_id": cid, "levels": per_clip[ci]}, ensure_ascii=False) + "\n")
                n_clips += 1
            out_fh.flush()
            print(f"[extract] flushed {n_clips}/{len(clips)} clips "
                  f"({n_calls} calls, parse_fail={n_parse_fail})")

    print(f"[extract] done: {n_clips} clips, {n_calls} level-calls, parse_fail={n_parse_fail} -> {out_path}")


if __name__ == "__main__":
    main()
