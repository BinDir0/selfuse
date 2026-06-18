"""Real codebase: end-to-end accuracy as a precision sanity check.

A numerically-broken accelerator often still *runs* a model - it just produces
slightly-wrong logits that wreck downstream accuracy. This catches that: run a
real small instruct model on known-answer questions (greedy) and score it. A
healthy model scores high; a PPU with bad precision tanks.

    python lm_eval_parity.py --model Qwen/Qwen2.5-1.5B-Instruct

For the rigorous version, run lm-eval-harness and compare the score to the
published / GPU number:
    lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct,dtype=bfloat16 \\
            --tasks arc_easy --limit 200 --device cuda
"""

from __future__ import annotations

import argparse

import torch

from _bench import DEVICE, IS_ACCEL, fail, ok, section, skip

# (question, list of acceptable answer substrings, lowercased)
QA = [
    ("What is 17 plus 26? Answer with just the number.", ["43"]),
    ("What is 9 times 8? Answer with just the number.", ["72"]),
    ("What is the capital of France?", ["paris"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("Which planet is known as the Red Planet?", ["mars"]),
    ("How many continents are there on Earth?", ["7", "seven"]),
    ("What is the largest ocean on Earth?", ["pacific"]),
    ("In what year did World War II end?", ["1945"]),
    ("What gas do plants absorb from the atmosphere?", ["carbon dioxide", "co2"]),
    ("What is the square root of 144?", ["12", "twelve"]),
]


def run(model_id: str) -> None:
    section(f"accuracy parity probe: {model_id}")
    if not IS_ACCEL:
        skip("no accelerator")
        return
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        skip(f"transformers unavailable: {type(exc).__name__}")
        return

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(DEVICE).eval()

        correct = 0
        for q, answers in QA:
            msgs = [{"role": "user", "content": q}]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            ids = tok(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=24, do_sample=False)
            ans = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).lower()
            hit = any(a in ans for a in answers)
            correct += hit
            mark = "ok " if hit else "MISS"
            print(f"    [{mark}] {q[:42]:<42} -> {ans.strip()[:40]!r}")

        acc = correct / len(QA)
        msg = f"accuracy {correct}/{len(QA)} = {acc * 100:.0f}%"
        if acc >= 0.8:
            ok(msg + "  (healthy - precision looks fine end-to-end)")
        elif acc >= 0.5:
            print(f"  ⚠️  {msg}  (degraded - suspect precision/kernel issues)")
        else:
            fail(msg + "  (broken - likely a real numerical problem on the PPU)")
    except Exception as exc:  # noqa: BLE001
        fail(f"{type(exc).__name__}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    run(ap.parse_args().model)


if __name__ == "__main__":
    main()
