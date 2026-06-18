"""Real codebase: vLLM offline inference + throughput.

The PPU natively supports vLLM (built-in 0.10.2, acext kernels). This is the
heaviest real-stack test: PagedAttention, custom kernels, continuous batching,
CUDA graphs. We load a small Qwen LLM, generate a batch, and report tokens/s.

    python vllm_smoke.py --model Qwen/Qwen2.5-0.5B-Instruct
    python vllm_smoke.py --model Qwen/Qwen2.5-7B-Instruct --quant awq   # int4

If vLLM isn't installed (e.g. you booted the training image, not the inference
one), it skips. On PPU use the inference-xpu-pytorch image.
"""

from __future__ import annotations

import argparse
import time

from _bench import fail, ok, section, skip


def run(model_id: str, quant: str | None, n_prompts: int) -> None:
    section(f"vLLM offline: {model_id}" + (f" [{quant}]" if quant else ""))
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # noqa: BLE001
        skip(f"vLLM not importable: {type(exc).__name__} (use inference-xpu-pytorch image)")
        return

    prompts = [
        "Explain what a vision-language-action model does, in one sentence.",
        "List three prime numbers greater than 50.",
        "Translate 'good morning' into French.",
        "What is the capital of Japan?",
    ] * max(1, n_prompts // 4)

    try:
        kwargs = {"model": model_id, "dtype": "bfloat16", "max_model_len": 2048,
                  "gpu_memory_utilization": 0.85}
        if quant:
            kwargs["quantization"] = quant
        llm = LLM(**kwargs)
        sp = SamplingParams(temperature=0.0, max_tokens=64)

        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        elapsed = time.perf_counter() - t0

        gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        tps = gen_tokens / elapsed
        ok(f"{len(prompts)} prompts, {gen_tokens} tokens in {elapsed:.2f}s -> {tps:.1f} tok/s")
        ok(f"sample: {outs[0].outputs[0].text.strip()[:80]!r}")
    except Exception as exc:  # noqa: BLE001
        fail(f"{type(exc).__name__}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--quant", default=None, help="awq | gptq | None")
    ap.add_argument("--n-prompts", type=int, default=8)
    a = ap.parse_args()
    run(a.model, a.quant, a.n_prompts)


if __name__ == "__main__":
    main()
