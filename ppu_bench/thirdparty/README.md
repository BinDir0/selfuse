# Third-party real-codebase probes

Dummy microbenches prove the *kernels* work; these prove the *stack* works, by
running real public libraries on the PPU. Each is self-contained (only depends
on `_bench.py` in this folder) and skips cleanly if its library is absent.

Run inside the PPU vendor container. The inference probes (vLLM) want the
`inference-xpu-pytorch` image; the rest run under either.

| Probe | Library | What it validates on the PPU | Risk |
|---|---|---|---|
| `hf_qwen3vl.py` | transformers | EgoVLA's real backbone family (Qwen VL): vision tower + attention, sdpa vs flash | med |
| `vllm_smoke.py` | vLLM | full inference engine: PagedAttention, custom kernels, batching, throughput | med (natively supported) |
| `diffusers_smoke.py` | diffusers 0.34 | EgoVLA's diffusion/flow-head dependency: UNet + scheduler math | low |
| `lm_eval_parity.py` | transformers | **end-to-end accuracy** as a precision check — broken numerics tank the score | — |

```bash
scp -r thirdparty/ <ppu-host>:~/ppu_bench/ && cd ~/ppu_bench/thirdparty
python hf_qwen3vl.py --model Qwen/Qwen2.5-VL-3B-Instruct
python vllm_smoke.py --model Qwen/Qwen2.5-0.5B-Instruct
python diffusers_smoke.py
python lm_eval_parity.py --model Qwen/Qwen2.5-1.5B-Instruct
```

## Version note

EgoVLA targets Qwen3-VL (transformers ~5.x). PPU SDK 1.7 ships transformers
4.57.x, which may only carry Qwen2.5-VL — hence the defaults above use Qwen2.5.
If you install transformers 5.x, switch `--model Qwen/Qwen3-VL-2B-Instruct` to
test EgoVLA's exact backbone. See `../egovla/PORTING.md`.

## Other libraries worth adding

- **SGLang** — also natively supported on PPU; alternative serving engine.
- **lm-eval-harness** — the rigorous accuracy parity (command in `lm_eval_parity.py`).
- **TransformerEngine** (PPU ships TE 2.5) — but cuDNN FusedAttention is
  unavailable on PPU, so TE must use FlashAttention, not FusedAttention.
- **accelerate** — real FSDP/multi-GPU launching (see `../layer1_distributed.py`).
