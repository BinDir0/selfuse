# PPU bench toolkit (Layers 0–2)

Vendor-agnostic PyTorch probes to answer: **does the Alibaba PPU run the
"latest DL features", correctly and fast?** Nothing here imports EgoVLA, so it
runs before the repo builds. The same harness runs on NVIDIA (A100/H20) for a
baseline.

## Run

```bash
# inside the PPU vendor torch container (training-xpu-pytorch ...)
scp -r ppu_bench/ <ppu-host>:~/ && ssh <ppu-host>
cd ~/ppu_bench

# set the PPU's dense bf16 peak so MFU% / GEMM% are meaningful (confirm spec!)
export PPU_PEAK_TFLOPS=148        # H20-class placeholder

python run_all.py                 # Layers 0–2 (single card)
python run_all.py --only ops compile      # subset

# multi-card (separate, needs torchrun). Do NOT source EgoVLA's NCCL_IB_* env.
torchrun --nproc_per_node=2 layer1_distributed.py
```

Results print live and land in `ppu_bench_report.json`.

## What each file probes

| File | Layer | Probes |
|---|---|---|
| `layer0_env.py` | 0 | torch/transformers/flash_attn/triton versions, device, SDPA backends, TF32 |
| `layer1_gemm.py` | 1 | GEMM TFLOPS sweep bf16/fp16/tf32/fp32 vs peak |
| `layer1_membw.py` | 1 | HBM bandwidth (copy / triad) |
| `layer1_ops.py` | 1 | **sdpa / flash_attn / flex_attention** + layernorm/softmax, each checked numerically |
| `layer1_compile.py` | 1 | `torch.compile` (default / max-autotune / cudagraphs) + **silent-fallback detection** |
| `layer1_models.py` | 1 | public-model smoke: resnet18, tiny GPT2, nn.Transformer |
| `feature_matrix.py` | 2 | autocast, fused AdamW, grad checkpointing, CUDA graphs, profiler/CUPTI, bitsandbytes |
| `layer1_distributed.py` | 1 | all_reduce busbw, FSDP2 `fully_shard` step, DeepSpeed ZeRO step |
| `repmodel.py` | — | shared representative transformer (Qwen3-VL-2B-ish dims) used below |
| `opt_ablation.py` | 4 | **worry #2**: fwd+bwd step time per optimization toggle vs naive baseline |
| `precision_parity.py` | — | **worry #1**: GPU golden capture → PPU compare, layer-wise divergence |

## Worry #1 — precision (GPU → PPU)

Per-op tolerance ≠ end-to-end fidelity. Capture a golden file on the NVIDIA box,
replay on the PPU, compare layer-by-layer:

```bash
# on the NVIDIA GPU where EgoVLA was trained
python precision_parity.py capture --out golden.pt
scp golden.pt <ppu-host>:~/ppu_bench/
# on the PPU
python precision_parity.py compare --golden golden.pt
```

It decomposes error into `PPU_bf16 vs GPU_bf16` (platform), `PPU_bf16 vs
GPU_fp32` (total), and compares the total against GPU's own bf16-vs-fp32 error.
Verdict: total error ≲ 2× GPU's bf16 error → fine; ≫ 5× → real divergence. Also
reports per-layer error growth and run-to-run nondeterminism. To test the real
EgoVLA model, import the module and pass your `(model, input)`.

## Worry #2 — do GPU optimizations survive

```bash
python opt_ablation.py        # or: python run_all.py --only ablation
```

Times one fwd+bwd step under each toggle (fp32/tf32/bf16 × sdpa/flash/flex ×
eager/compile/cudagraphs × grad-ckpt) and prints speedup vs naive baseline. If
the `compile` rows ≈ the `eager` rows, `torch.compile` isn't paying off on the PPU.

## Reading the statuses

| Status | Meaning |
|---|---|
| ✅ PASS | works and numerically correct |
| ❌ FAIL | errored, or returned wrong numbers — **the headline** |
| ➖ SKIP | dependency absent / N/A |
| 🐢 SLOW | runs but no speedup — suspect silent eager lowering |
| ⚠️ FALLBACK | runs but degraded (graph breaks, no CUDA timing, partial support) |

The mapping to EgoVLA: `flash_attn` PASS → keep `vision_attn_implementation:
flash_attention_2`; `flex_attention` FAIL/SLOW → set `text_attn_implementation:
sdpa`; `torch.compile` FAIL/SLOW → set `compile.enabled: False` and re-enable
per component. `profiler/CUPTI` FALLBACK → MFU% in `profiler_utils.py` stays
blank.

## Beyond the microbenches: real codebases

Microbenches prove kernels work; real codebases prove the *stack* works.

- **`thirdparty/`** — real public libraries on the PPU: HF transformers (Qwen
  VL backbone), **vLLM** (natively supported), diffusers, and an end-to-end
  accuracy parity probe. See `thirdparty/README.md`.
- **`egovla/`** — drive EgoVLA's own verification ladder
  (`pretrain_verification` / `full_chain_verification` / determinism / training
  smoke) via `egovla/run_ladder.sh`, after applying `egovla/PORTING.md` (the
  torch-2.8 / transformers-4.57 reconciliation + the verified repo footgun fixes).

Suggested order on a fresh PPU box: `layer0_env.py` → `run_all.py` →
`thirdparty/*` → apply `PORTING.md` → `egovla/run_ladder.sh` →
`precision_parity.py compare` (needs a GPU-captured `golden.pt`).
