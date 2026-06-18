# Porting EgoVLA onto the Alibaba PPU

EgoVLA is hard-pinned to **torch 2.10.0+cu128 / transformers 5.6.2**; PPU SDK
1.7 ships **torch 2.8 / transformers 4.57.1 / CUDA 12.9**. You cannot install
the repo's pinned torch on the PPU — run inside the vendor
`training-xpu-pytorch` image and patch the repo *down* to it. Do these edits
before running `run_ladder.sh`.

## 1. Dependencies (do NOT `pip install -r requirements.txt` blindly)

| requirements.txt line | action on PPU |
|---|---|
| `torch==2.10.0+cu128` etc. | ❌ skip — use the vendor image's torch 2.8 |
| `flash-attn` | ❌ skip pip build — use the PPU's bundled FlashAttention |
| `xformers==0.0.35` | ❌ drop — not imported anywhere in `src/` |
| `transformers==5.6.2` | ⚠️ **the crux** — see §4 |
| `bitsandbytes` | install but keep unused (only the optional 4-bit *load* path touches it) |
| everything else (hydra, diffusers, einops, webdataset, …) | ✅ install |

## 2. Apply `ppu.patch` (non-destructive — NVIDIA defaults unchanged)

```bash
cd <EgoVLA repo on the PPU box>
git apply ppu_bench/egovla/ppu.patch       # verified: reverse-check passed
```

It makes three changes, none of which alter the NVIDIA happy path:

1. **Adds** `src/config/experiment/legendvla_qwen3_vl_ppu.yaml` — a Hydra
   overlay that inherits the standard experiment and forces the safe baseline
   (`torch.compile` OFF, text attention `sdpa`; vision stays FA2). Launch with
   `experiment=legendvla_qwen3_vl_ppu`. The base configs are untouched, so the
   NVIDIA training path is unaffected — get a *correct* run here, then re-enable
   accelerated paths one at a time (vision FA2 → per-component compile).
2. **`src/utils/profiler_utils.py`** — `device_peak_tflops` now reads
   `EGOVLA_DEVICE_PEAK_TFLOPS` when not passed (default still 312/A800). Export
   `EGOVLA_DEVICE_PEAK_TFLOPS=<PPU bf16 peak>` so MFU% is meaningful.
3. **`scripts/debug_start.sh`** — the NVIDIA InfiniBand/RDMA env (`mlx5` HCAs,
   `eth0`, GDR) is now guarded; set `EGOVLA_DISABLE_NV_NCCL=1` on the PPU to skip
   it (otherwise the vendor CCL hangs). Default behavior on NVIDIA is unchanged.

`run_ladder.sh` already exports both env vars and uses the overlay experiment.

## 3. Footgun the patch does NOT cover

- `src/model/vlm/qwen3_vl_compile_patch.py:24` — top-level
  `from flash_attn import flash_attn_varlen_func`. Must resolve to the vendor
  FA build, or import fails before anything runs. Verify with
  `python -c "from flash_attn import flash_attn_varlen_func"` in the container.

## 4. The transformers gap (decide + iterate on the box)

The repo's pinned commit fixes Qwen3-VL video-RoPE + flex_attention deprecation
and assumes transformers 5.x (`qwen3_vl_2b.yaml` comment: "valid values in
transformers 5.3.0"). Two paths:

- **A (recommended): stay on PPU transformers 4.57.x.** May lack the
  `Qwen3VL` classes entirely → you'd run the Qwen2.5-VL backbone, or backport.
  Expect to touch the flex_attention call sites in `src/model/vlm/qwen3_expert.py`
  and `qwen3_vl_backbone.py`. This is real iteration, best done with the actual
  error messages in hand.
- **B: pip-install transformers 5.6.2 over the vendor image.** Quicker to try,
  but may break the vendor's adaptations — validate with the `thirdparty/`
  probes first.

Run `../layer0_env.py` first to see exactly which transformers/torch the image
has, and `../thirdparty/hf_qwen3vl.py` to see whether `Qwen3-VL` loads at all.
