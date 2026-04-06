# LegendVLA Full-Chain Verification Plan

> Goal: cover every potential error point across data loading, tokenization,
> model forward, loss computation, inference, and training loop.
> Check not just shape/dtype but **numerical content, semantic correctness,
> train-infer consistency, and convergence behavior**.

## Server Environment

| Item | Path |
|------|------|
| Code | `~/code/EgoVLA` |
| Data root | `~/data` (sub-dirs: `vla/`, `vlm/`, each with per-dataset folders) |
| Normalizer | `~/normalizers/2026.04.04-relative-mixed-all/normalizer.pkl` |
| Qwen3-VL weights | `~/checkpoints/Qwen3-VL-2B-Instruct` |
| Trained checkpoint | `~/checkpoints/update_step_100000` |
| Hydra config | `src/config/experiment/legendvla_qwen3_vl.yaml` |
| Python | System python (container, no venv needed) |

## Three-Machine Parallel Execution

| Machine | SSH Host | Tasks | Estimated Time |
|---------|----------|-------|---------------|
| **A** | `volc-transformers-exp-A100-1` | Part 0 (dataset audit) + Part 4 (flow math) + Part 1 (data pipeline) + Part 2 (collator) | ~10 min |
| **B** | `volc-transformers-exp-A100-2` | Part 3 (backbone/prefix cache) + Part 5 (expert/diffloss) | ~5 min |
| **C** | `volc-transformers-exp-A100-3` | Part 6 (inference parity) + Part 7 (E2E training + checkpoint) | ~8 min |

### Agent Workflow (per machine)

1. SSH login, `cd ~/code/EgoVLA`
2. Execute assigned Part scripts
3. Read generated `audit_report.json` + all PNG visualizations
4. Analyze each plot: distribution sanity, anomalies, NaN patterns
5. Write findings to `outputs/full_chain_verification/partN/REVIEW.md`
6. Report summary back (pass/fail per check, visualization analysis, file locations)

## Part Overview

| Part | Name | GPU | Real Data | Test Count |
|------|------|-----|-----------|------------|
| **Part 0** | Per-Dataset Audit | No | Yes | per-dataset |
| **Part 1** | Data Pipeline + Sample Level | No | Yes (shard) | 10 |
| **Part 2** | Collator / Tokenization / Mask | No (HF processor) | No (mock) | 23 |
| **Part 3** | Backbone Embedding + Prefix Cache | Yes | No (mock) | 11 |
| **Part 4** | Flow Matching Math + RTC Logic | No (pure math) | No | 26 |
| **Part 5** | Action Expert + DiffLoss + Loss | Yes | No (mock) | 13 |
| **Part 6** | Inference + Train-Infer Parity | Yes | No (mock) | 10 |
| **Part 7** | E2E Training + Checkpoint | Yes | Yes (optional shard) | 14 |
| **Part 8** | Accelerate Training Verification | Yes | No (config) | 13 |
| **Part 9** | Input Ablation (Real Data) | Yes | Yes (shard) | 19 |
| | | | **Total** | **139 + Part 0** |

---

## Machine A Commands

```bash
cd ~/code/EgoVLA

# Part 0: Per-dataset audit (all VLA + VLM sub-datasets)
python -m src.tests.full_chain_verification.part0_dataset_audit \
    --data-root ~/data \
    --normalizer-path ~/normalizers/2026.04.04-relative-mixed-all/normalizer.pkl \
    --model-path ~/checkpoints/Qwen3-VL-2B-Instruct \
    --max-samples 200

# Part 4: Flow math (CPU, no deps)
python -m src.tests.full_chain_verification.part4_flow_rtc_math

# Part 1: Data pipeline
python -m src.tests.full_chain_verification.part1_data_pipeline \
    --normalizer-path ~/normalizers/2026.04.04-relative-mixed-all/normalizer.pkl \
    --vla-shard "$(ls ~/data/vla/*/shard-*.tar | head -2 | tr '\n' ' ')"

# Part 2: Collator
python -m src.tests.full_chain_verification.part2_collator_tokenization \
    --model-path ~/checkpoints/Qwen3-VL-2B-Instruct
```

## Machine B Commands

```bash
cd ~/code/EgoVLA

# Part 3: Backbone + prefix cache
python -m src.tests.full_chain_verification.part3_backbone_prefix_cache \
    --config-path src/config/experiment/legendvla_qwen3_vl.yaml

# Part 5: Expert + DiffLoss + Loss
python -m src.tests.full_chain_verification.part5_expert_diffloss_loss \
    --config-path src/config/experiment/legendvla_qwen3_vl.yaml
```

## Machine C Commands

```bash
cd ~/code/EgoVLA

# Part 6: Inference parity
python -m src.tests.full_chain_verification.part6_inference_parity \
    --config-path src/config/experiment/legendvla_qwen3_vl.yaml

# Part 7: E2E training + checkpoint
python -m src.tests.full_chain_verification.part7_e2e_training \
    --config-path src/config/experiment/legendvla_qwen3_vl.yaml \
    --normalizer-path ~/normalizers/2026.04.04-relative-mixed-all/normalizer.pkl \
    --checkpoint-path ~/checkpoints/update_step_100000
```

## Output Structure

All outputs go to `~/code/EgoVLA/outputs/full_chain_verification/`:

```
outputs/full_chain_verification/
├── part0/
│   ├── audit_report.json          # structured per-dataset results
│   ├── REVIEW.md                  # agent analysis of visualizations
│   ├── egodex/
│   │   ├── field_boxplots.png     # per-field value distribution
│   │   ├── action_dim_hist.png    # 48D action per-dim histogram
│   │   ├── normalized_action_hist.png
│   │   ├── nan_heatmap.png        # sample x dim health indicator
│   │   └── sample_images.png      # image grid
│   ├── buildai/
│   │   └── ...
│   ├── FineVision/
│   │   ├── vlm_sample_images.png
│   │   └── text_length_hist.png
│   └── .../
├── part1/
│   ├── report.json
│   ├── normalizer_dist.png
│   └── raw_action_dist.png
├── part2/ ... part7/
│   └── report.json
└── part7/
    ├── report.json
    ├── lr_schedule.png
    └── overfit_loss.png
```

---

## Part 0: Per-Dataset Audit

**File**: `part0_dataset_audit.py`
**Deps**: webdataset + normalizer.pkl + (optional) Qwen3-VL processor
**Runs**: CPU

Per VLA dataset checks:
- Schema: required keys (`lowdim.npy`, `__key__`, image), lowdim shape `(116,)`
- Value health: NaN / Inf / all-zero sample count
- Per-field stats: mean, std, min, max for each of 6 lowdim fields
- Action range: max |action| < 1e6, warn if > 100
- Normalizer: outlier ratio (|normalized| > 5) < 10%
- Collator round-trip: real sample through `collate_raw` without error

Per VLM dataset checks:
- Schema: `meta.json` + `image_*.jpg` present
- Text stats: Q/A lengths, empty answer detection
- Image decodability

Visualizations per dataset:
- `field_boxplots.png` — 6 lowdim fields boxplot
- `action_dim_hist.png` — 48D per-dim histograms
- `normalized_action_hist.png` — post-normalizer distribution
- `nan_heatmap.png` — sample x dim health heatmap
- `sample_images.png` / `vlm_sample_images.png` — image grids
- `text_length_hist.png` — VLM Q/A length distribution

## Part 1: Data Pipeline + Sample Level

**File**: `part1_data_pipeline.py`  |  **Tests**: 8

- 1.1 Normalizer roundtrip, degenerate dims, output distribution
- 1.2 Relative action coordinate transform roundtrip
- 1.3 Sliding window config sanity (WindowConfig + LOWDIM_SLICES)
- 1.4 Real VLA sample schema + action values from shard
- 1.5 Real VLM sample schema from shard

## Part 2: Collator / Tokenization / Mask

**File**: `part2_collator_tokenization.py`  |  **Tests**: 23

- 2.1 VLA/VLM message structure, prompt_only mode
- 2.2 Special token counts, token order, answer_start_idx semantics
- 2.3 Labels: VLA all -100, VLM prompt masked / answer valid, padding masked
- 2.4 Attention mask: shape, right padding monotonic, pad_token_id
- 2.5 Visual tokens: video/image grid_thw consistency, mixed batch
- 2.6 Camera intrinsic text mode
- 2.7 Mixed batch: is_vla_data, labels, required keys
- 2.8 Varying action counts stress test

## Part 3: Backbone Embedding + Prefix Cache

**File**: `part3_backbone_prefix_cache.py`  |  **Tests**: 11

- 3.1 State embed replacement count, action noise training
- 3.2 Hidden states: shape, dtype, finite, not constant, answer region varies
- 3.3 Prefix cache: shape, mask content, KV meaningful, knowledge insulation
- 3.4 Visual embed non-zero

## Part 4: Flow Matching Math + RTC Logic

**File**: `part4_flow_rtc_math.py`  |  **Tests**: 26

- 4.1 psi_t: boundary (t=0, t=1), linearity, monotonicity, 2D time
- 4.2 Velocity target: analytic formula, finite-difference match
- 4.3 Flow time sampling: beta range/mean, uniform stratified
- 4.4 build_flow_inputs: shapes T=1/T=4, noisy action content, zero-mask loss, actions repeated
- 4.5 RTC: delay range (uniform/exp/clamped), prefix mask, token times, clean prefix, loss mask
- 4.6 build_dense_diffloss_inputs: chunk unfold, hidden positions, mask count, repeat_interleave

## Part 5: Action Expert + DiffLoss + Loss

**File**: `part5_expert_diffloss_loss.py`  |  **Tests**: 13

- 5.1 4D attention mask: prefix visible, same-chunk visible, T=1 bidirectional, invalid blocked
- 5.2 DiffLoss dense inputs: chunk unfold content, hidden positions, short action mask
- 5.3 DiffLoss module: zero-init, positive loss, sample shape
- 5.4 Total loss: pure VLA ce=0, all finite, gradients to expert/diffloss

## Part 6: Inference + Train-Infer Parity

**File**: `part6_inference_parity.py`  |  **Tests**: 10

- 6.1 Flow inference: shape, finite, invalid zeroed, deterministic, step count effect
- 6.2 AR inference: shape, finite
- 6.3 VLM inference: generates tokens, max_new_tokens respected
- 6.4 Train-infer parity: backbone hidden, prefix cache
- 6.5 RTC: prefix preservation in inference

## Part 7: E2E Training + Checkpoint

**File**: `part7_e2e_training.py`  |  **Tests**: 10

- 7.1 Parameter groups: all trainable in optimizer, no duplicates, frozen excluded
- 7.2 LR schedule: warmup + cosine curve shape
- 7.3 Overfit single batch 100 steps: loss decreases > 5%
- 7.4 Gradient health: per-component norms, no unused parameters
- 7.5 Batch dtypes correctness
- 7.6 Checkpoint: load without errors, inference produces reasonable values

## Part 8: Accelerate Training Verification

**File**: `part8_accelerate_training.py`  |  **Tests**: 16

- 8.1 FSDP wrap targets importable and present in model
- 8.2 Parameter groups: coverage, no duplicates, lr/wd/betas match config, decay/nodecay split
- 8.3 Production LR schedule shape, VLM freeze scheduler
- 8.4 preprocess_batch: required keys, dtypes
- 8.5 Gradient clipping: per-component param lists, clip values, clip isolation
- 8.6 NaN guard: scalar_metric_value + isfinite, skip logic
- 8.7 Gradient accumulation: 2 micro-steps → finite gradients

## Part 9: Input Ablation (Real Data)

**File**: `part9_input_ablation.py`  |  **Tests**: 19

- 9.1 Visual ablation: zeroing visual embeddings changes inference output / flow loss
- 9.2 State ablation: zeroing states changes inference output / flow loss
- 9.3 Per-field state ablation: wrist_state / hand_state individually affect output
- 9.4 Instruction/text ablation: different instructions → different actions, zeroing text embeddings
- 9.5 Component ablation (hooks): state_encoder / time_embedding / action_decoder
- 9.6 DiffLoss condition isolation: zeroing latent_condition_projector changes diffusion_loss but NOT flow_loss
- 9.7 Prefix cache ablation: zeroed prefix KV cache → very different actions
- 9.8 Action leakage: zeroed backbone action input does NOT change flow_loss, ar_action_encoder not called during inference
- 9.9 Attention weight distribution: temporarily switch to eager attention, measure prefix vs action attention share per layer, breakdown by visual/state/text tokens, per-head analysis, heatmap visualization
