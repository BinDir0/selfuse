# Pretrain Verification Suite

LegendVLA 万小时级训练前全链路正确性验证工具集。按数据流顺序组织为 10 个阶段（Phase 0-9），每个阶段包含独立可运行的验证项，同时提供可视化（PNG 图表）和定量（JSON 报告 + 断言）两个维度的检查。

## Quick Start

```bash
# 不需要真实模型权重的阶段（Phase 2-9）:
python -m src.tests.pretrain_verification.run_all

# 全部阶段（需要数据路径）:
python -m src.tests.pretrain_verification.run_all \
    --config-path src/config/experiment/legendvla_qwen3_vl.yaml \
    --normalizer-path /path/to/normalizer.pkl

# 只跑特定阶段:
python -m src.tests.pretrain_verification.run_all --phases 4 56 8

# 跳过可视化（CI 模式）:
python -m src.tests.pretrain_verification.run_all --skip-visual
```

## Phases

| Phase | 文件 | 内容 | 需要外部数据 |
|-------|------|------|:---:|
| 0 | `phase0_normalizer.py` | Normalizer 统计量、往返一致性、归一化后分布 | `normalizer.pkl` |
| 1 | `phase1_data_pipeline.py` | Collator schema、VLA/VLM 比率、视觉输入、Token 序列、2D 投影 | config + 数据 |
| 2 | `phase2_embeddings.py` | State/Action Encoder 分布、Time Embedding、Fourier 确定性 | - |
| 3 | `phase3_forward_stability.py` | Activation 审计、Prefix KV Cache、BF16 精度 | - |
| 4 | `phase4_loss_verification.py` | CE/Flow/DiffLoss 正确性、加权平衡、梯度流 | - |
| 5+6 | `phase5_rtc_and_phase6_flow.py` | RTC delay/mask/noisy actions、psi_t、velocity target、Euler 积分 | - |
| 7 | `phase7_diffloss.py` | Zero-init、Chunk 构造、采样质量 | - |
| 8 | `phase8_training_loop.py` | Optimizer 参数组、LR schedule、梯度裁剪、收敛性、过拟合 | - |
| 9 | `phase9_inference.py` | Flow/RTC/AR/VLM 推理、Train-Infer 闭环 | - |

## Output

所有结果输出到 `outputs/pretrain_verification/{phase}/`，包括：
- `report.json` — 每个 Check 的 pass/fail 状态和详细指标
- `*.png` — 可视化图表（使用 `--skip-visual` 跳过）

## Priority

| 优先级 | 阶段 | 理由 |
|--------|------|------|
| **P0** | 0, 1, 3, 4, 6, 8 | Normalizer / 数据 / NaN / Loss / Flow / 收敛性 — 静默失败高风险 |
| **P1** | 2, 5, 7, 9 | Embedding / RTC / DiffLoss / 推理 — 辅助诊断 |

## Single Phase Usage

每个 phase 文件可以独立运行，详见各文件顶部的 docstring。示例：

```bash
# Phase 0（需要 normalizer）:
python -m src.tests.pretrain_verification.phase0_normalizer \
    --normalizer-path /path/to/normalizer.pkl

# Phase 4（无需外部数据）:
python -m src.tests.pretrain_verification.phase4_loss_verification

# Phase 1（需要完整 config）:
python -m src.tests.pretrain_verification.phase1_data_pipeline \
    --config-path src/config/experiment/legendvla_qwen3_vl.yaml
```
