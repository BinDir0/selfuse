# LegendVLA with Qwen3-VL — 迁移计划与实施记录

## 概述

本文档定义了 LegendVLA 模型从手写 PaliGemma 栈迁移到 Hugging Face Qwen3-VL 的完整计划和实施记录。

### 设计目标

- 保留 state / action token-slot 注入语义
- 保留同批次 AR + flow 联合优化
- 保留 AdaLN-Zero 时间条件化
- 所有参数在一个 `nn.Module` 下，兼容 FSDP2 / DeepSpeed
- 子模块通过 Hydra `_target_` 实例化，配置驱动

### 简化约束（首版）

- 仅支持单图像
- 无多帧记忆
- 无深度输入
- 不兼容 pi0.5 检查点

---

## 架构设计

### 三空间模型

1. **残差空间**：backbone / expert 内部隐状态（hidden_size 可不同）
2. **注意力空间**：Q/K/V 投影定义，backbone 与 expert 共享（num_heads、num_kv_heads、head_dim）
3. **Token-slot 空间**：文本、图像、state、action 的序列布局

### 两流训练

- **Backbone 流**：原生多模态因果前向 → CE loss + AR/DiffLoss + 前缀 KV
- **Flow expert 流**：使用前缀 KV + 加噪动作 + AdaLN-Zero → 流速度预测

### 架构图

位于 `docs/architecture/` 目录下：
- `unified_vla_overview.png` — 端到端系统概览
- `unified_vla_model_blocks.png` — 内部模块布局
- `unified_vla_training_streams.png` — 两流训练流程
- `unified_vla_shared_kv.png` — 共享前缀注意力细节
- `unified_vla_file_map.png` — 文件级实现目标图

---

## 文件结构

### 核心实现文件

| 文件 | 用途 |
|------|------|
| `src/policy/legendvla.py` | 单根策略模块，接收 Hydra 实例化的子模块 |
| `src/policy/legendvla_loss.py` | CE + DiffLoss + Flow 联合损失 |
| `src/policy/legendvla_inference.py` | Flow / AR / VLM 推理函数 |
| `src/policy/legendvla_inference_wrapper.py` | 推理服务包装器 `LegendVLAInference` |
| `src/model/vlm/qwen3_vl_backbone.py` | Qwen3-VL HF 模型包装器 |
| `src/model/vlm/prefix_cache.py` | 前缀 KV 缓存规范化 |
| `src/model/action_expert/qwen_shared_kv_expert.py` | 共享 KV 专家解码器 |
| `src/model/action/action_head.py` | FourierActionEncoder、MLPProjector |
| `src/model/action/diffloss.py` | DiffLoss 实现 |
| `src/model/common/modules.py` | SinusoidalPosEmb、TimeEncoder、TimeEmbedding、AdaLNZero |
| `src/dataset/qwen3_vl_processing.py` | VLA / VLM 处理器 |
| `src/dataset/unified_vla_collator.py` | 统一批处理 collator |
| `src/workspace/train_legendvla_deepspeed_workspace.py` | 训练工作空间（唯一） |

### 配置文件

| 文件 | 用途 |
|------|------|
| `src/config/experiment/legendvla_qwen3_vl.yaml` | 训练主配置 |
| `src/config/experiment/inference.yaml` | 推理服务配置 |
| `src/config/train_config.yaml` | 入口配置（指向 legendvla_qwen3_vl） |
| `src/config/ds_config.json` | DeepSpeed ZeRO-2 配置 |

### 已删除的旧文件

- `src/model/vlm/paligemma/` — PaliGemma 视觉语言模型
- `src/model/vlm/nvila.py` — NVILA/LLaVA 模型
- `src/model/moe/` — MoE 混合专家
- `src/policy/unified_vla*.py` — 旧统一策略
- `src/policy/legendvla_utils.py` — 旧工具函数
- `src/dataset/paligemma_processing.py` — PaliGemma 处理器
- `src/workspace/train_unified_vla_workspace.py` — 已合并入基类

---

## LegendVLA 初始化接口

所有子模块通过 Hydra `_target_` 在 YAML 中声明，实例化后传入 `__init__`：

```python
class LegendVLA(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,           # Qwen3VLBackboneWrapper
        state_encoder: nn.Module,       # FourierActionEncoder — 状态 → backbone 嵌入空间
        ar_action_encoder: nn.Module,   # FourierActionEncoder — 动作 → backbone 嵌入空间 (AR stream)
        action_encoder: nn.Module,      # FourierActionEncoder — 动作 → expert 嵌入空间 (flow stream)
        time_embedding: nn.Module,      # TimeEmbedding
        flow_expert: nn.Module,         # ActionExpertDecoder
        action_decoder: nn.Module,      # MLPProjector — expert 隐状态 → 动作
        latent_condition_projector: nn.Module,  # MLPProjector — backbone 隐状态 → DiffLoss 条件
        shape_meta: dict,
        diffloss: nn.Module | None = None,
        action_hidden_size: int = 1024,
        # ... 标量配置参数
    ):
```

### YAML 共享维度变量

```yaml
# Qwen3-VL-4B 模型几何参数（切换 backbone 时需同步更新）
vlm_hidden_size: 2560
num_attention_heads: 32
num_key_value_heads: 8
head_dim: 80

# 模块维度
action_hidden_size: 1024
time_hidden_size: 1024
action_dim: 48
state_dim: 48
```

所有子模块配置通过 `${...}` 引用这些共享变量，保证一致性。

---

## 张量形状约定

### Backbone 流

| 张量 | 形状 | 说明 |
|------|------|------|
| `input_ids` | `[B, L]` | 文本 + 图像/state/action 占位符 |
| `attention_mask` | `[B, L]` | 原生因果掩码输入 |
| `pixel_values` | Qwen3-VL 格式 | 单图像 |
| `image_grid_thw` | Qwen3-VL 格式 | 处理器输出 |
| `mm_token_type_ids` | `[B, L]` | 多模态位置 ID |
| `position_ids` | `[3, B, L]` | Qwen3-VL M-RoPE 位置 ID |
| `hidden_states` | `[B, L, D_backbone]` | 主干输出 |

### 前缀缓存

| 张量 | 形状 | 说明 |
|------|------|------|
| `prefix_k` | `[B, H_kv, Lp_max, Dh]` | 每层前缀键 |
| `prefix_v` | `[B, H_kv, Lp_max, Dh]` | 每层前缀值 |
| `prefix_mask` | `[B, Lp_max]` | 布尔有效掩码 |

### Flow Expert 流

| 张量 | 形状 | 说明 |
|------|------|------|
| `flow_action_embeds` | `[B, A, D_expert]` | 流动作嵌入 |
| `time_cond` | `[B, D_time]` 或 `[B, A, D_time]` | AdaLN-Zero 时间条件 |
| `action_position_ids` | `[B, A]` | 动作位置 ID |
| `pred_v` | `[B, A, D_action]` | 流速度预测 |

---

## 同批次 AR + Flow 训练逻辑

1. 运行一次原生 backbone 因果前向
2. 从 backbone 输出计算 CE Loss 和 AR/DiffLoss
3. 从同一次前向结果切片前缀 KV
4. 运行独立的 flow expert 流
5. 计算 Flow Loss
6. 加权求和最终损失

前缀 KV 在因果解码下不受未来后缀影响，因此从完整因果运行提取的前缀缓存对 flow expert 有效。

---

## 推理服务

`LegendVLAInference`（`src/policy/legendvla_inference_wrapper.py`）实现 `RuntimeEngine` 所需接口：

| 方法 | 功能 |
|------|------|
| `prepare_process(obs)` | 原始观测 → processor 输入（图像选取、内参提取、状态归一化 + padding） |
| `build_model_inputs(prepared)` | 构建模型输入 tensor |
| `forward(inputs)` | flow / AR 推理，支持 RTC |
| `post_process(actions)` | normalizer 反归一化 |

配置文件：`src/config/experiment/inference.yaml`

---

## 迁移阶段与完成状态

| 阶段 | 内容 | 状态 |
|------|------|------|
| Stage 0 | 文档和脚手架 | ✅ 已完成 |
| Stage 1 | Processor 和 Collator | ✅ 已完成 |
| Stage 2 | Backbone 包装器 | ✅ 已完成 |
| Stage 3 | 前缀缓存工具 | ✅ 已完成 |
| Stage 4 | Flow Expert | ✅ 已完成 |
| Stage 5 | 统一策略和损失 | ✅ 已完成 |
| Stage 6 | 推理和工作空间 | ✅ 已完成 |
| Stage 7 | 清理和收尾 | ✅ 已完成 |
| 额外 | Hydra `_target_` 全面重构 | ✅ 已完成 |
| 额外 | 推理服务包装器 | ✅ 已完成 |
| 额外 | Workspace 合并 | ✅ 已完成 |
| 额外 | 单元测试（34 个） | ✅ 已完成 |

---

## 验证检查清单

### 单元测试

- ✅ Processor 格式验证（prompt 拼接、token 插入）
- ✅ UnifiedVLACollator（变长 padding、混合 VLA/VLM batch）
- ✅ PrefixKVCache（切片、掩码、dtype 转换）
- ✅ SharedPrefixAttention（flow/ar 掩码行为、prefix 可见性）
- ✅ 单批次 forward 产生 ce_loss + diffloss + flow_loss + total_loss
- ✅ Flow 推理运行
- ✅ AR 推理运行
- ✅ VLM 推理运行

### 训练级检查

- ✅ DeepSpeed 开关恢复（由环境变量控制）
- ✅ Evaluation 流程恢复（基类 evaluation → eval_utils）
- ✅ 优化器参数分组（backbone / expert / diffloss）
- ✅ 参数校验（trainable 参数必须在优化器中）

---

## 未来工作

### 多帧 Memory

与迁移前的方案一致，通过修改 vision tower 实现多帧输入：

- 当前限制：`single_image_only: True`，processor 和 backbone 仅处理单张图像
- 目标：支持 N 帧 RGB 历史输入，在 vision tower 内部做时序聚合
- 涉及改动：
  - `Qwen3VLBackboneWrapper.encode_image_features` — 扩展为多图像编码，处理多组 `pixel_values` 和 `image_grid_thw`
  - `Qwen3VLVLAProcessor` — prompt 结构需支持多个 `<|vision_start|>...<|vision_end|>` 块，或使用 Qwen3-VL 原生的视频输入接口
  - `VLAWdsDataset.sample_to_data` — 按 `image_horizon` 和 `image_stride` 传入多帧图像
  - `UnifiedVLACollator` — 多帧 `pixel_values` 的 batching 逻辑
  - YAML 配置 — `n_obs_image_steps` 从 1 改为 N，`single_image_only` 设为 False
- 注意：Qwen3-VL 原生支持多图像和视频输入，可以复用其 M-RoPE 时间维位置编码，不需要从零实现

### Depth 输入

在 backbone 中集成深度图编码：

- 当前状态：`train_depth: False`，depth 相关字段在 preprocess_batch 中已移除
- 目标：深度图作为额外模态输入，与 RGB 一同编码
- 涉及改动：
  - 新增 depth encoder 模块（可以用 DINOv2 或轻量 CNN），在 YAML 中通过 `_target_` 声明
  - `Qwen3VLBackboneWrapper` — 在 `build_inputs_embeds` 中融合 depth 特征到 inputs_embeds
  - `VLAWdsDataset.sample_to_data` — 恢复 `depth_values` 和 `has_depth_values` 字段
  - `preprocess_batch` — 恢复 depth_values 处理
  - `LegendVLA` — 新增 `depth_encoder` 子模块参数
- 设计决策：depth 特征是拼接到 image token 后面，还是通过 cross-attention 注入，需要实验确定

### FlexAttention 适配

用 PyTorch FlexAttention 替换手写 attention 实现，提升训练和推理效率：

- 当前状态：`SharedPrefixAttention` 手写了完整的 SDPA 流程（Q/K/V 投影、RoPE、mask 构建、softmax、输出投影）
- 目标：使用 `torch.nn.attention.flex_attention` 替换手写 attention kernel
- 涉及改动：
  - `SharedPrefixAttention.forward` — 将 prefix+suffix 拼接 + mask 构建逻辑改为 FlexAttention 的 `score_mod` / `block_mask` 接口
  - flow 模式的全可见 suffix mask 和 ar 模式的因果 suffix mask 用 `create_block_mask` 表达
  - prefix KV 的拼接逻辑需要适配 FlexAttention 的 document masking 语义
  - `Qwen3VLBackboneWrapper.forward_language_model` — 如果 backbone 的 attention 也要替换，需要考虑 HF 内部实现的兼容性
- 前置条件：PyTorch >= 2.5，且需要验证 FlexAttention 对 GQA（Grouped Query Attention）的支持
- 预期收益：消除手写 mask 的显存开销，利用 kernel fusion 加速

### torch.compile 加速

全模型或关键路径的 compile 优化：

- 当前状态：训练配置中 `compile.enabled: False`，推理包装器支持可选 `torch.compile`
- 目标：训练和推理均支持 compile 加速
- 涉及改动：
  - 训练路径：`TrainLegendVLAWorkspace.maybe_compile_model` 已存在，需要验证 `LegendVLA.forward` 在 `torch.compile(fullgraph=True)` 下的兼容性
  - 推理路径：`LegendVLAInference.maybe_compile_model` 已实现，需端到端验证
  - 关键障碍：
    - `slice_prefix_cache_from_full_kv` 中的动态 prefix length 切片可能导致 graph break
    - `build_attention_mask` 中的动态 mask 构建可能导致 graph break
    - HF Qwen3-VL 内部的 `get_image_features` 和 `compute_3d_position_ids` 是否 compile-safe 需要验证
  - 策略：先用 `torch._dynamo.explain()` 分析 graph break 点，逐步修复或标记 `torch.compiler.set_stance("force_eager")` 回退
- 预期收益：训练 20-40% 加速（取决于 graph break 数量），推理 30-50% 加速（flow inference 循环可完整 compile）

---

## 参考资料

- [Qwen3-VL HuggingFace 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)
- [Qwen3-VL 模型源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
- [Qwen3-VL-4B-Instruct 模型卡](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- transformers 最低版本要求：`>=4.57.0`
