# EgoVLA 代码重构计划

## 当前问题

### 1. 臃肿文件

| 文件 | 行数 | 问题 |
|------|------|------|
| `src/dataset/legendvla_dataset.py` | 2,327 | 数据集、几何变换、图像处理、collator、normalizer 全塞一起 |
| `src/policy/legendvla.py` | 1,765 | 模型结构、推理、loss、权重加载、mask 构建全塞一起 |
| `src/workspace/train_legendvla_deepspeed_workspace.py` | 939 | 训练循环 + 250 行评估逻辑 |
| `inference_pretrain_legendvla.py` | 845 | 推理 + 评估 + 指标 + 可视化 |

### 2. 职责混乱

- `legendvla.py` 是"上帝类"，包含两个类（LegendVLA + LegendVLAInference），混合了训练/推理/工具逻辑
- `legendvla_dataset.py` 包含 5 个类 + 大量辅助函数
- 推理逻辑分散在三处（policy、inference 脚本、workspace 的 evaluation）

### 3. EgoVLA vs LegendVLA 并行实现

两套独立实现（policy、dataset、workspace、processor），关系不清晰。如果 EgoVLA 已废弃，应移除。

---

## 重构方案

### Phase 1: 拆分 legendvla.py（优先级最高）

**当前结构**：
```
src/policy/legendvla.py (1,765 行)
├── LegendVLA(nn.Module)
│   ├── __init__                              # 模型结构
│   ├── load_pretrained_vlm_weights           # 权重加载
│   ├── load_pretrained_pi05_weights          # 权重加载
│   ├── freeze_non_lora_weights_in_vlm        # 冻结逻辑
│   ├── freeze_non_lora_weights_in_ae         # 冻结逻辑
│   ├── init_motion_token_embeddings          # 初始化
│   ├── _merge_input_ids_with_image_features  # Embedding 合并
│   ├── build_causal_mask_and_position_ids    # Mask 构建
│   ├── split_full_mask_into_submasks         # Mask 构建
│   ├── forward                               # 训练分发
│   ├── forward_train_ar                      # AR 训练
│   ├── forward_train_flow                    # Flow 训练
│   ├── compute_loss / compute_celoss         # Loss 计算
│   ├── compute_ar_loss / compute_flow_loss   # Loss 计算
│   ├── infer_action                          # 推理
│   ├── infer_vlm                             # 推理
│   ├── infer_single_step                     # 推理
│   └── 各种 @property (参数分组)              # 工具
└── LegendVLAInference(nn.Module)             # 推理封装类
```

**目标结构**：
```
src/policy/
├── legendvla.py              (~500 行)
│   └── LegendVLA(nn.Module)
│       ├── __init__
│       ├── _merge_input_ids_with_image_features
│       ├── forward              # 统一 forward，返回中间表示
│       ├── forward_train_ar
│       ├── forward_train_flow
│       └── 各种 @property
│
├── legendvla_loss.py         (~200 行)
│   ├── compute_celoss()
│   ├── compute_ar_loss()
│   └── compute_flow_loss()
│
├── legendvla_inference.py    (~200 行)
│   └── LegendVLAInference(nn.Module)
│       ├── prepare_inputs
│       ├── build_model_inputs
│       ├── infer_action
│       ├── infer_vlm
│       └── infer_single_step
│
└── legendvla_utils.py        (~300 行)
    ├── load_pretrained_vlm_weights()
    ├── load_pretrained_pi05_weights()
    ├── freeze_non_lora_weights_in_vlm()
    ├── freeze_non_lora_weights_in_ae()
    ├── init_motion_token_embeddings()
    ├── build_causal_mask_and_position_ids()
    └── split_full_mask_into_submasks()
```

**拆分原则**：
- Loss 计算改为独立函数，接收 model output 和 inputs，不依赖 self
- 权重加载、冻结逻辑改为接收 model 参数的独立函数
- Mask 构建改为纯函数
- LegendVLAInference 移到独立文件

---

### Phase 2: 拆分 legendvla_dataset.py

**当前结构**：
```
src/dataset/legendvla_dataset.py (2,327 行)
├── process_state_action()          # 几何变换
├── process_image()                 # 图像处理
├── LegendVLADataset                # VLA 数据集
├── LegendVLMDataset                # VLM 数据集
├── LegendUnifiedDataset            # 统一数据集
├── LegendVLALowLevelDataset        # Normalizer 用数据集
├── LegendVLDataCollator            # Collator
└── get_normalizer()                # Normalizer 计算
```

**目标结构**：
```
src/dataset/
├── legendvla_dataset.py      (~500 行)
│   ├── LegendVLADataset
│   ├── LegendUnifiedDataset
│   └── LegendVLALowLevelDataset
│
├── legendvlm_dataset.py      (~200 行)
│   └── LegendVLMDataset
│
├── collator.py               (~100 行)
│   └── LegendVLDataCollator
│
├── data_transforms.py        (~300 行)
│   ├── process_state_action()
│   ├── process_image()
│   └── 相关辅助函数
│
└── normalizer_utils.py       (~150 行)
    └── get_normalizer()
```

---

### Phase 3: 拆分训练和评估逻辑

**当前问题**：
- `train_legendvla_deepspeed_workspace.py` 的 `evaluation()` 方法有 250+ 行
- 评估逻辑与训练循环耦合

**目标**：
```
src/workspace/
├── train_legendvla_deepspeed_workspace.py  (~600 行)
│   └── TrainLegendVLAWorkspace
│       ├── __init__
│       ├── run (训练循环)
│       ├── preprocess_batch
│       └── get_grouped_parameters
│
└── eval_utils.py                           (~300 行)
    ├── evaluation()
    ├── save_topk_ckpt()
    └── save_interval_ckpt()
```

---

### Phase 4: 清理

1. **确认 EgoVLA 相关代码是否废弃**：
   - `src/policy/egovla.py`
   - `src/dataset/egovla_dataset.py`
   - `src/workspace/train_egovla_deepspeed_workspace.py`
   - `src/dataset/nvila_preprocessor.py`
   - `src/model/vlm/nvila.py`
   - 如果废弃，移除或移到 `archive/` 目录

2. **移除测试代码**：
   - `src/model/moe/joint_model.py` 末尾 ~150 行测试代码移到 `tests/`

3. **修复命名**：
   - `scripts/compuete_norm_stats.sh` → `scripts/compute_norm_stats.sh`

4. **清理 train.py 开头的注释掉的 debug 代码**

---

## 目标项目结构

```
EgoVLA/
├── train.py                           # 训练入口
├── inference_pretrain_legendvla.py     # 推理入口
├── src/
│   ├── config/                        # 配置（不变）
│   ├── dataset/
│   │   ├── base_dataset.py
│   │   ├── legendvla_dataset.py       # VLA 数据集（精简）
│   │   ├── legendvlm_dataset.py       # VLM 数据集（新拆出）
│   │   ├── collator.py               # Collator（新拆出）
│   │   ├── data_transforms.py        # 数据变换（新拆出）
│   │   ├── normalizer_utils.py       # Normalizer 工具（新拆出）
│   │   ├── paligemma_processing.py   # Processor（不变）
│   │   ├── sampler.py                # Sampler（不变）
│   │   └── streaming_replay_buffer.py
│   ├── model/                         # 模型组件（不变）
│   │   ├── action/
│   │   ├── common/
│   │   ├── moe/
│   │   └── vlm/
│   ├── policy/
│   │   ├── legendvla.py              # 模型结构（精简）
│   │   ├── legendvla_loss.py         # Loss 计算（新拆出）
│   │   ├── legendvla_inference.py    # 推理封装（新拆出）
│   │   └── legendvla_utils.py        # 工具函数（新拆出）
│   ├── workspace/
│   │   ├── base_workspace.py
│   │   ├── train_legendvla_deepspeed_workspace.py  # 训练（精简）
│   │   └── eval_utils.py             # 评估逻辑（新拆出）
│   ├── serving/                       # 服务化（不变）
│   └── utils/                         # 工具函数（不变）
├── data/                              # 数据处理脚本
├── scripts/                           # 启动脚本
└── tests/                             # 单元测试（新增）
```

---

## 执行顺序和注意事项

| 步骤 | 内容 | 注意 |
|------|------|------|
| 1 | 拆分 legendvla.py | 纯重构 commit，可 cherry-pick 到 diffloss-ar |
| 2 | 拆分 legendvla_dataset.py | 纯重构 commit，可 cherry-pick 到 diffloss-ar |
| 3 | 拆分训练/评估逻辑 | 纯重构 commit，可 cherry-pick 到 diffloss-ar |
| 4 | 清理废弃代码 | 仅在 LegendVLA 分支执行 |

**每个步骤**：
- 单独提交为纯重构 commit（不改功能）
- 提交后运行训练验证功能不变
- 确认无误后 cherry-pick 到 diffloss-ar 分支
