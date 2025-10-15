# Training Multiple GRVQ Tokenizers - Configuration Examples

## 基本概念

使用 YAML Anchors (`&`) 和 Merge Keys (`<<:`) 来实现配置复用：

- `&anchor_name`: 定义一个可复用的配置块
- `*anchor_name`: 引用整个配置块
- `<<: *anchor_name`: 合并配置块（可以被覆盖）

## 示例 1: 训练 wrist 和 hand tokenizer（不同 codebook 大小）

```yaml
base_tokenizer_args: &base_tokenizer_args
  codebook_dim: 256
  nb_code: 1024
  mu: 0.99
  # ... 其他参数

tokenizers:
  wrist:
    _target_: src.model.action.vq_tokenizer.PartialMotionVQModel
    shape_meta: ${shape_meta}
    args:
      <<: *base_tokenizer_args
      use_part: wrist
      nb_code: 1024  # 小的 codebook
  
  hand:
    _target_: src.model.action.vq_tokenizer.PartialMotionVQModel
    shape_meta: ${shape_meta}
    args:
      <<: *base_tokenizer_args
      use_part: hand
      nb_code: 2048  # 大的 codebook
```

## 示例 2: 训练 3 个 tokenizer（不同架构）

```yaml
base_tokenizer_args: &base_tokenizer_args
  codebook_dim: 256
  nb_code: 1024
  mu: 0.99
  model:
    down_t: 2
    stride_t: 2
    width: 256
    depth: 3
    # ...

tokenizers:
  wrist_small:
    _target_: src.model.action.vq_tokenizer.PartialMotionVQModel
    shape_meta: ${shape_meta}
    args:
      <<: *base_tokenizer_args
      use_part: wrist
      nb_code: 512
      model:
        width: 128  # 小模型
        depth: 2
  
  wrist_large:
    _target_: src.model.action.vq_tokenizer.PartialMotionVQModel
    shape_meta: ${shape_meta}
    args:
      <<: *base_tokenizer_args
      use_part: wrist
      nb_code: 2048
      model:
        width: 512  # 大模型
        depth: 4
  
  hand:
    _target_: src.model.action.vq_tokenizer.PartialMotionVQModel
    shape_meta: ${shape_meta}
    args:
      <<: *base_tokenizer_args
      use_part: hand
```

## 示例 3: 不同优化器配置

```yaml
base_optimizer: &base_optimizer
  lr: 2e-4
  betas: [0.9, 0.999]
  weight_decay: 1e-4

optimizers:
  wrist:
    <<: *base_optimizer
    lr: 1e-4  # 更小的学习率
  
  hand:
    <<: *base_optimizer
    lr: 3e-4  # 更大的学习率
    weight_decay: 1e-3  # 更大的 weight decay
```

## 示例 4: 同时训练 state 和 action tokenizer

```yaml
# 为 state 和 action 定义不同的基础配置
base_state_tokenizer: &base_state_tokenizer
  codebook_dim: 256
  nb_code: 2048
  mu: 0.99
  model:
    down_t: 2
    width: 256
    depth: 3
    # ...

base_action_tokenizer: &base_action_tokenizer
  codebook_dim: 256
  nb_code: 1024
  mu: 0.99
  model:
    down_t: 2
    width: 128  # action 用小一点的模型
    depth: 2
    # ...

tokenizers:
  state_wrist:
    _target_: src.model.action.vq_tokenizer.PartialMotionVQModel
    shape_meta: ${shape_meta}
    args:
      <<: *base_state_tokenizer
      use_part: wrist
  
  state_hand:
    _target_: src.model.action.vq_tokenizer.PartialMotionVQModel
    shape_meta: ${shape_meta}
    args:
      <<: *base_state_tokenizer
      use_part: hand
  
  action_full:
    _target_: src.model.action.vq_tokenizer.ActionVQModel  # 不同的模型类
    args:
      <<: *base_action_tokenizer
      action_dim: 48
```

## 示例 5: 使用嵌套的 anchor

```yaml
# 定义模型架构的 anchor
base_model_arch: &base_model_arch
  down_t: 2
  stride_t: 2
  width: 256
  depth: 3
  dilation_growth_rate: 3
  output_emb_width: 256
  activate: relu
  norm: null
  num_conv_layers: 3

# 定义 quantizer 的 anchor
base_quantizer: &base_quantizer
  quantizer_name: group_residualvq
  quantbeta: 1
  num_quantizers: 8
  num_groups: 1
  shared_codebook: True

# 定义 loss 的 anchor
base_loss: &base_loss
  recons_loss: l2
  commit_weight: 0.02

# 组合成完整的 tokenizer args
base_tokenizer_args: &base_tokenizer_args
  codebook_dim: 256
  nb_code: 1024
  mu: 0.99
  model:
    <<: *base_model_arch
  quantizer:
    <<: *base_quantizer
  loss:
    <<: *base_loss

# 现在可以在不同层级进行覆盖
tokenizers:
  wrist:
    _target_: src.model.action.vq_tokenizer.PartialMotionVQModel
    shape_meta: ${shape_meta}
    args:
      <<: *base_tokenizer_args
      use_part: wrist
      quantizer:
        <<: *base_quantizer
        num_quantizers: 4  # 只修改这一个参数
```

## 示例 6: 快速实验不同超参数

```yaml
# 为不同的实验定义不同的模板
exp_small: &exp_small
  codebook_dim: 128
  nb_code: 512
  model:
    width: 128
    depth: 2

exp_medium: &exp_medium
  codebook_dim: 256
  nb_code: 1024
  model:
    width: 256
    depth: 3

exp_large: &exp_large
  codebook_dim: 512
  nb_code: 2048
  model:
    width: 512
    depth: 4

# 快速组合实验
tokenizers:
  wrist_exp1:
    args:
      <<: *exp_small
      use_part: wrist
  
  wrist_exp2:
    args:
      <<: *exp_medium
      use_part: wrist
  
  wrist_exp3:
    args:
      <<: *exp_large
      use_part: wrist
```

## 运行命令

```bash
# 基本运行
python -m src.main experiment=train_multiple_grvq_tokenizer

# 覆盖特定参数
python -m src.main experiment=train_multiple_grvq_tokenizer \
    tokenizers.wrist.args.nb_code=2048 \
    optimizers.wrist.lr=1e-4

# 只训练特定的 tokenizer
python -m src.main experiment=train_multiple_grvq_tokenizer \
    ~tokenizers.hand  # 移除 hand tokenizer

# 添加新的 tokenizer
python -m src.main experiment=train_multiple_grvq_tokenizer \
    +tokenizers.full_state.args.use_part=null
```

## 性能优势

假设数据加载占 80%，模型训练占 20%：

| 方法 | 时间 | 加速比 |
|------|------|--------|
| 单独训练 2 个 tokenizer | 200% | 1.0x |
| 同时训练 2 个 tokenizer | 120% | 1.67x |
| 同时训练 3 个 tokenizer | 140% | 1.43x |
| 同时训练 4 个 tokenizer | 160% | 1.25x |

随着 tokenizer 数量增加，边际收益递减（因为训练时间占比增加）。
建议同时训练 2-3 个 tokenizer 以获得最佳性价比。

## 显存估算

- 单个 GRVQ tokenizer: ~500MB
- 同时训练 2 个: ~1.5GB (包括梯度和优化器状态)
- 同时训练 3 个: ~2.5GB
- Batch 数据 (bs=4096): ~10GB

**80G A100 可以轻松同时训练 5+ 个 tokenizer！**

