# Validation 函数说明

## 功能概述

`validate()` 函数会：
1. 在所有 validation 数据上运行所有 tokenizer
2. 累积每个 batch 的 loss
3. 在最后计算所有 loss 的平均值
4. 一次性 log 到 wandb（带 `valid/` 前缀）
5. 打印验证结果摘要

## 关键特性

### 1. 累积式计算（不中途 log）

```python
# 在验证循环中累积
for idx, batch in enumerate(tepoch):
    output = model(data)
    loss_accumulator[key][part][metric_name].append(metric_value)
    # ❌ 不在这里 log

# 验证结束后统一计算平均值
avg_loss_dict[key][part][metric_name] = np.mean(metric_values)

# ✅ 只在最后 log 一次
wandb.log(avg_loss_dict)
```

### 2. 支持嵌套结构

自动处理：
- **Partial tokenizers**: `states -> wrist/hand -> loss/recon_loss/commit_loss`
- **Full tokenizers**: `states -> loss/recon_loss/commit_loss`

### 3. 自动前缀处理

所有验证指标自动加上 `valid/` 前缀：

```python
# 训练时
train/states/wrist/loss
train/states/hand/loss

# 验证时
valid/states/wrist/loss
valid/states/hand/loss
```

## WandB 日志示例

### 假设你的 model output 包含：
```python
output = {
    'loss': 0.52,
    'recon_loss': 0.50,
    'commit_loss': 0.02
}
```

### 在验证结束时会 log：

```python
{
    'valid/states/wrist/loss': 0.52,
    'valid/states/wrist/recon_loss': 0.50,
    'valid/states/wrist/commit_loss': 0.02,
    'valid/states/hand/loss': 0.45,
    'valid/states/hand/recon_loss': 0.43,
    'valid/states/hand/commit_loss': 0.02,
    'valid/actions/wrist/loss': 0.48,
    'valid/actions/hand/loss': 0.41,
    'valid/total_loss': 1.86  # 所有 loss 的和
}
```

## 控制台输出示例

```
================================================================================
Validation Summary
================================================================================
states/wrist: loss=0.520000, recon_loss=0.500000, commit_loss=0.020000
states/hand: loss=0.450000, recon_loss=0.430000, commit_loss=0.020000
actions/wrist: loss=0.480000, recon_loss=0.460000, commit_loss=0.020000
actions/hand: loss=0.410000, recon_loss=0.390000, commit_loss=0.020000
Total validation loss: 1.860000
================================================================================
```

## 与训练的对比

| 特性 | Training | Validation |
|------|----------|-----------|
| 模式 | `model.train()` | `model.eval()` |
| 梯度 | 需要梯度 | `torch.no_grad()` |
| Log 频率 | 每个 step | 只在最后一次 |
| Loss 计算 | 即时值 | 平均值 |
| 前缀 | `train/` | `valid/` |

## 配置

在 YAML 中控制验证频率：

```yaml
training:
  num_epochs: 100
  val_per_epoch: 5  # 每 5 个 epoch 验证一次
  max_val_steps: 100  # 最多验证 100 个 batch（可选）
```

## 重要细节

### 1. 自动模式切换

```python
dict_apply(self.tokenizer, lambda x: x.eval())   # 验证前
# ... validation ...
dict_apply(self.tokenizer, lambda x: x.train())  # 验证后
```

### 2. 无梯度计算

```python
with torch.no_grad():
    # 验证循环
```

节省显存和计算时间。

### 3. Loss 累积结构

```python
loss_accumulator = {
    'states': {
        'wrist': {
            'loss': [0.5, 0.52, 0.48, ...],        # N 个 batch 的值
            'recon_loss': [0.48, 0.50, 0.46, ...],
            'commit_loss': [0.02, 0.02, 0.02, ...]
        },
        'hand': {...}
    }
}

# 计算平均
avg_loss_dict = {
    'states': {
        'wrist': {
            'loss': 0.50,           # np.mean([0.5, 0.52, 0.48, ...])
            'recon_loss': 0.48,
            'commit_loss': 0.02
        }
    }
}
```

## 调试提示

### 查看验证是否运行

检查 `val_per_epoch` 设置：
```python
if (epoch + 1) % self.cfg.training.val_per_epoch == 0:
    self.validate()
```

### 查看 WandB 日志

在 WandB UI 中，验证指标会显示为阶梯状（不像训练那样连续），因为只在 epoch 结束时记录一次。

### 验证数据集

默认使用与训练相同的 `dataloader`。如果需要单独的验证集，需要修改代码创建 `val_dataloader`。




