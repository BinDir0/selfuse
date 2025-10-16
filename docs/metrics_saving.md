# Metrics Saving Documentation

## 功能概述

`save_vq_tokenizer()` 函数现在会：
1. 保存所有 tokenizer 模型
2. 在**多个层级**保存 metrics.json 文件

## 文件结构

### 假设配置
```yaml
tokenizer:
  states:
    wrist: {...}
    hand: {...}
  actions:
    wrist: {...}
    hand: {...}
```

### 保存的文件结构

```
path/
├── metrics.json                          # 全局 metrics（所有 tokenizer 的汇总）
├── states/
│   ├── wrist/
│   │   └── loss=0.1234/
│   │       ├── config.json              # HuggingFace config
│   │       ├── pytorch_model.bin        # Model weights
│   │       └── metrics.json             # wrist tokenizer 的 metrics
│   └── hand/
│       └── loss=0.2345/
│           ├── config.json
│           ├── pytorch_model.bin
│           └── metrics.json             # hand tokenizer 的 metrics
└── actions/
    ├── wrist/
    │   └── loss=0.3456/
    │       ├── config.json
    │       ├── pytorch_model.bin
    │       └── metrics.json
    └── hand/
        └── loss=0.4567/
            ├── config.json
            ├── pytorch_model.bin
            └── metrics.json
```

## Metrics JSON 格式

### 1. 全局 metrics.json（根目录）

```json
{
  "states": {
    "wrist": {
      "loss": 0.1234,
      "recon_loss": 0.1100,
      "commit_loss": 0.0134
    },
    "hand": {
      "loss": 0.2345,
      "recon_loss": 0.2200,
      "commit_loss": 0.0145
    }
  },
  "actions": {
    "wrist": {
      "loss": 0.3456,
      "recon_loss": 0.3300,
      "commit_loss": 0.0156
    },
    "hand": {
      "loss": 0.4567,
      "recon_loss": 0.4400,
      "commit_loss": 0.0167
    }
  }
}
```

### 2. 部分 metrics.json（每个 tokenizer 目录下）

例如 `states/wrist/loss=0.1234/metrics.json`:

```json
{
  "loss": 0.1234,
  "recon_loss": 0.1100,
  "commit_loss": 0.0134
}
```

## 使用示例

### 训练时自动保存

```python
# 在 train() 方法中
if (epoch + 1) % self.cfg.training.ckpt_save_interval == 0:
    avg_loss_dict = self.validate()
    self.save_vq_tokenizer(
        path=os.path.join(self.output_dir, 'epoch_checkpoints', f'epoch={epoch}'), 
        metric_dict=avg_loss_dict
    )
```

### 加载 metrics

```python
import json

# 加载全局 metrics
with open('path/to/checkpoint/metrics.json', 'r') as f:
    all_metrics = json.load(f)
print(f"States wrist loss: {all_metrics['states']['wrist']['loss']}")

# 加载特定 tokenizer 的 metrics
with open('path/to/checkpoint/states/wrist/loss=0.1234/metrics.json', 'r') as f:
    wrist_metrics = json.load(f)
print(f"Wrist metrics: {wrist_metrics}")
```

## 自动类型转换

`_convert_to_json_serializable()` 会自动处理：

| Python 类型 | 转换为 |
|------------|--------|
| `numpy.ndarray` | `list` |
| `numpy.int32/int64` | `int` |
| `numpy.float32/float64` | `float` |
| `torch.Tensor` (标量) | `float` |
| `torch.Tensor` (多维) | `list` |
| `dict` | `dict` (递归转换) |
| `list` | `list` (递归转换) |

### 示例

```python
metric_dict = {
    'states': {
        'wrist': {
            'loss': np.float32(0.1234),          # → 0.1234 (float)
            'recon_loss': torch.tensor(0.1100),  # → 0.1100 (float)
            'commit_loss': 0.0134                # → 0.0134 (float)
        }
    }
}

# 自动转换并保存为 JSON
self.save_vq_tokenizer(path=save_path, metric_dict=metric_dict)
```

## 优点

1. **多层级保存**：
   - 根目录：完整的 metrics 汇总
   - 每个 tokenizer 目录：该 tokenizer 的 metrics
   
2. **便于对比**：
   - 可以快速查看不同 checkpoint 的性能
   - 文件夹名称包含 loss 值，一眼看出性能

3. **易于加载**：
   - JSON 格式，任何语言都能读取
   - 不需要 PyTorch/NumPy 也能查看

4. **版本控制友好**：
   - 纯文本格式
   - 可以 diff 对比

## 调试

如果遇到序列化错误，检查：

```python
# 打印类型信息
for key, value in metric_dict.items():
    print(f"{key}: {type(value)} = {value}")
```

`_convert_to_json_serializable()` 会递归处理所有嵌套结构，确保所有值都是 JSON 兼容的。

