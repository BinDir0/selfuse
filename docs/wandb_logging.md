# WandB 日志记录使用说明

## 功能说明

训练 VQ Tokenizer 时，现在会自动记录所有 loss 到 WandB。

## 记录的指标

### 训练指标 (每个 step)

根据你的 `loss_dict` 结构，会记录：

#### 如果使用 partial tokenizers (wrist/hand):
```
train/states/wrist/loss
train/states/wrist/recon_loss
train/states/wrist/commit_loss
train/states/hand/loss
train/states/hand/recon_loss
train/states/hand/commit_loss
train/actions/wrist/loss
train/actions/hand/loss
train/total_loss          # 所有 loss 的总和
train/epoch               # 当前 epoch
train/lr                  # 当前学习率
```

#### 如果使用 full tokenizers:
```
train/states/loss
train/states/recon_loss
train/actions/loss
train/total_loss
train/epoch
train/lr
```

## 配置

在你的 YAML 配置文件中添加（可选）：

```yaml
# WandB 配置
use_wandb: true                    # 是否使用 wandb (默认: true)
wandb_project: "vq-tokenizer"      # WandB 项目名 (默认: "vq-tokenizer")
exp_name: "my_experiment"          # 实验名称 (默认: "vq_tokenizer_training")
```

## 如何查看

1. **训练时自动上传**
   ```bash
   python train.py
   ```

2. **查看结果**
   - 访问 https://wandb.ai/your-username/vq-tokenizer
   - 或者运行时命令行会显示 wandb URL

3. **关闭 wandb**（如果需要）
   ```yaml
   use_wandb: false
   ```

## 日志示例

假设你的 `loss_dict` 如下：
```python
{
    'states': {
        'wrist': {
            'loss': 0.52,
            'recon_loss': 0.50,
            'commit_loss': 0.02
        },
        'hand': {
            'loss': 0.45,
            'recon_loss': 0.43,
            'commit_loss': 0.02
        }
    },
    'actions': {
        'wrist': {'loss': 0.48, ...},
        'hand': {'loss': 0.41, ...}
    }
}
```

会自动展平并记录为：
```python
{
    'train/states/wrist/loss': 0.52,
    'train/states/wrist/recon_loss': 0.50,
    'train/states/wrist/commit_loss': 0.02,
    'train/states/hand/loss': 0.45,
    'train/states/hand/recon_loss': 0.43,
    'train/states/hand/commit_loss': 0.02,
    'train/actions/wrist/loss': 0.48,
    'train/actions/hand/loss': 0.41,
    'train/total_loss': 1.86,  # 所有 loss 的和
    'train/epoch': 0,
    'train/lr': 0.0002
}
```

## 实现细节

- **自动展平**: `_flatten_loss_dict()` 会自动处理嵌套的 loss 字典
- **支持 Tensor**: 自动将 `torch.Tensor` 转换为 Python scalar
- **步数跟踪**: 使用 `global_step` 作为 x 轴
- **学习率**: 从 `optimizer.param_groups[0]['lr']` 获取

## 常见问题

**Q: 如何在本地不上传到 wandb？**

A: 设置环境变量：
```bash
export WANDB_MODE=offline
```

**Q: 如何更改项目名？**

A: 在 config.yaml 中添加：
```yaml
wandb_project: "my-custom-project-name"
```

**Q: 训练中断后如何恢复？**

A: wandb 会自动处理，使用 `resume='allow'` 参数

