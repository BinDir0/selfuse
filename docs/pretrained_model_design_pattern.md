# PreTrainedModel 设计模式详解

## 1. 为什么继承 PreTrainedModel？

`PartialMotionVQModel` 继承自 `transformers.PreTrainedModel`，这是 HuggingFace 的标准模型基类。

### 优势：
1. ✅ **标准化接口**：与整个 HuggingFace 生态系统兼容
2. ✅ **自动保存/加载**：获得 `save_pretrained()` 和 `from_pretrained()` 方法
3. ✅ **设备管理**：自动处理 CPU/GPU 转换
4. ✅ **梯度检查点**：支持大模型训练优化
5. ✅ **配置管理**：统一的配置系统

## 2. PreTrainedModel 的核心机制

```python
from transformers import PreTrainedModel, PretrainedConfig

class PartialMotionVQModel(PreTrainedModel):
    config_class = PretrainedConfig  # 关联配置类
    
    def __init__(self, shape_meta, args):
        # 步骤1: 创建配置对象
        config = PretrainedConfig()
        
        # 步骤2: 调用父类初始化
        super().__init__(config)
        
        # 步骤3: 初始化自己的模块
        self.use_part = args.use_part
        self.model = self._create_vq_model(args, ...)
        self.loss_fn = MotionReconstructionLoss(args.recons_loss)
```

### 关键点：
- `config`: 必须传入一个配置对象
- `super().__init__(config)`: 激活 PreTrainedModel 的功能
- 所有子模块自动注册到模型中

## 3. 保存机制

### 方法 1: 使用 transformers 的标准方法（推荐用于模型权重）

```python
# 保存
model = PartialMotionVQModel(shape_meta, args)
model.save_pretrained("/path/to/save")

# 这会创建：
# /path/to/save/
#   ├── config.json          # 模型配置
#   ├── pytorch_model.bin    # 模型权重
#   └── generation_config.json (可选)

# 加载
model = PartialMotionVQModel.from_pretrained("/path/to/save")
```

### 方法 2: 使用 ProcessorMixin（项目中使用的方法）

项目中通过 `VQActionProcessor` 包装模型，提供更灵活的保存：

```python
class VQActionProcessor(ProcessorMixin):
    def __init__(self, vq_model, vocab_size, ...):
        self.vq_model = vq_model  # PartialMotionVQModel
        self.vocab_size = vocab_size
        self.action_mean = action_mean
        self.action_std = action_std
    
    def save_pretrained(self, save_directory):
        # 保存 VQ 模型权重
        torch.save(
            self.vq_model.state_dict(),
            os.path.join(save_directory, "vq_model.pth")
        )
        
        # 保存配置（包括 normalizer 等）
        config = {
            "vocab_size": self.vocab_size,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
            "vq_model_config": {...}  # 模型架构信息
        }
        with open(os.path.join(save_directory, "config.pkl"), "wb") as f:
            pickle.dump(config, f)
```

### 方法 3: 直接保存 state_dict（最灵活）

```python
# 保存
torch.save(model.state_dict(), "model.pth")

# 加载
model = PartialMotionVQModel(shape_meta, args)
model.load_state_dict(torch.load("model.pth"))
```

## 4. 实际应用场景

### 场景 1: 训练时保存检查点

```python
class TrainMultipleGRVQTokenizerWorkspace(BaseWorkspace):
    def save_tokenizers(self):
        for name, tokenizer in self.tokenizers.items():
            save_dir = self.save_path / name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 使用 HuggingFace 的保存方法
            if hasattr(tokenizer, 'save_pretrained'):
                tokenizer.save_pretrained(str(save_dir))
            else:
                # 备用方案：直接保存 state_dict
                torch.save(tokenizer.state_dict(), save_dir / "model.pth")
```

### 场景 2: 加载预训练模型

```python
# 方式 A: 使用 transformers 标准方法
model = PartialMotionVQModel.from_pretrained("/path/to/checkpoint")

# 方式 B: 手动加载
model = PartialMotionVQModel(shape_meta, args)
state_dict = torch.load("/path/to/checkpoint/model.pth")
model.load_state_dict(state_dict)
```

### 场景 3: 保存到 HuggingFace Hub（可选）

```python
# 推送到 Hub
model.push_to_hub("your-username/model-name")

# 从 Hub 加载
model = PartialMotionVQModel.from_pretrained("your-username/model-name")
```

## 5. 完整的保存/加载示例

### 保存多个 tokenizer

```python
def save_tokenizers(self):
    """完整的保存逻辑"""
    for name, tokenizer in self.tokenizers.items():
        save_dir = self.save_path / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存模型权重（使用 PreTrainedModel 的方法）
        tokenizer.save_pretrained(str(save_dir))
        
        # 2. 保存训练状态（optimizer, scheduler 等）
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizers[name].state_dict(),
            'model_state_dict': tokenizer.state_dict(),
        }
        torch.save(checkpoint, save_dir / "training_state.pth")
        
        # 3. 保存配置（用于重建模型）
        config = {
            'shape_meta': self.cfg.shape_meta,
            'args': OmegaConf.to_container(self.cfg.tokenizers[name].args),
        }
        with open(save_dir / "config.yaml", 'w') as f:
            yaml.dump(config, f)
        
        print(f"Saved {name} to {save_dir}")
```

### 加载和恢复训练

```python
def load_tokenizers(self):
    """完整的加载逻辑"""
    for name in self.cfg.tokenizers.keys():
        load_dir = self.save_path / name
        
        # 1. 加载配置
        with open(load_dir / "config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # 2. 重建模型
        tokenizer = PartialMotionVQModel(
            shape_meta=config['shape_meta'],
            args=OmegaConf.create(config['args'])
        )
        
        # 3. 加载权重（方式 A：使用 PreTrainedModel）
        tokenizer = PartialMotionVQModel.from_pretrained(str(load_dir))
        
        # 或者（方式 B：手动加载）
        checkpoint = torch.load(load_dir / "training_state.pth")
        tokenizer.load_state_dict(checkpoint['model_state_dict'])
        
        # 4. 恢复优化器状态（如果继续训练）
        if 'optimizer_state_dict' in checkpoint:
            self.optimizers[name].load_state_dict(
                checkpoint['optimizer_state_dict']
            )
        
        self.tokenizers[name] = tokenizer.cuda()
        print(f"Loaded {name} from {load_dir}")
```

## 6. PreTrainedModel 提供的其他功能

```python
model = PartialMotionVQModel(shape_meta, args)

# 设备管理
model.to("cuda")
model.cuda()
model.cpu()

# 训练/评估模式
model.train()
model.eval()

# 参数冻结
model.requires_grad_(False)  # 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 梯度检查点（节省显存）
model.gradient_checkpointing_enable()

# 获取设备
device = model.device

# 参数统计
num_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

## 7. 注意事项

### ⚠️ 配置持久化问题

当前实现中，`PretrainedConfig()` 是空的：

```python
config = PretrainedConfig()  # 空配置
super().__init__(config)
```

**更好的做法**：创建自定义配置类

```python
from transformers import PretrainedConfig

class VQModelConfig(PretrainedConfig):
    model_type = "vq_model"
    
    def __init__(
        self,
        use_part="wrist",
        codebook_dim=256,
        nb_code=1024,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.use_part = use_part
        self.codebook_dim = codebook_dim
        self.nb_code = nb_code

class PartialMotionVQModel(PreTrainedModel):
    config_class = VQModelConfig  # 使用自定义配置
    
    def __init__(self, config: VQModelConfig):
        super().__init__(config)
        self.use_part = config.use_part
        # 从 config 读取所有参数
```

这样 `save_pretrained()` 会自动保存所有配置信息。

## 8. 项目中的最佳实践

根据当前项目结构，推荐：

```python
# 训练时
for epoch in range(num_epochs):
    # ... 训练 ...
    
    # 每个 epoch 保存
    for name, tokenizer in self.tokenizers.items():
        save_dir = f"{self.save_path}/{name}/epoch_{epoch}"
        tokenizer.save_pretrained(save_dir)

# 推理时
tokenizer = PartialMotionVQModel.from_pretrained(
    "/path/to/checkpoint/wrist/epoch_10"
)
```

## 总结

1. **PreTrainedModel** 提供了标准化的模型接口
2. **save_pretrained()** 和 **from_pretrained()** 是标准保存/加载方法
3. **项目中使用 ProcessorMixin** 包装，提供更灵活的保存（包括 normalizer）
4. **建议创建自定义 Config 类**，实现完整的配置持久化
5. **训练时保存 checkpoint**，包括 optimizer 状态以便恢复训练

