# PreTrainedModel 实现对比

## 当前实现 vs 改进实现

### 当前实现（存在的问题）

```python
class PartialMotionVQModel(PreTrainedModel):
    config_class = PretrainedConfig  # ❌ 使用空的通用配置
    
    def __init__(self, shape_meta, args):
        config = PretrainedConfig()  # ❌ 空配置，不保存任何参数
        super().__init__(config)
        
        # ❌ 参数来自 args（OmegaConf），不在 config 中
        self.use_part = args.use_part
        self.model = self._create_vq_model(args, ...)

# 保存时
model.save_pretrained("/path")
# ❌ 只保存了 pytorch_model.bin，config.json 是空的！
# ❌ 无法仅从保存的文件重建模型
```

**问题：**
- ❌ 配置没有持久化
- ❌ 需要额外保存 `args` 或 `shape_meta`
- ❌ `from_pretrained()` 无法直接工作
- ❌ 不能推送到 HuggingFace Hub

### 改进实现（推荐）

```python
class ImprovedPartialMotionVQModel(PreTrainedModel):
    config_class = PartialMotionVQModelConfig  # ✅ 自定义配置类
    
    def __init__(self, config: PartialMotionVQModelConfig):
        super().__init__(config)
        
        # ✅ 所有参数都从 config 读取
        self.use_part = config.use_part
        self.model = self._create_vq_model_from_config(config)

# 保存时
model.save_pretrained("/path")
# ✅ 保存了 pytorch_model.bin 和完整的 config.json
# ✅ 可以仅通过路径重建模型

# 加载时
model = ImprovedPartialMotionVQModel.from_pretrained("/path")
# ✅ 自动加载 config 和权重，无需额外参数
```

**优势：**
- ✅ 完整的配置持久化
- ✅ 标准化的保存/加载流程
- ✅ 可以推送到 HuggingFace Hub
- ✅ 版本控制和可复现性

## 详细对比表

| 特性 | 当前实现 | 改进实现 |
|------|---------|---------|
| **配置类** | `PretrainedConfig`（空） | `PartialMotionVQModelConfig`（完整） |
| **参数存储** | 在 `args` 中（外部） | 在 `config` 中（内部） |
| **保存内容** | 只有权重 | 权重 + 完整配置 |
| **加载方式** | `model = Model(shape_meta, args)`<br>`model.load_state_dict(...)` | `model = Model.from_pretrained(path)` |
| **可复现性** | ❌ 需要保存额外的 yaml | ✅ 所有信息在 config.json |
| **Hub 支持** | ❌ 不支持 | ✅ 完全支持 |
| **版本控制** | ❌ 困难 | ✅ 容易 |

## 文件结构对比

### 当前实现保存的文件

```
checkpoint/
├── pytorch_model.bin          # 模型权重
├── config.json                # ❌ 几乎为空
└── args.yaml                  # ❌ 需要额外保存
```

加载需要：
```python
# ❌ 复杂的加载流程
with open("args.yaml") as f:
    args = yaml.load(f)
model = PartialMotionVQModel(shape_meta, args)
model.load_state_dict(torch.load("pytorch_model.bin"))
```

### 改进实现保存的文件

```
checkpoint/
├── pytorch_model.bin          # 模型权重
└── config.json                # ✅ 包含所有配置
```

```json
{
  "model_type": "partial_motion_vq_model",
  "use_part": "wrist",
  "codebook_dim": 256,
  "nb_code": 1024,
  "width": 256,
  "depth": 3,
  ...
}
```

加载只需要：
```python
# ✅ 简单的加载流程
model = ImprovedPartialMotionVQModel.from_pretrained("checkpoint")
```

## 实际使用场景

### 场景 1: 训练和保存

**当前实现：**
```python
# 训练
model = PartialMotionVQModel(shape_meta, args)
# ... 训练 ...

# 保存（需要多个步骤）
model.save_pretrained("checkpoint")
# ❌ 还需要额外保存
with open("checkpoint/args.yaml", "w") as f:
    yaml.dump(OmegaConf.to_container(args), f)
with open("checkpoint/shape_meta.pkl", "wb") as f:
    pickle.dump(shape_meta, f)
```

**改进实现：**
```python
# 训练
config = PartialMotionVQModelConfig(use_part="wrist", ...)
model = ImprovedPartialMotionVQModel(config)
# ... 训练 ...

# 保存（一步完成）
model.save_pretrained("checkpoint")  # ✅ 全部搞定！
```

### 场景 2: 加载和推理

**当前实现：**
```python
# 加载（复杂）
with open("checkpoint/args.yaml") as f:
    args = yaml.load(f)
with open("checkpoint/shape_meta.pkl", "rb") as f:
    shape_meta = pickle.load(f)

model = PartialMotionVQModel(shape_meta, args)
model.load_state_dict(torch.load("checkpoint/pytorch_model.bin"))
model.eval()
```

**改进实现：**
```python
# 加载（简单）
model = ImprovedPartialMotionVQModel.from_pretrained("checkpoint")
model.eval()  # ✅ 就这么简单！
```

### 场景 3: 分享模型

**当前实现：**
```python
# ❌ 无法直接分享
# 需要打包：模型权重 + yaml 配置 + shape_meta + 代码
tar -czf model_package.tar.gz checkpoint/ config/ src/
# 其他人需要解压并理解结构
```

**改进实现：**
```python
# ✅ 推送到 Hub
model.push_to_hub("your-username/vq-wrist-tokenizer")

# ✅ 其他人直接使用
model = ImprovedPartialMotionVQModel.from_pretrained(
    "your-username/vq-wrist-tokenizer"
)
```

## 迁移指南

### 步骤 1: 创建配置类（已完成）

见 `src/model/action/vq_config.py`

### 步骤 2: 修改模型初始化

**从：**
```python
def __init__(self, shape_meta, args):
    config = PretrainedConfig()
    super().__init__(config)
    self.use_part = args.use_part
```

**到：**
```python
def __init__(self, config: PartialMotionVQModelConfig):
    super().__init__(config)
    self.use_part = config.use_part
```

### 步骤 3: 创建配置转换函数

```python
def create_config_from_hydra(hydra_cfg, shape_meta):
    """从 Hydra 配置创建 PreTrainedModel 配置"""
    return PartialMotionVQModelConfig(
        use_part=hydra_cfg.args.use_part,
        codebook_dim=hydra_cfg.args.codebook_dim,
        nb_code=hydra_cfg.args.nb_code,
        width=hydra_cfg.args.model.width,
        depth=hydra_cfg.args.model.depth,
        wrist_dim=shape_meta["obs"]["state"]["wrist"]["shape"][0],
        hand_dim=shape_meta["obs"]["state"]["hand"]["shape"][0],
        # ... 其他参数
    )
```

### 步骤 4: 修改训练代码

**从：**
```python
tokenizer = hydra.utils.instantiate(cfg.tokenizers.wrist)
```

**到：**
```python
# 创建配置
config = create_config_from_hydra(cfg.tokenizers.wrist, cfg.shape_meta)
# 创建模型
tokenizer = ImprovedPartialMotionVQModel(config)
```

### 步骤 5: 简化保存/加载

**从：**
```python
torch.save(tokenizer.state_dict(), "checkpoint/model.pth")
```

**到：**
```python
tokenizer.save_pretrained("checkpoint")  # 自动保存配置
```

## 推荐策略

### 渐进式迁移

1. **阶段 1**（当前）：使用现有实现，但改进保存逻辑
   ```python
   # 保存时多保存一些信息
   tokenizer.save_pretrained(save_dir)
   config_dict = {
       "use_part": args.use_part,
       "codebook_dim": args.codebook_dim,
       # ... 所有配置参数
   }
   torch.save(config_dict, f"{save_dir}/full_config.pth")
   ```

2. **阶段 2**（过渡）：新模型使用改进实现，旧模型兼容
   ```python
   # 提供兼容性函数
   def load_old_or_new_model(path):
       if os.path.exists(f"{path}/full_config.pth"):
           # 加载新格式
           return ImprovedPartialMotionVQModel.from_pretrained(path)
       else:
           # 加载旧格式
           return load_old_model(path)
   ```

3. **阶段 3**（最终）：全部迁移到新实现

## 总结

### 当前实现适合：
- 快速实验和原型开发
- 配置频繁变化的研究阶段
- 与 Hydra 紧密集成的场景

### 改进实现适合：
- 生产环境部署
- 模型分享和发布
- 需要版本控制的场景
- 与 HuggingFace 生态集成

**建议**：对于训练阶段，可以继续使用当前实现；对于最终发布的模型，迁移到改进实现以获得更好的可用性和兼容性。

