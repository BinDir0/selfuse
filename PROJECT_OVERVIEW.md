# LegendVLA 项目详细实现介绍

> **注意**: 本文档基于项目文件结构分析生成，部分代码示例和实现细节可能与实际代码存在差异。建议结合实际源码进行参考。（AI 生成）

## 项目概述

LegendVLA 是一个基于视觉-语言-动作（Vision-Language-Action）的机器人学习框架，专注于从第一人称视角学习复杂的双手操作任务。项目结合了视觉语言模型（VLM）和视觉语言动作模型（VLA），支持多数据集训练和分布式训练。

### 核心特性
- **多模态学习**: 融合视觉、语言和动作信息
- **双手操作**: 支持复杂的双手协调任务
- **相对动作表示**: 使用相对变换表示手部动作
- **分布式训练**: 支持DeepSpeed多机多卡训练
- **流式数据处理**: 支持大规模数据集的流式加载

## 项目文件树结构

```
LegendVLA/
├── README.md                          # 项目说明文档
├── requirements.txt                   # Python依赖包列表
├── train.py                          # 主训练入口脚本
├── inference.py                      # 推理入口脚本
├── visualize.py                      # 可视化工具脚本
├── ds_to_universal.py                # DeepSpeed checkpoint转换工具
├── progress.log                      # 训练进度日志
├── scripts/                          # 训练和部署脚本
│   ├── install.sh                    # 环境安装脚本
│   ├── pretrain_legendvla_deepspeed.sh  # LegendVLA分布式训练脚本
│   ├── pretrain_legendvla.sh         # LegendVLA单机训练脚本
│   ├── train_fast_tokenizer.sh        # 快速分词器训练脚本
│   ├── ds_to_universal.sh            # checkpoint转换脚本
│   └── nccl_test.sh                  # NCCL通信测试脚本
├── data/                            # 数据处理模块
│   ├── egodex/                      # EgoDex数据集处理
│   │   ├── build_zarr.py            # 构建zarr格式数据
│   │   └── mano_trans.py            # MANO参数转换
│   ├── HOI4D/                       # HOI4D数据集处理
│   │   ├── build_zarr.py
│   │   └── mano_trans.py
│   ├── taco/                        # TACO数据集处理
│   │   ├── build_zarr.py
│   │   └── mano_trans.py
│   ├── vlm/                         # VLM数据集处理
│   │   ├── filter.py                # 数据过滤
│   │   ├── test.py                  # 测试脚本
│   │   └── FineVision_elected.txt   # 精选数据列表
│   └── visualizer.py                # 数据可视化工具
├── src/                             # 核心源代码
│   ├── __init__.py
│   ├── config/                      # 配置文件模块
│   │   ├── __init__.py
│   │   ├── acc_config.yaml          # Accelerate配置
│   │   ├── ds_config.json           # DeepSpeed配置
│   │   ├── train_config.yaml        # 训练配置
│   │   ├── hostfile                 # 多机训练主机配置
│   │   └── experiment/              # 实验配置
│   │       ├── inference.yaml       # 推理配置
│   │       ├── legendvla_stage1.yaml # LegendVLA第一阶段配置
│   │       ├── pretrain_deepspeed.yaml # 分布式预训练配置
│   │       ├── pretrain_legendvla_deepspeed.yaml # LegendVLA分布式配置
│   │       ├── pretrain_legendvla.yaml # LegendVLA单机配置
│   │       ├── pretrain.yaml        # 基础预训练配置
│   │       ├── retargeting.yaml     # 重定向配置
│   │       └── train_fast_tokenizer.yaml # 快速分词器配置
│   ├── dataset/                     # 数据集处理模块
│   │   ├── __init__.py
│   │   ├── base_dataset.py          # 基础数据集类
│   │   ├── base_vl_preprocessor.py   # 视觉语言预处理器基类
│   │   ├── egovla_dataset.py        # EgoVLA数据集实现
│   │   ├── legendvla_dataset.py     # LegendVLA数据集实现
│   │   ├── nvila_preprocessor.py    # NVILA预处理器
│   │   └── paligemma_processing.py   # PaliGemma数据处理
│   ├── model/                       # 模型定义模块
│   │   ├── __init__.py
│   │   ├── action/                  # 动作相关模型
│   │   │   ├── action_head.py       # 动作预测头
│   │   │   ├── base_action_head.py  # 动作头基类
│   │   │   └── fast_tokenizer.py     # 快速动作分词器
│   │   ├── common/                  # 通用模型组件
│   │   │   ├── dict_of_tensor_mixin.py # 张量字典混入类
│   │   │   ├── kv_cache.py          # KV缓存实现
│   │   │   ├── lora.py              # LoRA适配器
│   │   │   ├── lr_scheduler.py      # 学习率调度器
│   │   │   ├── model_average.py     # 模型平均
│   │   │   ├── module_attr_mixin.py # 模块属性混入类
│   │   │   ├── modules.py           # 通用模块
│   │   │   ├── normalizer.py        # 数据标准化
│   │   │   └── utils.py             # 模型工具函数
│   │   ├── moe/                     # 混合专家模型
│   │   │   ├── joint_model.py       # 联合模型
│   │   │   └── mixture.py           # 混合模型
│   │   ├── retargeting/             # 重定向模块
│   │   │   ├── base_retargeting.py  # 重定向基类
│   │   │   ├── mlp_retargeting.py   # MLP重定向器
│   │   │   └── opt_retargeting.py   # 优化重定向器
│   │   └── vlm/                     # 视觉语言模型
│   │       ├── nvila.py             # NVILA模型实现
│   │       └── paligemma/            # PaliGemma模型
│   │           ├── __init__.py
│   │           ├── config.py        # 配置
│   │           ├── gemma.py         # Gemma语言模型
│   │           ├── load.py          # 模型加载
│   │           ├── processing.py    # 数据处理
│   │           └── siglip.py        # SigLIP视觉编码器
│   ├── policy/                      # 策略实现模块
│   │   ├── __init__.py
│   │   ├── base_policy.py           # 策略基类
│   │   ├── egovla.py                # EgoVLA策略实现
│   │   └── legendvla.py             # LegendVLA策略实现
│   ├── utils/                       # 工具函数模块
│   │   ├── __init__.py
│   │   ├── checkpoint_util.py         # checkpoint管理工具
│   │   ├── decorator.py             # 装饰器
│   │   ├── geometry.py              # 几何变换工具
│   │   ├── json_logger.py           # JSON日志记录器
│   │   ├── metric.py                # 评估指标
│   │   ├── monitor.py               # 监控工具
│   │   ├── optim.py                 # 优化器工具
│   │   ├── pytorch_util.py          # PyTorch工具函数
│   │   ├── replay_buffer.py         # 重放缓冲区
│   │   ├── sampler.py              # 序列采样器
│   │   ├── spec.py                  # 规格定义
│   │   └── streaming_replay_buffer.py # 流式重放缓冲区
│   └── workspace/                   # 训练工作空间模块
│       ├── __init__.py
│       ├── base_workspace.py        # 工作空间基类
│       ├── train_egovla_deepspeed_workspace.py # EgoVLA分布式训练
│       ├── train_egovla_workspace.py # EgoVLA单机训练
│       ├── train_fast_tokenizer.py  # 快速分词器训练
│       ├── train_legendvla_deepspeed_workspace.py # LegendVLA分布式训练
│       └── train_legendvla_workspace.py # LegendVLA单机训练
├── outputs/                         # 训练输出和日志
│   ├── 2025.09.11/                  # 按日期组织的训练输出
│   ├── 2025.09.15/
|   ├── ...
├── inference_utils/                 # 推理工具
│   └── robot_libs/                  # 机器人库
│       └── rh2-wrapper/             # RH2机器人包装器
├── reference/                       # 参考实现
│   ├── dex_ruiyan_vector.py        # DEX Ruiyan向量实现
│   ├── realman_tcp.py              # RealMan TCP通信
│   ├── realman.py                  # RealMan机器人接口
│   └── robot_inspire 1.py          # Inspire机器人接口
├── assets/                         # 资源文件
│   ├── maniskill_pp.png            # ManiSkill++图片
│   └── NVTX.png                    # NVTX标记图片
├── wandb/                          # Weights & Biases日志
└── _test/                          # 测试代码
    ├── Block-Sparse-Attention/     # 块稀疏注意力测试
    ├── test_block_sparse_attn.py   # 块稀疏注意力测试
    └── test.py                     # 通用测试
```

## 核心模块详细说明

### 1. 配置模块 (`src/config/`)

**功能**: 管理所有训练和实验配置，支持Hydra配置管理

#### 1.1 主要配置文件

**`experiment/pretrain_legendvla_deepspeed.yaml`** - LegendVLA分布式训练配置
```yaml
# 核心配置内容
name: legendvla
exp_name: "pretrain_deepspeed"
_target_: src.workspace.train_legendvla_deepspeed_workspace.TrainLegendVLAWorkspace

# 数据集配置
vla_dataset_paths:
  - /share_data/datasets/taco/taco_45pca_newpresence.zarr
  - /share_data/datasets/OakInk-v2/oakink2.zarr
  - /share_data/datasets/hoi4d_segmented/hoi4d_train.zarr
  - /share_data/datasets/EgoDex/egodex_train.zarr

vlm_dataset_paths:
  - "/share_data/datasets/VLM/FineVision/objects365_qa_filtered"
  - "/share_data/datasets/VLM/FineVision/spatialsense_filtered"
  # ... 更多VLM数据集

# 模型配置
shape_meta:
  obs:
    rgb:
      shape: [224, 224, 3]
      type: rgb
      horizon: 1
    state:
      wrist:
        shape: [18]
      hand: 
        shape: [30]
      shape: [48]
      type: low_dim
      horizon: 1

# 训练配置
training:
  num_epochs: 1000
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 0.01
  warmup_steps: 1000
  max_grad_norm: 1.0
```

**`experiment/train_fast_tokenizer.yaml`** - 快速分词器训练配置
```yaml
# 分词器训练专用配置
name: fast_tokenizer
exp_name: "train_fast_tokenizer"
_target_: src.workspace.train_fast_tokenizer.TrainFastTokenizerWorkspace

# 分词器参数
tokenizer:
  vocab_size: 8192
  max_seq_len: 2048
  num_epochs: 10
  batch_size: 64
```

#### 1.2 配置特点
- **模块化配置**: 使用Hydra支持配置组合和覆盖
- **多环境支持**: 支持单机和分布式训练配置
- **数据集管理**: 统一管理VLA和VLM数据集路径
- **超参数调优**: 支持不同实验的超参数配置

### 2. 数据集模块 (`src/dataset/`)

**功能**: 处理多种数据格式，支持VLA和VLM数据的统一处理

#### 2.1 核心数据集类

**`legendvla_dataset.py`** - LegendVLA数据集实现

```python
class LegendVLADataset(BaseImageDataset):
    """处理VLA数据（手部动作、状态、指令）"""
    
    def __init__(self, zarr_paths, horizon=1, pad_before=0, pad_after=0, 
                 shape_meta=None, seed=42, val_ratio=0.0, history=30, 
                 objective=None, normalizer_dataloader_cfg=dict(), 
                 max_train_episodes=None, train_mode=True, 
                 token_len_buckets=None):
        # 初始化replay buffers, samplers, 数据增强等
        
    def _sample_to_data(self, sample):
        """将原始样本转换为训练数据"""
        # 处理状态和动作
        state, action, action_valid_mask = process_state_action(...)
        # 处理图像
        image = process_image(sample['image'], ...)
        # 处理指令
        instruction = sample['instruction'][self.history]
        # 通过预处理器生成最终数据
        processed_results = self.preprocessor(...)
        
    def get_validation_dataset(self):
        """获取验证数据集"""
        
    def set_preprocessor(self, preprocessor):
        """设置预处理器"""
        
    def set_normalizer(self, normalizer):
        """设置标准化器"""
```

**`LegendVLMDataset`** - VLM数据集处理
```python
class LegendVLMDataset(BaseImageDataset):
    """处理VLM数据（图像-文本对）"""
    
    def _sample_to_data(self, sample, idx):
        """处理VLM样本"""
        images = sample['images']  # PIL图像列表
        text = sample['texts']     # 文本数据
        # 评分和选择最佳文本
        scores = formatting_ratings * weights[0] + \
                visual_dependency_ratings * weights[1] + \
                relevance_ratings * weights[2]
        text = text[np.argmax(scores)]
        # 数据增强
        augmented_images = self._augment_images(images)
        # 预处理
        processed_results = self.preprocessor(...)
```

**`LegendUnifiedDataset`** - 统一数据集
```python
class LegendUnifiedDataset(BaseImageDataset):
    """统一VLA和VLM数据集"""
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """根据索引返回对应数据集的数据"""
        if idx < len(self.vla_dataset):
            return self.vla_dataset[idx]
        elif self.vlm_dataset is not None:
            sample = self.vlm_dataset[idx - len(self.vla_dataset)]
            # 填充缺失的键值对
            for key in self.shape_meta.keys():
                if key not in sample and "valid_mask" not in key:
                    sample[key] = torch.zeros(self.shape_meta[key])
                    sample[f"{key}_valid_mask"] = torch.zeros(...)
            return sample
```

#### 2.2 数据处理流程

**数据加载流程**:
1. **Zarr格式加载**: 从zarr文件加载图像、状态、动作、指令等数据
2. **序列采样**: 使用SequenceSampler进行时间序列采样
3. **数据增强**: 应用颜色抖动、高斯模糊等增强
4. **坐标变换**: 将手部姿态转换到相机坐标系
5. **相对动作计算**: 计算相对于当前状态的动作
6. **标准化**: 对状态和动作进行标准化处理

**数据格式**:
```
zarr数据结构:
├── data/
│   ├── action/          # 动作数据
│   │   ├── hand        # 手部参数 (sum_frames, 30) float32
│   │   └── wrist       # 腕部参数 (sum_frames, 18) float32
│   ├── state/          # 状态数据  
│   │   ├── hand        # 手部状态 (sum_frames, 30) float32
│   │   └── wrist       # 腕部状态 (sum_frames, 18) float32
│   ├── image/          # 图像数据 (sum_frames, 384, 384, 3) uint8
│   ├── instruction/    # 指令文本 (sum_frames,) string
│   ├── instruction_num/ # 指令数量 (sum_frames,) int32
│   └── extrinsic/      # 相机外参 (sum_frames, 16) float32
└── meta/
    ├── episode_ends    # 回合结束索引 (num_episodes,) int64
    └── presence        # 手部可见性 (num_episodes,) int8
```

#### 2.3 关键处理函数

**`process_state_action`** - 状态动作处理
```python
def process_state_action(wrist_state, hand_state, wrist_action, hand_action, 
                        extrinsic, presence, hand_ndim, history, 
                        n_obs_state_steps, normalizer=None):
    """处理状态和动作数据"""
    # 1. 时间切片
    if n_obs_state_steps > 1:
        state_slice = [i for i in range(0, history + 1, history // (n_obs_state_steps - 1))]
    else:
        state_slice = [history]
    
    # 2. 手部参数处理
    all_hand_ndim = hand_state.shape[-1] // 2
    hand_state = np.concatenate([
        hand_state[state_slice, :hand_ndim], 
        hand_state[state_slice, all_hand_ndim:all_hand_ndim + hand_ndim]
    ], axis=-1)
    
    # 3. 坐标系变换
    processed_wrist_state = transform_wrist_to_target_frame(wrist_state[state_slice], extrinsic[history])
    processed_wrist_action = transform_wrist_to_target_frame(wrist_action[history:], extrinsic[history])
    
    # 4. 可见性处理
    processed_state, processed_action, action_valid_mask = get_presence_value(...)
    
    # 5. 相对动作计算
    processed_action = get_relative_action(processed_state[-1], processed_action)
    
    # 6. 标准化
    if normalizer is not None:
        state = normalizer['states'](processed_state)
        action = normalizer['actions'](processed_action)
    
    return state, action, action_valid_mask
```

**`get_relative_action`** - 相对动作计算
```python
def get_relative_action(state, action):
    """计算相对于当前状态的动作"""
    for idx in range(2):  # 左右手
        # 构建齐次变换矩阵
        wrist_action_homo_mat = homo_matrix_from_trans_6drot(
            action[..., idx*3 : idx*3+3], 
            action[..., 6+idx*6 : 6+idx*6+6]
        )
        wrist_state_homo_mat = homo_matrix_from_trans_6drot(
            state[idx*3 : idx*3+3], 
            state[6+idx*6 : 6+idx*6+6]
        )
        
        # 计算相对变换: T_relative = T_state^(-1) * T_action
        wrist_action_homo_mat = np.linalg.pinv(wrist_state_homo_mat) @ wrist_action_homo_mat
        trans, rot_6d = homo_matrix_to_trans_6drot(wrist_action_homo_mat)
        
        # 更新动作
        action[..., idx*3 : idx*3+3] = trans
        action[..., 6+idx*6 : 6+idx*6+6] = rot_6d
    
    # 手部参数的相对计算
    action[..., 18:] = action[..., 18:] - state[18:]
    return action
```

### 3. 模型模块 (`src/model/`)

**功能**: 定义各种神经网络模型组件，支持模块化设计

#### 3.1 动作模块 (`action/`)

**`action_head.py`** - 动作预测头
```python
class ActionHead(nn.Module):
    """动作预测头，将隐藏状态映射到动作空间"""
    
    def __init__(self, hidden_dim, action_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, hidden_states):
        x = hidden_states
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)
```

**`fast_tokenizer.py`** - 快速动作分词器
```python
class UniversalActionProcessor:
    """通用动作处理器，将连续动作离散化为token"""
    
    def __init__(self, vocab_size=8192, max_seq_len=2048):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tokenizer = None
        
    def train_tokenizer(self, actions, num_epochs=10):
        """训练动作分词器"""
        # 使用DCT变换和BPE分词器学习动作词汇表
        
    def encode(self, actions):
        """将连续动作编码为token序列"""
        
    def decode(self, tokens):
        """将token序列解码为连续动作"""
```

#### 3.2 视觉语言模型 (`vlm/`)

**`nvila.py`** - NVILA模型实现
```python
class NVILAModel(nn.Module):
    """NVILA视觉语言模型"""
    
    def __init__(self, config):
        super().__init__()
        self.vision_encoder = SigLIPVisionEncoder(config.vision_config)
        self.language_model = GemmaLanguageModel(config.language_config)
        self.fusion_layer = CrossModalFusion(config.fusion_config)
        
    def forward(self, images, text):
        # 视觉编码
        vision_features = self.vision_encoder(images)
        # 语言编码
        text_features = self.language_model(text)
        # 多模态融合
        fused_features = self.fusion_layer(vision_features, text_features)
        return fused_features
```

**`paligemma/`** - PaliGemma模型组件
```python
# gemma.py - Gemma语言模型
class GemmaModel(nn.Module):
    """Gemma语言模型实现"""
    
# siglip.py - SigLIP视觉编码器  
class SigLIPVisionEncoder(nn.Module):
    """SigLIP视觉编码器"""
    
# processing.py - 数据处理
class PaliGemmaProcessor:
    """PaliGemma数据处理器"""
```

#### 3.3 通用模块 (`common/`)

**`normalizer.py`** - 数据标准化
```python
class LinearNormalizer:
    """线性标准化器，支持流式统计计算"""
    
    def __init__(self):
        self.params_dict = {}
        
    def start_streaming_fit(self, keys):
        """开始流式拟合"""
        for key in keys:
            self.params_dict[key] = StreamingStats()
        
    def update_streaming_fit(self, data):
        """更新拟合统计量"""
        for key, value in data.items():
            self.params_dict[key].update(value)
        
    def finish_streaming_fit(self):
        """完成拟合，计算标准化参数"""
        for key, stats in self.params_dict.items():
            stats.finish()
        
    def __call__(self, data):
        """标准化数据"""
        return (data - self.offset) / self.scale
```

**`lora.py`** - LoRA适配器
```python
class LoRALinear(nn.Module):
    """LoRA线性层"""
    
    def __init__(self, in_features, out_features, rank=32, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x):
        return F.linear(x, self.lora_B @ self.lora_A) * self.alpha
```

#### 3.4 重定向模块 (`retargeting/`)

**`mlp_retargeting.py`** - MLP重定向器
```python
class MLPRetargeting(nn.Module):
    """MLP重定向器，将通用动作映射到特定机器人"""
    
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, actions):
        return self.network(actions)
```

### 4. 策略模块 (`src/policy/`)

**功能**: 实现不同的学习策略，支持多种训练目标

#### 4.1 LegendVLA策略

**`legendvla.py`** - LegendVLA策略实现
```python
class LegendVLA(nn.Module):
    """LegendVLA策略，结合VLA和VLM能力"""
    
    def __init__(self, config):
        super().__init__()
        # 视觉编码器
        self.vision_encoder = SigLIPVisionEncoder(config.vision_config)
        # 语言模型
        self.language_model = GemmaLanguageModel(config.language_config)
        # 动作预测头
        self.action_head = ActionHead(config.action_config)
        # 重定向器
        self.retargeting = MLPRetargeting(config.retargeting_config)
        
    def forward(self, images, text, states, actions=None):
        # 多模态编码
        vision_features = self.vision_encoder(images)
        text_features = self.language_model(text)
        state_features = self.state_encoder(states)
        
        # 特征融合
        fused_features = self.fusion_layer(
            vision_features, text_features, state_features
        )
        
        # 动作预测
        if self.training:
            actions = self.action_head(fused_features)
            return actions
        else:
            # 推理时生成动作序列
            return self.generate_actions(fused_features)
            
    def generate_actions(self, features, max_length=30):
        """生成动作序列"""
        actions = []
        for _ in range(max_length):
            action = self.action_head(features)
            actions.append(action)
            # 更新特征（如果需要）
            features = self.update_features(features, action)
        return torch.stack(actions, dim=1)
```

#### 4.2 训练目标

**自回归训练 (AR)**:
```python
def compute_ar_loss(self, predictions, targets, attention_mask):
    """计算自回归损失"""
    shift_logits = predictions[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    return loss
```

**Flow匹配训练**:
```python
def compute_flow_loss(self, predictions, targets, timesteps):
    """计算Flow匹配损失"""
    # Flow匹配损失计算
    noise = torch.randn_like(targets)
    noisy_targets = (1 - timesteps) * targets + timesteps * noise
    
    predicted_noise = self.flow_model(noisy_targets, timesteps)
    loss = F.mse_loss(predicted_noise, noise)
    return loss
```

### 5. 工具模块 (`src/utils/`)

**功能**: 提供各种实用工具，支持数据处理、训练和评估

#### 5.1 几何变换工具

**`geometry.py`** - 几何变换工具
```python
def rot_matrix_from_6drot(rot):
    """将6D旋转表示转换为旋转矩阵"""
    a = rot[..., :3]  # 第一个向量
    b = rot[..., 3:]  # 第二个向量
    
    # Schmidt正交化
    a = F.normalize(a, dim=-1)
    b = b - torch.sum(a * b, dim=-1, keepdim=True) * a
    b = F.normalize(b, dim=-1)
    c = torch.cross(a, b, dim=-1)
    
    # 构建旋转矩阵
    rot_matrix = torch.stack([a, b, c], dim=-1)
    return rot_matrix

def transform_wrist_to_target_frame(wrist_action, target_extrinsic):
    """将腕部动作转换到目标坐标系"""
    # 构建齐次变换矩阵
    wrist_pose = torch.zeros(wrist_rot_6d.shape[:-1] + (4, 4))
    wrist_pose[..., :3, 3] = wrist_action[..., :6].reshape(-1, 2, 3).reshape(-1, 6)
    wrist_pose[..., :3, :3] = rot_matrix_from_6drot(wrist_rot_6d)
    wrist_pose[..., 3, 3] = 1
    
    # 坐标系变换
    wrist_pose = transform_to_target_frame(wrist_pose, target_extrinsic)
    
    # 转换回6D表示
    wrist_rot_6d = rot_matrix_to_6drot(wrist_pose[..., :3, :3])
    wrist_action[..., :6] = wrist_pose[..., :3, 3].reshape(-1, 6)
    return wrist_action
```

#### 5.2 数据处理工具

**`streaming_replay_buffer.py`** - 流式重放缓冲区
```python
class StreamingReplayBuffer(ReplayBuffer):
    """流式重放缓冲区，支持大规模数据集的流式加载"""
    
    @classmethod
    def copy_from_path(cls, path, keys=None, lazy_load=True):
        """从zarr文件创建缓冲区"""
        buffer = cls()
        buffer.zarr_path = path
        src = zarr.open(path, mode='r')
        
        # 加载元数据
        buffer._meta = dict()
        if 'meta' in src:
            for key, value in src['meta'].items():
                buffer._meta[key] = value[:] if len(value.shape) > 0 else np.array(value)
        
        # 设置数据引用
        buffer._data = dict()
        src_data = src['data'] if 'data' in src else src
        
        for key in keys:
            data = src_data[key]
            if lazy_load:
                buffer._data[key] = ZarrReference(data)
            else:
                buffer._data[key] = data[:]
                
        return buffer
```

**`sampler.py`** - 序列采样器
```python
class SequenceSampler:
    """序列采样器，支持时间序列数据的采样"""
    
    def __init__(self, replay_buffer, sequence_length, pad_before=0, 
                 pad_after=0, keys=None, key_first_k=dict(), episode_mask=None):
        self.replay_buffer = replay_buffer
        self.sequence_length = sequence_length
        self.pad_before = pad_before
        self.pad_after = pad_after
        
        # 创建采样索引
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)
            
        self.indices = create_indices(
            episode_ends, sequence_length, episode_mask, 
            pad_before, pad_after
        )
        
    def sample_sequence(self, idx):
        """采样指定索引的序列"""
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # 性能优化：只加载使用的观测步数
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                sample = np.full((n_data,) + input_arr.shape[1:], 
                                  fill_value=0, dtype=input_arr.dtype)
                sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
            
            # 处理填充
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros((self.sequence_length,) + input_arr.shape[1:], 
                               dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            else:
                data = sample
            result[key] = data
        return result
```

#### 5.3 训练工具

**`checkpoint_util.py`** - Checkpoint管理
```python
class TopKCheckpointManager:
    """Top-K checkpoint管理器"""
    
    def __init__(self, save_dir, k=5, metric_name='loss', mode='min'):
        self.save_dir = save_dir
        self.k = k
        self.metric_name = metric_name
        self.mode = mode
        self.checkpoints = []
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics):
        """保存checkpoint"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'metrics': metrics
        }
        
        # 保存到临时文件
        temp_path = os.path.join(self.save_dir, f'checkpoint_temp_{epoch}.pt')
        torch.save(checkpoint, temp_path)
        
        # 更新checkpoint列表
        self.checkpoints.append((metrics[self.metric_name], temp_path, epoch))
        self.checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))
        
        # 保留Top-K
        if len(self.checkpoints) > self.k:
            _, old_path, _ = self.checkpoints.pop()
            if os.path.exists(old_path):
                os.remove(old_path)
                
        # 重命名最佳checkpoint
        best_metric, best_path, best_epoch = self.checkpoints[0]
        best_final_path = os.path.join(self.save_dir, f'best_checkpoint_epoch_{best_epoch}.pt')
        os.rename(best_path, best_final_path)
```

**`metric.py`** - 评估指标
```python
def get_action_accuracy(predictions, targets, valid_mask=None):
    """计算动作准确率"""
    if valid_mask is not None:
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
    
    # 计算L2距离
    l2_distances = torch.norm(predictions - targets, dim=-1)
    
    # 计算准确率（距离小于阈值的比例）
    threshold = 0.1  # 可调整的阈值
    accuracy = (l2_distances < threshold).float().mean()
    
    return accuracy.item()

def compute_pose_error(pred_poses, gt_poses, valid_mask=None):
    """计算姿态误差"""
    if valid_mask is not None:
        pred_poses = pred_poses[valid_mask]
        gt_poses = gt_poses[valid_mask]
    
    # 位置误差
    position_error = torch.norm(pred_poses[..., :3] - gt_poses[..., :3], dim=-1)
    
    # 旋转误差（使用四元数）
    pred_quat = rot_matrix_to_quaternion(pred_poses[..., 3:12])
    gt_quat = rot_matrix_to_quaternion(gt_poses[..., 3:12])
    rotation_error = torch.acos(torch.clamp(torch.abs(torch.sum(pred_quat * gt_quat, dim=-1)), 0, 1))
    
    return {
        'position_error': position_error.mean().item(),
        'rotation_error': rotation_error.mean().item(),
        'total_error': (position_error + rotation_error).mean().item()
    }
```

### 6. 工作空间模块 (`src/workspace/`)

**功能**: 管理训练和推理流程，提供统一的训练接口

#### 6.1 LegendVLA分布式训练工作空间

**`train_legendvla_deepspeed_workspace.py`** - LegendVLA分布式训练
```python
class TrainLegendVLAWorkspace(BaseWorkspace):
    """LegendVLA分布式训练工作空间"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setup_accelerator()
        self.setup_datasets()
        self.setup_models()
        self.setup_optimizers()
        self.setup_schedulers()
        self.setup_checkpoint_manager()
        
    def setup_accelerator(self):
        """设置Accelerator"""
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.training.gradient_accumulation_steps,
            mixed_precision=self.cfg.training.mixed_precision,
            log_with="wandb" if self.cfg.training.use_wandb else None,
            project_dir=self.cfg.training.output_dir
        )
        
    def setup_datasets(self):
        """设置数据集"""
        # VLA数据集
        self.vla_dataset = LegendVLADataset(
            zarr_paths=self.cfg.vla_dataset_paths,
            horizon=self.cfg.training.horizon,
            shape_meta=self.cfg.shape_meta,
            history=self.cfg.training.history,
            objective=self.cfg.objective,
            max_train_episodes=self.cfg.training.max_train_episodes
        )
        
        # VLM数据集
        if self.cfg.vlm_dataset_paths:
            self.vlm_dataset = LegendVLMDataset(
                dataset_paths=self.cfg.vlm_dataset_paths,
                split='train',
                weights=self.cfg.training.vlm_weights
            )
        else:
            self.vlm_dataset = None
            
        # 统一数据集
        self.unified_dataset = LegendUnifiedDataset(
            vla_dataset=self.vla_dataset,
            vlm_dataset=self.vlm_dataset,
            token_len_buckets=self.cfg.training.token_len_buckets
        )
        
        # 数据加载器
        self.train_dataloader = DataLoader(
            self.unified_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            collate_fn=self.unified_dataset.get_collator(),
            num_workers=self.cfg.training.num_workers
        )
        
    def setup_models(self):
        """设置模型"""
        # 预处理器
        self.preprocessor = PaliGemmaVLAProcessor(
            vision_model_name=self.cfg.model.vision_model_name,
            language_model_name=self.cfg.model.language_model_name,
            max_seq_len=self.cfg.model.max_seq_len
        )
        
        # 设置预处理器
        self.vla_dataset.set_preprocessor(self.preprocessor)
        if self.vlm_dataset:
            self.vlm_dataset.set_preprocessor(self.preprocessor)
            
        # 策略模型
        self.policy = LegendVLA(
            config=self.cfg.model,
            objective=self.cfg.objective
        )
        
        # 标准化器
        if self.cfg.training.use_normalizer:
            normalizer = self.vla_dataset.get_normalizer()
            self.vla_dataset.set_normalizer(normalizer)
            
    def setup_optimizers(self):
        """设置优化器"""
        # 参数分组
        vision_params = []
        language_params = []
        action_params = []
        
        for name, param in self.policy.named_parameters():
            if 'vision_encoder' in name:
                vision_params.append(param)
            elif 'language_model' in name:
                language_params.append(param)
            else:
                action_params.append(param)
                
        # 不同学习率
        self.optimizer = torch.optim.AdamW([
            {'params': vision_params, 'lr': self.cfg.training.vision_lr},
            {'params': language_params, 'lr': self.cfg.training.language_lr},
            {'params': action_params, 'lr': self.cfg.training.action_lr}
        ], weight_decay=self.cfg.training.weight_decay)
        
    def setup_schedulers(self):
        """设置学习率调度器"""
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=self.cfg.training.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=self.cfg.training.learning_rate,
            min_lr=self.cfg.training.min_learning_rate,
            warmup_steps=self.cfg.training.warmup_steps
        )
        
    def train_step(self, batch):
        """单步训练"""
        # 前向传播
        outputs = self.policy(
            images=batch['pixel_values'],
            text=batch['input_ids'],
            states=batch.get('states'),
            actions=batch.get('actions')
        )
        
        # 计算损失
        if self.cfg.objective == 'train_ar':
            loss = self.compute_ar_loss(outputs, batch['labels'], batch['attention_mask'])
        elif self.cfg.objective == 'train_flow':
            loss = self.compute_flow_loss(outputs, batch['actions'], batch['timesteps'])
        else:
            # 组合损失
            ar_loss = self.compute_ar_loss(outputs, batch['labels'], batch['attention_mask'])
            flow_loss = self.compute_flow_loss(outputs, batch['actions'], batch['timesteps'])
            loss = ar_loss + flow_loss
            
        # 反向传播
        self.accelerator.backward(loss)
        
        # 梯度裁剪
        if self.cfg.training.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(
                self.policy.parameters(), 
                self.cfg.training.max_grad_norm
            )
            
        # 优化器步骤
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return {
            'loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
    def train(self):
        """训练循环"""
        # 准备模型和优化器
        self.policy, self.optimizer, self.train_dataloader, self.scheduler = \
            self.accelerator.prepare(
                self.policy, self.optimizer, self.train_dataloader, self.scheduler
            )
            
        # 训练状态
        training_state = TrainingState()
        
        # 恢复checkpoint
        if self.cfg.training.resume_checkpoint_path:
            self.load_checkpoint(self.cfg.training.resume_checkpoint_path)
            
        # 训练循环
        for epoch in range(training_state.epoch, self.cfg.training.num_epochs):
            self.policy.train()
            
            for step, batch in enumerate(self.train_dataloader):
                # 训练步骤
                metrics = self.train_step(batch)
                
                # 记录指标
                if step % self.cfg.training.log_interval == 0:
                    self.accelerator.log(metrics, step=training_state.global_step)
                    
                # 保存checkpoint
                if step % self.cfg.training.save_interval == 0:
                    self.save_checkpoint(training_state)
                    
                training_state.global_step += 1
                
            training_state.epoch += 1
```

#### 6.2 快速分词器训练工作空间

**`train_fast_tokenizer.py`** - 快速分词器训练
```python
class TrainFastTokenizerWorkspace(BaseWorkspace):
    """快速分词器训练工作空间"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setup_accelerator()
        self.setup_datasets()
        self.setup_tokenizer()
        self.setup_optimizers()
        
    def setup_tokenizer(self):
        """设置分词器"""
        self.tokenizer = UniversalActionProcessor(
            vocab_size=self.cfg.tokenizer.vocab_size,
            max_seq_len=self.cfg.tokenizer.max_seq_len
        )
        
    def train_tokenizer(self):
        """训练分词器"""
        # 收集所有动作数据
        all_actions = []
        for batch in self.train_dataloader:
            actions = batch['actions']
            all_actions.append(actions.cpu().numpy())
            
        all_actions = np.concatenate(all_actions, axis=0)
        
        # 训练分词器
        self.tokenizer.train_tokenizer(
            all_actions, 
            num_epochs=self.cfg.tokenizer.num_epochs
        )
        
        # 保存分词器
        self.save_tokenizer()
```

## 数据处理流程详解

### 1. 原始数据处理

#### 1.1 MANO参数转换
```python
# mano_trans.py - 将45维MANO参数转换为15维PCA分量
def convert_mano_to_pca(mano_params, pca_components):
    """将MANO参数转换为PCA分量"""
    # 标准化
    normalized_params = (mano_params - pca_mean) / pca_std
    # PCA变换
    pca_params = np.dot(normalized_params, pca_components.T)
    return pca_params
```

#### 1.2 坐标系变换
```python
def transform_hand_to_camera(hand_pose, camera_extrinsic):
    """将手部姿态转换到相机坐标系"""
    # 构建手部齐次变换矩阵
    hand_matrix = pose_to_matrix(hand_pose)
    # 坐标系变换
    camera_matrix = camera_extrinsic @ hand_matrix
    # 转换回姿态表示
    camera_pose = matrix_to_pose(camera_matrix)
    return camera_pose
```

### 2. 数据格式规范

#### 2.1 Zarr数据结构
```
数据集根目录/
├── data/                           # 数据存储
│   ├── action/                     # 动作数据
│   │   ├── hand                   # 手部动作 (sum_frames, 30) float32
│   │   │   ├── 0                  # 左手15维PCA参数
│   │   │   └── 1                  # 右手15维PCA参数
│   │   └── wrist                  # 腕部动作 (sum_frames, 18) float32
│   │       ├── 0:3                # 左手平移 (x, y, z)
│   │       ├── 3:6                # 右手平移 (x, y, z)
│   │       ├── 6:12               # 左手旋转 (6D表示)
│   │       └── 12:18              # 右手旋转 (6D表示)
│   ├── state/                     # 状态数据
│   │   ├── hand                   # 手部状态 (sum_frames, 30) float32
│   │   └── wrist                  # 腕部状态 (sum_frames, 18) float32
│   ├── image/                     # 图像数据 (sum_frames, 384, 384, 3) uint8
│   ├── instruction/               # 指令文本 (sum_frames,) string
│   ├── instruction_num/           # 指令数量 (sum_frames,) int32
│   ├── extrinsic/                # 相机外参 (sum_frames, 16) float32
│   └── presence/                 # 手部可见性 (sum_frames,) int8
└── meta/                          # 元数据
    ├── episode_ends              # 回合结束索引 (num_episodes,) int64
    └── presence                  # 回合级可见性 (num_episodes,) int8
```

#### 2.2 数据增强策略
```python
class DataAugmentation:
    """数据增强策略"""
    
    def __init__(self, train_mode=True):
        if train_mode:
            self.aug_transform = transforms.Compose([
                # 颜色增强
                transforms.ColorJitter(
                    brightness=0.3, 
                    contrast=0.3, 
                    saturation=0.3, 
                    hue=0.1
                ),
                # 模糊增强
                transforms.GaussianBlur(
                    kernel_size=(3, 7), 
                    sigma=(0.1, 2.0)
                )
            ])
        else:
            self.aug_transform = None
            
    def __call__(self, images):
        """应用数据增强"""
        if self.aug_transform is None:
            return images
            
        augmented_images = []
        for img in images:
            # 转换为PIL图像
            img_pil = Image.fromarray(img)
            # 应用增强
            augmented_pil = self.aug_transform(img_pil)
            # 转换回numpy数组
            augmented_np = np.array(augmented_pil)
            augmented_images.append(augmented_np)
            
        return np.stack(augmented_images)
```

### 3. 训练流程详解

#### 3.1 单机训练流程
```bash
# 1. 环境设置
conda create -n legendvla python=3.8
conda activate legendvla
pip install -r requirements.txt

# 2. 数据准备
python data/egodex/mano_trans.py --data_root /path/to/egodex --output_root /path/to/output
python data/egodex/build_zarr.py --data_root /path/to/egodex --output /path/to/output.zarr

# 3. 训练
./scripts/pretrain_legendvla.sh
```

#### 3.2 分布式训练流程
```bash
# 1. 多机环境设置
# 确保所有机器可以SSH免密通信
# 确保共享存储访问权限

# 2. 配置文件设置
# 修改 src/config/acc_node0.yaml 中的多机信息

# 3. 启动分布式训练
./scripts/pretrain_legendvla_deepspeed.sh
```

#### 3.3 训练监控
```python
# 使用Weights & Biases监控训练
export WANDB_BASE_URL=https://api.bandw.top  # 国内镜像
wandb login

# 训练过程中会记录：
# - 损失曲线
# - 学习率变化
# - 梯度范数
# - 模型参数统计
# - 训练速度
```

### 4. 推理和可视化

#### 4.1 推理流程
```python
# inference.py - 推理脚本
def inference(model, dataloader, device):
    """推理函数"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 前向传播
            outputs = model(
                images=batch['pixel_values'].to(device),
                text=batch['input_ids'].to(device),
                states=batch.get('states', {}).to(device) if 'states' in batch else None
            )
            
            # 后处理
            actions = model.decode_actions(outputs)
            predictions.append(actions.cpu())
            
    return torch.cat(predictions, dim=0)
```

#### 4.2 可视化工具
```python
# visualize.py - 可视化脚本
def visualize_predictions(data_path, mano_root_dir, sample_id=0):
    """可视化预测结果"""
    # 加载数据
    data = torch.load(data_path)
    
    # 获取预测结果
    predictions = data['predictions'][sample_id]
    gt_actions = data['gt_actions'][sample_id]
    images = data['images'][sample_id]
    
    # 生成2D投影视频
    create_2d_projection_video(
        predictions, gt_actions, images,
        output_path='hand_motion_2d.mp4'
    )
    
    # 生成3D可视化视频
    create_3d_visualization_video(
        predictions, mano_root_dir,
        output_path='hand_motion_3d.mp4'
    )
```

## 技术特点总结

### 1. 多模态融合
- **视觉编码**: 使用SigLIP视觉编码器提取图像特征
- **语言理解**: 使用Gemma语言模型处理指令文本
- **状态编码**: 处理手部状态和动作信息
- **特征融合**: 通过交叉注意力机制融合多模态特征

### 2. 相对动作表示
- **坐标系变换**: 将手部姿态转换到相机坐标系
- **相对计算**: 计算相对于当前状态的动作变化
- **6D旋转表示**: 使用6D旋转表示避免四元数的奇异性
- **PCA降维**: 将45维MANO参数降维到15维PCA分量

### 3. 流式数据处理
- **大规模支持**: 支持TB级数据集的流式加载
- **内存优化**: 使用Zarr格式和懒加载减少内存占用
- **并行处理**: 支持多进程数据加载和预处理

### 4. 分布式训练
- **DeepSpeed集成**: 使用DeepSpeed ZeRO优化内存使用
- **多机支持**: 支持多机多卡分布式训练
- **混合精度**: 使用FP16加速训练
- **梯度累积**: 支持大批次训练

### 5. 模块化设计
- **组件解耦**: 各模块高度解耦，易于扩展
- **配置驱动**: 使用Hydra进行配置管理
- **插件化**: 支持不同的模型、策略和数据处理组件

这个项目实现了一个完整的视觉-语言-动作学习框架，从数据处理到模型训练再到推理可视化，提供了端到端的解决方案。通过模块化设计和分布式训练支持，可以处理大规模的多模态数据，学习复杂的双手操作任务。
