# LegendVLA 详细重构计划

> 本文件是重构的执行指南。每完成一个步骤，在对应条目前标记 `[x]`。
> 原则：纯重构，不改变任何模型功能和对外接口。

---

## Phase 1: 拆分 `src/policy/legendvla.py`（2093 行 → 4 个文件）

### 1.1 `src/policy/legendvla_utils.py`（新建 ~450 行）

**职责**：权重加载、冻结、mask 构建等工具函数。全部改为独立函数，尽量与 class 解耦。

#### 纯函数（不依赖 model 实例）

| 函数签名 | 来源 | 说明 |
|----------|------|------|
| `build_text_cache() -> KVCache` | L647-654 | 无依赖，纯工厂函数 |
| `build_causal_mask_and_position_ids(attention_mask, answer_start_idx, n_actions, num_action_tokens, dtype)` | L657-736 | 原 `self.num_action_tokens` 改为参数传入 |
| `split_full_mask_into_submasks(causal_mask, max_vlm_tokens, num_action_tokens)` | L738-757 | 原 `self.num_action_tokens` 改为参数传入 |
| `build_causal_mask_and_position_ids_for_text(q_len, attention_mask, kv_cache, dtype)` | L759-851 | 原本就不依赖 self，直接提取 |

#### 接收具体参数的函数（轻度依赖）

| 函数签名 | 来源 | 说明 |
|----------|------|------|
| `init_motion_token_embeddings(embed_tokens, vlm_hidden_size, motion_token_list)` | L284-300 | 传入 embed_tokens 和 hidden_size |
| `freeze_non_lora_weights_in_vlm(vision_tower, multi_modal_projector, joint_model)` | L591-612 | 传入三个子模块 |
| `freeze_non_lora_weights_in_ae(action_encoder, action_decoder, joint_model)` | L614-635 | 传入三个子模块 |
| `freeze_all_weights(model)` | L637-645 | 只需 named_parameters() |

#### 接收 model 实例的函数（重度依赖）

| 函数签名 | 来源 | 说明 |
|----------|------|------|
| `load_pretrained_vlm_weights(model)` | L302-406 | 访问 model.cfg, model.embed_tokens, model.vision_tower, model.multi_modal_projector, model.joint_model |
| `load_pretrained_pi05_weights(model)` | L408-589 | 访问大量 model 属性，传 model 最简洁 |

#### LegendVLA 中的委托方法

在 `legendvla.py` 的 `LegendVLA` 类中保留同名方法，内部委托给独立函数，保持对外接口不变：

```python
# legendvla.py 中
def load_pretrained_vlm_weights(self):
    from src.policy.legendvla_utils import load_pretrained_vlm_weights
    load_pretrained_vlm_weights(self)

def load_pretrained_pi05_weights(self):
    from src.policy.legendvla_utils import load_pretrained_pi05_weights
    load_pretrained_pi05_weights(self)

def freeze_non_lora_weights_in_vlm(self):
    from src.policy.legendvla_utils import freeze_non_lora_weights_in_vlm
    freeze_non_lora_weights_in_vlm(self.vision_tower, self.multi_modal_projector, self.joint_model)

def freeze_non_lora_weights_in_ae(self):
    from src.policy.legendvla_utils import freeze_non_lora_weights_in_ae
    freeze_non_lora_weights_in_ae(self.action_encoder, self.action_decoder, self.joint_model)

def freeze_all_weights(self):
    from src.policy.legendvla_utils import freeze_all_weights
    freeze_all_weights(self)

@torch.no_grad()
def init_motion_token_embeddings(self, motion_token_list):
    from src.policy.legendvla_utils import init_motion_token_embeddings
    init_motion_token_embeddings(self.embed_tokens, self.vlm_hidden_size, motion_token_list)

def build_causal_mask_and_position_ids(self, attention_mask, answer_start_idx, n_actions, dtype):
    from src.policy.legendvla_utils import build_causal_mask_and_position_ids
    return build_causal_mask_and_position_ids(
        attention_mask, answer_start_idx, n_actions, self.num_action_tokens, dtype
    )

def split_full_mask_into_submasks(self, causal_mask, max_vlm_tokens):
    from src.policy.legendvla_utils import split_full_mask_into_submasks
    return split_full_mask_into_submasks(causal_mask, max_vlm_tokens, self.num_action_tokens)

def build_causal_mask_and_position_ids_for_text(self, q_len, attention_mask, kv_cache, dtype):
    from src.policy.legendvla_utils import build_causal_mask_and_position_ids_for_text
    return build_causal_mask_and_position_ids_for_text(q_len, attention_mask, kv_cache, dtype)

def build_text_cache(self):
    from src.policy.legendvla_utils import build_text_cache
    return build_text_cache()
```

---

### 1.2 `src/policy/legendvla_loss.py`（新建 ~250 行）

**职责**：所有 loss 计算逻辑。接收 model 实例（因为需要调用 model 的 forward 子模块）。

| 函数签名 | 来源 | 说明 |
|----------|------|------|
| `compute_celoss(lm_head, final_logit_softcapping, ce_loss_fn, ignore_index, hidden_states, labels)` | L1711-1731 | 保留 `@torch.compile`，传入 `final_logit_softcapping` 值（float 或 None）而非函数引用 |
| `compute_ar_loss(model, batch)` | L1534-1607 | AR 训练：forward + CE loss。需要 model 做 forward |
| `compute_flow_loss(model, batch)` | L1609-1709 | Flow 训练：forward + flow loss。需要 model 做 forward |
| `compute_loss(model, batch)` | L1733-1896 | 联合训练：forward + CE + flow + diffusion loss |

#### `compute_celoss` 解耦示例

注意：`_apply_final_logit_softcapping` 是由 `self.final_logit_softcapping`（float 或 None）控制的，
Gemma2 有此特性，Gemma1 没有。传入值而非函数引用，避免依赖可能不存在的方法。

```python
def _apply_final_logit_softcapping(logits: torch.Tensor, final_logit_softcapping: float = None) -> torch.Tensor:
    """Apply final logit softcapping (Gemma2 feature). Pure function."""
    if final_logit_softcapping is not None:
        logits = logits / final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * final_logit_softcapping
    return logits

@torch.compile
def compute_celoss(
    lm_head: nn.Module,
    final_logit_softcapping: float,  # None 表示不启用 softcapping
    ce_loss_fn: nn.Module,           # model.CELoss
    ignore_index: int,
    hidden_states: torch.FloatTensor,
    labels: torch.LongTensor,
) -> torch.FloatTensor:
    logits = lm_head(hidden_states)
    logits = _apply_final_logit_softcapping(logits, final_logit_softcapping)
    logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
    labels = labels[:, 1:].contiguous().view(-1)
    ce_loss = ce_loss_fn(logits, labels)
    valid_num_labels = torch.sum(labels != ignore_index)
    return ce_loss / valid_num_labels.clamp(min=1)
```

#### LegendVLA 中的委托

```python
# legendvla.py 中
def compute_celoss(self, hidden_states, labels):
    from src.policy.legendvla_loss import compute_celoss
    return compute_celoss(
        self.lm_head, self.final_logit_softcapping,
        self.CELoss, self.ignore_index, hidden_states, labels
    )
```

`compute_ar_loss`、`compute_flow_loss`、`compute_loss` 因为需要调用 model 的多个子模块做 forward pass，传 model 实例更简洁：

```python
# legendvla.py 中
def forward(self, mode: str, batch: dict, **kwargs) -> dict:
    from src.policy.legendvla_loss import compute_loss, compute_ar_loss, compute_flow_loss
    if mode == "train":
        return compute_loss(self, batch)
    elif mode == "train_ar":
        return compute_ar_loss(self, batch)
    elif mode == "train_flow":
        return compute_flow_loss(self, batch)
    elif mode == "infer_action":
        return infer_action(self, batch, **kwargs)
    elif mode == "infer_vla":
        return infer_vla(self, batch, **kwargs)
    elif mode == "infer_vlm":
        return infer_vlm(self, batch, **kwargs)
    else:
        raise ValueError(f"Invalid mode: {mode}")
```

---

### 1.3 `src/policy/legendvla_inference.py`（新建 ~700 行）

**职责**：所有推理逻辑。

#### 独立推理函数（从 LegendVLA 方法提取）

| 函数签名 | 来源 | 说明 |
|----------|------|------|
| `infer_action(model, input, return_attn_weights=False)` | L988-1097 | Flow matching 动作推理 |
| `infer_single_step(model, input, kv_cache, dtype, return_attn_weights)` | L1099-1171 | 单步 VLM 推理 |
| `infer_vlm(model, input, max_new_tokens, ...)` | L1173-1350 | 多步自回归文本生成 |
| `infer_vla(model, input, max_new_tokens, ...)` | L1352-1509 | 自回归 VLA 动作推理 |

这些函数内部将 `self.xxx` 替换为 `model.xxx`，其余逻辑不变。

#### LegendVLAInference 类（完整搬迁）

| 内容 | 来源 | 说明 |
|------|------|------|
| `class LegendVLAInference` | L1916-2093 | 推理封装类，原样搬迁 |

---

### 1.4 `src/policy/legendvla.py`（精简后 ~550 行）

**职责**：LegendVLA 核心模型结构 + forward 分发 + 委托方法。

保留的内容：

| 内容 | 当前行号 | 说明 |
|------|----------|------|
| `LegendVLA.__init__` | L35-139 | 模型初始化 |
| `_apply_final_logit_softcapping` | L141-158 | |
| 所有 `@property` 方法 | L160-282 | 参数分组 |
| `_forward_siglip_and_text_embedding` | L854-986 | 视觉+文本 embedding |
| `psi_t` | L1511-1532 | Flow matching 插值 |
| `forward` | L1898-1913 | 分发器（改为调用外部函数） |
| 委托方法 | — | 见 1.1 和 1.2 中的委托代码 |

---

## Phase 2: 拆分 `src/dataset/legendvla_dataset.py`（~1500 行 → 6 个文件）

### 2.0 重构 `src/dataset/base_dataset.py`：提取 Zarr 数据集基类

**问题分析**：`LegendVLADataset` 和 `LegendVLALowLevelDataset` 存在大量重复代码。

#### 重复代码对比

| 逻辑 | LegendVLADataset | LegendVLALowLevelDataset | 差异 |
|------|------------------|--------------------------|------|
| 存储 shape_meta/motion_type/hand_ndim | ✅ | ✅ | 无 |
| 构建 sampler_cfg | ✅ (含 image steps) | ✅ (不含 image steps) | sampler_cfg 内容不同 |
| 遍历 zarr_paths 创建 replay_buffer | ✅ | ✅ | key_mapping 过滤不同 |
| 创建 train_mask + sampler | ✅ | ✅ | 无 |
| get_validation_dataset | ✅ | ✅ | 几乎相同 |
| __getitem__ | ✅ | ✅ | 相同模式 |
| __len__ | ✅ | ✅ | 完全相同 |
| _sample_to_data | ✅ (image+state+action) | ✅ (state+action only) | 完全不同 |
| get_collator | ✅ (LegendVLDataCollator) | ✅ (ConcatDataCollator) | 不同 |

#### 方案：新建 `BaseLegendZarrDataset` 基类

```python
class BaseLegendZarrDataset(torch.utils.data.Dataset):
    """
    Zarr 数据集基类，封装通用的 zarr 加载、采样、验证集创建逻辑。
    子类只需实现：
      - _build_sampler_cfg()  → 返回 sampler 配置 dict
      - _build_key_mapping(zarr_path) → 返回该 zarr 的 key mapping
      - _sample_to_data(sample) → 将采样结果转为模型输入
      - get_collator() → 返回 collator
    """

    def __init__(
        self,
        zarr_paths,
        shape_meta,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.motion_type = shape_meta['obs']['state']['type']
        self.hand_ndim = shape_meta['obs']['state']['hand']['shape'][-1] // 2

        # 子类实现
        self.sampler_cfg = self._build_sampler_cfg()

        # 通用存储
        self.replay_buffers = []
        self.train_masks = []
        self.samplers = []
        self.sampler_lens = []

        for zarr_path in zarr_paths:
            key_mapping = self._build_key_mapping(zarr_path)
            dataset_path = zarr_path['path']
            if not os.path.exists(dataset_path):
                print(f"Warning: Dataset path {dataset_path} does not exist, skipping.")
                continue

            replay_buffer = StreamingReplayBuffer.copy_from_path(
                dataset_path, key_mapping=key_mapping, lazy_load=self._lazy_load()
            )
            if len(replay_buffer) <= 1:
                print(f"Warning: Dataset has only {len(replay_buffer)} episodes, skipping.")
                continue

            try:
                val_mask = get_val_mask(n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
                train_mask = ~val_mask
                train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)
                sampler = SequenceSampler(
                    replay_buffer=replay_buffer, episode_mask=train_mask, **self.sampler_cfg
                )
            except Exception as e:
                print(f"Error creating sampler: {e}")
                continue

            self.replay_buffers.append(replay_buffer)
            self.train_masks.append(train_mask)
            self.samplers.append(sampler)
            self.sampler_lens.append(len(sampler))

            # 子类钩子：处理额外的 per-zarr 数据（如 weights, dataset_names）
            self._on_zarr_loaded(zarr_path, replay_buffer)

    # ---------- 子类必须实现 ---------- #
    def _build_sampler_cfg(self) -> dict:
        raise NotImplementedError

    def _build_key_mapping(self, zarr_path: dict) -> dict:
        raise NotImplementedError

    def _sample_to_data(self, sample):
        raise NotImplementedError

    def get_collator(self):
        raise NotImplementedError

    # ---------- 子类可选覆盖 ---------- #
    def _lazy_load(self) -> bool:
        """是否懒加载 zarr 数据。默认 True。"""
        return True

    def _on_zarr_loaded(self, zarr_path: dict, replay_buffer):
        """每个 zarr 加载完成后的钩子。默认空操作。"""
        pass

    def _on_validation_copy(self, val_set):
        """get_validation_dataset 中对 val_set 的额外处理。默认空操作。"""
        pass

    # ---------- 通用实现 ---------- #
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.samplers = []
        val_set.train_masks = []
        val_set.sampler_lens = []
        self._on_validation_copy(val_set)

        for i, replay_buffer in enumerate(self.replay_buffers):
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                episode_mask=~self.train_masks[i],
                **self.sampler_cfg
            )
            val_set.samplers.append(sampler)
            val_set.train_masks.append(~self.train_masks[i])
            val_set.sampler_lens.append(len(sampler))
        return val_set

    def __getitem__(self, idx):
        try:
            curr_idx, dataset_idx = idx, 0
            while curr_idx >= self.sampler_lens[dataset_idx]:
                curr_idx -= self.sampler_lens[dataset_idx]
                dataset_idx += 1
            sample = self.samplers[dataset_idx].sample_sequence(curr_idx)
            data = self._sample_to_data(sample)
            torch_data = dict_apply(
                data, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
            )
            return torch_data
        except Exception as e:
            warnings.warn(f"Error getting item {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self):
        return sum(self.sampler_lens)
```

#### LegendVLADataset 改为继承 BaseLegendZarrDataset

```python
class LegendVLADataset(BaseLegendZarrDataset):
    """VLA 数据集，继承通用 zarr 加载逻辑，增加图像处理、preprocessor、normalizer。"""

    def __init__(self, zarr_paths, shape_meta, seed=42, val_ratio=0.0, objective=None,
                 normalizer_dataloader_cfg=dict(), use_relative_action=False,
                 max_train_episodes=None, mode='train', depth_clip_range=None,
                 return_dataset_info=False):
        # VLA 特有字段
        self.preprocessor = None
        self.objective = objective
        self.normalizer_dataloader_cfg = normalizer_dataloader_cfg
        self.use_relative_action = use_relative_action
        self.normalizer = None
        self.depth_clip_range = depth_clip_range
        self.mode = mode
        self.return_dataset_info = return_dataset_info
        self.dataset_names = []
        self.aug_transform = ... if mode == 'train' else None

        # 用于 BaseRatioDataset 的 weights
        self._weights_list = []

        # 调用基类 __init__（会调用 _build_sampler_cfg, _build_key_mapping, _on_zarr_loaded）
        super().__init__(zarr_paths, shape_meta, seed, val_ratio, max_train_episodes)

        # 初始化 weights（BaseRatioDataset 逻辑）
        if self._weights_list:
            weights_sum = sum(self._weights_list)
            self.weights = [w / weights_sum for w in self._weights_list]
            self.dataset_lengths = self.sampler_lens
        else:
            self.weights = None
            self.dataset_lengths = None

    def _build_sampler_cfg(self):
        s = self.shape_meta
        return {
            'num_image_steps': s['obs']['rgb']['horizon'],
            'num_image_stride': s['obs']['rgb']['stride'],
            'num_state_steps': s['obs']['state']['horizon'],
            'num_state_stride': s['obs']['state']['stride'],
            'num_action_steps': s['action']['horizon'],
            'num_action_stride': s['action']['stride'],
        }

    def _build_key_mapping(self, zarr_path):
        return merge_key_mapping(zarr_path.get('mapping', None), self.motion_type)

    def _on_zarr_loaded(self, zarr_path, replay_buffer):
        weight = zarr_path.get('weight', None)
        if weight is not None:
            self._weights_list.append(weight)
        name = zarr_path.get('name', None) or pathlib.Path(str(zarr_path['path'])).stem
        self.dataset_names.append(name)

    def _on_validation_copy(self, val_set):
        val_set.mode = 'val' if self.mode == 'train' else self.mode
        val_set.aug_transform = None

    def _sample_to_data(self, sample):
        # ... 原有的图像+状态+动作处理逻辑（不变）
        ...

    # set_preprocessor, set_normalizer, get_normalizer, get_collator 保持不变
```

#### LegendVLALowLevelDataset 改为继承 BaseLegendZarrDataset

```python
class LegendVLALowLevelDataset(BaseLegendZarrDataset):
    """低级数据集，仅加载 state/action 用于 normalizer 拟合。"""

    def __init__(self, zarr_paths, shape_meta, seed=42, val_ratio=0.0,
                 max_train_episodes=None, return_numpy=True, use_relative_action=False,
                 normalizer_dataloader_cfg=None):
        self.return_numpy = return_numpy
        self.normalizer = None
        self.normalizer_dataloader_cfg = normalizer_dataloader_cfg
        self.use_relative_action = use_relative_action
        super().__init__(zarr_paths, shape_meta, seed, val_ratio, max_train_episodes)

    def _build_sampler_cfg(self):
        s = self.shape_meta
        return {
            'num_state_steps': s['obs']['state']['horizon'],
            'num_state_stride': s['obs']['state']['stride'],
            'num_action_steps': s['action']['horizon'],
            'num_action_stride': s['action']['stride'],
        }

    def _build_key_mapping(self, zarr_path):
        full_mapping = merge_key_mapping(zarr_path.get('mapping', None), self.motion_type)
        low_level_keys = ['wrist_state', 'hand_state', 'wrist_action', 'hand_action', 'extrinsic']
        return {k: full_mapping[k] for k in low_level_keys if k in full_mapping}

    def _lazy_load(self):
        return False  # normalizer 需要全量加载

    def _sample_to_data(self, sample):
        # ... 原有的 state+action 处理逻辑（不变）
        ...

    def get_collator(self):
        return ConcatDataCollator()

    # __getitem__ 覆盖：支持 return_numpy
    def __getitem__(self, idx):
        data = super().__getitem__(idx)  # 基类已经转为 torch
        if self.return_numpy:
            data = dict_apply(data, lambda x: x.numpy() if isinstance(x, torch.Tensor) else x)
        return data
```

---

### 2.1 `src/dataset/legendvla_dataset.py`（精简后 ~450 行）

保留：

| 内容 | 当前行号 | 说明 |
|------|----------|------|
| `build_default_key_mapping()` | L34-49 | |
| `merge_key_mapping()` | L52-59 | |
| `class LegendVLADataset` | L62-385 | 改为继承 BaseLegendZarrDataset |
| `class LegendUnifiedDataset` | L625-755 | 不变 |
| `class LegendVLALowLevelDataset` | L757-959 | 改为继承 BaseLegendZarrDataset |

imports 变更：
```python
from src.dataset.data_transforms import process_state_action, process_image
from src.dataset.collator import LegendVLDataCollator, ConcatDataCollator
from src.dataset.normalizer_utils import get_normalizer
from .base_dataset import BaseLegendZarrDataset, BaseDataCollator
```

---

### 2.2 `src/dataset/legendvlm_dataset.py`（新建 ~150 行）

| 内容 | 来源行号 | 说明 |
|------|----------|------|
| `class LegendVLMDataset` | L386-623 | 完整搬迁，不变 |

---

### 2.3 `src/dataset/collator.py`（新建 ~100 行）

| 内容 | 来源行号 | 说明 |
|------|----------|------|
| `class LegendVLDataCollator` | L962-1021 | VLA/VLM 数据 collator |
| `class ConcatDataCollator` | L1023-1044 | 拼接 collator |

---

### 2.4 `src/dataset/data_transforms.py`（新建 ~300 行）

| 内容 | 来源行号 | 说明 |
|------|----------|------|
| `get_relative_action()` | L1047-1065 | |
| `get_absolute_action()` | L1067-1100 | |
| `transform_hand_from_wrist_to_camera()` | L1102-1147 | |
| `process_state_action()` | L1149-1218 | |
| `process_image()` | L1220-1253 | |

---

### 2.5 `src/dataset/normalizer_utils.py`（新建 ~50 行）

| 内容 | 来源行号 | 说明 |
|------|----------|------|
| `get_normalizer()` | L1256-1289 | |

---

## Phase 3: 拆分 `src/workspace/train_legendvla_deepspeed_workspace.py`（949 行 → 2 个文件）

### 3.1 `src/workspace/eval_utils.py`（新建 ~300 行）

| 内容 | 来源行号 | 说明 |
|------|----------|------|
| `evaluation(workspace, accelerator, dataloader, step_log)` | L555-792 | 改为独立函数，接收 workspace 实例 |
| `save_topk_ckpt(workspace, accelerator, topk_manager, step_log)` | L805-822 | |
| `save_interval_ckpt(workspace, accelerator)` | L824-828 | |
| `save_checkpoint_accelerator(workspace, accelerator, path, tag)` | L794-803 | |
| `eval_with_averaged_model(accelerator, model, averaged_model)` | L912-935 | context manager，已经是独立函数 |

workspace 中改为委托调用：
```python
def evaluation(self, accelerator, dataloader, step_log):
    from src.workspace.eval_utils import evaluation
    evaluation(self, accelerator, dataloader, step_log)
```

### 3.2 `src/workspace/train_legendvla_deepspeed_workspace.py`（精简后 ~600 行）

保留：

| 内容 | 当前行号 | 说明 |
|------|----------|------|
| `TrainLegendVLAWorkspace.__init__` | L43-73 | |
| `TrainLegendVLAWorkspace.run` | L75-552 | 训练循环 |
| `TrainLegendVLAWorkspace.sample_fm_time` | L830-838 | |
| `TrainLegendVLAWorkspace.preprocess_batch` | L840-892 | |
| `TrainLegendVLAWorkspace.get_grouped_parameters` | L894-909 | |
| `main()` | L938-948 | |

---

## Phase 4: 测试代码迁移 + 清理

### 4.0 测试代码迁移到 `src/tests/`

所有散落在源码文件中的测试代码统一迁移到 `src/tests/` 目录，按源文件对应命名。

#### 目录结构

```
src/tests/
├── __init__.py
├── test_legendvla_dataset.py      ← legendvla_dataset.py 中的测试
├── test_joint_model.py            ← joint_model.py 中的测试
├── test_legendvla_hooks.py        ← legendvla_hooks_test.py 完整搬迁
├── test_metric.py                 ← metric.py 中的 __main__ 测试
├── test_plotting.py               ← plotting.py 中的 __main__ 测试
└── test_embedding_analysis.py     ← embedding_analysis.py 中的 __main__ 测试
```

#### 迁移明细

| 源文件 | 测试函数 | 目标文件 | 说明 |
|--------|----------|----------|------|
| `src/dataset/legendvla_dataset.py` L1292-1500 | `test_dataset_loading()` | `src/tests/test_legendvla_dataset.py` | 含 `if __name__` 入口 |
| `src/dataset/legendvla_dataset.py` L2197-2325 | `test_visualize_state_action()` | `src/tests/test_legendvla_dataset.py` | 含 `if __name__` 入口 |
| `src/model/moe/joint_model.py` L525-672 | `test_attention_functions()` | `src/tests/test_joint_model.py` | ~150 行测试代码 |
| `src/policy/legendvla_hooks_test.py` 全文 | 整个文件 | `src/tests/test_legendvla_hooks.py` | 完整搬迁，原文件删除 |
| `src/utils/metric.py` `if __name__` 部分 | main 测试代码 | `src/tests/test_metric.py` | |
| `src/utils/plotting.py` `if __name__` 部分 | main 测试代码 | `src/tests/test_plotting.py` | |
| `src/utils/embedding_analysis.py` `if __name__` 部分 | main 测试代码 | `src/tests/test_embedding_analysis.py` | |

#### 迁移后源文件处理

- 删除源文件中的 `def test_xxx()` 函数和 `if __name__ == "__main__"` 块
- `legendvla_hooks_test.py` 整个文件搬迁后删除
- 不影响 `compute_norm_stats.py`、`visualize_dataset_gt.py`、`serve_policy.py` 等独立可执行脚本的 `if __name__` 入口（这些是正式脚本，不是测试）

### 4.1 其他清理

- [ ] 确认 EgoVLA 相关代码是否废弃（`src/policy/egovla.py` 等）
- [ ] 修复 `scripts/compuete_norm_stats.sh` → `scripts/compute_norm_stats.sh`（如存在拼写错误）
- [ ] 清理 `train.py` 开头注释掉的 debug 代码

---

## 执行顺序

| 步骤 | 内容 | 验证方式 |
|------|------|----------|
| 1 | 新建 `legendvla_utils.py`，搬迁工具函数 | import 检查 |
| 2 | 新建 `legendvla_loss.py`，搬迁 loss 函数 | import 检查 |
| 3 | 新建 `legendvla_inference.py`，搬迁推理函数 + LegendVLAInference | import 检查 |
| 4 | 精简 `legendvla.py`，改为委托调用 | 训练验证 |
| 5 | 新建 `base_dataset.py` 中的 `BaseLegendZarrDataset` | import 检查 |
| 6 | 新建 `data_transforms.py` | import 检查 |
| 7 | 新建 `collator.py` | import 检查 |
| 8 | 新建 `normalizer_utils.py` | import 检查 |
| 9 | 新建 `legendvlm_dataset.py` | import 检查 |
| 10 | 精简 `legendvla_dataset.py`，改为继承 BaseLegendZarrDataset | 训练验证 |
| 11 | 新建 `eval_utils.py` | import 检查 |
| 12 | 精简 workspace | 训练验证 |
| 13 | 新建 `src/tests/`，迁移所有测试代码 | 测试运行 |
| 14 | Phase 4.1 其他清理 | 训练验证 |

**每个步骤**：单独提交为纯重构 commit（不改功能），提交后运行训练验证功能不变。

