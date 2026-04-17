# LegendVLA — Qwen3-VL + World Model

## Overview

当前仓库的训练主线基于 **Qwen3-VL** backbone，采用 **原生 torchrun + FSDP2** 的分布式训练框架（HSDP 自动开启：跨节点 replicate、节点内 NVLink shard）。流匹配动作专家与 DINOv3 未来帧蒸馏（world model）在同一训练 step 内联合优化，均由 Hydra 配置组装配。

| 组件 | 路径 |
|---|---|
| 模型主类 (LegendVLA) | `src/policy/legendvla.py` |
| Qwen3-VL backbone (含 MEM 时序注意力) | `src/model/vlm/qwen3_vl_backbone.py` |
| 统一 Qwen3 Expert (flow / world model) | `src/model/vlm/qwen3_expert.py` |
| DiffLoss (flow matching) | `src/model/common/diffloss.py` |
| 世界模型 frozen teacher (DINOv3) | `src/model/world_model/frozen_teacher.py` |
| 训练 Workspace | `src/workspace/train_legendvla_workspace.py` |
| FSDP2 / HSDP 工具 | `src/utils/distributed_utils.py` |
| 实验配置入口 | `src/config/experiment/legendvla_qwen3_vl.yaml` |
| 数据 Collator | `src/dataset/unified_vla_collator.py` |
| Qwen3-VL Batching | `src/dataset/qwen3_vl_batching.py` |

训练使用 Hydra 管理配置，WandB 记录日志，WebDataset 格式加载数据，FSDP2 原生 DCP 保存分片 checkpoint。

## Environment Setup

### 基础环境安装

```bash
git clone https://github.com/Psi-Robot/EgoVLA
cd EgoVLA
bash scripts/install.sh
```

`install.sh` 会完成以下操作：

1. 安装系统依赖（build-essential、NCCL、`pdsh`、`numactl` 等）
2. 创建 conda 环境 `legendvla`（Python 3.10）
3. 安装 PyTorch 2.10.0 + CUDA 12.8
4. 编译安装 FlashAttention
5. 安装 `requirements.txt` 中的其余依赖

> **注意**：`transformers` 当前 pin 到特定 commit（同时修复 Qwen3-VL batched video rope 和 flex_attention deprecated API 两个 issue），详见 `requirements.txt` 注释。

### 预训练权重

- **Qwen3-VL backbone**：`pretrained.model_name_or_path`（默认 `Qwen3-VL-2B-Instruct`）
- **DINOv3 frozen teacher**：`world_model.frozen_teacher.model_name_or_path`（默认 `dinov3-vitl16-pretrain-lvd1689m`，patch=16，4 register tokens，hidden=1024）

路径在 `src/config/model/qwen3_vl_2b.yaml` 与 `src/config/world_model/frozen_regression.yaml` 中修改。

### MANO 模型（仅数据处理 / 可视化需要）

前往 [MANO 官网](https://mano.is.tue.mpg.de/) 下载 `mano_v*_*.zip`，解压后放置为：

```text
manopth/
  mano/models/
    MANO_LEFT.pkl
    MANO_RIGHT.pkl
  manopth/__init__.py
```

### WandB 配置

```bash
# 可选国内镜像
export WANDB_BASE_URL=https://api.bandw.top
wandb login
```

## Data

### 数据格式

训练使用 **WebDataset** (`.tar` shards) 格式。数据集路径在以下配置中定义：

- VLA 数据集：`src/config/dataset_paths/vla_wds.yaml`
- VLM 数据集：`src/config/dataset_paths/vlm_wds.yaml`

训练时 VLA 和 VLM 数据按 `vla_ratio`（默认 `5/6`）混合采样。

**图像尺寸**：`data.target_image_size=[384, 384]`。启用 world model 时该值必须设置，保证 VLA 输入、MEM 时序注意力、以及 DINOv3 teacher 的 patch 对齐；`384/(16*upsample_factor=2)=12 → 12×12=144 query tokens/frame`，上采样后对齐 DINOv3 的 576 patch。

### 从 zarr 转换

每个原始数据集目录通常包含：

- `mano_trans.py`：将原始 MANO 参数转换为训练使用的手部状态表示
- `build_zarr.py`：读取视频、外参、语言和手部数据，封装成统一的 zarr 数据集

```bash
python mano_trans.py --data_root PATH/TO/DATASET --output_root PATH/TO/OUTPUT_DIR
python build_zarr.py --data_root PATH/TO/DATASET --output PATH/TO/OUTPUT_ZARR_DIR
```

zarr 转 WebDataset：

```bash
bash scripts/convert_zarr_to_wds.sh
```

VLM 原始数据（HuggingFace parquet）转 WebDataset：

```bash
bash scripts/convert_hf_to_wds.sh
```

### Normalizer

训练依赖预计算的 normalizer。计算方式：

```bash
bash scripts/compute_norm_stats.sh
```

或直接调用：

```bash
python -m src.workspace.compute_norm_stats \
    --config src/config/experiment/legendvla_qwen3_vl.yaml \
    --output_dir outputs/normalizer/<your-output-dir> \
    --num_workers 64 \
    --max_total_shards 5000
```

计算完成后把路径填入 `src/config/training/default.yaml` 的 `training.normalizer_path`。

### 数据可视化

```bash
python data/visualizer.py \
  --mano_root PATH/TO/MANO_MODEL \
  --manopth_path PATH/TO/MANOPTH_LIB \
  --zarr PATH/TO/DATASET \
  --index N \
  --intrinsic_path PATH/TO/INTR_NPY \
  --camera_view
```

## Training

### 配置结构

训练入口：`train.py`（Hydra）。默认 experiment：`legendvla_qwen3_vl`。

```text
src/config/train_config.yaml                    # 顶层入口
  └── experiment/legendvla_qwen3_vl.yaml        # 组合下列子配置
        ├── model/qwen3_vl_2b.yaml              # 模型结构、backbone、flow 专家、MEM
        ├── data/unified_wds.yaml               # 数据加载、collator、dataloader
        ├── training/default.yaml               # 优化器、调度器、FSDP、compile、梯度检查点
        ├── logging/default.yaml                # WandB、checkpoint 策略
        ├── dataset_paths/vla_wds.yaml          # VLA shard 路径
        ├── dataset_paths/vlm_wds.yaml          # VLM shard 路径
        ├── world_model/frozen_regression.yaml  # 世界模型（可切 none 关闭）
        └── diffloss/flow.yaml                  # DiffLoss（可切 none 关闭）
```

可通过 Hydra override 覆盖任意配置项：

```bash
# 关闭 world model、把 batch 改大、延长 warmup
python train.py \
    world_model=none \
    dataloader.loader.batch_size=16 \
    training.lr_warmup_steps=1000
```

### 本地快速启动

```bash
python train.py
```

这会读取默认 experiment，使用当前机器可见的所有 GPU（单卡亦可跑）。多卡训练请直接用下面的 launcher 脚本（内部走 `torchrun`）。

---

### 火山云（Volcengine MLP）训练

火山云机器学习平台（MLP）会为每个 worker 注入以下环境变量：

| 环境变量 | 含义 |
|---|---|
| `MLP_WORKER_0_HOST` | rank-0 worker 地址 |
| `MLP_WORKER_0_PORT` | rank-0 worker 端口 |
| `MLP_ROLE_INDEX` | 当前 worker 的 rank |
| `MLP_WORKER_NUM` | worker 总数 |
| `MLP_WORKER_GPU` | 每个 worker 的 GPU 数 |

脚本会自动读取这些变量，无需手动指定。

#### Single Node

使用 `pretrain_legendvla_fsdp2_single_node.sh`：

```bash
bash scripts/pretrain_legendvla_fsdp2_single_node.sh
```

在火山云控制台配置：

- **启动命令**：`bash`
- **启动参数**：`scripts/pretrain_legendvla_fsdp2_single_node.sh`

脚本内默认 8 卡。如需修改，编辑脚本顶部的 `CONFIGURATION` 区域：

```bash
# ---------------- CONFIGURATION ----------------
SCRIPT="train.py"
ARGS="experiment=legendvla_qwen3_vl"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-18276}"
GPU_COUNT="8"
# -----------------------------------------------
```

脚本会自动设置 NCCL 通信参数（IB RDMA、socket interface 等），默认网卡 `eth1`，可通过环境变量 `RDMA_IFNAME` 覆盖。

#### Multi-Node

使用 `pretrain_legendvla_fsdp2_volc.sh`。在火山云控制台配置：

- **启动命令**：`bash`
- **启动参数**：`/path/to/project/scripts/pretrain_legendvla_fsdp2_volc.sh`（共享盘上的绝对路径）
- **Worker 数量**：设置为需要的节点数（如 2、4）
- **每 Worker GPU 数**：通常为 8

脚本从 MLP 环境变量自动获取 `MASTER_ADDR`、`MASTER_PORT`、`NODE_RANK`、`NNODES`、`GPUS_PER_NODE`，然后通过 `torchrun --no-python` 调用 `scripts/numa_bind_wrapper.sh`，确保每个进程按 GPU 拓扑绑 NUMA（见 `Per-rank NUMA binding` 小节）。

**网络配置**（已预设适配火山云训练容器的 RDMA 参数）：

```bash
NCCL_SOCKET_IFNAME=eth1              # socket 网卡
NCCL_IB_HCA=mlx5_1,...,mlx5_4        # IB 设备
NCCL_IB_DISABLE=0                    # 启用 IB
```

如果容器内网卡名不同，在任务环境变量中设置 `RDMA_IFNAME=<your-interface>` 覆盖。

> **注意**：火山云脚本假设容器镜像中已预装所有训练依赖（PyTorch、FlashAttention、`numactl` 等），不执行 conda 激活。

---

### 裸金属多机训练（pdsh 方式）

适用于自行管理的多机集群，通过 pdsh + SSH 从 master 节点一键启动所有 worker。

前提条件：

- 多台服务器之间 SSH 免密通信
- 共享盘上的统一代码、数据、环境路径
- 安装 pdsh（`apt-get install pdsh`）与 numactl

使用 `pretrain_legendvla_fsdp2.sh`：

```bash
bash scripts/pretrain_legendvla_fsdp2.sh
```

修改脚本顶部的 `CONFIGURATION` 区域：

```bash
NODES=(
    "172.18.0.101"    # node 0 (master)
    "172.18.0.104"
    "172.18.0.105"
    "172.18.0.110"
)
SSH_USER=""
GPUS_PER_NODE=8
MASTER_PORT=18276
```

脚本会为每个 node 自动分配 rank，通过 pdsh 并行在所有节点上起 `torchrun --no-python bash scripts/numa_bind_wrapper.sh train.py ...`。按 Ctrl+C 触发 `remote_cleanup`，在所有节点 pkill `torchrun` 与 `train.py`。

### Per-rank NUMA binding

`scripts/numa_bind_wrapper.sh` 在每个 torchrun 子进程里先根据 `LOCAL_RANK` 从 `/sys/bus/pci/devices/.../numa_node` 查询对应 GPU 的 NUMA 节点，再用 `numactl --cpunodebind=N --membind=N` 启动 Python。Dataloader worker fork 自主进程，会继承该绑定。这避免了跨 socket 访问 DRAM 导致的吞吐抖动。

Rank 0 启动时会打印 CPU 亲和性；正常 2 socket 机器应看到约一半核心，若看到全部核心说明 wrapper 没生效。

### 单任务 Finetune

`scripts/finetune_single_task.py` 支持从 `vla_dataset_paths.yaml` 读取数据集列表，逐个或分发到多台机器串行 finetune：

```bash
# 本地逐个 finetune
python scripts/finetune_single_task.py

# 分发到多台机器并行（每台机器串行跑分配到的子集）
python scripts/finetune_single_task.py --hosts 172.18.1.150,172.18.1.151

# 预览命令但不执行
python scripts/finetune_single_task.py --dry_run
```

### Resume / Finetune from Checkpoint

训练输出的是 FSDP2 原生 DCP sharded checkpoint（目录形式，内含 `.metadata` 与 `__*_*.distcp`）：

```bash
# 恢复训练（加载 optimizer state + 调度器 + 训练状态）
python train.py training.resume=True \
    training.resume_checkpoint_path=/path/to/checkpoint

# Finetune（只加载模型权重，optimizer / scheduler 全新开始）
python train.py training.finetune_checkpoint_path=/path/to/checkpoint
```

如需把 DCP 分片 checkpoint 导出为单 `.pt`（便于部署、评估、权重分析）：

```bash
bash scripts/convert_fsdp_pt.sh
# 或
python -m src.utils.convert_fsdp_checkpoint \
    --checkpoint /path/to/update_step=NNNNN
```

支持原生 DCP 与历史 Accelerate 布局（`pytorch_model_fsdp_0/` 子目录）两种输入。

### 关键训练参数速查

| 参数 | 位置 | 说明 |
|---|---|---|
| `dataloader.loader.batch_size` | `data/unified_wds.yaml` | 每 GPU batch size（默认 20） |
| `training.steps_per_epoch` | `training/default.yaml` | 每 epoch 步数（默认 100000） |
| `training.num_epochs` | `training/default.yaml` | epoch 数（默认 2） |
| `training.gradient_accumulation_steps` | `training/default.yaml` | 梯度累积（默认 1） |
| `training.lr_warmup_steps` | `training/default.yaml` | 学习率 warmup 步数（默认 2000） |
| `training.vlm_freeze_steps` | `training/default.yaml` | 阶段性冻结 VLM 的步数（0 关闭） |
| `training.vlm_rewarmup_steps` | `training/default.yaml` | VLM 解冻后的 re-warmup 步数 |
| `training.eval_every` | `training/default.yaml` | 评估间隔（默认 5000） |
| `training.checkpoint_every` | `training/default.yaml` | topk checkpoint 间隔（默认 5000） |
| `training.ckpt_save_interval` | `training/default.yaml` | 整点保存间隔（默认 10000） |
| `training.fsdp.reshard_after_forward` | `training/default.yaml` | FSDP2 反向前 reshard（默认 False） |
| `training.fsdp.enable_prefetch` | `training/default.yaml` | 显式 all-gather prefetch（默认 True） |
| `training.compile.{vision,text,flow,diffloss,world_model}` | `training/default.yaml` | 分组件 `torch.compile` 开关 |
| `training.gradient_checkpointing.{text,vision,action_expert}` | `training/default.yaml` | 分组件 gradient checkpointing（`enabled` + `every_n`） |
| `training.clipping.max_grad_norm` | `training/default.yaml` | 梯度裁剪阈值（默认 1.0） |
| `optimizer.vlm.lr` | `training/default.yaml` | VLM backbone 学习率（默认 1e-4） |
| `optimizer.action.lr` | `training/default.yaml` | Flow 专家学习率（默认 3e-4） |
| `optimizer.ar_action_heads.lr` | `training/default.yaml` | DiffLoss / state 头学习率（默认 8e-4） |
| `optimizer.world_model.lr` | `training/default.yaml` | 世界模型专家学习率（默认 3e-4） |
| `flow.num_inference_steps` | `training/default.yaml` | flow 推理步数（默认 10） |
| `flow.num_parallel_t` | `training/default.yaml` | 每 step 并行采样的 flow 时间点数 |
| `runtime.knowledge_insulation` | `model/qwen3_vl_2b.yaml` | 知识隔离（True/False/int N） |
| `pretrained.text_attn_implementation` | `model/qwen3_vl_2b.yaml` | text/专家 attention backend（默认 `flex_attention`） |
| `policy.backbone.mem_temporal_attention.enabled` | `model/qwen3_vl_2b.yaml` | MEM 时序注意力开关 |
| `policy.backbone.mem_temporal_attention.every_n_layers` | `model/qwen3_vl_2b.yaml` | MEM 插入频率（默认每 4 层） |
| `world_model.enabled` + `world_model=frozen_regression\|none` | 配置组切换 | 开关 DINOv3 未来帧蒸馏分支 |
| `diffloss=flow\|none` | 配置组切换 | 开关 AR-action via diffloss |
| `world_model.future_frame.{horizon,stride}` | `world_model/frozen_regression.yaml` | 预测帧数与采样步长（默认 1 帧、30 步≈1 秒） |
| `world_model.action_conditioning` | `world_model/frozen_regression.yaml` | 是否把动作编码作为 WM 前缀条件 |
| `policy.loss_config.wm_loss_weight` | `model/qwen3_vl_2b.yaml` | 世界模型 loss 权重（默认 1.0） |
| `policy.loss_config.flow_loss_weight` | `model/qwen3_vl_2b.yaml` | flow matching loss 权重（默认 1.0） |
| `policy.loss_config.diffusion_loss_weight` | `model/qwen3_vl_2b.yaml` | diffloss 权重（默认 2.0） |

## Inference

### 推理配置

推理入口配置见 `src/config/experiment/inference.yaml`，通过 `src.policy.legendvla_inference_wrapper.LegendVLAInference` 包装：给定训练 run 的 `.hydra/config.yaml` 与 DCP checkpoint 即可重建整网，并可覆盖 `pretrained_vlm_path` / `teacher_path` 走本地镜像。

### WebSocket Serving

```bash
bash scripts/run_server.sh
```

相关代码位于 `src/serving/`，包含 WebSocket server / client 和 msgpack 序列化。

### 推理可视化

```bash
python rerun_inference_vis.py \
  --inference_zarr_path /path/to/predictions.zarr \
  --origin_zarr_path /path/to/origin.zarr \
  --sample_idx 200 \
  --save_path /path/to/output \
  --target_width 1920 \
  --target_height 1080
```

需要安装 `rerun-sdk`。

### 评估

```bash
bash scripts/run_eval.sh
```

离线评估入口 `evaluate.py`，配置在 `src/config/eval_config.yaml`。

## Debug

### 调试脚本

`scripts/debug_start.sh` 以少量步数、关闭持久 worker、开启 WandB disabled 的方式启动训练，方便 VS Code 挂 debugpy：

```bash
bash scripts/debug_start.sh
```

通过环境变量控制调试行为：

- `CUDA_VISIBLE_DEVICES`：指定调试用的 GPU
- `ENABLE_DEBUGPY=1`：启用 debugpy 远程调试（监听 rank 0）
- `DEBUGPY_PORT`：调试端口（默认 5679）
- `DEBUGPY_WAIT=1`：等待 VS Code 调试器连接
- `WANDB_MODE=disabled`：关闭 WandB

> **说明**：脚本顶部的历史 `accelerate launch` 调用仍保留做参考，但当前训练主链路已统一为 `torchrun`，新环境请以 `pretrain_legendvla_fsdp2*.sh` 为准。如需用 debugpy 跑，把脚本里的命令改为 `torchrun --nproc_per_node=$NUM_GPUS --master_port=29501 train.py experiment=legendvla_qwen3_vl data.dataloader.loader.num_workers=0 ...`。

### Nsight Profiling

可直接在 launcher 外套 `nsys`：

```bash
nsys launch bash scripts/pretrain_legendvla_fsdp2_single_node.sh
```

也可以在训练配置中开启内置 torch profiler：`training.profile=True`，trace 会写到 `${output_dir}/trace/`。

### VSCode 远程调试注意事项

- 不要在软链接路径下打开 VS Code
- 只在 Rank 0 上挂起调试器
- 设置 NCCL 超时避免其他 rank 提前退出：

```bash
export NCCL_TIMEOUT=3600000
export NCCL_ASYNC_ERROR_HANDLING=1
```

## Project Structure

```text
train.py                                  # training entry point (Hydra)
evaluate.py                               # offline evaluation entry point
visualize.py                              # training-sample visualization entry point
rerun_inference_vis.py                    # inference rerun visualization
rerun_datasets_zarr_vis.py                # raw-dataset rerun visualization

src/
├── config/
│   ├── train_config.yaml                 # top-level (defaults to legendvla_qwen3_vl)
│   ├── eval_config.yaml                  # offline evaluation config
│   ├── inference_config.yaml             # WebSocket serving config
│   ├── experiment/
│   │   ├── legendvla_qwen3_vl.yaml       # main experiment
│   │   └── inference.yaml                # WebSocket serving experiment
│   ├── model/
│   │   └── qwen3_vl_2b.yaml              # backbone + flow expert + MEM + losses
│   ├── data/
│   │   └── unified_wds.yaml              # WDS pipeline, collator, dataloader
│   ├── training/
│   │   └── default.yaml                  # optimizer, scheduler, FSDP, compile, GC
│   ├── logging/
│   │   └── default.yaml                  # WandB + Hydra output dir
│   ├── world_model/                      # world model config group
│   │   ├── frozen_regression.yaml        # DINOv3 teacher + Qwen3Expert
│   │   └── none.yaml                     # disable WM entirely
│   ├── diffloss/                         # diffloss config group
│   │   ├── flow.yaml                     # flow-matching diffloss (default)
│   │   └── none.yaml                     # disable AR-action via diffloss
│   └── dataset_paths/
│       ├── vla_wds.yaml                  # VLA WebDataset shard paths
│       └── vlm_wds.yaml                  # VLM WebDataset shard paths
│
├── dataset/
│   ├── vla_dataset.py                    # VLA WebDataset loader
│   ├── vlm_dataset.py                    # VLM WebDataset loader
│   ├── wds_dataset.py                    # WebDataset base utilities
│   ├── unified_vla_collator.py           # unified VLA+VLM collator
│   ├── qwen3_vl_batching.py              # Qwen3-VL chat formatting & batch processing
│   ├── data_transforms.py                # augmentation transforms
│   ├── normalizer_utils.py               # normalizer computation helpers
│   └── sanity_checks.py                  # non-finite / shape checks on samples
│
├── model/
│   ├── vlm/
│   │   ├── qwen3_vl_backbone.py          # Qwen3-VL backbone wrapper
│   │   ├── qwen3_vl_compile_patch.py     # torch.compile patches for Qwen3-VL
│   │   ├── prefix_cache.py               # KV prefix cache for flow / WM experts
│   │   ├── qwen3_expert.py               # Qwen3Expert (DiT for flow, plain for WM)
│   │   └── temporal_attention.py         # MEM multi-frame temporal attention
│   ├── action/
│   │   └── action_head.py                # Fourier encoders + MLP projectors
│   ├── common/
│   │   ├── diffloss.py                   # DiffLoss (flow-matching default)
│   │   ├── gaussian_diffusion.py         # DDPM / DDIM scheduler
│   │   ├── respace.py                    # timestep respacing
│   │   ├── diffusion_utils.py            # KL / log-likelihood helpers
│   │   ├── normalizer.py                 # action / state normalizer
│   │   ├── modules.py                    # TimeEmbedding, AdaLNZero, etc.
│   │   ├── model_average.py              # EMA / SWA averaging
│   │   ├── dict_of_tensor_mixin.py       # state-dict helpers
│   │   └── utils.py                      # misc model utils
│   └── world_model/
│       └── frozen_teacher.py             # frozen DINOv3 ViT target extractor
│
├── policy/
│   ├── legendvla.py                      # LegendVLA main policy class
│   ├── legendvla_loss.py                 # CE + flow + diffusion + WM loss
│   ├── legendvla_inference.py            # flow / AR / VLM inference
│   └── legendvla_inference_wrapper.py    # serving wrapper
│
├── workspace/
│   ├── base_workspace.py                 # Hydra-driven base workspace
│   ├── train_legendvla_workspace.py      # FSDP2 training workspace (train / eval loop)
│   ├── compute_norm_stats.py             # normalizer pre-computation
│   ├── eval_utils.py                     # evaluation + checkpoint save/load
│   └── visualize_dataset_gt.py           # ground-truth visualization
│
├── serving/
│   ├── websocket_policy_server.py        # WebSocket inference server
│   ├── websocket_policy_client.py        # WebSocket client
│   ├── serve_policy.py                   # serving entry
│   └── msgpack_numpy.py                  # msgpack for numpy
│
├── utils/
│   ├── distributed_utils.py              # init_distributed + apply_fsdp2 (HSDP-aware)
│   ├── fsdp_app_state.py                 # DCP Stateful app-state container
│   ├── checkpoint_util.py                # checkpoint save/load
│   ├── convert_fsdp_checkpoint.py        # DCP → single .pt exporter
│   ├── compile_utils.py                  # torch.compile helpers
│   ├── scheduler_utils.py                # LR scheduler builder
│   ├── training_utils.py                 # optimizer groups, grad clip, GC hooks
│   ├── sample_utils.py                   # span mask generator
│   ├── profiler_utils.py                 # torch profiler trace handler
│   └── ...                               # metrics, geometry, MANO vis, etc.
│
└── tests/                                # unit tests & pretrain verification suite

scripts/
├── install.sh                                    # environment setup
├── pretrain_legendvla_fsdp2_volc.sh              # Volcengine multi-node launcher (torchrun)
├── pretrain_legendvla_fsdp2_single_node.sh       # Volcengine single-node launcher (torchrun)
├── pretrain_legendvla_fsdp2.sh                   # pdsh multi-node launcher (torchrun)
├── numa_bind_wrapper.sh                          # per-rank NUMA binding wrapper
├── debug_start.sh                                # debug launcher with debugpy support
├── compute_norm_stats.sh                         # normalizer computation
├── finetune_single_task.py                       # per-dataset finetune scheduler
├── convert_zarr_to_wds.sh                        # zarr → WebDataset conversion
├── convert_hf_to_wds.sh                          # HuggingFace → WebDataset conversion
├── convert_fsdp_pt.sh                            # DCP checkpoint → single .pt
├── run_server.sh                                 # WebSocket inference server
├── run_eval.sh                                   # offline evaluation
├── sync_checkpoints.sh                           # rsync checkpoints across nodes
└── sync_ssh_config.sh                            # sync SSH config across nodes

data/                                    # dataset-specific processing scripts
├── egodex/
├── HOI4D/
├── taco/
├── lerobot/
├── vlm/                                 # VLM dataset processing
├── webdataset/                          # WebDataset utilities
├── zarr/                                # zarr utilities
├── convert_zarr_to_wds.py
├── convert_hf_to_wds.py
└── visualizer.py                        # dataset visualizer
```
