# LegendVLA — Qwen3-VL

## Overview

当前仓库的训练主线基于 **Qwen3-VL** backbone，采用 FSDP2 + Accelerate 分布式训练框架。核心组件：

| 组件 | 路径 |
|---|---|
| 模型主类 | `src/policy/legendvla.py` |
| VLM backbone | `src/model/vlm/qwen3_vl_backbone.py` |
| Action Expert (Qwen3 DiT) | `src/model/action_expert/qwen3_action_expert.py` |
| DiffLoss | `src/model/action/diffloss.py` |
| 训练 Workspace | `src/workspace/train_legendvla_workspace.py` |
| 实验配置 | `src/config/experiment/legendvla_qwen3_vl.yaml` |
| Accelerate FSDP2 配置 | `src/config/acc_config.yaml` |
| 数据 Collator | `src/dataset/unified_vla_collator.py` |
| Qwen3-VL Batching | `src/dataset/qwen3_vl_batching.py` |

训练使用 Hydra 管理配置，WandB 记录日志，WebDataset 格式加载数据。

## Environment Setup

### 基础环境安装

```bash
git clone <your-repo-url> EgoVLA
cd EgoVLA
bash scripts/install.sh
```

`install.sh` 会完成以下操作：

1. 安装系统依赖（build-essential、NCCL、pdsh 等）
2. 创建 conda 环境 `legendvla`（Python 3.10）
3. 安装 PyTorch 2.10.0 + CUDA 12.8
4. 编译安装 FlashAttention
5. 安装 `requirements.txt` 中的其余依赖

> **注意**：`transformers` 当前 pin 到特定 commit（修复了 Qwen3-VL batched video rope 和 flex_attention deprecated API 两个 issue），详见 `requirements.txt` 中的注释。

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

训练时 VLA 和 VLM 数据按 `vla_ratio`（默认 5:1）混合采样。

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

计算完成后将路径填入 `src/config/training/default.yaml` 的 `training.normalizer_path`。

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

训练入口：`train.py`，通过 Hydra 读取以下配置层级：

```text
src/config/train_config.yaml                    # 顶层入口，指定 experiment
  └── experiment/legendvla_qwen3_vl.yaml        # 实验配置，组合以下子配置
        ├── model/qwen3_vl_4b.yaml              # 模型结构、预训练路径、attention backend
        ├── data/unified_wds.yaml               # 数据加载、collator、dataloader
        ├── training/default.yaml               # 优化器、学习率、compile、gradient checkpointing
        ├── logging/default.yaml                # WandB、checkpoint 策略
        ├── dataset_paths/vla_wds.yaml          # VLA WebDataset shard 路径
        └── dataset_paths/vlm_wds.yaml          # VLM WebDataset shard 路径
```

可通过 Hydra override 覆盖任意配置项：

```bash
python train.py experiment=legendvla_qwen3_vl training.lr_warmup_steps=1000 dataloader.loader.batch_size=16
```

### 本地快速启动

```bash
python train.py
```

这会读取默认实验 `legendvla_qwen3_vl`，使用当前机器的所有可见 GPU。

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
ACC_CONFIG="src/config/acc_config.yaml"
SCRIPT="train.py"
ARGS="experiment=legendvla_qwen3_vl"
MASTER_ADDR="127.0.0.1"   # 修改为当前机器 IP
MASTER_PORT="18276"           # 修改为可用端口
GPU_COUNT="8"                 # 修改为实际 GPU 数
# -----------------------------------------------
```

脚本会自动设置 NCCL 通信参数（IB RDMA、socket interface 等），默认网卡 `eth1`，可通过环境变量 `RDMA_IFNAME` 覆盖。

#### Multi-Node

使用 `pretrain_legendvla_fsdp2_volc.sh`。在火山云控制台配置：

- **启动命令**：`bash`
- **启动参数**：`/path/to/project/scripts/pretrain_legendvla_fsdp2_volc.sh`（使用共享盘上的绝对路径）
- **Worker 数量**：设置为需要的节点数（如 2、4）
- **每 Worker GPU 数**：通常为 8

脚本会从 MLP 环境变量中自动获取 `MASTER_ADDR`、`MASTER_PORT`、`MACHINE_RANK`、`NNODES`，计算 `TOTAL_PROCESSES = NNODES * GPUS_PER_NODE`，然后调用 `accelerate launch` 启动分布式训练。

**网络配置**：脚本预设了适配火山云训练容器的 RDMA 参数：

```bash
NCCL_SOCKET_IFNAME=eth1          # socket 网卡
NCCL_IB_HCA=mlx5_1,...,mlx5_4   # IB 设备
NCCL_IB_DISABLE=0                # 启用 IB
```

如果容器内网卡名不同，在任务环境变量中设置 `RDMA_IFNAME=<your-interface>` 覆盖。

> **注意**：火山云脚本假设容器镜像中已预装所有训练依赖（PyTorch、Accelerate、FlashAttention 等），不需要 conda 激活。

---

### 裸金属多机训练（pdsh 方式）

适用于自行管理的多机集群，通过 pdsh + SSH 从 master 节点一键启动所有 worker。

前提条件：

- 多台服务器之间 SSH 免密通信
- 共享盘上的统一代码、数据、环境路径
- 安装 pdsh（`apt-get install pdsh`）

使用 `pretrain_legendvla_fsdp2.sh`：

```bash
bash scripts/pretrain_legendvla_fsdp2.sh
```

修改脚本顶部的 `CONFIGURATION` 区域：

```bash
NODES=(
    "172.18.1.150"    # node 0 (master)
    "172.18.1.151"    # node 1
)

SSH_USER=""                                    # SSH 用户名，留空则使用当前用户
PROJECT_DIR="/home/user/Projects/legendvla"    # 共享盘上的项目路径
GPUS_PER_NODE=8
MASTER_PORT=18276
```

脚本会自动为每个 node 分配 rank，拼接 NCCL 环境变量，通过 pdsh 在所有节点上并行启动 `accelerate launch`。按 Ctrl+C 会自动清理所有节点的训练进程。

### 单任务 Finetune

`scripts/finetune_single_task.py` 支持从 `vla_dataset_paths.yaml` 读取数据集列表，逐个或分布到多台机器上串行 finetune：

```bash
# 本地逐个 finetune
python scripts/finetune_single_task.py

# 分发到多台机器并行（每台机器串行跑分配到的子集）
python scripts/finetune_single_task.py --hosts 172.18.1.150,172.18.1.151

# 预览命令但不执行
python scripts/finetune_single_task.py --dry_run
```

### Resume / Finetune from Checkpoint

在 `src/config/training/default.yaml` 或命令行 override 中指定：

```bash
# 恢复训练（加载 optimizer state）
python train.py training.resume_checkpoint_path=/path/to/checkpoint

# Finetune（只加载模型权重）
python train.py training.finetune_checkpoint_path=/path/to/checkpoint
```

FSDP2 训练输出的 sharded checkpoint 可直接用于恢复。

### 关键训练参数速查

| 参数 | 位置 | 说明 |
|---|---|---|
| `dataloader.loader.batch_size` | `data/unified_wds.yaml` | 每 GPU batch size（默认 32） |
| `training.lr_warmup_steps` | `training/default.yaml` | 学习率 warmup 步数（默认 2000） |
| `training.eval_every` | `training/default.yaml` | 评估间隔（默认 5000 步） |
| `training.checkpoint_every` | `training/default.yaml` | checkpoint 保存间隔（默认 5000 步） |
| `optimizer.vlm.lr` | `training/default.yaml` | VLM backbone 学习率（默认 1e-4） |
| `optimizer.action.lr` | `training/default.yaml` | Action expert 学习率（默认 3e-4） |
| `optimizer.diffloss.lr` | `training/default.yaml` | DiffLoss 学习率（默认 8e-4） |
| `training.compile.enabled` | `training/default.yaml` | torch.compile 开关（默认 True） |
| `training.gradient_checkpointing` | `training/default.yaml` | 分组件 gradient checkpointing |
| `training.clipping.max_grad_norm` | `training/default.yaml` | 梯度裁剪阈值（默认 1.0） |
| `runtime.knowledge_insulation` | `model/qwen3_vl_4b.yaml` | 知识隔离（True/False/int N） |
| `pretrained.text_attn_implementation` | `model/qwen3_vl_4b.yaml` | Text attention backend（默认 flex_attention） |

## Inference

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

### WebSocket Serving

```bash
bash scripts/run_server.sh
```

相关代码位于 `src/serving/`，包含 WebSocket server/client 和 msgpack 序列化。

## Debug

### 调试脚本

`scripts/debug_start.sh` 提供了尽可能还原正式训练环境的调试启动方式：

```bash
bash scripts/debug_start.sh
```

通过环境变量控制调试行为：

- `CUDA_VISIBLE_DEVICES`：指定调试用的 GPU
- `ENABLE_DEBUGPY=1`：启用 debugpy 远程调试
- `DEBUGPY_PORT`：调试端口（默认 5679）
- `DEBUGPY_WAIT=1`：等待 VS Code 调试器连接
- `WANDB_MODE=disabled`：调试时关闭 WandB

### Nsight Profiling

```bash
nsys launch accelerate launch \
    --config_file src/config/acc_config.yaml \
    train.py experiment=legendvla_qwen3_vl
```

也可以在训练配置中开启内置 profiling：`training.profile=True`。

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

src/
├── config/
│   ├── train_config.yaml                 # top-level config (defaults to legendvla_qwen3_vl)
│   ├── acc_config.yaml                   # Accelerate FSDP2 config
│   ├── experiment/
│   │   ├── legendvla_qwen3_vl.yaml       # main experiment config
│   │   └── inference.yaml
│   ├── model/
│   │   └── qwen3_vl_4b.yaml             # model architecture & pretrained paths
│   ├── data/
│   │   └── unified_wds.yaml             # data pipeline, collator, dataloader
│   ├── training/
│   │   └── default.yaml                 # optimizer, scheduler, compile, checkpointing
│   ├── logging/
│   │   └── default.yaml                 # WandB & checkpoint config
│   └── dataset_paths/
│       ├── vla_wds.yaml                 # VLA WebDataset shard paths
│       └── vlm_wds.yaml                 # VLM WebDataset shard paths
│
├── dataset/
│   ├── vla_dataset.py                   # VLA WebDataset loader
│   ├── vlm_dataset.py                   # VLM WebDataset loader
│   ├── wds_dataset.py                   # WebDataset base utilities
│   ├── unified_vla_collator.py          # unified VLA+VLM collator
│   ├── qwen3_vl_batching.py            # Qwen3-VL chat formatting & batch processing
│   ├── data_transforms.py              # data augmentation transforms
│   └── normalizer_utils.py             # normalizer computation utilities
│
├── model/
│   ├── vlm/
│   │   ├── qwen3_vl_backbone.py        # Qwen3-VL backbone wrapper
│   │   ├── qwen3_vl_compile_patch.py   # torch.compile patches for Qwen3-VL
│   │   └── prefix_cache.py             # KV prefix cache for inference
│   ├── action/
│   │   ├── qwen3_action_expert.py      # Qwen3 DiT action expert
│   │   └── action_head.py              # action encoder/decoder heads
│   ├── common/
│   │   ├── diffloss.py                 # DiffLoss (shared diffusion/flow matching loss)
│   │   ├── gaussian_diffusion.py       # Gaussian diffusion scheduler (DDPM/DDIM)
│   │   ├── respace.py                  # timestep respacing
│   │   ├── diffusion_utils.py          # KL divergence, log-likelihood helpers
│   │   ├── normalizer.py               # action/state normalizer
│   │   ├── modules.py                  # shared modules (TimeEmbedding, AdaLNZero, etc.)
│   │   └── model_average.py            # EMA/SWA model averaging
│   └── vision/
│       ├── future_frame_encoder.py     # future frame target encoder
│       ├── temporal_attention.py       # temporal attention module
│       └── ema.py                      # EMA utilities
│
├── policy/
│   ├── legendvla.py                    # LegendVLA main policy class
│   ├── legendvla_loss.py               # loss computation
│   ├── legendvla_inference.py          # inference logic
│   └── legendvla_inference_wrapper.py  # inference wrapper for serving
│
├── workspace/
│   ├── base_workspace.py               # base workspace class
│   ├── train_legendvla_workspace.py    # training workspace (train/eval loop)
│   ├── compute_norm_stats.py           # normalizer pre-computation
│   ├── eval_utils.py                   # evaluation utilities
│   └── visualize_dataset_gt.py         # ground truth visualization
│
├── serving/
│   ├── websocket_policy_server.py      # WebSocket inference server
│   ├── websocket_policy_client.py      # WebSocket client
│   ├── serve_policy.py                 # policy serving entry
│   └── msgpack_numpy.py               # msgpack serialization for numpy
│
├── utils/                              # shared utilities (geometry, metrics, etc.)
└── tests/                              # unit tests & pretrain verification suite

scripts/
├── install.sh                           # environment setup
├── pretrain_legendvla_fsdp2_volc.sh     # Volcengine multi-node launcher
├── pretrain_legendvla_fsdp2_single_node.sh  # Volcengine single-node launcher
├── pretrain_legendvla_fsdp2.sh          # pdsh multi-node launcher (bare metal)
├── debug_start.sh                       # debug launcher with debugpy support
├── compute_norm_stats.sh                # normalizer computation
├── finetune_single_task.py              # per-dataset finetune scheduler
├── convert_zarr_to_wds.sh              # zarr → WebDataset conversion
├── convert_hf_to_wds.sh                # HuggingFace → WebDataset conversion
└── run_server.sh                        # inference server launcher

data/                                    # dataset-specific processing scripts
├── egodex/
├── HOI4D/
├── taco/
├── lerobot/
├── vlm/                                # VLM dataset processing
├── webdataset/                         # WebDataset utilities
├── zarr/                               # zarr utilities
├── convert_zarr_to_wds.py
├── convert_hf_to_wds.py
└── visualizer.py                       # dataset visualizer
```
