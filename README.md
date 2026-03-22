# LegendVLA with Qwen3-VL

## Overview

当前仓库仅保留 **最新的 Qwen3-VL LegendVLA 主线**：

- 模型主类：`src/policy/legendvla.py`
- 训练配置：`src/config/experiment/legendvla_qwen3_vl.yaml`
- 默认训练入口：`src/config/train_config.yaml`
- Qwen3-VL batching：`src/dataset/qwen3_vl_batching.py`
- Unified Collator：`src/dataset/unified_vla_collator.py`

旧的 PaliGemma / MoE 模型实现已经移除，不再作为可用训练路径保留。

## Environment Setup

如果需要多机训练，确保你的多台服务器满足以下要求：

- 有共享盘，代码、数据、环境最好都放在共享盘上，并通过统一路径访问。
- 不同服务器上的 UID 和 GID 尽量一致，避免权限问题。
- 多机之间配置好 ssh 免密通信，DeepSpeed / Accelerate 启动时会依赖这些连接。

安装环境：

```bash
git clone <your-repo-url> LegendVLA-transformers
cd LegendVLA-transformers
./scripts/install.sh
```

前往 MANO [官网](https://mano.is.tue.mpg.de/)，下载模型 `mano_v*_*.zip`，解压后按照如下方式放置：

```text
manopth/
  mano/
    models/
      MANO_LEFT.pkl
      MANO_RIGHT.pkl
      ...
  manopth/
    __init__.py
    ...
```

配置 wandb（可选国内镜像）：

```bash
export WANDB_BASE_URL=https://api.bandw.top
wandb login
```

## Data Processing

### For each dataset

每个数据集目录通常包含两个脚本：

- `mano_trans.py`：将原始 MANO 参数转换为训练使用的手部状态表示。
- `build_zarr.py`：读取视频、外参、语言和手部数据，封装成统一的 zarr 数据集。

使用方式：

```bash
python mano_trans.py --data_root PATH/TO/DATASET --output_root PATH/TO/OUTPUT_DIR
python build_zarr.py --data_root PATH/TO/DATASET --output PATH/TO/OUTPUT_ZARR_DIR
```

封装后的 zarr 目录格式示例：

```text
├── data
│   ├── action
│   │   ├── hand
│   │   └── wrist
│   ├── extrinsic
│   ├── image
│   ├── instruction
│   └── state
│       ├── hand
│       └── wrist
└── meta
    ├── episode_ends
    └── presence
```

### Dataset visualizer

`data/visualizer.py` 可用于可视化 zarr 数据集中的图像、语言指令和手部姿态。

示例：

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

### Default training entry

默认训练配置已经切到 `legendvla_qwen3_vl`。最直接的启动方式：

```bash
python train.py
```

这会读取：

- `src/config/train_config.yaml`
- `src/config/experiment/legendvla_qwen3_vl.yaml`

如果希望显式指定实验配置：

```bash
python train.py experiment=legendvla_qwen3_vl
```

### Single node with Accelerate

先配置 Accelerate：

```bash
accelerate config
```

然后启动：

```bash
accelerate launch --config_file src/config/acc_config.yaml train.py experiment=legendvla_qwen3_vl
```

也可以直接使用仓库脚本：

```bash
./scripts/pretrain_legendvla_fsdp2.sh
```

> 这个脚本现在实际也会启动 `experiment=legendvla_qwen3_vl`。

### Multi-node with FSDP2

当前多机启动脚本示例：

- `scripts/pretrain_legendvla_fsdp2.sh`

它已经切到新的 `legendvla_qwen3_vl` 配置。运行前请根据实际机器环境修改：

- `NODES`
- `PROJECT_DIR`
- `MASTER_PORT`
- 网卡和 NCCL 相关环境变量

### Resume from checkpoint

如果希望从 checkpoint 恢复训练，可以直接在配置中填写：

- `training.resume_checkpoint_path`
- `training.finetune_checkpoint_path`

当前主配置位置：

- `src/config/experiment/legendvla_qwen3_vl.yaml`

当前 FSDP2 训练输出的 checkpoint 可直接用于恢复。

## Normalizer

WebDataset 训练依赖预计算 normalizer。当前训练 workspace 会从配置中读取：

- `training.normalizer_path`

如果你需要重新计算，可以使用：

```bash
./scripts/compute_norm_stats.sh
```

或者直接调用：

```bash
python -m src.workspace.compute_norm_stats --config src/config/experiment/legendvla_qwen3_vl.yaml
```

## Inference Visualization

可视化推理结果建议使用：

```bash
python rerun_inference_vis.py \
  --inference_zarr_path /path/to/predictions.zarr \
  --origin_zarr_path /path/to/origin.zarr \
  --sample_idx 200 \
  --save_path /path/to/output \
  --target_width 1920 \
  --target_height 1080
```

环境建议使用 `legendvla` 环境，并按需安装 `rerun-sdk` 等依赖。

## Debug

### Nsight

Nsight 可用于分析 GPU kernel、内存访问、通信等瓶颈。

- 单机程序：可以在启动命令前加 `nsys launch`
- 多机程序：可以在训练脚本中使用 `nsys profile`

对于当前主线，建议直接在新的训练命令前套用，例如：

```bash
nsys launch accelerate launch --config_file src/config/acc_config.yaml train.py experiment=legendvla_qwen3_vl
```

### VSCode Debugger

如需在分布式训练中调试：

- 不要在软链接路径下打开 VS Code
- 只在 `Rank 0` 上挂起调试器
- 启动前设置 NCCL 超时，避免其他 rank 提前退出

在启动脚本（例如 `scripts/pretrain_legendvla_fsdp2.sh`）中常用：

```bash
export NCCL_TIMEOUT=3600000
export NCCL_ASYNC_ERROR_HANDLING=1
```

## Project Structure

当前与最新主线最相关的目录如下：

```text
src/
├── config/
│   ├── train_config.yaml
│   └── experiment/
│       └── legendvla_qwen3_vl.yaml
├── dataset/
│   ├── qwen3_vl_batching.py
│   ├── unified_vla_collator.py
│   ├── vla_dataset.py
│   └── vlm_dataset.py
├── model/
│   ├── action_expert/
│   └── vlm/
│       ├── prefix_cache.py
│       └── qwen3_vl_backbone.py
├── policy/
│   ├── legendvla.py
│   ├── legendvla_inference.py
│   └── legendvla_loss.py
└── workspace/
    ├── train_legendvla_workspace.py
    └── train_unified_vla_workspace.py
```

文档与架构图：

- `docs/unified_vla_qwen3vl_plan.md`
- `docs/architecture/unified_vla_montage.png`
- `docs/architecture/unified_vla_overview.png`
- `docs/architecture/unified_vla_model_blocks.png`
- `docs/architecture/unified_vla_training_streams.png`
- `docs/architecture/unified_vla_shared_kv.png`
- `docs/architecture/unified_vla_file_map.png`
- `docs/architecture/rendered/slide-1.png` ~ `docs/architecture/rendered/slide-5.png`
