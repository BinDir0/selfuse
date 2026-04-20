# LegendVLA — Qwen3-VL + World Model

基于 **Qwen3-VL** backbone 的 VLA (Vision-Language-Action) 训练系统。流匹配动作专家与 DINOv3 未来帧蒸馏（world model）在同一训练 step 内联合优化，采用 **torchrun + FSDP2** 分布式训练框架（HSDP 自动开启：跨节点 replicate、节点内 NVLink shard），Hydra 组装配置，WandB 记录日志，WebDataset 加载数据，FSDP2 原生 DCP 保存分片 checkpoint。

## Features

- **Qwen3-VL 2B** backbone + **MEM 多帧时序注意力**
- **Flow matching action expert**（DiT 风格 AdaLN，`Qwen3Expert` 复用 backbone 结构）
- **DINOv3 World Model**：冻结 ViT-L/16 teacher 监督未来帧特征
- **FSDP2 + HSDP**：跨节点 replicate、节点内 shard，原生 DCP sharded checkpoint
- **`torch.compile` 分组件开关**：vision / text / flow / diffloss / world_model 独立控制
- **`flex_attention` + FlashAttention**：text 走 flex_attention（torch.compile 友好），vision 走 flash-attn
- **Hydra 配置体系**：`src/config/` 下 yaml 组合，便于复现与 diff
- **RTC 推理**（Recurrent Temporal Conditioning）与 WebSocket serving
- **多机 launcher 开箱即用**：火山云 MLP / 腾讯云裸金属（pdsh）两套方案

## Quick Start

从 clone 到开始训练的最短路径（假设有 Ubuntu + CUDA 12.8 环境）：

```bash
# 1. 克隆仓库
git clone https://github.com/Psi-Robot/EgoVLA && cd EgoVLA

# 2. 安装训练环境（含 PyTorch 2.10.0+cu128、FlashAttention、NCCL、pdsh、numactl）
bash scripts/install.sh
conda activate legendvla

# 3. 配置数据与预训练权重路径
#    - VLA/VLM shards:   src/config/dataset_paths/{vla_wds,vlm_wds}.yaml
#    - Qwen3-VL 权重:     src/config/model/qwen3_vl_2b.yaml
#    - DINOv3 teacher:    src/config/world_model/frozen_regression.yaml

# 4. 计算 normalizer（训练前必做一次）
bash scripts/compute_norm_stats.sh
#    完成后把输出路径填进 src/config/training/default.yaml 的 training.normalizer_path

# 5. 启动训练（单节点 8 卡）
bash scripts/pretrain_legendvla_fsdp2_single_node.sh
```

> ⚠️ **版本严格要求**：务必按 `requirements.txt` 安装（PyTorch 2.10.0+cu128 + pinned `transformers` commit）；版本错会在训练与推理中都出现难诊断的错误。详见 [Installation](#installation)。

> ⚠️ **启动与配置约定**：不要直接 `python train.py`，请走 `scripts/pretrain_legendvla_fsdp2*.sh`；不要 CLI Hydra override，直接改 `src/config/` 下的 yaml。详见 [Training](#training)。

多机 / 腾讯云裸金属 / 火山云 MLP 启动方式见 [Training](#training)；本地推理环境精简版、单任务 finetune、checkpoint resume 等见对应章节。

## Architecture

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

## Installation

> ⚠️ **版本严格要求**：请务必严格按照 `requirements.txt` 安装对应的 PyTorch 与 `transformers` 版本。当前仓库与 **PyTorch 2.10.0+cu128** 及 pinned `transformers` commit 紧耦合（同时修复 Qwen3-VL batched video RoPE 与 `flex_attention` deprecated API 两个 issue）。**版本装错会在训练与推理中都出现难以诊断的错误**（attention API 崩溃、RoPE 形状错配、`torch.compile` 后端报错等）。不要擅自升级或降级。

### 安装场景

`scripts/install.sh` 面向**完整训练环境**，包含系统级依赖。**本地推理只需精简子集**。下表说明每一步的用途：

| 安装步骤 | 训练 | 本地推理 | 用途 |
|---|:-:|:-:|---|
| `apt-get` 系统包（build-essential、libjpeg/png、ffmpeg 等） | ✅ | 可选 | 编译工具链与视频/图像解码 |
| `libnccl2` / `libnccl-dev` | ✅ | ❌ | 分布式通信，单机推理不需要 |
| `pdsh` | ✅（多机） | ❌ | 仅裸金属多机 launcher 需要 |
| `numactl` | ✅（多机） | ❌ | per-rank NUMA 绑定（见 `scripts/numa_bind_wrapper.sh`） |
| conda 环境 `legendvla`（Python 3.10） | ✅ | ✅ | 基础 Python 环境 |
| **PyTorch 2.10.0 + CUDA 12.8**（严格版本） | ✅ | ✅ | 必须与 `requirements.txt` 完全一致 |
| FlashAttention（编译安装） | ✅ | ✅ | 训练与推理均需要，保持 attention backend 与训练一致 |
| `requirements.txt`（含 pinned `transformers` commit） | ✅ | ✅ | 必装；pin 的 commit 承载关键 Qwen3-VL / flex_attention 修复 |

### 训练环境（完整）

见 [Quick Start](#quick-start) 第 2 步，一行 `bash scripts/install.sh` 即可。该脚本会依次完成上表所有 ✅（训练）列。

**安装位置建议（按平台）**：

- **腾讯云（裸金属）**：容器系统盘通常不跨节点共享，为了让所有节点共用同一份环境，建议把 conda 环境装到 `/share_data` 共享盘（例如 `/share_data/conda_envs/legendvla`），然后在每台节点上用软链接把本地 `~/miniconda3/envs/legendvla` 指向共享盘对应目录（或直接把整个 conda 根目录链到共享盘）。这样一次安装、全部节点可用，`pdsh` launcher 也能解析到一致的 Python 路径。
- **火山云（Volcengine MLP）**：直接在系统盘正常安装（`bash scripts/install.sh`），装好后**保存为自定义镜像**。后续起任务时直接选用该镜像，无需重装；MLP 脚本也默认容器镜像已预装全部依赖，不再执行 conda 激活。

### 本地推理环境（精简）

不需要 NCCL / pdsh / numactl 等分布式系统级组件，但仍需安装 FlashAttention 以保持 attention backend 与训练一致：

```bash
# 1. 建 conda 环境
conda create -y -n legendvla python=3.10
conda activate legendvla

# 2. 严格安装 PyTorch 2.10.0 + cu128（版本不可改）
pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# 3. 编译安装 FlashAttention（需在 PyTorch 之后，以便匹配本地 torch / CUDA 工具链）
pip install packaging ninja psutil
pip install flash-attn --no-build-isolation

# 4. 其余依赖（含 pinned transformers commit）
pip install -r requirements.txt
```

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

## Data Preparation

### 数据格式

训练使用 **[WebDataset](https://github.com/webdataset/webdataset)** (`.tar` shards) 格式。数据集路径在以下配置中定义：

- VLA 数据集：`src/config/dataset_paths/vla_wds.yaml`
- VLM 数据集：`src/config/dataset_paths/vlm_wds.yaml`

训练时 VLA 和 VLM 数据按 `vla_ratio`（默认 `5/6`）混合采样。

**图像尺寸**：`data.target_image_size=[384, 384]`。启用 world model 时该值必须设置，保证 VLA 输入、MEM 时序注意力、以及 DINOv3 teacher 的 patch 对齐；`384/(16*upsample_factor=2)=12 → 12×12=144 query tokens/frame`，上采样后对齐 DINOv3 的 576 patch。

### WebDataset 约定

团队内部对 WebDataset 样本结构、字段命名、相机命名（`head` / `breast`）、动作/状态向量布局等有统一约定。新数据集接入前请先对齐此文档：

📖 [WebDataset 数据格式约定（飞书 Wiki）](https://psi-robot.feishu.cn/wiki/Tg2Vwtm3visp3akregXcTphZnQb)

### VLM 数据转换

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

### 启动方式（重要）

> ⚠️ **不建议直接 `python train.py` 启动**。请务必使用 `scripts/pretrain_legendvla_fsdp2*.sh` 系列脚本，内部走 `torchrun`，已配置好 NCCL / RDMA / NUMA 绑定 / 环境变量等，且与 FSDP2 初始化契约严格对齐。单机单卡调试请用 `scripts/debug_start.sh`。

> ⚠️ **不建议在命令行通过 Hydra override 修改配置**。为了便于观察、版本控制和复现，请直接修改 `src/config/` 下对应的 yaml（例如：关 world model 改 `src/config/experiment/legendvla_qwen3_vl.yaml` 里 `defaults` 中的 `- world_model: frozen_regression` 为 `none`；改 batch size 改 `src/config/data/unified_wds.yaml` 的 `dataloader.loader.batch_size`；改 warmup 改 `src/config/training/default.yaml` 的 `lr_warmup_steps`）。命令行 override 难以追踪与 diff，容易出实验记账问题。

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

### 裸金属多机训练（腾讯云，pdsh 方式）

适用于自行管理的多机集群，通过 pdsh + SSH 从 master 节点一键启动所有 worker。

前提条件：

- 多台服务器之间 SSH 免密通信（见下方 _Prepare_ 小节）
- 共享盘上的统一代码、数据、环境路径
- 安装 pdsh（`apt-get install pdsh`）与 numactl

#### Prepare：SSH 免密与 config 分发

pdsh launcher 要求**所有节点两两 SSH 免密**，且各节点使用统一的 `~/.ssh/config`（便于以 Host 别名引用节点）。项目提供 `scripts/sync_ssh_config.sh` 一键完成。

只需在**一台机器（通常是 master）**上完成以下准备，然后本机执行脚本：

1. 在本机 `~/.ssh/config` 写好所有节点的 Host 条目。Host 名既用于本脚本遍历，也可以直接用作后续 `pretrain_legendvla_fsdp2.sh` 中 `NODES` 数组的地址：

   ```text
   Host node0
       HostName 172.18.0.101
       User root
       Port 22

   Host node1
       HostName 172.18.0.104
       User root
   # ... 其他节点
   ```

2. 确保本机已有 SSH 密钥（脚本在缺失时会自动 `ssh-keygen -t rsa -f ~/.ssh/id_rsa`）。

3. 首次若全无免密，需安装 `sshpass`（脚本会提示输入一次统一密码）：

   ```bash
   sudo apt-get install -y sshpass pdsh numactl
   bash scripts/sync_ssh_config.sh
   ```

脚本会依次完成：

- 把本机 `~/.ssh/config` 同步到所有节点
- **本机 → 所有节点** 的免密（`ssh-copy-id`）
- **所有节点两两之间** 的免密（互为源与目标，确保 pdsh fan-out 与 NCCL 握手都走免密）
- 为本机与所有节点填充 `known_hosts`，避免 fingerprint 提示阻塞自动化

完成后在任意节点执行 `ssh <其他节点别名>` 都应直接登入且无交互。后续若有新节点加入，把它写入 `~/.ssh/config` 再跑一次脚本即可。

#### 启动训练

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

训练输出的是 FSDP2 原生 [DCP sharded checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)（目录形式，内含 `.metadata` 与 `__*_*.distcp`）。

在 `src/config/training/default.yaml` 中配置：

```yaml
training:
  # 恢复训练（加载 optimizer state + 调度器 + 训练状态）
  resume: true
  resume_checkpoint_path: /path/to/checkpoint

  # 或：Finetune（只加载模型权重，optimizer / scheduler 全新开始）
  # finetune_checkpoint_path: /path/to/checkpoint
```

然后通过 `scripts/pretrain_legendvla_fsdp2*.sh` 正常启动即可。

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

## Inference & Evaluation

### 推理配置

推理入口配置见 `src/config/experiment/inference.yaml`，通过 `src.policy.legendvla_inference_wrapper.LegendVLAInference` 包装：给定训练 run 的 `.hydra/config.yaml` 与 DCP checkpoint 即可重建整网，并可覆盖 `pretrained_vlm_path` / `teacher_path` 走本地镜像。

### WebSocket Serving

```bash
bash scripts/run_server.sh
```

相关代码位于 `src/serving/`，包含 WebSocket server / client 和 msgpack 序列化。

### 推理可视化

使用 [Rerun SDK](https://github.com/rerun-io/rerun) 可视化推理结果。具体命令行参数请直接查看脚本帮助：

```bash
python rerun_inference_vis.py --help
```

需要先 `pip install rerun-sdk`。

### 离线评估

```bash
bash scripts/run_eval.sh
```

离线评估入口 `evaluate.py`，配置在 `src/config/eval_config.yaml`。

## Debug & Profiling

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

### torch.profiler

项目内置 [`torch.profiler`](https://docs.pytorch.org/docs/stable/profiler.html) 支持（实现见 `src/utils/profiler_utils.py`）。在 `src/config/training/default.yaml` 中启用：

```yaml
training:
  profile: true            # 开启 profiler，trace 落在 ${output_dir}/trace/
```

生成的 `.pt.trace.json` / `.json.gz` 可以直接在 [Perfetto UI](https://ui.perfetto.dev/) 或 Chrome `chrome://tracing` 中打开查看时间线；若要做分布式训练的深度瓶颈分析，推荐 Meta 开源的 HTA。详细学习链接见 [References](#references)。

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
├── pretrain_legendvla_fsdp2_single_node.sh      # Volcengine single-node launcher (torchrun)
├── pretrain_legendvla_fsdp2.sh                   # pdsh multi-node launcher (torchrun)
├── numa_bind_wrapper.sh                          # per-rank NUMA binding wrapper
├── debug_start.sh                                # debug launcher with debugpy support
├── compute_norm_stats.sh                         # normalizer computation
├── finetune_single_task.py                       # per-dataset finetune scheduler
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
├── convert_hf_to_wds.py
└── visualizer.py                        # dataset visualizer
```

## References

核心依赖与扩展阅读：

| 主题 | 链接 |
|---|---|
| **PyTorch 2.10** Release Notes | [pytorch/pytorch v2.10.0](https://github.com/pytorch/pytorch/releases/tag/v2.10.0) |
| **FSDP2** `fully_shard` API | [torch.distributed.fsdp.fully_shard](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) |
| **`torch.compile`** 教程 | [Intro to torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) |
| **`flex_attention`** 介绍 | [PyTorch Blog: FlexAttention](https://pytorch.org/blog/flexattention/) |
| **FlashAttention** | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) |
| **Hydra** | [Hydra Docs](https://hydra.cc/docs/intro/) · [defaults list](https://hydra.cc/docs/advanced/defaults_list/) |
| **torchrun** Elastic Launch | [torch.distributed.elastic](https://docs.pytorch.org/docs/stable/elastic/run.html) |
| **DCP sharded checkpoint** | [torch.distributed.checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) |
| **`torch.profiler`** | [API](https://docs.pytorch.org/docs/stable/profiler.html) · [Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) · [TensorBoard](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) |
| **Trace 查看与分析** | [Perfetto UI](https://ui.perfetto.dev/) · [HolisticTraceAnalysis](https://github.com/facebookresearch/HolisticTraceAnalysis) · [Kineto](https://github.com/pytorch/kineto) |
| **WebDataset** | [webdataset/webdataset](https://github.com/webdataset/webdataset) |
| **WandB** | [Weights & Biases Docs](https://docs.wandb.ai/) |
| **Qwen3-VL** | [Qwen/Qwen3-VL (HF)](https://huggingface.co/collections/Qwen/qwen3-vl) |
| **DINOv3** | [facebook/DINOv3 (HF)](https://huggingface.co/collections/facebook/dinov3) |
| **Rerun SDK** | [rerun-io/rerun](https://github.com/rerun-io/rerun) |
