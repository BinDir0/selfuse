# EgoVLA and Our Better Solution

## Environment Setup

安装环境的完整流程，最好搞成一键式的，方便快捷。

## Data Processing



## Training




## Inference


## Visualization




```bash
# Step 1. git clone VILA & install VILA conda env & pip install -e .
# Step 2. git clone EgoVLA & pip install -r requirements.txt
```

配置多机之间通信，DeepSpeed 要求训练的多机之间要能 ssh 免密通信，因此需要配置好每台机器之间的 ssh 连接。

在每台服务器上安装好 pdsh 启动器，用于 DeepSpeed 多机启动：

```
sudo apt-get install pdsh
```

如果要使用 Deepspeed 多机训练，需要在每台机器上使用 conda 安装完全相同的环境，包括环境名。
最好是环境直接复制。

接着，将代码（包括 EgoVLA 和 VILA）放到 share_data/ 下，保证所有机器之间均能访问。然后在每台机器的相同位置创建一个软链接指向这个位置。

配置 wandb，使用国内的镜像站（可选）：

```bash
export WANDB_BASE_URL=https://api.bandw.top
```

wandb 登陆：

```bash
wandb login
```

### Pretraining

#### Single Node

本项目基于 Accelerate 包装的 DeepSeed 完成多机多卡训练，通过 Accelerate 设置 DeepSpeed Config：

```bash
Accelerate config
```

在询问是否使用 DeepSpeed 时，请勾选是，并选择使用 config 文件设置，预训练使用的 config 文件为：`/egovla/config/pretrain/ds_config.json`。

```bash
./scripts/pretrain.sh
```

#### Multi Node with DeepSpeed

```bash
/scripts/pretrain_deepspeed.sh
```

logging 结果将分别保存在每台机器上，位置可以在 `egovla/config/train_egovla_deepspeed_workspace.yaml` 中调整。

### Post-Traning

```bash
./scripts/post_train.sh
```

### Retargeting Training

```bash
./scripts/train_retargeting.sh
```

### NVILA Model Setup

#### **Step 1 — Download NVILA-Lite-2B**

从 HuggingFace 下载 NVILA-Lite-2B Model 到本地，或者在 load pretrained model 时， 从 HuggingFace 远程加载 Efficient-Large-Model/NVILA-Lite-2B

#### **Step 2 — Download and Set Up VILA**

从官方Repo 下载 VILA 

```bash
git clone https://github.com/NVlabs/VILA.git
./environment_setup.sh vila
```

Please Check the version of triton, which must be the same as the version shown in environment_setup.sh


