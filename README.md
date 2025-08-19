# EgoVLA and Our Better Solution

## Environment Setup

在需要多机并行的每台机器上，下载仓库并安装 conda 环境（或者放置到一个共享文件夹，可以共同访问）：

```bash
git clone https://github.com/Psi-Robot/EgoVLA
cd EgoVLA
./scripts/install.sh
```

前往 MANO [官网](https://mano.is.tue.mpg.de/)，下载模型 `mano_v*_*.zip`，然后解压按照如下方式放置：

```bash
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

配置多机之间通信，DeepSpeed 要求训练的多机之间要能 ssh 免密通信，因此需要配置好每台机器之间的 ssh 连接。

配置 wandb，使用国内的镜像站（可选）：

```bash
export WANDB_BASE_URL=https://api.bandw.top
```

wandb 登陆：

```bash
wandb login
```

## Data Processing



## Training

### Single Node

配置 Accelerate 默认 config：

```bash
Accelerate config
```

```bash
./scripts/pretrain.sh
```

### Multi Node with DeepSpeed

注意，在使用共享数据盘时，可能出现多机竞争读写的问题，因此你需要把 `egovla/config/experiment/pretrain_deepspeed.yaml` 下的 `multi_run/run_dir` 和 `hydra/run/dir` 修改成一个在每台机器上分别保存的路径。

在训练之前，请你根据 comments 修改 `egovla/config/acc_node0.yaml` 下有关多机信息的内容。

```bash
/scripts/pretrain_deepspeed.sh
```

## Inference


## Visualization

