# EgoTransformer

## 环境

```bash
conda create -y -n ego python=3.10 && conda activate ego
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

- **MANO**：安装 **manopth**；环境变量 **`MANO_ROOT`** 指向含 `MANO_LEFT.pkl` / `MANO_RIGHT.pkl` 的目录（如 `.../mano/models`）。也可用仓库下 `data/mano`（若已放置模型文件）。

## Usage

### 1. 样本 / 划分

- **数据**：WebDataset **tar** 目录或 glob，训练时用 `--data-path` 指向该路径。
- **episode 列表**：从 tar 扫出 episode 名并写 `train.txt` / 测试集等：

```bash
python scripts/make_episode_split.py --data-path /path/to/tar_root --out-dir splits/my_split
```

（可选：`--holdout`、`--train-max`、`--shard-glob` 等见脚本内说明。）

### 2. 训练

单卡：

```bash
python train.py --data-path /path/to/tar_root --episodes-file splits/my_split/train.txt --run-dir runs/exp --tensorboard-dir runs/exp/tb --mano-left-root-fix 
```

多卡（例：8 卡）：

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --data-path /path/to/tar_root \
  --episodes-file splits/my_split/train.txt \
  --run-dir runs/exp \
  --tensorboard-dir runs/exp/tb
  --mano-left-root-fix
```

验证集：`--val-episodes-file splits/my_split/val.txt`。仅验证：`--eval-only --resume .../latest.pt`。

TensorBoard：`tensorboard --logdir runs/exp/tb`
