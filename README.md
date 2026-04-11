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

推荐改为“配置文件驱动”调参：

1) 复制并修改 `configs/train_default.json`（按 data / augment / optim / mano / model / io / run 分组）。

2) 直接用配置启动：

```bash
python train.py --config configs/train_default.json
```

3) 需要临时覆盖时，命令行参数优先级更高：

```bash
python train.py --config configs/train_default.json --batch-size 16 --lr 1e-4
```

多卡同理：

```bash
torchrun --standalone --nproc_per_node=8 train.py --config configs/train_default.json
```

说明：
- `--config` 默认就是 `configs/train_default.json`，所以不传也会读取它。
- 若你想完全走纯 CLI（不读配置文件），可传 `--config ""`。

单卡：

```bash
python train.py --data-path /path/to/tar_root --episodes-file splits/my_split/train.txt --run-dir runs/exp --tensorboard-dir runs/exp/tb --mano-no-left-root-fix 
```

多卡（例：8 卡）：


单数据集：
```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --data-path /path/to/tar_root \
  --episodes-file splits/my_split/train.txt \
  --run-dir runs/exp \
  --tensorboard-dir runs/exp/tb \
  --mano-no-left-root-fix
```

多数据集混合：
```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --datasets-config configs/multi_datasets.json \
  --run-dir runs/mix_exp \
  --tensorboard-dir runs/mix_exp/tb
```

多数据集混合训练（同一 batch 可自然混合不同数据集样本）：

先准备 JSON（示例 `configs/multi_datasets.json`）：

```json
{
  "datasets": [
    {
      "name": "taco_sp",
      "data_path": "/path/to/taco_sp",
      "episodes_file": "splits/taco_sp/train.txt"
    },
    {
      "name": "oakink2_v5",
      "data_path": "/path/to/oakink2_v5",
      "episodes_file": "splits/oakink2_v5/train.txt",
      "shard_glob": "*.tar"
    }
  ]
}
```

训练命令：

```bash
python train.py \
  --datasets-config configs/multi_datasets.json \
  --run-dir runs/mix_exp2 \
  --tensorboard-dir runs/mix_exp2/tb \
  --batch-size 2
```

说明：
- 多数据集模式下，`--episodes-file` / `--episode-filter` 需要写在 JSON 的每个 dataset 项里。
- `--data-path` 在多数据集模式下可不传；若需要单独验证集，可额外传 `--data-path` + `--val-episodes-file`。

验证集：`--val-episodes-file splits/my_split/val.txt`。仅验证：`--eval-only --resume .../latest.pt`。

TensorBoard：`tensorboard --logdir runs/exp/tb`

可选：启用 W&B（仅 rank0 记录）：

```bash
python train.py --config configs/train_default.json --wandb --wandb-project egotransformer
```

### 3. 推理（`infer.py`）

默认 `--seq-len` / `--stride` / `--batch-size` 与 `train.py` 一致；`--args-json` 指向该次训练的 `args.json` 以匹配模型结构。

**视频**（长序列按窗滑动、`--stride 0` 表示窗无重叠即 `stride==seq-len`）：

```bash
python infer.py --checkpoint runs/exp/checkpoints/latest.pt \
  --args-json runs/exp/args.json --video /path/to/clip.mp4 --out preds.npz
```

**WebDataset**（与训练相同的 tar / `--episodes-file` 等；重叠窗按帧融合，**每 episode 一个** `npz`，按 `frame_index` 一行）：

```bash
python infer.py --checkpoint runs/exp/checkpoints/latest.pt \
  --args-json runs/exp/args.json --data-path /path/to/tar_root \
  --episodes-file splits/my_split/val.txt --out infer_out/
```

输出目录会生成“每个 episode 一个文件”，文件名示例：`taco_v2__episode_000123.npz`。
主要字段：`frame_index`、`per_frame_window_count`、`hand_existence_prob`、`mano_left_trans/root_orient/hand_pose/betas`、`mano_right_trans/root_orient/hand_pose/betas`。
