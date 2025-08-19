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

### For each dataset

每个数据集名字命名的文件夹中有两个代码：

- mano_trans：将 raw hand pose / 45 + 3d 的 MANO 参数和腕部平移信息一起，转化为 3D 腕部平移，rot6D 腕部旋转和前 15D mano PCA 分量。最好将 mano 信息存放到数据集根目录下的子文件夹中，方便 build_zarr。使用方式：

```bash
python mano_trans.py --data_root PATH/TO/DATASET --output_root PATH/TO/OUTPUT_DIR
```

- build_zarr：读取视频信息、相机外参、语言标注以及转换好的手部、腕部数据，将这些信息封装在一个 zarr 目录中。代码使用方式：

```bash
python build_zarr.py --data_root PATH/TO/DATASET --output PATH/TO/OUTPUT_ZARR_DIR
```

经过封装后，会生成如下格式的 zarr 目录：

```
├── data
│   ├── action       下一帧本体感知
│   │   ├── hand     (sum_frames, 30) float32 先左后右
│   │   └── wrist    (sum_frames, 18) float32 左平移，右平移，左旋转，右旋转
│   ├── extrinsic    (sum_frames, 16) float32
│   ├── image        (sum_frames, 384, 384, 3) uint8 注意图像都被插值成 384x384
│   ├── instruction  (sum_frames,) strin
│   └── state        当前帧本体感知
│       ├── hand     (sum_frames, 30) float32 同 action
│       └── wrist    (sum_frames, 18) float32 同 action
└── meta
    ├── episode_ends (num_episodes,) int64 同 dexgrasp
    └── presence     (num_episodes,) int8  左右手可见情况，1 左 2 右 3 均可见
```

### Dataset visualizer

`data/visualizer.py` 可以对生成好的 zarr 格式的数据集进行可视化。该 visualizer 会将数据集某一帧的语言指令、被插值后的图片以及两只手的 pose 可视化出来。

注意需要用 `--mano_root`, `--manopth_path` 两个参数传递 mano 模型和 manopth 库的路径。

默认会从所有帧中随机一帧进行可视化，也可以用 `--index n` 参数来指定可视化第 n 帧。

有两种模式：不打开 `--camera_view` 参数时，会将插值后的图片显示在图像的左半边，并将两只手的 mesh 显示在图像的右半边，注意 mesh 可视化的视角不是相机视角；打开该参数时，会将手部 mesh 的顶点以及 21 个关节位置重叠在左半边的图像上。注意如果不传递内参信息，则手部位置可视化结果可能不太准确。

外参信息可以存在 .npy 文件中（支持 4×4, 3×3 或直接以 `fx, fy, cx, cy` 形式给出），并调用 `--intrinsic_path PATH/TO/INTR_NPY`；也可以手动调用 `--fx --fy --cx --cy` 四个参数给出。

使用示例：
```bash
python data/visualizer.py --mano_root PATH/TO/MANO_MODEL --manopth_path PATH/TO/MANOPTH_LIB \
  --zarr PATH/TO/DATASET \
  --index N \
  --intrinsic_path PATH/TO/INTR_NPY \
  --camera_view
```

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

`visualize.py`提供模型预测结果的可视化，将未来30帧的预测结果渲染为视频，包含手部骨架连线以及3D mesh重建。

### 使用方法

基本使用：
```bash
python visualize.py --data_path /path/to/predictions.pt --mano_dir /path/to/manopth --show_mesh --show_gt
```

完整参数示例：
```bash
python visualize.py \
  --data_path /path/to/predictions.pt \
  --mano_dir /path/to/manopth \
  --sample_id 0 \
  --find_worst \
  --video_name hand_motion.mp4 \
  --output_dir ./output \
  --fps 30 \
  --show_mesh \
  --mesh_alpha 0.9 \
  --show_gt
```

### 参数说明

- `--data_path`: 包含预测结果、相机参数和背景图像的完整数据文件路径 (默认: `/share_data/yeyuyao/egovla/egovla_predictions_complete.pt`)
- `--mano_dir`: manopth仓库的父目录路径 (默认: `/home/yeyuyao`)
- `--sample_id`: 要可视化的样本索引 (默认: 0)
- `--find_worst`: 自动寻找并使用所有sample中loss最大的样本
- `--video_name`: 输出视频文件名 (默认: `inference_hand_motion.mp4`)
- `--output_dir`: 输出目录 (默认: `./output`)
- `--fps`: 视频帧率 (默认: 30)
- `--show_mesh`: 显示手部3D mesh重建结果
- `--mesh_alpha`: 网格透明度，范围0.0-1.0 (默认: 0.9)
- `--show_gt`: 显示ground truth标注（绿色）与预测结果对比

### 输出格式

生成的视频文件将包含：
- 静态背景图像（来自数据集）
- 30帧手部运动预测序列，包括手部骨架连线和3D mesh重建结果（如果启用`--show_mesh`）
- 可选的真值对比（如果启用`--show_gt`）


