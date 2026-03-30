<<<<<<< HEAD
## environment set up


### 1. 创建环境

```bash
conda create -n rowah python=3.10 -y
conda activate rowah
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

### 2. HaWoR / RoWaH Python 依赖

在**仓库根目录**：

```bash
cd /path/to/RoWaH
pip install -r requirements.txt
pip install pytorch-lightning==2.2.4 --no-deps
pip install lightning-utilities torchmetrics==1.4.0
```


### 3. 额外的库

```bash
pip install --no-build-isolation mmcv
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@stable
pip install --no-build-isolation git+https://github.com/mattloper/chumpy
```


### 4. 装模块（DPVO / Any4D / DROID-SLAM / Metric3D）

在仓库根目录执行；**权重另见 §5**，与下述安装配套。

**DPVO**（`--slam_backend dpvo`）

```bash
cd thirdparty/DPVO && pip install . --no-build-isolation
```

安装后拉取 **`thirdparty/DPVO/models/dpvo.pth`**：§5.1（`bash download_models_and_data.sh`）。

**Any4D**（`--any4d` / Any4D 深度）

```bash
pip install -e thirdparty/Any4D --no-deps
```

权重 **`any4d_4v_combined.pth`**：§5.2（仓库根 `checkpoints/` 或 `thirdparty/Any4D/checkpoints/`，与 `HAWOR_ANY4D_CHECKPOINT_PATH` 一致）。

**DROID-SLAM**（可选，`--slam_backend droid`）

```bash
cd thirdparty/DROID-SLAM && python setup.py install
```

权重 **`weights/external/droid.pth`**：§5.3。



### 5. 权重与 MANO


#### 5.1 Install DPVO weights:

With `--slam_backend dpvo`, RoWaH loads **`thirdparty/DPVO/models/dpvo.pth`** (see `lib/pipeline/dpvo_slam.py`).

After installing the DPVO package (§4), run the upstream script in the DPVO tree to fetch `models.zip` and extract `models/`:

```bash
cd thirdparty/DPVO
bash download_models_and_data.sh
```

Ensure **`thirdparty/DPVO/models/dpvo.pth`** exists (same layout as [princeton-vl/DPVO](https://github.com/princeton-vl/DPVO)).

#### 5.2 Install Any4D weights:

Two layouts work; pick one and match your env (see `lib/pipeline/any4d_depth.py` `resolve_any4d_paths`):

- **Usage (recommended):** set `HAWOR_ANY4D_CHECKPOINT_PATH=checkpoints/any4d_4v_combined.pth` (relative to repo root). Download into repo-root `checkpoints/` — same path as the `export` lines in §1 below.
- **No env override:** if `HAWOR_ANY4D_CHECKPOINT_PATH` is unset, the default is **`thirdparty/Any4D/checkpoints/any4d_4v_combined.pth`**.

Hugging Face (same file as [Any4D](https://github.com/Any-4D/Any4D)):

```bash
# If you use the Usage exports (repo-root checkpoints):
mkdir -p checkpoints
wget -O checkpoints/any4d_4v_combined.pth \
  https://huggingface.co/airlabshare/any4d-checkpoint/resolve/main/any4d_4v_combined.pth
```

```bash
# If you rely on code defaults (no CHECKPOINT_PATH export):
mkdir -p thirdparty/Any4D/checkpoints
wget -O thirdparty/Any4D/checkpoints/any4d_4v_combined.pth \
  https://huggingface.co/airlabshare/any4d-checkpoint/resolve/main/any4d_4v_combined.pth
```


#### 5.3 Install masked DROID-SLAM weights (optional):


Download DROID-SLAM official weights [droid.pth](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing), put it under `./weights/external/`.

#### 5.4 Install Metric3D weights:


Download Metric3D official weights [metric_depth_vit_large_800k.pth](https://drive.google.com/file/d/1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr/view?usp=drive_link), put it under `thirdparty/Metric3D/weights`.

#### 5.5 Download the model weights


```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./weights/external/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt -P ./weights/hawor/checkpoints/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt -P ./weights/hawor/checkpoints/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml -P ./weights/hawor/
```
It is also required to download MANO model from [MANO website](https://mano.is.tue.mpg.de). 
Create an account by clicking Sign Up and download the models (mano_v*_*.zip). Unzip and put the hand model to the `_DATA/data/mano/MANO_RIGHT.pkl` and `_DATA/data_left/mano_left/MANO_LEFT.pkl`. 

Note that MANO model falls under the [MANO license](https://mano.is.tue.mpg.de/license.html).
<<<<<<< HEAD



---

## Usage


```bash
conda activate rowah
export CUDA_HOME=/usr/local/cuda-12.8   # 按本机实际修改
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

### 0. extract image
python scripts/extract_frames.py --video_path /path/to/video.mp4



### 1. DPVO + Any4D


```bash
export HAWOR_ANY4D_REPO_ROOT=thirdparty/Any4D
export HAWOR_ANY4D_CHECKPOINT_PATH=checkpoints/any4d_4v_combined.pth
```

#### 1.1 分步式

```bash

python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,6 \
  --stages detect_track,motion --scheduler_mode wave

python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,1,2,3,4,5,6,7\
  --stages slam --scheduler_mode wave \
  --slam_backend dpvo --any4d

python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,6 \
  --stages infiller --scheduler_mode wave

python demo.py --video_path /path/to/video.mp4 --vis_mode cam --headless
python demo.py --video_path /path/to/video.mp4 --vis_mode world --headless
```


#### 1.2 一步式

```bash

python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,1,2,3,4,5,6,7 \
  --stages detect_track,motion,slam,infiller \
  --scheduler_mode wave \
  --slam_backend dpvo --any4d

python demo.py --video_path /path/to/video.mp4 --vis_mode cam --headless
python demo.py --video_path /path/to/video.mp4 --vis_mode world --headless
```

---

### 2. DROID-SLAM + Metric3D


#### 2.1 分步式

```bash
python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,6 \
  --stages detect_track,motion --scheduler_mode wave

python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,6 \
  --stages slam --scheduler_mode wave \
  --slam_backend droid

python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,6 \
  --stages infiller --scheduler_mode wave

python demo.py --video_path /path/to/video.mp4 --vis_mode cam --headless
python demo.py --video_path /path/to/video.mp4 --vis_mode world --headless
```

#### 2.2 一步式

```bash
python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,6 \
  --stages detect_track,motion,slam,infiller \
  --scheduler_mode wave \
  --slam_backend droid

python demo.py --video_path /path/to/video.mp4 --vis_mode cam --headless
python demo.py --video_path /path/to/video.mp4 --vis_mode world --headless
```


## analyze
### type 1:
python scripts/zarr_analysis/analyze_episodes_full.py \
  --zarr-paths /path/to/store /path/to/dataset.tar \
  --dataset-names A B

### type 2:

python scripts/zarr_analysis/analyze_episodes_full.py \
  --data-format webdataset --zarr-paths 'https://.../shards-{0000..0009}.tar' --dataset-names D1

