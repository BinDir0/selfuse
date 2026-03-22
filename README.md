

## Usage


```bash
conda activate rowah
export CUDA_HOME=/usr/local/cuda-12.8   # 按本机实际修改
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

### 0. extract image
python scripts/extract_frames.py --video_path /path/to/video.mp4

### SLAM 深度：整段逐帧（默认开启）

默认在 **`[start_idx, end_idx]` 片段内每一帧** 上跑 Metric3D / Any4D，写出 `dense_depth_*.npz`，再从中取出 **关键帧深度** 做尺度估计。

- **关闭（仅关键帧，更快）**：`batch_infer.py` / `batch_worker.py` 加 `--no_depth_predict_all_frames`，或 `export HAWOR_DEPTH_PREDICT_ALL_FRAMES=0`

会在 `video/视频名/SLAM/dense_depth_{metric3d|any4d}_{start}_{end}.npz` 写入：

- `frame_indices`：全局帧号（与 `extracted_images` 六位数命名一致）
- `depths_uint16`：`[T,H,W]`，**固定单位为毫米（mm）**，无单独 `depth_scale`；还原米制：`depth_m = depths_uint16.astype(float) * 1e-3`（超过 ~65.5 m 会饱和在 65535）
- `height`、`width`

Any4D **中间缓存**为单个 `SLAM/any4d_depth_{droid|dpvo}_{start}_{end}[_allframes].npz`（`depths` float32、`frame_indices`）；不再按 batch 拆多个 `_b*_*` 文件。重跑：`export HAWOR_ANY4D_FORCE_RERUN=1` 会删除该合并缓存。

SLAM 深度 batch 默认 **`--metric3d_batch_size 48`**（可用 `HAWOR_METRIC3D_BATCH_SIZE` 覆盖）；Any4D 多视图显存占用大，**OOM 时请改小**。

### 1. DPVO + Metric3D


#### 1.1 分步式

```bash

python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,6 \
  --stages detect_track,motion --scheduler_mode wave

python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,6 \
  --stages slam --scheduler_mode wave \
  --slam_backend dpvo

python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,6 \
  --stages infiller --scheduler_mode wave

python demo.py --video_path /path/to/video.mp4 --vis_mode cam --headless
python demo.py --video_path /path/to/video.mp4 --vis_mode world --headless
```

#### 1.2 一步式

```bash
python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,6 \
  --stages detect_track,motion,slam,infiller \
  --scheduler_mode wave \
  --slam_backend dpvo

python demo.py --video_path /path/to/video.mp4 --vis_mode cam --headless
python demo.py --video_path /path/to/video.mp4 --vis_mode world --headless
```

---

### 2. DPVO + Any4D

**依赖：** 同上 DPVO；统一环境内已可 `import` Any4D；**Any4D 深度需要**对应视频的 `extracted_images/`（务必先抽帧）。可选：

```bash
export HAWOR_ANY4D_REPO_ROOT=thirdparty/Any4D
export HAWOR_ANY4D_CHECKPOINT_PATH=checkpoints/any4d_4v_combined.pth
```

**性能：** 多段 keyframe batch 时会在**同一进程内复用**已加载的 Any4D 权重，避免每个 batch 重复初始化。

**默认开启（相对旧版「全 FP32 + 非 SDPA」可能有极小数值差；要与旧结果对齐可关）：**

- **AMP**：默认开 → 关闭：`export HAWOR_ANY4D_USE_AMP=0`
- **PyTorch SDPA**（DINO 注意力）：默认开 → 关闭：`export HAWOR_ANY4D_USE_PYTORCH_SDPA=0`

加载 Any4D 权重时默认**屏蔽**第三方库的冗长 `print`（含整段 kwargs、torch hub、`_IncompatibleKeys` 等）；需要排查问题时：`export HAWOR_ANY4D_VERBOSE=1`。

独立脚本 `scripts/scripts_test_video/run_any4d_depth.py` 可加 `--no_amp` 强制关 AMP。

#### 2.1 分步式

```bash
python scripts/extract_frames.py --video_path /path/to/video.mp4

python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,6 \
  --stages detect_track,motion --scheduler_mode wave

python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,6 \
  --stages slam --scheduler_mode wave \
  --slam_backend dpvo --any4d

python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,6 \
  --stages infiller --scheduler_mode wave

python demo.py --video_path /path/to/video.mp4 --vis_mode cam --headless
python demo.py --video_path /path/to/video.mp4 --vis_mode world --headless
```

（`--any4d` 等价于 `--depth_backend any4d`。）

#### 2.2 一步式

```bash
python scripts/extract_frames.py --video_path /path/to/video.mp4

python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,1,2,3,4,5,6,7 \
  --stages detect_track,motion,slam,infiller \
  --scheduler_mode wave \
  --slam_backend dpvo --any4d

python demo.py --video_path /path/to/video.mp4 --vis_mode cam --headless
python demo.py --video_path /path/to/video.mp4 --vis_mode world --headless
```

---

### 3. DROID-SLAM + Metric3D

**依赖：** 已安装 `thirdparty/DROID-SLAM`（`lietorch`、`droid_backends`）；`--slam_backend droid`。

#### 3.1 分步式

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

#### 3.2 一步式

```bash
python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,6 \
  --stages detect_track,motion,slam,infiller \
  --scheduler_mode wave \
  --slam_backend droid

python demo.py --video_path /path/to/video.mp4 --vis_mode cam --headless
python demo.py --video_path /path/to/video.mp4 --vis_mode world --headless
```

---

### 4. DROID-SLAM + Any4D

**依赖：** DROID 同上；Any4D 同上——**必须先** `extract_frames`，再跑带 `slam` 的阶段。Any4D 的 **AMP / SDPA 默认开启**，与上文「DPVO + Any4D」相同；可用 `HAWOR_ANY4D_USE_AMP=0`、`HAWOR_ANY4D_USE_PYTORCH_SDPA=0` 关闭。

#### 4.1 分步式

```bash
python scripts/extract_frames.py --video_path /path/to/video.mp4

python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,6 \
  --stages detect_track,motion --scheduler_mode wave

python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,6 \
  --stages slam --scheduler_mode wave \
  --slam_backend droid --any4d

python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,6 \
  --stages infiller --scheduler_mode wave

python demo.py --video_path /path/to/video.mp4 --vis_mode cam --headless
python demo.py --video_path /path/to/video.mp4 --vis_mode world --headless
```

#### 4.2 一步式

```bash
python scripts/extract_frames.py --video_path /path/to/video.mp4

python scripts/batch_infer.py \
  --video_list videos_10.txt --gpus 0,6 \
  --stages detect_track,motion,slam,infiller \
  --scheduler_mode wave \
  --slam_backend droid --any4d

python demo.py --video_path /path/to/video.mp4 --vis_mode cam --headless
python demo.py --video_path /path/to/video.mp4 --vis_mode world --headless
```
