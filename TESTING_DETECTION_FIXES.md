# 测试检测改进指南

本指南说明如何测试针对快速手部动作的检测改进。

## 改进内容

本次更新修复了 4 个关键问题：
1. **Fallback ID 碰撞** - 防止错误检测被合并
2. **置信度阈值提高** - 从 0.2 提高到 0.25，减少假阳性
3. **边界框大小验证** - 过滤大小异常的检测
4. **运动速度约束** - 过滤物理上不可能的运动

## 测试方法

### 方法 1：单个视频测试（推荐用于快速验证）

#### 1.1 准备测试视频
选择一个有快速手部动作的视频，最好是之前检测失败的视频。

```bash
# 设置视频路径
VIDEO_PATH="/path/to/your/fast_motion_video.mp4"
OUTPUT_DIR="./test_output/$(basename $VIDEO_PATH .mp4)"
```

#### 1.2 运行完整流程
```bash
# 使用改进后的代码运行完整流程
python demo.py \
    --video $VIDEO_PATH \
    --output_root $OUTPUT_DIR \
    --vis_mode world
```

#### 1.3 检查检测质量
```bash
# 查看检测结果
ls -lh $OUTPUT_DIR/tracks_0_*/

# 应该看到：
# - model_boxes.npy  (检测框)
# - model_tracks.npy (轨迹)
```

#### 1.4 可视化结果
```bash
# 如果有图形界面
python demo.py --video $VIDEO_PATH --output_root $OUTPUT_DIR --vis_mode world

# 如果是无头服务器
python demo_offline.py \
    --video $VIDEO_PATH \
    --seq_folder $OUTPUT_DIR \
    --vis_mode world \
    --output_video $OUTPUT_DIR/result.mp4
```

### 方法 2：对比测试（推荐用于评估改进效果）

#### 2.1 保存旧版本结果
如果你之前已经运行过检测，保存旧结果：
```bash
# 备份旧结果
cp -r $OUTPUT_DIR $OUTPUT_DIR.old
```

#### 2.2 删除检测阶段结果，重新运行
```bash
# 只删除检测结果，保留其他阶段
rm -rf $OUTPUT_DIR/tracks_0_*/model_boxes.npy
rm -rf $OUTPUT_DIR/tracks_0_*/model_tracks.npy

# 重新运行（会自动跳过已完成的其他阶段）
python demo.py --video $VIDEO_PATH --output_root $OUTPUT_DIR
```

#### 2.3 对比分析
```python
# 使用 Python 对比新旧结果
import numpy as np

# 加载旧结果
old_tracks = np.load('$OUTPUT_DIR.old/tracks_0_*/model_tracks.npy', allow_pickle=True).item()
# 加载新结果
new_tracks = np.load('$OUTPUT_DIR/tracks_0_*/model_tracks.npy', allow_pickle=True).item()

print(f"旧版本轨迹数: {len(old_tracks)}")
print(f"新版本轨迹数: {len(new_tracks)}")

# 检查轨迹长度和置信度
for tid, track in new_tracks.items():
    confs = [t['det_box'][0, 4] for t in track if t['det']]
    print(f"Track {tid}: {len(track)} frames, avg conf: {np.mean(confs):.3f}")
```

### 方法 3：批量测试（用于大规模验证）

#### 3.1 准备视频列表
```bash
# 创建包含多个测试视频的列表
cat > test_videos.txt <<EOF
/path/to/video1.mp4
/path/to/video2.mp4
/path/to/video3.mp4
EOF
```

#### 3.2 使用批量推理脚本
```bash
# 使用批量推理（如果已经实现）
python scripts/batch_infer.py \
    --video_list test_videos.txt \
    --output_root ./batch_test_output \
    --gpus 0,1 \
    --stages detect_track,motion,slam,infiller
```

## 评估标准

### 1. 检测质量指标

检查以下方面来评估改进效果：

#### a) 轨迹数量
```bash
# 查看生成的轨迹数
python -c "
import numpy as np
tracks = np.load('$OUTPUT_DIR/tracks_0_*/model_tracks.npy', allow_pickle=True).item()
print(f'总轨迹数: {len(tracks)}')
"
```
- **期望**: 轨迹数应该减少（过滤掉了假阳性）
- **正常范围**: 每只手 1-3 条主要轨迹

#### b) 平均置信度
```python
import numpy as np
tracks = np.load('$OUTPUT_DIR/tracks_0_*/model_tracks.npy', allow_pickle=True).item()

for tid, track in tracks.items():
    confs = [t['det_box'][0, 4] for t in track if t['det']]
    if len(confs) > 0:
        print(f"Track {tid}: avg_conf={np.mean(confs):.3f}, min_conf={np.min(confs):.3f}")
```
- **期望**: 平均置信度应该提高（>= 0.25）
- **好的轨迹**: avg_conf > 0.3

#### c) 轨迹连续性
```python
for tid, track in tracks.items():
    frames = [t['frame'] for t in track]
    gaps = np.diff(sorted(frames))
    num_gaps = np.sum(gaps > 1)
    print(f"Track {tid}: {len(frames)} frames, {num_gaps} gaps")
```
- **期望**: 间隙数量应该减少
- **好的轨迹**: 连续帧占比 > 80%

### 2. 视觉检查

最重要的是视觉检查最终结果：

```bash
# 生成可视化视频
python demo_offline.py \
    --video $VIDEO_PATH \
    --seq_folder $OUTPUT_DIR \
    --vis_mode world \
    --output_video $OUTPUT_DIR/result.mp4

# 查看结果
# 在本地机器上下载并播放 result.mp4
```

**检查要点**：
- ✅ 手部轨迹是否平滑连续
- ✅ 是否还有非手部区域被检测为手
- ✅ 左右手是否被正确区分
- ✅ 快速运动片段是否仍然有合理的轨迹

### 3. 问题诊断

如果结果仍然不理想，可以调整参数：

#### 调整置信度阈值
编辑 `scripts/scripts_test_video/detect_track_video.py`:
```python
# 第 58 行
boxes_, tracks_ = detect_track(imgfiles, thresh=0.3)  # 提高到 0.3
```

#### 调整大小变化阈值
编辑 `lib/eval_utils/custom_utils.py`:
```python
# interpolate_bboxes 函数
def interpolate_bboxes(bboxes, max_size_change_ratio=2.0):  # 降低到 2.0
```

#### 调整速度阈值
编辑 `lib/eval_utils/custom_utils.py`:
```python
# validate_motion_velocity 函数
def validate_motion_velocity(bboxes, max_relative_velocity=2.5):  # 降低到 2.5
```

## 常见问题

### Q1: 检测结果太少，很多有效帧被过滤了
**解决方案**: 降低阈值
- 置信度阈值降回 0.2
- 大小变化阈值提高到 3.0
- 速度阈值提高到 4.0

### Q2: 仍然有很多假阳性检测
**解决方案**: 提高阈值
- 置信度阈值提高到 0.3
- 大小变化阈值降低到 2.0
- 速度阈值降低到 2.5

### Q3: 如何只重新运行检测阶段
```bash
# 删除检测结果
rm -rf $OUTPUT_DIR/tracks_0_*/model_boxes.npy
rm -rf $OUTPUT_DIR/tracks_0_*/model_tracks.npy

# 重新运行
python scripts/scripts_test_video/detect_track_video.py \
    --video $VIDEO_PATH \
    --seq_folder $OUTPUT_DIR
```

### Q4: 如何查看详细的检测信息
```python
import numpy as np

tracks = np.load('$OUTPUT_DIR/tracks_0_*/model_tracks.npy', allow_pickle=True).item()

for tid, track in tracks.items():
    print(f"\n=== Track {tid} ===")
    print(f"Total frames: {len(track)}")

    # 检查 handedness
    handedness = [t['det_handedness'][0] for t in track if t['det']]
    is_right = np.mean(handedness) > 0.5
    print(f"Hand: {'Right' if is_right else 'Left'}")

    # 检查置信度分布
    confs = [t['det_box'][0, 4] for t in track if t['det']]
    print(f"Confidence: mean={np.mean(confs):.3f}, std={np.std(confs):.3f}")

    # 检查帧范围
    frames = [t['frame'] for t in track]
    print(f"Frame range: {min(frames)} - {max(frames)}")
```

## 反馈

如果测试后发现问题，请提供：
1. 视频特征（分辨率、帧率、手部运动速度）
2. 检测结果统计（轨迹数、平均置信度）
3. 具体的失败案例（哪些帧出错）
4. 当前使用的参数设置

这将帮助进一步优化参数和算法。
