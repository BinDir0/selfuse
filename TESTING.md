# HaWoR Batch Inference Testing Guide

这些脚本可以在有数据和GPU的机器上直接运行，用于验证批量推理系统。

## 快速开始

### 1. 环境验证
首先运行环境验证脚本，确保所有依赖都已安装：

```bash
bash scripts/validate_setup.sh
```

这会检查：
- Python 环境和必需包
- CUDA 和 GPU 可用性
- 模型权重文件
- 批处理脚本完整性

### 2. 烟雾测试（Smoke Test）
使用 2-3 个短视频和 2 张 GPU 进行基础功能测试：

```bash
# 准备测试视频目录
mkdir -p test_videos
# 将 2-3 个短视频复制到 test_videos/

# 运行烟雾测试
bash scripts/test_smoke.sh test_videos 0,1
```

测试内容：
- ✓ 完整流程执行（detect_track → motion → slam → infiller）
- ✓ 断点续跑（二次运行应跳过已完成的 stage）
- ✓ 输出产物验证（检查 world_space_res.pth 等文件）
- ✓ 部分重跑（删除某个产物后重新生成）

### 3. 生产环境测试（Production Test）
使用 20+ 视频和 8 张 GPU 进行压力测试：

```bash
# 创建视频列表文件
find /path/to/videos -name "*.mp4" > videos.txt

# 运行生产测试（会实时监控进度）
bash scripts/test_production.sh videos.txt 0,1,2,3,4,5,6,7
```

测试内容：
- 8 GPU 并行处理
- 实时进度监控
- 完整统计报告
- 失败率和成功率分析

## 手动运行批量推理

### 基本用法

```bash
# 从目录处理视频
python scripts/batch_infer.py \
  --video_dir /path/to/videos \
  --gpus 0,1,2,3,4,5,6,7

# 从列表文件处理视频
python scripts/batch_infer.py \
  --video_list videos.txt \
  --gpus 0,1,2,3,4,5,6,7
```

### 高级选项

```bash
# 自定义重试次数和 stage
python scripts/batch_infer.py \
  --video_list videos.txt \
  --gpus 0,1,2,3 \
  --retries 3 \
  --stages detect_track,motion,slam,infiller

# 强制重跑（忽略已有输出）
python scripts/batch_infer.py \
  --video_list videos.txt \
  --gpus 0,1,2,3,4,5,6,7 \
  --no-resume

# 指定运行目录（用于恢复中断的批次）
python scripts/batch_infer.py \
  --video_list videos.txt \
  --gpus 0,1,2,3,4,5,6,7 \
  --run_dir batch_runs/20260301_120000
```

## 结果分析

运行完成后，使用分析脚本查看详细报告：

```bash
python scripts/analyze_run.py batch_runs/20260301_120000
```

输出包括：
- 总体统计（完成/失败/部分完成）
- 各 stage 完成率
- 失败视频详情
- 事件统计
- Stage 平均耗时
- 视频平均处理时间
- 恢复建议

## 输出结构

```
batch_runs/<timestamp>/
├── status.json              # 所有视频的当前状态
├── events.jsonl             # 事件流日志（每行一个 JSON 事件）
└── logs/
    ├── video1_detect_track.log
    ├── video1_motion.log
    ├── video1_slam.log
    ├── video1_infiller.log
    └── ...
```

## 常见问题

### 中断后如何恢复？
直接使用相同的 `--run_dir` 重新运行即可：
```bash
python scripts/batch_infer.py \
  --video_list videos.txt \
  --gpus 0,1,2,3,4,5,6,7 \
  --run_dir batch_runs/20260301_120000
```

### 如何只处理失败的视频？
1. 使用 `analyze_run.py` 找出失败的视频
2. 创建新的视频列表文件，只包含失败的视频
3. 重新运行批量推理

### 如何调整 GPU 使用？
通过 `--gpus` 参数指定：
- 使用 4 张卡：`--gpus 0,1,2,3`
- 使用特定卡：`--gpus 2,3,6,7`
- 单卡测试：`--gpus 0`

### 如何查看实时进度？
1. 查看 `status.json`：`watch -n 5 cat batch_runs/<run_id>/status.json`
2. 查看最新事件：`tail -f batch_runs/<run_id>/events.jsonl`
3. 使用 `test_production.sh`（自带进度监控）

## 性能优化建议

### 当前版本（Phase 1）
- 每 GPU 处理 1 个视频
- 视频内部 stage 串行
- 每个 stage 会重新加载模型

### 预期吞吐
- 8×A100：可同时处理 8 个视频
- 单视频耗时：取决于视频长度和分辨率
- 批次总耗时 ≈ max(单视频耗时) × ceil(总视频数 / 8)

### 后续优化（Phase 2）
如果需要进一步提升吞吐，可以实现：
- 常驻 worker 进程（避免重复加载模型）
- 模型缓存复用
- 动态任务调度（减少尾部空转）

## 脚本说明

- `validate_setup.sh`：环境验证脚本
- `test_smoke.sh`：烟雾测试（2-3 视频，2 GPU）
- `test_production.sh`：生产测试（20+ 视频，8 GPU，带监控）
- `analyze_run.py`：结果分析工具
- `batch_infer.py`：批量推理主程序
- `batch_worker.py`：单 stage 执行器（被 batch_infer.py 调用）
