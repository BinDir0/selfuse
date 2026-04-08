# 从 HaWoR 推理到可规模化数据生产：RoWaH Pipeline 技术博客草稿

![RoWaH teaser](/root/.openclaw/workspace/projects/hawor_original/HaWoR/assets/teaser.png)

## 一句话总结

我们把原本偏单视频、偏研究脚本风格的 HaWoR 推理流程，整理成了一条面向大规模数据生产的 pipeline：输入是多种来源的视频或分片数据，输出是带有动作标注、相机内外参、深度几何与 MANO 参数的统一 WebDataset。  

这条 pipeline 的重点不是“又做了一遍 HaWoR”，而是把 **动作语义、几何信息和可重建的手部参数** 一起打包成一个稳定、可过滤、可复用、可重建的数据边界。

---

## 为什么要做这条 Pipeline

原始 HaWoR 更像是一个强大的单条视频手部重建与补全系统，但当目标变成大规模数据生产时，会很快遇到几个工程问题：

1. 输入来源不统一，难以直接复用同一套下游逻辑
2. 动作语义、几何、图像往往分散在不同脚本和中间目录中
3. 结果缺乏稳定的数据边界，难以复跑、过滤、重建
4. 大规模跑批时，速度、容错、断点续跑、过滤一致性都变成硬要求

RoWaH pipeline 的目标，就是把这些问题统一解决掉。

---

## Pipeline 最终产出了什么

这条 pipeline 的最终输出不是一堆零散缓存，而是统一的 WebDataset 样本。每个样本至少包含：

- `*.image.jpg`：RGB 图像
- `*.lowdim.npy`：116 维低维动作与几何表示
- `*.mano.npy`：可重建的双手 MANO 参数
- `*.meta.json`：动作文本、语言、presence 等元信息

其中 `lowdim.npy` 里不仅有双手腕部和平移/旋转状态，还显式包含：

- 相机外参 `w2c`，即 world-to-camera 变换
- 相机内参 `[fx, fy, cx, cy]`
- 与下一帧相关的 action/state 信息

这意味着最终数据集不只是“带手部关键点的视频帧”，而是一种 **图像、动作、相机、三维几何统一对齐** 的训练样本。

---

## 1. 动作标注：从旁路信息变成正式数据字段

这条 pipeline 里，动作标注不再是可有可无的补充文件，而是正式的数据协议。

### 标注协议

每个 clip 对应一个 sidecar annotation 文件：

```text
<annotation_root>/<clip_id>.annotation.json
```

最核心的字段有：

- `status`
- `language`
- `instruction`

当 `instruction` 缺失时，系统还可以从层级语义字段中回退生成 instruction。最终进入 WebDataset 时，动作标注会被写入 `meta.json`，包括：

- `instruction`
- `instruction_num`
- `language`

### 这样做的价值

这一步把“动作语义”从外部依赖变成了数据集的一部分，带来三个直接收益：

1. 训练时不需要再去额外查表拼文本
2. 过滤时可以直接把标注完整性纳入条件
3. 同一个 clip 的图像、动作几何、语言语义拥有同一个稳定主键

如果从 VLA 或机器人学习的角度看，这一点非常重要：模型看到的不是孤立的手，而是“手在做什么”。

---

## 2. 内外参：不是后处理附属品，而是样本的一部分

这条 pipeline 的另一个核心点，是把相机几何信息正式并入样本，而不是只在可视化时临时使用。

### 外参

最终样本中保存的是 `world-to-camera (w2c)` 外参矩阵。  
这意味着任何世界坐标下的手部状态，都可以稳定投影回当前图像坐标系。

### 内参

样本同时保存 pinhole 相机内参：

```text
[fx, fy, cx, cy]
```

### 为什么这很关键

有了显式内外参，数据集就不再只是“2D 图像 + 3D 手”。它变成了一种可以同时支持：

- world-space 监督
- image-space 投影检查
- 几何一致性过滤
- 离线重建与可视化复现

的统一表示。

这也是为什么我们能做更强的对齐调试和更可信的质量过滤：  
不是只检查手是不是“看起来对”，而是检查它在相机坐标系里是否也合理。

---

## 3. 深度：把 Any4D/SLAM 几何引入数据生产链

深度不是单独附加的一份结果，而是贯穿在 SLAM 与尺度估计中的关键几何信号。

### 深度来源

当前 pipeline 中，SLAM 阶段可以结合：

- DPVO 轨迹
- Any4D 深度预测

系统会在真实视频帧上预测深度，并与 DPVO 轨迹、mask、尺度估计一起工作，最终得到更稳定的相机与世界坐标关系。

### 为什么深度重要

如果没有深度，很多 3D 结果只能停留在“相对形状差不多”；  
有了深度和相机几何之后，才能更稳定地建立：

- 手在世界系中的位置
- 手与相机之间的真实前后关系
- frame-to-frame 的动作增量
- 更可靠的 camera-space 过滤条件

从数据质量的角度，这一步是从“视觉重建结果”走向“几何一致训练样本”的关键。

---

## 4. 为什么它比原始 HaWoR 更适合做数据集

如果只看单视频推理，HaWoR 已经很强；  
但如果看的是“大规模、可复跑、可过滤、可训练”的数据生产，这条 pipeline 有几个明显优势。

### 4.1 冻结 manifest，先统一边界再做重计算

所有 clip 会先进入 frozen manifest。后续的：

- 标注
- 推理
- 过滤
- build

都围绕同一个 clip 集合工作。这样做的好处是：

- 可以稳定复跑
- 可以先 filter 再 build
- 可以保证 pre-filter 与 post-build filter 的 clip 选择一致
- 可以避免因为上游扫描变化导致结果漂移

### 4.2 几何、语义、图像三者真正对齐

这条 pipeline 不只是把 HaWoR 结果塞进 tar 包。它会把：

- 图像
- 低维动作表示
- 相机内外参
- MANO 参数
- 文本动作标注

一起打包成统一样本，并保留调试与重建能力。

### 4.3 质量过滤前置，避免把明显脏样本写进最终数据集

在 build 之前，系统会基于 build 等价的特征生成逻辑先做 manifest filter。  
过滤标准不仅看文件是否存在，也会看：

- 低维特征是否存在 NaN/Inf
- 相机平移和旋转是否异常跳变
- wrist / hand 在 camera-space 中是否落在极端离群区间
- annotation 是否完整可用

这样能把显然脏到不可训练的样本，在落盘前就剔除掉。

---

## 5. 比 HaWoR 快多少

这一部分建议你们一定使用实测数，不要拍脑袋写。

目前仓库里已经有 benchmark 工具：

```bash
python tools/ops/benchmark_slam_stage.py \
  --factory_dir /path/to/factory \
  --gpus 0,1,2,3,4,5,6,7 \
  --num_videos 64 \
  --sample_mode random
```

它会输出：

- `videos_completed`
- `elapsed_sec`
- `avg_sec_per_video`
- GPU 利用率与显存统计

### 建议写法

如果你们已经有原始 HaWoR baseline，就在 blog 里写成下面这种格式：

> 在相同数据子集、相同 GPU 数量下，我们的 pipeline 将平均单视频处理时间从 **[X] 秒** 降到 **[Y] 秒**，整体达到 **[Z]x speedup**。

### 当前草稿中的建议占位

先不要硬写死数字，先留成：

> 在我们的 64-video / 8-GPU 基准设置下，RoWaH pipeline 的平均吞吐达到了 **[待填：Y sec/video]**，相对于原始 HaWoR baseline 实现为 **[待填：Zx]** 的加速。

### 这个“更快”主要来自哪里

可以在文中解释为三类优化：

1. **调度层优化**：从单脚本推理转为 batch scheduler / wave 调度 / 多 GPU 跑批
2. **数据边界优化**：manifest 冻结后避免重复扫描与无效重跑
3. **构建层优化**：过滤、build、MANO 特征缓存、resume 与并行写 shard

如果你想更稳一点，可以把标题写成：

> 比原始 HaWoR 数据生产流程更快，而不是简单宣称“模型更快”

因为这里提升的主要是 **pipeline throughput**，不一定是单模型纯推理 latency。

---

## 6. 为什么说它“更好”

“更好”这件事，建议不要空说精度，而是从 **结果可用性和一致性** 来讲。

### 更好的点 1：语义和几何统一了

以前很多结果只能做可视化，现在每个样本同时带有：

- 动作标注
- 图像
- world-space 手状态
- 相机内外参
- MANO 可重建参数

这让它更像真正的训练数据，而不是中间结果缓存。

### 更好的点 2：可以做严格对齐调试

仓库里已经有对齐调试工具，可以直接检查 WebDataset sample 是否和源 episode 对齐：

```bash
python tools/ops/debug_wds_sample_alignment.py \
  --input /path/to/shard.tar \
  --descriptor-manifest /path/to/clip_manifest.jsonl \
  --sample-index 0
```

这让“看起来差不多”变成“可以验证确实对齐”。

### 更好的点 3：可以直接回放成 demo

你们现在已经有基于 Rerun 的可视化工具，可以直接展示：

- 图像空间 keypoint overlay
- skeleton overlay
- mesh replay
- camera + world 双视角

命令例如：

```bash
python tools/ops/rerun_webdataset_visualizer.py \
  --input /path/to/final_dataset/train \
  --render-mode keypoint
```

或者导出 mesh 版本：

```bash
python tools/ops/rerun_webdataset_visualizer.py \
  --input /path/to/final_dataset/train \
  --render-mode mesh \
  --output-mode offline \
  --rrd-out /path/to/episode_mesh.rrd
```

---

## 7. 建议放哪些 Demo

你提到“更好（放几个 demo）”，我建议不要只放一类图，至少放 3 组。

### Demo A：动作标注 + 图像

展示同一个 clip：

- 原始 RGB
- 该 clip 的 instruction
- 若干关键帧上的手部 overlay

目的：强调这不是“只有手部姿态”，而是带动作语义的训练样本。

建议配图文案：

> 每个 clip 除了图像与三维手部状态外，还显式绑定动作文本描述，可直接作为具身动作学习或 VLA 数据样本使用。

### Demo B：内外参 + 3D 重放

展示：

- camera view overlay
- world view mesh replay

目的：强调相机几何是样本的一部分，不是离线调试附属品。

建议配图文案：

> 最终样本保存 world-to-camera 外参与 pinhole 内参，因此 3D 手部状态可以稳定重投影回图像，并在世界系中重放。

### Demo C：RoWaH vs 原始 HaWoR 对比

这里不建议直接比“更好看”三个字，最好放并排图：

- 左：baseline HaWoR / 原始输出
- 右：RoWaH pipeline build 后的 WDS replay

如果你们有 `compare-demo` 视图，甚至可以做：

```bash
python tools/ops/webdataset_visualizer.py \
  --input /path/to/shard.tar \
  --keypoint-source compare-demo
```

目的：强调结果一致性、几何对齐和最终数据可复现性。

### Demo D：深度与相机几何

如果你们手头有 Any4D 深度可视化图，建议单独放一组：

- RGB
- depth
- scale-aware camera / wrist replay

目的：解释为什么这套数据不是仅凭 2D 检测得到的“伪 3D”。

---

## 8. 一版可直接发的简洁版结论

如果你希望结尾写得更像技术博客收束，可以直接用下面这段：

> RoWaH pipeline 做的不是把 HaWoR 简单包一层脚本，而是把手部重建、深度几何、相机内外参和动作标注，统一到一个可规模化生产的数据边界中。  
> 对研究来说，它让结果更容易调试和复现；对数据生产来说，它让过滤、重建、回放和训练接入都变得更稳定。  
> 更重要的是，它把原本偏“单视频推理结果”的输出，变成了真正可训练、可验证、可扩展的数据资产。

---

## 9. 发布前待补的 3 个信息

这篇 blog 在正式发布前，建议你补齐下面三项：

1. **速度数字**
   - 填上 baseline HaWoR 的 `avg_sec_per_video`
   - 填上 pipeline 的 `avg_sec_per_video`
   - 计算 speedup 倍数

2. **3 组 demo 资源**
   - 最好至少有 keypoint / mesh / world-view 三种

3. **一组动作标注示例**
   - 给出一个真实 instruction 示例，增强“动作标注”这部分的说服力

---

## 10. 推荐标题候选

你可以从下面几个里挑一个：

1. **从 HaWoR 到可规模化数据生产：RoWaH Pipeline 的设计与实现**
2. **把手部重建变成训练数据：RoWaH Pipeline 如何统一动作标注、相机几何与深度**
3. **RoWaH Pipeline：面向大规模手部动作数据生产的几何与语义统一方案**
4. **不仅更快，而且更可用：RoWaH Pipeline 如何把 HaWoR 输出变成真正的数据集**

---

## 附：建议插图位置

建议最终 blog 至少放下面 5 张图：

1. 总览图：pipeline 全流程
2. 动作标注示例图：RGB + instruction
3. camera view overlay
4. world view mesh replay
5. baseline vs pipeline 对比图

如果你愿意，下一步我可以继续直接帮你把这篇草稿压成：

- 一个更像公众号/博客风格的短版
- 一个更像技术文档/项目主页风格的英文版
- 或者补成一个可直接发布到 Markdown 平台的最终稿
