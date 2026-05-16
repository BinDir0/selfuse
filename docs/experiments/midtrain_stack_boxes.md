# 叠盒子单任务 Mid-Train 实验手册

目标：用小规模、任务对齐的人手数据和已 ready 的真机 WDS，验证 EgoScale-style aligned mid-training 是否改善叠盒子的空间泛化和长程稳定性。

## 1. 数据产物

头环的 1 分钟原始切片不要直接作为训练 episode。先离线切成 attempt-level episode：

- 每个 episode 是一次完整或失败的叠盒子 attempt。
- reset、换初始位置、重新摆放、无效等待段必须切断，禁止训练窗口跨边界。
- 每帧输出当前 VLA WDS schema：`image.jpg`、`lowdim.npy`、`meta.json`。
- `lowdim.npy` 仍使用 48D state/action + head camera extrinsic/intrinsic 的现有布局。
- `meta.json` 至少包含 `instruction`、`instruction_num`、`dataset_name`、`episode_index`。

推荐第一轮数据划分：

- `human_train`：约 90 分钟，覆盖训练位置网格。
- `human_val`：约 30 分钟，只放 held-out 位置和组合扰动。
- `robot_train` / `robot_val`：使用同 schema 的已 ready 真机数据。

## 2. 配置入口

先编辑：

- `src/config/dataset_paths/vla_wds_midtrain_stack_boxes.yaml`
- `src/config/experiment/legendvla_qwen3_vl_midtrain_stack_boxes.yaml`

必须替换：

- `midtrain_wds_base_dir`
- `training.finetune_checkpoint_path`

默认主实验：

```bash
bash scripts/midtrain_stack_boxes_single_node.sh
```

smoke：

```bash
EXPERIMENT=legendvla_qwen3_vl_midtrain_stack_boxes_smoke \
  bash scripts/midtrain_stack_boxes_single_node.sh
```

## 3. 训练策略

当前 v0 recipe：

- 从人手预训练基座加载：`training.finetune_checkpoint_path`
- `objective: flow`
- `world_model: none`
- `dataset.vlm_dataset: null`
- aligned VLA-only mixture：human:robot = 4:1
- `training.vlm_train_scope: vision`
- action expert / state encoder / action encoder / vision tower 更新
- language/text/lm head 冻结
- `max_train_steps: 3000`，若方向正确再延到 8000

这个设置对应 EgoScale Stage II 的核心意图：保留语言主体和通用表征，主要让视觉和动作专家适配机器人 sensing/control。

## 4. 数据 Gate

配置路径填好后先跑：

```bash
bash scripts/check_midtrain_stack_boxes_wds.sh
```

输出在：

```text
outputs/preflight/midtrain_stack_boxes/
```

必须检查：

- `*_wds_report.json` 中 `failed == 0`
- `real_batch_train/contract_report.json`
- `real_batch_val/contract_report.json`
- `real_batch_*/report.html` 中图像、动作 overlay、instruction 和 episode 语义一致

## 5. 真机评测

每个 checkpoint 评测两类指标：

- 位置泛化：至少 5 个 held-out 初始/目标位置配置，每个配置 3 trials。
- 长程稳定性：最多连续叠盒 10 次或直到失败，记录最大连续成功次数。

Completion score：

- `+0.25` 抓取正确盒子
- `+0.25` 移动到目标区域上方
- `+0.25` 稳定放置在目标堆叠关系中
- `+0.25` 释放后不倒且可继续下一次

通过标准：

- 主实验 `human+robot midtrain` 在 held-out 位置成功率相对 baseline 提升至少 15 个百分点。
- 连续成功次数不低于当前 baseline。
- 失败模式不能从“位置泛化差”退化为“抓取/释放基本能力变差”。
