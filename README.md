# LegendVLA-Inference

## Environment Setup

1. **Prerequisites**: Install Docker and download the image file.
2. **Load Image**: 
```bash
docker load -i teleop.tar
```
3. **Create Container**: Update the host path in `create_container.sh` (line 137) to your local project directory, then execute:
```bash
./create_container.sh legendvla-inference
```

## 推理流程

### 1. 初始启动
1. **启动服务器**：首先启动模型推理服务器。
2. **一键启动客户端**：在本文件夹执行启动命令，系统将自动加载相机、机械臂、机械手及推理客户端节点。
### 2. 任务循环 (Loop)
系统按以下流程循环执行任务：
1. **配置任务**：在终端输入语言指令，并选择运行模式（`deploy` 或 `debug`）。
2. **触发开始**：踩下脚踏板 1，系统播放“开始执行”提示音并启动任务。
3. **过程控制**：
    - `deploy` 模式：连续执行。期间按空格键或踩下脚踏板 2 可实现 暂停/继续。
    - `debug` 模式：单步执行。每按一次空格键或踩下脚踏板 2，机器人向前执行一步预测动作。
4. **归位与重置**：踩下脚踏板 3，系统执行归位动作并结束当前任务。
5. **进入下一轮**：系统会再次提示输入新的指令和模式。