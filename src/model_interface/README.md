# Model Package

LegendVLA Inference系统的模型接口包，作为WebSocket客户端连接远程模型服务器。

## 架构说明

### Model Interface Node (WebSocket Client) - 本包实现
- **角色**: WebSocket Client（ROS节点）
- **频率**: ~10Hz（可配置）
- **功能**: 连接ROS系统和远程模型服务器

**订阅（输入）**:
- `Image`: `/camera/{head,chest}/rgb` - RGB图像
- `Image`: `/camera/{head,chest}/depth` - 深度图像
- `JointState`: `/state/{left,right}_arm/joints` - 机械臂状态
- `? DataType`: `/state/{left,right}_hand/keypoints` - 手部关键点状态

**发布（输出）**:
- `PoseStamped`: `/action/both_arms/wrist_poses` - 机械臂腕部位姿动作（相机坐标系）
- `? DataType`: `/action/{left,right}_hand/keypoints` - 手部关键点动作（腕部坐标系）

**WebSocket通信** (到远程Model Server):
- **发送**: camera_image, instruction, camera_meta_data, state_history
- **接收**: action_chunk_vlm, action_chunk_fm

### 远程Model Server（不在本包中）
- **部署位置**: Cloud / Local (other machine) / Local (host machine) / Local (docker)
- **功能**: 接收数据，运行VLA模型推理，返回动作
- **注意**: Model Server由其他团队/项目负责，本包只实现客户端

## 文件结构

```
model/
├── model/
│   ├── __init__.py
│   └── model_interface_node.py   # ✅ WebSocket Client节点
├── config/
│   └── model_config.yaml          # 配置文件
├── launch/
│   └── model_interface.launch.py  # ✅ 启动文件
├── resource/
│   └── model
├── CMakeLists.txt
├── package.xml
└── README.md
```

## 依赖项

### Python依赖
```bash
pip install websockets numpy opencv-python
```

## 编译

```bash
colcon build --packages-select model
source install/setup.bash
```

## 使用方法

### 启动Model Interface Node

```bash
# 连接到本地模型服务器
ros2 launch model model_interface.launch.py

# 连接到远程模型服务器
ros2 launch model model_interface.launch.py \
    model_server_url:=ws://192.168.1.100:8765

# 指定指令和频率
ros2 launch model model_interface.launch.py \
    model_server_url:=ws://10.0.0.50:8765 \
    instruction:="抓取桌上的杯子" \
    frequency:=15.0

# 连接到云端服务器
ros2 launch model model_interface.launch.py \
    model_server_url:=ws://model-server.example.com:8765
```

### 直接运行节点

```bash
ros2 run model model_interface_node.py --ros-args \
    -p model_server_url:=ws://localhost:8765 \
    -p frequency:=10.0 \
    -p instruction:="pick up the cup"
```

## 配置说明

### Model Interface Node参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| model_server_url | string | 'ws://localhost:8765' | 远程模型服务器URL |
| frequency | double | 10.0 | 推理请求频率 (Hz) |
| instruction | string | 'pick up the cup' | 默认指令 |

## Topic说明

### 订阅的Topic

| Topic名称 | 消息类型 | 说明 |
|-----------|----------|------|
| `/camera/head/rgb` | sensor_msgs/Image | Head相机RGB图像 |
| `/camera/head/depth` | sensor_msgs/Image | Head相机深度图像 |
| `/camera/chest/rgb` | sensor_msgs/Image | Chest相机RGB图像 |
| `/camera/chest/depth` | sensor_msgs/Image | Chest相机深度图像 |
| `/state/left_arm/joints` | sensor_msgs/JointState | 左臂关节状态 |
| `/state/right_arm/joints` | sensor_msgs/JointState | 右臂关节状态 |
| `/state/{left,right}_hand/keypoints` | ? DataType | 手部关键点（待定义） |

### 发布的Topic

| Topic名称 | 消息类型 | 说明 |
|-----------|----------|------|
| `/action/both_arms/wrist_poses` | geometry_msgs/PoseStamped | 双臂腕部位姿动作 |
| `/action/{left,right}_hand/keypoints` | ? DataType | 手部关键点动作（待定义） |

## WebSocket协议

### 请求格式（发送到Model Server）

```json
{
  "camera_images": {
    "head_rgb": { "shape": [480, 640, 3], "dtype": "uint8", "data": "..." },
    "head_depth": { "shape": [480, 640], "dtype": "uint16", "data": "..." },
    "chest_rgb": { "shape": [480, 640, 3], "dtype": "uint8", "data": "..." },
    "chest_depth": { "shape": [480, 640], "dtype": "uint16", "data": "..." }
  },
  "instruction": "pick up the cup",
  "camera_meta_data": {
    "width": 640,
    "height": 480,
    "fx": 500.0,
    "fy": 500.0
  },
  "state_history": [...],
  "current_state": {
    "left_arm": {"position": [...], "velocity": [...]},
    "right_arm": {"position": [...], "velocity": [...]},
    "left_hand": [...],
    "right_hand": [...]
  },
  "timestamp": 1234567890
}
```

### 响应格式（从Model Server接收）

```json
{
  "action_chunk_vlm": {
    "left_hand": [...],
    "right_hand": [...],
    "left_arm": [...],
    "right_arm": [...]
  },
  "action_chunk_fm": {
    "left_hand": [...],
    "right_hand": [...],
    "left_arm": [...],
    "right_arm": [...]
  },
  "timestamp": 1234567890,
  "status": "success"
}
```

## 部署架构

### 典型部署方案

```
┌─────────────────────────────┐
│   控制机 (ROS2)              │
│                             │
│  ┌───────────────────────┐  │
│  │  Camera Node          │  │
│  │  Hand Nodes           │  │
│  │  Arm Nodes            │  │
│  │  Model Interface Node │◄─┼──── WebSocket Client (本包实现)
│  └───────────┬───────────┘  │
└──────────────┼───────────────┘
               │ WebSocket
               │ ws://192.168.1.100:8765
               ↓
┌──────────────┼───────────────┐
│   GPU服务器 / 云端            │
│  ┌───────────▼───────────┐  │
│  │  Model Server         │◄─┼──── WebSocket Server (其他团队负责)
│  │  (VLA模型推理)         │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
```

## 开发说明

### TODO 列表

1. **Model Interface Node**:
   - [ ] 实现图像编码/解码（base64或其他格式）
   - [ ] 定义手部关键点消息类型
   - [ ] 实现动作转换逻辑（action_chunk → ROS messages）
   - [ ] 添加相机内参获取
   - [ ] 优化WebSocket重连机制
   - [ ] 添加动作平滑和插值
   - [ ] 添加请求/响应的错误处理
   - [ ] 添加超时处理
   - [ ] 实现状态历史管理

### 自定义消息类型

需要定义以下消息类型（在单独的msg包中）：
```bash
# 创建消息包
ros2 pkg create --build-type ament_cmake legend_msgs

# 定义消息
# HandKeypoints.msg
# ArmWristPoses.msg
```

## 测试

### 测试WebSocket连接

```bash
# 查看Model Interface Node状态
ros2 node info /model_interface_node

# 查看日志
ros2 node list
ros2 topic list | grep action

# 监控输出topic
ros2 topic hz /action/both_arms/wrist_poses

# Echo订阅的数据
ros2 topic echo /camera/head/rgb
```

### 使用模拟服务器测试

如果Model Server还未就绪，可以用简单的测试服务器：

```python
# test_server.py - 简单的模拟服务器
import asyncio
import websockets
import json

async def handler(websocket, path):
    print("Client connected")
    async for message in websocket:
        data = json.loads(message)
        print(f"Received request with instruction: {data.get('instruction')}")
        
        # 返回空动作作为响应
        response = {
            "action_chunk_vlm": {
                "left_arm": [],
                "right_arm": [],
                "left_hand": [],
                "right_hand": []
            },
            "action_chunk_fm": {
                "left_arm": [],
                "right_arm": [],
                "left_hand": [],
                "right_hand": []
            },
            "status": "success",
            "timestamp": data.get("timestamp", 0)
        }
        await websocket.send(json.dumps(response))

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("Test server running on ws://localhost:8765")
        await asyncio.Future()

asyncio.run(main())
```

运行测试：
```bash
# Terminal 1: 启动测试服务器
python3 test_server.py

# Terminal 2: 启动Model Interface Node
ros2 launch model model_interface.launch.py
```

## 与架构图的对应关系

```
┌─────────────────────────────────────┐
│  Cloud / Local (other machine) /    │
│  Local (host machine) / Local       │
│         (docker)                     │
│  ┌───────────────────────────────┐  │
│  │   Model (WebSocket Server)    │  │ ❌ 不在本包中
│  │   (其他团队负责)               │  │
│  └───────────────┬───────────────┘  │
└──────────────────┼───────────────────┘
                   │ WebSocket
                   │ action chunk (VLM/FM)
                   ↓
         ┌─────────────────────┐
         │ Model Interface     │        ✅ 本包唯一实现
         │      Node           │        ✅ model_interface_node.py
         │  (WebSocket Client) │
         └──────────┬──────────┘
                    │
         ┌──────────┴──────────┐
         ↓                     ↓
    Camera Images         State Data
    Instructions          History
```

## 常见问题

### Q: Model Server的代码在哪里？
A: Model Server由其他团队/项目负责实现和部署，本包只实现WebSocket客户端。

### Q: 如何连接到不同的Model Server？
A: 通过`model_server_url`参数指定：
```bash
ros2 launch model model_interface.launch.py \
    model_server_url:=ws://your-server:port
```

### Q: 支持哪些Model Server实现？
A: 只要Model Server遵循定义的WebSocket协议（请求/响应格式），就可以连接。

### Q: 如何调试WebSocket连接问题？
A: 查看节点日志，检查网络连接，使用测试服务器验证。

## 许可证

Apache-2.0

## 作者

LegendVLA Team
