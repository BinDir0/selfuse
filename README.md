# LegendVLA Inference System

LegendVLA机器人推理系统 - 基于ROS 2的模块化双臂双手机器人控制系统。

## 系统架构

本系统实现了完整的视觉-语言-动作（VLA）推理流程，包括：

- 🎥 **Camera** - 双相机RGB-D图像采集（30Hz）
- 🔗 **Model Interface** - WebSocket客户端节点，连接远程模型服务器（~10Hz）
- 🤚 **Hand** - 双手IK/FK/控制节点（80Hz）
- 🦾 **Arm** - 双臂IK/FK/控制节点（100Hz）
- 🌐 **Model Server** - 远程WebSocket服务器（由其他团队负责，不在本项目中）

## 包结构

```
src/
├── camera/          ✅ 相机包（30Hz）
│   ├── camera_node.py
│   └── realsense_image_module.py
│
├── hand/            ✅ 手部控制包（80Hz）
│   ├── hand_ik_node.py
│   ├── hand_fk_node.py
│   └── hand_control_node.py
│
├── arm/             ✅ 机械臂控制包（100Hz）
│   ├── arm_ik_node.py
│   ├── arm_fk_node.py
│   └── arm_control_node.py
│
└── model/           ✅ 模型接口包（WebSocket Client）
    └── model_interface_node.py    # WebSocket客户端节点
```

## 快速开始

### 1. 环境要求

- Ubuntu 20.04/22.04
- ROS 2 (Humble/Foxy/Galactic)
- Python 3.8+
- CUDA 11.x+ (可选，用于GPU加速)

### 2. 安装依赖

```bash
# ROS 2依赖
sudo apt update
sudo apt install -y \
    ros-$ROS_DISTRO-cv-bridge \
    ros-$ROS_DISTRO-sensor-msgs \
    ros-$ROS_DISTRO-geometry-msgs \
    librealsense2-dev \
    ros-$ROS_DISTRO-librealsense2*

# Python依赖
pip install -r requirements.txt
```

### 3. 编译

```bash
cd ~/workspace/legendvla-inference

# 编译所有包
colcon build

# 或分别编译
colcon build --packages-select camera hand arm model

# 加载环境
source install/setup.bash
```

### 4. 运行系统

#### 启动相机节点
```bash
# 单相机
ros2 launch camera camera_node.launch.py camera_name:=head

# 双相机
ros2 launch camera dual_cameras.launch.py
```

#### 启动手部系统
```bash
# 单手
ros2 launch hand hand_system.launch.py hand_side:=left

# 双手
ros2 launch hand dual_hands.launch.py
```

#### 启动机械臂系统
```bash
# 单臂
ros2 launch arm arm_system.launch.py arm_side:=left

# 双臂
ros2 launch arm dual_arms.launch.py
```

#### 启动模型接口节点
```bash
# 连接到本地模型服务器
ros2 launch model model_interface.launch.py

# 连接到远程模型服务器
ros2 launch model model_interface.launch.py \
    model_server_url:=ws://192.168.1.100:8765

# 指定指令
ros2 launch model model_interface.launch.py \
    instruction:="抓取桌上的杯子"
```

#### 远程模型服务器
```bash
# Model Server由其他团队负责
# 确保远程服务器已启动并可访问
# 例如：ws://192.168.1.100:8765
```

## Topic架构

### Camera Node (30Hz)
```
发布:
  /camera/head/rgb              sensor_msgs/Image
  /camera/head/depth            sensor_msgs/Image
  /camera/chest/rgb             sensor_msgs/Image
  /camera/chest/depth           sensor_msgs/Image
```

### Hand System (80Hz)
```
订阅/发布:
  /action/{left,right}_hand/joints      sensor_msgs/JointState
  /state/{left,right}_hand/joints       sensor_msgs/JointState
  /action/{left,right}_hand/keypoints   (待定义)
  /state/{left,right}_hand/keypoints    (待定义)
```

### Arm System (100Hz)
```
订阅/发布:
  /action/{left,right}_arm/joints       sensor_msgs/JointState
  /state/{left,right}_arm/joints        sensor_msgs/JointState
  /state/{left,right}_arm/wrist_pose    sensor_msgs/JointState
```

### Model Interface Node (~10Hz)
```
订阅:
  /camera/{head,chest}/rgb              sensor_msgs/Image
  /camera/{head,chest}/depth            sensor_msgs/Image
  /state/{left,right}_arm/joints        sensor_msgs/JointState
  /state/{left,right}_hand/keypoints    (待定义)

发布:
  /action/both_arms/wrist_poses         geometry_msgs/PoseStamped
  /action/{left,right}_hand/keypoints   (待定义)

WebSocket Client → 远程Model Server:
  发送: camera_image, instruction, camera_meta_data, state_history
  接收: action_chunk_vlm, action_chunk_fm
```

## 开发状态

### ✅ 已完成
- [x] Camera包 - 完整实现
- [x] Model包 - Model Interface Node框架完成（WebSocket Client）
- [x] Hand包 - 框架搭建完成
- [x] Arm包 - 框架搭建完成

### 🚧 待实现
- [ ] Model Interface Node - 图像编码、动作转换等具体实现
- [ ] Hand IK/FK/Control - 具体算法实现
- [ ] Arm IK/FK/Control - 具体算法实现
- [ ] Model Server - VLA模型集成（远程部署）
- [ ] 自定义消息类型定义
- [ ] 完整的系统集成测试

## 开发指南

### 添加新节点

1. 在对应包的模块目录下创建节点文件
2. 在`CMakeLists.txt`中添加可执行文件配置
3. 创建对应的launch文件
4. 更新README文档

### 自定义消息类型

如需定义新的消息类型（如HandKeypoints）：

1. 创建msg包：
```bash
ros2 pkg create --build-type ament_cmake legend_msgs
```

2. 定义.msg文件
3. 在各包中添加依赖
4. 重新编译

### 调试技巧

```bash
# 查看所有topic
ros2 topic list

# 监控topic频率
ros2 topic hz /camera/head/rgb

# 查看topic内容
ros2 topic echo /camera/head/rgb

# 查看节点信息
ros2 node info /camera_node

# 使用rqt查看系统图
rqt_graph
```

## 性能要求

- **Camera Node**: 30Hz，延迟 < 33ms
- **Hand Nodes**: 80Hz，延迟 < 12.5ms
- **Arm Nodes**: 100Hz，延迟 < 10ms
- **Model Server**: 根据模型复杂度，目标 < 100ms

## 硬件要求

### 最低配置
- CPU: Intel i5-8代 或 AMD Ryzen 5
- RAM: 16GB
- GPU: NVIDIA GTX 1060 6GB（用于模型推理）

### 推荐配置
- CPU: Intel i7-10代+ 或 AMD Ryzen 7
- RAM: 32GB+
- GPU: NVIDIA RTX 3070+ 或 A100（用于模型推理）
- 存储: SSD 500GB+

## 网络部署

### 典型部署架构

```
┌─────────────────────────────┐
│   控制机 (ROS2)              │
│  ┌───────────────────────┐  │
│  │ Camera                │  │
│  │ Hand Nodes            │  │
│  │ Arm Nodes             │  │
│  │ Model Interface Node  │◄─┼─── WebSocket Client
│  └───────────┬───────────┘  │
└──────────────┼───────────────┘
               │ WebSocket
               │ ws://gpu-server:8765
               ↓
┌──────────────┼───────────────┐
│   GPU服务器 / 云端            │
│  ┌───────────▼───────────┐  │
│  │ Model Server          │◄─┼─── WebSocket Server
│  │ (VLA模型推理)         │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
```

### 部署说明

- **控制机**: 运行ROS节点（Camera, Hand, Arm, Model Interface）
- **GPU服务器/云端**: 运行Model Server（由其他团队提供和部署）

```bash
# 在控制机上启动ROS系统
# 确保Model Server已在远程服务器上运行

# 启动Model Interface Node，连接到远程服务器
ros2 launch model model_interface.launch.py \
    model_server_url:=ws://192.168.1.100:8765

# 或连接到云端服务器
ros2 launch model model_interface.launch.py \
    model_server_url:=ws://model-server.example.com:8765
```

**注意**: Model Server的实现和部署不在本项目范围内，由VLA模型团队负责。

## 故障排除

### Camera无法启动
```bash
# 检查RealSense设备
rs-enumerate-devices

# 检查USB权限
sudo chmod 666 /dev/bus/usb/*/*
```

### Topic频率不足
- 检查系统负载：`htop`
- 降低图像分辨率
- 使用性能模式
- 检查网络延迟（分布式部署）

### 模型推理慢
- 启用GPU加速
- 使用模型量化（FP16/INT8）
- 减少batch size
- 使用TensorRT优化

## 贡献指南

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

Apache-2.0 License

## 作者

LegendVLA Team

## 致谢

- ROS 2社区
- Intel RealSense团队
- 所有贡献者
