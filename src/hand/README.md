# Hand Package

LegendVLA Inference系统的手部控制包，包含IK、FK和控制节点。

## 架构说明

按照系统架构图：

### Hand IK Node
- **频率**: 80Hz
- **输入**: `/action/{left,right}_hand/joints` (JointState)
- **输出**: `/state/{left,right}_hand/joints` (JointState)

### Hand FK Node
- **频率**: 80Hz
- **输入**: `/state/{left,right}_hand/joints` (JointState)
- **输出**: 发送到Hand Control Node

### Hand Control Node
- **频率**: 80Hz
- **输入**: `/state/{left,right}_hand/keypoints` (? DataType)
- **输出**: `/action/{left,right}_hand/keypoints` (? DataType)

## 文件结构

```
hand/
├── hand/
│   ├── __init__.py
│   ├── hand_ik_node.py         # 逆运动学节点
│   ├── hand_fk_node.py         # 正运动学节点
│   └── hand_control_node.py   # 控制节点
├── launch/
│   ├── hand_system.launch.py  # 单手系统启动
│   └── dual_hands.launch.py   # 双手系统启动
├── resource/
│   └── hand
├── CMakeLists.txt
├── package.xml
└── README.md
```

## 编译

```bash
colcon build --packages-select hand
source install/setup.bash
```

## 使用方法

### 1. 启动单手系统

```bash
# 启动左手
ros2 launch hand hand_system.launch.py hand_side:=left

# 启动右手
ros2 launch hand hand_system.launch.py hand_side:=right

# 自定义频率
ros2 launch hand hand_system.launch.py hand_side:=left frequency:=100.0
```

### 2. 启动双手系统

```bash
# 同时启动左右手
ros2 launch hand dual_hands.launch.py

# 自定义频率
ros2 launch hand dual_hands.launch.py frequency:=60.0
```

### 3. 单独运行节点

```bash
# 运行IK节点
ros2 run hand hand_ik_node.py --ros-args -p hand_side:=left

# 运行FK节点
ros2 run hand hand_fk_node.py --ros-args -p hand_side:=left

# 运行Control节点
ros2 run hand hand_control_node.py --ros-args -p hand_side:=left
```

## 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| hand_side | string | 'left' | 手部侧别：left 或 right |
| frequency | double | 80.0 | 节点运行频率 (Hz) |

## Topic说明

### Hand IK Node
- **订阅**: `/action/{left,right}_hand/joints` (JointState)
- **发布**: `/state/{left,right}_hand/joints` (JointState)

### Hand FK Node
- **订阅**: `/state/{left,right}_hand/joints` (JointState)
- **发布**: 待定义

### Hand Control Node
- **订阅**: `/state/{left,right}_hand/keypoints` (待定义消息类型)
- **发布**: `/action/{left,right}_hand/keypoints` (待定义消息类型)

## 开发说明

### TODO 列表

1. **Hand IK Node**:
   - [ ] 实现IK solver
   - [ ] 实现动作回调逻辑
   - [ ] 实现状态发布逻辑

2. **Hand FK Node**:
   - [ ] 实现FK solver
   - [ ] 实现状态回调逻辑
   - [ ] 定义输出消息类型和发布者

3. **Hand Control Node**:
   - [ ] 定义关键点消息类型
   - [ ] 实现控制算法
   - [ ] 实现状态处理逻辑

### 自定义消息类型

可能需要定义以下消息类型：
- `HandKeypoints.msg` - 手部关键点数据结构

## 与架构图的对应关系

```
┌─────────────────────┐
│  Hand Control Node  │ ✅ 框架已创建
│       80Hz          │
└──────────┬──────────┘
           │ /action/{left,right}_hand/keypoints
           ↓
┌─────────────────────┐
│   Hand IK Node      │ ✅ 框架已创建
│       80Hz          │
└──────────┬──────────┘
           │ /action/{left,right}_hand/joints
           ↓
┌─────────────────────┐
│   Hand FK Node      │ ✅ 框架已创建
│       80Hz          │
└─────────────────────┘
```

## 许可证

Apache-2.0

## 作者

LegendVLA Team

