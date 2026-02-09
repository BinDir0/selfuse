# Arm Package

LegendVLA Inference系统的机械臂控制包，包含IK、FK和控制节点。

## 架构说明

按照系统架构图：

### Arm IK Node
- **频率**: 100Hz
- **输入**: `/action/{left,right}_arm/joints` (JointState)
- **输出**: `/state/{left,right}_arm/joints` (JointState)

### Arm FK Node
- **频率**: 100Hz
- **输入**: `/state/{left,right}_arm/joints` (JointState)
- **输出**: 发送到Arm Control Node

### Arm Control Node
- **频率**: 100Hz
- **输入**: 
  - `/state/{left,right}_arm/wrist_pose` (JointState) - arm states (wrist poses in camera frame)
  - `/camera/{head,chest}/rgb` (Image) - camera RGB-D images
  - `/camera/{head,chest}/depth` (Image) - camera RGB-D images
- **输出**: `/action/{left,right}_arm/joints` (JointState)

## 文件结构

```
arm/
├── arm/
│   ├── __init__.py
│   ├── arm_ik_node.py         # 逆运动学节点
│   ├── arm_fk_node.py         # 正运动学节点
│   └── arm_control_node.py   # 控制节点
├── launch/
│   ├── arm_system.launch.py  # 单臂系统启动
│   └── dual_arms.launch.py   # 双臂系统启动
├── resource/
│   └── arm
├── CMakeLists.txt
├── package.xml
└── README.md
```

## 编译

```bash
colcon build --packages-select arm
source install/setup.bash
```

## 使用方法

### 1. 启动单臂系统

```bash
# 启动左臂
ros2 launch arm arm_system.launch.py arm_side:=left

# 启动右臂
ros2 launch arm arm_system.launch.py arm_side:=right

# 自定义频率
ros2 launch arm arm_system.launch.py arm_side:=left frequency:=50.0
```

### 2. 启动双臂系统

```bash
# 同时启动左右臂
ros2 launch arm dual_arms.launch.py

# 自定义频率
ros2 launch arm dual_arms.launch.py frequency:=80.0
```

### 3. 单独运行节点

```bash
# 运行IK节点
ros2 run arm arm_ik_node.py --ros-args -p arm_side:=left

# 运行FK节点
ros2 run arm arm_fk_node.py --ros-args -p arm_side:=left

# 运行Control节点
ros2 run arm arm_control_node.py --ros-args -p arm_side:=left
```

## 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| arm_side | string | 'left' | 机械臂侧别：left 或 right |
| frequency | double | 100.0 | 节点运行频率 (Hz) |

## Topic说明

### Arm IK Node
- **订阅**: `/action/{left,right}_arm/joints` (JointState)
- **发布**: `/state/{left,right}_arm/joints` (JointState)

### Arm FK Node
- **订阅**: `/state/{left,right}_arm/joints` (JointState)
- **发布**: 待定义

### Arm Control Node
- **订阅**: 
  - `/state/{left,right}_arm/wrist_pose` (JointState)
  - `/camera/head/rgb` (Image)
  - `/camera/head/depth` (Image)
  - `/camera/chest/rgb` (Image)
  - `/camera/chest/depth` (Image)
- **发布**: `/action/{left,right}_arm/joints` (JointState)

## 开发说明

### TODO 列表

1. **Arm IK Node**:
   - [ ] 实现IK solver（考虑使用KDL、MoveIt或自定义solver）
   - [ ] 实现动作回调逻辑
   - [ ] 实现状态发布逻辑

2. **Arm FK Node**:
   - [ ] 实现FK solver
   - [ ] 实现状态回调逻辑
   - [ ] 定义输出消息类型和发布者

3. **Arm Control Node**:
   - [ ] 实现控制算法（轨迹规划、碰撞检测等）
   - [ ] 实现图像数据融合
   - [ ] 实现腕部位姿处理逻辑
   - [ ] 优化100Hz控制频率性能

### 依赖库

可能需要的依赖：
- `python3-kdl-parser` - 用于机器人运动学
- `moveit2` - 用于高级运动规划（可选）
- `scipy` - 用于数值优化
- `numpy` - 数值计算

## 与架构图的对应关系

```
┌─────────────────────┐
│  Arm Control Node   │ ✅ 框架已创建
│      100Hz          │
└──────────┬──────────┘
           │ /action/{left,right}_arm/joints
           ↓
┌─────────────────────┐
│   Arm IK Node       │ ✅ 框架已创建
│      100Hz          │
└──────────┬──────────┘
           │ /state/{left,right}_arm/joints
           ↓
┌─────────────────────┐
│   Arm FK Node       │ ✅ 框架已创建
│      100Hz          │
└─────────────────────┘
           ↑
           │ Image: /camera/{head,chest}/{rgb,depth}
┌─────────────────────┐
│   Camera Node       │
│      30Hz           │
└─────────────────────┘
```

## 许可证

Apache-2.0

## 作者

LegendVLA Team

