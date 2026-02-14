# Arm Package

LegendVLA Inference系统的机械臂控制包，包含IK、FK和控制节点。

## 当前进度
### 待编辑：
- `package.xml`
- debug `launch` 目录
### 已验证：
- `arm_ik_node`
- `arm_fk_node`
- `arm_control_node`无插值，无sdk

## 架构说明

按照系统架构图：

### Arm IK Node
- **频率**: 100Hz
- **输入**: `/action/both_arms/wrist_poses` (PoseArray)
- **输出**: `/state/{left,right}_arm/joints` (JointState)

### Arm FK Node
- **频率**: 100Hz
- **输入**: `/state/{left,right}_arm/joints` (JointState)
- **输出**: 发送到Arm Control Node
> 注：目前考虑去除该节点，用直接读control结果的位置代替

### Arm Control Node
- **频率**: 100Hz
- **输入**: `/state/{left,right}_arm/joints` (JointState)
- **输出**: `/action/{left,right}_arm/joints` (JointState)

## 文件结构

```
arm/
├── arm/
│   ├── __init__.py
│   ├── arm_ik_node.py         # 逆运动学节点
│   ├── arm_fk_node.py         # 正运动学节点
│   └── arm_control_node.py   # 控制节点
│   └── Ruckig_Interpolator.py   # 插值
│   └── connector_config.py   # 连接件相关参数（已弃用）
├── launch/
│   ├── arm_system.launch.py  # 单臂系统启动（已弃用）
│   └── dual_arms.launch.py   # 双臂系统启动（已弃用）
│   └── launch.xml   # 双臂系统启动
│   └── launch_arm_ik.launch.py   # arm_ik_node启动
│   └── launch_arm_fk.launch.py   # arm_fk_node启动
│   └── launch_ruckig_control.xml   # arm_control_node启动
├── resource/
│   └── arm
│   └── Robotic_arm  # sdk
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

### 最新使用方法
```bash
# 启动双臂
ros2 launch arm launch.xml
```

### 单独运行节点

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

### 依赖库

可能需要的依赖：
- `scipy` - 用于数值优化
- `numpy` - 数值计算
- `mujoco`
- `mink`

## 与架构图的对应关系（已弃用）

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

