# RuiYan Hand 运动学求解器

基于 MuJoCo + Mink 实现的灵巧手FK/IK求解器，支持左手和右手。

## 功能特性

- ✅ 正运动学(FK): 关节角度 → 指尖位姿
- ✅ 逆运动学(IK): 指尖位姿 → 关节角度
- ✅ MuJoCo 可视化
- ✅ 无 ROS2 依赖
- ✅ 支持左手/右手
- ✅ 支持单指或多指控制

## 安装依赖

```bash
# 如果使用虚拟环境
source .venv/bin/activate
pip install -r requirements.txt
```

## 首次使用：转换URDF

**重要：** 首次使用前需要将URDF转换为MuJoCo MJCF场景格式：

```bash
# 转换左手
python convert_urdf_to_mjcf.py \
  InspiredHand_RuiYan/0611_v1.4/Version_3.0/RuiYan_Hand_Left_Mimic/RuiYan_Hand_Left_Mimic.urdf

# 转换右手
python convert_urdf_to_mjcf.py \
  InspiredHand_RuiYan/0611_v1.4/Version_3.0/RuiYan_Hand_Right_Mimic/RuiYan_Hand_Right_Mimic.urdf
```

# 重要！！！
现在xml里的mimic参数（equality参数）是我暂时修改的，因为原urdf中此参数也有问题。后续手动调整了一下，但也不保证正确。可能需要更进一步的验证！

**转换后生成的场景文件：**
- 📁 位置：`meshes/RuiYan_Hand_XXX_Mimic_scene.xml`（与OBJ文件同级）
- ✅ `<asset>` 标签：声明所有visual mesh
- ✅ `<geom>` 标签：为每个link添加可视化mesh
- ✅ Site定义（5个指尖）
- ✅ Mocap目标（用于IK可视化）
- ✅ Keyframe（home姿态）

**参考设计：** 类似 `src/vive_arm_control/assets/PsiRobot_DC_02_OnlyArm/meshes/xiaozi.xml`

## 使用方法

### 1. FK 求解器

```bash
# 交互式模式
python hand_fk.py --hand left --viewer

# 或在代码中使用
from hand_fk import HandFK
import numpy as np

fk = HandFK(hand_type='left', enable_viewer=True)

# 输入关节角度
joint_positions = np.array([0.5, 0.5, 1.2, 1.2, 1.2, 1.2])  # 6个关节

# 计算FK
results = fk.compute_fk(joint_positions)

# 输出：5个指尖的位姿
for finger, pose in results.items():
    print(f"{finger}: pos={pose['pos']}, quat={pose['quat']}")
```

### 2. IK 求解器

```bash
# 交互式模式
python hand_ik.py --hand left --viewer

# 或在代码中使用
from hand_ik import HandIK
import numpy as np

ik = HandIK(hand_type='left', enable_viewer=True)

# 指定目标位姿（可以只控制部分手指）
target_poses = {
    'index': {
        'pos': np.array([0.0, -0.03, 0.15])  # 目标位置
        # 'quat': np.array([0, 0, 0, 1])     # 可选：目标姿态
    }
}

# 求解IK
result = ik.compute_ik(target_poses, max_iterations=50)

print(f"关节角度: {result['joint_positions']}")
print(f"残差误差: {result['error']:.6f}")
print(f"迭代次数: {result['iterations']}")
```

### 3. 运行测试

```bash
python test_hand_kinematics.py
```

测试内容：
- FK求解准确性
- IK求解准确性
- FK-IK一致性验证

#### 4.解ik和fk的接口使用请参考
    example_usage.py


## 手指名称

- `thumb` - 拇指
- `index` - 食指
- `middle` - 中指
- `ring` - 无名指
- `pinky` - 小指

## 关节自由度

**左手/右手各6个自由度：**
- 拇指: 2个关节 (CMC屈曲, MCP屈曲)
- 食指: 1个关节 (MCP屈曲)
- 中指: 1个关节 (MCP屈曲)
- 无名指: 1个关节 (MCP屈曲)
- 小指: 1个关节 (MCP屈曲)

## 文件结构

```
ruiyan_hand/
├── hand_fk.py              # FK求解器
├── hand_ik.py              # IK求解器
├── convert_urdf_to_mjcf.py # URDF→MJCF转换脚本
├── test_hand_kinematics.py # 测试脚本
├── requirements.txt        # Python依赖
├── README.md              # 本文档
└── InspiredHand_RuiYan/   # 原始URDF和生成的场景文件
    └── 0611_v1.4/Version_3.0/
        ├── RuiYan_Hand_Left_Mimic/
        │   ├── RuiYan_Hand_Left_Mimic.urdf  # 原始URDF（未修改）
        │   └── meshes/                      # Mesh文件目录
        │       ├── *.obj                    # 原始mesh文件（未修改）
        │       ├── *.mtl                    # 原始材质文件（未修改）
        │       └── RuiYan_Hand_Left_Mimic_scene.xml  # 生成的场景文件 ✨
        └── RuiYan_Hand_Right_Mimic/
            ├── RuiYan_Hand_Right_Mimic.urdf # 原始URDF（未修改）
            └── meshes/
                ├── *.obj                    # 原始mesh文件（未修改）
                ├── *.mtl                    # 原始材质文件（未修改）
                └── RuiYan_Hand_Right_Mimic_scene.xml  # 生成的场景文件 ✨
```

**重要说明：**
- ✅ 原始URDF文件完全未修改
- ✅ 所有OBJ/MTL mesh文件完全未修改
- ✨ 场景文件（`*_scene.xml`）放在meshes目录，与OBJ文件同级，避免路径问题

## 技术细节

### 为什么需要转换URDF？

MuJoCo在加载URDF时：
1. 无法正确处理外部mesh文件的相对路径
2. `mujoco.mj_saveLastXML`转换时会丢失visual mesh的geom定义
3. 参考代码（`xiaozi.xml`）使用的是**手动创建的MJCF格式**

**解决方案：**
1. 手动创建MJCF文件（类似`xiaozi.xml`）
2. 在`<asset>`中统一声明所有mesh
3. 为每个link添加`<geom type="mesh" mesh="..."/>`
4. 将场景文件放在meshes目录，与OBJ文件同级
5. 自动添加site和mocap定义

### MJCF vs URDF

| 特性 | URDF | MJCF |
|------|------|------|
| Mesh声明 | 分散在各个link中 | `<asset>`标签统一声明 |
| Mesh路径 | 相对路径常出问题 | 相对于XML文件的简单路径 |
| Site支持 | ❌ 不支持 | ✅ 原生支持 |
| Mocap支持 | ❌ 不支持 | ✅ 原生支持 |
| Fixed Joint | 独立的link | 自动合并到父link |

### 指尖Body名称变化

由于MuJoCo会合并fixed joint，指尖body名称与URDF中不同：

| 手指 | URDF原始 | MJCF转换后 |
|------|----------|------------|
| 拇指 | `hand1_link_1_4` | `hand1_link_1_3` |
| 食指 | `hand1_link_2_3` | `hand1_link_2_2` |
| 中指 | `hand1_link_3_3` | `hand1_link_3_2` |
| 无名指 | `hand1_link_4_3` | `hand1_link_4_2` |
| 小指 | `hand1_link_5_3` | `hand1_link_5_2` |

## 已知问题

1. **IK误差较大** - 当目标位置距离较远时，IK求解误差可能达到10-15mm
   - 可能原因：关节限制、IK参数需要调优
   - 解决方向：调整`mink.FrameTask`的cost权重，增加迭代次数

2. **只保留visual mesh** - collision mesh已移除
   - 原因：避免MuJoCo加载问题
   - 影响：无碰撞检测（对FK/IK影响不大）

## 参考代码

本项目参考了以下ROS2节点的实现：
- `src/vive_arm_control/vive_arm_control/mink_ik_node.py`
- `src/vive_arm_control/vive_arm_control/mink_fk_node.py`
- `src/vive_arm_control/vive_arm_control/xiaozi_mink_ik_node.py`
- `src/vive_arm_control/vive_arm_control/xiaozi_mink_fk_node.py`

主要区别：
- ✅ 去除了所有ROS2依赖
- ✅ 直接加载MJCF而不是URDF
- ✅ 使用site而不是body作为IK目标
- ✅ 提供命令行交互界面

## License





与主项目相同

## 作者

基于原有ROS2节点改编，移除ROS2依赖并适配RuiYan Hand
