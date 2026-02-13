# Robotic_Arm (本地版本)

第二代睿尔曼机械臂Python版本二次开发包（本地集成）

## 说明

这是睿尔曼机械臂SDK的本地版本，已集成到项目中，**无需单独安装**。

## 使用方式

代码中通过相对路径自动引用此目录：

```python
# 自动添加resource/Robotic_Arm到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
robotic_arm_path = os.path.join(current_dir, '..', 'resource', 'Robotic_Arm')
sys.path.insert(0, robotic_arm_path)

from rm_robot_interface import *
```

## 目录结构

```
Robotic_Arm/
├── __init__.py
├── rm_robot_interface.py    # 主接口文件
├── rm_ctypes_wrap.py        # 底层封装
└── libs/                    # 平台相关库文件
    ├── linux_arm/
    ├── linux_x86/
    ├── win_32/
    └── win_64/
```