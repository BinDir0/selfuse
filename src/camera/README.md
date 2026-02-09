# Camera

LegendVLA Inference系统的相机包，用于采集和发布RGB-D图像数据。

## 架构说明

按照系统架构图：
- **频率**: 30Hz
- **输出类型**: Image (sensor_msgs/Image)
- **输出Topic**: 
  - `/camera/{head,chest}/rgb` - RGB图像
  - `/camera/{head,chest}/depth` - 深度图像

## 功能特性

- 支持RealSense系列相机（D435、D455等）
- 自动对齐深度图到彩色图
- 支持多相机同时运行（head和chest）
- 可配置图像分辨率和发布频率
- 统一的时间戳和frame_id

## 文件结构

```
camera/
├── camera/
│   ├── __init__.py                    # 包初始化
│   ├── camera_node.py                 # 主节点实现
│   └── realsense_image_module.py      # RealSense相机封装
├── launch/
│   ├── camera_node.launch.py          # 单相机启动文件
│   └── dual_cameras.launch.py         # 双相机启动文件
├── resource/
│   └── camera                         # 资源标记文件
├── CMakeLists.txt                     # CMake配置
├── package.xml                        # 包描述文件
└── README.md                          # 本文件
```

## 依赖项

### 系统依赖
- ROS 2 (Humble/Foxy/Galactic)
- Python 3.8+
- librealsense2

### Python依赖
- rclpy
- sensor_msgs
- cv_bridge
- opencv-python
- pyrealsense2
- numpy

### 安装依赖

```bash
# 安装RealSense SDK
sudo apt-get install ros-$ROS_DISTRO-librealsense2*

# 安装Python依赖
pip install pyrealsense2 opencv-python numpy
```

## 编译

在工作空间根目录下：

```bash
# 编译camera包
colcon build --packages-select camera

# 加载环境
source install/setup.bash
```

## 使用方法

### 1. 启动单个相机

```bash
# 使用默认参数（head相机，30Hz）
ros2 launch camera camera_node.launch.py

# 指定相机名称
ros2 launch camera camera_node.launch.py camera_name:=chest

# 指定序列号（用于多相机识别）
ros2 launch camera camera_node.launch.py camera_name:=head serial_number:=134222070573

# 自定义频率和分辨率
ros2 launch camera camera_node.launch.py frequency:=15.0 width:=1280 height:=720
```

### 2. 启动双相机系统

```bash
# 使用默认参数
ros2 launch camera dual_cameras.launch.py

# 指定两个相机的序列号
ros2 launch camera dual_cameras.launch.py \
    head_serial:=134222070573 \
    chest_serial:=836612070298

# 自定义频率
ros2 launch camera dual_cameras.launch.py frequency:=20.0
```

### 3. 直接运行节点

```bash
ros2 run camera camera_node --ros-args \
    -p camera_name:=head \
    -p frequency:=30.0
```

## 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| camera_name | string | 'head' | 相机名称，可选：head, chest |
| serial_number | string | '' | 相机序列号，留空使用默认设备 |
| frequency | double | 30.0 | 发布频率 (Hz) |
| width | int | 640 | 图像宽度 |
| height | int | 480 | 图像高度 |

## Topic说明

### 发布的Topic

| Topic名称 | 消息类型 | 说明 |
|-----------|----------|------|
| `/camera/{camera_name}/rgb` | sensor_msgs/Image | RGB彩色图像 (rgb8) |
| `/camera/{camera_name}/depth` | sensor_msgs/Image | 深度图像 (16UC1) |

其中`{camera_name}`会被替换为实际的相机名称（head或chest）。

## 查看相机序列号

如果你有多个RealSense相机，需要获取它们的序列号：

```bash
# 安装realsense-viewer
sudo apt-get install librealsense2-utils

# 列出所有连接的相机
rs-enumerate-devices

# 或使用Python脚本
python3 -c "import pyrealsense2 as rs; ctx = rs.context(); \
[print(f'{d.get_info(rs.camera_info.serial_number)}: {d.get_info(rs.camera_info.name)}') \
for d in ctx.devices]"
```

## 测试

### 查看发布的topic

```bash
# 查看所有camera相关的topic
ros2 topic list | grep camera

# 查看topic信息
ros2 topic info /camera/head/rgb
ros2 topic info /camera/head/depth

# 查看发布频率
ros2 topic hz /camera/head/rgb
```

### 查看图像

```bash
# 使用rqt_image_view查看RGB图像
ros2 run rqt_image_view rqt_image_view /camera/head/rgb

# 使用rviz2查看
rviz2
# 然后添加Image显示插件，选择对应的topic
```

### 录制数据

```bash
# 录制所有camera topic
ros2 bag record -a -o camera_data

# 只录制特定相机的数据
ros2 bag record /camera/head/rgb /camera/head/depth
```

## 故障排除

### 问题：找不到RealSense设备

**解决方案**：
1. 检查USB连接
2. 确认设备权限：`sudo chmod 666 /dev/bus/usb/*/*`
3. 重新插拔相机

### 问题：两个相机无法同时工作

**解决方案**：
1. 确保使用USB 3.0接口
2. 将相机连接到不同的USB控制器
3. 降低分辨率或帧率

### 问题：图像发布频率不稳定

**解决方案**：
1. 检查系统负载
2. 降低图像分辨率
3. 确保USB带宽充足

## 与架构图的对应关系

```
┌──────────────────┐
│  Camera Node     │  ← 本包实现
│     30Hz         │
└────────┬─────────┘
         │ Image
         │ /camera/{head,chest}/depth
         │ /camera/{head,chest}/rgb
         ↓
┌──────────────────┐
│ Model Interface  │
│      Node        │
└──────────────────┘
```

## 开发说明

### 扩展相机支持

如果需要支持其他类型的相机，可以：

1. 在`camera/`目录下创建新的相机模块（如`other_camera_module.py`）
2. 实现相同的接口：
   - `__init__(SN_number, width, height, fps)`
   - `capture_rgb_depth_frames()` -> (rgb, depth)
   - `close()`
3. 在`camera_node.py`中导入并使用新的模块

### 添加相机内参发布

如果需要发布相机内参，可以参考`realsense_image_module.py`中的`get_camera_intrinsics()`方法。

## 许可证

Apache-2.0

## 作者

LegendVLA Team

