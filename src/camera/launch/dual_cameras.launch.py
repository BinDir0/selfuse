#!/usr/bin/env python3
"""
Dual Camera Launch File
一次启动 head + breast 两路 RealSense 相机节点（pi0.5 EgoHands use_breast 部署需要）。

序列号是机器特定的，做成启动参数。下面的默认值是本机(psibot)枚举到的两台相机：
  rs-enumerate-devices -s   或
  python3 -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()])"
head/breast 与序列号的对应关系需自行确认（遮镜头看 viewer，或看物理 USB 口）；
若发现 head/breast 装反，交换 head_serial / breast_serial 两个参数即可。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """启动描述 - head + breast 两个 RealSense 相机节点"""

    head_serial = LaunchConfiguration('head_serial')
    breast_serial = LaunchConfiguration('breast_serial')
    frequency = LaunchConfiguration('frequency')
    width = LaunchConfiguration('width')
    height = LaunchConfiguration('height')

    head_camera_node = Node(
        package='camera',
        executable='camera_node',
        name='head_camera_node',
        output='screen',
        parameters=[{
            'camera_name': 'head',
            'serial_number': ParameterValue(head_serial, value_type=str),
            'frequency': frequency,
            'width': width,
            'height': height,
        }],
    )

    breast_camera_node = Node(
        package='camera',
        executable='camera_node',
        name='breast_camera_node',
        output='screen',
        parameters=[{
            'camera_name': 'breast',
            'serial_number': ParameterValue(breast_serial, value_type=str),
            'frequency': frequency,
            'width': width,
            'height': height,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'head_serial',
            default_value='046322250624',
            description='head 相机序列号（本机默认值，换机务必覆盖）',
        ),
        DeclareLaunchArgument(
            'breast_serial',
            default_value='135122250353',
            description='breast 相机序列号（本机默认值，换机务必覆盖）',
        ),
        DeclareLaunchArgument('frequency', default_value='30.0', description='图像发布频率 (Hz)'),
        DeclareLaunchArgument('width', default_value='640', description='图像宽度'),
        DeclareLaunchArgument('height', default_value='480', description='图像高度'),
        head_camera_node,
        breast_camera_node,
    ])
