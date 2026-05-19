#!/usr/bin/env python3
"""
Camera Node Launch File
启动单个相机节点
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """生成启动描述 - 启动单个RealSense相机节点"""

    # 声明启动参数
    camera_name_arg = DeclareLaunchArgument(
        'camera_name',
        default_value='head',
        description='相机名称 (head 或 chest)'
    )
    
    serial_number_arg = DeclareLaunchArgument(
        'serial_number',
        default_value='',
        description='相机序列号，留空则使用默认设备（写死序列号会导致换机/换相机后报 No device connected）'
    )
    
    frequency_arg = DeclareLaunchArgument(
        'frequency',
        default_value='30.0',
        description='图像发布频率 (Hz)'
    )
    
    width_arg = DeclareLaunchArgument(
        'width',
        default_value='640',
        description='图像宽度'
    )
    
    height_arg = DeclareLaunchArgument(
        'height',
        default_value='480',
        description='图像高度'
    )

    # 相机节点
    camera_node = Node(
        package='camera',
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[{
            'camera_name': LaunchConfiguration('camera_name'),
            'serial_number': ParameterValue(LaunchConfiguration('serial_number'), value_type=str),
            'frequency': LaunchConfiguration('frequency'),
            'width': LaunchConfiguration('width'),
            'height': LaunchConfiguration('height'),
        }],
    )

    return LaunchDescription([
        camera_name_arg,
        serial_number_arg,
        frequency_arg,
        width_arg,
        height_arg,
        camera_node,
    ])

