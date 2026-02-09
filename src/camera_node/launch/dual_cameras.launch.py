#!/usr/bin/env python3
"""
Dual Camera Node Launch File
同时启动head和chest两个相机节点
按照架构图要求，两个相机都以30Hz频率运行
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """生成启动描述 - 启动两个RealSense相机节点 (head和chest)"""

    # 声明启动参数
    head_serial_arg = DeclareLaunchArgument(
        'head_serial',
        default_value='',
        description='Head相机序列号，留空则使用默认设备'
    )
    
    chest_serial_arg = DeclareLaunchArgument(
        'chest_serial',
        default_value='',
        description='Chest相机序列号，留空则使用默认设备'
    )
    
    frequency_arg = DeclareLaunchArgument(
        'frequency',
        default_value='30.0',
        description='图像发布频率 (Hz)，默认30Hz'
    )

    # Head相机节点
    head_camera_node = Node(
        package='camera_node',
        executable='camera_node',
        name='head_camera_node',
        namespace='head_camera',
        output='screen',
        parameters=[{
            'camera_name': 'head',
            'serial_number': LaunchConfiguration('head_serial'),
            'frequency': LaunchConfiguration('frequency'),
            'width': 640,
            'height': 480,
        }],
    )
    
    # Chest相机节点
    chest_camera_node = Node(
        package='camera_node',
        executable='camera_node',
        name='chest_camera_node',
        namespace='chest_camera',
        output='screen',
        parameters=[{
            'camera_name': 'chest',
            'serial_number': LaunchConfiguration('chest_serial'),
            'frequency': LaunchConfiguration('frequency'),
            'width': 640,
            'height': 480,
        }],
    )

    return LaunchDescription([
        head_serial_arg,
        chest_serial_arg,
        frequency_arg,
        head_camera_node,
        chest_camera_node,
    ])

