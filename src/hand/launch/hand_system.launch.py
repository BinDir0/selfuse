#!/usr/bin/env python3
"""
Hand System Launch File
启动完整的手部系统（IK + FK + Control 节点）
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """生成启动描述 - 启动单侧手部的所有节点"""

    # 声明启动参数
    hand_side_arg = DeclareLaunchArgument(
        'hand_side',
        default_value='left',
        description='手部侧别: left 或 right'
    )
    
    frequency_arg = DeclareLaunchArgument(
        'frequency',
        default_value='80.0',
        description='节点运行频率 (Hz)，默认80Hz'
    )

    mjcf_path_arg = DeclareLaunchArgument(
        'mjcf_path',
        default_value='',
        description='MJCF 场景文件路径 (留空使用默认路径)'
    )

    # Hand IK Node
    hand_ik_node = Node(
        package='hand',
        executable='hand_ik_node.py',
        name='hand_ik_node',
        output='screen',
        parameters=[{
            'hand_side': LaunchConfiguration('hand_side'),
            'frequency': LaunchConfiguration('frequency'),
            'mjcf_path': LaunchConfiguration('mjcf_path'),
        }],
    )
    
    # Hand FK Node
    hand_fk_node = Node(
        package='hand',
        executable='hand_fk_node.py',
        name='hand_fk_node',
        output='screen',
        parameters=[{
            'hand_side': LaunchConfiguration('hand_side'),
            'frequency': LaunchConfiguration('frequency'),
        }],
    )
    
    # Hand Control Node
    hand_control_node = Node(
        package='hand',
        executable='hand_control_node.py',
        name='hand_control_node',
        output='screen',
        parameters=[{
            'hand_side': LaunchConfiguration('hand_side'),
            'frequency': LaunchConfiguration('frequency'),
        }],
    )

    return LaunchDescription([
        hand_side_arg,
        frequency_arg,
        mjcf_path_arg,
        hand_ik_node,
        hand_fk_node,
        hand_control_node,
    ])

