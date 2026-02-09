#!/usr/bin/env python3
"""
Arm System Launch File
启动完整的机械臂系统（IK + FK + Control 节点）
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """生成启动描述 - 启动单侧机械臂的所有节点"""

    # 声明启动参数
    arm_side_arg = DeclareLaunchArgument(
        'arm_side',
        default_value='left',
        description='机械臂侧别: left 或 right'
    )
    
    frequency_arg = DeclareLaunchArgument(
        'frequency',
        default_value='100.0',
        description='节点运行频率 (Hz)，默认100Hz'
    )

    # Arm IK Node
    arm_ik_node = Node(
        package='arm',
        executable='arm_ik_node.py',
        name='arm_ik_node',
        output='screen',
        parameters=[{
            'arm_side': LaunchConfiguration('arm_side'),
            'frequency': LaunchConfiguration('frequency'),
        }],
    )
    
    # Arm FK Node
    arm_fk_node = Node(
        package='arm',
        executable='arm_fk_node.py',
        name='arm_fk_node',
        output='screen',
        parameters=[{
            'arm_side': LaunchConfiguration('arm_side'),
            'frequency': LaunchConfiguration('frequency'),
        }],
    )
    
    # Arm Control Node
    arm_control_node = Node(
        package='arm',
        executable='arm_control_node.py',
        name='arm_control_node',
        output='screen',
        parameters=[{
            'arm_side': LaunchConfiguration('arm_side'),
            'frequency': LaunchConfiguration('frequency'),
        }],
    )

    return LaunchDescription([
        arm_side_arg,
        frequency_arg,
        arm_ik_node,
        arm_fk_node,
        arm_control_node,
    ])

