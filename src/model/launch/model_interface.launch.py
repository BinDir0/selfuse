#!/usr/bin/env python3
"""
Model Interface Node Launch File
启动模型接口节点（WebSocket Client）
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """生成启动描述 - 启动Model Interface Node"""

    # 声明启动参数
    model_server_url_arg = DeclareLaunchArgument(
        'model_server_url',
        default_value='ws://localhost:8765',
        description='远程模型服务器的WebSocket URL'
    )
    
    frequency_arg = DeclareLaunchArgument(
        'frequency',
        default_value='10.0',
        description='推理请求频率 (Hz)'
    )
    
    instruction_arg = DeclareLaunchArgument(
        'instruction',
        default_value='pick up the cup',
        description='默认指令'
    )

    # Model Interface Node
    model_interface_node = Node(
        package='model',
        executable='model_interface_node.py',
        name='model_interface_node',
        output='screen',
        parameters=[{
            'model_server_url': LaunchConfiguration('model_server_url'),
            'frequency': LaunchConfiguration('frequency'),
            'instruction': LaunchConfiguration('instruction'),
        }],
    )

    return LaunchDescription([
        model_server_url_arg,
        frequency_arg,
        instruction_arg,
        model_interface_node,
    ])

