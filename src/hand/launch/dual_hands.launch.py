#!/usr/bin/env python3
"""
Dual Hands Launch File
同时启动左右手的完整系统
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """生成启动描述 - 启动双手系统"""

    frequency_arg = DeclareLaunchArgument(
        'frequency',
        default_value='80.0',
        description='节点运行频率 (Hz)，默认80Hz'
    )
    
    hand_pkg_dir = get_package_share_directory('hand')
    
    # 左手系统
    left_hand = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(hand_pkg_dir, 'launch', 'hand_system.launch.py')
        ),
        launch_arguments={
            'hand_side': 'left',
            'frequency': LaunchConfiguration('frequency'),
        }.items()
    )
    
    # 右手系统
    right_hand = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(hand_pkg_dir, 'launch', 'hand_system.launch.py')
        ),
        launch_arguments={
            'hand_side': 'right',
            'frequency': LaunchConfiguration('frequency'),
        }.items()
    )

    return LaunchDescription([
        frequency_arg,
        left_hand,
        right_hand,
    ])

