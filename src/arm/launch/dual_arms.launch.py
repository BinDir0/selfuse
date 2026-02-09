#!/usr/bin/env python3
"""
Dual Arms Launch File
同时启动左右机械臂的完整系统
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """生成启动描述 - 启动双臂系统"""

    frequency_arg = DeclareLaunchArgument(
        'frequency',
        default_value='100.0',
        description='节点运行频率 (Hz)，默认100Hz'
    )
    
    arm_pkg_dir = get_package_share_directory('arm')
    
    # 左臂系统
    left_arm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(arm_pkg_dir, 'launch', 'arm_system.launch.py')
        ),
        launch_arguments={
            'arm_side': 'left',
            'frequency': LaunchConfiguration('frequency'),
        }.items()
    )
    
    # 右臂系统
    right_arm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(arm_pkg_dir, 'launch', 'arm_system.launch.py')
        ),
        launch_arguments={
            'arm_side': 'right',
            'frequency': LaunchConfiguration('frequency'),
        }.items()
    )

    return LaunchDescription([
        frequency_arg,
        left_arm,
        right_arm,
    ])

