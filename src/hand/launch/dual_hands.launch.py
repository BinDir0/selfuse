#!/usr/bin/env python3
"""
Dual Hands Launch File
同时启动左右手的完整系统 (IK + FK + Control)

使用方式:
    # 默认参数启动
    ros2 launch hand dual_hands.launch.py

    # 自定义串口
    ros2 launch hand dual_hands.launch.py \
        left_serial_port:=/dev/ttyACM0 \
        right_serial_port:=/dev/ttyACM1

    # 自定义频率
    ros2 launch hand dual_hands.launch.py frequency:=40.0

    # 自定义电机速度和电流
    ros2 launch hand dual_hands.launch.py default_velocity:=2000 default_current:=800
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """生成启动描述 - 启动双手系统"""

    # ----------------------------------------------------------
    # 共享参数
    # ----------------------------------------------------------
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

    data_timeout_sec_arg = DeclareLaunchArgument(
        'data_timeout_sec',
        default_value='0.5',
        description='数据超时时间 (秒)'
    )

    baudrate_arg = DeclareLaunchArgument(
        'baudrate',
        default_value='460800',
        description='串口波特率'
    )

    enable_interpolation_arg = DeclareLaunchArgument(
        'enable_interpolation',
        default_value='true',
        description='是否启用 Ruckig 平滑插值'
    )

    ik_solver_arg = DeclareLaunchArgument(
        'ik_solver',
        default_value='daqp',
        description='IK QP 求解器类型 (daqp, quadprog, proxqp)'
    )

    default_velocity_arg = DeclareLaunchArgument(
        'default_velocity',
        default_value='3000',
        description='电机默认速度 (与遥操 ry_hand_485_node 一致)'
    )

    default_current_arg = DeclareLaunchArgument(
        'default_current',
        default_value='1000',
        description='电机默认电流 (与遥操 ry_hand_485_node 一致)'
    )

    # ----------------------------------------------------------
    # 左右手独立参数 (串口设备不同)
    # ----------------------------------------------------------
    left_serial_port_arg = DeclareLaunchArgument(
        'left_serial_port',
        default_value='/dev/ttyACM0',
        description='左手 RS485 串口设备路径'
    )

    right_serial_port_arg = DeclareLaunchArgument(
        'right_serial_port',
        default_value='/dev/ttyACM1',
        description='右手 RS485 串口设备路径'
    )

    # ----------------------------------------------------------
    # 获取 launch 文件路径
    # ----------------------------------------------------------
    hand_pkg_dir = get_package_share_directory('hand')
    hand_system_launch = os.path.join(
        hand_pkg_dir, 'launch', 'hand_system.launch.py'
    )

    # ----------------------------------------------------------
    # 左手系统
    # ----------------------------------------------------------
    left_hand = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(hand_system_launch),
        launch_arguments={
            'hand_side': 'left',
            'frequency': LaunchConfiguration('frequency'),
            'mjcf_path': LaunchConfiguration('mjcf_path'),
            'data_timeout_sec': LaunchConfiguration('data_timeout_sec'),
            'ik_solver': LaunchConfiguration('ik_solver'),
            'serial_port': LaunchConfiguration('left_serial_port'),
            'baudrate': LaunchConfiguration('baudrate'),
            'enable_interpolation': LaunchConfiguration('enable_interpolation'),
            'default_velocity': LaunchConfiguration('default_velocity'),
            'default_current': LaunchConfiguration('default_current'),
        }.items()
    )

    # ----------------------------------------------------------
    # 右手系统
    # ----------------------------------------------------------
    right_hand = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(hand_system_launch),
        launch_arguments={
            'hand_side': 'right',
            'frequency': LaunchConfiguration('frequency'),
            'mjcf_path': LaunchConfiguration('mjcf_path'),
            'data_timeout_sec': LaunchConfiguration('data_timeout_sec'),
            'ik_solver': LaunchConfiguration('ik_solver'),
            'serial_port': LaunchConfiguration('right_serial_port'),
            'baudrate': LaunchConfiguration('baudrate'),
            'enable_interpolation': LaunchConfiguration('enable_interpolation'),
            'default_velocity': LaunchConfiguration('default_velocity'),
            'default_current': LaunchConfiguration('default_current'),
        }.items()
    )

    return LaunchDescription([
        # 共享参数
        frequency_arg,
        mjcf_path_arg,
        data_timeout_sec_arg,
        ik_solver_arg,
        baudrate_arg,
        enable_interpolation_arg,
        default_velocity_arg,
        default_current_arg,
        # 左右手串口
        left_serial_port_arg,
        right_serial_port_arg,
        # 节点组
        left_hand,
        right_hand,
    ])
