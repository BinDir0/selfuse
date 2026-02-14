#!/usr/bin/env python3
"""
Hand System Launch File
启动完整的手部系统（IK + FK + Control 节点）

使用方式:
    # 单侧启动 (默认左手)
    ros2 launch hand hand_system.launch.py

    # 指定右手
    ros2 launch hand hand_system.launch.py hand_side:=right serial_port:=/dev/ttyACM1

    # 自定义频率
    ros2 launch hand hand_system.launch.py frequency:=40.0

    # 自定义电机速度和电流
    ros2 launch hand hand_system.launch.py default_velocity:=2000 default_current:=800
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    """生成启动描述 - 启动单侧手部的所有节点"""

    # ----------------------------------------------------------
    # 共享参数
    # ----------------------------------------------------------
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

    data_timeout_sec_arg = DeclareLaunchArgument(
        'data_timeout_sec',
        default_value='0.5',
        description='数据超时时间 (秒)，超时后停止发布/保持最后位置'
    )

    # ----------------------------------------------------------
    # Hand IK Node 专用参数
    # ----------------------------------------------------------
    ik_solver_arg = DeclareLaunchArgument(
        'ik_solver',
        default_value='daqp',
        description='IK QP 求解器类型 (daqp, quadprog, proxqp)'
    )

    ik_max_iterations_arg = DeclareLaunchArgument(
        'ik_max_iterations',
        default_value='10',
        description='IK 最大迭代次数'
    )

    # ----------------------------------------------------------
    # Hand Control Node 专用参数
    # ----------------------------------------------------------
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyACM0',
        description='RS485 串口设备路径'
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
    # 节点定义 (使用 namespace 避免左右手节点名冲突)
    # ----------------------------------------------------------
    # 使用 hand_side 作为 namespace，节点名变为:
    #   /left/hand_ik_node, /left/hand_fk_node, /left/hand_control_node
    #   /right/hand_ik_node, /right/hand_fk_node, /right/hand_control_node
    # 注意: topic 使用绝对路径 (以 / 开头)，不受 namespace 影响
    hand_nodes = GroupAction(
        actions=[
            PushRosNamespace(LaunchConfiguration('hand_side')),

            # Hand IK Node (架构图: Model Interface → IK → Control)
            Node(
                package='hand',
                executable='hand_ik_node.py',
                name='hand_ik_node',
                output='screen',
                parameters=[{
                    'hand_side': LaunchConfiguration('hand_side'),
                    'frequency': LaunchConfiguration('frequency'),
                    'mjcf_path': LaunchConfiguration('mjcf_path'),
                    'ik_solver': LaunchConfiguration('ik_solver'),
                    'ik_max_iterations': LaunchConfiguration('ik_max_iterations'),
                    'data_timeout_sec': LaunchConfiguration('data_timeout_sec'),
                }],
            ),

            # Hand FK Node (架构图: Control → FK → Model Interface)
            Node(
                package='hand',
                executable='hand_fk_node.py',
                name='hand_fk_node',
                output='screen',
                parameters=[{
                    'hand_side': LaunchConfiguration('hand_side'),
                    'frequency': LaunchConfiguration('frequency'),
                    'mjcf_path': LaunchConfiguration('mjcf_path'),
                }],
            ),

            # Hand Control Node (架构图: IK → Control → Hardware)
            Node(
                package='hand',
                executable='hand_control_node.py',
                name='hand_control_node',
                output='screen',
                parameters=[{
                    'hand_side': LaunchConfiguration('hand_side'),
                    'frequency': LaunchConfiguration('frequency'),
                    'serial_port': LaunchConfiguration('serial_port'),
                    'baudrate': LaunchConfiguration('baudrate'),
                    'enable_interpolation': LaunchConfiguration('enable_interpolation'),
                    'data_timeout_sec': LaunchConfiguration('data_timeout_sec'),
                    'default_velocity': LaunchConfiguration('default_velocity'),
                    'default_current': LaunchConfiguration('default_current'),
                }],
            ),
        ]
    )

    return LaunchDescription([
        # 共享参数
        hand_side_arg,
        frequency_arg,
        mjcf_path_arg,
        data_timeout_sec_arg,
        # IK Node 参数
        ik_solver_arg,
        ik_max_iterations_arg,
        # Control Node 参数
        serial_port_arg,
        baudrate_arg,
        enable_interpolation_arg,
        default_velocity_arg,
        default_current_arg,
        # 节点组
        hand_nodes,
    ])
