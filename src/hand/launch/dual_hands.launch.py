#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    left_hand_ik_node = Node(
        package='hand',
        executable='hand_ik_node.py',
        name='left_hand_ik_node',
        output='screen',
        parameters=[{
            'hand_side': 'left',
            'frequency': 80.0,
        }]
    )

    right_hand_ik_node = Node(
        package='hand',
        executable='hand_ik_node.py',
        name='right_hand_ik_node',
        output='screen',
        parameters=[{
            'hand_side': 'right',
            'frequency': 80.0,
        }]
    )

    left_hand_control_node = Node(
        package='hand',
        executable='hand_control_node.py',
        name='left_hand_control_node',
        output='screen',
        parameters=[{
            'hand_side': 'left',
            'frequency': 80.0,
            'serial_port': '/dev/ttyACM0',
            'baudrate': 460800,
            'enable_interpolation': True,
        }]
    )

    right_hand_control_node = Node(
        package='hand',
        executable='hand_control_node.py',
        name='right_hand_control_node',
        output='screen',
        parameters=[{
            'hand_side': 'right',
            'frequency': 80.0,
            'serial_port': '/dev/ttyACM1',
            'baudrate': 460800,
            'enable_interpolation': True,
        }]
    )

    left_hand_fk_node = Node(
        package='hand',
        executable='hand_fk_node.py',
        name='left_hand_fk_node',
        output='screen',
        parameters=[{
            'hand_side': 'left',
            'frequency': 80.0,
        }]
    )

    right_hand_fk_node = Node(
        package='hand',
        executable='hand_fk_node.py',
        name='right_hand_fk_node',
        output='screen',
        parameters=[{
            'hand_side': 'right',
            'frequency': 80.0,
        }]
    )

    return LaunchDescription([
        left_hand_ik_node,
        right_hand_ik_node,
        left_hand_control_node,
        right_hand_control_node,
        left_hand_fk_node,
        right_hand_fk_node,
    ])
