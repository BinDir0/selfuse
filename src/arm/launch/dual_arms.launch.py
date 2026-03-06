#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    arm_ik_node = Node(
        package='arm',
        executable='arm_ik_node.py',
        name='arm_ik_node',
        output='screen',
        parameters=[{
            'frequency': 100.0,
            'enable_viewer': True,
        }]
    )
    
    left_arm_control_node = Node(
        package='arm',
        executable='arm_control_node.py',
        name='left_arm_control_node',
        output='screen',
        parameters=[{
            'arm_side': 'left',
            'frequency': 100.0,
        }]
    )

    right_arm_control_node = Node(
        package='arm',
        executable='arm_control_node.py',
        name='right_arm_control_node',
        output='screen',
        parameters=[{
            'arm_side': 'right',
            'frequency': 100.0,
        }]
    )
    
    return LaunchDescription([
        arm_ik_node,
        left_arm_control_node,
        right_arm_control_node,
    ])
