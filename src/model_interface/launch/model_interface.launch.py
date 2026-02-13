import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='model_interface',
            executable='model_interface_node',
            name='model_interface_node',
            output='screen',
            parameters=[{
                'control_frequency': 30.0,
                'model_server_host': '81.68.132.224',
                'model_server_port': 18020,
                'calibration_path': '/root/workspace/legendvla-inference/examples/calibration_outputs',
                'camera_name': 'head',
                'ui_service_host': 'localhost',
                'ui_service_port': 8080,
                'data_frequency': 30.0,
                'state_horizon': 16,
                'state_stride': 2,
                'image_horizon': 1,
                'image_stride': 1,
                'action_execution_len': 6,
                'buffer_size': 200,
            }]
        )
    ])