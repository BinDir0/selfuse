import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='model_interface',
            executable='model_interface_node',
            name='model_interface_node',
            output='screen',
            parameters=[{
                'arm_frequency': '100.0',
                'hand_frequency': '80.0',
                'model_server_host': '81.68.132.224',
                'model_server_port': '18020',
                'camera_name': 'head',
                'calibration_path': '/root/workspace/legendvla-inference/examples/calibration_outputs',
                'assets_folder': '/root/workspace/legendvla-inference/assets'
            }]
        )
    ])