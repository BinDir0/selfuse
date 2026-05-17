from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mock_robot_data',
            executable='mock_publisher_node',
            name='mock_robot_data',
            output='screen',
            parameters=[{
                'camera_name': 'head',
                # pi0.5 EgoHands needs the breast view too (include_breast=True).
                'breast_camera_name': 'breast',
                'use_breast': True,
            }]
        )
    ])