from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model_server_host', default_value='172.16.1.10'),
        DeclareLaunchArgument('model_server_port', default_value='8000'),
        DeclareLaunchArgument('action_execution_len', default_value='6'),
        DeclareLaunchArgument('duration_sec', default_value='30.0'),
        DeclareLaunchArgument('instruction', default_value='Pick up the test tube.'),
        Node(
            package='mock_robot_data',
            executable='mock_publisher_node',
            name='mock_robot_data',
            output='screen',
            parameters=[{
                'camera_name': 'head',
            }],
        ),
        Node(
            package='model_interface',
            executable='virtual_smoke_test_node',
            name='virtual_smoke_test_node',
            output='screen',
            on_exit=Shutdown(reason='virtual smoke monitor finished'),
            parameters=[{
                'ui_service_port': 18080,
                'instruction': LaunchConfiguration('instruction'),
                'mode': 'deploy',
                'duration_sec': ParameterValue(LaunchConfiguration('duration_sec'), value_type=float),
                'min_arm_msgs': 3,
                'min_hand_msgs': 3,
            }],
        ),
        Node(
            package='model_interface',
            executable='model_interface_node',
            name='model_interface_node',
            output='screen',
            parameters=[{
                'control_frequency': 30.0,
                'model_server_host': LaunchConfiguration('model_server_host'),
                'model_server_port': ParameterValue(LaunchConfiguration('model_server_port'), value_type=int),
                'calibration_path': '',
                'camera_name': 'head',
                'ui_service_host': 'localhost',
                'ui_service_port': 18080,
                'data_frequency': 30.0,
                'state_horizon': 1,
                'state_stride': 1,
                'image_horizon': 1,
                'image_stride': 1,
                'action_execution_len': ParameterValue(LaunchConfiguration('action_execution_len'), value_type=int),
                'buffer_size': 200,
                'debug_code': False,
                'do_resize': True,
                'require_depth': False,
                'auto_start': True,
                'auto_start_delay_sec': 1.0,
                'enable_keyboard': False,
                'use_mock_calibration': True,
            }],
        ),
    ])
