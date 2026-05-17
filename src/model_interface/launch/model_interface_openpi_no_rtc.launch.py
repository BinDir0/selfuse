from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model_server_host', default_value='172.16.1.10'),
        DeclareLaunchArgument('model_server_port', default_value='8000'),
        DeclareLaunchArgument('calibration_path', default_value='/root/workspace/legendvla-inference/examples/calibration_outputs'),
        DeclareLaunchArgument('camera_name', default_value='head'),
        DeclareLaunchArgument('breast_camera_name', default_value='breast'),
        DeclareLaunchArgument('use_breast', default_value='true'),
        DeclareLaunchArgument('use_mock_calibration', default_value='false'),
        DeclareLaunchArgument('ui_service_host', default_value='localhost'),
        DeclareLaunchArgument('ui_service_port', default_value='8080'),
        DeclareLaunchArgument('action_execution_len', default_value='6'),
        DeclareLaunchArgument('require_depth', default_value='false'),
        DeclareLaunchArgument('debug_code', default_value='false'),
        Node(
            package='model_interface',
            executable='model_interface_node',
            name='model_interface_node',
            output='screen',
            parameters=[{
                'control_frequency': 30.0,
                # Point this to the machine running /root/openpi/scripts/serve_policy.py.
                'model_server_host': LaunchConfiguration('model_server_host'),
                'model_server_port': ParameterValue(LaunchConfiguration('model_server_port'), value_type=int),
                'calibration_path': LaunchConfiguration('calibration_path'),
                'camera_name': LaunchConfiguration('camera_name'),
                'breast_camera_name': LaunchConfiguration('breast_camera_name'),
                # pi0.5 EgoHands was trained with include_breast=True; serve must match.
                'use_breast': ParameterValue(LaunchConfiguration('use_breast'), value_type=bool),
                # L2 mock stage: identity-ish calibration so no result.npz needed.
                'use_mock_calibration': ParameterValue(LaunchConfiguration('use_mock_calibration'), value_type=bool),
                'ui_service_host': LaunchConfiguration('ui_service_host'),
                'ui_service_port': ParameterValue(LaunchConfiguration('ui_service_port'), value_type=int),
                'data_frequency': 30.0,
                # OpenPI pi0.5 EgoHands currently consumes only the latest state/image.
                'state_horizon': 1,
                'state_stride': 1,
                'image_horizon': 1,
                'image_stride': 1,
                # No RTC: execute a short prefix of the predicted chunk, then replan.
                'action_execution_len': ParameterValue(LaunchConfiguration('action_execution_len'), value_type=int),
                'buffer_size': 200,
                'debug_code': ParameterValue(LaunchConfiguration('debug_code'), value_type=bool),
                # pi0.5 config was trained on 224x224 WDS images.
                'do_resize': True,
                # The current OpenPI pi0.5 policy ignores depth. Send zero depth placeholder.
                'require_depth': ParameterValue(LaunchConfiguration('require_depth'), value_type=bool),
            }]
        )
    ])
