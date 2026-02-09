import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('model_interface')
    
    # 修改默认路径为你给出的路径结构
    default_calib_path = os.path.join(
        os.environ.get('HOME', '/root'), 
        'Documents/projects/LegendaryVLA/LegendVLA-Inference/examples/calibration_outputs'
    )

    # Arguments
    freq_arg = DeclareLaunchArgument('frequency', default_value='100.0')
    host_arg = DeclareLaunchArgument('model_server_host', default_value='0.0.0.0')
    port_arg = DeclareLaunchArgument('model_server_port', default_value='8000')
    
    # 这里的 calibration_path 应该指向包含 xiaozi1-head... 等文件夹的父目录
    calib_arg = DeclareLaunchArgument('calibration_path', default_value=default_calib_path)
    
    # camera_name 决定了去搜索 head 还是 chest 的标定文件
    cam_arg = DeclareLaunchArgument('camera_name', default_value='head', description="'head' or 'chest'")

    interface_node = Node(
        package='model_interface',
        executable='model_interface_node',
        name='model_interface_node',
        output='screen',
        parameters=[{
            'frequency': LaunchConfiguration('frequency'),
            'model_server_host': LaunchConfiguration('model_server_host'),
            'model_server_port': LaunchConfiguration('model_server_port'),
            'calibration_path': LaunchConfiguration('calibration_path'),
            'camera_name': LaunchConfiguration('camera_name'),
        }]
    )

    return LaunchDescription([
        freq_arg,
        host_arg,
        port_arg,
        calib_arg,
        cam_arg,
        interface_node
    ])