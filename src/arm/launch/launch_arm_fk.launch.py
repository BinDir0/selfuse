#!/usr/bin/env python3
"""
MinkFK Node Launch File (Python version)
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# get ['VENVPYTHONPATH'] from env variables
venv_python_path = os.environ['VENVPYTHONPATH']

if venv_python_path is None:
    # venv_python_path = "/root/workspace/a2d-tele/.venv/bin/python3"
    venv_python_path = "/usr/bin/python3"

    print(f"VENVPYTHONPATH is not set, using default path: {venv_python_path}")
else:
    print(f"VENVPYTHONPATH is set to: {venv_python_path}")


def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('arm')
    xml_path = os.path.join(os.path.dirname(__file__), "../../../assets/PsiRobot_DC_02_OnlyArm/meshes/psi_robot_scene_transformed.xml")
    xml_path = os.path.abspath(xml_path)
    
    # Launch arguments
    fk_arm_side_arg = DeclareLaunchArgument(
        'fk_arm_side',
        default_value='left',
        description='FK arm side, left or right'
    )

    fk_frequency_arg = DeclareLaunchArgument(
        'fk_frequency',
        default_value='100.0',
        description='FK solving frequency in Hz'
    )
    
    fk_xml_path_arg = DeclareLaunchArgument(
        'fk_xml_path',
        default_value=xml_path,
        description='XML path for MinkFK'
    )
    
    # MinkFK Node
    arm_fk_node = Node(
        package='arm',
        executable='arm_fk_node',
        name=['arm_fk_node_', LaunchConfiguration('fk_arm_side')],
        output='screen',
        prefix=f'{venv_python_path}',
        parameters=[
            {
                'arm_side': LaunchConfiguration('fk_arm_side'),
                'frequency': LaunchConfiguration('fk_frequency'),
                'xml_path': LaunchConfiguration('fk_xml_path'),
            }
        ],
    )
    
    return LaunchDescription([
        fk_arm_side_arg,
        fk_frequency_arg,
        fk_xml_path_arg,
        arm_fk_node,
    ]) 


"""
VENVPYTHONPATH=$(which python) ros2 launch vive_arm_control launch_mink_fk.launch.py
"""