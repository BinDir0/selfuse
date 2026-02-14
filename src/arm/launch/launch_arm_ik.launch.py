#!/usr/bin/env python3
"""
MinkIK Node Launch File (Python version)
"""

'''
log: changed xml path: {xml_path} to {ik_xml_path}, same to arm_ik_node
delete xml path in launch.xml, use default mode
'''

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
    ik_enable_viewer_arg = DeclareLaunchArgument(
        'ik_enable_viewer',
        default_value='true',
        description='Enable MuJoCo viewer visualization'
    )
    
    ik_frequency_arg = DeclareLaunchArgument(
        'ik_frequency',
        default_value='100.0',
        description='IK solving frequency in Hz'
    )
    
    ik_solver_arg = DeclareLaunchArgument(
        'ik_solver',
        default_value='daqp',
        description='IK solver type'
    )
    
    ik_xml_path_arg = DeclareLaunchArgument(
        'ik_xml_path',
        default_value=xml_path,
        description='XML path for MinkIK'
    )
    
    # MinkIK Node
    arm_ik_node = Node(
        package='arm',
        executable='arm_ik_node',
        name='arm_ik_node',
        output='screen',
        prefix=f'{venv_python_path}',
        parameters=[
            {
                'enable_viewer': LaunchConfiguration('ik_enable_viewer'),
                'frequency': LaunchConfiguration('ik_frequency'),
                'solver': LaunchConfiguration('ik_solver'),
                'ik_xml_path': LaunchConfiguration('ik_xml_path'),
            }
        ],
        # Topic remappings (commented out as in original XML)
        # remappings=[
        #     ('left_target_pose', '/mink_ik/left_target_pose'),
        #     ('right_target_pose', '/mink_ik/right_target_pose'),
        #     ('joint_states', '/mink_ik/joint_states'),
        # ]
    )
    
    return LaunchDescription([
        ik_enable_viewer_arg,
        ik_frequency_arg,
        ik_solver_arg,
        ik_xml_path_arg,
        arm_ik_node,
    ]) 


"""
VENVPYTHONPATH=$(which python) ros2 launch vive_arm_control launch_mink_ik.launch.py
"""