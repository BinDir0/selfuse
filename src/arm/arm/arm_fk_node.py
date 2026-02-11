#!/usr/bin/env python3
"""
Arm FK Node - 机械臂正运动学节点
根据架构图：
- 频率: 100Hz
- 输入: /state/{left,right}_arm/joints (JointState) - arm states (joints)
- 输出: /state/{left,right}_arm/wrist_pose (PoseStamped in arm_base frame)

实现说明：
- 核心FK计算100%遵循mj-controller (xiaozi_mink_fk_node.py)
- 坐标变换参考visualize_psirobot_with_rgbd_calib.py
- 输出坐标系：arm_base frame (arm1_link0 或 arm2_link0)
- 输出位置：connector顶端（手部安装点）
- 输出姿态：wrist TCP原始姿态（保持不变）
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
import threading
from pathlib import Path

# Import connector configuration
from arm.connector_config import (
    LEFT_CONNECTOR_OFFSET_1_1, LEFT_CONNECTOR_RPY_1_1,
    LEFT_CONNECTOR_OFFSET_1_2, LEFT_CONNECTOR_RPY_1_2,
    RIGHT_CONNECTOR_OFFSET_2_1, RIGHT_CONNECTOR_RPY_2_1,
    RIGHT_CONNECTOR_OFFSET_2_2, RIGHT_CONNECTOR_RPY_2_2,
    CONNECTOR_HEIGHT
)


class ArmFKNode(Node):
    """
    机械臂正运动学节点
    输入：关节角度 (7-DOF)
    输出：Wrist位姿 (in Arm Base Frame)
    """
    
    def __init__(self):
        super().__init__('arm_fk_node')
        
        # ========== 参数声明 ==========
        self.declare_parameter('arm_side', 'left')   # 'left' or 'right'
        self.declare_parameter('frequency', 100.0)   # 100Hz
        self.declare_parameter('xml_path', '')       # MuJoCo XML path
        
        # 获取参数
        self.arm_side = self.get_parameter('arm_side').get_parameter_value().string_value
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        xml_path = self.get_parameter('xml_path').get_parameter_value().string_value
        
        # 默认XML路径
        if xml_path == '':
            # Use repository assets directory
            # Current file is at: src/arm/arm/arm_fk_node.py
            # Assets directory is at: assets/PsiRobot_DC_02_OnlyArm/meshes/
            repo_root = Path(__file__).parent.parent.parent.parent
            xml_path = str(
                repo_root
                / 'assets'
                / 'PsiRobot_DC_02_OnlyArm'
                / 'meshes'
                / 'psi_robot_scene_transformed.xml'
            )
            self.get_logger().info(f'Using default XML path: {xml_path}')
        
        # ========== 加载MuJoCo模型 (遵循mj-controller) ==========
        self.get_logger().info(f'Loading MuJoCo model from: {xml_path}')
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
            self.data_lock = threading.Lock()
            self.get_logger().info(f'✅ MuJoCo model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'❌ Failed to load MuJoCo model: {e}')
            raise
        
        # ========== 获取关键ID ==========
        # Wrist site ID (遵循mj-controller)
        wrist_site_name = f"{self.arm_side}_wrist"
        try:
            self.wrist_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, wrist_site_name
            )
            self.get_logger().info(f'✅ Found wrist site: {wrist_site_name} (ID: {self.wrist_site_id})')
        except:
            self.get_logger().error(f'❌ Wrist site not found: {wrist_site_name}')
            raise
        
        # Arm base body ID (用于剥离XML world偏移)
        if self.arm_side == 'left':
            base_body_name = "arm1_link0"
        else:
            base_body_name = "arm2_link0"
        
        try:
            self.arm_base_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name
            )
            self.get_logger().info(f'✅ Found arm base body: {base_body_name} (ID: {self.arm_base_body_id})')
        except:
            self.get_logger().error(f'❌ Arm base body not found: {base_body_name}')
            raise
        
        # ========== ROS2接口 ==========
        # 订阅：关节状态
        self.state_sub = self.create_subscription(
            JointState,
            f'/state/{self.arm_side}_arm/joints',
            self.state_callback,
            10
        )
        
        # 发布：腕部位姿 (在arm_base坐标系)
        self.wrist_pose_pub = self.create_publisher(
            PoseStamped,
            f'/state/{self.arm_side}_arm/wrist_pose',
            10
        )
        
        # 当前关节状态
        self.current_joints = None
        
        # ========== 100Hz定时器 ==========
        timer_period = 1.0 / self.frequency
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # ========== 加载Connector参数 ==========
        if self.arm_side == 'left':
            self.connector_offset_1 = LEFT_CONNECTOR_OFFSET_1_1
            self.connector_rpy_1 = LEFT_CONNECTOR_RPY_1_1
            self.connector_offset_2 = LEFT_CONNECTOR_OFFSET_1_2
            self.connector_rpy_2 = LEFT_CONNECTOR_RPY_1_2
        else:
            self.connector_offset_1 = RIGHT_CONNECTOR_OFFSET_2_1
            self.connector_rpy_1 = RIGHT_CONNECTOR_RPY_2_1
            self.connector_offset_2 = RIGHT_CONNECTOR_OFFSET_2_2
            self.connector_rpy_2 = RIGHT_CONNECTOR_RPY_2_2
        
        self.hand_z_offset = CONNECTOR_HEIGHT  # 0.011m (hand_z_offset from visualizer)
        
        self.get_logger().info(f'✅ Connector parameters loaded for {self.arm_side} arm')
        
        # ========== 日志输出 ==========
        self.get_logger().info('='*60)
        self.get_logger().info('✅ Arm FK Node initialized')
        self.get_logger().info(f'   Arm: {self.arm_side}')
        self.get_logger().info(f'   Frequency: {self.frequency} Hz')
        self.get_logger().info(f'   Input: /state/{self.arm_side}_arm/joints (JointState)')
        self.get_logger().info(f'   Output: /state/{self.arm_side}_arm/wrist_pose (PoseStamped)')
        self.get_logger().info(f'   Output Position: Connector top (hand base connection point)')
        self.get_logger().info(f'   Output Rotation: Wrist TCP orientation (unchanged)')
        self.get_logger().info(f'   Output Frame: {base_body_name} (Arm Base)')
        self.get_logger().info('='*60)
    
    def state_callback(self, msg):
        """
        接收关节状态 (遵循mj-controller的回调模式)
        """
        if msg.position is None or len(msg.position) < 7:
            self.get_logger().warn(f'Invalid joint state: expected 7 joints, got {len(msg.position) if msg.position else 0}')
            return
        
        self.current_joints = np.array(msg.position[:7])
    
    def timer_callback(self):
        """
        100Hz FK计算和发布
        核心实现100%遵循mj-controller的xiaozi_mink_fk_node.py
        """
        if self.current_joints is None:
            return
        
        try:
            # ========== 核心FK计算 (100%遵循mj-controller) ==========
            with self.data_lock:
                # Step 1: 设置关节角度 (遵循mj-controller)
                if self.arm_side == 'left':
                    self.data.qpos[:7] = self.current_joints
                else:
                    self.data.qpos[7:14] = self.current_joints
                
                # Step 2: 执行MuJoCo FK (遵循mj-controller)
                mujoco.mj_forward(self.model, self.data)
                
                # Step 3: 提取wrist在XML world中的位姿 (遵循mj-controller)
                wrist_pos_xml = self.data.site(self.wrist_site_id).xpos.copy()
                wrist_mat_xml = self.data.site(self.wrist_site_id).xmat.copy().reshape(3, 3)
            
            # ========== 坐标变换：剥离XML world偏移 (参考visualizer) ==========
            wrist_pos_base, wrist_mat_base = self._transform_to_base_frame(
                wrist_pos_xml, wrist_mat_xml
            )
            
            # Step 4: 旋转矩阵 → 四元数 (遵循mj-controller)
            wrist_quat_base = R.from_matrix(wrist_mat_base).as_quat()  # [x,y,z,w]
            
            # ========== 计算Connector顶端位置 (新增) ==========
            connector_top_pos = self._compute_connector_top_position(
                wrist_pos_base, wrist_mat_base
            )
            
            # ========== 发布位姿 ==========
            # Position: connector顶端（手部安装点）
            # Rotation: wrist TCP姿态（保持不变）
            self._publish_wrist_pose(connector_top_pos, wrist_quat_base)
            
        except Exception as e:
            self.get_logger().error(f'FK computation error: {e}')
    
    def _transform_to_base_frame(self, pos_xml, mat_xml):
        """
        将FK结果从XML world坐标系转换到arm_base坐标系
        参考：visualize_psirobot_with_rgbd_calib.py的get_world_transform思想
        
        核心思想：
        1. FK结果包含XML的worldbody位置偏移
        2. 通过计算相对于base的局部变换，剥离XML的位置影响
        
        Args:
            pos_xml: wrist在XML world中的位置
            mat_xml: wrist在XML world中的旋转矩阵
        
        Returns:
            pos_base, mat_base: wrist在arm_base坐标系中的位置和姿态
        """
        # 获取base在XML world中的变换
        base_pos_xml = self.data.xpos[self.arm_base_body_id].copy()
        base_mat_xml = self.data.xmat[self.arm_base_body_id].copy().reshape(3, 3)
        
        # 构建4x4变换矩阵
        T_xml_base = np.eye(4)
        T_xml_base[:3, :3] = base_mat_xml
        T_xml_base[:3, 3] = base_pos_xml
        
        T_xml_wrist = np.eye(4)
        T_xml_wrist[:3, :3] = mat_xml
        T_xml_wrist[:3, 3] = pos_xml
        
        # 计算wrist相对于base的变换 (剥离XML world偏移)
        T_base_wrist = np.linalg.inv(T_xml_base) @ T_xml_wrist
        
        # 提取位姿
        pos_base = T_base_wrist[:3, 3]
        mat_base = T_base_wrist[:3, :3]
        
        return pos_base, mat_base
    
    def _compute_connector_top_position(self, wrist_pos, wrist_mat):
        """
        计算connector顶端位置（手部安装点）
        参考：visualize_psirobot_with_rgbd_calib.py的connector计算逻辑
        
        核心思想：
        1. 从wrist TCP开始，依次应用两个connector的变换
        2. 最后沿Z轴偏移得到手部安装点
        3. 姿态使用wrist原始姿态（不改变）
        
        Args:
            wrist_pos: wrist在arm_base中的位置
            wrist_mat: wrist在arm_base中的旋转矩阵
        
        Returns:
            connector_top_pos: connector顶端位置（手部安装点）
        """
        # Connector 1 变换
        rot_1 = R.from_euler('xyz', self.connector_rpy_1).as_matrix()
        pos_1 = wrist_pos + wrist_mat @ self.connector_offset_1
        mat_1 = wrist_mat @ rot_1
        
        # Connector 2 变换
        rot_2 = R.from_euler('xyz', self.connector_rpy_2).as_matrix()
        pos_2 = pos_1 + mat_1 @ self.connector_offset_2
        mat_2 = mat_1 @ rot_2
        
        # Hand Base位置（沿connector 2的Z轴偏移）
        connector_top_pos = pos_2 + mat_2[:, 2] * self.hand_z_offset
        
        return connector_top_pos
    
    def _publish_wrist_pose(self, position, quaternion):
        """
        发布wrist位姿
        
        注意：
        - position: connector顶端位置（手部安装点）
        - quaternion: wrist TCP原始姿态（保持不变）
        
        Args:
            position: [x, y, z] connector顶端在arm_base frame
            quaternion: [x, y, z, w] wrist TCP姿态
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = f"{self.arm_side}_arm_base"
        
        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])
        
        pose_msg.pose.orientation.x = float(quaternion[0])
        pose_msg.pose.orientation.y = float(quaternion[1])
        pose_msg.pose.orientation.z = float(quaternion[2])
        pose_msg.pose.orientation.w = float(quaternion[3])
        
        self.wrist_pose_pub.publish(pose_msg)


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    node = None
    try:
        node = ArmFKNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            node.get_logger().error(f'节点运行错误: {str(e)}')
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

