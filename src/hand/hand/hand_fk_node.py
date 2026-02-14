#!/usr/bin/env python3
"""
Hand FK Node - 手部正运动学节点
根据架构图：
- 频率: 80Hz
- 输入: /state/{left,right}_hand/joints (JointState) - hand states (joints)
- 输出: /state/{left,right}_hand/keypoints (PoseArray) - hand keypoints (keypoints in wrist frame)

实现说明：
- 内部实现HandFK类（方法与hand_fk.py相同）
- 应用手部安装变换 (wrist → hand_base)
- 输出坐标系：wrist frame
- 功能：接收手指关节角度，计算并输出指尖关键点3D位姿
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading
from pathlib import Path
import mujoco


class HandFKSolver:
    """
    Internal Hand FK Solver (same functionality as hand_fk.py)
    
    Features:
    - Compute fingertip poses from joint angles
    - Support left/right hand
    - Handle mimic joints
    - Support normalized input [0-1]
    """
    
    def __init__(self, hand_type='left', mjcf_path=None):
        """
        Initialize Hand FK Solver
        
        Args:
            hand_type: 'left' or 'right'
            mjcf_path: Path to hand MJCF file (if None, use default path)
        """
        self.hand_type = hand_type
        
        # Load MJCF model
        if mjcf_path is None:
            # Default path: repository assets directory
            # Current file is at: src/hand/hand/hand_fk_node.py
            # Assets directory is at: assets/ruiyan_hand/...
            repo_root = Path(__file__).parent.parent.parent.parent
            hand_prefix = 'Left' if hand_type == 'left' else 'Right'
            mjcf_dir = (
                repo_root
                / 'assets'
                / 'ruiyan_hand'
                / 'InspiredHand_RuiYan'
                / '0611_v1.4'
                / 'Version_3.0'
                / f'RuiYan_Hand_{hand_prefix}_Mimic'
                / 'meshes'
            )
            mjcf_path = mjcf_dir / f'RuiYan_Hand_{hand_prefix}_Mimic_scene.xml'
        
        if not Path(mjcf_path).exists():
            raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")
        
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.data_lock = threading.Lock()
        
        # Setup joint and site information
        self._setup_joint_info()
        self._setup_site_info()
        
        # Warm up simulation
        self._warm_up_sim()
    
    def _setup_joint_info(self):
        """Setup active joint information (same as hand_fk.py)"""
        # Get all joints
        all_joints = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) 
                     for i in range(self.model.njnt)]
        
        # Define active joints (6 actuated joints)
        prefix = 'hand1' if self.hand_type == 'left' else 'hand2'
        active_joints = [
            f'{prefix}_joint_link_1_1',  # Thumb joint 1
            f'{prefix}_joint_link_1_2',  # Thumb joint 2
            f'{prefix}_joint_link_2_1',  # Index finger
            f'{prefix}_joint_link_3_1',  # Middle finger
            f'{prefix}_joint_link_4_1',  # Ring finger
            f'{prefix}_joint_link_5_1',  # Pinky finger
        ]
        
        # Keep only existing active joints
        self.joint_names = [j for j in active_joints if j in all_joints]
        
        # Store joint limits (for normalization)
        self.joint_limits = {}
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_adr = self.model.jnt_qposadr[joint_id]
            lower = self.model.jnt_range[joint_id][0]
            upper = self.model.jnt_range[joint_id][1]
            self.joint_limits[joint_name] = {
                'lower': lower,
                'upper': upper,
                'range': upper - lower
            }
    
    def _setup_site_info(self):
        """Setup fingertip site information (same as hand_fk.py)"""
        side_prefix = self.hand_type  # 'left' or 'right'
        
        # 5 fingertip site names
        self.fingertip_sites = [
            f'{side_prefix}_thumb_tip',
            f'{side_prefix}_index_tip',
            f'{side_prefix}_middle_tip',
            f'{side_prefix}_ring_tip',
            f'{side_prefix}_pinky_tip',
        ]
        
        # Get site IDs
        self.site_ids = {}
        for site_name in self.fingertip_sites:
            try:
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                self.site_ids[site_name] = site_id
            except:
                pass  # Site might not exist
    
    def _normalized_to_radians(self, normalized_positions):
        """
        Convert normalized values [0-1] to radians (same as hand_fk.py)
        
        Args:
            normalized_positions: Normalized joint angles [0-1]
        
        Returns:
            Joint angles in radians
        """
        radians = np.zeros_like(normalized_positions)
        for i, joint_name in enumerate(self.joint_names):
            limits = self.joint_limits[joint_name]
            radians[i] = limits['lower'] + normalized_positions[i] * limits['range']
        return radians
    
    def _apply_mimic_joints(self):
        """
        Manually apply mimic joint constraints (same as hand_fk.py)
        
        Critical: MuJoCo equality constraints only work in dynamics simulation.
        For pure kinematics (FK/IK), we must manually apply them.
        """
        hand_prefix = 'hand1' if self.hand_type == 'left' else 'hand2'
        
        # Mimic relations extracted from URDF
        # 与 hand_ik_solver.py / psirobot_visualizer/hand_ik.py 保持一致
        mimic_relations = [
            (f'{hand_prefix}_joint_link_1_2', f'{hand_prefix}_joint_link_1_3', 0.325, 0.0),  # Thumb
            (f'{hand_prefix}_joint_link_2_1', f'{hand_prefix}_joint_link_2_2', 1.0, 0.0),    # Index
            (f'{hand_prefix}_joint_link_3_1', f'{hand_prefix}_joint_link_3_2', 1.0, 0.0),    # Middle
            (f'{hand_prefix}_joint_link_4_1', f'{hand_prefix}_joint_link_4_2', 1.0, 0.0),    # Ring
            (f'{hand_prefix}_joint_link_5_1', f'{hand_prefix}_joint_link_5_2', 1.0, 0.0),    # Pinky
        ]
        
        with self.data_lock:
            for leader_name, follower_name, multiplier, offset in mimic_relations:
                try:
                    leader_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, leader_name)
                    follower_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, follower_name)
                    
                    # 使用 jnt_qposadr 获取正确的 qpos 地址
                    # (与 hand_ik_solver.py 保持一致，不能直接用 joint_id 作为 qpos 索引)
                    leader_addr = self.model.jnt_qposadr[leader_id]
                    follower_addr = self.model.jnt_qposadr[follower_id]
                    
                    # Apply constraint: follower = offset + multiplier * leader
                    self.data.qpos[follower_addr] = offset + multiplier * self.data.qpos[leader_addr]
                except:
                    pass  # Some joints might not exist
    
    def _warm_up_sim(self):
        """Warm up simulation (same as hand_fk.py)"""
        for _ in range(10):
            with self.data_lock:
                mujoco.mj_forward(self.model, self.data)
    
    def compute_fk(self, joint_positions, use_normalized=False):
        """
        Compute forward kinematics (same as hand_fk.py)
        
        Args:
            joint_positions: Joint angles array
                           - If use_normalized=False: radians
                           - If use_normalized=True: normalized [0-1]
                           Length should be 6 (active joints)
            use_normalized: Whether input is normalized
        
        Returns:
            dict: Fingertip poses
                {
                    'thumb': {'pos': [x,y,z], 'quat': [x,y,z,w], 'rot': mat},
                    'index': {'pos': [x,y,z], 'quat': [x,y,z,w], 'rot': mat},
                    ...
                }
        """
        joint_positions = np.array(joint_positions)
        
        if len(joint_positions) != len(self.joint_names):
            raise ValueError(
                f"Joint count mismatch: expected {len(self.joint_names)}, got {len(joint_positions)}"
            )
        
        # Convert normalized to radians if needed
        if use_normalized:
            joint_positions = self._normalized_to_radians(joint_positions)
        
        # Update joint positions
        with self.data_lock:
            # Set active joint positions
            for i, joint_name in enumerate(self.joint_names):
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_adr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_adr] = joint_positions[i]
        
        # Apply mimic joints constraints (critical step!)
        self._apply_mimic_joints()
        
        with self.data_lock:
            # Execute forward kinematics
            mujoco.mj_forward(self.model, self.data)
            
            # Extract fingertip poses
            results = {}
            finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
            
            for finger_name, site_name in zip(finger_names, self.fingertip_sites):
                if site_name in self.site_ids:
                    site_id = self.site_ids[site_name]
                    
                    # Get site position
                    pos = self.data.site(site_id).xpos.copy()
                    
                    # Get site orientation (rotation matrix -> quaternion)
                    xmat = self.data.site(site_id).xmat.copy().reshape(3, 3)
                    quat = R.from_matrix(xmat).as_quat()  # [x, y, z, w]
                    
                    results[finger_name] = {
                        'pos': pos,
                        'quat': quat,
                        'rot': xmat
                    }
        
        return results


class HandFKNode(Node):
    """
    灵巧手正运动学节点
    输入：手指关节角度 (6-DOF, 归一化 [0-1])
    输出：指尖位姿 (in Wrist Frame)
    """
    
    def __init__(self):
        super().__init__('hand_fk_node')
        
        # ========== 参数声明 ==========
        self.declare_parameter('hand_side', 'left')  # 'left' or 'right'
        self.declare_parameter('frequency', 80.0)    # 80Hz
        self.declare_parameter('mjcf_path', '')      # Hand MJCF path
        
        # 获取参数
        self.hand_side = self.get_parameter('hand_side').get_parameter_value().string_value
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        mjcf_path = self.get_parameter('mjcf_path').get_parameter_value().string_value
        
        # Default MJCF path if not specified
        if mjcf_path == '':
            mjcf_path = None  # HandFKSolver will use default path
        
        # ========== 初始化内部Hand FK Solver ==========
        try:
            self.hand_fk = HandFKSolver(hand_type=self.hand_side, mjcf_path=mjcf_path)
            self.get_logger().info(f'✅ Hand FK Solver initialized successfully')
            self.get_logger().info(f'   Active joints: {len(self.hand_fk.joint_names)}')
            self.get_logger().info(f'   Fingertip sites: {len(self.hand_fk.fingertip_sites)}')
        except Exception as e:
            self.get_logger().error(f'❌ Failed to initialize Hand FK Solver: {e}')
            raise
        
        # ========== 手部安装变换 (hand_base → wrist) ==========
        # 重要: 此变换必须与 Hand IK Node 中的 _T_wrist_hand_base 完全一致!
        # FK: T_wrist_hand_base 将 hand_base frame → wrist frame (输出给 Model Interface)
        # IK: inv(T_wrist_hand_base) 将 wrist frame → hand_base frame (输入给 IK solver)
        # 当前配置: Z 轴 180° 旋转
        self.T_wrist_hand_base = np.eye(4)
        self.T_wrist_hand_base[:3, :3] = R.from_euler('z', 180, degrees=True).as_matrix()
        # 如果有偏移，可以设置 (同时需修改 hand_ik_node.py 中的对应变换!):
        # self.T_wrist_hand_base[:3, 3] = np.array([0, 0, 0.05])
        
        self.get_logger().info(f'Hand installation transform: Z-axis 180° rotation')
        
        # ========== ROS2接口 ==========
        # 订阅：手指关节状态
        self.state_sub = self.create_subscription(
            JointState,
            f'/state/{self.hand_side}_hand/joints',
            self.state_callback,
            10
        )
        
        # 发布：指尖关键点 (PoseArray格式，包含5个指尖的3D位姿)
        self.keypoints_pub = self.create_publisher(
            PoseArray,
            f'/state/{self.hand_side}_hand/keypoints',
            10
        )
        
        # 当前关节状态
        self.current_joints = None
        
        # ========== 80Hz定时器 ==========
        timer_period = 1.0 / self.frequency
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # ========== 日志输出 ==========
        self.get_logger().info('='*60)
        self.get_logger().info('✅ Hand FK Node initialized')
        self.get_logger().info(f'   Hand: {self.hand_side}')
        self.get_logger().info(f'   Frequency: {self.frequency} Hz')
        self.get_logger().info(f'   Input: /state/{self.hand_side}_hand/joints (JointState)')
        self.get_logger().info(f'   Output: /state/{self.hand_side}_hand/keypoints (PoseArray)')
        self.get_logger().info(f'   Output Frame: {self.hand_side}_wrist')
        self.get_logger().info('='*60)
    
    def state_callback(self, msg):
        """
        接收手指关节状态
        """
        if msg.position is None or len(msg.position) < 6:
            self.get_logger().warn(f'Invalid hand joint state: expected 6 joints, got {len(msg.position) if msg.position else 0}')
            return
        
        self.current_joints = np.array(msg.position[:6])
    
    def timer_callback(self):
        """
        80Hz FK计算和发布
        功能：接收关节角度，计算指尖关键点3D坐标
        """
        if self.current_joints is None:
            return
        
        try:
            # ========== Hand FK计算 (内部HandFKSolver) ==========
            # 调用内部HandFKSolver的compute_fk方法
            fingertip_results = self.hand_fk.compute_fk(
                self.current_joints,
                use_normalized=True  # 假设输入是归一化的 [0-1]
            )
            
            # ========== 转换到wrist坐标系 ==========
            fingertip_poses_wrist = {}
            
            for finger_name, finger_data in fingertip_results.items():
                # 指尖在hand_base坐标系
                pos_hand = finger_data['pos']
                mat_hand = finger_data['rot']
                
                # 构建变换矩阵
                T_hand_fingertip = np.eye(4)
                T_hand_fingertip[:3, :3] = mat_hand
                T_hand_fingertip[:3, 3] = pos_hand
                
                # 应用手部安装变换：wrist → hand_base → fingertip
                T_wrist_fingertip = self.T_wrist_hand_base @ T_hand_fingertip
                
                # 提取位姿（位置和姿态）
                pos_wrist = T_wrist_fingertip[:3, 3]
                mat_wrist = T_wrist_fingertip[:3, :3]
                quat_wrist = R.from_matrix(mat_wrist).as_quat()  # [x,y,z,w]
                
                fingertip_poses_wrist[finger_name] = {
                    'pos': pos_wrist,
                    'quat': quat_wrist
                }
            
            # ========== 发布指尖关键点 (JointState格式) ==========
            self._publish_fingertip_keypoints(fingertip_poses_wrist)
            
        except Exception as e:
            self.get_logger().error(f'Hand FK computation error: {e}')
    
    def _publish_fingertip_keypoints(self, fingertip_poses):
        """
        发布指尖关键点 (PoseArray格式)
        
        格式:
        - header.frame_id: "{side}_wrist"
        - poses: 5个Pose对象，按 thumb, index, middle, ring, pinky 顺序
          - Pose.position: Point(x, y, z) 指尖3D坐标
          - Pose.orientation: Quaternion(x, y, z, w) 指尖朝向
        
        Args:
            fingertip_poses: dict，包含5个指尖的位姿
                {'thumb': {'pos': [x,y,z], 'quat': [x,y,z,w]}, ...}
        """
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"{self.hand_side}_wrist"
        
        # 按固定顺序排列指尖
        finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        for finger_name in finger_order:
            pose = Pose()
            
            if finger_name in fingertip_poses:
                pos = fingertip_poses[finger_name]['pos']
                quat = fingertip_poses[finger_name]['quat']
                
                # 设置位置
                pose.position = Point(
                    x=float(pos[0]),
                    y=float(pos[1]),
                    z=float(pos[2])
                )
                
                # 设置姿态 (scipy输出格式: [x, y, z, w])
                pose.orientation = Quaternion(
                    x=float(quat[0]),
                    y=float(quat[1]),
                    z=float(quat[2]),
                    w=float(quat[3])
                )
            else:
                # 缺失的指尖: 零位置 + identity quaternion
                pose.position = Point(x=0.0, y=0.0, z=0.0)
                pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            
            msg.poses.append(pose)
        
        # 发布
        self.keypoints_pub.publish(msg)


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    node = None
    try:
        node = HandFKNode()
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

