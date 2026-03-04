#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import String
import numpy as np
import mujoco
from pathlib import Path
import threading
from scipy.spatial.transform import Rotation as R
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor


class HandFKSolver:
    """灵巧手正运动学求解器 - 针对 80Hz 实时反馈优化"""
    def __init__(self, hand_type='left'):
        self.hand_type = hand_type
        self.data_lock = threading.Lock()

        # 路径处理
        script_dir = Path(__file__).parent.parent
        hand_prefix = 'Left' if hand_type == 'left' else 'Right'
        mjcf_file = script_dir / 'resource' / 'RuiYan' / '0611_v1.4' / 'Version_3.0' / f'RuiYan_Hand_{hand_prefix}_Mimic' / 'meshes' / f'RuiYan_Hand_{hand_prefix}_Mimic_scene.xml'

        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_file))
        self.data = mujoco.MjData(self.model)

        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        # 预记录核心信息，避免循环中查找字符串
        self._setup_joint_info()
        self._record_joint_metadata()
        self._setup_site_info()
        
        # 初始计算一次
        with self.data_lock:
            mujoco.mj_forward(self.model, self.data)

    def _setup_joint_info(self):
        """记录主动关节名称"""
        all_joints = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]
        prefix = 'hand1' if self.hand_type == 'left' else 'hand2'
        active_candidates = [
            f'{prefix}_joint_link_1_1', f'{prefix}_joint_link_1_2',
            f'{prefix}_joint_link_2_1', f'{prefix}_joint_link_3_1',
            f'{prefix}_joint_link_4_1', f'{prefix}_joint_link_5_1',
        ]
        self.joint_names = [j for j in active_candidates if j in all_joints]

    def _record_joint_metadata(self):
        """性能优化：预先缓存关节内存地址和限位范围"""
        self.joint_metadata = []
        for name in self.joint_names:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            lower, upper = self.model.jnt_range[jnt_id]
            
            self.joint_metadata.append({
                'name': name,
                'adr': qpos_adr,
                'low': lower,
                'range': (upper - lower) if (upper - lower) != 0 else 1.0
            })
        print(f"[FK Solver] {self.hand_type.upper()} 预记录 {len(self.joint_metadata)} 个主动关节")

    def _setup_site_info(self):
        """缓存指尖 site 的 ID"""
        side_prefix = self.hand_type
        self.fingertip_sites = [f'{side_prefix}_{name}_tip' for name in self.finger_names]
        self.site_ids_list = []
        for site_name in self.fingertip_sites:
            try:
                self.site_ids_list.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name))
            except:
                pass

    def _get_mimic_relations(self):
        """定义联动关节关系"""
        p = 'hand1' if self.hand_type == 'left' else 'hand2'
        return [
            (f'{p}_joint_link_1_2', f'{p}_joint_link_1_3', 1.675, 0.0),
            (f'{p}_joint_link_2_1', f'{p}_joint_link_2_2', 1.0, 0.0),
            (f'{p}_joint_link_3_1', f'{p}_joint_link_3_2', 1.0, 0.0),
            (f'{p}_joint_link_4_1', f'{p}_joint_link_4_2', 1.0, 0.0),
            (f'{p}_joint_link_5_1', f'{p}_joint_link_5_2', 1.0, 0.0),
        ]

    def _apply_mimic_joints(self):
        """手动应用联动约束到 qpos"""
        with self.data_lock:
            for leader_name, follower_name, multiplier, offset in self._get_mimic_relations():
                try:
                    l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, leader_name)
                    f_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, follower_name)
                    l_adr = self.model.jnt_qposadr[l_id]
                    f_adr = self.model.jnt_qposadr[f_id]
                    self.data.qpos[f_adr] = offset + multiplier * self.data.qpos[l_adr]
                except:
                    pass

    def compute_fk(self, normalized_positions):
        """
        根据归一化关节角度计算 FK
        normalized_positions: 长度为 6 的数组 (0.0~1.0)
        """
        if len(normalized_positions) != len(self.joint_metadata):
            return None

        with self.data_lock:
            # 1. 映射归一化值到物理弧度
            for i, m in enumerate(self.joint_metadata):
                val_rad = m['low'] + normalized_positions[i] * m['range']
                self.data.qpos[m['adr']] = val_rad
        
        # 2. 应用联动
        self._apply_mimic_joints()
        
        # 3. 前向计算
        with self.data_lock:
            mujoco.mj_forward(self.model, self.data)
            
            # 4. 提取指尖 Site 位姿
            results = []
            for s_id in self.site_ids_list:
                pos = self.data.site(s_id).xpos.copy()
                xmat = self.data.site(s_id).xmat.copy().reshape(3, 3)
                quat = R.from_matrix(xmat).as_quat() # [x, y, z, w]
                results.append((pos, quat))
        return results


class HandFKNode(Node):
    def __init__(self):
        super().__init__('hand_fk_node')

        self.declare_parameter('hand_side', 'left')
        self.declare_parameter('frequency', 80.0)
        self.hand_side = self.get_parameter('hand_side').value
        self.frequency = self.get_parameter('frequency').value

        self.get_logger().info(f"Initializing {self.hand_side.upper()} Hand FK Node")

        # 初始化 FK Solver
        self.fk_solver = HandFKSolver(hand_type=self.hand_side)

        # 回调组：确保实时性
        cb_group = MutuallyExclusiveCallbackGroup()

        # 订阅：当前手的关节状态 (由 Hand Control Node 发出，position 为 0~1)
        self.sub_joint_states = self.create_subscription(
            JointState,
            f'/state/{self.hand_side}_hand/joints',
            self.joint_state_callback,
            10,
            callback_group=cb_group
        )

        # 发布：当前指尖的 PoseArray (发布给 Interface/Model)
        self.pub_keypoints = self.create_publisher(
            PoseArray,
            f'/state/{self.hand_side}_hand/keypoints',
            10,
            callback_group=cb_group
        )

    def joint_state_callback(self, msg: JointState):
        """当收到手部真实关节反馈时，计算 FK 并发布指尖位置"""
        if not msg.position:
            return

        # 获取归一化关节角 (Hand Control Node 已经除以了 4095)
        # 假设顺序与 FK Solver 中的 joint_metadata 映射一致
        # (即 ID 1-6 对应的顺序)
        normalized_positions = np.array(msg.position)

        # 计算 FK
        fingertip_poses = self.fk_solver.compute_fk(normalized_positions)
        
        if fingertip_poses:
            self.publish_keypoints(fingertip_poses)

    def publish_keypoints(self, poses_data):
        """组装并发布 PoseArray"""
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"{self.hand_side}_hand_base"

        for pos, quat in poses_data:
            p = Pose()
            p.position.x = float(pos[0])
            p.position.y = float(pos[1])
            p.position.z = float(pos[2])
            p.orientation.x = float(quat[0])
            p.orientation.y = float(quat[1])
            p.orientation.z = float(quat[2])
            p.orientation.w = float(quat[3])
            msg.poses.append(p)

        self.pub_keypoints.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    node = HandFKNode()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()