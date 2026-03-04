#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import numpy as np
import mujoco
import mink
from pathlib import Path
import threading
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor


class HandIKSolver:
    """灵巧手逆运动学求解器 - 针对 80Hz 实时控制优化的 Mocap 稳定版"""
    def __init__(self, hand_type='left', solver='daqp', frequency=80.0):
        self.hand_type = hand_type
        self.solver = solver
        self.dt = 1.0 / frequency
        self.data_lock = threading.Lock()

        # 路径处理
        script_dir = Path(__file__).parent.parent
        hand_prefix = 'Left' if hand_type == 'left' else 'Right'
        mjcf_file = script_dir / 'resource' / 'RuiYan' / '0611_v1.4' / 'Version_3.0' / f'RuiYan_Hand_{hand_prefix}_Mimic' / 'meshes' / f'RuiYan_Hand_{hand_prefix}_Mimic_scene.xml'

        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_file))
        self.configuration = mink.Configuration(self.model)

        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        # 1. 预记录所有核心信息，确保 control_loop 是 "Zero-String-Lookup"
        self._setup_joint_info()      # 确定主动关节
        self._record_joint_metadata() # 缓存地址与限位
        self._setup_site_info()       # 缓存 site ID
        self._setup_mocap_targets()   # 缓存 mocap ID
        self._setup_tasks()           # 配置 IK 任务
        
        self.limits = [mink.ConfigurationLimit(self.model)]
        
        # 初始化 Home 姿态
        self.reset_to_home()

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
        """核心性能优化：预先缓存关节内存地址、限位范围，避免 80Hz 循环中的查找开销"""
        self.joint_metadata = []
        for name in self.joint_names:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            lower, upper = self.model.jnt_range[jnt_id]
            
            self.joint_metadata.append({
                'name': name,
                'id': jnt_id,
                'adr': qpos_adr,
                'low': lower,
                'high': upper,
                'range': (upper - lower) if (upper - lower) != 0 else 1.0
            })
        
        # 记录联动关节的 ID，用于更快的 clamp 操作
        self.limited_joint_adrs = [self.model.jnt_qposadr[i] for i in range(self.model.njnt) if self.model.jnt_limited[i]]
        self.limited_joint_ranges = [self.model.jnt_range[i] for i in range(self.model.njnt) if self.model.jnt_limited[i]]
        
        print(f"[IK Solver] {self.hand_type.upper()} 手初始化成功，已缓存 {len(self.joint_metadata)} 个主动关节限位")

    def _setup_site_info(self):
        side_prefix = self.hand_type
        self.fingertip_sites = [f'{side_prefix}_{name}_tip' for name in self.finger_names]
        self.site_ids_list = [] # 记录为列表方便循环
        for site_name in self.fingertip_sites:
            try:
                self.site_ids_list.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name))
            except:
                pass

    def _setup_mocap_targets(self):
        mocap_body_names = [f'{name}_target' for name in self.finger_names]
        self.mocap_ids_list = []
        for body_name in mocap_body_names:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                self.mocap_ids_list.append(self.model.body_mocapid[body_id])
            except:
                pass

    def _setup_tasks(self):
        self.tasks = []
        self.hand_tasks = []
        for site_name in self.fingertip_sites:
            task = mink.FrameTask(
                frame_name=site_name,
                frame_type="site",
                position_cost=1000.0,
                orientation_cost=0.001,
                lm_damping=1.0,
            )
            self.hand_tasks.append(task)
        self.tasks.extend(self.hand_tasks)

    def _get_mimic_relations(self):
        """预存联动关系，避免重复构建列表"""
        p = 'hand1' if self.hand_type == 'left' else 'hand2'
        return [
            (f'{p}_joint_link_1_2', f'{p}_joint_link_1_3', 1.675, 0.0),
            (f'{p}_joint_link_2_1', f'{p}_joint_link_2_2', 1.0, 0.0),
            (f'{p}_joint_link_3_1', f'{p}_joint_link_3_2', 1.0, 0.0),
            (f'{p}_joint_link_4_1', f'{p}_joint_link_4_2', 1.0, 0.0),
            (f'{p}_joint_link_5_1', f'{p}_joint_link_5_2', 1.0, 0.0),
        ]

    def _couple_mimic_velocities(self, vel):
        """将 Mink 算出的速度按比例重新分配给 Leader 关节（必须赋值返回）"""
        vel = vel.copy()
        for leader_name, follower_name, multiplier, offset in self._get_mimic_relations():
            try:
                l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, leader_name)
                f_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, follower_name)
                l_dof = self.model.jnt_dofadr[l_id]
                f_dof = self.model.jnt_dofadr[f_id]
                
                combined_vel = (vel[l_dof] + vel[f_dof]) / (1.0 + multiplier)
                vel[l_dof] = combined_vel
                vel[f_dof] = multiplier * combined_vel
            except:
                pass
        return vel

    def _apply_mimic_joints(self):
        """在位置层面强制同步联动关节"""
        with self.data_lock:
            for leader_name, follower_name, multiplier, offset in self._get_mimic_relations():
                try:
                    l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, leader_name)
                    f_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, follower_name)
                    l_adr = self.model.jnt_qposadr[l_id]
                    f_adr = self.model.jnt_qposadr[f_id]
                    self.configuration.data.qpos[f_adr] = offset + multiplier * self.configuration.data.qpos[l_adr]
                except:
                    pass

    def _clamp_joint_limits(self):
        """快速限位裁剪"""
        with self.data_lock:
            qpos = self.configuration.data.qpos
            for adr, (low, high) in zip(self.limited_joint_adrs, self.limited_joint_ranges):
                qpos[adr] = np.clip(qpos[adr], low, high)

    def reset_to_home(self):
        """完全重置到 Home 姿态并同步 mocap 球"""
        try:
            self.configuration.update_from_keyframe("home")
        except:
            self.configuration.q = np.zeros(self.model.nq)
            
        with self.data_lock:
            mujoco.mj_forward(self.model, self.configuration.data)
            for m_id, s_id in zip(self.mocap_ids_list, self.site_ids_list):
                self.configuration.data.mocap_pos[m_id] = self.configuration.data.site(s_id).xpos.copy()

    def update_mocap_targets(self, target_dict):
        """更新 Mocap 目标（由 ROS 回调触发）"""
        with self.data_lock:
            for finger_name, pos in target_dict.items():
                # 注意：此处假设 finger_name 顺序与 mocap_ids_list 对应，
                # 若不确定顺序，可保留原来的逻辑
                if finger_name in self.mocap_ids:
                    mid = self.mocap_ids[finger_name]
                    self.configuration.data.mocap_pos[mid] = pos

    def step_ik_from_mocap(self, max_iterations=10):
        """核心步进逻辑：利用 mj_forward 确保约束和雅可比的绝对准确"""
        # 1. 设置任务目标 (基于上一帧 mj_forward 算好的姿态)
        for i, (m_id, s_id) in enumerate(zip(self.mocap_ids_list, self.site_ids_list)):
            target_pos = self.configuration.data.mocap_pos[m_id].copy()
            with self.data_lock:
                rot_mat = self.configuration.data.site(s_id).xmat.reshape(3, 3).copy()

            transform = np.eye(4)
            transform[:3, :3] = rot_mat
            transform[:3, 3] = target_pos
            self.hand_tasks[i].set_target(mink.SE3.from_matrix(transform))

        # 2. IK 求解循环
        for _ in range(max_iterations):
            with self.data_lock:
                # 在循环内部保留 mj_forward 以确保联动约束和雅可比的一致性
                mujoco.mj_forward(self.model, self.configuration.data)

            vel = mink.solve_ik(
                configuration=self.configuration,
                tasks=self.tasks,
                dt=self.dt,
                solver=self.solver,
                damping=1e-3,
                safety_break=False,
                limits=self.limits,
            )

            # 关键修正：重新赋值耦合速度
            vel = self._couple_mimic_velocities(vel)
            
            self.configuration.integrate_inplace(vel, self.dt)
            self._apply_mimic_joints()
            self._clamp_joint_limits()

            if np.linalg.norm(vel) < 1e-8:
                break
                
        # 3. 同步最终状态
        with self.data_lock:
            mujoco.mj_forward(self.model, self.configuration.data)

    def get_active_joints_normalized(self):
        """利用预记录的 Metadata 极速获取归一化值 (0.0 ~ 1.0)"""
        with self.data_lock:
            qpos = self.configuration.data.qpos
            # 列表推导式 + 内存地址索引，耗时几乎为 0
            normalized = [
                np.clip((qpos[m['adr']] - m['low']) / m['range'], 0.0, 1.0)
                for m in self.joint_metadata
            ]
        return np.array(normalized)


class HandIKNode(Node):
    def __init__(self):
        super().__init__('hand_ik_node')

        self.declare_parameter('hand_side', 'left')
        self.declare_parameter('frequency', 80.0)

        self.hand_side = self.get_parameter('hand_side').value
        self.frequency = self.get_parameter('frequency').value

        self.get_logger().info(f"Initializing {self.hand_side.upper()} Hand IK Node at {self.frequency}Hz")

        try:
            self.ik_solver = HandIKSolver(
                hand_type=self.hand_side,
                frequency=self.frequency
            )
        except Exception as e:
            self.get_logger().error(f"Failed to load IK Solver: {e}")
            raise e

        # 状态机：记录系统状态及是否收到了首帧 inference 数据
        self.current_mode = "inference"
        self.received_first_action = False
        
        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        self.pub_joints = self.create_publisher(
            JointState, 
            f'/action/{self.hand_side}_hand/joints', 
            10,
            MutuallyExclusiveCallbackGroup()
        )

        self.sub_keypoints = self.create_subscription(
            PoseArray,
            f'/action/{self.hand_side}_hand/keypoints',
            self.keypoints_callback,
            10,
            MutuallyExclusiveCallbackGroup()
        )
        
        self.sub_mode = self.create_subscription(
            String,
            '/system/mode',
            self.mode_callback,
            10,
            MutuallyExclusiveCallbackGroup()
        )

        self.timer = self.create_timer(1.0 / self.frequency, self.control_loop, MutuallyExclusiveCallbackGroup())
        self.get_logger().info(f"{self.hand_side.upper()} Hand IK Node ready. Waiting for action chunk...")

    def mode_callback(self, msg: String):
        """监听模式切换"""
        new_mode = msg.data.lower()
        if new_mode in ["inference", "reset"]:
            if self.current_mode != new_mode:
                self.get_logger().info(f"[{self.hand_side.upper()}] Mode Switched: {self.current_mode} -> {new_mode}")
                self.current_mode = new_mode
                
                if self.current_mode == "reset":
                    # 重置内部求解器，将所有目标球归零
                    self.ik_solver.reset_to_home()
                    # 清除首帧标记，这样下次切回 inference 时终端会有“等待动作”的提示
                    self.received_first_action = False
                    self.get_logger().info(f"[{self.hand_side.upper()}] Hand completely reset to Home posture. Interface data will be ignored.")
                
                elif self.current_mode == "inference":
                    self.get_logger().info(f"[{self.hand_side.upper()}] Ready for inference. Waiting for the first predicted action...")

    def keypoints_callback(self, msg: PoseArray):
        """接收Interface按需(30Hz)下发的动作预测"""
        # 如果系统在 reset，直接抛弃数据，防患于未然
        if self.current_mode != "inference":
            return

        if len(msg.poses) != 5:
            self.get_logger().warn(f"[{self.hand_side.upper()}] Expected 5 keypoints, got {len(msg.poses)}. Ignored.")
            return

        # 首次接收到有效数据时，打印一条日志
        if not self.received_first_action:
            self.received_first_action = True
            self.get_logger().info(f"[{self.hand_side.upper()}] Received first action chunk. Started tracking targets.")

        # 将收到的 5 个指尖位置丢给底层的 Mocap 目标球
        target_dict = {}
        for i, finger_name in enumerate(self.finger_names):
            pos = msg.poses[i].position
            target_dict[finger_name] = np.array([pos.x, pos.y, pos.z])
            
        self.ik_solver.update_mocap_targets(target_dict)

    def publish_joints(self, joint_angles):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.ik_solver.joint_names
        msg.position = joint_angles.tolist()
        self.pub_joints.publish(msg)

    def control_loop(self):
        """以80Hz持续执行。无论有没有新数据，都要发布 JointState 以供控制板平滑插值"""
        if self.current_mode == "reset":
            # reset状态下不解IK，直接输出 home 状态 (0 消耗)
            active_joints = self.ik_solver.get_active_joints_normalized()
            self.publish_joints(active_joints)
            return

        if self.current_mode == "inference":
            # 不论是否收到新数据，都向着当前的 Mocap 球追赶 10 步
            # 如果没收到新数据，它会向着上一次的 Mocap 位置收敛，非常稳定
            self.ik_solver.step_ik_from_mocap(max_iterations=10)
            
            # 发布计算好的关节角度
            active_joints = self.ik_solver.get_active_joints_normalized()
            self.publish_joints(active_joints)


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    node = HandIKNode()
    executor.add_node(node)
    try: executor.spin()
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()