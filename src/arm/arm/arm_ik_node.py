#!/usr/bin/env python3

import os
import time
import threading

# 设置 MuJoCo 使用 GLX 渲染后端（Docker 环境需要）
os.environ['MUJOCO_GL'] = 'glx'

import mink
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor


class ArmIKNode(Node):
    def __init__(self):
        super().__init__('arm_ik_node')
        
        # 声明和获取ROS2参数
        self.declare_parameter('enable_viewer', True)
        self.declare_parameter('frequency', 100.0)
        self.declare_parameter('solver', 'daqp')
        self.declare_parameter('robot_xml_path', '')
        
        # 获取参数值
        self.enable_viewer = self.get_parameter('enable_viewer').value
        self.frequency = self.get_parameter('frequency').value
        self.dt = 1.0 / self.frequency
        self.solver = self.get_parameter('solver').value
        
        # PsiRobot双臂配置 - 使用wrist site作为end-effector
        self.hands = ['left_wrist', 'right_wrist']  # 左臂和右臂的手腕site
        self.targets = ['left_wrist_target', 'right_wrist_target']
        self.bases = ['arm1_link0', 'arm2_link0']
        self.xml_path = self.get_parameter('robot_xml_path').value
        if self.xml_path == '':
            self.xml_path = "/root/workspace/legendvla-inference/assets/PsiRobot_DC_02_OnlyArm/meshes/psi_robot_scene_transformed.xml"
        
        # 初始化MuJoCo模型和配置
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.configuration = mink.Configuration(self.model)
        self.data_lock = threading.Lock()

        # PsiRobot关节名称配置
        self.left_arm_joint_names = [f'arm1_joint_link{i+1}' for i in range(7)]  # ARM1 = 左臂
        self.right_arm_joint_names = [f'arm2_joint_link{i+1}' for i in range(7)]  # ARM2 = 右臂

        # 初始化Mink
        self.mink_setup()

        # 如果启用viewer，启动可视化线程
        self.viewer = None
        if self.enable_viewer:
            self._start_viewer()

        self._warm_up_sim()

        # 创建ROS2发布器和订阅器
        self._publish_count = 0
        self.mode = 'inference'

        self.wrist_poses_sub = self.create_subscription(
            PoseArray,
            '/action/both_arms/wrist_poses',
            self.wrist_poses_callback,
            10,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.system_mode_sub = self.create_subscription(
            String,
            '/system/mode',
            self.system_mode_callback,
            10,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.left_arm_command_pub = self.create_publisher(
            JointState, 
            '/action/left_arm/joints',
            10
        )

        self.right_arm_command_pub = self.create_publisher(
            JointState, 
            '/action/right_arm/joints',
            10
        )
        
        # 创建定时器，用于定期求解IK并发布关节状态
        self.timer = self.create_timer(self.dt, self.timer_callback, callback_group=MutuallyExclusiveCallbackGroup())
        
        self.get_logger().info(f'ArmIKNode initialized with frequency: {self.frequency}Hz')
        self.get_logger().info(f'PsiRobot sites: {self.hands}')
        self.get_logger().info(f'PsiRobot targets: {self.targets}')
        self.get_logger().info(f'Viewer enabled: {self.enable_viewer}')

    def mink_setup(self):
        """设置IK任务 - 专为PsiRobot双臂配置"""
        # PsiRobot关节阻尼成本 - 14个DOF（双臂）
        damping_joints_cost = np.ones(self.model.nv) * 1.0  # 所有关节阻尼相同
        self.damping_task = mink.DampingTask(self.model, cost=damping_joints_cost)
        
        # 创建动能正则化任务
        self.kinetic_energy_task = mink.KineticEnergyRegularizationTask(cost=1e-3)
        self.kinetic_energy_task.set_dt(self.dt)

        self.posture_task = mink.PostureTask(self.model, cost=0.1)

        # 添加基础任务
        self.tasks = [
            self.damping_task,
            self.kinetic_energy_task,
            self.posture_task,
        ]
        
        # 为每个手部创建帧任务
        self.hand_tasks = []
        for hand_frame_name in self.hands:
            task = mink.FrameTask(
                frame_name=hand_frame_name,
                frame_type="site",
                position_cost=5.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )
            self.hand_tasks.append(task)
        self.tasks.extend(self.hand_tasks)

        self.configuration.update_from_keyframe("home")  # 从"home"关键帧更新配置
        self.posture_task.set_target(self.configuration.q)
        self.initial_mocap_pos = {}
        self.initial_mocap_quat = {}
        self.T_base2world_list = []
        with self.data_lock:
            mujoco.mj_forward(self.model, self.configuration.data)
            # 打印双臂site的位置和四元数
            for i in range(len(self.hands)):
                hand = self.hands[i]
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, hand)
                site_xpos = self.configuration.data.site(site_id).xpos.copy()
                site_xmat = self.configuration.data.site(site_id).xmat.copy()
                arm_name = "左臂" if i == 0 else "右臂"
                self.get_logger().info(f"🎈 {arm_name} ({hand}) site初始位置: {site_xpos}, 旋转矩阵: {site_xmat}")

                target = self.targets[i]
                target_mocap_id = self.model.body(target).mocapid[0]
                xml_pos = self.configuration.data.mocap_pos[target_mocap_id].copy()
                xml_quat = self.configuration.data.mocap_quat[target_mocap_id].copy()
                self.initial_mocap_pos[target] = xml_pos
                self.initial_mocap_quat[target] = xml_quat
                self.get_logger().info(f"🟢 {arm_name} ({target}) 保留XML定义的mocap位置: {xml_pos}")
                self.get_logger().info(f"🟢 {arm_name} ({target}) 保留XML定义的mocap姿态: {xml_quat}")

                base = self.bases[i]
                base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base)
                base_pos = self.configuration.data.xpos[base_id].copy()
                base_quat_mujoco = self.configuration.data.xquat[base_id].copy()
                T_base2world = np.eye(4)
                T_base2world[:3, :3] = R.from_quat(base_quat_mujoco, scalar_first=True).as_matrix()
                T_base2world[:3, 3] = base_pos
                self.T_base2world_list.append(T_base2world)
                self.get_logger().info(f"💾 已缓存 {arm_name} 基座 ({base}) 的全局 4x4 变换矩阵")

        _mink_joint_limit = mink.ConfigurationLimit(self.model)
        self.limits = [_mink_joint_limit]

    def _start_viewer(self):
        """在单独线程中启动MuJoCo viewer"""
        try:
            self.viewer = mujoco.viewer.launch_passive(
                    model=self.model, 
                    data=self.configuration.data, 
                    show_left_ui=True,
                    show_right_ui=True
                )
            mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)
        except Exception as e:
            self.get_logger().warning(f"Failed to start viewer (headless environment?): {e}")
            self.viewer = None
            self.enable_viewer = False

    def _warm_up_sim(self):
        """预热仿真"""
        for _ in range(10):
            with self.data_lock:
                mujoco.mj_forward(self.model, self.configuration.data)
                if self.enable_viewer and self.viewer:
                    self.viewer.sync()
                time.sleep(0.01)

    def wrist_poses_callback(self, msg: PoseArray):
        """接收手腕位姿的回调函数"""
        for i in range(len(msg.poses)):
            pose = msg.poses[i]
            target = self.targets[i]
            target_mocap_id = self.model.body(target).mocapid[0]

            T_base2world = self.T_base2world_list[i]
            target_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
            target_quat_scipy =[pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            T_target2base = np.eye(4)
            T_target2base[:3, :3] = R.from_quat(target_quat_scipy).as_matrix()
            T_target2base[:3, 3] = target_pos

            T_target2world = T_base2world @ T_target2base
            target_pos_world = T_target2world[:3, 3]
            target_rot_world = T_target2world[:3, :3]
            target_quat_world_mujoco = R.from_matrix(target_rot_world).as_quat(scalar_first=True)

            
            with self.data_lock:
                self.configuration.data.mocap_pos[target_mocap_id] = target_pos_world
                self.configuration.data.mocap_quat[target_mocap_id] = target_quat_world_mujoco

    def system_mode_callback(self, msg: String):
        """接收系统模式切换的回调函数"""
        self.mode = msg.data

    def timer_callback(self):
        """定时器回调函数，用于求解IK并发布关节状态"""
        t0 = time.time()

        if self.mode == 'reset':
            self.mode = 'inference'
            self.configuration.update_from_keyframe("home")
            for target in self.targets:
                target_mocap_id = self.model.body(target).mocapid[0]
                with self.data_lock:
                    self.configuration.data.mocap_pos[target_mocap_id] = self.initial_mocap_pos[target]
                    self.configuration.data.mocap_quat[target_mocap_id] = self.initial_mocap_quat[target]

        # MuJoCo更新
        with self.data_lock:
            mujoco.mj_forward(self.model, self.configuration.data)
            if self.enable_viewer and self.viewer:
                self.viewer.sync()
        
        # 更新任务目标
        for i in range(len(self.hand_tasks)):
            hand_task = self.hand_tasks[i]
            target = self.targets[i]
            with self.data_lock:
                _hand_target = mink.SE3.from_mocap_name(self.model, self.configuration.data, target)
                hand_task.set_target(_hand_target)
        
        # 求解IK
        vel = mink.solve_ik(
            configuration=self.configuration, 
            tasks=self.tasks, 
            dt=self.dt, 
            solver=self.solver, 
            damping=1e-1,
            safety_break=False,
            limits=self.limits,
        )
        self.configuration.integrate_inplace(vel, self.dt)
        
        with self.data_lock:
            _solved_joints = self.configuration.data.qpos.copy()  # PsiRobot: [arm1_joints, arm2_joints]
        
        self.get_logger().debug(f'Solved joints: {_solved_joints}')
        
        # 发布关节状态
        self._publish_arms_joints_cmd(target_arm_joints=_solved_joints)

        t1 = time.time()
        self.get_logger().debug(f'IK solve loop time: {(t1-t0)*1000:.2f}ms')
    
    def _publish_arms_joints_cmd(self, target_arm_joints):
        """发布PsiRobot双臂关节角度（匹配控制节点格式）"""
        # PsiRobot关节位置
        if isinstance(target_arm_joints, np.ndarray):
            joint_positions = target_arm_joints.tolist()
        else:
            joint_positions = target_arm_joints
            
        # 确保关节数量正确（14个DOF）
        expected_joints = self.left_arm_joint_names + self.right_arm_joint_names
        assert len(joint_positions) == len(expected_joints), f"joint_positions length mismatch: {len(joint_positions)} != {len(expected_joints)}"
        
        # 创建JointState消息（带时间戳）
        timestamp = self.get_clock().now().to_msg()
        left_arm_joint_state_msg = JointState()
        left_arm_joint_state_msg.header.stamp = timestamp
        left_arm_joint_state_msg.header.frame_id = "base_link"
        left_arm_joint_state_msg.name = self.left_arm_joint_names
        left_arm_joint_state_msg.position = [float(pos) for pos in joint_positions[:7]]
        
        right_arm_joint_state_msg = JointState()
        right_arm_joint_state_msg.header.stamp = timestamp
        right_arm_joint_state_msg.header.frame_id = "base_link"
        right_arm_joint_state_msg.name = self.right_arm_joint_names
        right_arm_joint_state_msg.position = [float(pos) for pos in joint_positions[7:]]

        # 发布关节角度消息
        self.left_arm_command_pub.publish(left_arm_joint_state_msg)
        self.right_arm_command_pub.publish(right_arm_joint_state_msg)

        self._publish_count += 1
            
        if self._publish_count % 100 == 0:  # 每100次输出一次
            self.get_logger().info(f"🔍 [调试] 发布IK结果到/action/left_arm/joints 和 /action/right_arm/joints: 长度={len(joint_positions)}, 前3个关节={joint_positions[:3]}")

    def destroy_node(self):
        """节点销毁时的清理函数"""
        if self.viewer:
            self.viewer.close()
        super().destroy_node()

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    node = ArmIKNode()
    executor.add_node(node)
    try: executor.spin()
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()
        

if __name__ == '__main__':
    main()