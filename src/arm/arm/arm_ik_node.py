"""
Arm IK Node - 机械臂逆运动学节点

这个节点为PsiRobot双臂机器人实现Mink IK求解器，包含以下功能：
1. 从topic接收PoseArray目标位姿（armbase坐标）
2. 将目标位姿转换为世界坐标系
3. 使用Mink求解器计算逆运动学
4. 分别发布左右臂JointState消息到topic
5. 可选择启用MuJoCo viewer进行可视化
"""

# TODO: reset 多少状态？要不要清空ik_solver？要不要重置ik目标as home?

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import threading
from rclpy.callback_groups import ReentrantCallbackGroup

import mink  # Mink IK求解器

# Import connector configuration parameters
from connector_config import (
    LEFT_CONNECTOR_OFFSET_1_1, LEFT_CONNECTOR_RPY_1_1,
    LEFT_CONNECTOR_OFFSET_1_2, LEFT_CONNECTOR_RPY_1_2,
    RIGHT_CONNECTOR_OFFSET_2_1, RIGHT_CONNECTOR_RPY_2_1,
    RIGHT_CONNECTOR_OFFSET_2_2, RIGHT_CONNECTOR_RPY_2_2,
    CONNECTOR_HEIGHT, TCP_ROTATION
)


class ArmIKNode(Node):
    """
    机械臂逆运动学节点 - 专为PsiRobot双臂机器人设计
    
    主要功能：
    - 接收双臂目标位姿（armbase坐标）并求解逆运动学
    - 分别发布左右臂关节状态
    - 可选的可视化功能
    """
    
    def __init__(self):
        """初始化ArmIKNode"""
        super().__init__('arm_ik_node')

        # ik节点模式：inference接收位姿信息，发送ik求解关节角，reset发送复位信息
        self.mode = "inference"   # or "reset"
        self.mode_lock = threading.Lock()

        # ROS2 parallel callback groups
        self.both_arms_sub_group = ReentrantCallbackGroup()
        self.publisher_group = ReentrantCallbackGroup()
        self.mode_sub_group = ReentrantCallbackGroup()
        
        # 声明和获取ROS2参数
        self.declare_parameter('enable_viewer', False)
        self.declare_parameter('frequency', 100.0)
        self.declare_parameter('solver', 'daqp')
        self.declare_parameter('ik_xml_path', '')
        
        # 获取参数值
        self.enable_viewer = self.get_parameter('enable_viewer').value
        self.frequency = self.get_parameter('frequency').value
        self.dt = 1.0 / self.frequency
        self.solver = self.get_parameter('solver').value
        
        # PsiRobot双臂配置 - 使用wrist site作为end-effector
        self.hands = ['left_wrist', 'right_wrist']  # 左臂和右臂的手腕site
        self.xml_path = self.get_parameter('ik_xml_path').value

        self.is_env_initialized = False

        if self.xml_path == '':
            # Use absolute path to assets directory
            # Current file is at: src/arm/arm/arm_ik_node.py
            # Assets directory is at: assets/PsiRobot_DC_02_OnlyArm/meshes/
            import os
            from pathlib import Path
            repo_root = Path(__file__).parent.parent.parent.parent
            self.xml_path = str(repo_root / "assets" / "PsiRobot_DC_02_OnlyArm" / "meshes" / "psi_robot_scene_transformed.xml")
        
        # 创建ROS2发布器和订阅器
        self.left_arm_pub = self.create_publisher(
            JointState, 
            '/action/left_arm/joints',
            10,
            callback_group=self.publisher_group
        )
        
        self.right_arm_pub = self.create_publisher(
            JointState, 
            '/action/right_arm/joints',
            10,
            callback_group=self.publisher_group
        )
        
        # 订阅器：接收双臂手腕位姿
        self.both_arms_pose_sub = self.create_subscription(
            PoseArray,
            '/action/both_arms/wrist_poses',
            self.action_callback,
            10,
            callback_group=self.both_arms_sub_group
        )

        # 订阅器：接收复位命令
        self.reset_sub = self.create_subscription(
            String,
            '/system/mode',
            self._reset_command_callback,
            10,
            callback_group=self.mode_sub_group
        )

        
        # 初始化MuJoCo模型和配置
        self.model = self._load_robot_model(self.xml_path)
        self.configuration = mink.Configuration(self.model)
        self.data_lock = threading.Lock()
        
        # 检查是否为URDF模式（缺少MuJoCo特定特性）
        self.is_urdf_mode = self._check_urdf_mode()

        # PsiRobot关节名称配置
        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]
        self.arm_joint_names = self.joint_names  # PsiRobot只有手臂关节
        self.left_arm_joint_names = [f'arm1_joint_link{i+1}' for i in range(7)]  # ARM1 = 左臂
        self.right_arm_joint_names = [f'arm2_joint_link{i+1}' for i in range(7)]  # ARM2 = 右臂

        # 获取左右臂base的body ID（用于坐标转换）
        self.left_arm_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'arm1_link0')
        self.right_arm_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'arm2_link0')

        # 获取home_key_id用于复位
        self.home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        assert self.home_key_id >= 0, "❌ MuJoCo XML does not contain a keyframe named 'home'"

        # 初始化IK任务
        self._setup_tasks()

        # 初始化mink环境
        self._init_mink_env()

        # 设置关节限制
        self._setup_limits()

        # 初始化mocap物体
        self._init_mocap_bodies()

        # 如果启用viewer，启动可视化线程
        self.viewer = None
        if self.enable_viewer:
            self._start_viewer()

        self._warm_up_sim()
        
        # 创建定时器，用于定期求解IK并发布关节状态
        self.timer = self.create_timer(1.0/self.frequency, self.timer_callback)
        
        self.get_logger().info(f'Arm IK Node initialized with frequency: {self.frequency}Hz')
        self.get_logger().info(f'PsiRobot hands: {self.hands}')
        self.get_logger().info(f'Viewer enabled: {self.enable_viewer}')

    def _load_robot_model(self, model_path):
        """
        通用机器人模型加载方法，支持URDF和MuJoCo XML格式
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            MuJoCo模型对象
        """
        import os
        
        # 获取文件扩展名
        _, ext = os.path.splitext(model_path.lower())
        
        if ext == '.urdf':
            self.get_logger().info(f'Loading URDF model from: {model_path}')
            
            # MuJoCo 2.3+ 支持直接加载URDF
            # 需要切换到URDF文件所在目录，以便正确解析相对路径的mesh文件
            try:
                urdf_dir = os.path.dirname(os.path.abspath(model_path))
                original_cwd = os.getcwd()
                
                # 切换到URDF所在目录
                os.chdir(urdf_dir)
                self.get_logger().info(f'🔄 Changed working directory to: {urdf_dir}')
                
                try:
                    model = mujoco.MjModel.from_xml_path(os.path.basename(model_path))
                    self.get_logger().info('✅ Successfully loaded URDF model')
                    return model
                finally:
                    # 恢复原始工作目录
                    os.chdir(original_cwd)
                    
            except Exception as e:
                self.get_logger().error(f'❌ Failed to load URDF model: {e}')
                raise
                
        elif ext in ['.xml', '.mjcf']:
            self.get_logger().info(f'Loading MuJoCo XML model from: {model_path}')
            
            try:
                model = mujoco.MjModel.from_xml_path(str(model_path))
                self.get_logger().info('✅ Successfully loaded MuJoCo XML model')
                return model
            except Exception as e:
                self.get_logger().error(f'❌ Failed to load MuJoCo XML model: {e}')
                raise
                
        else:
            error_msg = f'Unsupported model format: {ext}. Supported formats: .urdf, .xml, .mjcf'
            self.get_logger().error(error_msg)
            raise ValueError(error_msg)
    
    def _check_urdf_mode(self):
        """
        检查是否为URDF模式（缺少MuJoCo特定特性）
        
        Returns:
            bool: True if URDF mode, False if full MuJoCo mode
        """
        # 检查必需的MuJoCo特性是否存在
        has_end_effector_sites = True
        has_mocap_targets = True
        has_keyframes = True
        
        try:
            # 检查end-effector sites
            for hand in self.hands:
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, hand)
        except:
            has_end_effector_sites = False
            
        try:
            # 检查mocap targets
            self.model.body("left_wrist_target")
            self.model.body("right_wrist_target")
        except:
            has_mocap_targets = False
            
        try:
            # 检查keyframes
            if self.model.nkey == 0:
                has_keyframes = False
        except:
            has_keyframes = False
            
        is_urdf = not (has_end_effector_sites and has_mocap_targets and has_keyframes)
        
        if is_urdf:
            self.get_logger().warn('🔄 URDF compatibility mode detected - some features will be limited')
            self.get_logger().warn(f'   End-effector sites: {"✅" if has_end_effector_sites else "❌"}')
            self.get_logger().warn(f'   Mocap targets: {"✅" if has_mocap_targets else "❌"}')  
            self.get_logger().warn(f'   Keyframes: {"✅" if has_keyframes else "❌"}')
        else:
            self.get_logger().info('✅ Full MuJoCo mode - all features available')
            
        return is_urdf

    def _warm_up_sim(self):
        """预热仿真"""
        for _ in range(10):
            with self.data_lock:
                mujoco.mj_forward(self.model, self.configuration.data)
                if self.enable_viewer:
                    self.viewer.sync()
                time.sleep(0.01)

    def _setup_limits(self):
        """设置关节限制"""
        _mink_joint_limit = mink.ConfigurationLimit(self.model)
        self.limits = [_mink_joint_limit]
    
    def _setup_tasks(self):
        """设置IK任务 - 专为PsiRobot双臂配置"""
        hz = self.frequency
        
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
        
        # 获取mocap物体的ID（仅在非URDF模式下）
        if not self.is_urdf_mode:
            self.hands_target_model_id = [
                self.model.body("left_wrist_target").mocapid[0],  # 左臂目标
                self.model.body("right_wrist_target").mocapid[0]   # 右臂目标
            ]
        else:
            self.hands_target_model_id = []
            self.get_logger().warn('🔄 URDF mode: Skipping mocap target initialization')
        
    def _init_mink_env(self):
        """初始化mink环境"""
        self.configuration.update_from_keyframe("home")  # 从"home"关键帧更新配置
        self.posture_task.set_target(self.configuration.q)
        
        with self.data_lock:
            # 打印双臂site的位置和四元数
            for i, hand in enumerate(self.hands):
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, hand)
                site_pos = self.configuration.data.site(site_id).xpos.copy()
                site_xmat = self.configuration.data.site(site_id).xmat.copy()
                arm_name = "左臂" if i == 0 else "右臂"
                self.get_logger().info(f"🎈 {arm_name} ({hand}) site初始位置: {site_pos}, 旋转矩阵: {site_xmat}")

        self.is_env_initialized = True
    
    def _init_mocap_bodies(self):
        """初始化mocap物体 - 保留XML中定义的target位置"""
        assert self.is_env_initialized, "Mink environment is not initialized"
        
        with self.data_lock:
            for hand in self.hands:
                # 🔍 调试：打印XML中定义的mocap位置（不移动）
                target_name = f"{hand}_target"
                target_mocap_id = self.model.body(target_name).mocapid[0]
                xml_pos = self.configuration.data.mocap_pos[target_mocap_id].copy()
                xml_quat = self.configuration.data.mocap_quat[target_mocap_id].copy()
                print(f"🟢 {hand} 保留XML定义的mocap位置: {xml_pos}")
                print(f"🟢 {hand} 保留XML定义的mocap姿态: {xml_quat}")
                
                # ❌ 注释掉这行：不要覆盖XML中定义的target位置！
                # mink.move_mocap_to_frame(self.model, self.configuration.data, target_name, hand, "site")
                
            mujoco.mj_forward(self.model, self.configuration.data)

    def _start_viewer(self):
        """在单独线程中启动MuJoCo viewer"""
        self.viewer = mujoco.viewer.launch_passive(
                model=self.model, 
                data=self.configuration.data, 
                show_left_ui=True,
                show_right_ui=True
            )
        mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)

    def _reset_command_callback(self, msg: String):
        """
        接收 reset / inference 模式切换
        """
        cmd = msg.data.strip().lower()

        if cmd == "reset":
            self.get_logger().info("🔄 收到 RESET 指令，切换到 reset 模式")

            with self.data_lock:
                # 重置 MuJoCo 到 home keyframe
                mujoco.mj_resetDataKeyframe(
                    self.model,
                    self.configuration.data,
                    self.home_key_id
                )
                mujoco.mj_forward(self.model, self.configuration.data)

                # 清空 velocity/增量残差
                self.configuration.integrate_inplace(np.zeros_like(self.configuration.data.qvel), self.dt)

                with self.mode_lock:
                    # 切换模式
                    self.mode = "reset"

        elif cmd == "inference":
            self.get_logger().info("▶️ 切换到 inference 模式")
            with self.mode_lock:
                self.mode = "inference"

        else:
            self.get_logger().warn(f"⚠️ 未识别的 system mode: {cmd}")

    def _hand_to_tcp(self, hand_pos, hand_mat, arm='left'):
        """
        将hand位姿经过连接件转换为TCP位姿（即手腕位姿）

        :param hand_pos: hand位置 (3,) numpy array
        :param hand_mat: hand旋转矩阵 (3,3) numpy array
        :param arm: 'left' 或 'right'
        :return: (wrist_pos, wrist_mat) TCP位置和旋转矩阵
        所有坐标为相对base计算
        """
        # 固定变换矩阵
        T_conn2tcp = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.0345],
            [0, 0, 0, 1]
        ])

        T_hand2conn_left = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])

        T_hand2conn_right = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ])

        # 选择手臂对应的 hand2conn
        if arm == 'left':
            T_hand2conn = T_hand2conn_left
        elif arm == 'right':
            T_hand2conn = T_hand2conn_right
        else:
            raise ValueError(f"无效的arm参数: '{arm}'。必须是 'left' 或 'right'")

        # 复合变换：hand -> TCP
        T_hand2tcp = T_conn2tcp @ T_hand2conn   # 4x4

        # 求逆：TCP -> hand
        # 对于刚体变换，逆可以直接计算：[R^T, -R^T * t; 0 0 0 1]
        R_ht = T_hand2tcp[:3, :3]
        t_ht = T_hand2tcp[:3, 3]
        T_tcp2hand = np.eye(4)
        T_tcp2hand[:3, :3] = R_ht.T
        T_tcp2hand[:3, 3] = -R_ht.T @ t_ht

        # 构造 hand 在参考系中的变换矩阵
        T_ref_hand = np.eye(4)
        T_ref_hand[:3, :3] = hand_mat
        T_ref_hand[:3, 3] = hand_pos

        # 计算 TCP 在参考系中的变换矩阵
        T_ref_tcp = T_ref_hand @ T_tcp2hand

        # 提取位置和旋转矩阵
        wrist_pos = T_ref_tcp[:3, 3]
        wrist_mat = T_ref_tcp[:3, :3]

        return wrist_pos, wrist_mat
    
    def _arm_base_to_world(self, position, quaternion_xyzw, arm='left'):
        """
        将相对于arm base的位姿转换为世界坐标系
        
        Args:
            position: [3] 相对于arm base的位置
            quaternion_xyzw: [4] 相对于arm base的四元数 (x,y,z,w)
            arm: 'left' 或 'right'
            
        Returns:
            world_position: [3] 世界坐标系位置
            world_quaternion_wxyz: [4] 世界坐标系四元数 (w,x,y,z) MuJoCo格式
        """
        # 获取arm base在世界坐标系中的位姿
        if arm == 'left':
            base_id = self.left_arm_base_id
        else:
            base_id = self.right_arm_base_id
        
        with self.data_lock:
            base_pos = self.configuration.data.body(base_id).xpos.copy()
            base_mat = self.configuration.data.body(base_id).xmat.copy().reshape(3, 3)
        
        # 位置变换: world_pos = base_pos + base_mat @ relative_pos
        world_position = base_pos + base_mat @ position
        
        # 姿态变换: world_mat = base_mat @ relative_mat
        relative_mat = R.from_quat(quaternion_xyzw).as_matrix()
        world_mat = base_mat @ relative_mat
        
        # 转换为MuJoCo四元数格式 (w,x,y,z)
        world_quaternion_xyzw = R.from_matrix(world_mat).as_quat()
        world_quaternion_wxyz = np.array([
            world_quaternion_xyzw[3],  # w
            world_quaternion_xyzw[0],  # x
            world_quaternion_xyzw[1],  # y
            world_quaternion_xyzw[2]   # z
        ])
        
        return world_position, world_quaternion_wxyz

    def action_callback(self, msg: PoseArray):
        """接收双臂hand_base目标位姿的回调函数（armbase坐标系） - 逆变换为TCP位姿后转换为世界坐标并设置为IK目标
        
        Args:
            msg: PoseArray消息，包含2个Pose（相对于arm base坐标系）：
                 poses[0] - 左臂hand_base位姿（相对于左臂base）
                 poses[1] - 右臂hand_base位姿（相对于右臂base）
        """

        with self.mode_lock:
            if self.mode == "reset":
                return # reset状态不再接收PoseArray

        if len(msg.poses) < 2:
            self.get_logger().warn(f"⚠️ PoseArray消息包含的poses数量不足: {len(msg.poses)}, 需要至少2个")
            return
        
        # 处理左臂 (poses[0]) - hand_base位姿 -> TCP位姿 -> 世界坐标
        left_hand_base_pose = msg.poses[0]
        left_hand_base_pos_armbase = np.array([left_hand_base_pose.position.x, left_hand_base_pose.position.y, left_hand_base_pose.position.z])
        left_hand_base_quat_xyzw = np.array([
            left_hand_base_pose.orientation.x,
            left_hand_base_pose.orientation.y,
            left_hand_base_pose.orientation.z,
            left_hand_base_pose.orientation.w
        ])
        left_hand_base_mat_armbase = R.from_quat(left_hand_base_quat_xyzw).as_matrix()
        
        left_tcp_pos_armbase, left_tcp_mat_armbase = self._hand_to_tcp(
            left_hand_base_pos_armbase, 
            left_hand_base_mat_armbase, 
            'left'
        )
        
        # 转换TCP姿态矩阵为四元数
        left_tcp_quat_xyzw = R.from_matrix(left_tcp_mat_armbase).as_quat()
        
        # 坐标转换: armbase -> world
        left_position_world, left_quat_mujoco = self._arm_base_to_world(
            left_tcp_pos_armbase, 
            left_tcp_quat_xyzw, 
            arm='left'
        )

        left_target_name = "left_wrist_target"
        left_target_mocap_id = self.model.body(left_target_name).mocapid[0]
        
        # 设置左臂世界坐标目标
        with self.data_lock:
            with self.mode_lock:
                if self.mode == "inference": # 线程锁，只在inference模式下更新目标
                    self.configuration.data.mocap_pos[left_target_mocap_id] = left_position_world
                    self.configuration.data.mocap_quat[left_target_mocap_id] = left_quat_mujoco
        
        # 处理右臂 (poses[1]) - hand_base位姿 -> TCP位姿 -> 世界坐标
        right_hand_base_pose = msg.poses[1]
        right_hand_base_pos_armbase = np.array([right_hand_base_pose.position.x, right_hand_base_pose.position.y, right_hand_base_pose.position.z])
        right_hand_base_quat_xyzw = np.array([
            right_hand_base_pose.orientation.x,
            right_hand_base_pose.orientation.y,
            right_hand_base_pose.orientation.z,
            right_hand_base_pose.orientation.w
        ])
        right_hand_base_mat_armbase = R.from_quat(right_hand_base_quat_xyzw).as_matrix()
        
        right_tcp_pos_armbase, right_tcp_mat_armbase = self._hand_to_tcp(
            right_hand_base_pos_armbase, 
            right_hand_base_mat_armbase, 
            'right'
        )
        
        # 转换TCP姿态矩阵为四元数
        right_tcp_quat_xyzw = R.from_matrix(right_tcp_mat_armbase).as_quat()
        
        # 坐标转换: armbase -> world
        right_position_world, right_quat_mujoco = self._arm_base_to_world(
            right_tcp_pos_armbase, 
            right_tcp_quat_xyzw, 
            arm='right'
        )

        right_target_name = "right_wrist_target"
        right_target_mocap_id = self.model.body(right_target_name).mocapid[0]
        
        # 设置右臂世界坐标目标
        with self.data_lock:
            with self.mode_lock:
                if self.mode == "inference": # 线程锁，只在inference模式下更新目标
                    self.configuration.data.mocap_pos[right_target_mocap_id] = right_position_world
                    self.configuration.data.mocap_quat[right_target_mocap_id] = right_quat_mujoco  
    
    def timer_callback(self):
        """定时器回调函数，用于求解IK并发布关节状态"""

        t0 = time.time()

        with self.data_lock:
            with self.mode_lock:
                if self.mode == "reset":
                    # 直接发布 home qpos
                    home_joints = self.configuration.data.qpos.copy()
                    self._publish_arms_joints_cmd(home_joints)
                    return

            # MuJoCo更新
            mujoco.mj_forward(self.model, self.configuration.data)
            if self.enable_viewer:
                self.viewer.sync()
        
            # 更新任务目标
            for i, hand_task in enumerate(self.hand_tasks):
                target_names = ["left_wrist_target", "right_wrist_target"]
                _hand_target = mink.SE3.from_mocap_name(self.model, self.configuration.data, target_names[i])
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

        with self.data_lock:
            with self.mode_lock:
                if self.mode == "inference": # 线程锁：只在inference状态更新关节角
                    self.configuration.integrate_inplace(vel, self.dt)
        
            _solved_joints = self.configuration.data.qpos.copy()  # PsiRobot: [arm1_joints, arm2_joints]
        
        self.get_logger().debug(f'Solved joints: {_solved_joints}')
        
        # 发布关节状态
        self._publish_arms_joints_cmd(target_arm_joints=_solved_joints)

        t1 = time.time()
        self.get_logger().debug(f'IK solve loop time: {(t1-t0)*1000:.2f}ms')
    
    def _publish_arms_joints_cmd(self, target_arm_joints):
        """分别发布左臂和右臂关节角度"""
        # PsiRobot关节位置
        if isinstance(target_arm_joints, np.ndarray):
            joint_positions = target_arm_joints.tolist()
        else:
            joint_positions = target_arm_joints
            
        # 确保关节数量正确（14个DOF）
        expected_joints = self.left_arm_joint_names + self.right_arm_joint_names
        assert len(joint_positions) == len(expected_joints), f"joint_positions length mismatch: {len(joint_positions)} != {len(expected_joints)}"
        
        # 分割左右臂关节（前7个是左臂，后7个是右臂）
        left_joint_positions = joint_positions[:7]
        right_joint_positions = joint_positions[7:]
        
        timestamp = self.get_clock().now().to_msg()
        
        # 创建并发布左臂JointState消息
        left_joint_state_msg = JointState()
        left_joint_state_msg.header.stamp = timestamp
        left_joint_state_msg.header.frame_id = "base_link"
        left_joint_state_msg.name = self.left_arm_joint_names
        left_joint_state_msg.position = [float(pos) for pos in left_joint_positions]
        self.left_arm_pub.publish(left_joint_state_msg)
        
        # 创建并发布右臂JointState消息
        right_joint_state_msg = JointState()
        right_joint_state_msg.header.stamp = timestamp
        right_joint_state_msg.header.frame_id = "base_link"
        right_joint_state_msg.name = self.right_arm_joint_names
        right_joint_state_msg.position = [float(pos) for pos in right_joint_positions]
        self.right_arm_pub.publish(right_joint_state_msg)
        
        # 调试信息：显示发布的关节角度
        if hasattr(self, '_publish_count'):
            self._publish_count += 1
        else:
            self._publish_count = 1
            
        if self._publish_count % 50 == 1:  # 每50次输出一次
            self.get_logger().info(f"🔍 [调试] 发布左臂到/action/left_arm/joints: {left_joint_positions[:3]}")
            self.get_logger().info(f"🔍 [调试] 发布右臂到/action/right_arm/joints: {right_joint_positions[:3]}")

    def destroy_node(self):
        """节点销毁时的清理函数"""
        if self.viewer:
            self.viewer.close()
        super().destroy_node()

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    node = None
    
    try:
        node = ArmIKNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 收到中断信号")
    except Exception as e:
        print(f"❌ IK节点异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🧹 开始清理IK节点...")
        
        if node:
            try:
                node.destroy_node()
                print("📴 IK节点已销毁")
            except Exception as e:
                print(f"IK节点销毁异常: {e}")
        
        # 只有当rclpy还在运行时才关闭
        if rclpy.ok():
            try:
                rclpy.shutdown()
                print("📴 ROS2已关闭")
            except Exception as e:
                print(f"ROS2关闭异常: {e}")
        
        print("✅ IK节点完全退出")


if __name__ == '__main__':
    main()