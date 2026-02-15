#!/usr/bin/env python3
"""
Hand IK Node - 手部逆运动学 ROS2 节点

根据架构图:
- 频率: 80Hz
- 输入: /action/{left,right}_hand/keypoints (PoseArray)
        含义: 5 个指尖在手腕坐标系 (wrist frame) 下的 3D 位姿
        格式: header.frame_id = "{side}_wrist"
              poses = [Pose × 5]，按 thumb, index, middle, ring, pinky 顺序
              每个 Pose.position = Point(x, y, z) 为指尖 3D 坐标 (米)
              每个 Pose.orientation = Quaternion (预留，IK 仅使用 position)

- 输入: /system/mode (String)
        含义: 系统模式切换命令
        支持: "reset" (复位模式), "inference" (推理模式)

- 输出: /action/{left,right}_hand/joints (JointState)
        含义: 6 个归一化关节角度 (0-1)
        格式: name  = ["thumb_rotation", "thumb_bend", "index", "middle", "ring", "pinky"]
              position = [6 个 float, 范围 0-1]
        与遥操作代码 (ry_hand_node) 的关节命名和归一化约定完全一致

数据流:
    Model Interface Node
        -> /action/{side}_hand/keypoints (PoseArray: keypoints in wrist frame)
        -> /system/mode (String: mode switch command)
    Hand IK Node  [本节点]
        -> /action/{side}_hand/joints (JointState: normalized joint angles)
    Hand Control Node
        -> 硬件控制
"""

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R

from hand.hand_ik_solver import FINGER_NAMES, JOINT_NAMES, HandIKSolver

# keypoints 消息中期望的 Pose 数量 (5 fingers)
_EXPECTED_NUM_FINGERS = len(FINGER_NAMES)


def _compute_wrist_to_hand_base_transform(arm='left'):
    """
    Compute the fixed 4x4 homogeneous transform: hand_base pose in wrist frame.

    Combines two sub-transforms derived from hardware geometry:
      1. wrist → TCP   (from arm connector, see arm_ik_node.py _hand_to_tcp)
      2. TCP → hand_base (from hand connector, see visualize_psirobot_with_rgbd_calib.py)

    Naming conventions (across different code files):
      - wrist     : actual wrist joint  ("hand" in arm_ik_node.py)
      - tcp       : tool center point   ("wrist" in visualize code, "tcp" in arm_ik_node.py)
      - hand_base : dexterous hand base ("hand_base" in visualize code)

    IMPORTANT: This function must be kept in sync with hand_fk_node.py.

    Args:
        arm: 'left' or 'right'

    Returns:
        T_wrist_hand_base: 4x4 numpy array
    """
    # ---- Part 1: wrist → TCP (arm_ik_node.py: _hand_to_tcp) ----
    # connector → tcp: z-axis translation 0.0345m
    T_conn2tcp = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.0345],
        [0, 0, 0, 1]
    ])

    # wrist(hand) → connector: pure rotation (left/right differ)
    if arm == 'left':
        T_wrist2conn = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])
    elif arm == 'right':
        T_wrist2conn = np.array([
            [0,  1,  0, 0],
            [0,  0, -1, 0],
            [-1, 0,  0, 0],
            [0,  0,  0, 1]
        ])
    else:
        raise ValueError(f"Invalid arm: '{arm}'. Must be 'left' or 'right'.")

    # T_wrist2tcp maps wrist-frame coords to tcp-frame coords
    T_wrist2tcp = T_conn2tcp @ T_wrist2conn

    # T_wrist_tcp = pose of TCP in wrist frame = inv(T_wrist2tcp)
    R_wt = T_wrist2tcp[:3, :3]
    t_wt = T_wrist2tcp[:3, 3]
    T_wrist_tcp = np.eye(4)
    T_wrist_tcp[:3, :3] = R_wt.T
    T_wrist_tcp[:3, 3] = -R_wt.T @ t_wt

    # ---- Part 2: TCP → hand_base (visualize_psirobot_with_rgbd_calib.py) ----
    # Connector geometry parameters (left/right differ)
    if arm == 'left':
        offset_1 = np.array([-0.0004094, 0.0000127, 0.0036907])
        rpy_1 = np.array([0.0, 0.0000038, 1.5707998])
        offset_2 = np.array([0.0, 0.0, 0.0036596])
        rpy_2 = np.array([0.0, 0.0, 0.0])
    else:  # right
        offset_1 = np.array([-0.0004101, -0.0000127, 0.0036911])
        rpy_1 = np.array([0.0, -0.0000037, -1.5707928])
        offset_2 = np.array([0.0, -0.0000002, 0.0036597])
        rpy_2 = np.array([0.0000001, -0.0000001, 3.1415925])

    rot_1 = R.from_euler('xyz', rpy_1).as_matrix()
    rot_2 = R.from_euler('xyz', rpy_2).as_matrix()
    hand_z_offset = 0.0105

    # Translation of hand_base in TCP frame
    # Derived by expanding the stepwise position computation in visualize code:
    #   hand_base_pos = tcp_pos + R_tcp @ (offset_1 + rot_1 @ offset_2
    #                                      + rot_1 @ rot_2 @ [0,0,h])
    t_tcp_hb = (offset_1
                + rot_1 @ offset_2
                + rot_1 @ rot_2 @ np.array([0.0, 0.0, hand_z_offset]))

    # Rotation of hand_base relative to TCP: R_z(180°)
    # From visualize code: hand_base_mat = tcp_mat @ R_z_180
    R_z_180 = np.array([
        [-1.0,  0.0, 0.0],
        [ 0.0, -1.0, 0.0],
        [ 0.0,  0.0, 1.0]
    ])

    T_tcp_hb = np.eye(4)
    T_tcp_hb[:3, :3] = R_z_180
    T_tcp_hb[:3, 3] = t_tcp_hb

    # ---- Compose: wrist → TCP → hand_base ----
    return T_wrist_tcp @ T_tcp_hb


class HandIKNode(Node):
    """
    手部逆运动学 ROS2 节点

    职责:
        1. 订阅指尖关键点 (keypoints in wrist frame)
        2. 调用 HandIKSolver 进行 IK 求解
        3. 发布归一化关节角度 (0-1) 到 joints topic
    """

    def __init__(self):
        super().__init__("hand_ik_node")

        # ----------------------------------------------------------
        # 系统模式：inference 接收 keypoints 并求解 IK，reset 发送复位关节角
        # ----------------------------------------------------------
        self.mode = "inference"  # or "reset"

        # ----------------------------------------------------------
        # ROS2 并发回调组
        # ----------------------------------------------------------
        self.keypoints_sub_group = ReentrantCallbackGroup()
        self.publisher_group = ReentrantCallbackGroup()
        self.mode_sub_group = ReentrantCallbackGroup()

        # ----------------------------------------------------------
        # 参数声明
        # ----------------------------------------------------------
        self.declare_parameter("hand_side", "left")
        self.declare_parameter("frequency", 80.0)
        self.declare_parameter("mjcf_path", "")
        self.declare_parameter("ik_solver", "daqp")
        self.declare_parameter("ik_max_iterations", 10)
        self.declare_parameter("data_timeout_sec", 0.5)  # keypoints 数据超时时间

        # ----------------------------------------------------------
        # 参数获取
        # ----------------------------------------------------------
        self.hand_side: str = (
            self.get_parameter("hand_side").get_parameter_value().string_value
        )
        self.frequency: float = (
            self.get_parameter("frequency").get_parameter_value().double_value
        )
        mjcf_path_param: str = (
            self.get_parameter("mjcf_path").get_parameter_value().string_value
        )
        ik_solver_type: str = (
            self.get_parameter("ik_solver").get_parameter_value().string_value
        )
        self.ik_max_iterations: int = (
            self.get_parameter("ik_max_iterations").get_parameter_value().integer_value
        )
        self.data_timeout_sec: float = (
            self.get_parameter("data_timeout_sec").get_parameter_value().double_value
        )

        # 空字符串视为未设置，使用默认路径
        mjcf_path = mjcf_path_param if mjcf_path_param else None

        # ----------------------------------------------------------
        # 初始化 IK 求解器
        # ----------------------------------------------------------
        self.get_logger().info(f"Initializing IK solver for {self.hand_side} hand...")
        self.ik_solver = HandIKSolver(
            hand_type=self.hand_side,
            mjcf_path=mjcf_path,
            solver=ik_solver_type,
            frequency=self.frequency,
        )
        self.get_logger().info("IK solver initialized successfully")

        # ----------------------------------------------------------
        # 手部安装变换: wrist frame <-> hand_base frame (MuJoCo 模型坐标系)
        # ----------------------------------------------------------
        # 综合 arm_ik_node.py (wrist→TCP) 和 visualize (TCP→hand_base) 的完整变换
        # FK 使用 T_wrist_hand_base 将 hand_base → wrist (输出给 Model Interface)
        # IK 使用其逆矩阵 T_hand_base_wrist 将 wrist → hand_base (输入给 IK solver)
        # 重要: 必须与 Hand FK Node 中的 T_wrist_hand_base 完全一致!
        self._T_wrist_hand_base = _compute_wrist_to_hand_base_transform(self.hand_side)

        # 逆变换: wrist → hand_base
        self._T_hand_base_wrist = np.linalg.inv(self._T_wrist_hand_base)
        # 预缓存旋转矩阵和平移 (仅对位置做变换时直接使用，避免每帧齐次运算)
        self._R_hand_base_wrist = self._T_hand_base_wrist[:3, :3]
        self._t_hand_base_wrist = self._T_hand_base_wrist[:3, 3]

        self.get_logger().info(
            f"Wrist→hand_base transform: "
            f"translation={np.array2string(self._T_wrist_hand_base[:3, 3], precision=6)}"
        )

        # ----------------------------------------------------------
        # Home 姿态配置 (reset 模式发布的关节角度)
        # ----------------------------------------------------------
        # 全伸直姿态：所有关节归一化值为 0.0 (完全伸展)
        self.home_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.get_logger().info(f"Home joints configuration: {self.home_joints}")

        # ----------------------------------------------------------
        # 状态变量
        # ----------------------------------------------------------
        self._latest_keypoints: dict = {}  # {finger_name: np.array([x,y,z])}
        self._has_new_keypoints: bool = False
        self._last_keypoints_time: float = 0.0  # 最近一次收到 keypoints 的时间 (monotonic)
        self._keypoints_stamp = None  # 最近一次 keypoints 消息的 header.stamp
        self._first_solve_logged: bool = False  # 是否已记录首次求解日志
        self._last_solved_joints: list = None  # 缓存上次 IK 求解的关节角 (80Hz 一直发)

        # ----------------------------------------------------------
        # ROS2 接口
        # ----------------------------------------------------------
        # 订阅1: 指尖关键点 (PoseArray, 5 个指尖在 wrist frame 下的 3D 位姿)
        self.keypoints_sub = self.create_subscription(
            PoseArray,
            f"/action/{self.hand_side}_hand/keypoints",
            self._keypoints_callback,
            10,
            callback_group=self.keypoints_sub_group,
        )

        # 订阅2: 系统模式切换
        self.mode_sub = self.create_subscription(
            String,
            "/system/mode",
            self._mode_callback,
            10,
            callback_group=self.mode_sub_group,
        )

        # 发布: 归一化关节角度
        self.joints_pub = self.create_publisher(
            JointState,
            f"/action/{self.hand_side}_hand/joints",
            10,
            callback_group=self.publisher_group,
        )

        # 定时器: 按指定频率运行 IK 并发布
        timer_period = 1.0 / self.frequency
        self.timer = self.create_timer(timer_period, self._timer_callback)

        # ----------------------------------------------------------
        # 启动日志
        # ----------------------------------------------------------
        self.get_logger().info(f"Hand IK Node started")
        self.get_logger().info(f"  Hand side : {self.hand_side}")
        self.get_logger().info(f"  Frequency : {self.frequency} Hz")
        self.get_logger().info(f"  IK solver : {ik_solver_type}")
        self.get_logger().info(f"  IK iters  : {self.ik_max_iterations}")
        self.get_logger().info(f"  Mode      : {self.mode}")
        self.get_logger().info(
            f"  Subscribe : /action/{self.hand_side}_hand/keypoints (PoseArray)"
        )
        self.get_logger().info(
            f"  Subscribe : /system/mode (String)"
        )
        self.get_logger().info(
            f"  Publish   : /action/{self.hand_side}_hand/joints (JointState)"
        )

    # ------------------------------------------------------------------
    # 回调函数
    # ------------------------------------------------------------------

    def _mode_callback(self, msg: String):
        """
        处理 /system/mode 消息的回调函数
        
        支持的模式：
        - "reset": 复位模式，手部伸直，停止接收 keypoints
        - "inference": 推理模式，正常 IK 求解
        """
        cmd = msg.data.strip().lower()
        
        if cmd == "reset":
            self.get_logger().info("🔄 切换到 reset 模式 (手部复位)")
            self.mode = "reset"
            # 清空缓存的 keypoints 和 IK 结果
            self._has_new_keypoints = False
            self._latest_keypoints = {}
            self._last_solved_joints = None
            
        elif cmd == "inference":
            self.get_logger().info("▶️ 切换到 inference 模式")
            self.mode = "inference"
            
        else:
            self.get_logger().warn(f"⚠️ 未识别的 system mode: {cmd}")

    def _keypoints_callback(self, msg: PoseArray):
        """
        处理接收到的指尖关键点消息 (PoseArray)

        期望 msg.poses 包含 5 个 Pose，按以下顺序:
            [thumb, index, middle, ring, pinky]
        每个 Pose.position 包含指尖在 wrist frame 下的 3D 坐标 (x, y, z)
        """
        # 模式检查：reset 模式下不接收 keypoints
        if self.mode == "reset":
            return
        
        if len(msg.poses) != _EXPECTED_NUM_FINGERS:
            self.get_logger().warn(
                f"Invalid keypoints count: {len(msg.poses)}, "
                f"expected {_EXPECTED_NUM_FINGERS}. Message ignored.",
                throttle_duration_sec=2.0,
            )
            return

        # 解析为 {finger_name: [x, y, z]} 字典
        # 坐标变换: wrist frame → hand_base frame (MuJoCo 模型坐标系)
        # IK solver 在 hand_base frame 中工作，输入必须转换到该坐标系
        keypoints = {}
        for i, finger_name in enumerate(FINGER_NAMES):
            p = msg.poses[i].position
            pos_wrist = np.array([p.x, p.y, p.z])
            pos_hand_base = self._R_hand_base_wrist @ pos_wrist + self._t_hand_base_wrist
            keypoints[finger_name] = pos_hand_base.tolist()

        self._latest_keypoints = keypoints
        self._has_new_keypoints = True
        self._last_keypoints_time = time.monotonic()
        self._keypoints_stamp = msg.header.stamp

    def _timer_callback(self):
        """
        定时回调: 80Hz 一直发布关节角度

        逻辑 (与架构图一致):
        - reset 模式: 直接发布 home 关节角度（全伸直），80Hz 一直发
        - inference 模式:
            - 有新 keypoints 时: IK 求解 → 更新缓存 → 发布
            - 无新 keypoints 时: 重发上次缓存的结果 → 保持 80Hz 持续输出
            - 首次 IK 求解前: 不发布（尚无有效数据）
        """
        # ========== Reset 模式：直接发布 home 关节角度 ==========
        if self.mode == "reset":
            self._publish_home_joints()
            return

        # ========== Inference 模式 ==========
        # 有新 keypoints 到达时：运行 IK 求解，更新缓存
        if self._has_new_keypoints:
            target_positions = self._latest_keypoints
            self._has_new_keypoints = False

            try:
                normalized_joints, info = self.ik_solver.compute_ik(
                    target_positions=target_positions,
                    max_iterations=self.ik_max_iterations,
                )
            except Exception as e:
                self.get_logger().error(
                    f"IK solve failed: {e}", throttle_duration_sec=2.0
                )
                # 求解失败不更新缓存，下面会重发上次结果
            else:
                # 输出范围验证（与 ry_hand_node.joint_command_callback 的 0-1 clip 对齐）
                joint_positions = normalized_joints.tolist()
                for i, val in enumerate(joint_positions):
                    if not (0.0 <= val <= 1.0):
                        self.get_logger().warn(
                            f"Joint {JOINT_NAMES[i]} out of range [0,1]: {val:.4f}, clipping.",
                            throttle_duration_sec=2.0,
                        )
                        joint_positions[i] = max(0.0, min(1.0, val))

                # 更新缓存
                self._last_solved_joints = joint_positions

                # 首次求解成功日志
                if not self._first_solve_logged:
                    self._first_solve_logged = True
                    self.get_logger().info(
                        f"First IK solve completed: "
                        f"joints=[{', '.join(f'{j:.3f}' for j in joint_positions)}], "
                        f"converged={info['converged']}, "
                        f"pos_err={info['position_error']*1000:.3f}mm"
                    )

                # 调试日志（节流输出，避免刷屏）
                self.get_logger().debug(
                    f"IK solved: iters={info['iterations']}, "
                    f"converged={info['converged']}, "
                    f"pos_err={info['position_error']*1000:.3f}mm, "
                    f"joints=[{', '.join(f'{j:.3f}' for j in joint_positions)}]"
                )

        # ========== 80Hz 一直发：发布缓存的关节角度 ==========
        # 首次 IK 求解前无缓存，不发布
        if self._last_solved_joints is None:
            return

        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = f"{self.hand_side}_hand_base_link"
        joint_msg.name = list(JOINT_NAMES)
        joint_msg.position = self._last_solved_joints
        self.joints_pub.publish(joint_msg)

    def _publish_home_joints(self):
        """
        发布 home 姿态的关节角度（reset 模式）
        
        Home 姿态：所有关节归一化值为 0.0（完全伸展）
        """
        # 构建 JointState 消息
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = f"{self.hand_side}_hand_base_link"
        joint_msg.name = list(JOINT_NAMES)
        joint_msg.position = self.home_joints
        self.joints_pub.publish(joint_msg)
        
        # 调试日志（节流输出）
        self.get_logger().debug(
            f"Reset mode: publishing home joints {self.home_joints}"
        )

    # ------------------------------------------------------------------
    # 生命周期管理（与 ry_hand_node.shutdown 模式对齐）
    # ------------------------------------------------------------------

    def shutdown(self):
        """节点关闭时的清理工作"""
        self.get_logger().info("Shutting down Hand IK Node...")

        if hasattr(self, "timer"):
            self.timer.cancel()

        self.get_logger().info("Hand IK Node shut down complete")


def main(args=None):
    """主函数"""
    rclpy.init(args=args)

    node = None
    try:
        node = HandIKNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        if node:
            node.get_logger().error(f"Node error: {e}")
        else:
            print(f"[HandIKNode] Initialization error: {e}")
    finally:
        if node:
            node.shutdown()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
