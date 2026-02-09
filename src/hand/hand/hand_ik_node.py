#!/usr/bin/env python3
"""
Hand IK Node - 手部逆运动学 ROS2 节点

根据架构图:
- 频率: 80Hz
- 输入: /action/{left,right}_hand/keypoints (JointState)
        含义: 5 个指尖在手腕坐标系 (wrist frame) 下的 3D 位置
        格式: position 字段包含 15 个 float，按以下顺序排列:
              [thumb_x, thumb_y, thumb_z,
               index_x, index_y, index_z,
               middle_x, middle_y, middle_z,
               ring_x, ring_y, ring_z,
               pinky_x, pinky_y, pinky_z]

- 输出: /action/{left,right}_hand/joints (JointState)
        含义: 6 个归一化关节角度 (0-1)
        格式: name  = ["thumb_rotation", "thumb_bend", "index", "middle", "ring", "pinky"]
              position = [6 个 float, 范围 0-1]
        与遥操作代码 (ry_hand_node) 的关节命名和归一化约定完全一致

数据流:
    Model Interface Node
        -> /action/{side}_hand/keypoints (keypoints in wrist frame)
    Hand IK Node  [本节点]
        -> /action/{side}_hand/joints (normalized joint angles)
    Hand Control Node
        -> 硬件控制
"""

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from hand.hand_ik_solver import FINGER_NAMES, JOINT_NAMES, HandIKSolver

# keypoints 消息中期望的 position 长度 (5 fingers x 3 coords)
_EXPECTED_KEYPOINT_DIM = len(FINGER_NAMES) * 3


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
        # 状态变量
        # ----------------------------------------------------------
        self._latest_keypoints: dict = {}  # {finger_name: np.array([x,y,z])}
        self._has_new_keypoints: bool = False
        self._last_keypoints_time: float = 0.0  # 最近一次收到 keypoints 的时间 (monotonic)
        self._keypoints_stamp = None  # 最近一次 keypoints 消息的 header.stamp
        self._first_solve_logged: bool = False  # 是否已记录首次求解日志

        # ----------------------------------------------------------
        # ROS2 接口
        # ----------------------------------------------------------
        # 订阅: 指尖关键点，这里的datatype存疑，先用jointstate填了一下
        self.keypoints_sub = self.create_subscription(
            JointState,
            f"/action/{self.hand_side}_hand/keypoints",
            self._keypoints_callback,
            10,
        )

        # 发布: 归一化关节角度
        self.joints_pub = self.create_publisher(
            JointState,
            f"/action/{self.hand_side}_hand/joints",
            10,
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
        self.get_logger().info(
            f"  Subscribe : /action/{self.hand_side}_hand/keypoints"
        )
        self.get_logger().info(
            f"  Publish   : /action/{self.hand_side}_hand/joints"
        )

    # ------------------------------------------------------------------
    # 回调函数
    # ------------------------------------------------------------------

    def _keypoints_callback(self, msg: JointState):
        """
        处理接收到的指尖关键点消息

        期望 msg.position 包含 15 个 float:
            [thumb_x, thumb_y, thumb_z,
             index_x, index_y, index_z,
             middle_x, middle_y, middle_z,
             ring_x, ring_y, ring_z,
             pinky_x, pinky_y, pinky_z]
        """
        positions = msg.position
        if len(positions) != _EXPECTED_KEYPOINT_DIM:
            self.get_logger().warn(
                f"Invalid keypoints length: {len(positions)}, "
                f"expected {_EXPECTED_KEYPOINT_DIM}. Message ignored.",
                throttle_duration_sec=2.0,
            )
            return

        # 解析为 {finger_name: [x, y, z]} 字典
        keypoints = {}
        for i, finger_name in enumerate(FINGER_NAMES):
            offset = i * 3
            keypoints[finger_name] = [
                positions[offset],
                positions[offset + 1],
                positions[offset + 2],
            ]

        self._latest_keypoints = keypoints
        self._has_new_keypoints = True
        self._last_keypoints_time = time.monotonic()
        self._keypoints_stamp = msg.header.stamp

    def _timer_callback(self):
        """
        定时回调: 运行 IK 求解并发布关节角度

        保护机制:
        - 没有收到过 keypoints 时不发布
        - keypoints 数据超时后停止发布（避免上游断开后手一直维持旧姿态）
        """
        if not self._has_new_keypoints:
            return

        # 数据过时检测: 如果 keypoints 消息超时，停止发布
        elapsed = time.monotonic() - self._last_keypoints_time
        if elapsed > self.data_timeout_sec:
            self.get_logger().warn(
                f"Keypoints data expired ({elapsed:.2f}s > {self.data_timeout_sec}s), "
                "skipping IK publish.",
                throttle_duration_sec=2.0,
            )
            return

        # 取出最新 keypoints 并重置标志
        target_positions = self._latest_keypoints
        self._has_new_keypoints = False

        # IK 求解
        try:
            normalized_joints, info = self.ik_solver.compute_ik(
                target_positions=target_positions,
                max_iterations=self.ik_max_iterations,
            )
        except Exception as e:
            self.get_logger().error(
                f"IK solve failed: {e}", throttle_duration_sec=2.0
            )
            return

        # 输出范围验证（与 ry_hand_node.joint_command_callback 的 0-1 clip 对齐）
        joint_positions = normalized_joints.tolist()
        for i, val in enumerate(joint_positions):
            if not (0.0 <= val <= 1.0):
                self.get_logger().warn(
                    f"Joint {JOINT_NAMES[i]} out of range [0,1]: {val:.4f}, clipping.",
                    throttle_duration_sec=2.0,
                )
                joint_positions[i] = max(0.0, min(1.0, val))

        # 构建 JointState 消息并发布
        # 格式与 ry_hand_485_node._status_list_to_joint_state 保持一致:
        #   header.frame_id, name, position
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = f"{self.hand_side}_hand_base_link"
        joint_msg.name = list(JOINT_NAMES)
        joint_msg.position = joint_positions
        self.joints_pub.publish(joint_msg)

        # 首次求解成功日志（与 haptic_glove_node 的 once=True 模式一致）
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
