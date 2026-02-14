#!/usr/bin/env python3
"""
Hand Control Node - 手部硬件控制 ROS2 节点

根据架构图:
- 频率: 80Hz
- 输入: /action/{left,right}_hand/joints (JointState)
        含义: 6 个归一化关节角度 (0-1)
        格式: name  = ["thumb_rotation", "thumb_bend", "index", "middle", "ring", "pinky"]
              position = [6 个 float, 范围 0-1]

- 输出: /state/{left,right}_hand/joints (JointState)
        含义: 6 个归一化关节角度 (0-1) - 硬件反馈的当前状态
        格式: name  = ["thumb_rotation", "thumb_bend", "index", "middle", "ring", "pinky"]
              position = [6 个 float, 范围 0-1]
              velocity = [6 个 float]
              effort   = [6 个 float] (current)

数据流:
    Hand IK Node
        -> /action/{side}_hand/joints (normalized joint angles)
    Hand Control Node  [本节点]
        -> RS485 硬件通信 (RuiyanHandController)
        -> /state/{side}_hand/joints (hardware feedback)
    Hand FK Node
        -> /state/{side}_hand/keypoints (fingertip positions)

与遥操代码 (ry_hand_485_node) 的关系:
    - 硬件通信层完全一致: RuiyanHandController + SerialInterface
    - 关节命名和归一化约定完全一致: 6-DOF, 0-1 range
    - Ruckig 平滑插值逻辑一致
    - 主控制循环逻辑一致: send commands -> read status -> publish
"""

import logging
import time
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from hand.ry_hand_controller import RuiyanHandController
from hand.ry_hand_interface import (
    RuiyanFingerStatusMessage,
    RuiyanInstructionType,
    SerialInterface,
)

# Optional: Ruckig smooth interpolation
try:
    from hand.smooth_interpolator import SmoothJointInterpolator

    _HAS_RUCKIG = True
except ImportError:
    _HAS_RUCKIG = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (consistent with ry_hand_485_node and hand_ik_node)
# ---------------------------------------------------------------------------
JOINT_NAMES = [
    "thumb_rotation",
    "thumb_bend",
    "index",
    "middle",
    "ring",
    "pinky",
]
DOF = 6
MOTOR_POSITION_MAX = 4095  # Motor position range: 0-4095

# Motor ID to joint name mapping (consistent with ry_hand_485_node)
MOTOR_ID_TO_JOINT = {
    1: "thumb_rotation",
    2: "thumb_bend",
    3: "index",
    4: "middle",
    5: "ring",
    6: "pinky",
}


class HandControlNode(Node):
    """
    手部硬件控制 ROS2 节点

    职责:
        1. 订阅关节指令 (归一化 0-1 角度)
        2. Ruckig 平滑插值 (可选)
        3. 转换为电机位置 (0-4095) 通过 RS485 发送
        4. 读取电机状态并发布关节状态

    与遥操代码 ry_hand_485_node 的核心区别:
        - Topic 命名: /action/{side}_hand/joints, /state/{side}_hand/joints
        - 默认频率: 80Hz (遥操为 40Hz)
        - 增加了数据超时保护 (与 hand_ik_node 一致)
    """

    def __init__(self):
        super().__init__("hand_control_node")

        # ----------------------------------------------------------
        # 参数声明
        # ----------------------------------------------------------
        self.declare_parameter("hand_side", "left")
        self.declare_parameter("frequency", 80.0)
        self.declare_parameter("serial_port", "/dev/ttyACM0")
        self.declare_parameter("baudrate", 460800)
        self.declare_parameter("auto_connect", True)
        self.declare_parameter("motors_id", [1, 2, 3, 4, 5, 6])
        self.declare_parameter("instruction_type", "0xAA")
        self.declare_parameter("enable_interpolation", True)
        self.declare_parameter("data_timeout_sec", 0.5)
        self.declare_parameter("default_velocity", 3000)
        self.declare_parameter("default_current", 1000)

        # ----------------------------------------------------------
        # 参数获取
        # ----------------------------------------------------------
        self.hand_side: str = (
            self.get_parameter("hand_side").get_parameter_value().string_value
        )
        self.frequency: float = (
            self.get_parameter("frequency").get_parameter_value().double_value
        )
        serial_port: str = (
            self.get_parameter("serial_port").get_parameter_value().string_value
        )
        baudrate: int = (
            self.get_parameter("baudrate").get_parameter_value().integer_value
        )
        auto_connect: bool = (
            self.get_parameter("auto_connect").get_parameter_value().bool_value
        )
        motors_id = list(
            self.get_parameter("motors_id")
            .get_parameter_value()
            .integer_array_value
        )
        instruction_type_str: str = (
            self.get_parameter("instruction_type")
            .get_parameter_value()
            .string_value
        )
        self.enable_interpolation: bool = (
            self.get_parameter("enable_interpolation")
            .get_parameter_value()
            .bool_value
        )
        self.data_timeout_sec: float = (
            self.get_parameter("data_timeout_sec")
            .get_parameter_value()
            .double_value
        )
        self.default_velocity: int = (
            self.get_parameter("default_velocity")
            .get_parameter_value()
            .integer_value
        )
        self.default_current: int = (
            self.get_parameter("default_current")
            .get_parameter_value()
            .integer_value
        )

        instruction_type = int(instruction_type_str, 16)

        # ----------------------------------------------------------
        # 初始化硬件通信 (与 ry_hand_485_node 一致)
        # ----------------------------------------------------------
        self.get_logger().info(
            f"Initializing RS485 interface: port={serial_port}, baud={baudrate}"
        )
        hand_interface = SerialInterface(
            port=serial_port,
            baudrate=baudrate,
            mock=False,
            auto_connect=auto_connect,
        )
        self.hand = RuiyanHandController(
            communication_interface=hand_interface,
            motors_id=motors_id,
            instruction=RuiyanInstructionType(instruction_type),
        )

        # 初始化电机状态 (与 ry_hand_485_node 一致)
        self.hand.position_list = [0] * DOF
        self.hand.velocity_list = [self.default_velocity] * DOF
        self.hand.current_list = [self.default_current] * DOF

        self.get_logger().info("Hardware interface initialized successfully")

        # ----------------------------------------------------------
        # 初始化平滑插值器 (与 ry_hand_485_node 一致)
        # ----------------------------------------------------------
        if self.enable_interpolation:
            if _HAS_RUCKIG:
                self._init_interpolator()
            else:
                self.get_logger().warn(
                    "Ruckig not available, interpolation disabled. "
                    "Install with: pip install ruckig"
                )
                self.interpolator = None
                self.enable_interpolation = False
        else:
            self.interpolator = None

        # ----------------------------------------------------------
        # 状态变量
        # ----------------------------------------------------------
        self._target_angles: Optional[np.ndarray] = None
        self._last_command_time: float = 0.0
        self._first_publish_logged: bool = False

        # ----------------------------------------------------------
        # ROS2 接口
        # ----------------------------------------------------------
        # 订阅: 关节指令 (来自 Hand IK Node)
        self.joint_cmd_sub = self.create_subscription(
            JointState,
            f"/action/{self.hand_side}_hand/joints",
            self._joint_command_callback,
            10,
        )

        # 发布: 关节状态 (反馈给 Hand FK Node)
        self.joint_state_pub = self.create_publisher(
            JointState,
            f"/state/{self.hand_side}_hand/joints",
            10,
        )

        # 定时器: 主控制循环
        timer_period = 1.0 / self.frequency
        self.timer = self.create_timer(timer_period, self._timer_callback)

        # ----------------------------------------------------------
        # 启动日志
        # ----------------------------------------------------------
        self.get_logger().info("Hand Control Node started")
        self.get_logger().info(f"  Hand side      : {self.hand_side}")
        self.get_logger().info(f"  Frequency      : {self.frequency} Hz")
        self.get_logger().info(f"  Serial port    : {serial_port}")
        self.get_logger().info(f"  Baudrate       : {baudrate}")
        self.get_logger().info(
            f"  Interpolation  : "
            f"{'enabled' if self.enable_interpolation else 'disabled'}"
        )
        self.get_logger().info(f"  Data timeout   : {self.data_timeout_sec}s")
        self.get_logger().info(
            f"  Velocity       : {self.default_velocity}"
        )
        self.get_logger().info(
            f"  Current        : {self.default_current}"
        )
        self.get_logger().info(
            f"  Subscribe      : /action/{self.hand_side}_hand/joints"
        )
        self.get_logger().info(
            f"  Publish        : /state/{self.hand_side}_hand/joints"
        )

    # ------------------------------------------------------------------
    # 初始化方法
    # ------------------------------------------------------------------

    def _init_interpolator(self):
        """
        初始化 Ruckig 平滑插值器
        与 ry_hand_485_node._init_interpolator 逻辑一致
        """
        dt = 1.0 / self.frequency

        self.interpolator = SmoothJointInterpolator(
            dof=DOF,
            step=dt,
            alpha=0.5,
        )

        # 设置运动学限制 (适配 80Hz 控制频率)
        max_velocity = [200.0] * DOF
        max_acceleration = [400.0] * DOF
        max_jerk = [400.0] * DOF

        self.interpolator.set_kinematic_limits(
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            max_jerk=max_jerk,
        )

        # 初始化为零位置
        self.interpolator.set_input_param(
            current_position=np.zeros(DOF),
            current_velocity=np.zeros(DOF),
            current_acceleration=np.zeros(DOF),
        )

        self.get_logger().info(
            f"Ruckig interpolator initialized: "
            f"{DOF} DOF, {self.frequency} Hz"
        )

    # ------------------------------------------------------------------
    # 回调函数
    # ------------------------------------------------------------------

    def _joint_command_callback(self, msg: JointState):
        """
        处理接收到的关节指令
        与 ry_hand_485_node.set_angles_callback 逻辑一致

        期望 msg.position 包含 6 个 float (归一化 0-1):
            [thumb_rotation, thumb_bend, index, middle, ring, pinky]
        """
        positions = msg.position
        if len(positions) != DOF:
            self.get_logger().warn(
                f"Invalid joint command length: {len(positions)}, "
                f"expected {DOF}. Message ignored.",
                throttle_duration_sec=2.0,
            )
            return

        # Clip to valid range [0, 1] (与 hand_ik_node 的输出范围对齐)
        target = np.clip(np.array(positions), 0.0, 1.0)
        self._target_angles = target
        self._last_command_time = time.monotonic()

        # 设置速度和电流 (与 ry_hand_485_node 一致)
        self.hand.velocity_list = [self.default_velocity] * DOF
        self.hand.current_list = [self.default_current] * DOF

    def _timer_callback(self):
        """
        主控制循环 (与 ry_hand_485_node.loop 逻辑一致)

        流程:
            1. 检查是否有目标指令
            2. 数据过时检测 (与 hand_ik_node 一致)
            3. 平滑插值 (如果启用)
            4. 转换为电机位置并发送
            5. 读取电机状态
            6. 发布关节状态
        """
        try:
            # 有目标指令时: 插值 + 更新电机位置
            if self._target_angles is not None:
                # 数据过时检测 (与 hand_ik_node 一致)
                elapsed = time.monotonic() - self._last_command_time
                if elapsed <= self.data_timeout_sec:
                    # 平滑插值 (与 ry_hand_485_node 一致)
                    if (
                        self.enable_interpolation
                        and self.interpolator is not None
                    ):
                        smoothed_angles, _, _, _ = self.interpolator.update(
                            target_pos=self._target_angles
                        )
                        if isinstance(smoothed_angles, np.ndarray):
                            smoothed_angles = smoothed_angles.tolist()
                    else:
                        smoothed_angles = self._target_angles.tolist()

                    # 转换为电机位置 (0-4095)
                    # 注意: 遥操代码 ry_hand_485_node 用 int(p * 4096)，
                    # p=1.0 时会溢出到 4096 (超出 12-bit)。
                    # 这里使用 int(p * 4095) 更安全: p=1.0 -> 4095 (0xFFF)。
                    self.hand.position_list = [
                        int(p * MOTOR_POSITION_MAX) for p in smoothed_angles
                    ]
                else:
                    self.get_logger().warn(
                        f"Joint command expired ({elapsed:.2f}s > "
                        f"{self.data_timeout_sec}s), holding last position.",
                        throttle_duration_sec=2.0,
                    )
                    # 不更新 position_list，保持最后发送的位置

            # 发送指令并读取状态 (与 ry_hand_485_node.loop 一致)
            status_list = self.hand.loop()

            # 有效状态时发布 (与 ry_hand_485_node 一致)
            if status_list:
                joint_state = self._status_list_to_joint_state(status_list)
                self.joint_state_pub.publish(joint_state)

                # 首次发布成功日志 (与 hand_ik_node 的 once 模式一致)
                if not self._first_publish_logged:
                    self._first_publish_logged = True
                    self.get_logger().info(
                        f"First joint state published: "
                        f"pos=[{', '.join(f'{p:.3f}' for p in joint_state.position)}]"
                    )

            # 调试日志
            self.get_logger().debug(
                f"Loop: target={'set' if self._target_angles is not None else 'none'}, "
                f"motor_pos={self.hand.position_list}, "
                f"status_count={len(status_list)}"
            )

        except Exception as e:
            self.get_logger().warn(
                f"Control loop error: {e}. Skipping this cycle.",
                throttle_duration_sec=2.0,
            )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _status_list_to_joint_state(
        self, status_list: List[Optional[RuiyanFingerStatusMessage]]
    ) -> JointState:
        """
        将电机状态列表转换为 JointState 消息
        与 ry_hand_485_node._status_list_to_joint_state 完全一致
        """
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.header.frame_id = f"{self.hand_side}_hand_base_link"

        # 过滤 None 并按 motor_id 排序 (与 ry_hand_485_node 一致)
        valid_status = [s for s in status_list if s is not None]
        sorted_status = sorted(valid_status, key=lambda x: x.motor_id)

        joint_names = []
        positions = []
        velocities = []
        efforts = []

        for status in sorted_status:
            joint_name = MOTOR_ID_TO_JOINT.get(
                status.motor_id, f"joint_{status.motor_id}"
            )
            joint_names.append(joint_name)
            # 归一化位置: 0-4095 -> 0-1 (与 ry_hand_485_node 一致)
            positions.append(float(status.position or 0) / MOTOR_POSITION_MAX)
            velocities.append(float(status.velocity or 0))
            efforts.append(float(status.current or 0))

        joint_state.name = joint_names
        joint_state.position = positions
        joint_state.velocity = velocities
        joint_state.effort = efforts

        return joint_state

    # ------------------------------------------------------------------
    # 生命周期管理 (与 hand_ik_node.shutdown 模式对齐)
    # ------------------------------------------------------------------

    def shutdown(self):
        """节点关闭时的清理工作"""
        self.get_logger().info("Shutting down Hand Control Node...")

        # 取消定时器
        if hasattr(self, "timer"):
            self.timer.cancel()

        # 关闭插值器 (与 ry_hand_485_node 一致)
        if hasattr(self, "interpolator") and self.interpolator is not None:
            self.interpolator.close()
            self.get_logger().info("Interpolator closed")

        # 断开硬件连接
        if hasattr(self, "hand"):
            try:
                self.hand.disconnect()
                self.get_logger().info("Hardware disconnected")
            except Exception as e:
                self.get_logger().warn(f"Error disconnecting hardware: {e}")

        self.get_logger().info("Hand Control Node shut down complete")


def main(args=None):
    """主函数"""
    rclpy.init(args=args)

    node = None
    try:
        node = HandControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info(
                "Received keyboard interrupt, shutting down..."
            )
    except Exception as e:
        if node:
            node.get_logger().error(f"Node error: {e}")
        else:
            print(f"[HandControlNode] Initialization error: {e}")
    finally:
        if node:
            node.shutdown()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
