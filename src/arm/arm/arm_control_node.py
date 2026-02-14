#!/usr/bin/env python3
"""
Arm Control Node - 机械臂硬件控制 ROS2 节点

根据架构图:
- 频率: 100Hz
- 输入: /action/{left,right}_arm/joints (JointState)
        含义: 7 个关节角度 (弧度)
        格式: name  = ["joint_1", ..., "joint_7"]
              position = [7 个 float, 单位：弧度]

- 输出: /state/{left,right}_arm/joints (JointState)
        含义: 7 个关节角度 (弧度) - 硬件反馈的当前状态
        格式: name  = ["joint_1", ..., "joint_7"]
              position = [7 个 float, 单位：弧度]
              velocity = [7 个 float]
              effort   = [7 个 float] (current)

数据流:
    Arm IK Node
        -> /action/{side}_arm/joints (joint angles)
    Arm Control Node  [本节点]
        -> 睿尔曼机械臂硬件通信 (rm_movej_canfd)
        -> /state/{side}_arm/joints (hardware feedback)
    Arm FK Node
        -> /state/{side}_arm/wrist_pose (wrist poses)

与 mj-controller (xiaozi_ruckig_control_node) 的关系:
    - 硬件通信层完全一致: RealMan SDK (rm_movej_canfd)
    - Ruckig 平滑插值逻辑一致
    - 主控制循环逻辑一致: receive -> interpolate -> send -> feedback

与 hand_control_node 的架构对齐:
    - Topic 命名模式: /action/{side}_xxx/joints, /state/{side}_xxx/joints
    - 数据超时保护
    - 平滑插值器
    - 主控制循环逻辑
"""

import logging
import time
import signal
from typing import Optional, List
import sys
import os

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Import RealMan SDK (same as mj-controller)
try:
    # Add Robotic_Arm path for SDK import
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, '..', '..', '..', '..', 'resource', 'Robotic_Arm'),
        os.path.join(current_dir, '..', 'resource', 'Robotic_Arm'),
        os.path.join(current_dir, 'resource', 'Robotic_Arm'),
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            parent_path = os.path.dirname(abs_path)
            if parent_path not in sys.path:
                sys.path.insert(0, parent_path)
            break
    
    from Robotic_Arm import RoboticArm, rm_thread_mode_e
    _HAS_RM_SDK = True
except ImportError:
    _HAS_RM_SDK = False
    # Mock class for testing without hardware
    class RoboticArm:
        def __init__(self, *args, **kwargs):
            pass

# Optional: Ruckig smooth interpolation (same as hand_control_node)
try:
    # Try import from local Ruckig_Interpolator module
    from Ruckig_Interpolator import SmoothJointInterpolator
    _HAS_RUCKIG = True
except ImportError:
    try:
        # Fallback to hand package interpolator
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
# Constants (consistent with architecture)
# ---------------------------------------------------------------------------
DOF = 7  # 7-DOF arm
RAD_TO_DEG = 180.0 / np.pi
DEG_TO_RAD = np.pi / 180.0

# Joint names (consistent with mj-controller)
def get_joint_names(arm_side):
    """Get joint names based on arm side (consistent with mj-controller)"""
    if arm_side == 'left':
        return [f'arm1_joint_link{i+1}' for i in range(7)]
    else:
        return [f'arm2_joint_link{i+1}' for i in range(7)]


class ArmControlNode(Node):
    """
    机械臂硬件控制 ROS2 节点
    
    职责:
        1. 订阅关节指令 (弧度)
        2. Ruckig 平滑插值 (可选)
        3. 转换为度并通过 RealMan SDK 发送
        4. 读取机械臂状态并发布关节状态
    
    与 hand_control_node 的核心对齐:
        - Topic 命名: /action/{side}_arm/joints, /state/{side}_arm/joints
        - 默认频率: 100Hz
        - 增加了数据超时保护
    """
    
    def __init__(self):
        super().__init__("arm_control_node")
        
        # ----------------------------------------------------------
        # 参数声明
        # ----------------------------------------------------------
        self.declare_parameter("arm_side", "left")
        self.declare_parameter("frequency", 100.0)
        self.declare_parameter("robot_ip", "192.168.100.100")
        self.declare_parameter("robot_port", 8080)
        self.declare_parameter("enable_interpolation", True)
        self.declare_parameter("data_timeout_sec", 0.5)
        self.declare_parameter("v_percent", 30.0)
        self.declare_parameter("a_percent", 30.0)
        self.declare_parameter("joint_margin_deg", 1.0)  # Joint limit margin (degrees)
        self.declare_parameter("mock_mode", False)  # Mock mode for testing
        
        # ----------------------------------------------------------
        # 参数获取
        # ----------------------------------------------------------
        self.arm_side: str = (
            self.get_parameter("arm_side").get_parameter_value().string_value
        )
        self.frequency: float = (
            self.get_parameter("frequency").get_parameter_value().double_value
        )
        self.robot_ip: str = (
            self.get_parameter("robot_ip").get_parameter_value().string_value
        )
        self.robot_port: int = (
            self.get_parameter("robot_port").get_parameter_value().integer_value
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
        self.v_percent: float = (
            self.get_parameter("v_percent").get_parameter_value().double_value
        )
        self.a_percent: float = (
            self.get_parameter("a_percent").get_parameter_value().double_value
        )
        self.joint_margin_deg: float = (
            self.get_parameter("joint_margin_deg").get_parameter_value().double_value
        )
        self.mock_mode: bool = (
            self.get_parameter("mock_mode").get_parameter_value().bool_value
        )
        
        # ----------------------------------------------------------
        # 初始化硬件通信 (与 mj-controller 一致)
        # ----------------------------------------------------------
        # 关节限位（从硬件获取，用于安全检查）
        self.joint_limits_min: Optional[List[float]] = None
        self.joint_limits_max: Optional[List[float]] = None
        
        if not self.mock_mode and _HAS_RM_SDK:
            self._init_realman_arm()
        else:
            self.arm_instance = None
            self.arm_connected = False
            if self.mock_mode:
                self.get_logger().info("Mock mode enabled, hardware disabled")
            else:
                self.get_logger().warn("RealMan SDK not available, running in mock mode")
        
        # ----------------------------------------------------------
        # 初始化平滑插值器 (与 hand_control_node 一致)
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
        self._target_joints: Optional[np.ndarray] = None
        self._last_command_time: float = 0.0
        self._first_publish_logged: bool = False
        self._current_joint_state: Optional[np.ndarray] = None
        
        # ----------------------------------------------------------
        # ROS2 接口 (与 hand_control_node 架构对齐)
        # ----------------------------------------------------------
        # 订阅: 关节指令 (来自 Arm IK Node)
        self.joint_cmd_sub = self.create_subscription(
            JointState,
            f"/action/{self.arm_side}_arm/joints",
            self._joint_command_callback,
            10,
        )
        
        # 发布: 关节状态 (反馈给 Arm FK Node)
        self.joint_state_pub = self.create_publisher(
            JointState,
            f"/state/{self.arm_side}_arm/joints",
            10,
        )
        
        # 定时器: 主控制循环
        timer_period = 1.0 / self.frequency
        self.timer = self.create_timer(timer_period, self._timer_callback)
        
        # ----------------------------------------------------------
        # 信号处理 (资源管理增强, 遵循 mj-controller)
        # ----------------------------------------------------------
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # ----------------------------------------------------------
        # 启动日志
        # ----------------------------------------------------------
        self.get_logger().info("Arm Control Node started")
        self.get_logger().info(f"  Arm side       : {self.arm_side}")
        self.get_logger().info(f"  Frequency      : {self.frequency} Hz")
        self.get_logger().info(f"  Robot IP       : {self.robot_ip}")
        self.get_logger().info(f"  Robot Port     : {self.robot_port}")
        self.get_logger().info(
            f"  Interpolation  : "
            f"{'enabled' if self.enable_interpolation else 'disabled'}"
        )
        self.get_logger().info(f"  Data timeout   : {self.data_timeout_sec}s")
        self.get_logger().info(f"  Velocity       : {self.v_percent}%")
        self.get_logger().info(f"  Acceleration   : {self.a_percent}%")
        self.get_logger().info(f"  Joint margin   : {self.joint_margin_deg}°")
        self.get_logger().info(
            f"  Subscribe      : /action/{self.arm_side}_arm/joints"
        )
        self.get_logger().info(
            f"  Publish        : /state/{self.arm_side}_arm/joints"
        )
    
    # ------------------------------------------------------------------
    # 初始化方法
    # ------------------------------------------------------------------
    
    def _init_realman_arm(self):
        """
        初始化睿尔曼机械臂连接 (遵循 mj-controller)
        """
        try:
            self.get_logger().info(
                f"Initializing RealMan arm: {self.robot_ip}:{self.robot_port}"
            )
            
            # 创建机械臂实例 (遵循 mj-controller)
            self.arm_instance = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            
            # 设置超时
            self.arm_instance.rm_set_timeout(30000)  # 30s timeout
            
            # 连接机械臂
            handle = self.arm_instance.rm_create_robot_arm(
                self.robot_ip, self.robot_port, level=1
            )
            
            # 检查连接
            if handle is None or not hasattr(handle, 'id') or handle.id <= 0:
                self.get_logger().error("Failed to connect to RealMan arm")
                self.arm_connected = False
                return
            
            self.get_logger().info(f"✅ Connected to RealMan arm, handle ID: {handle.id}")
            
            # 设置运行模式
            ret = self.arm_instance.rm_set_arm_run_mode(1)  # 1 = real robot
            if not self._check_rm_result(ret, "set_run_mode"):
                self.arm_connected = False
                return
            
            # 上电检查
            power_state = self.arm_instance.rm_get_arm_power_state()
            if self._check_rm_result(power_state, "get_power_state"):
                if isinstance(power_state, tuple) and len(power_state) >= 2:
                    if power_state[1] == 0:
                        self.get_logger().info("Powering on the arm...")
                        ret = self.arm_instance.rm_set_arm_power(1)
                        self._check_rm_result(ret, "power_on")
                        time.sleep(0.5)
                    else:
                        self.get_logger().info("Arm is already powered on")
            
            # 获取关节限位 (安全性增强, 遵循 mj-controller)
            self._get_joint_limits()
            
            self.arm_connected = True
            self.get_logger().info("✅ RealMan arm initialized successfully")
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize RealMan arm: {e}")
            self.arm_instance = None
            self.arm_connected = False
    
    def _check_rm_result(self, ret, operation: str) -> bool:
        """检查睿尔曼 API返回结果 (遵循 mj-controller)"""
        if ret is None:
            success = True
        elif isinstance(ret, int):
            success = ret == 0
        elif isinstance(ret, tuple) and len(ret) >= 1:
            success = ret[0] == 0
        elif isinstance(ret, dict) and "code" in ret:
            success = ret["code"] == 0
        else:
            success = False
        
        if not success:
            self.get_logger().error(f"RealMan API {operation} failed: {ret}")
        else:
            self.get_logger().debug(f"RealMan API {operation} success")
        return success
    
    def _init_interpolator(self):
        """
        初始化 Ruckig 平滑插值器 (与 hand_control_node 逻辑一致)
        """
        dt = 1.0 / self.frequency
        
        self.interpolator = SmoothJointInterpolator(
            dof=DOF,
            step=dt,
            alpha=0.5,
        )
        
        # 设置运动学限制 (适配 100Hz 控制频率)
        # 参考 mj-controller 中的配置
        max_velocity = [4.0] * DOF     # rad/s
        max_acceleration = [10.0] * DOF  # rad/s^2
        max_jerk = [100.0] * DOF        # rad/s^3
        
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
    
    def _get_joint_limits(self):
        """
        获取机械臂的关节限位 (遵循 mj-controller)
        安全性增强：用于限位检查，防止超限运动
        """
        try:
            ret_min = self.arm_instance.rm_get_joint_min_pos()
            ret_max = self.arm_instance.rm_get_joint_max_pos()
            
            if self._check_rm_result(ret_min, "get_joint_min_pos") and \
               self._check_rm_result(ret_max, "get_joint_max_pos"):
                # 解析返回值 (兼容tuple和dict格式)
                min_limits = ret_min[1] if isinstance(ret_min, tuple) else ret_min.get("data", [])
                max_limits = ret_max[1] if isinstance(ret_max, tuple) else ret_max.get("data", [])
                
                self.joint_limits_min = min_limits
                self.joint_limits_max = max_limits
                
                self.get_logger().info(f"✅ 关节限位 - 最小: {min_limits}")
                self.get_logger().info(f"✅ 关节限位 - 最大: {max_limits}")
            else:
                self.get_logger().warn("⚠️  获取关节限位失败，将跳过限位检查")
                
        except Exception as e:
            self.get_logger().error(f"获取关节限位时出错: {e}")
    
    def _clamp_joints(self, target: List[float]) -> List[float]:
        """
        对关节角度进行限位裁剪 (遵循 mj-controller)
        安全性增强：防止机械臂超限运动导致硬件损坏
        
        Args:
            target: 目标关节角度列表（度）
        
        Returns:
            裁剪后的关节角度列表（度）
        """
        if self.joint_limits_min is None or self.joint_limits_max is None:
            self.get_logger().debug("限位未知，跳过裁剪")
            return target
            
        clamped = []
        for i, val in enumerate(target):
            if i < len(self.joint_limits_min) and i < len(self.joint_limits_max):
                # 添加安全边界，防止贴边风险 (遵循 mj-controller)
                min_val = self.joint_limits_min[i] + self.joint_margin_deg
                max_val = self.joint_limits_max[i] - self.joint_margin_deg
                clamped_val = min(max(val, min_val), max_val)
                
                # 如果发生裁剪，记录日志
                if clamped_val != val:
                    self.get_logger().debug(
                        f"关节{i}限位裁剪: {val:.2f}° -> {clamped_val:.2f}°"
                    )
                clamped.append(clamped_val)
            else:
                clamped.append(val)
        return clamped
    
    def _check_arm_connection(self) -> bool:
        """
        检查机械臂连接状态 (遵循 mj-controller)
        鲁棒性增强：避免在断开连接后调用API导致段错误
        
        Returns:
            连接是否正常
        """
        if not self.arm_connected or self.arm_instance is None:
            self.get_logger().debug("机械臂未连接或实例为空")
            return False
            
        # 通过获取关节状态检查连接是否正常
        try:
            current_state = self.arm_instance.rm_get_current_arm_state()
            
            if isinstance(current_state, tuple) and len(current_state) >= 2:
                # 检查返回码
                if current_state[0] == 0:  # 成功
                    return True
                else:
                    self.get_logger().warn(
                        f"连接检查失败，返回码: {current_state[0]}"
                    )
                    return False
            else:
                self.get_logger().error(
                    f"连接检查返回格式异常: {type(current_state)}"
                )
                return False
        except Exception as e:
            self.get_logger().error(f"连接检查时出错: {e}")
            return False
    
    # ------------------------------------------------------------------
    # 回调函数
    # ------------------------------------------------------------------
    
    def _joint_command_callback(self, msg: JointState):
        """
        处理接收到的关节指令 (与 hand_control_node 逻辑一致)
        
        期望 msg.position 包含 7 个 float (弧度):
            [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7]
        """
        positions = msg.position
        if len(positions) != DOF:
            self.get_logger().warn(
                f"Invalid joint command length: {len(positions)}, "
                f"expected {DOF}. Message ignored.",
                throttle_duration_sec=2.0,
            )
            return
        
        # 更新目标关节角度
        self._target_joints = np.array(positions)
        self._last_command_time = time.monotonic()
        
        self.get_logger().debug(
            f"Received joint command: "
            f"[{', '.join(f'{j:.3f}' for j in self._target_joints)}]"
        )
    
    def _timer_callback(self):
        """
        主控制循环 (架构对齐 hand_control_node)
        
        流程:
            1. 检查是否有目标指令
            2. 数据过时检测 (与 hand_control_node 一致)
            3. 平滑插值 (如果启用)
            4. 转换为度并发送到硬件
            5. 读取硬件状态
            6. 发布关节状态
        """
        try:
            # Step 1: 检查是否有目标指令
            if self._target_joints is not None:
                # Step 2: 数据过时检测 (与 hand_control_node 一致)
                elapsed = time.monotonic() - self._last_command_time
                if elapsed <= self.data_timeout_sec:
                    # Step 3: 平滑插值 (与 hand_control_node 一致)
                    if (
                        self.enable_interpolation
                        and self.interpolator is not None
                    ):
                        smoothed_joints, _, _, _ = self.interpolator.update(
                            target_pos=self._target_joints
                        )
                        if isinstance(smoothed_joints, np.ndarray):
                            current_joints = smoothed_joints
                        else:
                            current_joints = self._target_joints
                    else:
                        current_joints = self._target_joints
                    
                    # Step 4: 发送硬件指令 (遵循 mj-controller)
                    if self.arm_connected and self.arm_instance is not None:
                        self._send_joint_command(current_joints)
                    
                    # Update current state for feedback
                    self._current_joint_state = current_joints
                    
                else:
                    self.get_logger().warn(
                        f"Joint command expired ({elapsed:.2f}s > "
                        f"{self.data_timeout_sec}s), holding last position.",
                        throttle_duration_sec=2.0,
                    )
                    # Don't update, hold last position
            
            # Step 5 & 6: 读取状态并发布 (与 hand_control_node 一致)
            self._read_and_publish_state()
            
        except Exception as e:
            self.get_logger().warn(
                f"Control loop error: {e}. Skipping this cycle.",
                throttle_duration_sec=2.0,
            )
    
    def _send_joint_command(self, joint_angles_rad: np.ndarray):
        """
        发送关节角度指令到硬件 (遵循 mj-controller 的 rm_movej_canfd)
        增强：连接检查 + 限位检查
        
        Args:
            joint_angles_rad: 7个关节角度 (弧度)
        """
        # 鲁棒性增强：检查连接状态，避免段错误 (遵循 mj-controller)
        if not self._check_arm_connection():
            self.get_logger().error(
                "机械臂连接已断开，跳过指令发送",
                throttle_duration_sec=2.0
            )
            # 更新连接状态
            self.arm_connected = False
            return
        
        try:
            # 验证输入数据
            if len(joint_angles_rad) != DOF:
                self.get_logger().error(
                    f"关节位置长度{len(joint_angles_rad)} != {DOF}"
                )
                return
            
            # 转换为度 (RealMan SDK使用度)
            joint_degrees = (joint_angles_rad * RAD_TO_DEG).tolist()
            
            # 安全性增强：限位检查 (遵循 mj-controller)
            joint_degrees = self._clamp_joints(joint_degrees)
            
            # 检查是否有无效值
            if any(not np.isfinite(x) for x in joint_degrees):
                self.get_logger().error("关节角度包含无效值 (NaN or Inf)")
                return
            
            # 使用 rm_movej_canfd 进行高跟随透传 (遵循 mj-controller)
            ret = self.arm_instance.rm_movej_canfd(
                joint_degrees,  # 7个关节角度，单位：度
                True,          # follow=True: 高跟随模式
                0,             # trajectory_mode=0: 完全透传模式
                0              # radio: 完全透传下无效
            )
            
            if not self._check_rm_result(ret, "movej_canfd"):
                # 如果是通信失败(-1)，尝试重试 (遵循 mj-controller)
                if ret == -1:
                    self.get_logger().warn(
                        "透传失败(-1)，尝试重试...",
                        throttle_duration_sec=2.0
                    )
                    time.sleep(0.05)  # 短暂等待
                    
                    # 重试一次
                    retry_ret = self.arm_instance.rm_movej_canfd(
                        joint_degrees, True, 0, 0
                    )
                    if self._check_rm_result(retry_ret, "movej_canfd_retry"):
                        self.get_logger().info("✅ 透传重试成功")
                    else:
                        self.get_logger().warn(
                            "⚠️  透传重试失败，但保持连接避免段错误"
                        )
                else:
                    self.get_logger().debug(
                        f"Failed to send joint command: {ret}",
                        throttle_duration_sec=2.0
                    )
            else:
                self.get_logger().debug(
                    f"Sent joint command: [{', '.join(f'{d:.2f}°' for d in joint_degrees)}]"
                )
        
        except Exception as e:
            self.get_logger().error(
                f"Error sending joint command: {e}",
                throttle_duration_sec=2.0
            )
            # 发生异常时标记为断开连接
            self.arm_connected = False
    
    def _read_and_publish_state(self):
        """
        读取硬件状态并发布 (与 hand_control_node 逻辑一致)
        """
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.header.frame_id = f"{self.arm_side}_arm_base_link"
        
        # 设置关节名称
        joint_state.name = get_joint_names(self.arm_side)
        
        # 读取硬件状态 (如果连接)
        if self.arm_connected and self.arm_instance is not None:
            try:
                # 获取当前关节状态 (遵循 mj-controller)
                ret = self.arm_instance.rm_get_current_arm_state()
                
                if self._check_rm_result(ret, "get_current_arm_state"):
                    if isinstance(ret, tuple) and len(ret) >= 2:
                        arm_state = ret[1]
                        if 'joint' in arm_state:
                            joint_degrees = arm_state['joint']
                            
                            # 转换为弧度
                            joint_state.position = (
                                np.array(joint_degrees) * DEG_TO_RAD
                            ).tolist()
                            
                            # Velocity and effort (如果可用)
                            joint_state.velocity = [0.0] * DOF
                            joint_state.effort = [0.0] * DOF
                            
                            # Update current state
                            self._current_joint_state = np.array(joint_state.position)
                        else:
                            self.get_logger().debug("No joint data in arm state")
                            return
                else:
                    self.get_logger().debug("Failed to get arm state")
                    return
                    
            except Exception as e:
                self.get_logger().error(
                    f"Error reading arm state: {e}",
                    throttle_duration_sec=2.0
                )
                return
        else:
            # Mock mode: use current target as feedback
            if self._current_joint_state is not None:
                joint_state.position = self._current_joint_state.tolist()
                joint_state.velocity = [0.0] * DOF
                joint_state.effort = [0.0] * DOF
            else:
                # No data yet
                return
        
        # 发布状态
        self.joint_state_pub.publish(joint_state)
        
        # 首次发布成功日志 (与 hand_control_node 一致)
        if not self._first_publish_logged:
            self._first_publish_logged = True
            self.get_logger().info(
                f"First joint state published: "
                f"pos=[{', '.join(f'{p:.3f}' for p in joint_state.position)}]"
            )
    
    # ------------------------------------------------------------------
    # 清理方法 (资源管理增强, 遵循 mj-controller)
    # ------------------------------------------------------------------
    
    def _signal_handler(self, signum, frame):
        """
        信号处理函数，用于Ctrl+C安全中断 (遵循 mj-controller)
        资源管理增强：确保安全停止机械臂
        """
        self.get_logger().warn("SIGINT received. Trying slow stop...")
        self._safe_emergency_stop()
        sys.exit(0)
    
    def _safe_emergency_stop(self):
        """
        安全急停功能 (遵循 mj-controller)
        资源管理增强：使用缓停而非急停，保护硬件
        """
        if self.arm_connected and self.arm_instance is not None:
            try:
                self.get_logger().info("执行机械臂缓停...")
                self.arm_instance.rm_set_arm_slow_stop()
                self.get_logger().info("✅ 机械臂缓停执行完成")
            except Exception as e:
                self.get_logger().error(f"机械臂缓停时出错: {e}")
    
    def cleanup(self):
        """
        清理资源 (遵循 mj-controller)
        资源管理增强：完整的清理流程
        """
        # Step 1: 安全停止运动
        self._safe_emergency_stop()
        
        # Step 2: 清理Ruckig插值器
        if hasattr(self, 'interpolator') and self.interpolator is not None:
            try:
                if hasattr(self.interpolator, 'close'):
                    self.interpolator.close()
                    self.get_logger().info("✅ Ruckig插值器已清理")
            except Exception as e:
                self.get_logger().error(f"清理插值器时出错: {e}")
        
        # Step 3: 断开机械臂连接
        if self.arm_instance is not None:
            try:
                if self.arm_connected:
                    # 只对仍然连接的机械臂调用断开API
                    self.arm_instance.rm_delete_robot_arm()
                    self.get_logger().info("✅ 机械臂已断开连接")
            except Exception as e:
                self.get_logger().error(f"断开机械臂连接时出错: {e}")
            finally:
                self.arm_connected = False
                self.arm_instance = None
        
        self.get_logger().info("✅ Arm Control Node cleanup completed")
    
    def destroy_node(self):
        """节点销毁时清理资源 (遵循 mj-controller)"""
        self.get_logger().info("Shutting down Arm Control Node...")
        
        # 调用完整的清理方法
        self.cleanup()
        
        super().destroy_node()


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    node = None
    try:
        node = ArmControlNode()
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
