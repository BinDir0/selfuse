import os
import sys
import time
import signal
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from ruckig import InputParameter, OutputParameter, Result, Ruckig

# 添加SDK路径
current_dir = os.path.dirname(os.path.abspath(__file__))
resource_paths = [
    os.path.join(current_dir, '..', 'resource'),
]

for path in resource_paths:
    if os.path.exists(path):
        sys.path.insert(0, path)

try:
    from Robotic_Arm import *
    print(f"✅ 成功从本地resource目录导入睿尔曼SDK: {path}")
except ImportError as e:
    print(f"❌ 导入睿尔曼SDK失败: {e}")
    sys.exit(1)


@dataclass
class RealmanConfig:
    """睿尔曼机械臂配置"""
    ip: str = ""
    port: int = 8080
    thread_mode: int = 2  # RM_TRIPLE_MODE_E
    run_mode: int = 1     # 1=实机, 0=仿真
    timeout_ms: int = 30000  # 30秒超时
    v_percent: float = 80.0
    a_percent: float = 100.0
    blend_r_deg: float = 0.0
    blocking: int = 0


class ArmControlNode(Node):
    def __init__(self):
        super().__init__('arm_control_node')

        self.declare_parameter('arm_side', "")
        self.declare_parameter('frequency', 100.0)
        self.arm_side = self.get_parameter('arm_side').value
        self.frequency = self.get_parameter('frequency').value
        self.dt = 1.0 / self.frequency

        # 配置
        self.rm_config = RealmanConfig()
        if self.arm_side == "left":
            self.rm_config.ip = "192.168.100.100"
            self.arm_name = "左臂"
        elif self.arm_side == "right":
            self.rm_config.ip = "192.168.100.101"
            self.arm_name = "右臂"
        self.dof = 7
        
        # 1. 初始化机械臂
        if not self._init_robotic_arm():
            self.get_logger().error(f"❌ {self.arm_name}初始化失败")
            return
        
        # 2. 初始化Ruckig插值器
        self._init_ruckig()
        
        # 3. 初始化订阅器和发布器
        self.arm_command_sub = self.create_subscription(
            JointState,
            f'/action/{self.arm_side}_arm/joints',
            self.arm_command_callback,
            10,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.wrist_pose_pub = self.create_publisher(
            PoseStamped,
            f'/state/{self.arm_side}_arm/wrist_pose',
            10,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.control_timer = self.create_timer(
            self.dt,
            self.control_callback,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.status_timer = self.create_timer(
            self.dt,
            self.publish_status,
            callback_group=MutuallyExclusiveCallbackGroup()
        )
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)

        self.get_logger().info(f"🤖 {self.arm_name}专用控制节点启动完成")

    def _init_robotic_arm(self) -> bool:
        """初始化机械臂"""
        try:
            self.get_logger().info(f"🔌 正在连接{self.arm_name}: {self.rm_config.ip}:{self.rm_config.port}")
            
            # 创建机械臂实例
            self.rm_arm = RoboticArm(self.rm_config.thread_mode)
            
            # 设置超时
            self.rm_arm.rm_set_timeout(self.rm_config.timeout_ms)
            
            # 连接
            self.rm_handle = self.rm_arm.rm_create_robot_arm(
                self.rm_config.ip,
                self.rm_config.port,
                level=2  # warning模式
            )
            
            if self.rm_handle.id <= 0:
                self.get_logger().error(f"❌ {self.arm_name}连接失败，句柄ID: {self.rm_handle.id}")
                return False
            
            self.get_logger().info(f"✅ {self.arm_name}连接成功，句柄ID: {self.rm_handle.id}")
            
            # 设置运行参数
            self.rm_arm.rm_set_arm_run_mode(self.rm_config.run_mode)
            
            # 检查上电状态
            power_state = self.rm_arm.rm_get_arm_power_state()
            if isinstance(power_state, tuple) and len(power_state) >= 2:
                if power_state[1] == 0:
                    self.get_logger().info(f"🔌 {self.arm_name}上电...")
                    self.rm_arm.rm_set_arm_power(1)
                    time.sleep(1)
                else:
                    self.get_logger().info(f"✅ {self.arm_name}已上电")
            
            # 获取当前位置
            current_state = self.rm_arm.rm_get_current_arm_state()
            if isinstance(current_state, tuple) and current_state[0] == 0:
                joint_degrees = current_state[1]['joint'][:7]
                self.current_joint_positions = np.radians(joint_degrees)
                self.get_logger().info(f"✅ {self.arm_name}当前位置: {np.degrees(self.current_joint_positions)}")
            else:
                self.get_logger().warn(f"⚠️  无法获取{self.arm_name}当前位置，使用默认位置")
                self.current_joint_positions = np.array([0.0813, -1.0521, 0.0675, -1.6824, -1.4881, 1.4717, -0.1232])
            self.target_joint_positions = None
            
            self.rm_connected = True
            return True
            
        except Exception as e:
            self.get_logger().error(f"❌ {self.arm_name}初始化异常: {e}")
            import traceback
            self.get_logger().error(f"{self.arm_name}异常堆栈: {traceback.format_exc()}")
            return False

    def _init_ruckig(self):
        """初始化Ruckig插值器"""
        try:
            # 创建插值器
            self.interpolator = Ruckig(self.dof, self.dt)
            self.input_param = InputParameter(self.dof)
            self.output_param = OutputParameter(self.dof)
            
            # 设置运动限制
            max_velocity = np.array([4, 4, 4, 4, 4, 4, 4])
            max_acceleration = np.array([10, 10, 10, 10, 10, 10, 10])
            max_jerk = np.array([100, 100, 100, 100, 100, 100, 100])
            
            self.input_param.max_velocity = max_velocity.tolist()
            self.input_param.max_acceleration = max_acceleration.tolist()
            self.input_param.max_jerk = max_jerk.tolist()
            
            # 设置初始状态
            if self.current_joint_positions is not None:
                self.input_param.current_position = self.current_joint_positions.tolist()
                self.input_param.current_velocity = [0.0] * self.dof
                self.input_param.current_acceleration = [0.0] * self.dof
                self.input_param.target_position = self.current_joint_positions.tolist()
                self.input_param.target_velocity = [0.0] * self.dof
                self.input_param.target_acceleration = [0.0] * self.dof
            
            self.get_logger().info(f"✅ Ruckig插值器初始化完成 ({self.dof}DOF, {self.frequency}Hz)")
            
        except Exception as e:
            self.get_logger().error(f"❌ Ruckig初始化异常: {e}")
            import traceback
            self.get_logger().error(f"异常堆栈: {traceback.format_exc()}")

    def arm_command_callback(self, msg: JointState):
        """处理关节指令"""
        self.target_joint_positions = np.array(msg.position)

    def control_callback(self):
        """控制回调"""
        # 如果有新的目标位置，更新Ruckig
        if self.target_joint_positions is not None and self.interpolator is not None:
            
            # 设置目标
            self.input_param.target_position = self.target_joint_positions.tolist()
            
            # 执行插值
            result = self.interpolator.update(self.input_param, self.output_param)
            
            if result == Result.Working or result == Result.Finished:
                # 获取插值后的位置
                interpolated_positions = np.array(self.output_param.new_position)
                
                # 发送到右臂
                success = self._send_to_arm(interpolated_positions)
                
                if success:
                    # 更新当前位置
                    self.current_joint_positions = interpolated_positions
                    
                    # 更新输入参数为下一次插值
                    self.input_param.current_position = self.output_param.new_position
                    self.input_param.current_velocity = self.output_param.new_velocity
                    self.input_param.current_acceleration = self.output_param.new_acceleration

    def _send_to_arm(self, joint_positions: np.ndarray) -> bool:
        """发送关节位置到机械臂"""
        try:
            if not self.rm_connected or self.rm_arm is None:
                return False
            
            # 转换为度数
            joint_degrees = np.degrees(joint_positions).tolist()
            
            # 发送透传指令
            result = self.rm_arm.rm_movej_canfd(
                joint_degrees,
                True,  # follow=True 高跟随模式
                0,     # trajectory_mode=0 完全透传
                0      # radio无效
            )
            
            if result == 0:
                return True
            else:
                if result == -1:  # 通信失败
                    self.get_logger().warn("⚠️  右臂通信失败，重试...")
                    time.sleep(0.01)
                    retry_result = self.rm_arm.rm_movej_canfd(joint_degrees, True, 0, 0)
                    return retry_result == 0
                else:
                    self.get_logger().warn(f"⚠️  {self.arm_name}指令异常: {result}")
                    return False
            
        except Exception as e:
            self.get_logger().error(f"❌ 发送{self.arm_name}指令异常: {e}")
            return False

    def publish_status(self):
        """发布关节状态"""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"{self.arm_name}_base"

        _, state = self.rm_arm.rm_get_current_arm_state()
        pose = state['pose']
        msg.position.x = pose[0]
        msg.position.y = pose[1]
        msg.position.z = pose[2]
        quat = R.from_euler('xyz', [pose[3], pose[4], pose[5]]).as_quat()
        msg.orientation.x = quat[0]
        msg.orientation.y = quat[1]
        msg.orientation.z = quat[2]
        msg.orientation.w = quat[3]
        
        self.wrist_pose_pub.publish(msg)

    def _signal_handler(self, signum, frame):
        """信号处理"""
        self.get_logger().info("🛑 收到中断信号，开始清理...")
        self._cleanup()
        rclpy.shutdown()

    def _cleanup(self):
        """清理资源"""
        try:
            # 停止定时器
            if hasattr(self, 'control_timer'):
                self.control_timer.cancel()
            if hasattr(self, 'status_timer'):
                self.status_timer.cancel()
            
            # 断开右臂
            if self.rm_right_arm is not None:
                try:
                    self.rm_right_arm.rm_delete_robot_arm()
                    self.get_logger().info("✅ 右臂连接已清理")
                except:
                    pass
            
            # 清理SDK状态
            try:
                RoboticArm.rm_destory()
                self.get_logger().info("✅ SDK状态已清理")
            except:
                pass
            
            self.get_logger().info("✅ 清理完成")
            
        except Exception as e:
            self.get_logger().error(f"清理异常: {e}")

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    node = ArmControlNode()
    executor.add_node(node)
    try: executor.spin()
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
