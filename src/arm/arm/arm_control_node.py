#!/usr/bin/env python3
"""
Arm Control Node - 机械臂控制节点
根据架构图：
- 频率: 100Hz
- 输入: /state/{left,right}_arm/wrist_pose - arm states (wrist poses in camera frame) [JointState]
- 输出: /action/{left,right}_arm/joints - arm actions (joints) [JointState]
- 同时从Camera Node接收图像: /camera/{head,chest}/depth 和 /camera/{head,chest}/rgb [Image]
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image


class ArmControlNode(Node):
    """
    机械臂控制节点
    处理机械臂腕部位姿，生成关节控制指令
    同时接收相机图像数据
    """
    
    def __init__(self):
        super().__init__('arm_control_node')
        
        # 声明参数
        self.declare_parameter('arm_side', 'left')   # 'left' or 'right'
        self.declare_parameter('frequency', 100.0)   # 100Hz
        
        # 获取参数
        self.arm_side = self.get_parameter('arm_side').get_parameter_value().string_value
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        
        # 订阅机械臂腕部位姿状态
        self.wrist_pose_sub = self.create_subscription(
            JointState,
            f'/state/{self.arm_side}_arm/wrist_pose',
            self.wrist_pose_callback,
            10
        )
        
        # 订阅相机图像 - Head相机
        self.head_rgb_sub = self.create_subscription(
            Image,
            '/camera/head/rgb',
            self.head_rgb_callback,
            10
        )
        
        self.head_depth_sub = self.create_subscription(
            Image,
            '/camera/head/depth',
            self.head_depth_callback,
            10
        )
        
        # 订阅相机图像 - Chest相机
        self.chest_rgb_sub = self.create_subscription(
            Image,
            '/camera/chest/rgb',
            self.chest_rgb_callback,
            10
        )
        
        self.chest_depth_sub = self.create_subscription(
            Image,
            '/camera/chest/depth',
            self.chest_depth_callback,
            10
        )
        
        # 发布机械臂关节动作
        self.action_pub = self.create_publisher(
            JointState,
            f'/action/{self.arm_side}_arm/joints',
            10
        )
        
        # 创建定时器
        timer_period = 1.0 / self.frequency
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info(f'Arm Control Node 已启动')
        self.get_logger().info(f'机械臂: {self.arm_side}')
        self.get_logger().info(f'频率: {self.frequency} Hz')
        self.get_logger().info(f'订阅: /state/{self.arm_side}_arm/wrist_pose')
        self.get_logger().info(f'订阅: /camera/{{head,chest}}/{{rgb,depth}}')
        self.get_logger().info(f'发布: /action/{self.arm_side}_arm/joints')
        
        # TODO: 初始化控制器
    
    def wrist_pose_callback(self, msg):
        """处理接收到的腕部位姿状态"""
        # TODO: 实现位姿处理逻辑
        pass
    
    def head_rgb_callback(self, msg):
        """处理Head相机RGB图像"""
        # TODO: 实现图像处理逻辑
        pass
    
    def head_depth_callback(self, msg):
        """处理Head相机深度图像"""
        # TODO: 实现深度图处理逻辑
        pass
    
    def chest_rgb_callback(self, msg):
        """处理Chest相机RGB图像"""
        # TODO: 实现图像处理逻辑
        pass
    
    def chest_depth_callback(self, msg):
        """处理Chest相机深度图像"""
        # TODO: 实现深度图处理逻辑
        pass
    
    def timer_callback(self):
        """定时生成和发布控制指令"""
        # TODO: 实现控制逻辑
        pass


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

