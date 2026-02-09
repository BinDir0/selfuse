#!/usr/bin/env python3
"""
Hand IK Node - 手部逆运动学节点
根据架构图：
- 频率: 80Hz
- 输入: /action/{left,right}_hand/joints (JointState) - hand actions (joints)
- 输出: /state/{left,right}_hand/joints (JointState) - hand states (joints)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class HandIKNode(Node):
    """
    手部逆运动学节点
    将手部动作转换为关节状态
    """
    
    def __init__(self):
        super().__init__('hand_ik_node')
        
        # 声明参数
        self.declare_parameter('hand_side', 'left')  # 'left' or 'right'
        self.declare_parameter('frequency', 80.0)    # 80Hz
        
        # 获取参数
        self.hand_side = self.get_parameter('hand_side').get_parameter_value().string_value
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        
        # 订阅手部动作指令
        self.action_sub = self.create_subscription(
            JointState,
            f'/action/{self.hand_side}_hand/joints',
            self.action_callback,
            10
        )
        
        # 发布手部关节状态
        self.state_pub = self.create_publisher(
            JointState,
            f'/state/{self.hand_side}_hand/joints',
            10
        )
        
        # 创建定时器
        timer_period = 1.0 / self.frequency
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info(f'Hand IK Node 已启动')
        self.get_logger().info(f'手部: {self.hand_side}')
        self.get_logger().info(f'频率: {self.frequency} Hz')
        self.get_logger().info(f'订阅: /action/{self.hand_side}_hand/joints')
        self.get_logger().info(f'发布: /state/{self.hand_side}_hand/joints')
        
        # TODO: 初始化IK solver
    
    def action_callback(self, msg):
        """处理接收到的手部动作指令"""
        # TODO: 实现IK计算逻辑
        pass
    
    def timer_callback(self):
        """定时发布手部状态"""
        # TODO: 实现状态发布逻辑
        pass


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    node = None
    try:
        node = HandIKNode()
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

